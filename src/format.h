// gpusqz container format: self-describing, chunk-based, raw fallback per chunk.
//
//   FileHeader
//   u32 compressed_size[chunk_count]  each chunk's payload bytes, flag included
//   u32 checksum[chunk_count]         of those bytes (chunk_hash)
//   TableGroup[table_group_count]
//   payload: each chunk's [ChunkFlag][data], back to back in chunk order
//   tables:  each group's quantised counts, coded by table_codec.h,
//            TableGroup::table_bytes each, in group order, ending the file
//
// A chunk's payload offset is the sum of the sizes before it, and every
// chunk holds chunk_size bytes of the original except the last, which
// holds the rest of original_size.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "rans_codes.h"

namespace gpusqz {

constexpr uint32_t kMagic = 0x5A515347; // "GSQZ" in file byte order
constexpr uint32_t kVersion = 3;

// One warp compresses one chunk. Larger chunks give the match finder more
// history, so they compress better but are coarser units of parallelism.
// The 1MB cap is a policy, not a format limit (offsets are u32): it bounds
// per-chunk scratch, SeqRec's 21-bit fields and the hash table's reach.
constexpr uint32_t kDefaultChunkSize = 65536;
constexpr uint32_t kMaxChunkSize = 1u << 20;

// Chunk sizes for `gpusqz c --profile speed|balance|ratio` (see docs/usage.md).
constexpr uint32_t kProfileSpeedChunkSize = 65536;
constexpr uint32_t kProfileBalanceChunkSize = 262144;
constexpr uint32_t kProfileRatioChunkSize = kMaxChunkSize;

// Shortest match the LZ stage emits; token and rANS match-length codes are
// relative to it. (3 measured ~3% worse and slower on text.)
constexpr int kMinMatch = 4;

// A TableGroup covers this much input, so which chunks share tables — and
// so the bytes gpusqz writes — depend only on the file, not on how much GPU
// memory a batch happened to get. A group is small enough to fit any usable
// GPU, and a compression batch holds whole groups (see plan_batches).
constexpr uint64_t kGroupBytes = 64ull << 20;
inline uint32_t group_chunks(uint32_t chunk_size) {
  // Tests only: GPUSQZ_FORCE_GROUP_CHUNKS puts several groups in a batch
  // without needing a kGroupBytes-sized input.
  static const uint32_t forced = [] {
    const char* e = std::getenv("GPUSQZ_FORCE_GROUP_CHUNKS");
    unsigned long v = e ? std::strtoul(e, nullptr, 10) : 0;
    return (uint32_t)v;
  }();
  if (forced) return forced;
  uint64_t n = kGroupBytes / chunk_size;
  return (uint32_t)(n < 1 ? 1 : n);
}

// Bytes reserved per chunk in the fixed-slot output layout: encoders give
// up (and store the chunk raw) before writing past the input size, so a
// slot needs only the flag byte plus header headroom beyond chunk_size.
inline uint32_t worst_case_size(uint32_t chunk_size) { return (chunk_size + 512 + 15) & ~15u; }

enum class ChunkFlag : uint8_t {
  Raw = 0,
  Lz = 1,     // LZ4-style token stream, see lz_warp.cuh
  LzRans = 2, // rANS-coded LZ sequences against its TableGroup's tables, see rans.cuh
};

#pragma pack(push, 1)
struct FileHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t chunk_size;
  uint64_t original_size;
  uint32_t chunk_count;
  uint32_t table_group_count;
  uint64_t tables_offset; // absolute offset of the table section, which ends the file
};


// LzRans chunks share rANS tables with the rest of their compression batch,
// a "table group". Groups cover [0, chunk_count) contiguously, in order.
// Each group picks its own literal-context rule (rans_codes.h), so its
// quantised counts vary in size; they live in the table section.
struct TableGroup {
  uint32_t start_chunk;
  uint32_t chunk_count;
  uint32_t lit_ctx_shift; // 8, 4 or 0: 1, 16 or 256 literal contexts
  uint32_t table_bytes;   // of this group's coded counts in the table section
  uint32_t checksum;      // chunk_hash of those coded bytes
};
#pragma pack(pop)

// Every stored byte is covered by a 32-bit check: each chunk's payload by
// checksum[c], each group's coded counts by TableGroup::checksum. That is
// what catches a corrupted literal, table or length byte that would
// otherwise decode "successfully" into wrong output. It is a plain hash,
// not a cryptographic one, and the decoder rejects a chunk whose payload
// doesn't match.
//
// The bytes are split over 32 lanes (lane l takes l, l + 32, ...), each
// lane folds its own bytes, and the lanes combine in order, so one warp,
// one workgroup or a plain loop all produce the same value.
constexpr uint32_t kHashInit = 2166136261u; // FNV-1a's offset basis and prime
constexpr uint32_t kHashMul = 16777619u;
#ifdef __CUDACC__
__host__ __device__
#endif
    inline uint32_t
    hash_fold(uint32_t h, uint32_t byte) {
  return (h ^ byte) * kHashMul;
}
// The scalar definition (the GPUs compute the same value lane by lane).
inline uint32_t chunk_hash(const uint8_t* p, size_t n) {
  uint32_t lane_h[32];
  for (int l = 0; l < 32; ++l) {
    uint32_t h = kHashInit;
    for (size_t i = (size_t)l; i < n; i += 32) h = hash_fold(h, p[i]);
    lane_h[l] = h;
  }
  uint32_t h = kHashInit;
  for (int l = 0; l < 32; ++l) {
    for (int b = 0; b < 4; ++b) h = hash_fold(h, (lane_h[l] >> (8 * b)) & 0xFFu);
  }
  return h ^ (uint32_t)n;
}

// Bytes of a group's quantised counts once decoded (table_codec.h).
inline size_t group_quant_bytes(const TableGroup& g) { return (size_t)quant_bytes(lit_ctx_count(g.lit_ctx_shift)); }

// Where a chunk is, derived from the stored sizes (not itself stored).
struct ChunkEntry {
  uint64_t offset;          // of this chunk's data within the payload
  uint32_t compressed_size; // including the flag byte
  uint32_t original_size;   // only the file's last chunk may be short of chunk_size
};

// Rebuilds every chunk's entry from its stored compressed size. Returns
// false if chunk_count doesn't match original_size and chunk_size, or a
// size is 0 (every chunk has at least its flag byte).
inline bool chunk_entries(const uint32_t* sizes, uint32_t chunk_count, uint32_t chunk_size, uint64_t original_size,
                          ChunkEntry* out) {
  if (chunk_size == 0 || chunk_count != (original_size + chunk_size - 1) / chunk_size) return false;
  uint64_t off = 0;
  for (uint32_t c = 0; c < chunk_count; ++c) {
    if (sizes[c] == 0) return false;
    uint64_t start = (uint64_t)c * chunk_size;
    uint64_t left = original_size - start;
    out[c] = ChunkEntry{off, sizes[c], (uint32_t)(left < chunk_size ? left : chunk_size)};
    off += sizes[c];
  }
  return true;
}

} // namespace gpusqz
