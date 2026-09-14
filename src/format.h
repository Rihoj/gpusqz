// gpusqz container format: self-describing, chunk-based, raw fallback per chunk.
//
//   FileHeader
//   ChunkEntry[chunk_count]
//   TableGroup[table_group_count]
//   payload: each chunk's [ChunkFlag][data], back to back in chunk order
//   tables:  each group's quantised counts, in group order, ending the file
#pragma once
#include <cstddef>
#include <cstdint>

#include "rans_codes.h"

namespace gpusqz {

constexpr uint32_t kMagic = 0x5A515347; // "GSQZ" in file byte order
constexpr uint32_t kVersion = 1;

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

struct ChunkEntry {
  uint64_t offset;          // of this chunk's data within the payload
  uint32_t compressed_size; // including the flag byte
  uint32_t original_size;   // only the file's last chunk may be short of chunk_size
};

// LzRans chunks share rANS tables with the rest of their compression batch,
// a "table group". Groups cover [0, chunk_count) contiguously, in order.
// Each group picks its own literal-context rule (rans_codes.h), so its
// quantised counts vary in size; they live in the table section.
struct TableGroup {
  uint32_t start_chunk;
  uint32_t chunk_count;
  uint32_t lit_ctx_shift; // 8, 4 or 0: 1, 16 or 256 literal contexts
};
#pragma pack(pop)

inline size_t group_quant_bytes(const TableGroup& g) { return (size_t)quant_bytes(lit_ctx_count(g.lit_ctx_shift)); }

} // namespace gpusqz
