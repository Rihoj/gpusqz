// gzp container format: self-describing, chunk-based, raw-fallback per chunk.
#pragma once
#include <cstdint>

#include "rans_codes.h"

namespace gzp {

constexpr uint32_t kMagic = 0x50475A47; // "GZGP" little-endian
constexpr uint32_t kVersion = 4;

// One warp compresses one chunk, and ~400-500 warps are resident on this
// GPU at once, so batches of a few hundred chunks fill it regardless of
// chunk size; larger chunks then mostly buy compression ratio (longer
// history for matches). Match offsets are u32 (see lz_warp.cuh), so the
// cap here is a deliberate limit, not a format one: 1MB keeps per-chunk
// scratch and the match-finding hash table's reach reasonable given a
// fixed table size (see kHashBits) rather than growing them further.
constexpr uint32_t kDefaultChunkSize = 65536;
constexpr uint32_t kMaxChunkSize = 1u << 20;

// Named chunk-size presets for `gzp c --profile <name>` (see main.cu),
// measured on a 283MB text corpus: speed keeps today's default (fastest
// on both ends); ratio uses the largest chunk this format allows (beats
// zstd -1's ratio there, at a real compress/decompress speed cost — see
// README for the full sweep); balance is a middle ground that closes
// most of that ratio gap while staying well above gzip -1's speed.
constexpr uint32_t kProfileSpeedChunkSize = 65536;
constexpr uint32_t kProfileBalanceChunkSize = 262144;
constexpr uint32_t kProfileRatioChunkSize = kMaxChunkSize;

// Shortest match the LZ stage emits. Baked into both payload formats:
// token match codes and rANS match-length codes are relative to it.
// (3 was measured to be ~3% worse and slower than 4 on text.)
constexpr int kMinMatch = 4;

// Bytes reserved per chunk in the fixed-slot output layout. Encoders give
// up (and the chunk is stored raw) before writing past the input size, so
// the slot only needs the flag byte plus header headroom beyond chunk_size.
inline uint32_t worst_case_size(uint32_t chunk_size) {
  return (chunk_size + 512 + 15) & ~15u;
}

enum class ChunkFlag : uint8_t {
  Raw = 0,
  Lz = 1,     // LZ4-style token stream, see lz_warp.cuh
  LzRans = 2, // rANS-coded LZ sequences, table shared across a TableGroup
};

#pragma pack(push, 1)
struct FileHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t chunk_size;
  uint64_t original_size;
  uint32_t chunk_count;
  uint32_t table_group_count;
};

// One per chunk, written after FileHeader, before the compressed chunk data.
struct ChunkEntry {
  uint64_t offset;          // byte offset of this chunk's data within the compressed data section
  uint32_t compressed_size; // size of this chunk's data (including the 1-byte flag)
  uint32_t original_size;   // uncompressed size of this chunk (last chunk may be short)
};

// Every LzRans chunk's rANS table is shared with the other chunks that were
// compressed in the same host batch (a "table group"), rather than stored
// per chunk: groups always cover a contiguous, gap-free range of chunk
// indices (chunk c belongs to whichever group has start_chunk <= c <
// start_chunk + chunk_count), in ascending order. table_group_count of
// these follow the ChunkEntry array, before the chunk payload data.
struct TableGroup {
  uint32_t start_chunk;
  uint32_t chunk_count;
  uint8_t q[kQuantBytes]; // quantised frequency counts, see rans_codes.h
};
#pragma pack(pop)

} // namespace gzp
