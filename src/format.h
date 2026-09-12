// gzp container format: self-describing, chunk-based, raw-fallback per chunk.
#pragma once
#include <cstdint>

namespace gzp {

constexpr uint32_t kMagic = 0x50475A47; // "GZGP" little-endian
constexpr uint32_t kVersion = 1;

// Default chunk size: small enough that a modest input file produces enough
// chunks to fill an SM-heavy GPU with one thread per chunk (see README), and
// small enough that an 8KB LZSS window covers the whole chunk.
constexpr uint32_t kDefaultChunkSize = 8192;

// Worst case a chunk can expand to. The LZSS encoder writes speculatively
// into this buffer before deciding Raw vs Lzss (see kernels.cu), so this
// must bound the LZSS *encoder's own* worst case, not just the final
// raw-fallback size: an all-literals chunk emits 1 flag byte per 8 items
// plus the items themselves, then the container adds its own 1-byte
// Raw/Lzss flag on top.
inline uint32_t worst_case_size(uint32_t chunk_size) {
  return chunk_size + (chunk_size + 7) / 8 + 1;
}

enum class ChunkFlag : uint8_t {
  Raw = 0,
  Lzss = 1,
};

#pragma pack(push, 1)
struct FileHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t chunk_size;
  uint64_t original_size;
  uint32_t chunk_count;
};

// One per chunk, written after FileHeader, before the compressed chunk data.
struct ChunkEntry {
  uint64_t offset;          // byte offset of this chunk's data within the compressed data section
  uint32_t compressed_size; // size of this chunk's data (including the 1-byte flag)
  uint32_t original_size;   // uncompressed size of this chunk (last chunk may be short)
};
#pragma pack(pop)

} // namespace gzp
