// Per-chunk LZSS compression/decompression kernels.
//
// Parallelism model: one CUDA thread per chunk. Chunks are fully independent
// (no cross-chunk references), so this scales with however many chunks the
// input has, not with any cooperation between threads. This is simpler than
// nvCOMP's warp/block-per-chunk LZ4 design, at the cost of per-thread serial
// work; it trades some peak throughput for a much simpler, easier-to-verify
// implementation.
//
// Wire format per chunk (written after the chunk's 1-byte flag, which lives
// in the container, not here):
//   Repeating groups of: [1 flag byte] [up to 8 items]
//   Item bit (LSB first) 0 = literal: 1 raw byte follows.
//   Item bit 1 = match: 2 bytes little-endian offset, 1 byte (length - 3).
//     Offset is the backward distance from the current output position.
//     Length is 3..258.
//   Decoding stops as soon as `original_size` output bytes have been
//   produced, so a partial final group's unused flag bits are never read.
#pragma once
#include <cstdint>
#include <cstring>

namespace gzp {

constexpr int kHashBits = 9;              // 512-entry hash table
constexpr int kHashSize = 1 << kHashBits; // per-thread, lives in local memory
constexpr int kMinMatch = 3;
constexpr int kMaxMatch = 258; // 3 + 255, length byte encodes (len - 3)

__device__ __forceinline__ uint32_t hash3(const uint8_t* p) {
  uint32_t v = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
  return (v * 2654435761u) >> (32 - kHashBits);
}

// Encodes one chunk. `in` has `in_len` valid bytes (in_len <= chunk_size).
// `out` must have room for at least worst_case_size(chunk_size) bytes
// (the caller reserves this; this function does NOT itself fall back to
// raw — the caller compares the returned size against in_len and chooses
// the container's Raw/Lzss flag accordingly).
// Returns the number of bytes written to `out`.
__device__ inline uint32_t lzss_encode_chunk(const uint8_t* in, uint32_t in_len, uint8_t* out) {
  // Local (per-thread) hash table: maps hash(3 bytes) -> last position seen.
  // 0xFFFFFFFF means empty. Sized to fit comfortably in local memory.
  uint32_t htab[kHashSize];
  for (int i = 0; i < kHashSize; ++i) htab[i] = 0xFFFFFFFFu;

  uint32_t out_pos = 0;
  uint32_t i = 0;
  while (i < in_len) {
    uint32_t group_start = out_pos;
    out[out_pos++] = 0; // flag byte, filled in below
    uint8_t flags = 0;

    for (int item = 0; item < 8 && i < in_len; ++item) {
      uint32_t best_len = 0;
      uint32_t best_off = 0;

      if (i + kMinMatch <= in_len) {
        uint32_t h = hash3(in + i);
        uint32_t cand = htab[h];
        htab[h] = i;
        if (cand != 0xFFFFFFFFu && cand < i) {
          uint32_t max_len = in_len - i;
          if (max_len > kMaxMatch) max_len = kMaxMatch;
          uint32_t len = 0;
          while (len < max_len && in[cand + len] == in[i + len]) ++len;
          if (len >= kMinMatch) {
            best_len = len;
            best_off = i - cand;
          }
        }
      }

      if (best_len >= kMinMatch) {
        out[out_pos + 0] = (uint8_t)(best_off & 0xFF);
        out[out_pos + 1] = (uint8_t)((best_off >> 8) & 0xFF);
        out[out_pos + 2] = (uint8_t)(best_len - kMinMatch);
        out_pos += 3;
        flags |= (uint8_t)(1u << item);
        i += best_len;
      } else {
        out[out_pos++] = in[i];
        ++i;
      }
    }
    out[group_start] = flags;
  }
  return out_pos;
}

// Decodes one chunk. `in` holds `in_len` compressed bytes; `out` must have
// room for exactly `original_size` bytes.
__device__ inline void lzss_decode_chunk(const uint8_t* in, uint32_t in_len, uint8_t* out,
                                          uint32_t original_size) {
  uint32_t ip = 0;
  uint32_t op = 0;
  while (op < original_size) {
    uint8_t flags = in[ip++];
    for (int item = 0; item < 8 && op < original_size; ++item) {
      if (flags & (uint8_t)(1u << item)) {
        uint32_t off = (uint32_t)in[ip] | ((uint32_t)in[ip + 1] << 8);
        uint32_t len = (uint32_t)in[ip + 2] + kMinMatch;
        ip += 3;
        uint32_t src = op - off;
        for (uint32_t k = 0; k < len; ++k) out[op + k] = out[src + k];
        op += len;
      } else {
        out[op++] = in[ip++];
      }
    }
  }
  (void)in_len;
}

} // namespace gzp
