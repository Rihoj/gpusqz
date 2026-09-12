// Warp-per-chunk LZ codec.
//
// Token format (ChunkFlag::Lz payload), LZ4-style:
//   sequence := token literals* [offset:u16le match_ext*]
//   token    := (lit_len:4 << 4) | ml_code:4
//   lit_len 15 and ml_code 15 are followed by extension bytes: 255s then a
//   final byte < 255, all summed onto the base value.
//   match length = ml_code + kMinMatch. Offset is the backward distance,
//   1..65535, and must not reach before the chunk start.
//   The last sequence may be literals only: the decoder stops as soon as
//   the output is full, so no offset follows if literals complete it.
//
// All device functions here are warp-collective: every lane of the warp
// must call them with the same arguments, and their control flow is
// warp-uniform so no lane ever skips a shuffle or ballot.
#pragma once
#include <cstdint>

namespace gzp {

constexpr int kMinMatch = 4;
constexpr int kHashBits = 11;
constexpr int kHashSize = 1 << kHashBits; // u32 buckets holding two u16 positions
constexpr int kWarpsPerBlock = 4;
constexpr unsigned kFullMask = 0xFFFFFFFFu;
constexpr uint32_t kEmptyPos = 0xFFFFu;

__device__ __forceinline__ uint32_t load4(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

__device__ __forceinline__ uint32_t hash4(uint32_t v) { return (v * 2654435761u) >> (32 - kHashBits); }

__device__ __forceinline__ uint32_t match_len(const uint8_t* in, uint32_t a, uint32_t b, uint32_t max_len) {
  uint32_t l = 0;
  while (l < max_len && in[a + l] == in[b + l]) ++l;
  return l;
}

__device__ __forceinline__ uint32_t ext_bytes(uint32_t v) { return v >= 15 ? 1 + (v - 15) / 255 : 0; }

__device__ __forceinline__ uint32_t seq_size(uint32_t lit_len, uint32_t ml) {
  uint32_t s = 1 + ext_bytes(lit_len) + lit_len;
  if (ml) s += 2 + ext_bytes(ml - kMinMatch);
  return s;
}

__device__ __forceinline__ uint32_t write_ext(uint8_t* out, uint32_t p, uint32_t v) {
  uint32_t r = v - 15;
  while (r >= 255) {
    out[p++] = 255;
    r -= 255;
  }
  out[p++] = (uint8_t)r;
  return p;
}

// Appends one sequence at out[op]. ml == 0 means a literals-only tail.
// Returns false, uniformly, if the sequence would not fit within cap.
__device__ __forceinline__ bool emit_seq(const uint8_t* in, uint8_t* out, uint32_t cap, uint32_t& op,
                                         uint32_t lit_start, uint32_t lit_len, uint32_t off, uint32_t ml) {
  uint32_t size = seq_size(lit_len, ml);
  if (op + size > cap) return false;
  int lane = threadIdx.x & 31;
  uint32_t lit_pos = op + 1 + ext_bytes(lit_len);
  if (lane == 0) {
    uint32_t mlc = ml ? ml - kMinMatch : 0;
    out[op] = (uint8_t)(((lit_len >= 15 ? 15 : lit_len) << 4) | (mlc >= 15 ? 15 : mlc));
    if (lit_len >= 15) write_ext(out, op + 1, lit_len);
    if (ml) {
      uint32_t p = lit_pos + lit_len;
      out[p++] = (uint8_t)(off & 0xFF);
      out[p++] = (uint8_t)(off >> 8);
      if (mlc >= 15) write_ext(out, p, mlc);
    }
  }
  for (uint32_t k = lane; k < lit_len; k += 32) out[lit_pos + k] = in[lit_start + k];
  op += size;
  return true;
}

// Encodes in[0..n) into out, writing at most cap bytes. htab is this
// warp's kHashSize-entry shared-memory table. Returns false if the output
// would exceed cap (caller stores the chunk raw instead).
//
// Match finding currently runs serially on lane 0 while the other lanes
// wait at the broadcast; the token emission is warp-cooperative.
__device__ inline bool lz_encode_warp(const uint8_t* in, uint32_t n, uint8_t* out, uint32_t cap,
                                      uint32_t* out_len, uint32_t* htab) {
  int lane = threadIdx.x & 31;
  for (int i = lane; i < kHashSize; i += 32) htab[i] = 0xFFFFFFFFu;
  __syncwarp();

  uint32_t op = 0;
  uint32_t lit_start = 0;
  uint32_t pos = 0; // lane 0 only
  for (;;) {
    uint32_t cmd = 0, m_pos = 0, m_off = 0, m_len = 0;
    if (lane == 0) {
      while (pos + kMinMatch <= n) {
        uint32_t h = hash4(load4(in + pos));
        uint32_t b = htab[h];
        uint32_t best = 0, best_off = 0;
        for (int w = 0; w < 2; ++w) {
          uint32_t cand = (b >> (16 * w)) & 0xFFFF;
          if (cand != kEmptyPos && cand < pos) {
            uint32_t l = match_len(in, cand, pos, n - pos);
            if (l > best) {
              best = l;
              best_off = pos - cand;
            }
          }
        }
        htab[h] = (b << 16) | pos;
        if (best >= (uint32_t)kMinMatch) {
          cmd = 1;
          m_pos = pos;
          m_off = best_off;
          m_len = best;
          pos += best;
          break;
        }
        ++pos;
      }
      if (cmd == 0) cmd = 2;
    }
    cmd = __shfl_sync(kFullMask, cmd, 0);
    m_pos = __shfl_sync(kFullMask, m_pos, 0);
    m_off = __shfl_sync(kFullMask, m_off, 0);
    m_len = __shfl_sync(kFullMask, m_len, 0);
    if (cmd == 1) {
      if (!emit_seq(in, out, cap, op, lit_start, m_pos - lit_start, m_off, m_len)) return false;
      lit_start = m_pos + m_len;
    } else {
      if (lit_start < n && !emit_seq(in, out, cap, op, lit_start, n - lit_start, 0, 0)) return false;
      break;
    }
  }
  *out_len = op;
  return true;
}

__device__ __forceinline__ bool read_ext(const uint8_t* in, uint32_t in_len, uint32_t& ip, uint32_t& v) {
  uint8_t b;
  do {
    if (ip >= in_len) return false;
    b = in[ip++];
    v += b;
  } while (b == 255);
  return true;
}

// Decodes exactly `orig` bytes from in[0..in_len) into out. Returns false
// on any malformed input; never reads or writes out of bounds.
__device__ inline bool lz_decode_warp(const uint8_t* in, uint32_t in_len, uint8_t* out, uint32_t orig) {
  int lane = threadIdx.x & 31;
  uint32_t ip = 0, op = 0;
  while (op < orig) {
    if (ip >= in_len) return false;
    uint8_t tok = in[ip++];
    uint32_t lit = tok >> 4;
    if (lit == 15 && !read_ext(in, in_len, ip, lit)) return false;
    if (ip + lit > in_len || op + lit > orig) return false;
    for (uint32_t k = lane; k < lit; k += 32) out[op + k] = in[ip + k];
    ip += lit;
    op += lit;
    if (op >= orig) break;

    if (ip + 2 > in_len) return false;
    uint32_t off = (uint32_t)in[ip] | ((uint32_t)in[ip + 1] << 8);
    ip += 2;
    uint32_t ml = (tok & 15) + kMinMatch;
    if ((tok & 15) == 15) {
      uint32_t e = 15;
      if (!read_ext(in, in_len, ip, e)) return false;
      ml = e + kMinMatch;
    }
    if (off == 0 || off > op || op + ml > orig) return false;

    __syncwarp(); // literals above must be visible before an overlapping match reads them
    const uint8_t* src = out + op - off;
    if (off >= ml) {
      for (uint32_t k = lane; k < ml; k += 32) out[op + k] = src[k];
    } else {
      // Overlapping match: byte k repeats byte k mod off of the already
      // written history, so lanes never depend on each other's writes.
      for (uint32_t k = lane; k < ml; k += 32) out[op + k] = src[k % off];
    }
    __syncwarp();
    op += ml;
  }
  return ip == in_len;
}

} // namespace gzp
