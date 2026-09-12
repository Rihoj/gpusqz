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

#include "format.h"

namespace gzp {

static_assert(kMinMatch == 3 || kMinMatch == 4, "hashing supports 3- or 4-byte minimum matches");
constexpr int kProbe = 32; // per-lane match-length cap before cooperative extension
constexpr int kHashBits = 11;
constexpr int kHashSize = 1 << kHashBits; // u32 buckets holding two u16 positions
constexpr int kWarpsPerBlock = 4;
constexpr unsigned kFullMask = 0xFFFFFFFFu;
constexpr uint32_t kEmptyPos = 0xFFFFu;

// One parsed sequence: lit_len literals followed by a match of ml bytes
// at backward distance off (ml == 0 only for a literals-only tail).
struct SeqRec {
  uint32_t lit_len;
  uint32_t ml;
  uint32_t off;
};

// Every match covers at least kMinMatch bytes, plus one optional tail.
__host__ __device__ inline uint32_t max_sequences(uint32_t chunk_size) { return chunk_size / kMinMatch + 1; }

__device__ __forceinline__ uint32_t load4(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

__device__ __forceinline__ uint32_t hash_at(const uint8_t* p) {
  uint32_t v = kMinMatch == 4 ? load4(p) : ((uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16));
  return (v * 2654435761u) >> (32 - kHashBits);
}

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

// Extends a match at p (backward distance off) that already covers `len`
// bytes, 32 bytes per step, until the first mismatch or the chunk end.
__device__ __forceinline__ uint32_t warp_extend(const uint8_t* in, uint32_t p, uint32_t off, uint32_t n,
                                                uint32_t len) {
  int lane = threadIdx.x & 31;
  uint32_t max_len = n - p;
  while (len < max_len) {
    uint32_t k = len + lane;
    bool mism = k >= max_len || in[p + k] != in[p - off + k];
    unsigned m = __ballot_sync(kFullMask, mism);
    if (m) return len + (__ffs(m) - 1);
    len += 32;
  }
  return len;
}

// Emits LZ4-style tokens straight into an output buffer.
struct TokenEmitter {
  const uint8_t* in;
  uint8_t* out;
  uint32_t cap;
  uint32_t op = 0;
  __device__ __forceinline__ bool operator()(uint32_t lit_start, uint32_t lit_len, uint32_t off, uint32_t ml) {
    return emit_seq(in, out, cap, op, lit_start, lit_len, off, ml);
  }
};

// Records sequences and copies literals into scratch for a later stage.
struct SeqEmitter {
  const uint8_t* in;
  SeqRec* seqs;
  uint8_t* lits;
  uint32_t n_seq = 0;
  uint32_t n_lit = 0;
  __device__ __forceinline__ bool operator()(uint32_t lit_start, uint32_t lit_len, uint32_t off, uint32_t ml) {
    int lane = threadIdx.x & 31;
    if (lane == 0) seqs[n_seq] = SeqRec{lit_len, ml, off};
    for (uint32_t k = lane; k < lit_len; k += 32) lits[n_lit + k] = in[lit_start + k];
    ++n_seq;
    n_lit += lit_len;
    return true;
  }
};

// Re-emits recorded sequences as tokens (literals re-read from `in`).
__device__ inline bool tokens_from_seqs(const uint8_t* in, const SeqRec* seqs, uint32_t n_seq, uint8_t* out,
                                        uint32_t cap, uint32_t* out_len) {
  uint32_t op = 0, pos = 0;
  for (uint32_t i = 0; i < n_seq; ++i) {
    SeqRec r = seqs[i];
    if (!emit_seq(in, out, cap, op, pos, r.lit_len, r.off, r.ml)) return false;
    pos += r.lit_len + r.ml;
  }
  *out_len = op;
  return true;
}

// Parses in[0..n) into sequences, calling emit(lit_start, lit_len, off, ml)
// for each match (and once more with ml = 0 for any trailing literals).
// htab is this warp's kHashSize-entry shared-memory table. Returns false
// as soon as emit does.
//
// The warp walks the chunk in 32-byte windows. Every lane hashes its own
// position, probes the two candidates in its bucket (capped at kProbe
// bytes so per-lane work is bounded), and the window's matches are then
// selected warp-uniformly with a one-position lazy lookahead; matches
// that hit the probe cap are extended cooperatively. Candidates only ever
// come from earlier windows because insertion happens after every lane
// has read; a repeat that starts and recurs inside the same 32 bytes is
// picked up from the next window on.
template <class Emit>
__device__ inline bool lz_parse_warp(const uint8_t* in, uint32_t n, uint32_t* htab, Emit& emit) {
  int lane = threadIdx.x & 31;
  for (int i = lane; i < kHashSize; i += 32) htab[i] = 0xFFFFFFFFu;
  __syncwarp();

  uint32_t lit_start = 0;
  uint32_t pos = 0;
  while (pos + kMinMatch <= n) {
    uint32_t p = pos + lane;
    bool valid = p + kMinMatch <= n;
    uint32_t h = 0, b = 0xFFFFFFFFu;
    uint32_t best_len = 0, best_off = 0;
    if (valid) {
      h = hash_at(in + p);
      b = htab[h];
      uint32_t max_len = min((uint32_t)kProbe, n - p);
      for (int w = 0; w < 2; ++w) {
        uint32_t cand = (b >> (16 * w)) & 0xFFFF;
        if (cand != kEmptyPos && cand < p) {
          uint32_t l = match_len(in, cand, p, max_len);
          if (l > best_len) {
            best_len = l;
            best_off = p - cand;
          }
        }
      }
    }
    __syncwarp(); // all reads of htab precede any insert

    // One insert per distinct bucket per window: the highest valid lane
    // (the most recent position) wins, deterministically.
    unsigned valid_mask = __ballot_sync(kFullMask, valid);
    unsigned peers = __match_any_sync(kFullMask, h) & valid_mask;
    if (valid && lane == 31 - __clz(peers)) htab[h] = (b << 16) | p;

    unsigned mask = __ballot_sync(kFullMask, best_len >= (uint32_t)kMinMatch);
    uint32_t cur = 0;
    while (cur < 32) {
      unsigned m = mask & (~0u << cur);
      if (!m) break;
      uint32_t j = __ffs(m) - 1;
      uint32_t lj = __shfl_sync(kFullMask, best_len, j);
      uint32_t lj1 = __shfl_sync(kFullMask, best_len, min(j + 1, 31u));
      // Lazy match: if the next position has a clearly longer match, emit
      // this byte as a literal and take that one instead.
      if (j < 31 && ((mask >> (j + 1)) & 1) && lj1 > lj + 1) {
        ++j;
        lj = lj1;
      }
      uint32_t off = __shfl_sync(kFullMask, best_off, j);
      uint32_t len = lj;
      if (len == (uint32_t)kProbe) len = warp_extend(in, pos + j, off, n, len);
      if (!emit(lit_start, pos + j - lit_start, off, len)) return false;
      lit_start = pos + j + len;
      cur = j + len;
    }
    pos += cur > 32 ? cur : 32;
  }
  if (lit_start < n && !emit(lit_start, n - lit_start, 0, 0)) return false;
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
