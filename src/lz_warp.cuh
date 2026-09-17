// Warp-per-chunk LZ codec.
//
// Token format (ChunkFlag::Lz payload), LZ4-style:
//   sequence := token literals* [offset:u32le match_ext*]
//   token    := (lit_len:4 << 4) | ml_code:4
//   lit_len 15 and ml_code 15 are followed by extension bytes: 255s then a
//   final byte < 255, all summed onto the base value.
//   match length = ml_code + kMinMatch. Offset is the backward distance,
//   1..chunk_size, and must not reach before the chunk start.
//   The last sequence may be literals only: the decoder stops as soon as
//   the output is full, so no offset follows if literals complete it.
//
// All device functions here are warp-collective: every lane of the warp
// must call them with the same arguments, and their control flow is
// warp-uniform so no lane ever skips a shuffle or ballot.
#pragma once
#include <cstdint>

#include "format.h"
#include "kernels.h"

namespace gpusqz {

static_assert(kMinMatch == 3 || kMinMatch == 4, "hashing supports 3- or 4-byte minimum matches");
// Per-lane match-length cap before cooperative extension (64 measured
// ~0.1% smaller output for 1-3% slower compression).
constexpr int kProbe = 32;
// Set-associative hash table, one u32 chunk-relative position per word
// (more buckets beat deeper buckets at equal size). It lives in global
// memory, one region per chunk: a bigger shared-memory table cost more in
// occupancy than it gained (see docs/performance-history.md).
constexpr int kBucketWays = kHashBucketWays;
constexpr int kWarpsPerBlock = 1;
// How many positions ahead lz_parse_warp's lazy matching looks before
// committing to a match (zstd's "lazy2"; a third step never paid off).
constexpr int kLazySteps = 2;
// Acceleration on data that doesn't match (LZ4's and zstd's "skip ahead"):
// after 2^kSkipTrigger windows in a row without a match, the window's 32
// lanes sample every stride-th position, the stride growing by one every
// 2^kSkipTrigger further windows up to kMaxSkip, and the first match found
// is extended back over the positions skipped. Only sampled positions are
// hashed, so a later copy of sampled data is found only if some sample in
// it lines up with one in the original: short copies of random-looking
// data get missed (2.4% bigger output on random runs followed by copies of
// their tails). Measured, against no sampling: random data 3.0x (`speed`)
// to 6.7x (`ratio`) faster compress kernel; text unchanged; shared
// libraries at most 0.02% bigger; already-compressed files (gz, png, jpg,
// ...) -0.22% to +0.11% and 14-61% faster. Waiting 2 or 4 windows instead
// of 8 lost more; skipping whole windows instead of spreading the lanes
// lost 8-12% on the copies.
constexpr int kSkipTrigger = 3;
constexpr int kMaxSkip = 32;
constexpr unsigned kFullMask = 0xFFFFFFFFu;
constexpr uint32_t kEmptyPos = 0xFFFFFFFFu;

// One parsed sequence: lit_len literals, then a match of ml bytes at
// backward distance off (ml == 0 only for a literals-only tail). Each field
// is at most kMaxChunkSize = 2^20, so all three pack into 21 bits of one
// u64, keeping per-chunk scratch small. Decode rejects any off or ml above
// the chunk size before packing it.
struct SeqRec {
  uint64_t v;
  static constexpr uint64_t kMask = (1ull << 21) - 1;
  __device__ __forceinline__ SeqRec() : v(0) {}
  __device__ __forceinline__ SeqRec(uint32_t lit_len, uint32_t off, uint32_t ml)
      : v((uint64_t)lit_len | ((uint64_t)off << 21) | ((uint64_t)ml << 42)) {}
  __device__ __forceinline__ uint32_t lit_len() const { return (uint32_t)(v & kMask); }
  __device__ __forceinline__ uint32_t off() const { return (uint32_t)((v >> 21) & kMask); }
  __device__ __forceinline__ uint32_t ml() const { return (uint32_t)((v >> 42) & kMask); }
};
static_assert(sizeof(SeqRec) == kSeqRecBytes, "scratch_bytes() (kernels.h) assumes kSeqRecBytes-byte records");
static_assert(kMaxChunkSize <= SeqRec::kMask + 1, "SeqRec's 21-bit fields must hold any chunk-sized value");

__device__ __forceinline__ uint32_t load4(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

__device__ __forceinline__ uint32_t hash_at(const uint8_t* p, int hash_bits) {
  uint32_t v = kMinMatch == 4 ? load4(p) : ((uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16));
  return (v * 2654435761u) >> (32 - hash_bits);
}

__device__ __forceinline__ uint32_t match_len(const uint8_t* in, uint32_t a, uint32_t b, uint32_t max_len) {
  uint32_t l = 0;
  while (l < max_len && in[a + l] == in[b + l]) ++l;
  return l;
}

// Checks all kBucketWays candidates (one u32 chunk-relative position per
// word, bucket[0] = most recently inserted) and keeps the longest match.
__device__ __forceinline__ void probe_bucket(const uint8_t* in, uint32_t p, const uint32_t* bucket,
                                             uint32_t max_len, uint32_t& best_len, uint32_t& best_off) {
#pragma unroll
  for (int i = 0; i < kBucketWays; ++i) {
    uint32_t cand = bucket[i];
    if (cand != kEmptyPos && cand < p) {
      uint32_t l = match_len(in, cand, p, max_len);
      if (l > best_len) {
        best_len = l;
        best_off = p - cand;
      }
    }
  }
}

// What a match costs in the rANS stream, in bits, roughly: a symbol is
// about 5 bits, an explicit offset also carries floor(log2 off) raw bits,
// and a repeat offset carries none (rans_codes.h). Literals are counted at
// 8 bits. Only differences matter here, so the constants are coarse.
constexpr int kBitsSymbol = 5;
constexpr int kBitsLiteral = 8;
__device__ __forceinline__ int match_price(uint32_t ml, uint32_t off, const uint32_t* rep_off, int n_rep) {
  int price = 2 * kBitsSymbol; // length and offset symbols
  bool is_rep = false;
  for (int r = 0; r < kMaxRepProbes; ++r) {
    if (r < n_rep && rep_off[r] == off) is_rep = true;
  }
  if (!is_rep) price += (int)floor_log2(off); // the offset's raw bits
  uint32_t code, nb, bits;
  len_code(ml - kMinMatch, code, nb, bits);
  price += (int)nb;
  return price;
}
// Bits saved by coding `ml` bytes as this match instead of as literals.
__device__ __forceinline__ int match_gain(uint32_t ml, uint32_t off, const uint32_t* rep_off, int n_rep) {
  return (int)(kBitsLiteral * ml) - match_price(ml, off, rep_off, n_rep);
}

__device__ __forceinline__ uint32_t ext_bytes(uint32_t v) { return v >= 15 ? 1 + (v - 15) / 255 : 0; }

__device__ __forceinline__ uint32_t seq_size(uint32_t lit_len, uint32_t ml) {
  uint32_t s = 1 + ext_bytes(lit_len) + lit_len;
  if (ml) s += 4 + ext_bytes(ml - kMinMatch); // 4-byte offset field
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
      out[p++] = (uint8_t)(off >> 16);
      out[p++] = (uint8_t)(off >> 24);
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

// How far a match at p (backward distance off) also extends backwards,
// 32 bytes per step, without reaching before `floor` or the chunk start.
__device__ __forceinline__ uint32_t warp_extend_back(const uint8_t* in, uint32_t p, uint32_t off, uint32_t floor) {
  int lane = threadIdx.x & 31;
  uint32_t max_back = min(p - floor, p - off);
  for (uint32_t back = 0; back < max_back; back += 32) {
    uint32_t k = back + lane; // compares the byte at p - 1 - k
    bool mism = k >= max_back || in[p - 1 - k] != in[p - 1 - k - off];
    unsigned m = __ballot_sync(kFullMask, mism);
    if (m) return back + (__ffs(m) - 1);
  }
  return max_back;
}

// Records sequences and copies literals into the chunk's scratch.
struct SeqEmitter {
  const uint8_t* in;
  SeqRec* seqs;
  uint8_t* lits;
  uint32_t n_seq = 0;
  uint32_t n_lit = 0;
  __device__ __forceinline__ void operator()(uint32_t lit_start, uint32_t lit_len, uint32_t off, uint32_t ml) {
    int lane = threadIdx.x & 31;
    if (lane == 0) seqs[n_seq] = SeqRec{lit_len, off, ml};
    for (uint32_t k = lane; k < lit_len; k += 32) lits[n_lit + k] = in[lit_start + k];
    ++n_seq;
    n_lit += lit_len;
  }
};

// Re-emits recorded sequences as tokens (literals re-read from `in`).
__device__ inline bool tokens_from_seqs(const uint8_t* in, const SeqRec* seqs, uint32_t n_seq, uint8_t* out,
                                        uint32_t cap, uint32_t* out_len) {
  uint32_t op = 0, pos = 0;
  for (uint32_t i = 0; i < n_seq; ++i) {
    SeqRec r = seqs[i];
    if (!emit_seq(in, out, cap, op, pos, r.lit_len(), r.off(), r.ml())) return false;
    pos += r.lit_len() + r.ml();
  }
  *out_len = op;
  return true;
}

// Parses in[0..n) into sequences, calling emit(lit_start, lit_len, off, ml)
// for each match and once more with ml = 0 for any trailing literals.
// htab is this chunk's (1 << hash_bits) * kBucketWays-word table; n_rep
// (at most kMaxRepProbes, see rep_probes() in kernels.h) is how many recent
// match offsets each position also tries.
//
// The warp walks the chunk in 32-byte windows. Every lane hashes its own
// position and probes its bucket's candidates and the recent offsets
// (capped at kProbe bytes so per-lane work is bounded; zstd's parsers
// check their repeat offsets too). One lane per bucket then inserts its
// position, and lanes that found nothing probe again, which catches repeats
// within the same window. The window's matches are selected warp-uniformly
// with a kLazySteps lookahead; matches that hit the probe cap are extended
// cooperatively. After a long stretch without matches the lanes sample
// positions kSkipTrigger-style (see above) until one matches.
__device__ inline void lz_parse_warp(const uint8_t* in, uint32_t n, uint32_t* htab, int hash_bits, int n_rep,
                                     SeqEmitter& emit) {
  int lane = threadIdx.x & 31;
  uint32_t hash_words = (1u << hash_bits) * (uint32_t)kBucketWays;
  for (uint32_t i = lane; i < hash_words; i += 32) htab[i] = kEmptyPos;
  __syncwarp();

  uint32_t lit_start = 0;
  uint32_t pos = 0;
  uint32_t rep[kMaxRepProbes] = {}; // recent distinct offsets, most recent first; 0 = none yet
  // Moves off to the front of rep (warp-uniform): shifts the others down
  // until off's old slot, or the last, is overwritten.
  auto use_offset = [&](uint32_t off) {
    uint32_t carry = off;
#pragma unroll
    for (int r = 0; r < kMaxRepProbes; ++r) {
      uint32_t t = rep[r];
      rep[r] = carry;
      carry = t;
      if (t == off) break;
    }
  };
  uint32_t miss = 0; // windows in a row without a match
  while (pos + kMinMatch <= n) {
    uint32_t stride = min(1u + (miss >> kSkipTrigger), (uint32_t)kMaxSkip);
    uint32_t p = pos + lane * stride;
    bool valid = p + kMinMatch <= n;
    uint32_t h = 0;
    uint32_t bucket[kBucketWays]; // only read after being populated below, when valid
    uint32_t best_len = 0, best_off = 0;
    if (valid) {
      h = hash_at(in + p, hash_bits);
#pragma unroll
      for (int w = 0; w < kBucketWays; ++w) bucket[w] = htab[kBucketWays * h + w];
      uint32_t max_len = min((uint32_t)kProbe, n - p);
      probe_bucket(in, p, bucket, max_len, best_len, best_off);
      // The n_rep most recent match offsets too: a match there codes as a
      // cheap repeat-offset code, so it wins ties with the bucket's
      // candidates, and the more recent offset wins between them.
#pragma unroll
      for (int r = kMaxRepProbes - 1; r >= 0; --r) {
        uint32_t off = rep[r];
        if (r < n_rep && off != 0 && off <= p) {
          uint32_t l = match_len(in, p - off, p, max_len);
          if (l >= (uint32_t)kMinMatch && l >= best_len) {
            best_len = l;
            best_off = off;
          }
        }
      }
    }
    __syncwarp(); // all reads of htab precede any insert

    // One insert per distinct bucket, by its lowest valid lane (so the
    // output is deterministic). The new position becomes the most recent
    // candidate and the oldest is dropped.
    unsigned valid_mask = __ballot_sync(kFullMask, valid);
    unsigned peers = __match_any_sync(kFullMask, h) & valid_mask;
    if (valid && lane == __ffs(peers) - 1) {
#pragma unroll
      for (int w = kBucketWays - 1; w > 0; --w) htab[kBucketWays * h + w] = bucket[w - 1];
      htab[kBucketWays * h] = p;
    }
    __syncwarp();

    if (stride == 1 && valid && best_len < (uint32_t)kMinMatch) {
#pragma unroll
      for (int w = 0; w < kBucketWays; ++w) bucket[w] = htab[kBucketWays * h + w];
      uint32_t max_len = min((uint32_t)kProbe, n - p);
      probe_bucket(in, p, bucket, max_len, best_len, best_off);
    }

    unsigned mask = __ballot_sync(kFullMask, best_len >= (uint32_t)kMinMatch);
    if (stride > 1) {
      // Sampling: take the first match, extend it back over the positions
      // the stride skipped, and resume one position at a time after it.
      if (!mask) {
        pos += 32 * stride;
        ++miss;
        continue;
      }
      uint32_t j = __ffs(mask) - 1;
      uint32_t len = __shfl_sync(kFullMask, best_len, j);
      uint32_t off = __shfl_sync(kFullMask, best_off, j);
      uint32_t start = pos + j * stride;
      if (len == (uint32_t)kProbe) len = warp_extend(in, start, off, n, len);
      uint32_t back = warp_extend_back(in, start, off, lit_start);
      start -= back;
      len += back;
      emit(lit_start, start - lit_start, off, len);
      lit_start = start + len;
      use_offset(off);
      pos = lit_start;
      miss = 0;
      continue;
    }
    miss = mask ? 0 : miss + 1;
    uint32_t cur = 0;
    while (cur < 32) {
      unsigned m = mask & (~0u << cur);
      if (!m) break;
      uint32_t j = __ffs(m) - 1;
      uint32_t lj = __shfl_sync(kFullMask, best_len, j);
      uint32_t oj = __shfl_sync(kFullMask, best_off, j);
      // Lazy matching, priced: take the next position's match when it saves
      // more bits than this one plus the literal that deferring adds (up to
      // kLazySteps positions on). Length alone would ignore what a far
      // offset costs and what a repeat offset saves.
      int gj = match_gain(lj, oj, rep, n_rep);
#pragma unroll
      for (int step = 0; step < kLazySteps; ++step) {
        if (j >= 31 || !((mask >> (j + 1)) & 1)) break;
        uint32_t lj1 = __shfl_sync(kFullMask, best_len, j + 1);
        uint32_t oj1 = __shfl_sync(kFullMask, best_off, j + 1);
        int gj1 = match_gain(lj1, oj1, rep, n_rep);
        if (gj1 <= gj + kBitsLiteral) break;
        ++j;
        lj = lj1;
        oj = oj1;
        gj = gj1;
      }
      uint32_t off = oj;
      uint32_t len = lj;
      if (len == (uint32_t)kProbe) len = warp_extend(in, pos + j, off, n, len);
      emit(lit_start, pos + j - lit_start, off, len);
      lit_start = pos + j + len;
      cur = j + len;
      use_offset(off);
    }
    pos += cur > 32 ? cur : 32;
  }
  if (lit_start < n) emit(lit_start, n - lit_start, 0, 0);
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

    if (ip + 4 > in_len) return false;
    uint32_t off = (uint32_t)in[ip] | ((uint32_t)in[ip + 1] << 8) | ((uint32_t)in[ip + 2] << 16) |
                   ((uint32_t)in[ip + 3] << 24);
    ip += 4;
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

} // namespace gpusqz
