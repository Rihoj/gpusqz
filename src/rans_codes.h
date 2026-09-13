// Symbol alphabets and frequency tables for the rANS stage. Plain C++ so
// the CPU reference decoder shares it with the GPU code; both sides must
// derive bit-identical tables from the same quantised counts.
//
// Alphabets (zstd-style, so tables stay small):
//   literal      byte value, 256 symbols
//   lit_len v    v < 16: code v, no extra bits
//                else:   code 12 + floor(log2 v), floor(log2 v) extra bits
//   match_len    same coding of v = ml - (kMinMatch - 1) (v = 0 marks a
//                literals-only final sequence; real matches give v >= 1)
//   offset       code floor(log2 off), that many extra bits -- except the
//                top 3 codes (kOffRepBase..kOffRepBase+2), which are never
//                a real floor(log2 off) for any offset this format allows
//                and instead mean "reuse the 1st/2nd/3rd most-recently-used
//                distinct match offset" (zstd-style repeat offsets), with
//                zero extra bits: see kOffRepBase below and
//                compute_repeat_codes()/its decode-side mirror in rans.cuh.
// Extra bits are written raw (rANS bypass), so there is no side stream.
#pragma once
#include <cstdint>

namespace gzp {

constexpr int kProbBits = 12;
constexpr uint32_t kProbScale = 1u << kProbBits;
constexpr uint32_t kRansL = 1u << 16; // state lower bound; 16-bit output words
constexpr int kLitSyms = 256;
constexpr int kSmallSyms = 32; // lit_len / match_len / offset alphabets (29, 28, 29 used)
// Offset codes >= kOffRepBase are repeat-offset codes, not real
// floor(log2 off) values: kMaxChunkSize (1MB, format.h) caps a real
// off_code() at floor_log2(1<<20) = 20, comfortably below this.
constexpr uint32_t kOffRepBase = kSmallSyms - 3;
static_assert(kOffRepBase > 20, "repeat-offset codes must exceed any real off_code() for kMaxChunkSize");
constexpr int kRansStates = 32;
constexpr int kQuantBytes = kLitSyms + 3 * kSmallSyms;
// Payload header after the chunk flag: n_seq, n_lit, states. The quantised
// tables live once per table group (see format.h's TableGroup), not per
// chunk, so they are not part of this per-chunk header.
constexpr int kRansHeaderBytes = 8 + kRansStates * 4;

#ifdef __CUDACC__
#define GZP_HD __host__ __device__ inline
#else
#define GZP_HD inline
#endif

GZP_HD uint32_t floor_log2(uint32_t v) { // v >= 1
#if defined(__CUDA_ARCH__)
  return 31 - __clz(v);
#else
  uint32_t r = 0;
  while (v >>= 1) ++r;
  return r;
#endif
}

GZP_HD void len_code(uint32_t v, uint32_t& code, uint32_t& nb, uint32_t& bits) {
  if (v < 16) {
    code = v;
    nb = 0;
    bits = 0;
  } else {
    uint32_t l = floor_log2(v);
    code = 12 + l;
    nb = l;
    bits = v - (1u << l);
  }
}
GZP_HD uint32_t len_nb(uint32_t code) { return code < 16 ? 0 : code - 12; }
GZP_HD uint32_t len_value(uint32_t code, uint32_t bits) { return code < 16 ? code : (1u << (code - 12)) + bits; }

GZP_HD void off_code(uint32_t off, uint32_t& code, uint32_t& nb, uint32_t& bits) {
  uint32_t l = floor_log2(off);
  code = l;
  nb = l;
  bits = off - (1u << l);
}
GZP_HD uint32_t off_value(uint32_t code, uint32_t bits) { return (1u << code) + bits; }
// Bypass-bit width for a decoded offset code: for a real off_code() this
// is the code itself (nb == floor(log2 off), see off_code above); a
// repeat-offset code (>= kOffRepBase) carries no extra bits at all.
GZP_HD uint32_t off_nb(uint32_t code) { return code >= kOffRepBase ? 0 : code; }

// Quantises counts to one byte each for the header: zero iff absent,
// otherwise 1..255 scaled to the largest count.
GZP_HD void quantize_counts(const uint32_t* cnt, int k, uint8_t* q) {
  uint32_t mx = 0;
  for (int s = 0; s < k; ++s) mx = cnt[s] > mx ? cnt[s] : mx;
  for (int s = 0; s < k; ++s) {
    if (cnt[s] == 0) {
      q[s] = 0;
    } else {
      uint32_t v = (uint32_t)(((uint64_t)cnt[s] * 255 + mx / 2) / mx);
      q[s] = (uint8_t)(v < 1 ? 1 : v);
    }
  }
}

// Expands quantised bytes into frequencies summing to exactly kProbScale,
// every present symbol getting at least 1. Deterministic integer math so
// encoder and decoder agree. Returns false if no symbol is present.
GZP_HD bool normalize_table(const uint8_t* q, int k, uint16_t* freq, uint16_t* cum) {
  uint32_t total = 0;
  for (int s = 0; s < k; ++s) total += q[s];
  if (total == 0) {
    for (int s = 0; s < k; ++s) freq[s] = cum[s] = 0;
    return false;
  }
  int sum = 0;
  for (int s = 0; s < k; ++s) {
    uint32_t f = 0;
    if (q[s]) {
      f = (uint32_t)(((uint64_t)q[s] << kProbBits) / total);
      if (f < 1) f = 1;
    }
    freq[s] = (uint16_t)f;
    sum += (int)f;
  }
  int adj = (int)kProbScale - sum;
  // Hand the rounding remainder to the most frequent symbol; if we
  // overshot, trim the largest symbols one at a time, never below 1.
  while (adj != 0) {
    int amax = 0;
    for (int s = 1; s < k; ++s) {
      if (freq[s] > freq[amax]) amax = s;
    }
    if (adj > 0) {
      freq[amax] = (uint16_t)(freq[amax] + adj);
      adj = 0;
    } else {
      if (freq[amax] <= 1) return false; // cannot happen with k <= 256; guard anyway
      freq[amax]--;
      adj++;
    }
  }
  uint32_t c = 0;
  for (int s = 0; s < k; ++s) {
    cum[s] = (uint16_t)c;
    c += freq[s];
  }
  return true;
}

#undef GZP_HD

} // namespace gzp
