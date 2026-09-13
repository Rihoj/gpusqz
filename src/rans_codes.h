// Symbol alphabets and frequency tables for the rANS stage. Plain C++ so
// the CPU reference decoder shares it with the GPU code; both sides must
// derive bit-identical tables from the same quantised counts.
//
// Alphabets (zstd-style, so tables stay small):
//   literal      byte value, 256 symbols, coded with one of lit_ctx_count()
//                tables chosen by the previous literal (see lit_ctx below)
//   lit_len v    v < 16: code v, no extra bits
//                else:   code 12 + floor(log2 v), floor(log2 v) extra bits
//   match_len    same coding of v = ml - (kMinMatch - 1) (v = 0 marks a
//                literals-only final sequence; real matches give v >= 1)
//   offset       code floor(log2 off), that many extra bits; the top 3
//                codes instead mean "reuse the 1st/2nd/3rd most recent
//                distinct offset" (zstd-style repeat offsets, no extra bits)
// Extra bits are written raw (rANS bypass), so there is no side stream.
#pragma once
#include <cstdint>

namespace gpusqz {

constexpr int kProbBits = 12;
constexpr uint32_t kProbScale = 1u << kProbBits;
constexpr uint32_t kRansL = 1u << 16; // state lower bound; 16-bit output words
constexpr int kLitSyms = 256;
constexpr int kSmallSyms = 32; // lit_len / match_len / offset alphabets (29, 28, 29 used)
// Offset codes >= kOffRepBase are repeat-offset codes. A real off_code() is
// at most floor_log2(kMaxChunkSize) = 20, so they never collide.
constexpr uint32_t kOffRepBase = kSmallSyms - 3;
static_assert(kOffRepBase > 20, "repeat-offset codes must exceed any real off_code() for kMaxChunkSize");
// Largest lit_len / match_len len_code() can represent (its code must stay
// below kSmallSyms). Only a match-free kMaxChunkSize chunk, one literal run
// of exactly 2^20, exceeds it; such a chunk is stored as Lz tokens instead.
constexpr uint32_t kMaxLenValue = (1u << (kSmallSyms - 12)) - 1;
constexpr int kRansStates = 32;
// Per-chunk rANS payload header after the flag byte: n_seq, n_lit, states.
constexpr int kRansHeaderBytes = 8 + kRansStates * 4;

#ifdef __CUDACC__
#define GPUSQZ_HD __host__ __device__ inline
#else
#define GPUSQZ_HD inline
#endif

// ---- Order-1 literal contexts ----
//
// Literals are coded with one of lit_ctx_count(shift) tables chosen by the
// previous literal: ctx = prev >> shift, so shift 8 is order-0 (1 table),
// 4 keys on the previous byte's high nibble (16) and 0 on the whole byte
// (256). Each table group picks its shift (see build_table_kernel).
//
// So the decoder knows the previous literal, each of the 32 rANS lanes owns
// one contiguous run of the chunk's literals: lane j owns [j*L, (j+1)*L)
// with L = lit_run_len(n_lit), and each run's first literal uses context 0.
// L is a multiple of 4 so runs start 4-byte aligned (the decoder stores 4
// literals at a time). The histogram, the encoder and both decoders must
// all follow exactly this rule.
constexpr uint32_t kLitShiftOrder0 = 8, kLitShiftNibble = 4, kLitShiftByte = 0;
GPUSQZ_HD bool lit_shift_valid(uint32_t shift) {
  return shift == kLitShiftOrder0 || shift == kLitShiftNibble || shift == kLitShiftByte;
}
GPUSQZ_HD int lit_ctx_count(uint32_t shift) { return 256 >> shift; }
GPUSQZ_HD uint32_t lit_ctx(uint32_t prev, uint32_t shift) { return prev >> shift; }
GPUSQZ_HD uint32_t lit_run_len(uint32_t n_lit) { return ((n_lit + 31) / 32 + 3) & ~3u; }

// Layout of every quantised-count / frequency array: the three small
// alphabets first, then one 256-entry literal table per context, so the
// number of contexts moves nothing else.
constexpr int kLlBase = 0;
constexpr int kMlBase = kSmallSyms;
constexpr int kOffBase = 2 * kSmallSyms;
constexpr int kLitBase = 3 * kSmallSyms;
constexpr int kMaxLitCtx = 256;
GPUSQZ_HD int lit_entry(uint32_t ctx, uint32_t sym) { return kLitBase + (int)(ctx * kLitSyms + sym); }
GPUSQZ_HD int quant_bytes(int n_ctx) { return kLitBase + n_ctx * kLitSyms; }
constexpr int kMaxQuantBytes = kLitBase + kMaxLitCtx * kLitSyms;

GPUSQZ_HD uint32_t floor_log2(uint32_t v) { // v >= 1
#if defined(__CUDA_ARCH__)
  return 31 - __clz(v);
#else
  uint32_t r = 0;
  while (v >>= 1) ++r;
  return r;
#endif
}

GPUSQZ_HD void len_code(uint32_t v, uint32_t& code, uint32_t& nb, uint32_t& bits) {
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
GPUSQZ_HD uint32_t len_nb(uint32_t code) { return code < 16 ? 0 : code - 12; }
GPUSQZ_HD uint32_t len_value(uint32_t code, uint32_t bits) { return code < 16 ? code : (1u << (code - 12)) + bits; }

GPUSQZ_HD void off_code(uint32_t off, uint32_t& code, uint32_t& nb, uint32_t& bits) {
  uint32_t l = floor_log2(off);
  code = l;
  nb = l;
  bits = off - (1u << l);
}
GPUSQZ_HD uint32_t off_value(uint32_t code, uint32_t bits) { return (1u << code) + bits; }
// Bypass-bit width for a decoded offset code: for a real off_code() this
// is the code itself (nb == floor(log2 off), see off_code above); a
// repeat-offset code (>= kOffRepBase) carries no extra bits at all.
GPUSQZ_HD uint32_t off_nb(uint32_t code) { return code >= kOffRepBase ? 0 : code; }

// Quantises counts to one byte each: zero iff absent, else 1..255 scaled
// to the largest count.
GPUSQZ_HD void quantize_counts(const uint32_t* cnt, int k, uint8_t* q) {
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
GPUSQZ_HD bool normalize_table(const uint8_t* q, int k, uint16_t* freq, uint16_t* cum) {
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
      // Round to nearest (flooring made a flat histogram a noisy mix of 15s
      // and 16s out of 4096).
      f = (uint32_t)((((uint64_t)q[s] << kProbBits) + total / 2) / total);
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

#undef GPUSQZ_HD

} // namespace gpusqz
