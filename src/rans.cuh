// Warp-interleaved rANS entropy coder for the parsed LZ stream.
//
// 32 rANS states, one per lane, share ONE stream of 16-bit words. Every
// lane encodes/decodes in lockstep; at each step the lanes that need to
// renormalise write (or read) their word contiguously in lane order, found
// with a ballot + popcount, so the stream needs no per-lane offsets. The
// encoder runs in reverse (last symbol first) writing backward from the
// end of the slot; the decoder reads forward. With a 2^16 state floor and
// 16-bit words each step moves at most one word per lane, which is what
// keeps the encoder's and decoder's step-by-step word counts identical.
//
// The frequency table is NOT part of the per-chunk payload: it is shared
// by every chunk in the same TableGroup (see format.h), expanded once
// (RansGroupTable, built by expand_group_table_warp) and referenced from
// global memory by every chunk's decode. This removes the ~490-byte/chunk
// table overhead the earlier per-chunk-table design paid, and also means
// decode no longer rebuilds a table per chunk.
//
// Payload layout (ChunkFlag::LzRans, after the flag byte):
//   u32 n_seq, u32 n_lit, u32 state[32], u16 words[]
// Decode order: sequence groups of 32 (sub-steps ll_code, ll_bits,
// ml_code, ml_bits, off_code, off_bits), then literal groups of 32.
#pragma once
#include <cstdint>
#include <initializer_list>

#include "format.h"
#include "rans_codes.h"
#include "lz_warp.cuh"

namespace gzp {

constexpr int kLitBase = 0;
constexpr int kLlBase = kLitSyms;
constexpr int kMlBase = kLitSyms + kSmallSyms;
constexpr int kOffBase = kLitSyms + 2 * kSmallSyms;

// Coarse index for the 32-symbol alphabets: lut[i] is the largest symbol s
// with cum[s] <= i * (kProbScale >> kSmallLutBits), i.e. a lower bound on
// the true answer for any slot in that bucket. A short forward scan from
// there (never more than a few symbols, since buckets are much narrower
// than a typical run of same-cum zero-frequency symbols) replaces what
// would otherwise be a full 5-step binary search per length/offset code.
constexpr int kSmallLutBits = 7;
constexpr int kSmallLutSize = 1 << kSmallLutBits;

__device__ __forceinline__ void build_small_lut(const uint16_t* cum, uint8_t* lut) {
  int lane = threadIdx.x & 31;
  for (int i = lane; i < kSmallLutSize; i += 32) {
    uint32_t target = (uint32_t)i << (kProbBits - kSmallLutBits);
    uint32_t s = 0;
    for (uint32_t k = 1; k < (uint32_t)kSmallSyms; ++k) {
      if (cum[k] <= target) s = k;
    }
    lut[i] = (uint8_t)s;
  }
}

// One table, shared by every chunk in a TableGroup. Built once per group
// (expand_group_table_warp), then read (never written) by every chunk's
// decode, straight out of global memory — small enough that L2 keeps a
// hot group cached across the many chunks that reference it.
struct RansGroupTable {
  uint8_t sym[kProbScale]; // literal alphabet slot -> symbol
  uint16_t freq[kQuantBytes];
  uint16_t cum[kQuantBytes];
  uint8_t ll_lut[kSmallLutSize];
  uint8_t ml_lut[kSmallLutSize];
  uint8_t off_lut[kSmallLutSize];
};

// Quantises a batch-wide histogram (as accumulated by accumulate_hist) into
// the q[] bytes stored in the file for this TableGroup, and expands them
// into freq/cum for the encoder to use directly. One warp, called once per
// batch (not once per chunk).
__device__ inline void build_batch_table_warp(const uint32_t* cnt, uint8_t* q_out, uint16_t* freq, uint16_t* cum) {
  int lane = threadIdx.x & 31;
  if (lane == 0) {
    quantize_counts(cnt + kLitBase, kLitSyms, q_out + kLitBase);
    normalize_table(q_out + kLitBase, kLitSyms, freq + kLitBase, cum + kLitBase);
    for (int base : {kLlBase, kMlBase, kOffBase}) {
      quantize_counts(cnt + base, kSmallSyms, q_out + base);
      normalize_table(q_out + base, kSmallSyms, freq + base, cum + base);
    }
  }
  __syncwarp();
}

// Expands one group's quantised bytes into a full RansGroupTable. Callers
// must zero-initialize `t` first (or otherwise ensure it starts zeroed):
// a group with no LzRans chunks has an all-zero q[] and normalize_table
// leaves freq/cum untouched on that path, which is fine as long as `t`
// started zeroed and nothing ever decodes against it (guaranteed: a chunk
// only gets ChunkFlag::LzRans if its group's histogram was non-empty).
__device__ inline void expand_group_table_warp(const uint8_t* q, RansGroupTable& t) {
  int lane = threadIdx.x & 31;
  if (lane == 0) {
    normalize_table(q + kLitBase, kLitSyms, t.freq + kLitBase, t.cum + kLitBase);
    for (int base : {kLlBase, kMlBase, kOffBase}) {
      normalize_table(q + base, kSmallSyms, t.freq + base, t.cum + base);
    }
  }
  __syncwarp();
  for (int s = lane; s < kLitSyms; s += 32) {
    uint32_t f = t.freq[kLitBase + s], c = t.cum[kLitBase + s];
    for (uint32_t k = 0; k < f; ++k) t.sym[c + k] = (uint8_t)s;
  }
  build_small_lut(t.cum + kLlBase, t.ll_lut);
  build_small_lut(t.cum + kMlBase, t.ml_lut);
  build_small_lut(t.cum + kOffBase, t.off_lut);
  __syncwarp();
}

__device__ __forceinline__ unsigned lanemask_lt() {
  unsigned m;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
  return m;
}

__device__ __forceinline__ uint32_t load_u32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
__device__ __forceinline__ void store_u32(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}

// Adds one chunk's symbol counts into a batch-wide histogram (kQuantBytes
// entries, indexed like everything else by kLitBase/kLlBase/kMlBase/
// kOffBase) via atomics, so every chunk in a host batch can contribute to
// one shared table. Call once per chunk, all lanes, after that chunk's LZ
// parse has filled `seqs`/`lits`.
__device__ inline void accumulate_hist(const SeqRec* seqs, uint32_t n_seq, const uint8_t* lits, uint32_t n_lit,
                                       uint32_t* batch_cnt) {
  int lane = threadIdx.x & 31;
  for (uint32_t i = lane; i < n_lit; i += 32) atomicAdd(&batch_cnt[kLitBase + lits[i]], 1u);
  for (uint32_t i = lane; i < n_seq; i += 32) {
    SeqRec r = seqs[i];
    uint32_t c, nb, b;
    len_code(r.lit_len, c, nb, b);
    atomicAdd(&batch_cnt[kLlBase + c], 1u);
    len_code(r.ml ? (uint32_t)r.ml - (kMinMatch - 1) : 0, c, nb, b);
    atomicAdd(&batch_cnt[kMlBase + c], 1u);
    if (r.ml) {
      off_code(r.off, c, nb, b);
      atomicAdd(&batch_cnt[kOffBase + c], 1u);
    }
  }
}

// ---- encoder steps (all lanes call; return false uniformly on overflow) ----

__device__ __forceinline__ bool rans_enc_flush(uint32_t& x, bool need, uint16_t*& wp, const uint16_t* wlimit) {
  unsigned m = __ballot_sync(kFullMask, need);
  uint32_t cnt = __popc(m);
  if (wp - cnt < wlimit) return false;
  wp -= cnt;
  if (need) {
    wp[__popc(m & lanemask_lt())] = (uint16_t)(x & 0xFFFF);
    x >>= 16;
  }
  return true;
}

__device__ __forceinline__ bool rans_enc_put(uint32_t& x, bool active, uint32_t f, uint32_t c, uint16_t*& wp,
                                             const uint16_t* wlimit) {
  // Flush when x >= f << 20 (i.e. ((L >> kProbBits) << 16) * f), written
  // as a right shift so f = kProbScale does not overflow.
  bool need = active && (x >> (16 + 16 - kProbBits)) >= f;
  if (!rans_enc_flush(x, need, wp, wlimit)) return false;
  if (active) x = ((x / f) << kProbBits) + (x % f) + c;
  return true;
}

// Encodes at most 16 raw ("bypass") bits in one step. A single flush only
// ever drops one 16-bit word (rans_enc_flush), so nb must not exceed 16
// here — see rans_enc_bits for the wider-nb wrapper around this.
__device__ __forceinline__ bool rans_enc_bits16(uint32_t& x, bool active, uint32_t nb, uint32_t bits, uint16_t*& wp,
                                                const uint16_t* wlimit) {
  bool on = active && nb != 0;
  bool need = on && x >= (1u << (32 - nb));
  if (!rans_enc_flush(x, need, wp, wlimit)) return false;
  if (on) x = (x << nb) | bits;
  return true;
}

// Encodes nb raw ("bypass") bits, nb possibly > 16 now that wide chunks
// give length/offset codes up to kSmallSyms-1, needing up to ~19 extra
// bits (kSmallLutBits assumes < 32; this comfortably fits under that).
// Split into an explicit low-16-bit step (encoded first) and a high-
// remainder step (encoded second, at most 16 bits since nb is capped well
// under 32); decode reads in the opposite call order, so it sees the high
// chunk first and the low chunk second, matching how it reassembles them.
//
// Both rans_enc_bits16 calls below are made by every lane every time
// (their own nb may be 0, a harmless no-op), never skipped based on this
// lane's own nb: rans_enc_bits16 -> rans_enc_flush -> __ballot_sync
// requires every lane in the mask to reach the same call, so nb (which
// can differ per lane, since it comes from each lane's own decoded
// symbol) must never gate which calls execute, only what they encode.
__device__ __forceinline__ bool rans_enc_bits(uint32_t& x, bool active, uint32_t nb, uint32_t bits, uint16_t*& wp,
                                              const uint16_t* wlimit) {
  uint32_t lo_nb = nb > 16 ? 16 : nb;
  uint32_t hi_nb = nb > 16 ? nb - 16 : 0;
  if (!rans_enc_bits16(x, active, lo_nb, bits & 0xFFFFu, wp, wlimit)) return false;
  return rans_enc_bits16(x, active, hi_nb, bits >> 16, wp, wlimit);
}

// Encodes a chunk already parsed into `seqs`/`lits` against a table shared
// by its whole batch (freq/cum, kQuantBytes entries, indexed by
// kLitBase/kLlBase/kMlBase/kOffBase as usual). Words are written backward
// from the end of the slot and the header is placed right before them, so
// the payload starts at *out_start within the slot. The caller guarantees
// in_len > kRansHeaderBytes + 1.
__device__ inline bool rans_encode_warp(const SeqRec* seqs, uint32_t n_seq, const uint8_t* lits, uint32_t n_lit,
                                        const uint16_t* freq, const uint16_t* cum, uint8_t* slot,
                                        uint32_t slot_stride, uint32_t in_len, uint32_t* out_start,
                                        uint32_t* out_size) {
  int lane = threadIdx.x & 31;

  // The whole payload must not beat raw storage, so the stream may use at
  // most in_len - flag - header bytes; wlimit is where that runs out.
  uint32_t max_stream = in_len - 1 - kRansHeaderBytes;
  uint16_t* slot_end = reinterpret_cast<uint16_t*>(slot + slot_stride);
  const uint16_t* wlimit = slot_end - max_stream / 2;
  uint16_t* wp = slot_end;
  uint32_t x = kRansL;

  for (int g = (int)((n_lit + 31) / 32) - 1; g >= 0; --g) {
    uint32_t idx = g * 32 + lane;
    bool act = idx < n_lit;
    uint32_t s = act ? lits[idx] : 0;
    if (!rans_enc_put(x, act, freq[kLitBase + s], cum[kLitBase + s], wp, wlimit)) return false;
  }
  for (int g = (int)((n_seq + 31) / 32) - 1; g >= 0; --g) {
    uint32_t idx = g * 32 + lane;
    bool act = idx < n_seq;
    SeqRec r{0, 0, 0};
    if (act) r = seqs[idx];
    uint32_t ml = r.ml, off = r.off;
    uint32_t llc, llnb, llb, mlc, mlnb, mlb, oc = 0, onb = 0, ob = 0;
    len_code(r.lit_len, llc, llnb, llb);
    len_code(ml ? ml - (kMinMatch - 1) : 0, mlc, mlnb, mlb);
    bool has_off = act && ml != 0;
    if (has_off) off_code(off, oc, onb, ob);

    if (!rans_enc_bits(x, has_off, onb, ob, wp, wlimit)) return false;
    if (!rans_enc_put(x, has_off, freq[kOffBase + oc], cum[kOffBase + oc], wp, wlimit)) return false;
    if (!rans_enc_bits(x, act, mlnb, mlb, wp, wlimit)) return false;
    if (!rans_enc_put(x, act, freq[kMlBase + mlc], cum[kMlBase + mlc], wp, wlimit)) return false;
    if (!rans_enc_bits(x, act, llnb, llb, wp, wlimit)) return false;
    if (!rans_enc_put(x, act, freq[kLlBase + llc], cum[kLlBase + llc], wp, wlimit)) return false;
  }

  uint8_t* hdr = reinterpret_cast<uint8_t*>(wp) - kRansHeaderBytes;
  uint8_t* start = hdr - 1;
  if (lane == 0) {
    start[0] = (uint8_t)ChunkFlag::LzRans;
    store_u32(hdr, n_seq);
    store_u32(hdr + 4, n_lit);
  }
  store_u32(hdr + 8 + 4 * lane, x);
  *out_start = (uint32_t)(start - slot);
  *out_size = (uint32_t)((slot + slot_stride) - start);
  return true;
}

// ---- decoder steps ----

__device__ __forceinline__ bool rans_dec_renorm(uint32_t& x, bool active, const uint8_t*& rp, const uint8_t* rend) {
  bool need = active && x < kRansL;
  unsigned m = __ballot_sync(kFullMask, need);
  uint32_t cnt = __popc(m);
  if (rp + 2 * cnt > rend) return false;
  if (need) {
    const uint8_t* p = rp + 2 * __popc(m & lanemask_lt());
    x = (x << 16) | ((uint32_t)p[0] | ((uint32_t)p[1] << 8));
  }
  rp += 2 * cnt;
  return true;
}

__device__ __forceinline__ uint32_t rans_dec_lit(uint32_t& x, const RansGroupTable& t) {
  uint32_t slot = x & (kProbScale - 1);
  uint32_t s = t.sym[slot];
  x = t.freq[kLitBase + s] * (x >> kProbBits) + slot - t.cum[kLitBase + s];
  return s;
}

// Small alphabets: look up a lower-bound symbol from the coarse LUT, then
// scan forward to the exact answer (skipping zero-frequency symbols,
// which share cum with their successor).
__device__ __forceinline__ uint32_t rans_dec_small(uint32_t& x, const uint16_t* freq, const uint16_t* cum,
                                                   const uint8_t* lut) {
  uint32_t slot = x & (kProbScale - 1);
  uint32_t s = lut[slot >> (kProbBits - kSmallLutBits)];
  while (s + 1 < (uint32_t)kSmallSyms && cum[s + 1] <= slot) ++s;
  x = freq[s] * (x >> kProbBits) + slot - cum[s];
  return s;
}

// Decodes at most 16 raw ("bypass") bits in one step, renormalising once
// afterward. See rans_dec_bits for the wider-nb wrapper around this.
__device__ __forceinline__ bool rans_dec_bits16(uint32_t& x, bool active, uint32_t nb, uint32_t& out,
                                                const uint8_t*& rp, const uint8_t* rend) {
  bool on = active && nb != 0;
  out = on ? (x & ((1u << nb) - 1)) : 0;
  if (on) x >>= nb;
  return rans_dec_renorm(x, on, rp, rend);
}

// Decodes nb raw ("bypass") bits, renormalising as needed. Mirrors
// rans_enc_bits's split: for nb > 16 (possible now that wide chunks give
// length/offset codes needing up to ~19 extra bits), a single renorm can
// only ever supply one 16-bit word, so the high (nb-16) bits are decoded
// first and the low 16 bits second — the reverse of encode's call order,
// which is what lets the two sides' bit chunks line up.
//
// Both rans_dec_bits16 calls below are made by every lane every time
// (their own width may be 0, a harmless no-op), never skipped based on
// this lane's own nb: rans_dec_bits16 -> rans_dec_renorm -> __ballot_sync
// requires every lane in the mask to reach the same call, so nb (which
// can differ per lane, since it comes from each lane's own decoded
// symbol) must never gate which calls execute, only what they decode.
__device__ __forceinline__ bool rans_dec_bits(uint32_t& x, bool active, uint32_t nb, uint32_t& out,
                                              const uint8_t*& rp, const uint8_t* rend) {
  uint32_t hi_nb = nb > 16 ? nb - 16 : 0;
  uint32_t lo_nb = nb > 16 ? 16 : nb;
  uint32_t hi = 0, lo = 0;
  if (!rans_dec_bits16(x, active, hi_nb, hi, rp, rend)) return false;
  if (!rans_dec_bits16(x, active, lo_nb, lo, rp, rend)) return false;
  out = (hi << 16) | lo;
  return true;
}

// Decodes the sequence and literal streams into scratch, using an
// already-expanded group table `t`. Rejects any malformed input (empty
// tables in use, truncated or over-long streams) without touching memory
// beyond the scratch bounds.
__device__ inline bool rans_decode_warp(const uint8_t* payload, uint32_t len, const RansGroupTable& t, SeqRec* seqs,
                                        uint32_t max_seq, uint8_t* lits, uint32_t max_lit, uint32_t* n_seq_out,
                                        uint32_t* n_lit_out) {
  int lane = threadIdx.x & 31;
  if (len < (uint32_t)kRansHeaderBytes) return false;
  uint32_t n_seq = load_u32(payload), n_lit = load_u32(payload + 4);
  if (n_seq > max_seq || n_lit > max_lit) return false;

  uint32_t x = load_u32(payload + 8 + 4 * lane);
  const uint8_t* rp = payload + kRansHeaderBytes;
  const uint8_t* rend = payload + len;
  bool bad = false;

  for (uint32_t g = 0; g < (n_seq + 31) / 32; ++g) {
    uint32_t idx = g * 32 + lane;
    bool act = idx < n_seq;
    uint32_t llc = 0, mlc = 0, oc = 0, llb = 0, mlb = 0, ob = 0;

    if (act) {
      llc = rans_dec_small(x, t.freq + kLlBase, t.cum + kLlBase, t.ll_lut);
      bad |= t.freq[kLlBase + llc] == 0;
    }
    if (!rans_dec_renorm(x, act, rp, rend)) return false;
    if (!rans_dec_bits(x, act, len_nb(llc), llb, rp, rend)) return false;

    if (act) {
      mlc = rans_dec_small(x, t.freq + kMlBase, t.cum + kMlBase, t.ml_lut);
      bad |= t.freq[kMlBase + mlc] == 0;
    }
    if (!rans_dec_renorm(x, act, rp, rend)) return false;
    if (!rans_dec_bits(x, act, len_nb(mlc), mlb, rp, rend)) return false;

    uint32_t mlv = len_value(mlc, mlb);
    uint32_t ml = mlv ? mlv + (kMinMatch - 1) : 0;
    bool has_off = act && ml != 0;
    if (has_off) {
      oc = rans_dec_small(x, t.freq + kOffBase, t.cum + kOffBase, t.off_lut);
      // oc must stay < 32 regardless of chunk size: it's a shift count
      // below (nb = oc), and rans_dec_bits(x, 32) would be undefined.
      bad |= t.freq[kOffBase + oc] == 0 || oc >= (uint32_t)kSmallSyms;
    }
    if (!rans_dec_renorm(x, has_off, rp, rend)) return false;
    if (!rans_dec_bits(x, has_off, oc, ob, rp, rend)) return false;

    if (act) {
      if (ml == 0 && idx != n_seq - 1) bad = true;
      if (ml > max_lit) bad = true; // a match can never be longer than the chunk itself
      seqs[idx] = SeqRec{len_value(llc, llb), has_off ? off_value(oc, ob) : 0, ml};
    }
    if (__any_sync(kFullMask, bad)) return false;
  }

  for (uint32_t g = 0; g < (n_lit + 31) / 32; ++g) {
    uint32_t idx = g * 32 + lane;
    bool act = idx < n_lit;
    if (act) {
      uint32_t s = rans_dec_lit(x, t);
      bad |= t.freq[kLitBase + s] == 0;
      lits[idx] = (uint8_t)s;
    }
    if (!rans_dec_renorm(x, act, rp, rend)) return false;
  }
  if (__any_sync(kFullMask, bad)) return false;
  if (rp != rend) return false;
  *n_seq_out = n_seq;
  *n_lit_out = n_lit;
  return true;
}

// Rebuilds the chunk from decoded sequences and literals.
__device__ inline bool lz_reconstruct_warp(const SeqRec* seqs, uint32_t n_seq, const uint8_t* lits, uint32_t n_lit,
                                           uint8_t* out, uint32_t orig) {
  int lane = threadIdx.x & 31;
  uint32_t op = 0, lp = 0;
  for (uint32_t i = 0; i < n_seq; ++i) {
    SeqRec r = seqs[i];
    if (lp + r.lit_len > n_lit || op + r.lit_len > orig) return false;
    for (uint32_t k = lane; k < r.lit_len; k += 32) out[op + k] = lits[lp + k];
    op += r.lit_len;
    lp += r.lit_len;
    uint32_t ml = r.ml, off = r.off;
    if (ml) {
      if (off == 0 || off > op || op + ml > orig) return false;
      __syncwarp();
      const uint8_t* src = out + op - off;
      if (off >= ml) {
        for (uint32_t k = lane; k < ml; k += 32) out[op + k] = src[k];
      } else {
        for (uint32_t k = lane; k < ml; k += 32) out[op + k] = src[k % off];
      }
      __syncwarp();
      op += ml;
    }
  }
  return op == orig && lp == n_lit;
}

} // namespace gzp
