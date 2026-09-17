// Warp-interleaved rANS entropy coder for the parsed LZ stream.
//
// 32 rANS states, one per lane, share ONE stream of 16-bit words. All lanes
// step in lockstep; at each step the lanes that need to renormalise write
// (or read) their word contiguously in lane order, located with a ballot
// and a popcount, so the stream needs no per-lane offsets. The encoder runs
// in reverse, writing backward from the end of the slot; the decoder reads
// forward. With a 2^16 state floor and 16-bit words, each step moves at
// most one word per lane, which keeps both sides' word counts in step.
//
// Tables are not in the per-chunk payload: every chunk of a TableGroup
// shares them, expanded per decode batch for the groups it touches
// (expand_group_table).
//
// Payload (ChunkFlag::LzRans, after the flag byte):
//   u32 n_seq, u32 n_lit, u32 state[32], u16 words[]
// Decode order: sequences in groups of 32, lane j owning sequence 32g+j
// (sub-steps ll_code, ll_bits, ml_code, ml_bits, off_code, off_bits), then
// the literals, lane j owning one contiguous run (see lit_run_len).
//
// Repeat offsets (kOffRepBase) need the most-recent-offsets state at every
// sequence, which is inherently sequential: the encoder computes it in one
// serial pass (compute_repeat_codes), and the decoder replays it per group
// of 32 with shuffles. Both use RepOffsets.
#pragma once
#include <cstdint>

#include "format.h"
#include "rans_codes.h"
#include "lz_warp.cuh"

namespace gpusqz {

// Coarse index for the 32-symbol alphabets: lut[i] is the largest symbol s
// with cum[s] <= i * (kProbScale >> kSmallLutBits), i.e. a lower bound on
// the true answer for any slot in that bucket. A short forward scan from
// there (never more than a few symbols, since buckets are much narrower
// than a typical run of same-cum zero-frequency symbols) replaces what
// would otherwise be a full 5-step binary search per length/offset code.
constexpr int kSmallLutBits = 7;
constexpr int kSmallLutSize = 1 << kSmallLutBits;

// Called by every thread of the block; entries are split across threads.
__device__ __forceinline__ void build_small_lut(const uint16_t* cum, uint8_t* lut) {
  for (int i = threadIdx.x; i < kSmallLutSize; i += blockDim.x) {
    uint32_t target = (uint32_t)i << (kProbBits - kSmallLutBits);
    uint32_t s = 0;
    for (uint32_t k = 1; k < (uint32_t)kSmallSyms; ++k) {
      if (cum[k] <= target) s = k;
    }
    lut[i] = (uint8_t)s;
  }
}

// One group's decode tables, built once (expand_group_table) and read by
// every chunk of the group straight from global memory. Their size depends
// on the group's literal context count (group_table_bytes); table_view()
// finds the parts within the group's region:
//   sym[n_ctx][kProbScale]   literal slot -> symbol, per context
//   freq, cum[quant_bytes(n_ctx)]   indexed like the quantised counts
//   ll_lut, ml_lut, off_lut[kSmallLutSize]
// At 256 contexts that is ~1.2MB per group, which L2 (32MB here) holds for
// the few groups a decode batch touches.
struct RansTableView {
  uint8_t* sym;
  uint16_t* freq;
  uint16_t* cum;
  uint8_t* ll_lut;
  uint8_t* ml_lut;
  uint8_t* off_lut;
};

__host__ __device__ inline size_t align16(size_t v) { return (v + 15) & ~(size_t)15; }
__host__ __device__ inline size_t group_table_bytes(int n_ctx) {
  size_t sym = align16((size_t)n_ctx << kProbBits);
  size_t fc = align16(sizeof(uint16_t) * (size_t)quant_bytes(n_ctx));
  return sym + 2 * fc + align16(3 * kSmallLutSize);
}
__device__ __forceinline__ RansTableView table_view(uint8_t* base, int n_ctx) {
  size_t sym = align16((size_t)n_ctx << kProbBits);
  size_t fc = align16(sizeof(uint16_t) * (size_t)quant_bytes(n_ctx));
  RansTableView v;
  v.sym = base;
  v.freq = reinterpret_cast<uint16_t*>(base + sym);
  v.cum = reinterpret_cast<uint16_t*>(base + sym + fc);
  v.ll_lut = base + sym + 2 * fc;
  v.ml_lut = v.ll_lut + kSmallLutSize;
  v.off_lut = v.ml_lut + kSmallLutSize;
  return v;
}

// Table t of n_ctx + 3 (the literal contexts, then the three small
// alphabets): its base and size in the rans_codes.h layout.
__device__ __forceinline__ void table_row(int t, int n_ctx, int& base, int& k) {
  if (t < n_ctx) {
    base = lit_entry((uint32_t)t, 0);
    k = kLitSyms;
  } else {
    base = (t - n_ctx) * kSmallSyms; // kLlBase, kMlBase, kOffBase
    k = kSmallSyms;
  }
}

// Quantises a batch histogram into the counts stored in the file and
// expands them into the encoder's freq/cum. One block; each thread handles
// whole tables.
__device__ inline void build_batch_table(const uint32_t* cnt, int n_ctx, uint8_t* q_out, uint16_t* freq,
                                         uint16_t* cum) {
  for (int t = threadIdx.x; t < n_ctx + 3; t += blockDim.x) {
    int base, k;
    table_row(t, n_ctx, base, k);
    quantize_counts(cnt + base, k, q_out + base);
    normalize_table(q_out + base, k, freq + base, cum + base);
  }
}

// Expands one group's quantised counts into its decode tables; one block
// per group. The region must be zeroed first (see launch_expand_group_tables).
__device__ inline void expand_group_table(const uint8_t* q, int n_ctx, uint8_t* region) {
  RansTableView v = table_view(region, n_ctx);
  for (int t = threadIdx.x; t < n_ctx + 3; t += blockDim.x) {
    int base, k;
    table_row(t, n_ctx, base, k);
    normalize_table(q + base, k, v.freq + base, v.cum + base);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < n_ctx * kLitSyms; i += blockDim.x) {
    uint32_t ctx = (uint32_t)i / kLitSyms, s = (uint32_t)i % kLitSyms;
    uint32_t f = v.freq[kLitBase + i], c = v.cum[kLitBase + i];
    uint8_t* row = v.sym + ((size_t)ctx << kProbBits);
    for (uint32_t k = 0; k < f; ++k) row[c + k] = (uint8_t)s;
  }
  build_small_lut(v.cum + kLlBase, v.ll_lut);
  build_small_lut(v.cum + kMlBase, v.ml_lut);
  build_small_lut(v.cum + kOffBase, v.off_lut);
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

// The 3 most recent distinct match offsets (zstd-style repeat offsets).
// Slot 0..2 means "reuse that recent offset"; kRepNew means a new one.
constexpr uint32_t kRepNew = 3;
struct RepOffsets {
  uint32_t r0 = 0, r1 = 0, r2 = 0; // 0 never matches: real offsets are >= 1
  __device__ __forceinline__ uint32_t find(uint32_t off) const {
    return off == r0 ? 0 : off == r1 ? 1 : off == r2 ? 2 : kRepNew;
  }
  __device__ __forceinline__ uint32_t get(uint32_t slot) const { return slot == 0 ? r0 : slot == 1 ? r1 : r2; }
  // Moves `off`, found in `slot` (or kRepNew), to the front. Written as
  // selects, not branches: the decoder runs this 32 times per group.
  __device__ __forceinline__ void use(uint32_t slot, uint32_t off) {
    bool front = slot == 0;
    r2 = slot >= 2 ? r1 : r2;
    r1 = front ? r1 : r0;
    r0 = front ? r0 : off;
  }
};

// Writes rep_code[i] for every match sequence: 0 = code the offset
// explicitly, 1..3 = reuse the 1st/2nd/3rd recent offset. Serial, on lane
// 0: the state at sequence i depends on every offset before it.
__device__ inline void compute_repeat_codes(const SeqRec* seqs, uint32_t n_seq, uint8_t* rep_code) {
  int lane = threadIdx.x & 31;
  if (lane == 0) {
    RepOffsets rep;
    for (uint32_t i = 0; i < n_seq; ++i) {
      uint32_t off = seqs[i].off();
      uint8_t code = 0;
      if (seqs[i].ml()) {
        uint32_t slot = rep.find(off);
        if (slot != kRepNew) code = (uint8_t)(slot + 1);
        rep.use(slot, off);
      }
      rep_code[i] = code;
    }
  }
  __syncwarp();
}

// Adds one chunk's symbol counts to the batch histogram (kMaxQuantBytes
// entries: the full order-1 literal histogram, which build_table_kernel
// folds to the batch's context rule). Callers skip chunks that can't be
// rANS-coded (see parse_hist_kernel), so every lit_len fits the alphabet.
__device__ inline void accumulate_hist(const SeqRec* seqs, uint32_t n_seq, const uint8_t* rep_code,
                                       const uint8_t* lits, uint32_t n_lit, uint32_t* batch_cnt) {
  int lane = threadIdx.x & 31;
  // Literals: each lane walks its own run, as the encoder will.
  uint32_t run_len = lit_run_len(n_lit);
  uint32_t run0 = lane * run_len, run_end = min(n_lit, run0 + run_len);
  uint32_t prev = 0;
  for (uint32_t i = run0; i < run_end; ++i) {
    uint32_t sym = lits[i];
    atomicAdd(&batch_cnt[lit_entry(lit_ctx(prev, kLitShiftByte), sym)], 1u);
    prev = sym;
  }
  for (uint32_t i = lane; i < n_seq; i += 32) {
    SeqRec r = seqs[i];
    uint32_t c, nb, b;
    len_code(r.lit_len(), c, nb, b);
    atomicAdd(&batch_cnt[kLlBase + c], 1u);
    len_code(r.ml() ? (uint32_t)r.ml() - (kMinMatch - 1) : 0, c, nb, b);
    atomicAdd(&batch_cnt[kMlBase + c], 1u);
    if (r.ml()) {
      uint8_t rc = rep_code[i];
      if (rc) {
        c = kOffRepBase + (rc - 1);
      } else {
        off_code(r.off(), c, nb, b);
      }
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

// Encodes nb (up to 20) raw bits as a low-16-bit step, then the rest; the
// decoder reads them in the opposite order. Every lane makes both calls
// whatever its own nb (0 is a no-op): each contains a ballot that every
// lane must reach.
__device__ __forceinline__ bool rans_enc_bits(uint32_t& x, bool active, uint32_t nb, uint32_t bits, uint16_t*& wp,
                                              const uint16_t* wlimit) {
  uint32_t lo_nb = nb > 16 ? 16 : nb;
  uint32_t hi_nb = nb > 16 ? nb - 16 : 0;
  if (!rans_enc_bits16(x, active, lo_nb, bits & 0xFFFFu, wp, wlimit)) return false;
  return rans_enc_bits16(x, active, hi_nb, bits >> 16, wp, wlimit);
}

// Encodes a parsed chunk against its batch's freq/cum (literal-context rule
// lit_shift). Words are written backward from the end of the slot with the
// header just before them, so the payload starts at *out_start. Returns
// false if the result would not beat raw storage. Requires in_len >
// kRansHeaderBytes + 1 and a chunk that passed rans_eligible (kernels.cu).
__device__ inline bool rans_encode_warp(const SeqRec* seqs, uint32_t n_seq, const uint8_t* rep_code,
                                        const uint8_t* lits, uint32_t n_lit, uint32_t lit_shift,
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

  // Literals, last step first, each in the context of the one before it
  // in its lane's run.
  uint32_t run_len = lit_run_len(n_lit);
  for (int t = (int)run_len - 1; t >= 0; --t) {
    uint32_t idx = lane * run_len + (uint32_t)t;
    bool act = idx < n_lit;
    uint32_t e = 0;
    if (act) e = lit_entry(lit_ctx(t ? lits[idx - 1] : 0, lit_shift), lits[idx]);
    if (!rans_enc_put(x, act, freq[e], cum[e], wp, wlimit)) return false;
  }
  for (int g = (int)((n_seq + 31) / 32) - 1; g >= 0; --g) {
    uint32_t idx = g * 32 + lane;
    bool act = idx < n_seq;
    SeqRec r{0, 0, 0};
    if (act) r = seqs[idx];
    uint32_t ml = r.ml(), off = r.off();
    uint32_t llc, llnb, llb, mlc, mlnb, mlb, oc = 0, onb = 0, ob = 0;
    len_code(r.lit_len(), llc, llnb, llb);
    len_code(ml ? ml - (kMinMatch - 1) : 0, mlc, mlnb, mlb);
    bool has_off = act && ml != 0;
    if (has_off) {
      uint8_t rc = rep_code[idx];
      if (rc) {
        oc = kOffRepBase + (rc - 1); // onb, ob stay 0: repeat codes carry no extra bits
      } else {
        off_code(off, oc, onb, ob);
      }
    }

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

// Decodes one literal in context ctx. Sets `bad` if the table entry it
// lands on has zero frequency, which a valid stream never produces (an
// unused context's sym row is all zeros, and so is its symbol 0's freq).
__device__ __forceinline__ uint32_t rans_dec_lit(uint32_t& x, const RansTableView& t, uint32_t ctx, bool& bad) {
  uint32_t slot = x & (kProbScale - 1);
  uint32_t s = t.sym[(ctx << kProbBits) + slot];
  int e = lit_entry(ctx, s);
  uint32_t f = t.freq[e];
  bad |= f == 0;
  x = f * (x >> kProbBits) + slot - t.cum[e];
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

// Decodes at most 16 raw bits, then renormalises.
__device__ __forceinline__ bool rans_dec_bits16(uint32_t& x, bool active, uint32_t nb, uint32_t& out,
                                                const uint8_t*& rp, const uint8_t* rend) {
  bool on = active && nb != 0;
  out = on ? (x & ((1u << nb) - 1)) : 0;
  if (on) x >>= nb;
  return rans_dec_renorm(x, on, rp, rend);
}

// Decodes nb raw bits: the high part first, then the low 16, mirroring
// rans_enc_bits. As there, every lane makes both calls.
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

// Decodes the sequences and literals into scratch using the group's tables
// `t`. Rejects malformed input (zero-frequency symbols, truncated or
// over-long streams) without writing beyond the scratch bounds.
__device__ inline bool rans_decode_warp(const uint8_t* payload, uint32_t len, const RansTableView& t,
                                        uint32_t lit_shift, SeqRec* seqs, uint32_t max_seq, uint8_t* lits,
                                        uint32_t max_lit, uint32_t* n_seq_out, uint32_t* n_lit_out) {
  int lane = threadIdx.x & 31;
  if (len < (uint32_t)kRansHeaderBytes) return false;
  uint32_t n_seq = load_u32(payload), n_lit = load_u32(payload + 4);
  if (n_seq > max_seq || n_lit > max_lit) return false;

  uint32_t x = load_u32(payload + 8 + 4 * lane);
  const uint8_t* rp = payload + kRansHeaderBytes;
  const uint8_t* rend = payload + len;
  bool bad = false;
  RepOffsets rep; // identical in every lane, advanced by the replay below

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
      bad |= t.freq[kOffBase + oc] == 0;
    }
    if (!rans_dec_renorm(x, has_off, rp, rend)) return false;
    if (!rans_dec_bits(x, has_off, off_nb(oc), ob, rp, rend)) return false;

    // Resolve the group's offsets: every lane replays the recent-offset
    // state over all 32 sequences in order, keeping its own lane's result.
    uint32_t off = 0;
    for (int j = 0; j < 32; ++j) {
      bool j_has = __shfl_sync(kFullMask, has_off, j);
      uint32_t j_oc = __shfl_sync(kFullMask, oc, j);
      uint32_t j_ob = __shfl_sync(kFullMask, ob, j);
      uint32_t jo = 0;
      if (j_has) {
        if (j_oc >= kOffRepBase) {
          jo = rep.get(j_oc - kOffRepBase);
          rep.use(j_oc - kOffRepBase, jo);
        } else {
          jo = off_value(j_oc, j_ob);
          rep.use(kRepNew, jo);
        }
      }
      if (j == lane) off = jo;
    }

    if (act) {
      if (ml == 0 && idx != n_seq - 1) bad = true;
      // Reject lengths/offsets beyond the chunk before packing them into
      // SeqRec's 21-bit fields.
      if (ml > max_lit || (has_off && off > max_lit)) bad = true;
      seqs[idx] = SeqRec{len_value(llc, llb), has_off ? off : 0, ml};
    }
    if (__any_sync(kFullMask, bad)) return false;
  }

  // Literals: each lane decodes its own run in order, the previous literal
  // giving the context, and stores them 4 at a time (runs are 4-aligned).
  uint32_t run_len = lit_run_len(n_lit);
  uint32_t run0 = lane * run_len;
  uint32_t prev = 0, pack = 0;
  for (uint32_t k = 0; k < run_len; ++k) {
    uint32_t idx = run0 + k;
    bool act = idx < n_lit;
    if (act) {
      uint32_t s = rans_dec_lit(x, t, lit_ctx(prev, lit_shift), bad);
      prev = s;
      pack |= s << (8 * (k & 3));
      if ((k & 3) == 3) {
        *reinterpret_cast<uint32_t*>(lits + idx - 3) = pack;
        pack = 0;
      }
    }
    if (!rans_dec_renorm(x, act, rp, rend)) return false;
  }
  if (run0 < n_lit) {
    uint32_t run_end = min(n_lit, run0 + run_len);
    uint32_t rem = (run_end - run0) & 3;
    for (uint32_t k = 0; k < rem; ++k) lits[run_end - rem + k] = (uint8_t)(pack >> (8 * k));
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
    if (lp + r.lit_len() > n_lit || op + r.lit_len() > orig) return false;
    for (uint32_t k = lane; k < r.lit_len(); k += 32) out[op + k] = lits[lp + k];
    op += r.lit_len();
    lp += r.lit_len();
    uint32_t ml = r.ml(), off = r.off();
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

} // namespace gpusqz
