#include "kernels.h"
#include "lz_warp.cuh"
#include "rans.cuh"
#include "format.h"

#include <cub/device/device_scan.cuh>

namespace gpusqz {

constexpr int kBlockThreads = kWarpsPerBlock * 32;

// GPUSQZ_MIN_BLOCKS_PER_SM (a CMake option) makes ptxas keep register use
// low enough for that many blocks per SM, for occupancy experiments.
#ifdef GPUSQZ_MIN_BLOCKS_PER_SM
#define GPUSQZ_LAUNCH_BOUNDS __launch_bounds__(kBlockThreads, GPUSQZ_MIN_BLOCKS_PER_SM)
#else
#define GPUSQZ_LAUNCH_BOUNDS __launch_bounds__(kBlockThreads)
#endif

__device__ __forceinline__ void chunk_scratch(uint8_t* scratch, uint32_t c, uint32_t chunk_size, SeqRec*& seqs,
                                              uint8_t*& rep_code, uint8_t*& lits) {
  uint8_t* base = scratch + (size_t)c * scratch_bytes(chunk_size);
  seqs = reinterpret_cast<SeqRec*>(base);
  rep_code = base + scratch_seqs_bytes(chunk_size);
  lits = base + scratch_lits_offset(chunk_size);
}

// Whether a parsed chunk could end up rANS-coded, and the size of its plain
// token stream (ChunkFlag::Lz payload); warp-collective. rANS is ruled out
// by a token stream shorter than a rANS header (rANS is kept only if it is
// no bigger than the tokens) or by a literal run too long for its length
// alphabet (kMaxLenValue). Both kernels below use this one test: such
// chunks stay out of the batch histogram and never try rANS.
__device__ __forceinline__ bool rans_eligible(const SeqRec* seqs, uint32_t n_seq, uint32_t& token_bytes) {
  int lane = threadIdx.x & 31;
  uint32_t tot = 0;
  bool too_long = false;
  for (uint32_t i = lane; i < n_seq; i += 32) {
    SeqRec r = seqs[i];
    tot += seq_size(r.lit_len(), r.ml());
    too_long |= r.lit_len() > kMaxLenValue;
  }
  for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(kFullMask, tot, o);
  token_bytes = tot;
  return !__any_sync(kFullMask, too_long) && tot >= (uint32_t)kRansHeaderBytes;
}

// ---------------------------------------------------------------------------
// Compress: parse + histogram, build the batch's tables, encode.
// ---------------------------------------------------------------------------

// One warp per chunk. Chunks of at most 1 byte are stored raw right here.
// Others are parsed into scratch (n_seq/n_lit record how much) and, if they
// could be rANS-coded, counted into the batch histogram.
__global__ void GPUSQZ_LAUNCH_BOUNDS
parse_hist_kernel(const uint8_t* in, uint32_t chunk_size, uint32_t chunk_count, const uint32_t* in_lens,
                  uint8_t* out, uint32_t out_slot_stride, uint32_t* out_start, uint32_t* out_sizes,
                  uint8_t* scratch, uint32_t* htab, uint32_t hash_bits, uint32_t* n_seq_arr, uint32_t* n_lit_arr,
                  uint32_t* batch_cnt) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t c = blockIdx.x * kWarpsPerBlock + warp;
  if (c >= chunk_count) return;

  const uint8_t* chunk_in = in + (size_t)c * chunk_size;
  uint32_t in_len = in_lens[c];
  uint8_t* slot = out + (size_t)c * out_slot_stride;

  if (in_len <= 1) {
    for (uint32_t k = lane; k < in_len; k += 32) slot[1 + k] = chunk_in[k];
    if (lane == 0) {
      slot[0] = (uint8_t)ChunkFlag::Raw;
      out_start[c] = 0;
      out_sizes[c] = 1 + in_len;
      n_seq_arr[c] = 0;
      n_lit_arr[c] = 0;
    }
    return;
  }

  SeqRec* seqs;
  uint8_t* rep_code;
  uint8_t* lits;
  chunk_scratch(scratch, c, chunk_size, seqs, rep_code, lits);
  uint32_t hash_words = (1u << hash_bits) * (uint32_t)kBucketWays;
  SeqEmitter em{chunk_in, seqs, lits};
  lz_parse_warp(chunk_in, in_len, htab + (size_t)c * hash_words, (int)hash_bits, rep_probes(chunk_size), em);
  if (lane == 0) {
    n_seq_arr[c] = em.n_seq;
    n_lit_arr[c] = em.n_lit;
  }
  __syncwarp(); // the sequences and literals below were written by other lanes
  uint32_t token_bytes;
  if (!rans_eligible(seqs, em.n_seq, token_bytes)) return;
  compute_repeat_codes(seqs, em.n_seq, rep_code);
  accumulate_hist(seqs, em.n_seq, rep_code, lits, em.n_lit, batch_cnt);
}

// Bits to code a 256-symbol histogram row with the table the encoder would
// actually build from it (8-bit quantised, normalised to kProbScale, every
// present symbol at least 1/kProbScale), rather than the exact-probability
// entropy, which ignores what quantisation costs thinly populated contexts.
__device__ float row_coded_bits(const uint32_t* row) {
  uint8_t q[kLitSyms];
  uint16_t f[kLitSyms], c[kLitSyms];
  quantize_counts(row, kLitSyms, q);
  if (!normalize_table(q, kLitSyms, f, c)) return 0.f;
  float b = 0.f;
  for (int s = 0; s < kLitSyms; ++s) {
    if (row[s]) b += (float)row[s] * ((float)kProbBits - __log2f((float)f[s]));
  }
  return b;
}

// One block per batch. Picks the batch's literal-context rule from its full
// order-1 histogram, folds the histogram to that rule, and builds the
// tables (freq/cum for the encoder, q for the file). The rule minimises
// estimated literal bits (row_coded_bits) plus 8 * 256 table bytes per
// context, so literal-poor or incompressible batches keep a single table.
// forced_shift >= 0 skips the choice (GPUSQZ_FORCE_LIT_SHIFT, tests). The
// costs are summed in a fixed order, so the output is deterministic.
constexpr int kTableThreads = 256;
__global__ void __launch_bounds__(kTableThreads)
build_table_kernel(uint32_t* cnt, int forced_shift, uint32_t* shift_out, uint8_t* q_out, uint16_t* freq,
                   uint16_t* cum) {
  __shared__ uint32_t nib[16][kLitSyms]; // counts folded to 16 contexts (shift 4)
  __shared__ uint32_t col[kLitSyms];     // folded to 1 context (shift 8)
  __shared__ float part[3][kTableThreads];
  __shared__ int chosen;
  int t = threadIdx.x; // one literal symbol per thread while folding
  uint32_t colsum = 0;
  for (int c = 0; c < 16; ++c) {
    uint32_t v = 0;
    for (int r = 0; r < 16; ++r) v += cnt[lit_entry(c * 16 + r, t)];
    nib[c][t] = v;
    colsum += v;
  }
  col[t] = colsum;
  __syncthreads();

  if (forced_shift < 0) {
    // part[k][t]: thread t's share of shift 8 / 4 / 0's cost, summed in a
    // fixed order below.
    part[0][t] = t == 0 ? row_coded_bits(col) : 0.f;
    part[1][t] = t < 16 ? row_coded_bits(nib[t]) : 0.f;
    part[2][t] = row_coded_bits(cnt + lit_entry(t, 0));
    __syncthreads();
    if (t == 0) {
      const uint32_t shifts[3] = {kLitShiftOrder0, kLitShiftNibble, kLitShiftByte};
      int best = 0;
      float best_cost = 0.f;
      for (int k = 0; k < 3; ++k) {
        float c = 8.f * kLitSyms * lit_ctx_count(shifts[k]);
        for (int i = 0; i < kTableThreads; ++i) c += part[k][i];
        if (k == 0 || c < best_cost) {
          best = k;
          best_cost = c;
        }
      }
      chosen = (int)shifts[best];
    }
  } else if (t == 0) {
    chosen = forced_shift;
  }
  __syncthreads();

  uint32_t shift = (uint32_t)chosen;
  int n_ctx = lit_ctx_count(shift);
  if (t == 0) *shift_out = shift;
  // Every read of the full histogram is done; fold into its first rows.
  if (shift == kLitShiftNibble) {
    for (int c = 0; c < 16; ++c) cnt[lit_entry(c, t)] = nib[c][t];
  } else if (shift == kLitShiftOrder0) {
    cnt[lit_entry(0, t)] = col[t];
  }
  __syncthreads();
  build_batch_table(cnt, n_ctx, q_out, freq, cum);
}

// One warp per chunk (chunks of at most 1 byte are already stored). Keeps
// the smallest of rANS against the batch's tables, plain tokens, and raw.
__global__ void GPUSQZ_LAUNCH_BOUNDS
rans_encode_kernel(const uint8_t* in, uint32_t chunk_size, uint32_t chunk_count, const uint32_t* in_lens,
                   uint8_t* out, uint32_t out_slot_stride, uint32_t* out_start, uint32_t* out_sizes,
                   uint8_t* scratch, const uint32_t* n_seq_arr, const uint32_t* n_lit_arr,
                   const uint32_t* lit_shift_p, const uint16_t* freq, const uint16_t* cum) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t c = blockIdx.x * kWarpsPerBlock + warp;
  if (c >= chunk_count) return;

  uint32_t in_len = in_lens[c];
  if (in_len <= 1) return; // already finalized by parse_hist_kernel

  const uint8_t* chunk_in = in + (size_t)c * chunk_size;
  uint8_t* slot = out + (size_t)c * out_slot_stride;
  SeqRec* seqs;
  uint8_t* rep_code;
  uint8_t* lits;
  chunk_scratch(scratch, c, chunk_size, seqs, rep_code, lits);
  uint32_t n_seq = n_seq_arr[c], n_lit = n_lit_arr[c];
  uint32_t lit_shift = *lit_shift_p; // chosen by build_table_kernel for this batch

  uint32_t tok_total;
  bool eligible = rans_eligible(seqs, n_seq, tok_total);

  bool ok = false;
  uint32_t start = 0, size = 0;
  // An ineligible chunk contributed nothing to the batch tables, so it
  // must not use them either.
  if (eligible && in_len > (uint32_t)kRansHeaderBytes + 1) {
    ok = rans_encode_warp(seqs, n_seq, rep_code, lits, n_lit, lit_shift, freq, cum, slot, out_slot_stride, in_len,
                          &start, &size);
    if (ok && 1 + tok_total < size) ok = false;
  }
  // Each attempt below overwrites bytes other lanes wrote in the one
  // before, so the lanes sync first.
  if (!ok) {
    __syncwarp();
    uint32_t len = 0;
    ok = tokens_from_seqs(chunk_in, seqs, n_seq, slot + 1, in_len - 1, &len);
    if (ok) {
      start = 0;
      size = 1 + len;
      if (lane == 0) slot[0] = (uint8_t)ChunkFlag::Lz;
    }
  }
  if (!ok) {
    __syncwarp();
    for (uint32_t k = lane; k < in_len; k += 32) slot[1 + k] = chunk_in[k];
    start = 0;
    size = 1 + in_len;
    if (lane == 0) slot[0] = (uint8_t)ChunkFlag::Raw;
  }
  if (lane == 0) {
    out_start[c] = start;
    out_sizes[c] = size;
  }
}

// ---------------------------------------------------------------------------
// Decompress (both Lz and LzRans chunks may appear in the same file).
// ---------------------------------------------------------------------------

// One warp per chunk. Chunk c's compressed data lives at in + in_offsets[c]
// (in_lens[c] bytes, flag first); output goes to out + c*chunk_size,
// out_lens[c] bytes. LzRans chunks read their tables from group g =
// group_id[c]'s region at group_tables + group_off[g], expanded for its
// literal-context rule group_shift[g]. Any malformed chunk sets *err.
__global__ void GPUSQZ_LAUNCH_BOUNDS
decompress_kernel(const uint8_t* in, const uint32_t* in_offsets, uint32_t chunk_count, const uint32_t* in_lens,
                  uint8_t* out, uint32_t chunk_size, const uint32_t* out_lens, uint8_t* scratch,
                  uint8_t* group_tables, const uint64_t* group_off, const uint32_t* group_shift,
                  const uint32_t* group_id, uint32_t* err) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t c = blockIdx.x * kWarpsPerBlock + warp;
  if (c >= chunk_count) return;

  const uint8_t* slot = in + in_offsets[c];
  uint32_t len = in_lens[c];
  uint8_t* chunk_out = out + (size_t)c * chunk_size;
  uint32_t orig = out_lens[c];

  bool ok = false;
  if (len >= 1) {
    ChunkFlag flag = (ChunkFlag)slot[0];
    if (flag == ChunkFlag::Raw) {
      ok = len == 1 + orig;
      if (ok) {
        for (uint32_t k = lane; k < orig; k += 32) chunk_out[k] = slot[1 + k];
      }
    } else if (flag == ChunkFlag::Lz) {
      ok = lz_decode_warp(slot + 1, len - 1, chunk_out, orig);
    } else if (flag == ChunkFlag::LzRans) {
      SeqRec* seqs;
      uint8_t* rep_code;
      uint8_t* lits;
      chunk_scratch(scratch, c, chunk_size, seqs, rep_code, lits); // rep_code unused: decode keeps its state in registers
      uint32_t n_seq = 0, n_lit = 0;
      uint32_t g = group_id[c], lit_shift = group_shift[g];
      RansTableView gt = table_view(group_tables + group_off[g], lit_ctx_count(lit_shift));
      ok = rans_decode_warp(slot + 1, len - 1, gt, lit_shift, seqs, max_sequences(chunk_size), lits, chunk_size,
                            &n_seq, &n_lit);
      if (ok) {
        __syncwarp();
        ok = lz_reconstruct_warp(seqs, n_seq, lits, n_lit, chunk_out, orig);
      }
    }
  }
  if (!ok && lane == 0) *err = 1;
}

// One block per group: group g's quantised counts start at q_all + q_off[g]
// and expand into its region at out + table_off[g].
__global__ void expand_group_tables_kernel(const uint8_t* q_all, const uint64_t* q_off, const uint64_t* table_off,
                                           const uint32_t* shift, uint32_t group_count, uint8_t* out) {
  uint32_t g = blockIdx.x;
  if (g >= group_count) return;
  expand_group_table(q_all + q_off[g], lit_ctx_count(shift[g]), out + table_off[g]);
}

// ---------------------------------------------------------------------------
// Host-callable launchers
// ---------------------------------------------------------------------------

void launch_compress(const uint8_t* d_in, uint32_t chunk_size, uint32_t chunk_count,
                      const uint32_t* d_in_lens, uint8_t* d_out, uint32_t out_slot_stride,
                      uint32_t* d_out_start, uint32_t* d_out_sizes, uint8_t* d_scratch, uint32_t* d_htab,
                      int forced_lit_shift, const RansBatchBufs& d_rans, cudaStream_t stream) {
  uint32_t blocks = (chunk_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
  uint32_t hash_bits = (uint32_t)hash_table_bits(chunk_size);
  // Parse+histogram everything in the batch, build one shared set of
  // tables, then encode. Sequential on `stream`, so each stage sees the
  // previous one's complete output.
  cudaMemsetAsync(d_rans.cnt, 0, (size_t)kMaxQuantBytes * sizeof(uint32_t), stream);
  parse_hist_kernel<<<blocks, kBlockThreads, 0, stream>>>(d_in, chunk_size, chunk_count, d_in_lens, d_out,
                                                          out_slot_stride, d_out_start, d_out_sizes, d_scratch,
                                                          d_htab, hash_bits, d_rans.n_seq, d_rans.n_lit, d_rans.cnt);
  build_table_kernel<<<1, kTableThreads, 0, stream>>>(d_rans.cnt, forced_lit_shift, d_rans.lit_shift, d_rans.q,
                                                      d_rans.freq, d_rans.cum);
  rans_encode_kernel<<<blocks, kBlockThreads, 0, stream>>>(d_in, chunk_size, chunk_count, d_in_lens, d_out,
                                                           out_slot_stride, d_out_start, d_out_sizes, d_scratch,
                                                           d_rans.n_seq, d_rans.n_lit, d_rans.lit_shift, d_rans.freq,
                                                           d_rans.cum);
}

void launch_decompress(const uint8_t* d_in, const uint32_t* d_in_offsets, uint32_t chunk_count,
                        const uint32_t* d_in_lens, uint8_t* d_out, uint32_t chunk_size,
                        const uint32_t* d_out_lens, uint8_t* d_scratch, uint8_t* d_group_tables,
                        const uint64_t* d_group_off, const uint32_t* d_group_shift, const uint32_t* d_group_id,
                        uint32_t* d_err, cudaStream_t stream) {
  uint32_t blocks = (chunk_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
  decompress_kernel<<<blocks, kBlockThreads, 0, stream>>>(d_in, d_in_offsets, chunk_count, d_in_lens, d_out,
                                                          chunk_size, d_out_lens, d_scratch, d_group_tables,
                                                          d_group_off, d_group_shift, d_group_id, d_err);
}

size_t rans_group_table_bytes(uint32_t lit_shift) { return group_table_bytes(lit_ctx_count(lit_shift)); }

void launch_expand_group_tables(const uint8_t* d_q, const uint64_t* d_q_off, const uint64_t* d_table_off,
                                const uint32_t* d_shift, uint32_t group_count, uint8_t* d_tables,
                                cudaStream_t stream) {
  expand_group_tables_kernel<<<group_count, kTableThreads, 0, stream>>>(d_q, d_q_off, d_table_off, d_shift,
                                                                        group_count, d_tables);
}

// One warp per chunk; byte copies because slot payloads have arbitrary
// alignment. This is memory-bound and tiny next to the compress kernel.
__global__ void compact_kernel(const uint8_t* slots, uint32_t slot_stride, const uint32_t* src_off,
                               const uint32_t* sizes, const uint32_t* offsets, uint32_t n, uint8_t* packed) {
  uint32_t c = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  uint32_t lane = threadIdx.x & 31;
  if (c >= n) return;
  const uint8_t* src = slots + (size_t)c * slot_stride + src_off[c];
  uint8_t* dst = packed + offsets[c];
  uint32_t sz = sizes[c];
  for (uint32_t k = lane; k < sz; k += 32) dst[k] = src[k];
}

size_t compaction_temp_bytes(uint32_t max_chunks) {
  size_t bytes = 0;
  cub::DeviceScan::InclusiveSum(nullptr, bytes, (const uint32_t*)nullptr, (uint32_t*)nullptr,
                                (int)max_chunks);
  return bytes;
}

cudaError_t launch_compact(const uint8_t* d_slots, uint32_t slot_stride, const uint32_t* d_src_off,
                           const uint32_t* d_sizes, uint32_t n, uint32_t* d_offsets, uint8_t* d_packed,
                           void* d_temp, size_t temp_bytes, cudaStream_t stream) {
  cudaError_t err = cudaMemsetAsync(d_offsets, 0, sizeof(uint32_t), stream);
  if (err != cudaSuccess) return err;
  err = cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_sizes, d_offsets + 1, (int)n, stream);
  if (err != cudaSuccess) return err;
  constexpr int kThreads = 128;
  uint32_t blocks = (n * 32 + kThreads - 1) / kThreads;
  compact_kernel<<<blocks, kThreads, 0, stream>>>(d_slots, slot_stride, d_src_off, d_sizes, d_offsets, n, d_packed);
  return cudaGetLastError();
}

} // namespace gpusqz
