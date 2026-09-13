#include "kernels.h"
#include "lz_warp.cuh"
#include "rans.cuh"
#include "format.h"

#include <cub/device/device_scan.cuh>

namespace gzp {

constexpr int kBlockThreads = kWarpsPerBlock * 32;

// GZP_MIN_BLOCKS_PER_SM (set via CMake's GZP_MIN_BLOCKS_PER_SM cache var)
// forces ptxas to keep register usage low enough for that many blocks per
// SM, for A/B occupancy testing. Left unset, __launch_bounds__ takes only
// the thread-count argument and ptxas picks registers freely.
#ifdef GZP_MIN_BLOCKS_PER_SM
#define GZP_LAUNCH_BOUNDS __launch_bounds__(kBlockThreads, GZP_MIN_BLOCKS_PER_SM)
#else
#define GZP_LAUNCH_BOUNDS __launch_bounds__(kBlockThreads)
#endif

__device__ __forceinline__ void chunk_scratch(uint8_t* scratch, uint32_t c, uint32_t chunk_size, SeqRec*& seqs,
                                              uint8_t*& rep_code, uint8_t*& lits) {
  uint8_t* base = scratch + (size_t)c * scratch_bytes(chunk_size);
  seqs = reinterpret_cast<SeqRec*>(base);
  rep_code = base + scratch_seqs_bytes(chunk_size);
  lits = base + scratch_lits_offset(chunk_size);
}

// ---------------------------------------------------------------------------
// Compress: 3 kernels sharing one rANS table per batch. Each chunk still
// individually falls back to a plain LZ token stream, then to Raw storage,
// whichever is smallest — see rans_encode_kernel.
// ---------------------------------------------------------------------------

// One warp per chunk. in_len <= 1 chunks are finalized here directly (Raw)
// since they never reach the encode kernel. Others are parsed into scratch
// and contribute to the batch histogram; n_seq[c]/n_lit[c] record how much
// of scratch is valid so the encode kernel doesn't need to re-parse.
// htab is chunk_count * hash_table_bytes(chunk_size) of global memory (see
// kernels.h): chunk c's region starts at htab + c * hash_table_words.
__global__ void GZP_LAUNCH_BOUNDS
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
  lz_parse_warp(chunk_in, in_len, htab + (size_t)c * hash_words, (int)hash_bits, em);
  compute_repeat_codes(seqs, em.n_seq, rep_code);
  accumulate_hist(seqs, em.n_seq, rep_code, lits, em.n_lit, batch_cnt);
  if (lane == 0) {
    n_seq_arr[c] = em.n_seq;
    n_lit_arr[c] = em.n_lit;
  }
}

// One block, launched once per batch: quantises+normalises the batch
// histogram into freq/cum for the encode kernel and q[] for the host to
// store in this batch's TableGroup.
__global__ void build_table_kernel(const uint32_t* cnt, uint8_t* q_out, uint16_t* freq, uint16_t* cum) {
  build_batch_table_warp(cnt, q_out, freq, cum);
}

// One warp per chunk (skips in_len <= 1 chunks, already finalized). Tries
// rANS against the batch's shared table, then plain tokens, then raw,
// keeping whichever is smallest.
__global__ void GZP_LAUNCH_BOUNDS
rans_encode_kernel(const uint8_t* in, uint32_t chunk_size, uint32_t chunk_count, const uint32_t* in_lens,
                   uint8_t* out, uint32_t out_slot_stride, uint32_t* out_start, uint32_t* out_sizes,
                   uint8_t* scratch, const uint32_t* n_seq_arr, const uint32_t* n_lit_arr, const uint16_t* freq,
                   const uint16_t* cum) {
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

  uint32_t tok_total = 0;
  for (uint32_t i = lane; i < n_seq; i += 32) {
    SeqRec r = seqs[i];
    tok_total += seq_size(r.lit_len(), r.ml());
  }
  for (int o = 16; o > 0; o >>= 1) tok_total += __shfl_xor_sync(kFullMask, tok_total, o);

  bool ok = false;
  uint32_t start = 0, size = 0;
  if (in_len > (uint32_t)kRansHeaderBytes + 1) {
    ok = rans_encode_warp(seqs, n_seq, rep_code, lits, n_lit, freq, cum, slot, out_slot_stride, in_len, &start,
                          &size);
    if (ok && 1 + tok_total < size) ok = false;
  }
  if (!ok) {
    uint32_t len = 0;
    ok = tokens_from_seqs(chunk_in, seqs, n_seq, slot + 1, in_len - 1, &len);
    if (ok) {
      start = 0;
      size = 1 + len;
      if (lane == 0) slot[0] = (uint8_t)ChunkFlag::Lz;
    }
  }
  if (!ok) {
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
// out_lens[c] bytes. LzRans chunks read their table from
// group_tables[group_id[c]] (already expanded). Any malformed chunk sets *err.
__global__ void GZP_LAUNCH_BOUNDS
decompress_kernel(const uint8_t* in, const uint32_t* in_offsets, uint32_t chunk_count, const uint32_t* in_lens,
                  uint8_t* out, uint32_t chunk_size, const uint32_t* out_lens, uint8_t* scratch,
                  const RansGroupTable* group_tables, const uint32_t* group_id, uint32_t* err) {
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
      const RansGroupTable& gt = group_tables[group_id[c]];
      ok = rans_decode_warp(slot + 1, len - 1, gt, seqs, max_sequences(chunk_size), lits, chunk_size, &n_seq,
                            &n_lit);
      if (ok) {
        __syncwarp();
        ok = lz_reconstruct_warp(seqs, n_seq, lits, n_lit, chunk_out, orig);
      }
    }
  }
  if (!ok && lane == 0) *err = 1;
}

__global__ void expand_group_tables_kernel(const uint8_t* q_all, uint32_t group_count, RansGroupTable* out) {
  uint32_t g = blockIdx.x;
  if (g >= group_count) return;
  expand_group_table_warp(q_all + (size_t)g * kQuantBytes, out[g]);
}

// ---------------------------------------------------------------------------
// Host-callable launchers
// ---------------------------------------------------------------------------

void launch_compress(const uint8_t* d_in, uint32_t chunk_size, uint32_t chunk_count,
                      const uint32_t* d_in_lens, uint8_t* d_out, uint32_t out_slot_stride,
                      uint32_t* d_out_start, uint32_t* d_out_sizes, uint8_t* d_scratch, uint32_t* d_htab,
                      const RansBatchBufs& d_rans, cudaStream_t stream) {
  uint32_t blocks = (chunk_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
  uint32_t hash_bits = (uint32_t)hash_table_bits(chunk_size);
  // Parse+histogram everything in the batch, build one shared table, then
  // encode. Sequential on `stream`, so each stage sees the previous one's
  // complete output.
  cudaMemsetAsync(d_rans.cnt, 0, kQuantBytes * sizeof(uint32_t), stream);
  parse_hist_kernel<<<blocks, kBlockThreads, 0, stream>>>(d_in, chunk_size, chunk_count, d_in_lens, d_out,
                                                          out_slot_stride, d_out_start, d_out_sizes, d_scratch,
                                                          d_htab, hash_bits, d_rans.n_seq, d_rans.n_lit, d_rans.cnt);
  build_table_kernel<<<1, 32, 0, stream>>>(d_rans.cnt, d_rans.q, d_rans.freq, d_rans.cum);
  rans_encode_kernel<<<blocks, kBlockThreads, 0, stream>>>(d_in, chunk_size, chunk_count, d_in_lens, d_out,
                                                           out_slot_stride, d_out_start, d_out_sizes, d_scratch,
                                                           d_rans.n_seq, d_rans.n_lit, d_rans.freq, d_rans.cum);
}

void launch_decompress(const uint8_t* d_in, const uint32_t* d_in_offsets, uint32_t chunk_count,
                        const uint32_t* d_in_lens, uint8_t* d_out, uint32_t chunk_size,
                        const uint32_t* d_out_lens, uint8_t* d_scratch, const void* d_group_tables,
                        const uint32_t* d_group_id, uint32_t* d_err, cudaStream_t stream) {
  uint32_t blocks = (chunk_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
  decompress_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      d_in, d_in_offsets, chunk_count, d_in_lens, d_out, chunk_size, d_out_lens, d_scratch,
      reinterpret_cast<const RansGroupTable*>(d_group_tables), d_group_id, d_err);
}

size_t rans_group_table_bytes() { return sizeof(RansGroupTable); }

void launch_expand_group_tables(const uint8_t* d_q, uint32_t group_count, void* d_tables, cudaStream_t stream) {
  expand_group_tables_kernel<<<group_count, 32, 0, stream>>>(d_q, group_count,
                                                             reinterpret_cast<RansGroupTable*>(d_tables));
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

} // namespace gzp
