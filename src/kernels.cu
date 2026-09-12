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
                                              uint8_t*& lits) {
  uint8_t* base = scratch + (size_t)c * scratch_bytes(chunk_size);
  seqs = reinterpret_cast<SeqRec*>(base);
  lits = base + ((sizeof(SeqRec) * max_sequences(chunk_size) + 15) & ~(size_t)15);
}

// One warp per chunk. Chunk c's input lives at in + c*chunk_size (in_lens[c]
// valid bytes); its output goes into the fixed slot out + c*out_slot_stride
// as [flag byte][payload] starting at out_start[c], total size out_sizes[c].
__global__ void GZP_LAUNCH_BOUNDS
compress_kernel(const uint8_t* in, uint32_t chunk_size, uint32_t chunk_count, const uint32_t* in_lens,
                uint8_t* out, uint32_t out_slot_stride, uint32_t* out_start, uint32_t* out_sizes,
                uint8_t* scratch, Mode mode) {
  __shared__ uint32_t htab[kWarpsPerBlock][kHashSize];
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t c = blockIdx.x * kWarpsPerBlock + warp;
  if (c >= chunk_count) return;

  const uint8_t* chunk_in = in + (size_t)c * chunk_size;
  uint32_t in_len = in_lens[c];
  uint8_t* slot = out + (size_t)c * out_slot_stride;

  // Any encoding must not beat raw storage by less than nothing: payloads
  // are capped at in_len - 1 so flag + payload <= in_len.
  bool ok = false;
  uint32_t start = 0, size = 0;
  if (in_len > 1) {
    if (mode == Mode::LzRans) {
      SeqRec* seqs;
      uint8_t* lits;
      chunk_scratch(scratch, c, chunk_size, seqs, lits);
      SeqEmitter em{chunk_in, seqs, lits};
      lz_parse_warp(chunk_in, in_len, htab[warp], em);
      __syncwarp();

      // Plain tokens have no per-chunk table overhead, so on highly
      // repetitive chunks they beat rANS; size both and keep the smaller.
      uint32_t tok_total = 0;
      for (uint32_t i = lane; i < em.n_seq; i += 32) {
        SeqRec r = seqs[i];
        tok_total += seq_size(r.lit_len, r.ml);
      }
      for (int o = 16; o > 0; o >>= 1) tok_total += __shfl_xor_sync(kFullMask, tok_total, o);

      if (in_len > (uint32_t)kRansHeaderBytes + 1) {
        // The hash table is free now; reuse it for the coder's tables.
        RansEncTables& t = *reinterpret_cast<RansEncTables*>(htab[warp]);
        ok = rans_encode_warp(seqs, em.n_seq, lits, em.n_lit, t, slot, out_slot_stride, in_len, &start, &size);
        if (ok && 1 + tok_total < size) ok = false;
      }
      if (!ok) {
        uint32_t len = 0;
        ok = tokens_from_seqs(chunk_in, seqs, em.n_seq, slot + 1, in_len - 1, &len);
        if (ok) {
          start = 0;
          size = 1 + len;
          if (lane == 0) slot[0] = (uint8_t)ChunkFlag::Lz;
        }
      }
    } else {
      TokenEmitter em{chunk_in, slot + 1, in_len - 1};
      ok = lz_parse_warp(chunk_in, in_len, htab[warp], em);
      if (ok) {
        start = 0;
        size = 1 + em.op;
        if (lane == 0) slot[0] = (uint8_t)ChunkFlag::Lz;
      }
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

// One warp per chunk. Chunk c's compressed data lives at in + in_offsets[c]
// (in_lens[c] bytes, flag first); output goes to out + c*chunk_size,
// out_lens[c] bytes. Any malformed chunk sets *err.
__global__ void GZP_LAUNCH_BOUNDS
decompress_kernel(const uint8_t* in, const uint32_t* in_offsets, uint32_t chunk_count, const uint32_t* in_lens,
                  uint8_t* out, uint32_t chunk_size, const uint32_t* out_lens, uint8_t* scratch, uint32_t* err) {
  __shared__ RansDecTables tables[kWarpsPerBlock];
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
      uint8_t* lits;
      chunk_scratch(scratch, c, chunk_size, seqs, lits);
      uint32_t n_seq = 0, n_lit = 0;
      ok = rans_decode_warp(slot + 1, len - 1, tables[warp], seqs, max_sequences(chunk_size), lits, chunk_size,
                            &n_seq, &n_lit);
      if (ok) {
        __syncwarp();
        ok = lz_reconstruct_warp(seqs, n_seq, lits, n_lit, chunk_out, orig);
      }
    }
  }
  if (!ok && lane == 0) *err = 1;
}

void launch_compress(const uint8_t* d_in, uint32_t chunk_size, uint32_t chunk_count,
                      const uint32_t* d_in_lens, uint8_t* d_out, uint32_t out_slot_stride,
                      uint32_t* d_out_start, uint32_t* d_out_sizes, uint8_t* d_scratch, Mode mode,
                      cudaStream_t stream) {
  uint32_t blocks = (chunk_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
  compress_kernel<<<blocks, kBlockThreads, 0, stream>>>(d_in, chunk_size, chunk_count, d_in_lens, d_out,
                                                         out_slot_stride, d_out_start, d_out_sizes, d_scratch,
                                                         mode);
}

void launch_decompress(const uint8_t* d_in, const uint32_t* d_in_offsets, uint32_t chunk_count,
                        const uint32_t* d_in_lens, uint8_t* d_out, uint32_t chunk_size,
                        const uint32_t* d_out_lens, uint8_t* d_scratch, uint32_t* d_err,
                        cudaStream_t stream) {
  uint32_t blocks = (chunk_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
  decompress_kernel<<<blocks, kBlockThreads, 0, stream>>>(d_in, d_in_offsets, chunk_count, d_in_lens, d_out,
                                                           chunk_size, d_out_lens, d_scratch, d_err);
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
