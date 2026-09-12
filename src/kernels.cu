#include "kernels.h"
#include "lz_warp.cuh"
#include "format.h"

#include <cub/device/device_scan.cuh>

namespace gzp {

constexpr int kBlockThreads = kWarpsPerBlock * 32;

// One warp per chunk. Chunk c's input lives at in + c*chunk_size (in_lens[c]
// valid bytes); its output goes to the fixed slot out + c*out_slot_stride
// as [flag byte][payload], with the total size in out_sizes[c].
__global__ void __launch_bounds__(kBlockThreads)
compress_kernel(const uint8_t* in, uint32_t chunk_size, uint32_t chunk_count, const uint32_t* in_lens,
                uint8_t* out, uint32_t out_slot_stride, uint32_t* out_sizes) {
  __shared__ uint32_t htab[kWarpsPerBlock][kHashSize];
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t c = blockIdx.x * kWarpsPerBlock + warp;
  if (c >= chunk_count) return;

  const uint8_t* chunk_in = in + (size_t)c * chunk_size;
  uint32_t in_len = in_lens[c];
  uint8_t* slot = out + (size_t)c * out_slot_stride;

  // Only worth encoding if the result (flag + payload) is no larger than
  // the raw alternative, so cap the payload at in_len - 1.
  uint32_t enc_len = 0;
  bool ok = in_len > 1 && lz_encode_warp(chunk_in, in_len, slot + 1, in_len - 1, &enc_len, htab[warp]);
  if (ok) {
    if (lane == 0) {
      slot[0] = (uint8_t)ChunkFlag::Lz;
      out_sizes[c] = 1 + enc_len;
    }
  } else {
    for (uint32_t k = lane; k < in_len; k += 32) slot[1 + k] = chunk_in[k];
    if (lane == 0) {
      slot[0] = (uint8_t)ChunkFlag::Raw;
      out_sizes[c] = 1 + in_len;
    }
  }
}

// One warp per chunk. Chunk c's compressed data lives at in + in_offsets[c]
// (in_lens[c] bytes, flag first); output goes to out + c*chunk_size,
// out_lens[c] bytes. Any malformed chunk sets *err.
__global__ void __launch_bounds__(kBlockThreads)
decompress_kernel(const uint8_t* in, const uint32_t* in_offsets, uint32_t chunk_count, const uint32_t* in_lens,
                  uint8_t* out, uint32_t chunk_size, const uint32_t* out_lens, uint32_t* err) {
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
    }
  }
  if (!ok && lane == 0) *err = 1;
}

void launch_compress(const uint8_t* d_in, uint32_t chunk_size, uint32_t chunk_count,
                      const uint32_t* d_in_lens, uint8_t* d_out, uint32_t out_slot_stride,
                      uint32_t* d_out_sizes, cudaStream_t stream) {
  uint32_t blocks = (chunk_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
  compress_kernel<<<blocks, kBlockThreads, 0, stream>>>(d_in, chunk_size, chunk_count, d_in_lens, d_out,
                                                         out_slot_stride, d_out_sizes);
}

void launch_decompress(const uint8_t* d_in, const uint32_t* d_in_offsets, uint32_t chunk_count,
                        const uint32_t* d_in_lens, uint8_t* d_out, uint32_t chunk_size,
                        const uint32_t* d_out_lens, uint32_t* d_err, cudaStream_t stream) {
  uint32_t blocks = (chunk_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
  decompress_kernel<<<blocks, kBlockThreads, 0, stream>>>(d_in, d_in_offsets, chunk_count, d_in_lens, d_out,
                                                           chunk_size, d_out_lens, d_err);
}

// One warp per chunk; byte copies because slot payloads have arbitrary
// alignment. This is memory-bound and tiny next to the compress kernel.
__global__ void compact_kernel(const uint8_t* slots, uint32_t slot_stride, const uint32_t* sizes,
                               const uint32_t* offsets, uint32_t n, uint8_t* packed) {
  uint32_t c = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  uint32_t lane = threadIdx.x & 31;
  if (c >= n) return;
  const uint8_t* src = slots + (size_t)c * slot_stride;
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

cudaError_t launch_compact(const uint8_t* d_slots, uint32_t slot_stride, const uint32_t* d_sizes,
                           uint32_t n, uint32_t* d_offsets, uint8_t* d_packed, void* d_temp,
                           size_t temp_bytes, cudaStream_t stream) {
  cudaError_t err = cudaMemsetAsync(d_offsets, 0, sizeof(uint32_t), stream);
  if (err != cudaSuccess) return err;
  err = cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_sizes, d_offsets + 1, (int)n, stream);
  if (err != cudaSuccess) return err;
  constexpr int kThreads = 128;
  uint32_t blocks = (n * 32 + kThreads - 1) / kThreads;
  compact_kernel<<<blocks, kThreads, 0, stream>>>(d_slots, slot_stride, d_sizes, d_offsets, n, d_packed);
  return cudaGetLastError();
}

} // namespace gzp
