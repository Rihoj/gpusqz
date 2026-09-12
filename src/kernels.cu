#include "kernels.h"
#include "lzss_kernels.cuh"
#include "format.h"

#include <cub/device/device_scan.cuh>

namespace gzp {

// Fixed-slot layout: chunk c's input lives at in + c*chunk_size (chunk_size
// bytes, last chunk zero-tail padded by the host); chunk c's output slot is
// out + c*out_slot_stride (out_slot_stride = worst_case_size(chunk_size)).
// out_sizes[c] receives the actual bytes written (including the 1-byte flag).
__global__ void compress_kernel(const uint8_t* in, uint32_t chunk_size, uint32_t chunk_count,
                                 const uint32_t* in_lens, uint8_t* out, uint32_t out_slot_stride,
                                 uint32_t* out_sizes) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= chunk_count) return;

  const uint8_t* chunk_in = in + (size_t)c * chunk_size;
  uint32_t in_len = in_lens[c];
  uint8_t* slot = out + (size_t)c * out_slot_stride;

  // Encode into the slot starting at offset 1, reserving byte 0 for the
  // container flag (Raw/Lzss), decided after we see the encoded size.
  uint32_t enc_size = lzss_encode_chunk(chunk_in, in_len, slot + 1);
  if (enc_size < in_len) {
    slot[0] = (uint8_t)ChunkFlag::Lzss;
    out_sizes[c] = 1 + enc_size;
  } else {
    slot[0] = (uint8_t)ChunkFlag::Raw;
    for (uint32_t k = 0; k < in_len; ++k) slot[1 + k] = chunk_in[k];
    out_sizes[c] = 1 + in_len;
  }
}

// Chunk c's compressed data (flag byte + payload) lives at in + in_offsets[c],
// with in_lens[c] valid bytes; chunk c's decompressed output lives at
// out + c*chunk_size, with out_lens[c] valid bytes.
__global__ void decompress_kernel(const uint8_t* in, const uint32_t* in_offsets, uint32_t chunk_count,
                                   const uint32_t* in_lens, uint8_t* out, uint32_t chunk_size,
                                   const uint32_t* out_lens) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= chunk_count) return;

  const uint8_t* slot = in + in_offsets[c];
  uint32_t payload_len = in_lens[c] - 1;
  uint8_t* chunk_out = out + (size_t)c * chunk_size;
  uint32_t original_size = out_lens[c];

  if ((ChunkFlag)slot[0] == ChunkFlag::Raw) {
    for (uint32_t k = 0; k < original_size; ++k) chunk_out[k] = slot[1 + k];
  } else {
    lzss_decode_chunk(slot + 1, payload_len, chunk_out, original_size);
  }
}

void launch_compress(const uint8_t* d_in, uint32_t chunk_size, uint32_t chunk_count,
                      const uint32_t* d_in_lens, uint8_t* d_out, uint32_t out_slot_stride,
                      uint32_t* d_out_sizes, cudaStream_t stream) {
  constexpr int kThreads = 128;
  uint32_t blocks = (chunk_count + kThreads - 1) / kThreads;
  compress_kernel<<<blocks, kThreads, 0, stream>>>(d_in, chunk_size, chunk_count, d_in_lens, d_out,
                                                     out_slot_stride, d_out_sizes);
}

void launch_decompress(const uint8_t* d_in, const uint32_t* d_in_offsets, uint32_t chunk_count,
                        const uint32_t* d_in_lens, uint8_t* d_out, uint32_t chunk_size,
                        const uint32_t* d_out_lens, cudaStream_t stream) {
  constexpr int kThreads = 128;
  uint32_t blocks = (chunk_count + kThreads - 1) / kThreads;
  decompress_kernel<<<blocks, kThreads, 0, stream>>>(d_in, d_in_offsets, chunk_count, d_in_lens,
                                                       d_out, chunk_size, d_out_lens);
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
