#include "kernels.h"
#include "lzss_kernels.cuh"
#include "format.h"

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

// Fixed-slot layout: chunk c's compressed data (flag byte + payload) lives at
// in + c*in_slot_stride, with in_lens[c] valid bytes; chunk c's decompressed
// output lives at out + c*chunk_size, with out_lens[c] valid bytes.
__global__ void decompress_kernel(const uint8_t* in, uint32_t in_slot_stride, uint32_t chunk_count,
                                   const uint32_t* in_lens, uint8_t* out, uint32_t chunk_size,
                                   const uint32_t* out_lens) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= chunk_count) return;

  const uint8_t* slot = in + (size_t)c * in_slot_stride;
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

void launch_decompress(const uint8_t* d_in, uint32_t in_slot_stride, uint32_t chunk_count,
                        const uint32_t* d_in_lens, uint8_t* d_out, uint32_t chunk_size,
                        const uint32_t* d_out_lens, cudaStream_t stream) {
  constexpr int kThreads = 128;
  uint32_t blocks = (chunk_count + kThreads - 1) / kThreads;
  decompress_kernel<<<blocks, kThreads, 0, stream>>>(d_in, in_slot_stride, chunk_count, d_in_lens,
                                                       d_out, chunk_size, d_out_lens);
}

} // namespace gzp
