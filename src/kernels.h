#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace gzp {

void launch_compress(const uint8_t* d_in, uint32_t chunk_size, uint32_t chunk_count,
                      const uint32_t* d_in_lens, uint8_t* d_out, uint32_t out_slot_stride,
                      uint32_t* d_out_sizes, cudaStream_t stream);

void launch_decompress(const uint8_t* d_in, uint32_t in_slot_stride, uint32_t chunk_count,
                        const uint32_t* d_in_lens, uint8_t* d_out, uint32_t chunk_size,
                        const uint32_t* d_out_lens, cudaStream_t stream);

} // namespace gzp
