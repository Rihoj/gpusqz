#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace gzp {

// Chunk c's input lives at d_in + c*chunk_size with d_in_lens[c] valid
// bytes; its output (flag byte + payload) is written to the fixed slot
// d_out + c*out_slot_stride and its size to d_out_sizes[c].
void launch_compress(const uint8_t* d_in, uint32_t chunk_size, uint32_t chunk_count,
                      const uint32_t* d_in_lens, uint8_t* d_out, uint32_t out_slot_stride,
                      uint32_t* d_out_sizes, cudaStream_t stream);

// Chunk c's compressed data lives at d_in + d_in_offsets[c] with
// d_in_lens[c] bytes; its output at d_out + c*chunk_size, d_out_lens[c] bytes.
void launch_decompress(const uint8_t* d_in, const uint32_t* d_in_offsets, uint32_t chunk_count,
                        const uint32_t* d_in_lens, uint8_t* d_out, uint32_t chunk_size,
                        const uint32_t* d_out_lens, cudaStream_t stream);

// Scratch needed by launch_compact for up to max_chunks chunks.
size_t compaction_temp_bytes(uint32_t max_chunks);

// Packs fixed-slot chunk outputs contiguously: d_offsets (n+1 entries) gets
// the exclusive prefix sum of d_sizes with d_offsets[n] = total, and
// d_packed[d_offsets[c] ..] receives chunk c's d_sizes[c] bytes.
cudaError_t launch_compact(const uint8_t* d_slots, uint32_t slot_stride, const uint32_t* d_sizes,
                           uint32_t n, uint32_t* d_offsets, uint8_t* d_packed, void* d_temp,
                           size_t temp_bytes, cudaStream_t stream);

} // namespace gzp
