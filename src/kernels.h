#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "format.h"

namespace gzp {

enum class Mode : uint32_t {
  Lz = 0,     // LZ tokens only
  LzRans = 1, // LZ parse + rANS entropy coding (falls back to Lz, then Raw)
};

// Every match covers at least kMinMatch bytes, plus one optional tail.
__host__ __device__ inline uint32_t max_sequences(uint32_t chunk_size) { return chunk_size / kMinMatch + 1; }

// Per-chunk device scratch used by the LzRans paths: an 8-byte record per
// sequence followed by the literals, each section 16-byte aligned. Must
// agree with chunk_scratch() in kernels.cu.
__host__ __device__ inline size_t scratch_bytes(uint32_t chunk_size) {
  size_t seqs = ((size_t)8 * max_sequences(chunk_size) + 15) & ~(size_t)15;
  size_t lits = ((size_t)chunk_size + 15) & ~(size_t)15;
  return seqs + lits;
}

// Chunk c's input lives at d_in + c*chunk_size with d_in_lens[c] valid
// bytes. Its output (flag byte + payload) is written into the fixed slot
// d_out + c*out_slot_stride starting at byte d_out_start[c], with total
// size d_out_sizes[c]. d_scratch holds chunk_count * scratch_bytes().
void launch_compress(const uint8_t* d_in, uint32_t chunk_size, uint32_t chunk_count,
                      const uint32_t* d_in_lens, uint8_t* d_out, uint32_t out_slot_stride,
                      uint32_t* d_out_start, uint32_t* d_out_sizes, uint8_t* d_scratch, Mode mode,
                      cudaStream_t stream);

// Chunk c's compressed data lives at d_in + d_in_offsets[c] with
// d_in_lens[c] bytes; its output at d_out + c*chunk_size, d_out_lens[c] bytes.
// *d_err (zeroed by the caller) is set nonzero if any chunk is malformed.
void launch_decompress(const uint8_t* d_in, const uint32_t* d_in_offsets, uint32_t chunk_count,
                        const uint32_t* d_in_lens, uint8_t* d_out, uint32_t chunk_size,
                        const uint32_t* d_out_lens, uint8_t* d_scratch, uint32_t* d_err,
                        cudaStream_t stream);

// Scratch needed by launch_compact for up to max_chunks chunks.
size_t compaction_temp_bytes(uint32_t max_chunks);

// Packs fixed-slot chunk outputs contiguously: d_offsets (n+1 entries) gets
// the exclusive prefix sum of d_sizes with d_offsets[n] = total, and
// d_packed[d_offsets[c] ..] receives chunk c's d_sizes[c] bytes read from
// slot c at byte offset d_src_off[c].
cudaError_t launch_compact(const uint8_t* d_slots, uint32_t slot_stride, const uint32_t* d_src_off,
                           const uint32_t* d_sizes, uint32_t n, uint32_t* d_offsets, uint8_t* d_packed,
                           void* d_temp, size_t temp_bytes, cudaStream_t stream);

} // namespace gzp
