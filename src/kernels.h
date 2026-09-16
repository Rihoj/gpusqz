#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "format.h"
#include "rans_codes.h"

namespace gpusqz {

// Every match covers at least kMinMatch bytes, plus one optional tail.
__host__ __device__ inline uint32_t max_sequences(uint32_t chunk_size) { return chunk_size / kMinMatch + 1; }

// The LZ parse's match-finding hash table: one global-memory region per
// chunk, kHashBucketWays positions per bucket. (Declared here, not in
// lz_warp.cuh, because the host sizes it; lz_warp.cuh aliases it.)
constexpr int kHashBucketWays = 4;
// Buckets per profile, measured on the benchmark corpora (bigger tables
// compress better but add global-memory misses):
//   speed   (<= 64KB)  2048: 4096 saves 1.6% of output for ~12% of wall speed
//   balance (<= 256KB) 4096: saves 2.9% for ~11%
//   ratio   (1MB)     32768: saves 2.2% over 8192 at no wall-clock cost;
//                            65536 adds 0.4% for ~18% of kernel speed
__host__ __device__ inline int hash_table_bits(uint32_t chunk_size) {
  return chunk_size <= kDefaultChunkSize ? 11 : chunk_size <= 4 * kDefaultChunkSize ? 12 : 15;
}
// How many recent match offsets the parse also tries at every position
// (see lz_parse_warp; at most kMaxRepProbes). Against none, on the three
// corpora: one gave 0.03-0.33% smaller output for 2-4% of compress kernel
// throughput; a second, another 0.03-0.15% for another ~3%, which `speed`
// doesn't take.
constexpr int kMaxRepProbes = 2;
__host__ __device__ inline int rep_probes(uint32_t chunk_size) { return chunk_size <= kDefaultChunkSize ? 1 : 2; }
__host__ __device__ inline size_t hash_table_bytes(uint32_t chunk_size) {
  return ((size_t)1 << hash_table_bits(chunk_size)) * kHashBucketWays * sizeof(uint32_t);
}

// Per-chunk scratch for the LzRans path, each section 16-byte aligned: an
// 8-byte SeqRec per sequence (lz_warp.cuh static_asserts the size), one
// repeat-offset code byte per sequence, then the literals. Compress reuses
// it afterwards as the destination of the output compaction.
constexpr size_t kSeqRecBytes = 8;
__host__ __device__ inline size_t scratch_seqs_bytes(uint32_t chunk_size) {
  return (kSeqRecBytes * max_sequences(chunk_size) + 15) & ~(size_t)15;
}
__host__ __device__ inline size_t scratch_rep_bytes(uint32_t chunk_size) {
  return ((size_t)max_sequences(chunk_size) + 15) & ~(size_t)15;
}
__host__ __device__ inline size_t scratch_lits_offset(uint32_t chunk_size) {
  return scratch_seqs_bytes(chunk_size) + scratch_rep_bytes(chunk_size);
}
__host__ __device__ inline size_t scratch_bytes(uint32_t chunk_size) {
  return scratch_lits_offset(chunk_size) + (((size_t)chunk_size + 15) & ~(size_t)15);
}

// Per-batch buffers for the LzRans pipeline. cnt, freq, cum and q hold
// kMaxQuantBytes entries each (rans_codes.h): the batch histogram, the
// encode tables, and the quantised counts stored in the file.
struct RansBatchBufs {
  uint32_t* cnt;       // zeroed by launch_compress before parsing
  uint16_t* freq;
  uint16_t* cum;
  uint8_t* q;
  uint32_t* lit_shift; // 1 entry: the literal-context rule this batch chose
  uint32_t* n_seq;     // per chunk: sequences in its scratch
  uint32_t* n_lit;     // per chunk: literals in its scratch
};

// Compresses a batch: chunk c's input is at d_in + c*chunk_size
// (d_in_lens[c] bytes); its output (flag byte + payload) goes into slot
// d_out + c*out_slot_stride, starting d_out_start[c] bytes in, d_out_sizes[c]
// bytes long. d_scratch holds chunk_count * scratch_bytes() and d_htab
// chunk_count * hash_table_bytes(). Runs three kernels on `stream`: parse +
// histogram, build the batch's tables, encode. The batch picks its
// literal-context rule and reports it in *d_rans.lit_shift, unless
// forced_lit_shift >= 0. Each chunk keeps the smallest of rANS, plain LZ
// tokens (ChunkFlag::Lz) and raw storage.
void launch_compress(const uint8_t* d_in, uint32_t chunk_size, uint32_t chunk_count,
                      const uint32_t* d_in_lens, uint8_t* d_out, uint32_t out_slot_stride,
                      uint32_t* d_out_start, uint32_t* d_out_sizes, uint8_t* d_scratch, uint32_t* d_htab,
                      int forced_lit_shift, const RansBatchBufs& d_rans, cudaStream_t stream);

// Decompresses a batch: chunk c's data is at d_in + d_in_offsets[c]
// (d_in_lens[c] bytes); its output at d_out + c*chunk_size (d_out_lens[c]
// bytes). An LzRans chunk decodes against group g = d_group_id[c]'s tables
// at d_group_tables + d_group_off[g] (literal-context rule d_group_shift[g]).
// *d_err (zeroed by the caller) is set nonzero if any chunk is malformed.
void launch_decompress(const uint8_t* d_in, const uint32_t* d_in_offsets, uint32_t chunk_count,
                        const uint32_t* d_in_lens, uint8_t* d_out, uint32_t chunk_size,
                        const uint32_t* d_out_lens, uint8_t* d_scratch, uint8_t* d_group_tables,
                        const uint64_t* d_group_off, const uint32_t* d_group_shift, const uint32_t* d_group_id,
                        uint32_t* d_err, cudaStream_t stream);

// Bytes of one group's expanded decode tables.
size_t rans_group_table_bytes(uint32_t lit_shift);

// Expands each group g's quantised counts (d_q + d_q_off[g], rule
// d_shift[g]) into its decode tables at d_tables + d_table_off[g].
// d_tables must be zeroed first: unused tables stay all-zero, and a zero
// frequency is what flags a corrupt chunk decoding against one.
void launch_expand_group_tables(const uint8_t* d_q, const uint64_t* d_q_off, const uint64_t* d_table_off,
                                const uint32_t* d_shift, uint32_t group_count, uint8_t* d_tables,
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

} // namespace gpusqz
