#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "format.h"
#include "rans_codes.h"

namespace gzp {

// Every match covers at least kMinMatch bytes, plus one optional tail.
__host__ __device__ inline uint32_t max_sequences(uint32_t chunk_size) { return chunk_size / kMinMatch + 1; }

// Sizing for the LZ parse's per-chunk match-finding hash table (lz_warp.cuh).
// This table now lives in *global* memory (one region per chunk, allocated
// alongside the rest of compress's per-chunk scratch) rather than shared
// memory, specifically so its size can scale with chunk_size without
// touching the 48KB-per-block shared-memory threshold that caused a real
// regression the last time this table was grown (see the comment in
// lz_warp.cuh). kHashBucketWays here must match lz_warp.cuh's
// kBucketWays -- duplicated for the same layering reason scratch_bytes()
// duplicates SeqRec's size (lz_warp.cuh includes this header, not the
// other way around).
constexpr int kHashBucketWays = 4;
// Only 3 chunk sizes actually get exercised in practice (the `speed`/
// `balance`/`ratio` profile presets, format.h), so this is a measured
// three-tier choice rather than a generic formula extrapolated to sizes
// nothing has validated. Re-measured once batches were sized for
// occupancy (main.cu's plan_batches): the earlier "a bigger table costs
// ~2x" results were taken with ~32 warps in flight, where every extra
// global-memory miss was fully exposed. Size change and compress-kernel
// change per step, on the repetitive 283MB and the varied 1GB corpora:
// - <= the default 64KB (`speed`): 2048 buckets. 4096 saves 1.6% of
//   output but costs ~18-20% of kernel and ~12% of wall throughput, the
//   wrong trade for the default, fastest profile.
// - <= 4x the default (`balance`'s 256KB): 4096 buckets. Saves ~2.9% of
//   output for ~21-25% of kernel and ~11% of wall throughput.
// - above that (`ratio`'s 1MB): 32768 buckets. 8192 -> 16384 saved 1.4%,
//   -> 32768 another 0.8%, for ~3-4% of kernel throughput in total and no
//   measurable wall-clock cost. 65536 saved only 0.4% more and cost ~18%
//   of kernel throughput on the varied corpus.
__host__ __device__ inline int hash_table_bits(uint32_t chunk_size) {
  return chunk_size <= kDefaultChunkSize ? 11 : chunk_size <= 4 * kDefaultChunkSize ? 12 : 15;
}
__host__ __device__ inline size_t hash_table_bytes(uint32_t chunk_size) {
  return ((size_t)1 << hash_table_bits(chunk_size)) * kHashBucketWays * sizeof(uint32_t);
}

// Per-chunk device scratch used by the LzRans paths: an 8-byte record per
// sequence (SeqRec, lz_warp.cuh, which static_asserts its size against
// kSeqRecBytes -- that header includes this one, not the other way
// around), then one repeat-offset-code byte per sequence
// (compute_repeat_codes(), rans.cuh), then the literals -- each section
// 16-byte aligned. chunk_scratch() in kernels.cu carves it up with the
// helpers below.
//
// Compress also reuses the batch's scratch as the destination of the
// output compaction (launch_compact's d_packed): scratch is dead once the
// encode kernel has run, and main.cu allocates it at least one output slot
// per chunk (compress_scratch_bytes).
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

// Per-batch buffers for the LzRans pipeline, sized independently of
// chunk_size/chunk_count: the running histogram, its expansion into
// freq/cum, and the quantised bytes the host reads back to store in this
// batch's TableGroup. Allocate kMaxQuantBytes entries each (rans_codes.h);
// a batch uses quant_bytes(lit_ctx_count(*lit_shift)) of them.
struct RansBatchBufs {
  uint32_t* cnt;       // zeroed by launch_compress before parsing
  uint16_t* freq;
  uint16_t* cum;
  uint8_t* q;
  uint32_t* lit_shift; // 1 entry: the literal-context rule this batch chose
  uint32_t* n_seq;  // batch-sized: valid sequence count per chunk's scratch
  uint32_t* n_lit;  // batch-sized: valid literal count per chunk's scratch
};

// Chunk c's input lives at d_in + c*chunk_size with d_in_lens[c] valid
// bytes. Its output (flag byte + payload) is written into the fixed slot
// d_out + c*out_slot_stride starting at byte d_out_start[c], with total
// size d_out_sizes[c]. d_scratch holds chunk_count * scratch_bytes();
// d_htab holds chunk_count * hash_table_bytes(chunk_size) -- the LZ
// parse's match-finding table, one region per chunk (see hash_table_bits
// above for why this is a separate global-memory buffer rather than
// living in d_scratch or shared memory).
//
// This is actually 3 kernel launches on `stream` (parse + histogram, build
// the shared tables, encode against them) rather than 1; d_rans is scratch
// for that (see RansBatchBufs). The batch picks its own literal-context
// rule (rans_codes.h) and reports it in *d_rans.lit_shift, unless
// forced_lit_shift >= 0 forces one. Each chunk still individually falls
// back to a plain LZ token stream (ChunkFlag::Lz) when that's smaller than
// the rANS encoding, or to Raw storage when neither beats the input.
void launch_compress(const uint8_t* d_in, uint32_t chunk_size, uint32_t chunk_count,
                      const uint32_t* d_in_lens, uint8_t* d_out, uint32_t out_slot_stride,
                      uint32_t* d_out_start, uint32_t* d_out_sizes, uint8_t* d_scratch, uint32_t* d_htab,
                      int forced_lit_shift, const RansBatchBufs& d_rans, cudaStream_t stream);

// Chunk c's compressed data lives at d_in + d_in_offsets[c] with
// d_in_lens[c] bytes; its output at d_out + c*chunk_size, d_out_lens[c]
// bytes. LzRans chunks look up their tables in group g = d_group_id[c]'s
// region at d_group_tables + d_group_off[g], expanded (by
// launch_expand_group_tables) for its literal-context rule d_group_shift[g].
// *d_err (zeroed by the caller) is set nonzero if any chunk is malformed.
void launch_decompress(const uint8_t* d_in, const uint32_t* d_in_offsets, uint32_t chunk_count,
                        const uint32_t* d_in_lens, uint8_t* d_out, uint32_t chunk_size,
                        const uint32_t* d_out_lens, uint8_t* d_scratch, uint8_t* d_group_tables,
                        const uint64_t* d_group_off, const uint32_t* d_group_shift, const uint32_t* d_group_id,
                        uint32_t* d_err, cudaStream_t stream);

// Bytes of one group's expanded decode tables (opaque here so main.cu need
// not include rans.cuh just to size a buffer of them).
size_t rans_group_table_bytes(uint32_t lit_shift);

// Expands each group g's quantised bytes (at d_q + d_q_off[g], for
// literal-context rule d_shift[g]) into its table region at d_tables +
// d_table_off[g], rans_group_table_bytes(d_shift[g]) bytes. d_tables must
// already be zeroed: an all-zero table (an unused context, or a group with
// no LzRans chunks) leaves its region untouched, and a zero frequency is
// what flags a corrupt chunk that decodes against one.
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

} // namespace gzp
