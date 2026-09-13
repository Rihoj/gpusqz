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
// two-tier choice rather than a generic formula extrapolated to sizes
// nothing has validated:
// - <= 4x the default (up to and including `balance`'s 256KB): stays at
//   2048 buckets, the original shared-memory-era table size. Growing
//   `balance`'s table by even one step measured a real ~25-30%
//   compress-kernel regression on varied real data (this table's now in
//   global memory rather than shared, so that's a genuine bandwidth/
//   latency cost, not the shared-memory occupancy or cache-partition
//   costs a bigger *shared* table paid -- see lz_warp.cuh) for a ratio
//   gain that didn't justify it there.
// - above that (in practice, just `ratio`'s 1MB): 8192 buckets (4x).
//   Measured a real, consistent ~2x compress-kernel cost on both a
//   repetitive and a genuinely varied large corpus (not a corpus-
//   dependent cliff like the shared-memory attempts), in exchange for
//   gzp's biggest ratio win this session -- a trade `ratio`'s whole
//   purpose is to prefer. Growing further wasn't tried; the two
//   corpora above are what it should be validated against before
//   moving this constant.
__host__ __device__ inline int hash_table_bits(uint32_t chunk_size) {
  return chunk_size <= 4 * kDefaultChunkSize ? 11 : 13;
}
__host__ __device__ inline size_t hash_table_bytes(uint32_t chunk_size) {
  return ((size_t)1 << hash_table_bits(chunk_size)) * kHashBucketWays * sizeof(uint32_t);
}

// Per-chunk device scratch used by the LzRans paths: a 12-byte record per
// sequence, then one repeat-offset-code byte per sequence
// (compute_repeat_codes(), rans.cuh), then the literals -- each section
// 16-byte aligned. Must agree with chunk_scratch() in kernels.cu (which
// uses sizeof(SeqRec) directly; this copy exists only because lz_warp.cuh,
// where SeqRec is defined, includes this header, not the other way
// around).
__host__ __device__ inline size_t scratch_bytes(uint32_t chunk_size) {
  size_t seqs = ((size_t)12 * max_sequences(chunk_size) + 15) & ~(size_t)15;
  size_t rep = ((size_t)max_sequences(chunk_size) + 15) & ~(size_t)15;
  size_t lits = ((size_t)chunk_size + 15) & ~(size_t)15;
  return seqs + rep + lits;
}

// Per-batch buffers for the LzRans pipeline, all sized independently of
// chunk_size/chunk_count (kQuantBytes is a small fixed constant): the
// running histogram, its expansion into freq/cum, and the quantised bytes
// the host reads back to store in this batch's TableGroup.
struct RansBatchBufs {
  uint32_t* cnt;    // kQuantBytes, zeroed by the caller before parsing
  uint16_t* freq;   // kQuantBytes
  uint16_t* cum;    // kQuantBytes
  uint8_t* q;       // kQuantBytes
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
// the shared table, encode against it) rather than 1; d_rans is scratch
// for that (see RansBatchBufs). Each chunk still individually falls back
// to a plain LZ token stream (ChunkFlag::Lz) when that's smaller than the
// rANS encoding, or to Raw storage when neither beats the input.
void launch_compress(const uint8_t* d_in, uint32_t chunk_size, uint32_t chunk_count,
                      const uint32_t* d_in_lens, uint8_t* d_out, uint32_t out_slot_stride,
                      uint32_t* d_out_start, uint32_t* d_out_sizes, uint8_t* d_scratch, uint32_t* d_htab,
                      const RansBatchBufs& d_rans, cudaStream_t stream);

// Chunk c's compressed data lives at d_in + d_in_offsets[c] with
// d_in_lens[c] bytes; its output at d_out + c*chunk_size, d_out_lens[c]
// bytes. LzRans chunks look up their table at d_group_tables[d_group_id[c]]
// (already expanded by launch_expand_group_tables). *d_err (zeroed by the
// caller) is set nonzero if any chunk is malformed.
void launch_decompress(const uint8_t* d_in, const uint32_t* d_in_offsets, uint32_t chunk_count,
                        const uint32_t* d_in_lens, uint8_t* d_out, uint32_t chunk_size,
                        const uint32_t* d_out_lens, uint8_t* d_scratch, const void* d_group_tables,
                        const uint32_t* d_group_id, uint32_t* d_err, cudaStream_t stream);

// Bytes needed by one RansGroupTable (opaque here so main.cu need not
// include rans.cuh just to size a buffer of them).
size_t rans_group_table_bytes();

// Expands group_count groups' quantised bytes (d_q, group_count *
// kQuantBytes) into group_count RansGroupTables at d_tables (must already
// be zeroed: groups with no LzRans chunks have an all-zero q[], which
// leaves their table's freq/cum untouched — harmless since no chunk ever
// references such a group, but the memory must not be garbage regardless).
void launch_expand_group_tables(const uint8_t* d_q, uint32_t group_count, void* d_tables, cudaStream_t stream);

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
