#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "format.h"
#include "rans_codes.h"

namespace gzp {

// Every match covers at least kMinMatch bytes, plus one optional tail.
__host__ __device__ inline uint32_t max_sequences(uint32_t chunk_size) { return chunk_size / kMinMatch + 1; }

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
// size d_out_sizes[c]. d_scratch holds chunk_count * scratch_bytes().
//
// This is actually 3 kernel launches on `stream` (parse + histogram, build
// the shared table, encode against it) rather than 1; d_rans is scratch
// for that (see RansBatchBufs). Each chunk still individually falls back
// to a plain LZ token stream (ChunkFlag::Lz) when that's smaller than the
// rANS encoding, or to Raw storage when neither beats the input.
void launch_compress(const uint8_t* d_in, uint32_t chunk_size, uint32_t chunk_count,
                      const uint32_t* d_in_lens, uint8_t* d_out, uint32_t out_slot_stride,
                      uint32_t* d_out_start, uint32_t* d_out_sizes, uint8_t* d_scratch,
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
