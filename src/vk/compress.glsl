// Descriptor bindings and push constants shared by the compress shaders
// (one descriptor set per buffer set; see vk_backend.cpp's CompressBinding).
// Per chunk c: input at c * chunk_size, output slot at c * slot_stride,
// sequences at c * max_seq, literals at c * chunk_size, hash table at
// c * (4 << hash_bits) words. After compaction `inb` holds the packed output.
// SCRATCH_QUAL is `coherent` in shaders where lanes read scratch that other
// lanes wrote during the same dispatch.
#ifndef SCRATCH_QUAL
#define SCRATCH_QUAL
#endif
layout(std430, binding = 0) buffer InB { uint8_t inb[]; };
layout(std430, binding = 1) buffer InLens { uint in_lens[]; };
layout(std430, binding = 2) buffer Slots { uint8_t slots[]; };
layout(std430, binding = 3) buffer Start { uint out_start[]; };
layout(std430, binding = 4) buffer Sizes { uint out_sizes[]; };
layout(std430, binding = 5) buffer Offsets { uint out_offsets[]; };
layout(std430, binding = 6) SCRATCH_QUAL buffer Seqs { uvec2 seqs[]; };
layout(std430, binding = 7) SCRATCH_QUAL buffer Rep { uint8_t rep_code[]; };
layout(std430, binding = 8) SCRATCH_QUAL buffer Lits { uint8_t lits[]; };
layout(std430, binding = 9) SCRATCH_QUAL buffer Htab { uint htab[]; };
layout(std430, binding = 10) buffer NSeq { uint n_seq_arr[]; };
layout(std430, binding = 11) buffer NLit { uint n_lit_arr[]; };
layout(std430, binding = 12) buffer Cnt { uint batch_cnt[]; };
layout(std430, binding = 13) buffer Fc { uint fc[]; };
layout(std430, binding = 14) buffer Q { uint8_t q_out[]; };
layout(std430, binding = 15) buffer Shift { uint lit_shift_buf[]; };
layout(std430, binding = 16) buffer OutHash { uint out_hash[]; }; // format.h's chunk_hash per chunk

layout(push_constant) uniform PC {
  uint chunk_size;
  uint chunk_count;
  uint slot_stride;
  uint hash_bits;
  uint max_seq;
  int forced_shift;
  uint group_chunks; // chunks per TableGroup (format.h); batch_cnt, fc and
                     // q_out hold kMaxQuantBytes entries per group
} pc;
