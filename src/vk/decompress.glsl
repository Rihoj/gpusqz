// Descriptor bindings and push constants shared by the decompress shaders
// (see vk_backend.cpp's DecompressBinding). Chunk c's compressed data is at
// inb + in_offsets[c]; its output at c * chunk_size in outb; its scratch
// sequences at c * max_seq and literals, 4 per word, at word
// c * ((chunk_size + 3) / 4).
//
// Group tables: group g's info is ginfo[g] = (q_off, sym_off, fc_off,
// shift). Its literal slot -> symbol rows (n_ctx * kProbScale bytes) start
// at gsym[sym_off], followed by the ll, ml and off coarse LUTs
// (kSmallLutSize bytes each); its packed freq/cum entries, indexed like
// the quantised counts, at gfc[fc_off].
layout(std430, binding = 0) buffer InB { uint8_t inb[]; };
layout(std430, binding = 1) buffer InOffsets { uint in_offsets[]; };
layout(std430, binding = 2) buffer InLens { uint in_lens[]; };
layout(std430, binding = 3) buffer OutLens { uint out_lens[]; };
layout(std430, binding = 4) buffer GroupId { uint group_id[]; };
layout(std430, binding = 5) buffer OutB { uint8_t outb[]; };
layout(std430, binding = 6) buffer Seqs { uvec2 seqs[]; };
layout(std430, binding = 7) buffer Lits { uint lits[]; };
layout(std430, binding = 8) buffer Err { uint err[]; };
layout(std430, binding = 9) buffer GInfo { uvec4 ginfo[]; };
layout(std430, binding = 10) buffer GSym { uint8_t gsym[]; };
layout(std430, binding = 11) buffer GFc { uint gfc[]; };
layout(std430, binding = 12) buffer GQ { uint8_t gq[]; };

layout(push_constant) uniform PC {
  uint chunk_size;
  uint chunk_count;
  uint max_seq;
} pc;
