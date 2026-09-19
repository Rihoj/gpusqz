// Shared by the Vulkan compute shaders: format constants, the rANS symbol
// coding (a port of rans_codes.h), and the lane-group layer.
//
// Every chunk is handled by one workgroup of 32 invocations, the "lane
// group" standing in for a CUDA warp. The format depends on exactly 32
// lanes (32 interleaved rANS states, lit_run_len), whatever the hardware's
// subgroup size, so the lane-group operations come in two builds:
//   LG_SUBGROUP=1  the 32 lanes are one aligned 32-lane window of a single
//                  subgroup (subgroup size 32, as on NVIDIA and Apple, or 64
//                  with half the lanes idle, as on AMD GCN). The Vulkan
//                  backend checks this at startup with probe.comp.
//   LG_SUBGROUP=0  any subgroup size (e.g. 8 or 16 on Intel and llvmpipe):
//                  ballots and shuffles go through shared memory and
//                  workgroup barriers. Correct everywhere, slower.
// Like the CUDA code, all lane-group functions are collective: all 32 lanes
// must call them with uniform control flow.
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#if LG_SUBGROUP
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_vote : require
#endif

// ---- Constants: must match format.h, rans_codes.h and lz_warp.cuh ----
const uint kMinMatch = 4;
const uint kProbBits = 12;
const uint kProbScale = 1u << kProbBits;
const uint kRansL = 1u << 16;
const uint kLitSyms = 256;
const uint kSmallSyms = 32;
const uint kOffRepBase = kSmallSyms - 3;
const uint kMaxLenValue = (1u << (kSmallSyms - 12)) - 1;
const uint kRansStates = 32;
const uint kRansHeaderBytes = 8 + kRansStates * 4;
const uint kLitShiftOrder0 = 8, kLitShiftNibble = 4, kLitShiftByte = 0;
const uint kLlBase = 0;
const uint kMlBase = kSmallSyms;
const uint kOffBase = 2 * kSmallSyms;
const uint kLitBase = 3 * kSmallSyms;
const uint kMaxQuantBytes = kLitBase + 256 * kLitSyms;
const uint kFlagRaw = 0, kFlagLz = 1, kFlagLzRans = 2;
const uint kSmallLutBits = 7;
const uint kSmallLutSize = 1u << kSmallLutBits;
const uint kEmptyPos = 0xFFFFFFFFu;

// ---- rans_codes.h ----
uint lit_ctx_count(uint shift) { return 256u >> shift; }
uint lit_ctx(uint prev, uint shift) { return prev >> shift; }
uint lit_run_len(uint n_lit) { return ((n_lit + 31) / 32 + 3) & ~3u; }
uint lit_entry(uint ctx, uint sym) { return kLitBase + ctx * kLitSyms + sym; }
uint quant_bytes(uint n_ctx) { return kLitBase + n_ctx * kLitSyms; }

void len_code(uint v, out uint code, out uint nb, out uint bits) {
  if (v < 16) {
    code = v;
    nb = 0;
    bits = 0;
  } else {
    uint l = uint(findMSB(v));
    code = 12 + l;
    nb = l;
    bits = v - (1u << l);
  }
}
uint len_nb(uint code) { return code < 16 ? 0 : code - 12; }
uint len_value(uint code, uint bits) { return code < 16 ? code : (1u << (code - 12)) + bits; }
void off_code(uint off, out uint code, out uint nb, out uint bits) {
  uint l = uint(findMSB(off));
  code = l;
  nb = l;
  bits = off - (1u << l);
}
uint off_value(uint code, uint bits) { return (1u << code) + bits; }
uint off_nb(uint code) { return code >= kOffRepBase ? 0 : code; }

// Encode/decode tables pack freq (low 16 bits) and cum (high 16) per entry.
uint fc_freq(uint v) { return v & 0xFFFFu; }
uint fc_cum(uint v) { return v >> 16; }

// ---- lz_warp.cuh's SeqRec, as two words: three 21-bit fields ----
uvec2 seq_pack(uint lit_len, uint off, uint ml) { return uvec2(lit_len | (off << 21), (off >> 11) | (ml << 10)); }
uint seq_ll(uvec2 r) { return r.x & 0x1FFFFFu; }
uint seq_off(uvec2 r) { return (r.x >> 21) | ((r.y & 0x3FFu) << 11); }
uint seq_ml(uvec2 r) { return r.y >> 10; }

uint ext_bytes(uint v) { return v >= 15 ? 1 + (v - 15) / 255 : 0; }
uint seq_size(uint lit_len, uint ml) {
  uint s = 1 + ext_bytes(lit_len) + lit_len;
  if (ml != 0) s += 4 + ext_bytes(ml - kMinMatch);
  return s;
}

// ---- Repeat offsets (rans.cuh's RepOffsets) ----
const uint kRepNew = 3;
uint rep_find(uvec3 r, uint off) { return off == r.x ? 0 : off == r.y ? 1 : off == r.z ? 2 : kRepNew; }
uint rep_get(uvec3 r, uint slot) { return slot == 0 ? r.x : slot == 1 ? r.y : r.z; }
void rep_use(inout uvec3 r, uint slot, uint off) {
  bool front = slot == 0;
  r.z = slot >= 2 ? r.y : r.z;
  r.y = front ? r.y : r.x;
  r.x = front ? r.x : off;
}

// format.h's chunk_hash: lane l folds bytes l, l + 32, ..., then every lane
// folds all 32 lane hashes in lane order, so all lanes end with the same
// value. The buffer the bytes come from differs per shader, so the caller
// passes its fold of one lane's bytes and this combines them.
const uint kHashInit = 2166136261u;
const uint kHashMul = 16777619u;
uint hash_fold(uint h, uint b) { return (h ^ b) * kHashMul; }

// ---- Lane-group layer ----
#if LG_SUBGROUP

uint lg_lane() { return gl_SubgroupInvocationID & 31u; }
// This lane group's word of a subgroup-wide ballot, and its first lane.
uint lg_word() { return gl_SubgroupInvocationID >> 5; }
uint lg_base() { return gl_SubgroupInvocationID & ~31u; }
uint lg_ballot(bool p) { return subgroupBallot(p)[lg_word()]; }
bool lg_any(bool p) { return subgroupAny(p); }
uint lg_shfl(uint v, uint src) { return subgroupShuffle(v, lg_base() | src); }
uint lg_sum(uint v) { return subgroupAdd(v); }
// __syncwarp(): orders the lanes' buffer accesses around it
// (subgroupBarrier is an execution and memory barrier for all memory).
void lg_sync() { subgroupBarrier(); }

#else

shared uint lg_x[32];
uint lg_lane() { return gl_LocalInvocationIndex; }
uint lg_ballot(bool p) {
  barrier(); // earlier readers of lg_x are done
  lg_x[lg_lane()] = p ? 1u << lg_lane() : 0u;
  barrier();
  uint m = 0;
  for (uint i = 0; i < 32; ++i) m |= lg_x[i];
  return m;
}
bool lg_any(bool p) { return lg_ballot(p) != 0; }
uint lg_shfl(uint v, uint src) {
  barrier();
  lg_x[lg_lane()] = v;
  barrier();
  return lg_x[src];
}
uint lg_sum(uint v) {
  barrier();
  lg_x[lg_lane()] = v;
  barrier();
  uint s = 0;
  for (uint i = 0; i < 32; ++i) s += lg_x[i];
  return s;
}
void lg_sync() {
  memoryBarrierBuffer();
  barrier();
}

#endif

// Combines the 32 lane folds (see hash_fold) into the chunk's checksum.
uint hash_combine(uint lane_h, uint n) {
  uint out_h = kHashInit;
  for (uint l = 0; l < 32u; ++l) {
    uint v = lg_shfl(lane_h, l);
    for (int b = 0; b < 4; ++b) out_h = hash_fold(out_h, (v >> (8 * b)) & 0xFFu);
  }
  return out_h ^ n;
}


uint lanemask_lt() { return (1u << lg_lane()) - 1u; }
