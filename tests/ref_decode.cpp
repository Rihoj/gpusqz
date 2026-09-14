// Independent CPU decoder for the gpusqz container, used by the tests to
// check that GPU-compressed files decode correctly by an implementation
// that shares no code with the GPU path (a symmetric bug in the GPU
// encoder and decoder would still round-trip on the GPU alone).
//
//   gpusqz_refdec <input.gsz> <output>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>

#include "file_io.h"
#include "format.h"
#include "rans_codes.h"

using namespace gpusqz;

namespace {

[[noreturn]] void fail(const std::string& msg) {
  std::fprintf(stderr, "gpusqz_refdec: %s\n", msg.c_str());
  std::exit(1);
}

uint32_t load_u32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

struct Seq {
  uint32_t lit_len, ml, off;
};

// One TableGroup's tables, expanded from its quantised counts: the three
// small alphabets, then one literal table per context (layout in
// rans_codes.h).
struct Tables {
  int n_ctx = 1;
  std::vector<uint16_t> freq, cum;
  std::vector<uint8_t> sym; // n_ctx * kProbScale, literal slot -> symbol per context

  Tables(const uint8_t* q, int nctx) : n_ctx(nctx), freq(quant_bytes(nctx), 0), cum(quant_bytes(nctx), 0),
      sym((size_t)nctx * kProbScale, 0) {
    for (int base : {kLlBase, kMlBase, kOffBase}) normalize_table(q + base, kSmallSyms, &freq[base], &cum[base]);
    for (int c = 0; c < n_ctx; ++c) {
      int base = lit_entry(c, 0);
      normalize_table(q + base, kLitSyms, &freq[base], &cum[base]);
      for (int s = 0; s < kLitSyms; ++s) {
        for (uint32_t k = 0; k < freq[base + s]; ++k) sym[(size_t)c * kProbScale + cum[base + s] + k] = (uint8_t)s;
      }
    }
  }
};

// Serial emulation of the GPU's 32-lane interleaved rANS decoder. The GPU
// runs each sub-step for all lanes at once, and the lanes that need a word
// take one in lane order, so visiting lanes 0..31 within each sub-step and
// reading a word whenever that lane needs one consumes the stream
// identically.
struct RansDecoder {
  const uint8_t* p;
  size_t len, rp;
  uint32_t x[32];
  const Tables& t;
  bool bad = false;

  RansDecoder(const uint8_t* payload, size_t n, const Tables& tables) : p(payload), len(n), rp(kRansHeaderBytes),
      t(tables) {
    for (int l = 0; l < 32; ++l) x[l] = load_u32(p + 8 + 4 * l);
  }

  void renorm(int l) {
    if (x[l] < kRansL) {
      if (rp + 2 > len) {
        bad = true;
        return;
      }
      x[l] = (x[l] << 16) | ((uint32_t)p[rp] | ((uint32_t)p[rp + 1] << 8));
      rp += 2;
    }
  }
  // Decodes one symbol for lane l from a small alphabet at `base`, then
  // renormalises that lane.
  uint32_t get(int l, int base, int k) {
    uint32_t slot = x[l] & (kProbScale - 1);
    uint32_t s = 0;
    for (int i = 0; i < k; ++i) {
      if (t.cum[base + i] <= slot) s = i;
    }
    step(l, base + (int)s, slot);
    return s;
  }
  // Decodes one literal for lane l in context ctx.
  uint32_t get_lit(int l, uint32_t ctx) {
    if ((int)ctx >= t.n_ctx) {
      bad = true;
      return 0;
    }
    uint32_t slot = x[l] & (kProbScale - 1);
    uint32_t s = t.sym[(size_t)ctx * kProbScale + slot];
    step(l, lit_entry(ctx, s), slot);
    return s;
  }
  void step(int l, int e, uint32_t slot) {
    if (t.freq[e] == 0) bad = true;
    x[l] = t.freq[e] * (x[l] >> kProbBits) + slot - t.cum[e];
    renorm(l);
  }
  // Mirrors the GPU's rans_dec_bits16: extracts at most 16 bits for lane l
  // (0 if !on), then renormalises only if on. Called for every lane at the
  // same step, like the GPU's whole-warp ballot (see bits_pass).
  uint32_t bits16(int l, bool on, uint32_t nb) {
    on = on && nb != 0;
    uint32_t b = on ? (x[l] & ((1u << nb) - 1)) : 0;
    if (on) x[l] >>= nb;
    if (on && x[l] < kRansL) {
      if (rp + 2 > len) {
        bad = true;
        return b;
      }
      x[l] = (x[l] << 16) | ((uint32_t)p[rp] | ((uint32_t)p[rp + 1] << 8));
      rp += 2;
    }
    return b;
  }
};

// Decodes nb[l] raw bits for every lane in two whole-warp phases, all
// lanes' high parts (nb > 16) first, then all lanes' low 16 bits, as the
// GPU does. Doing one lane's full nb before the next would consume the
// shared stream's words in the wrong order.
void bits_pass(RansDecoder& d, bool active[32], const uint32_t nb[32], uint32_t out[32]) {
  uint32_t hi[32];
  for (int l = 0; l < 32; ++l) {
    uint32_t hi_nb = nb[l] > 16 ? nb[l] - 16 : 0;
    hi[l] = d.bits16(l, active[l], hi_nb);
  }
  for (int l = 0; l < 32; ++l) {
    uint32_t lo_nb = nb[l] > 16 ? 16 : nb[l];
    uint32_t lo = d.bits16(l, active[l], lo_nb);
    out[l] = (hi[l] << 16) | lo;
  }
}

bool decode_lzrans(const uint8_t* in, size_t in_len, uint8_t* out, size_t orig, uint32_t chunk_size,
                   const Tables& tables, uint32_t lit_shift) {
  if (in_len < (size_t)kRansHeaderBytes) return false;
  uint32_t n_seq = load_u32(in), n_lit = load_u32(in + 4);
  if (n_seq > chunk_size / kMinMatch + 1 || n_lit > chunk_size) return false;
  RansDecoder d(in, in_len, tables);
  std::vector<Seq> seqs(n_seq);
  std::vector<uint8_t> lits(n_lit);
  uint32_t llc[32], mlc[32], oc[32], llb[32], mlb[32], ob[32], ml[32], nb[32], off[32];
  bool active[32], has_off[32];
  // Recent-offset state for repeat codes (kOffRepBase), carried across
  // groups; walking lanes 0..31 in order reproduces the GPU's replay.
  uint32_t rep0 = 0, rep1 = 0, rep2 = 0;

  for (uint32_t g = 0; g < (n_seq + 31) / 32; ++g) {
    for (int l = 0; l < 32; ++l) active[l] = g * 32 + l < n_seq;
    for (int l = 0; l < 32; ++l) if (active[l]) llc[l] = d.get(l, kLlBase, kSmallSyms);
    for (int l = 0; l < 32; ++l) nb[l] = active[l] ? len_nb(llc[l]) : 0;
    bits_pass(d, active, nb, llb);
    for (int l = 0; l < 32; ++l) if (active[l]) mlc[l] = d.get(l, kMlBase, kSmallSyms);
    for (int l = 0; l < 32; ++l) nb[l] = active[l] ? len_nb(mlc[l]) : 0;
    bits_pass(d, active, nb, mlb);
    for (int l = 0; l < 32; ++l) {
      if (!active[l]) continue;
      uint32_t v = len_value(mlc[l], mlb[l]);
      ml[l] = v ? v + (kMinMatch - 1) : 0;
    }
    for (int l = 0; l < 32; ++l) has_off[l] = active[l] && ml[l] != 0;
    for (int l = 0; l < 32; ++l) if (has_off[l]) oc[l] = d.get(l, kOffBase, kSmallSyms);
    for (int l = 0; l < 32; ++l) nb[l] = has_off[l] ? off_nb(oc[l]) : 0;
    bits_pass(d, has_off, nb, ob);
    for (int l = 0; l < 32; ++l) {
      uint32_t jo = 0;
      if (has_off[l]) {
        if (oc[l] >= kOffRepBase) {
          uint32_t slot = oc[l] - kOffRepBase;
          jo = slot == 0 ? rep0 : slot == 1 ? rep1 : rep2;
          if (slot == 1) {
            rep1 = rep0;
            rep0 = jo;
          } else if (slot == 2) {
            rep2 = rep1;
            rep1 = rep0;
            rep0 = jo;
          }
        } else {
          jo = off_value(oc[l], ob[l]);
          rep2 = rep1;
          rep1 = rep0;
          rep0 = jo;
        }
      }
      off[l] = jo;
    }
    for (int l = 0; l < 32; ++l) {
      if (!active[l]) continue;
      uint32_t idx = g * 32 + l;
      if (ml[l] == 0 && idx != n_seq - 1) return false;
      if (ml[l] > chunk_size) return false; // a match can never be longer than the chunk itself
      seqs[idx] = Seq{len_value(llc[l], llb[l]), ml[l], ml[l] ? off[l] : 0};
    }
    if (d.bad) return false;
  }
  // Literals: lane l owns the run [l*L, (l+1)*L) and codes each literal in
  // the context of the previous one in its run (the first in context 0);
  // every step visits lanes 0..31 in order, as the GPU's ballot does.
  uint32_t run_len = lit_run_len(n_lit);
  uint32_t prev[32] = {};
  for (uint32_t k = 0; k < run_len; ++k) {
    for (int l = 0; l < 32; ++l) {
      uint32_t idx = (uint32_t)l * run_len + k;
      if (idx >= n_lit) continue;
      uint32_t s = d.get_lit(l, lit_ctx(prev[l], lit_shift));
      lits[idx] = (uint8_t)s;
      prev[l] = s;
    }
  }
  if (d.bad || d.rp != in_len) return false;

  size_t op = 0, lp = 0;
  for (const Seq& s : seqs) {
    if (lp + s.lit_len > n_lit || op + s.lit_len > orig) return false;
    for (uint32_t k = 0; k < s.lit_len; ++k) out[op + k] = lits[lp + k];
    op += s.lit_len;
    lp += s.lit_len;
    if (s.ml) {
      if (s.off == 0 || s.off > op || op + s.ml > orig) return false;
      for (uint32_t k = 0; k < s.ml; ++k) out[op + k] = out[op - s.off + k];
      op += s.ml;
    }
  }
  return op == orig && lp == n_lit;
}

bool read_ext(const uint8_t* in, size_t in_len, size_t& ip, uint32_t& v) {
  uint8_t b;
  do {
    if (ip >= in_len) return false;
    b = in[ip++];
    v += b;
  } while (b == 255);
  return true;
}

bool decode_lz(const uint8_t* in, size_t in_len, uint8_t* out, size_t orig) {
  size_t ip = 0, op = 0;
  while (op < orig) {
    if (ip >= in_len) return false;
    uint8_t tok = in[ip++];
    uint32_t lit = tok >> 4;
    if (lit == 15 && !read_ext(in, in_len, ip, lit)) return false;
    if (ip + lit > in_len || op + lit > orig) return false;
    for (uint32_t k = 0; k < lit; ++k) out[op + k] = in[ip + k];
    ip += lit;
    op += lit;
    if (op >= orig) break;
    if (ip + 4 > in_len) return false;
    uint32_t off = (uint32_t)in[ip] | ((uint32_t)in[ip + 1] << 8) | ((uint32_t)in[ip + 2] << 16) |
                   ((uint32_t)in[ip + 3] << 24);
    ip += 4;
    uint32_t ml = (tok & 15) + kMinMatch;
    if ((tok & 15) == 15) {
      uint32_t e = 15;
      if (!read_ext(in, in_len, ip, e)) return false;
      ml = e + kMinMatch;
    }
    if (off == 0 || off > op || op + ml > orig) return false;
    for (uint32_t k = 0; k < ml; ++k) out[op + k] = out[op - off + k];
    op += ml;
  }
  return ip == in_len;
}

} // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "--version") == 0) {
    std::printf("gpusqz_refdec %s\n", GPUSQZ_VERSION_STRING);
    return 0;
  }
  if (argc != 3) fail("usage: gpusqz_refdec <input.gsz> <output> | --version");
  FILE* in = std::fopen(argv[1], "rb");
  if (!in) fail("cannot open input");
  FILE* out = std::fopen(argv[2], "wb");
  if (!out) fail("cannot open output");

  FileHeader h;
  if (std::fread(&h, sizeof(h), 1, in) != 1) fail("truncated header");
  if (h.magic != kMagic) fail("bad magic");
  if (h.version != kVersion) fail("unsupported version");
  if (h.chunk_size == 0 || h.chunk_size > kMaxChunkSize) fail("bad chunk_size");

  std::vector<ChunkEntry> entries(h.chunk_count);
  if (h.chunk_count && std::fread(entries.data(), sizeof(ChunkEntry), h.chunk_count, in) != h.chunk_count) {
    fail("truncated chunk table");
  }

  std::vector<TableGroup> groups(h.table_group_count);
  if (h.table_group_count &&
      std::fread(groups.data(), sizeof(TableGroup), h.table_group_count, in) != h.table_group_count) {
    fail("truncated table group directory");
  }
  uint64_t payload_start = file_tell(in);
  // The table section: each group's quantised counts, in group order, at
  // tables_offset (after the payload). Each group has its own
  // literal-context rule and so its own size.
  std::vector<Tables> tables;
  tables.reserve(h.table_group_count);
  if (!file_seek(in, h.tables_offset)) fail("bad tables_offset");
  for (const TableGroup& g : groups) {
    if (!lit_shift_valid(g.lit_ctx_shift)) fail("bad lit_ctx_shift");
    std::vector<uint8_t> q(group_quant_bytes(g));
    if (std::fread(q.data(), 1, q.size(), in) != q.size()) fail("truncated table section");
    tables.emplace_back(q.data(), lit_ctx_count(g.lit_ctx_shift));
  }
  // Every chunk's group, found independently of any GPU-side batching:
  // this is exactly the directory-coverage logic the file format promises,
  // read straight off disk with no runtime batch size involved at all.
  std::vector<uint32_t> chunk_group(h.chunk_count, 0);
  if (!groups.empty()) {
    uint64_t covered = 0;
    for (uint32_t g = 0; g < groups.size(); ++g) {
      const TableGroup& tg = groups[g];
      if (tg.start_chunk != covered || tg.chunk_count == 0) fail("corrupt table group directory: not contiguous");
      for (uint32_t c = tg.start_chunk; c < tg.start_chunk + tg.chunk_count; ++c) chunk_group[c] = g;
      covered += tg.chunk_count;
    }
    if (covered != h.chunk_count) fail("corrupt table group directory: doesn't cover all chunks");
  }

  std::vector<uint8_t> cbuf, obuf(h.chunk_size);
  uint64_t produced = 0;
  for (uint32_t c = 0; c < h.chunk_count; ++c) {
    const ChunkEntry& e = entries[c];
    if (e.original_size > h.chunk_size || e.compressed_size < 1) fail("bad chunk entry");
    cbuf.resize(e.compressed_size);
    if (!file_seek(in, payload_start + e.offset)) fail("seek failed");
    if (std::fread(cbuf.data(), 1, e.compressed_size, in) != e.compressed_size) fail("short payload read");

    bool ok = false;
    switch ((ChunkFlag)cbuf[0]) {
      case ChunkFlag::Raw:
        ok = e.compressed_size == 1 + e.original_size;
        if (ok) std::copy(cbuf.begin() + 1, cbuf.end(), obuf.begin());
        break;
      case ChunkFlag::Lz:
        ok = decode_lz(cbuf.data() + 1, e.compressed_size - 1, obuf.data(), e.original_size);
        break;
      case ChunkFlag::LzRans:
        ok = decode_lzrans(cbuf.data() + 1, e.compressed_size - 1, obuf.data(), e.original_size, h.chunk_size,
                           tables[chunk_group[c]], groups[chunk_group[c]].lit_ctx_shift);
        break;
      default:
        break;
    }
    if (!ok) fail("malformed chunk " + std::to_string(c));
    if (std::fwrite(obuf.data(), 1, e.original_size, out) != e.original_size) fail("write failed");
    produced += e.original_size;
  }
  if (produced != h.original_size) fail("size mismatch");
  std::fclose(in);
  if (std::fclose(out) != 0) fail("write failed");
  return 0;
}
