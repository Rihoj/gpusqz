// Independent CPU decoder for the gzp container, used by the tests to
// check that GPU-compressed files decode correctly by an implementation
// that shares no code with the GPU path (a symmetric bug in the GPU
// encoder and decoder would still round-trip on the GPU alone).
//
//   gzp_refdec <input.gzp> <output>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>

#include "format.h"
#include "rans_codes.h"

using namespace gzp;

namespace {

[[noreturn]] void fail(const std::string& msg) {
  std::fprintf(stderr, "gzp_refdec: %s\n", msg.c_str());
  std::exit(1);
}

uint32_t load_u32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

struct Seq {
  uint32_t lit_len, ml, off;
};

// Serial emulation of the GPU's 32-lane interleaved rANS decoder. The GPU
// runs each sub-step for all lanes at once and lets the lanes that need a
// word take one in lane order, so visiting lanes 0..31 within each
// sub-step, reading a word whenever that lane needs one, consumes the
// stream identically.
struct RansDecoder {
  const uint8_t* p;
  size_t len, rp;
  uint32_t x[32];
  uint16_t freq[kQuantBytes], cum[kQuantBytes];
  std::vector<uint8_t> sym;
  bool bad = false;

  static constexpr int kLit = 0, kLl = kLitSyms, kMl = kLitSyms + kSmallSyms, kOff = kLitSyms + 2 * kSmallSyms;

  // `q` is this chunk's TableGroup's quantised bytes (shared by every
  // chunk in the group, not stored in the per-chunk payload anymore).
  RansDecoder(const uint8_t* payload, size_t n, const uint8_t* q) : p(payload), len(n), rp(kRansHeaderBytes),
      sym(kProbScale, 0) {
    normalize_table(q + kLit, kLitSyms, freq + kLit, cum + kLit);
    for (int base : {kLl, kMl, kOff}) normalize_table(q + base, kSmallSyms, freq + base, cum + base);
    for (int s = 0; s < kLitSyms; ++s) {
      for (uint32_t k = 0; k < freq[kLit + s]; ++k) sym[cum[kLit + s] + k] = (uint8_t)s;
    }
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
  uint32_t get(int l, int base, int k) {
    uint32_t slot = x[l] & (kProbScale - 1);
    uint32_t s;
    if (base == kLit) {
      s = sym[slot];
    } else {
      s = 0;
      for (int i = 0; i < k; ++i) {
        if (cum[base + i] <= slot) s = i;
      }
    }
    if (freq[base + s] == 0) bad = true;
    x[l] = freq[base + s] * (x[l] >> kProbBits) + slot - cum[base + s];
    renorm(l);
    return s;
  }
  uint32_t bits(int l, uint32_t nb) {
    if (nb == 0) return 0;
    uint32_t b = x[l] & ((1u << nb) - 1);
    x[l] >>= nb;
    renorm(l);
    return b;
  }
};

bool decode_lzrans(const uint8_t* in, size_t in_len, uint8_t* out, size_t orig, uint32_t chunk_size,
                   const uint8_t* group_q) {
  if (in_len < (size_t)kRansHeaderBytes) return false;
  uint32_t n_seq = load_u32(in), n_lit = load_u32(in + 4);
  if (n_seq > chunk_size / kMinMatch + 1 || n_lit > chunk_size) return false;
  RansDecoder d(in, in_len, group_q);
  std::vector<Seq> seqs(n_seq);
  std::vector<uint8_t> lits(n_lit);
  uint32_t llc[32], mlc[32], oc[32], llb[32], mlb[32], ob[32], ml[32];

  for (uint32_t g = 0; g < (n_seq + 31) / 32; ++g) {
    auto active = [&](int l) { return g * 32 + l < n_seq; };
    for (int l = 0; l < 32; ++l) if (active(l)) llc[l] = d.get(l, RansDecoder::kLl, kSmallSyms);
    for (int l = 0; l < 32; ++l) if (active(l)) llb[l] = d.bits(l, len_nb(llc[l]));
    for (int l = 0; l < 32; ++l) if (active(l)) mlc[l] = d.get(l, RansDecoder::kMl, kSmallSyms);
    for (int l = 0; l < 32; ++l) if (active(l)) mlb[l] = d.bits(l, len_nb(mlc[l]));
    for (int l = 0; l < 32; ++l) {
      if (!active(l)) continue;
      uint32_t v = len_value(mlc[l], mlb[l]);
      ml[l] = v ? v + (kMinMatch - 1) : 0;
    }
    for (int l = 0; l < 32; ++l) if (active(l) && ml[l]) oc[l] = d.get(l, RansDecoder::kOff, kSmallSyms);
    for (int l = 0; l < 32; ++l) if (active(l) && ml[l]) ob[l] = d.bits(l, oc[l]);
    for (int l = 0; l < 32; ++l) {
      if (!active(l)) continue;
      uint32_t idx = g * 32 + l;
      if (ml[l] == 0 && idx != n_seq - 1) return false;
      if (ml[l] && oc[l] > 15) return false;
      seqs[idx] = Seq{len_value(llc[l], llb[l]), ml[l], ml[l] ? off_value(oc[l], ob[l]) : 0};
    }
    if (d.bad) return false;
  }
  for (uint32_t g = 0; g < (n_lit + 31) / 32; ++g) {
    for (int l = 0; l < 32; ++l) {
      uint32_t idx = g * 32 + l;
      if (idx < n_lit) lits[idx] = (uint8_t)d.get(l, RansDecoder::kLit, kLitSyms);
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
    if (ip + 2 > in_len) return false;
    uint32_t off = (uint32_t)in[ip] | ((uint32_t)in[ip + 1] << 8);
    ip += 2;
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
  if (argc != 3) fail("usage: gzp_refdec <input.gzp> <output>");
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

  long payload_start = std::ftell(in);

  std::vector<uint8_t> cbuf, obuf(h.chunk_size);
  uint64_t produced = 0;
  for (uint32_t c = 0; c < h.chunk_count; ++c) {
    const ChunkEntry& e = entries[c];
    if (e.original_size > h.chunk_size || e.compressed_size < 1) fail("bad chunk entry");
    cbuf.resize(e.compressed_size);
    if (std::fseek(in, payload_start + (long)e.offset, SEEK_SET) != 0) fail("seek failed");
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
                           groups[chunk_group[c]].q);
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
