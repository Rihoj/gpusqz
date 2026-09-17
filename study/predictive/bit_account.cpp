// Where a .gsz file's bits go. A fork of tests/ref_decode.cpp that decodes
// every chunk exactly as the reference decoder does, and prices each rANS
// symbol at its exact code length, log2(kProbScale / freq), and each bypass
// bit at 1 bit. Totals are reported per stream class and checked against
// the file's real size.
//
// Optionally dumps every rANS chunk's literal stream, with each literal's
// offset in the chunk's output, for lit_models.cpp:
//   per chunk: u32 chunk, u32 group, u32 lit_shift, u32 n_lit, u32 orig,
//              u8 lits[n_lit], u32 pos[n_lit]
//
//   bit_account <input.gsz> [literals.dump]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "file_io.h"
#include "format.h"
#include "rans_codes.h"
#include "table_codec.h"

using namespace gpusqz;

namespace {

[[noreturn]] void fail(const std::string& msg) {
  std::fprintf(stderr, "bit_account: %s\n", msg.c_str());
  std::exit(1);
}

uint32_t load_u32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

enum Cls { kLlCode, kLlBits, kMlCode, kMlBits, kOffCode, kOffBits, kLit, kNumCls };
const char* kClsName[kNumCls] = {"lit_len codes", "lit_len extra bits", "match_len codes", "match_len extra bits",
                                 "offset codes", "offset extra bits", "literals"};
double g_bits[kNumCls];
uint64_t g_syms[kNumCls];
uint64_t g_rep_offsets, g_matches, g_match_bytes, g_lit_bytes;

struct Seq {
  uint32_t lit_len, ml, off;
};

struct Tables {
  int n_ctx = 1;
  std::vector<uint16_t> freq, cum;
  std::vector<uint8_t> sym;

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

struct RansDecoder {
  const uint8_t* p;
  size_t len, rp;
  uint32_t x[32];
  const Tables& t;
  bool bad = false;
  int cls = kLlCode; // class charged by the next symbol

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
  uint32_t get(int l, int base, int k) {
    uint32_t slot = x[l] & (kProbScale - 1);
    uint32_t s = 0;
    for (int i = 0; i < k; ++i) {
      if (t.cum[base + i] <= slot) s = i;
    }
    step(l, base + (int)s, slot);
    return s;
  }
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
    if (t.freq[e] == 0) {
      bad = true;
      return;
    }
    g_bits[cls] += std::log2((double)kProbScale / t.freq[e]);
    g_syms[cls]++;
    x[l] = t.freq[e] * (x[l] >> kProbBits) + slot - t.cum[e];
    renorm(l);
  }
  uint32_t bits16(int l, bool on, uint32_t nb) {
    on = on && nb != 0;
    uint32_t b = on ? (x[l] & ((1u << nb) - 1)) : 0;
    if (on) {
      x[l] >>= nb;
      g_bits[cls] += nb;
    }
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
                   const Tables& tables, uint32_t lit_shift, std::vector<uint8_t>& lits_out,
                   std::vector<uint32_t>& pos_out) {
  if (in_len < (size_t)kRansHeaderBytes) return false;
  uint32_t n_seq = load_u32(in), n_lit = load_u32(in + 4);
  if (n_seq > chunk_size / kMinMatch + 1 || n_lit > chunk_size) return false;
  RansDecoder d(in, in_len, tables);
  std::vector<Seq> seqs(n_seq);
  std::vector<uint8_t> lits(n_lit);
  uint32_t llc[32], mlc[32], oc[32], llb[32], mlb[32], ob[32], ml[32], nb[32], off[32];
  bool active[32], has_off[32];
  uint32_t rep0 = 0, rep1 = 0, rep2 = 0;

  for (uint32_t g = 0; g < (n_seq + 31) / 32; ++g) {
    for (int l = 0; l < 32; ++l) active[l] = g * 32 + l < n_seq;
    d.cls = kLlCode;
    for (int l = 0; l < 32; ++l) if (active[l]) llc[l] = d.get(l, kLlBase, kSmallSyms);
    for (int l = 0; l < 32; ++l) nb[l] = active[l] ? len_nb(llc[l]) : 0;
    d.cls = kLlBits;
    bits_pass(d, active, nb, llb);
    d.cls = kMlCode;
    for (int l = 0; l < 32; ++l) if (active[l]) mlc[l] = d.get(l, kMlBase, kSmallSyms);
    for (int l = 0; l < 32; ++l) nb[l] = active[l] ? len_nb(mlc[l]) : 0;
    d.cls = kMlBits;
    bits_pass(d, active, nb, mlb);
    for (int l = 0; l < 32; ++l) {
      if (!active[l]) continue;
      uint32_t v = len_value(mlc[l], mlb[l]);
      ml[l] = v ? v + (kMinMatch - 1) : 0;
    }
    for (int l = 0; l < 32; ++l) has_off[l] = active[l] && ml[l] != 0;
    d.cls = kOffCode;
    for (int l = 0; l < 32; ++l) if (has_off[l]) oc[l] = d.get(l, kOffBase, kSmallSyms);
    for (int l = 0; l < 32; ++l) nb[l] = has_off[l] ? off_nb(oc[l]) : 0;
    d.cls = kOffBits;
    bits_pass(d, has_off, nb, ob);
    for (int l = 0; l < 32; ++l) {
      uint32_t jo = 0;
      if (has_off[l]) {
        if (oc[l] >= kOffRepBase) {
          g_rep_offsets++;
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
      if (ml[l] > chunk_size) return false;
      seqs[idx] = Seq{len_value(llc[l], llb[l]), ml[l], ml[l] ? off[l] : 0};
    }
    if (d.bad) return false;
  }
  d.cls = kLit;
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

  lits_out.assign(lits.begin(), lits.end());
  pos_out.resize(n_lit);
  size_t op = 0, lp = 0;
  for (const Seq& s : seqs) {
    if (lp + s.lit_len > n_lit || op + s.lit_len > orig) return false;
    for (uint32_t k = 0; k < s.lit_len; ++k) {
      out[op + k] = lits[lp + k];
      pos_out[lp + k] = (uint32_t)(op + k);
    }
    op += s.lit_len;
    lp += s.lit_len;
    g_lit_bytes += s.lit_len;
    if (s.ml) {
      if (s.off == 0 || s.off > op || op + s.ml > orig) return false;
      for (uint32_t k = 0; k < s.ml; ++k) out[op + k] = out[op - s.off + k];
      op += s.ml;
      g_matches++;
      g_match_bytes += s.ml;
    }
  }
  return op == orig && lp == n_lit;
}

void put_u32(FILE* f, uint32_t v) { std::fwrite(&v, 4, 1, f); }

} // namespace

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) fail("usage: bit_account <input.gsz> [literals.dump]");
  FILE* in = std::fopen(argv[1], "rb");
  if (!in) fail("cannot open input");
  FILE* dump = argc == 3 ? std::fopen(argv[2], "wb") : nullptr;
  if (argc == 3 && !dump) fail("cannot open dump");

  FileHeader h;
  if (std::fread(&h, sizeof(h), 1, in) != 1) fail("truncated header");
  if (h.magic != kMagic || h.version != kVersion) fail("bad magic or version");
  if (!file_seek(in, 0, SEEK_END)) fail("seek failed");
  uint64_t file_bytes = file_tell(in);
  if (!file_seek(in, sizeof(h))) fail("seek failed");

  std::vector<uint32_t> sizes(h.chunk_count);
  if (h.chunk_count && std::fread(sizes.data(), 4, h.chunk_count, in) != h.chunk_count) fail("truncated chunk table");
  std::vector<ChunkEntry> entries(h.chunk_count);
  if (!chunk_entries(sizes.data(), h.chunk_count, h.chunk_size, h.original_size, entries.data())) {
    fail("chunk table doesn't match the header");
  }
  std::vector<TableGroup> groups(h.table_group_count);
  if (h.table_group_count &&
      std::fread(groups.data(), sizeof(TableGroup), h.table_group_count, in) != h.table_group_count) {
    fail("truncated table group directory");
  }
  uint64_t payload_start = file_tell(in);
  std::vector<Tables> tables;
  if (!file_seek(in, h.tables_offset)) fail("bad tables_offset");
  uint64_t coded_total = 0;
  int shift_hist[9] = {};
  for (const TableGroup& g : groups) {
    std::vector<uint8_t> coded(g.table_bytes), q(group_quant_bytes(g));
    if (std::fread(coded.data(), 1, coded.size(), in) != coded.size()) fail("truncated table section");
    if (!decode_table_counts(coded.data(), coded.size(), q.data(), q.size())) fail("corrupt table section");
    tables.emplace_back(q.data(), lit_ctx_count(g.lit_ctx_shift));
    coded_total += g.table_bytes;
    shift_hist[g.lit_ctx_shift]++;
  }
  std::vector<uint32_t> chunk_group(h.chunk_count, 0);
  for (uint32_t g = 0; g < groups.size(); ++g) {
    for (uint32_t c = groups[g].start_chunk; c < groups[g].start_chunk + groups[g].chunk_count; ++c) chunk_group[c] = g;
  }

  std::vector<uint8_t> cbuf, obuf(h.chunk_size), lits;
  std::vector<uint32_t> pos;
  uint64_t n_raw = 0, n_lz = 0, n_rans = 0, raw_bytes = 0, lz_bytes = 0, rans_payload_bytes = 0, rans_orig = 0;
  for (uint32_t c = 0; c < h.chunk_count; ++c) {
    const ChunkEntry& e = entries[c];
    cbuf.resize(e.compressed_size);
    if (!file_seek(in, payload_start + e.offset)) fail("seek failed");
    if (std::fread(cbuf.data(), 1, e.compressed_size, in) != e.compressed_size) fail("short payload read");
    switch ((ChunkFlag)cbuf[0]) {
      case ChunkFlag::Raw:
        n_raw++;
        raw_bytes += e.compressed_size;
        break;
      case ChunkFlag::Lz:
        n_lz++;
        lz_bytes += e.compressed_size;
        break;
      case ChunkFlag::LzRans: {
        uint32_t g = chunk_group[c];
        if (!decode_lzrans(cbuf.data() + 1, e.compressed_size - 1, obuf.data(), e.original_size, h.chunk_size,
                           tables[g], groups[g].lit_ctx_shift, lits, pos)) {
          fail("malformed chunk " + std::to_string(c));
        }
        n_rans++;
        rans_payload_bytes += e.compressed_size - 1;
        rans_orig += e.original_size;
        if (dump) {
          put_u32(dump, c);
          put_u32(dump, g);
          put_u32(dump, groups[g].lit_ctx_shift);
          put_u32(dump, (uint32_t)lits.size());
          put_u32(dump, e.original_size);
          std::fwrite(lits.data(), 1, lits.size(), dump);
          std::fwrite(pos.data(), 4, pos.size(), dump);
        }
        break;
      }
      default:
        fail("bad chunk flag");
    }
  }
  if (dump) std::fclose(dump);

  double sym_bits = 0;
  for (int i = 0; i < kNumCls; ++i) sym_bits += g_bits[i];
  uint64_t dir_bytes = sizeof(FileHeader) + 4ull * h.chunk_count + sizeof(TableGroup) * (uint64_t)groups.size();
  // Everything in a rANS payload that isn't a symbol's code length: the
  // n_seq/n_lit header, the 32 initial states, and the final-state slack.
  double rans_framing = (double)rans_payload_bytes - sym_bits / 8.0;
  auto pct = [&](double bytes) { return 100.0 * bytes / (double)file_bytes; };
  std::printf("%s: %llu bytes, original %llu, ratio %.4f, chunk %u\n", argv[1], (unsigned long long)file_bytes,
              (unsigned long long)h.original_size, (double)file_bytes / (double)h.original_size, h.chunk_size);
  std::printf("chunks: %llu rANS, %llu LZ-token, %llu raw; groups %zu (literal ctx shift 0/4/8: %d/%d/%d)\n",
              (unsigned long long)n_rans, (unsigned long long)n_lz, (unsigned long long)n_raw, groups.size(),
              shift_hist[0], shift_hist[4], shift_hist[8]);
  std::printf("rANS chunks: %llu bytes = %llu literals (%.1f%%) + %llu match bytes in %llu matches "
              "(avg %.1f, %.1f%% repeat offsets)\n",
              (unsigned long long)rans_orig, (unsigned long long)g_lit_bytes, 100.0 * g_lit_bytes / rans_orig,
              (unsigned long long)g_match_bytes, (unsigned long long)g_matches,
              g_matches ? (double)g_match_bytes / g_matches : 0.0, 100.0 * g_rep_offsets / (g_matches ? g_matches : 1));
  std::printf("  %-22s %12s %12s %8s %9s\n", "class", "bytes", "symbols", "of file", "bits/sym");
  for (int i = 0; i < kNumCls; ++i) {
    std::printf("  %-22s %12.0f %12llu %7.2f%% %9.3f\n", kClsName[i], g_bits[i] / 8, (unsigned long long)g_syms[i],
                pct(g_bits[i] / 8), g_syms[i] ? g_bits[i] / g_syms[i] : 0.0);
  }
  std::printf("  %-22s %12.0f %12s %7.2f%%  (%.1f bytes/chunk; state header is %d)\n", "rANS framing", rans_framing,
              "", pct(rans_framing), n_rans ? rans_framing / n_rans : 0.0, kRansHeaderBytes);
  std::printf("  %-22s %12llu %12s %7.2f%%\n", "chunk flags", (unsigned long long)n_rans, "", pct((double)n_rans));
  std::printf("  %-22s %12llu %12s %7.2f%%\n", "tables", (unsigned long long)coded_total, "", pct((double)coded_total));
  std::printf("  %-22s %12llu %12s %7.2f%%\n", "header+directories", (unsigned long long)dir_bytes, "",
              pct((double)dir_bytes));
  std::printf("  %-22s %12llu %12s %7.2f%%\n", "raw/LZ-token chunks", (unsigned long long)(raw_bytes + lz_bytes), "",
              pct((double)(raw_bytes + lz_bytes)));
  // Framing is the remainder, so the classes always sum to the file; the
  // real check is that framing per chunk comes out near the 136-byte state
  // header plus a few bytes of final-state slack.
  return 0;
}
