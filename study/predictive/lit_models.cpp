// Literal-stream models: how many bits other literal models would spend on
// the literals a .gsz file's LZ parse left, read from bit_account's dump.
//
//   lit_models <literals.dump> <original file> <chunk_size> <model> [options]
//
// Static models, one table per context per table group, priced as gpusqz
// prices them (8-bit quantised counts, 12-bit rANS frequencies) plus the
// coded table bytes (table_codec.h):
//   lits1   previous literal in the lane's run, the group's own context rule
//           (calibration: must match bit_account's literal bytes)
//   lits2   previous literal and the top 2 bits of the one before, in the
//           run, 1024 contexts (decodable by today's two-phase decoder)
//   lits2b  the same with the top 4 bits, 4096 contexts
//   text1   the byte before the literal in the output, 256 contexts
//   text2   that byte and the top 2 bits of the one before, 1024 contexts
//   text2b  that byte and the top 4 bits of the one before, 4096 contexts
// Adaptive models (cm.h), no tables:
//   lane    reset at every lane run
//   chunk   reset every chunk; the 32 runs step in lockstep with one model
//   options: --src lits|text (history: the run's literals, or the output
//            before the literal), --orders, --bits, --word
// "text" contexts need the decoder to have rebuilt the output up to each
// literal, which gpusqz's two-phase decode doesn't do: it's an upper bound
// on what that restructuring would buy.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cm.h"
#include "rans_codes.h"
#include "table_codec.h"

using namespace gpusqz;

namespace {

struct Chunk {
  uint32_t idx, group, shift, n_lit, orig;
  std::vector<uint8_t> lits;
  std::vector<uint32_t> pos;
};

[[noreturn]] void fail(const std::string& msg) {
  std::fprintf(stderr, "lit_models: %s\n", msg.c_str());
  std::exit(1);
}

std::vector<uint8_t> read_file(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) fail(std::string("cannot open ") + path);
  std::vector<uint8_t> v;
  uint8_t buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + n);
  std::fclose(f);
  return v;
}

std::vector<Chunk> read_dump(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) fail(std::string("cannot open ") + path);
  std::vector<Chunk> out;
  uint32_t hdr[5];
  while (std::fread(hdr, 4, 5, f) == 5) {
    Chunk c{hdr[0], hdr[1], hdr[2], hdr[3], hdr[4], {}, {}};
    c.lits.resize(c.n_lit);
    c.pos.resize(c.n_lit);
    if (std::fread(c.lits.data(), 1, c.n_lit, f) != c.n_lit || std::fread(c.pos.data(), 4, c.n_lit, f) != c.n_lit) {
      fail("truncated dump");
    }
    out.push_back(std::move(c));
  }
  std::fclose(f);
  return out;
}

// Context of literal j of chunk c under a static model.
uint32_t static_ctx(const std::string& model, const Chunk& c, uint32_t j, const uint8_t* text, uint32_t run_len) {
  if (model == "lits1") {
    uint32_t prev = j % run_len ? c.lits[j - 1] : 0;
    return lit_ctx(prev, c.shift);
  }
  if (model == "lits2" || model == "lits2b") {
    uint32_t k = j % run_len;
    uint32_t b1 = k >= 1 ? c.lits[j - 1] : 0, b2 = k >= 2 ? c.lits[j - 2] : 0;
    return b1 | (b2 >> (model == "lits2" ? 6 : 4)) << 8;
  }
  uint32_t p = c.pos[j];
  uint32_t b1 = p >= 1 ? text[p - 1] : 0, b2 = p >= 2 ? text[p - 2] : 0;
  if (model == "text1") return b1;
  if (model == "text2") return b1 | (b2 >> 6) << 8;
  return b1 | (b2 >> 4) << 8; // text2b
}

int static_nctx(const std::string& model, uint32_t shift) {
  if (model == "lits1") return lit_ctx_count(shift);
  if (model == "text1") return 256;
  if (model == "text2" || model == "lits2") return 1024;
  return 4096;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 5) fail("usage: lit_models <literals.dump> <original file> <chunk_size> <model> [options]");
  std::vector<Chunk> chunks = read_dump(argv[1]);
  std::vector<uint8_t> data = read_file(argv[2]);
  uint64_t chunk_size = std::stoull(argv[3]);
  std::string model = argv[4], src = "text";
  cm::Config cfg;
  cfg.orders = {0, 1, 2, 3};
  cfg.bits = 20;
  for (int i = 5; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--src" && i + 1 < argc) src = argv[++i];
    else if (a == "--orders" && i + 1 < argc) cfg.orders = cm::parse_orders(argv[++i]);
    else if (a == "--bits" && i + 1 < argc) cfg.bits = std::stoi(argv[++i]);
    else if (a == "--word") cfg.word = true;
    else fail("unknown option " + a);
  }

  uint64_t n_lits = 0;
  for (const Chunk& c : chunks) n_lits += c.n_lit;
  double bits = 0, table_bytes = 0;

  if (model == "lits1" || model == "lits2" || model == "lits2b" || model == "text1" || model == "text2" ||
      model == "text2b") {
    size_t g0 = 0;
    while (g0 < chunks.size()) {
      size_t g1 = g0;
      while (g1 < chunks.size() && chunks[g1].group == chunks[g0].group) ++g1;
      int nctx = static_nctx(model, chunks[g0].shift);
      std::vector<uint32_t> counts((size_t)nctx * 256, 0);
      for (size_t ci = g0; ci < g1; ++ci) {
        const Chunk& c = chunks[ci];
        const uint8_t* text = data.data() + (uint64_t)c.idx * chunk_size;
        uint32_t run_len = lit_run_len(c.n_lit);
        for (uint32_t j = 0; j < c.n_lit; ++j) counts[(size_t)static_ctx(model, c, j, text, run_len) * 256 + c.lits[j]]++;
      }
      std::vector<uint8_t> q((size_t)kLitBase + (size_t)nctx * 256, 0);
      std::vector<uint16_t> freq((size_t)nctx * 256), cum((size_t)nctx * 256);
      for (int x = 0; x < nctx; ++x) {
        quantize_counts(&counts[(size_t)x * 256], 256, &q[kLitBase + (size_t)x * 256]);
        normalize_table(&q[kLitBase + (size_t)x * 256], 256, &freq[(size_t)x * 256], &cum[(size_t)x * 256]);
      }
      for (size_t i = 0; i < counts.size(); ++i) {
        if (counts[i]) bits += counts[i] * std::log2((double)kProbScale / freq[i]);
      }
      table_bytes += (double)encode_table_counts(q.data(), q.size()).size();
      g0 = g1;
    }
  } else if (model == "lane" || model == "chunk") {
    cm::Model m(cfg);
    for (const Chunk& c : chunks) {
      const uint8_t* text = data.data() + (uint64_t)c.idx * chunk_size;
      uint32_t run_len = lit_run_len(c.n_lit);
      auto code_one = [&](uint32_t l, uint32_t k) {
        uint32_t j = l * run_len + k;
        if (src == "lits") return m.code(c.lits[j], c.lits.data() + l * run_len, k, -1, 0);
        return m.code(c.lits[j], text, c.pos[j], -1, 0);
      };
      if (model == "lane") {
        for (uint32_t l = 0; l < 32; ++l) {
          m.reset();
          for (uint32_t k = 0; k < run_len && l * run_len + k < c.n_lit; ++k) bits += code_one(l, k);
        }
      } else {
        m.reset();
        for (uint32_t k = 0; k < run_len; ++k) {
          for (uint32_t l = 0; l < 32; ++l) {
            if (l * run_len + k < c.n_lit) bits += code_one(l, k);
          }
        }
      }
    }
  } else {
    fail("unknown model " + model);
  }

  std::string desc = model;
  if (model == "lane" || model == "chunk") {
    desc += " src=" + src + " orders=";
    for (size_t i = 0; i < cfg.orders.size(); ++i) desc += (i ? "," : "") + std::to_string(cfg.orders[i]);
    desc += cfg.word ? " +word" : "";
    desc += " bits=" + std::to_string(cfg.bits);
  }
  std::printf("%s %s: %llu literals -> %.0f bytes + %.0f table bytes = %.0f (%.3f bits/literal)\n", argv[1],
              desc.c_str(), (unsigned long long)n_lits, bits / 8, table_bytes, bits / 8 + table_bytes,
              n_lits ? (bits + 8 * table_bytes) / n_lits : 0.0);
  return 0;
}
