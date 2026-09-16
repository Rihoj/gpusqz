// "Drop LZ": what a context-mixing model would spend coding a file's raw
// bytes directly, under gpusqz's constraints.
//
//   cm_raw <file> [options]
//     --chunk N       reset the model every N bytes (0: never; default 65536)
//     --lanes L       1: one stream per chunk; 32: the chunk split into 32
//                     segments stepped in lockstep with one shared model, each
//                     segment's contexts from its own bytes and matches only
//                     into bytes already decoded (default 1)
//     --orders a,b,.. context orders (default 0,1,2,3,4,6)
//     --word          add a word context
//     --no-match      no match model
//     --bits B        log2 slots per hashed order (default 22)
//     --limit N       adaptation count cap (default 255)
//     --lr X          mixer learning rate (default 0.02)
//     --prior FILE    start every chunk from a model trained on FILE (up to
//                     --prior-bytes, default 16MB), instead of from nothing
//     --from A --to B only evaluate chunks starting in [A, B)
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "cm.h"

namespace {

std::vector<uint8_t> read_file(const std::string& path, uint64_t max = ~0ull) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "cm_raw: cannot open %s\n", path.c_str());
    std::exit(1);
  }
  std::vector<uint8_t> v;
  uint8_t buf[1 << 16];
  size_t n;
  while (v.size() < max && (n = std::fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + n);
  std::fclose(f);
  if (v.size() > max) v.resize(max);
  return v;
}

constexpr uint32_t kMinLen = 6;

uint32_t hash_at(const uint8_t* p) { // the kMinLen bytes ending just before p
  uint32_t h = 0;
  for (uint32_t k = kMinLen; k > 0; --k) h = (h + p[-(int)k] + 1) * 0x2F0F3A35u;
  return h;
}

// Codes buf[cs, ce) as one stream: contexts from the chunk's own history, a
// match model over the chunk so far.
double code_serial(cm::Model& m, const uint8_t* buf, size_t cs, size_t ce, bool use_match, std::vector<int32_t>& ht,
                   int hbits) {
  double bits = 0;
  size_t mptr = 0;
  uint32_t mlen = 0;
  for (size_t i = cs; i < ce; ++i) {
    if (use_match && i > cs) {
      if (mlen && buf[mptr] == buf[i - 1]) {
        ++mlen;
        ++mptr;
      } else {
        mlen = 0;
      }
      if (i - cs >= kMinLen) {
        uint32_t h = hash_at(buf + i) >> (32 - hbits);
        if (mlen == 0 && ht[h] > 0) {
          size_t p = (size_t)ht[h];
          uint32_t len = 0;
          while (len < 65535 && p - len > cs && buf[p - 1 - len] == buf[i - 1 - len]) ++len;
          if (len >= kMinLen) {
            mlen = len;
            mptr = p;
          }
        }
        ht[h] = (int32_t)i;
      }
    }
    int mbyte = use_match && mlen ? buf[mptr] : -1;
    bits += m.code(buf[i], buf + cs, i - cs, mbyte, mlen);
  }
  return bits;
}

// The same chunk as 32 segments in lockstep: step k codes byte k of every
// segment in lane order, so when lane l codes its byte, lanes < l have done
// k+1 bytes and lanes >= l have done k.
double code_lanes(cm::Model& m, const uint8_t* buf, size_t cs, size_t ce, bool use_match, std::vector<int32_t>& ht,
                  int hbits) {
  constexpr int L = 32;
  size_t len = ce - cs, seg = (len + L - 1) / L;
  size_t mptr[L] = {};
  uint32_t mlen[L] = {};
  double bits = 0;
  size_t cur_k = 0;
  int cur_l = 0;
  auto decoded = [&](size_t pos) {
    size_t s = (pos - cs) / seg, off = (pos - cs) % seg;
    return off < ((int)s < cur_l ? cur_k + 1 : cur_k);
  };
  for (size_t k = 0; k < seg; ++k) {
    for (int l = 0; l < L; ++l) {
      size_t ss = cs + (size_t)l * seg, i = ss + k;
      if (i >= ce) continue;
      cur_k = k;
      cur_l = l;
      if (use_match && k > 0) {
        if (mlen[l] && decoded(mptr[l]) && buf[mptr[l]] == buf[i - 1]) {
          ++mlen[l];
          ++mptr[l];
        } else {
          mlen[l] = 0;
        }
        if (k >= kMinLen) {
          uint32_t h = hash_at(buf + i) >> (32 - hbits);
          if (mlen[l] == 0 && ht[h] > 0) {
            size_t p = (size_t)ht[h];
            uint32_t n = 0;
            while (n < 65535 && p - n > cs && decoded(p - 1 - n) && buf[p - 1 - n] == buf[i - 1 - n]) ++n;
            if (n >= kMinLen) {
              mlen[l] = n;
              mptr[l] = p;
            }
          }
          ht[h] = (int32_t)i;
        }
      }
      int mbyte = use_match && mlen[l] && decoded(mptr[l]) ? buf[mptr[l]] : -1;
      bits += m.code(buf[i], buf + ss, k, mbyte, mlen[l]);
    }
  }
  return bits;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: cm_raw <file> [options], see the source\n");
    return 1;
  }
  std::string path = argv[1], prior_path;
  cm::Config cfg;
  uint64_t chunk = 65536, prior_bytes = 16u << 20, from = 0, to = ~0ull;
  int lanes = 1;
  bool use_match = true;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "cm_raw: %s needs a value\n", a.c_str());
        std::exit(1);
      }
      return argv[++i];
    };
    if (a == "--chunk") chunk = std::stoull(next());
    else if (a == "--lanes") lanes = std::stoi(next());
    else if (a == "--orders") cfg.orders = cm::parse_orders(next());
    else if (a == "--word") cfg.word = true;
    else if (a == "--no-match") use_match = false;
    else if (a == "--bits") cfg.bits = std::stoi(next());
    else if (a == "--limit") cfg.limit = std::stoi(next());
    else if (a == "--lr") cfg.lr = std::stod(next());
    else if (a == "--prior") prior_path = next();
    else if (a == "--prior-bytes") prior_bytes = std::stoull(next());
    else if (a == "--from") from = std::stoull(next());
    else if (a == "--to") to = std::stoull(next());
    else {
      std::fprintf(stderr, "cm_raw: unknown option %s\n", a.c_str());
      return 1;
    }
  }
  std::vector<uint8_t> data = read_file(path);
  if (chunk == 0) chunk = data.size();
  if (to > data.size()) to = data.size();
  int hbits = 22;

  auto t0 = std::chrono::steady_clock::now();
  cm::Model model(cfg);
  std::unique_ptr<cm::Model> prior;
  std::vector<int32_t> ht((size_t)1 << hbits);
  if (!prior_path.empty()) {
    std::vector<uint8_t> pd = read_file(prior_path, prior_bytes);
    prior = std::make_unique<cm::Model>(cfg);
    code_serial(*prior, pd.data(), 0, pd.size(), use_match, ht, hbits);
  }
  auto t1 = std::chrono::steady_clock::now();

  double bits = 0;
  uint64_t coded = 0;
  for (uint64_t cs = 0; cs < data.size(); cs += chunk) {
    if (cs < from || cs >= to) continue;
    size_t ce = cs + chunk < data.size() ? cs + chunk : data.size();
    if (prior) model = *prior;
    else model.reset();
    std::fill(ht.begin(), ht.end(), 0);
    bits += lanes == 32 ? code_lanes(model, data.data(), cs, ce, use_match, ht, hbits)
                        : code_serial(model, data.data(), cs, ce, use_match, ht, hbits);
    coded += ce - cs;
  }
  auto t2 = std::chrono::steady_clock::now();
  double secs = std::chrono::duration<double>(t2 - t1).count();
  std::string orders;
  for (int o : cfg.orders) orders += (orders.empty() ? "" : ",") + std::to_string(o);
  std::printf("%s chunk=%llu lanes=%d orders=%s%s%s bits=%d limit=%d lr=%g prior=%s: %llu bytes -> %.0f, "
              "ratio %.4f, %.3f bpc (%.2f MB/s, prior %.1fs)\n",
              path.c_str(), (unsigned long long)chunk, lanes, orders.c_str(), cfg.word ? " +word" : "",
              use_match ? " +match" : "", cfg.bits, cfg.limit, cfg.lr, prior_path.empty() ? "-" : prior_path.c_str(),
              (unsigned long long)coded, bits / 8, bits / 8 / coded, bits / coded, coded / 1e6 / secs,
              std::chrono::duration<double>(t1 - t0).count());
  return 0;
}
