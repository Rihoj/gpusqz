// A small lpaq-style context-mixing model, used only to *measure* how many
// bits a predictive model would spend. It returns ideal code lengths; it
// codes nothing. Probabilities are clamped to 12 bits, as gpusqz's rANS
// would need, but the mixer is floating point: fine for a measurement, not
// for a format (both GPU backends would have to agree bit for bit).
//
// Per bit: one adaptive probability per context order (order 0 and 1
// direct-indexed, higher orders hashed, no collision check), an optional
// word context, and an optional match input fed by the caller, mixed by a
// logistic mixer whose weight set is picked by match state and the partial
// byte.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace cm {

struct Config {
  std::vector<int> orders{0, 1, 2, 3, 4, 6};
  bool word = false;     // hash of the current alphabetic word (text)
  int bits = 22;         // log2 slots per hashed order
  int limit = 60;        // adaptation count cap: rate is 1/(n+1.5)
  double lr = 0.01;      // mixer learning rate
};

inline std::vector<int> parse_orders(const std::string& s) {
  std::vector<int> v;
  size_t i = 0;
  while (i < s.size()) {
    size_t j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    v.push_back(std::stoi(s.substr(i, j - i)));
    i = j + 1;
  }
  return v;
}

class Model {
 public:
  explicit Model(const Config& c) : cfg_(c) {
    n_in_ = (int)cfg_.orders.size() + (cfg_.word ? 1 : 0) + 2; // + match + bias
    for (int o : cfg_.orders) {
      int b = o == 0 ? 8 : o == 1 ? 16 : cfg_.bits;
      tables_.emplace_back((size_t)1 << b);
      tbits_.push_back(b);
    }
    if (cfg_.word) {
      tables_.emplace_back((size_t)1 << cfg_.bits);
      tbits_.push_back(cfg_.bits);
    }
    match_sm_.resize(64 * 256);
    weights_.resize((size_t)kSel * n_in_);
    for (int i = 0; i < 4096; ++i) {
      double p = (i + 0.5) / 4096.0;
      stretch_[i] = std::log(p / (1 - p));
    }
    for (int n = 0; n < 1024; ++n) rate_[n] = 1.0 / (n + 1.5);
    reset();
  }

  void reset() {
    uint32_t init = (uint32_t)1 << 31; // p = 1/2, n = 0
    for (auto& t : tables_) std::fill(t.begin(), t.end(), init);
    std::fill(match_sm_.begin(), match_sm_.end(), init);
    std::fill(weights_.begin(), weights_.end(), 0.3);
  }

  // Ideal code length of `byte`, given its history hist[0..n) (most recent
  // last) and, if mbyte >= 0, a predicted byte from a match of length mlen.
  // Then learns from it.
  double code(uint8_t byte, const uint8_t* hist, size_t n, int mbyte, uint32_t mlen) {
    size_t n_ctx = tables_.size();
    uint64_t h[16];
    for (size_t i = 0; i < cfg_.orders.size(); ++i) {
      int o = cfg_.orders[i];
      if (o == 0) {
        h[i] = 0;
      } else if (o == 1) {
        h[i] = n ? hist[n - 1] : 0;
      } else {
        uint64_t x = 0x9E3779B97F4A7C15ull * (uint64_t)(o + 1);
        size_t k = (size_t)o < n ? (size_t)o : n;
        for (size_t j = 0; j < k; ++j) x = (x + hist[n - 1 - j] + 1) * 0x100000001B3ull;
        h[i] = x ^ (k << 56);
      }
    }
    if (cfg_.word) {
      uint64_t x = 0xCBF29CE484222325ull;
      for (size_t j = n; j > 0 && n - j < 32; --j) {
        uint8_t ch = hist[j - 1];
        if (ch >= 'A' && ch <= 'Z') ch = (uint8_t)(ch + 32);
        if (!(ch >= 'a' && ch <= 'z')) break;
        x = (x ^ ch) * 0x100000001B3ull;
      }
      h[cfg_.orders.size()] = x;
    }
    double bits = 0, x_in[18];
    size_t idx[16];
    uint32_t c0 = 1;
    int mbucket = mbyte < 0 ? 0 : mlen < 16 ? 1 : mlen < 32 ? 2 : 3;
    for (int b = 7; b >= 0; --b) {
      int y = (byte >> b) & 1;
      for (size_t i = 0; i < n_ctx; ++i) {
        if (i < cfg_.orders.size() && cfg_.orders[i] <= 1) {
          idx[i] = (size_t)((h[i] << 8) | c0) & (((size_t)1 << tbits_[i]) - 1);
        } else {
          idx[i] = (size_t)(((h[i] + c0 * 0xD6E8FEB86659FD93ull) * 0x9E3779B97F4A7C15ull) >> (64 - tbits_[i]));
        }
        x_in[i] = stretch_[tables_[i][idx[i]] >> 20];
      }
      int ebit = -1;
      size_t mslot = 0;
      if (mbyte >= 0 && (uint32_t)((mbyte | 256) >> (b + 1)) == c0) {
        ebit = (mbyte >> b) & 1;
        mslot = (size_t)(mlen < 63 ? mlen : 63) * 256 + (ebit ? 128 : 0) + (size_t)(7 - b);
        x_in[n_ctx] = stretch_[match_sm_[mslot] >> 20];
      } else {
        x_in[n_ctx] = 0;
      }
      x_in[n_ctx + 1] = 0.25;
      int sel = (ebit < 0 ? 0 : mbucket) * 256 + (int)c0;
      double* w = &weights_[(size_t)sel * n_in_];
      double dot = 0;
      for (int i = 0; i < n_in_; ++i) dot += w[i] * x_in[i];
      if (dot > 30) dot = 30;
      if (dot < -30) dot = -30;
      double p = 1.0 / (1.0 + std::exp(-dot));
      // rANS with 12-bit probabilities can't code a bit more cheaply than this.
      double pq = p < 1.0 / 4096 ? 1.0 / 4096 : p > 4095.0 / 4096 ? 4095.0 / 4096 : p;
      bits -= std::log2(y ? pq : 1 - pq);
      double err = (double)y - p;
      for (int i = 0; i < n_in_; ++i) w[i] += cfg_.lr * err * x_in[i];
      for (size_t i = 0; i < n_ctx; ++i) update(tables_[i][idx[i]], y);
      if (ebit >= 0) update(match_sm_[mslot], y);
      c0 = c0 * 2 + (uint32_t)y;
    }
    return bits;
  }

 private:
  static constexpr int kSel = 4 * 256;

  // Slot: probability in the top 22 bits, count in the low 10.
  void update(uint32_t& s, int y) {
    uint32_t n = s & 1023;
    double p = (double)(s >> 10);
    double target = y ? (double)((1u << 22) - 1) : 0.0;
    p += (target - p) * rate_[n];
    if (n < (uint32_t)cfg_.limit) ++n;
    s = ((uint32_t)p << 10) | n;
  }

  Config cfg_;
  int n_in_;
  std::vector<std::vector<uint32_t>> tables_;
  std::vector<int> tbits_;
  std::vector<uint32_t> match_sm_;
  std::vector<double> weights_;
  double stretch_[4096];
  double rate_[1024];
};

} // namespace cm
