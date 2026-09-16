// Compact coding of each table group's quantised counts in the file's
// table section. Host-only plain C++, shared with the CPU reference decoder.
//
// The counts (rans_codes.h layout: three 32-entry small alphabets, then 256
// bytes per literal context) are mostly zeros and small values, and each
// literal row resembles the one before it. They are coded one byte at a
// time as 8 binary decisions (a bit tree, most significant bit first) with
// an adaptive binary range coder, LZMA's: 11-bit probabilities moving 1/16
// of the way per bit. The bit tree's context is the previous byte's size
// class, the size class of the same symbol in the previous literal row,
// and whether the byte belongs to a small alphabet. That codes the 1GB
// benchmark corpus's tables ~10x smaller (zstd -19 manages ~9x per group).
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rans_codes.h"

namespace gpusqz {

namespace table_codec_detail {

constexpr int kBitProbBits = 11;
constexpr uint32_t kProbOne = 1u << kBitProbBits;
constexpr int kMoveBits = 4;
constexpr uint32_t kTop = 1u << 24;
constexpr int kClasses = 5;
constexpr int kContexts = kClasses * kClasses * 2;

inline int size_class(uint32_t v) { return v == 0 ? 0 : v < 4 ? 1 : v < 16 ? 2 : v < 64 ? 3 : 4; }

inline int context(const uint8_t* q, size_t i) {
  uint32_t prev = i ? q[i - 1] : 0;
  uint32_t above = i >= (size_t)(kLitBase + kLitSyms) ? q[i - kLitSyms] : 0;
  return (size_class(prev) * kClasses + size_class(above)) * 2 + (i < (size_t)kLitBase ? 1 : 0);
}

// One bit-tree probability per (context, tree node), all starting at 1/2.
struct Model {
  std::vector<uint16_t> p = std::vector<uint16_t>((size_t)kContexts * 256, (uint16_t)(kProbOne / 2));
  uint16_t& at(int ctx, uint32_t node) { return p[(size_t)ctx * 256 + node]; }
};

} // namespace table_codec_detail

// Codes q[0..n) and returns the bytes.
inline std::vector<uint8_t> encode_table_counts(const uint8_t* q, size_t n) {
  using namespace table_codec_detail;
  Model m;
  std::vector<uint8_t> out;
  uint64_t low = 0;
  uint32_t range = 0xFFFFFFFFu;
  uint8_t cache = 0;
  uint64_t cache_size = 1;
  auto shift_low = [&] {
    if ((uint32_t)low < 0xFF000000u || (low >> 32) != 0) {
      uint8_t carry = (uint8_t)(low >> 32);
      uint8_t b = cache;
      do {
        out.push_back((uint8_t)(b + carry));
        b = 0xFF;
      } while (--cache_size != 0);
      cache = (uint8_t)(low >> 24);
    }
    ++cache_size;
    low = (low & 0x00FFFFFFu) << 8;
  };
  for (size_t i = 0; i < n; ++i) {
    int ctx = context(q, i);
    uint32_t node = 1;
    for (int b = 7; b >= 0; --b) {
      uint32_t bit = (q[i] >> b) & 1;
      uint16_t& p = m.at(ctx, node);
      uint32_t bound = (range >> kBitProbBits) * p;
      if (!bit) {
        range = bound;
        p = (uint16_t)(p + ((kProbOne - p) >> kMoveBits));
      } else {
        low += bound;
        range -= bound;
        p = (uint16_t)(p - (p >> kMoveBits));
      }
      while (range < kTop) {
        range <<= 8;
        shift_low();
      }
      node = node * 2 + bit;
    }
  }
  for (int k = 0; k < 5; ++k) shift_low();
  return out;
}

// Decodes n counts into q from in[0..in_len). Returns false if the coded
// data runs out before all n are decoded (a truncated or corrupt table);
// any other corruption yields wrong counts, which the rANS decoders then
// reject or which decode to wrong output (the format has no checksum).
inline bool decode_table_counts(const uint8_t* in, size_t in_len, uint8_t* q, size_t n) {
  using namespace table_codec_detail;
  Model m;
  size_t ip = 0;
  bool overrun = false;
  auto next = [&]() -> uint32_t {
    if (ip < in_len) return in[ip++];
    overrun = true;
    return 0;
  };
  uint32_t range = 0xFFFFFFFFu, code = 0;
  for (int k = 0; k < 5; ++k) code = (code << 8) | next();
  for (size_t i = 0; i < n; ++i) {
    int ctx = context(q, i);
    uint32_t node = 1;
    for (int b = 0; b < 8; ++b) {
      uint16_t& p = m.at(ctx, node);
      uint32_t bound = (range >> kBitProbBits) * p;
      uint32_t bit;
      if (code < bound) {
        range = bound;
        p = (uint16_t)(p + ((kProbOne - p) >> kMoveBits));
        bit = 0;
      } else {
        code -= bound;
        range -= bound;
        p = (uint16_t)(p - (p >> kMoveBits));
        bit = 1;
      }
      while (range < kTop) {
        range <<= 8;
        code = (code << 8) | next();
      }
      node = node * 2 + bit;
    }
    q[i] = (uint8_t)(node - 256);
  }
  return !overrun;
}

} // namespace gpusqz
