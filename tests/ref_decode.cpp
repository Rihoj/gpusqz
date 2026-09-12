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

using namespace gzp;

namespace {

[[noreturn]] void fail(const std::string& msg) {
  std::fprintf(stderr, "gzp_refdec: %s\n", msg.c_str());
  std::exit(1);
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
    uint32_t ml = (tok & 15) + 4;
    if ((tok & 15) == 15) {
      uint32_t e = 15;
      if (!read_ext(in, in_len, ip, e)) return false;
      ml = e + 4;
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
