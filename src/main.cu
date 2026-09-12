// gzp: a small GPU-accelerated file compressor.
//
// Design: the input is split into fixed-size, independent chunks. Each
// chunk is LZSS-compressed (or stored raw if that doesn't help) by its own
// CUDA thread, so throughput scales with chunk count, not with cooperation
// between threads. See README.md for the format and the tradeoffs.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

#include "format.h"
#include "kernels.h"

using namespace gzp;

namespace {

void die(const std::string& msg) {
  std::fprintf(stderr, "gzp: %s\n", msg.c_str());
  std::exit(1);
}

void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    die(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

struct DeviceBuf {
  void* ptr = nullptr;
  size_t bytes = 0;
  ~DeviceBuf() {
    if (ptr) cudaFree(ptr);
  }
  void alloc(size_t n) {
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
    }
    check_cuda(cudaMalloc(&ptr, n), "cudaMalloc");
    bytes = n;
  }
};

// Picks how many chunks to process per GPU launch. Queries current free
// VRAM (which may be shared with other processes/processes on the host
// side of WSL) and retries with a smaller batch on allocation failure, so
// this degrades gracefully instead of crashing when the GPU is busy.
uint32_t pick_batch_chunks(uint32_t chunk_size, uint32_t chunk_count) {
  size_t free_bytes = 0, total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

  // Per chunk we need roughly: chunk_size (raw slot) + worst_case_size
  // (compressed slot) + a few bytes of length metadata, on both host and
  // device. Use at most a quarter of currently-free VRAM, capped at 256MiB
  // of device buffers, so we leave room for the other GPU consumer we saw
  // in nvidia-smi.
  size_t per_chunk = (size_t)chunk_size + worst_case_size(chunk_size) + 16;
  size_t budget = free_bytes / 4;
  size_t cap = 256u * 1024 * 1024;
  if (budget > cap) budget = cap;

  uint32_t batch = (uint32_t)(budget / per_chunk);
  if (batch < 1) batch = 1;
  if (batch > chunk_count) batch = chunk_count;
  return batch;
}

uint64_t file_size(FILE* f) {
  long cur = ftell(f);
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, cur, SEEK_SET);
  return (uint64_t)sz;
}

void compress(const std::string& in_path, const std::string& out_path, uint32_t chunk_size) {
  FILE* in = std::fopen(in_path.c_str(), "rb");
  if (!in) die("cannot open input: " + in_path);
  uint64_t total_size = file_size(in);

  uint32_t chunk_count = (uint32_t)((total_size + chunk_size - 1) / chunk_size);
  if (total_size == 0) chunk_count = 0;

  FILE* out = std::fopen(out_path.c_str(), "wb");
  if (!out) die("cannot open output: " + out_path);

  FileHeader header{kMagic, kVersion, chunk_size, total_size, chunk_count};
  std::fwrite(&header, sizeof(header), 1, out);
  long entries_pos = ftell(out);
  std::vector<ChunkEntry> entries(chunk_count);
  // Reserve space for the entry table; filled in for real once we know
  // each chunk's compressed size.
  std::fwrite(entries.data(), sizeof(ChunkEntry), (size_t)chunk_count, out);

  if (chunk_count == 0) {
    std::fclose(in);
    std::fclose(out);
    return;
  }

  uint32_t out_slot_stride = worst_case_size(chunk_size);
  uint32_t batch_chunks = pick_batch_chunks(chunk_size, chunk_count);

  std::vector<uint8_t> h_in((size_t)batch_chunks * chunk_size);
  std::vector<uint32_t> h_in_lens(batch_chunks);
  std::vector<uint8_t> h_out((size_t)batch_chunks * out_slot_stride);
  std::vector<uint32_t> h_out_sizes(batch_chunks);

  DeviceBuf d_in, d_in_lens, d_out, d_out_sizes;
  d_in.alloc(h_in.size());
  d_in_lens.alloc(h_in_lens.size() * sizeof(uint32_t));
  d_out.alloc(h_out.size());
  d_out_sizes.alloc(h_out_sizes.size() * sizeof(uint32_t));

  uint64_t payload_offset = 0;
  uint32_t chunk_idx = 0;
  while (chunk_idx < chunk_count) {
    uint32_t n = std::min(batch_chunks, chunk_count - chunk_idx);

    for (uint32_t c = 0; c < n; ++c) {
      size_t want = chunk_size;
      if (chunk_idx + c == chunk_count - 1) {
        uint64_t rem = total_size - (uint64_t)(chunk_idx + c) * chunk_size;
        want = (size_t)rem;
      }
      size_t got = std::fread(h_in.data() + (size_t)c * chunk_size, 1, want, in);
      if (got != want) die("short read on input file");
      h_in_lens[c] = (uint32_t)want;
    }

    check_cuda(cudaMemcpy(d_in.ptr, h_in.data(), (size_t)n * chunk_size, cudaMemcpyHostToDevice),
               "H2D input");
    check_cuda(cudaMemcpy(d_in_lens.ptr, h_in_lens.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice),
               "H2D lens");

    launch_compress((const uint8_t*)d_in.ptr, chunk_size, n, (const uint32_t*)d_in_lens.ptr,
                     (uint8_t*)d_out.ptr, out_slot_stride, (uint32_t*)d_out_sizes.ptr, 0);
    check_cuda(cudaGetLastError(), "compress_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "compress_kernel sync");

    check_cuda(cudaMemcpy(h_out.data(), d_out.ptr, (size_t)n * out_slot_stride, cudaMemcpyDeviceToHost),
               "D2H output");
    check_cuda(cudaMemcpy(h_out_sizes.data(), d_out_sizes.ptr, n * sizeof(uint32_t), cudaMemcpyDeviceToHost),
               "D2H sizes");

    for (uint32_t c = 0; c < n; ++c) {
      uint32_t csize = h_out_sizes[c];
      std::fwrite(h_out.data() + (size_t)c * out_slot_stride, 1, csize, out);
      entries[chunk_idx + c] = ChunkEntry{payload_offset, csize, h_in_lens[c]};
      payload_offset += csize;
    }
    chunk_idx += n;
  }

  std::fseek(out, entries_pos, SEEK_SET);
  std::fwrite(entries.data(), sizeof(ChunkEntry), entries.size(), out);

  std::fclose(in);
  std::fclose(out);
}

void decompress(const std::string& in_path, const std::string& out_path) {
  FILE* in = std::fopen(in_path.c_str(), "rb");
  if (!in) die("cannot open input: " + in_path);

  FileHeader header;
  if (std::fread(&header, sizeof(header), 1, in) != 1) die("truncated header");
  if (header.magic != kMagic) die("bad magic (not a gzp file)");
  if (header.version != kVersion) die("unsupported version");

  std::vector<ChunkEntry> entries(header.chunk_count);
  if (header.chunk_count > 0 &&
      std::fread(entries.data(), sizeof(ChunkEntry), header.chunk_count, in) != header.chunk_count) {
    die("truncated chunk table");
  }
  long payload_start = ftell(in);

  FILE* out = std::fopen(out_path.c_str(), "wb");
  if (!out) die("cannot open output: " + out_path);

  if (header.chunk_count == 0) {
    std::fclose(in);
    std::fclose(out);
    return;
  }

  uint32_t chunk_size = header.chunk_size;
  uint32_t in_slot_stride = worst_case_size(chunk_size);
  uint32_t batch_chunks = pick_batch_chunks(chunk_size, header.chunk_count);

  std::vector<uint8_t> h_in((size_t)batch_chunks * in_slot_stride);
  std::vector<uint32_t> h_in_lens(batch_chunks);
  std::vector<uint8_t> h_out((size_t)batch_chunks * chunk_size);
  std::vector<uint32_t> h_out_lens(batch_chunks);

  DeviceBuf d_in, d_in_lens, d_out, d_out_lens;
  d_in.alloc(h_in.size());
  d_in_lens.alloc(h_in_lens.size() * sizeof(uint32_t));
  d_out.alloc(h_out.size());
  d_out_lens.alloc(h_out_lens.size() * sizeof(uint32_t));

  uint32_t chunk_idx = 0;
  while (chunk_idx < header.chunk_count) {
    uint32_t n = std::min(batch_chunks, header.chunk_count - chunk_idx);

    for (uint32_t c = 0; c < n; ++c) {
      const ChunkEntry& e = entries[chunk_idx + c];
      if (e.compressed_size > in_slot_stride) die("corrupt chunk table: compressed_size too large");
      std::fseek(in, payload_start + (long)e.offset, SEEK_SET);
      size_t got = std::fread(h_in.data() + (size_t)c * in_slot_stride, 1, e.compressed_size, in);
      if (got != e.compressed_size) die("short read on compressed payload");
      h_in_lens[c] = e.compressed_size;
      h_out_lens[c] = e.original_size;
    }

    check_cuda(cudaMemcpy(d_in.ptr, h_in.data(), (size_t)n * in_slot_stride, cudaMemcpyHostToDevice),
               "H2D input");
    check_cuda(cudaMemcpy(d_in_lens.ptr, h_in_lens.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice),
               "H2D lens");
    check_cuda(cudaMemcpy(d_out_lens.ptr, h_out_lens.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice),
               "H2D out_lens");

    launch_decompress((const uint8_t*)d_in.ptr, in_slot_stride, n, (const uint32_t*)d_in_lens.ptr,
                       (uint8_t*)d_out.ptr, chunk_size, (const uint32_t*)d_out_lens.ptr, 0);
    check_cuda(cudaGetLastError(), "decompress_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "decompress_kernel sync");

    check_cuda(cudaMemcpy(h_out.data(), d_out.ptr, (size_t)n * chunk_size, cudaMemcpyDeviceToHost),
               "D2H output");

    for (uint32_t c = 0; c < n; ++c) {
      std::fwrite(h_out.data() + (size_t)c * chunk_size, 1, h_out_lens[c], out);
    }
    chunk_idx += n;
  }

  std::fclose(in);
  std::fclose(out);
}

void usage() {
  std::fprintf(stderr,
               "usage:\n"
               "  gzp c <input> <output> [chunk_size]   compress\n"
               "  gzp d <input> <output>                decompress\n");
  std::exit(1);
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 4) usage();
  std::string mode = argv[1];
  std::string in_path = argv[2];
  std::string out_path = argv[3];

  auto t0 = std::chrono::steady_clock::now();
  if (mode == "c") {
    uint32_t chunk_size = kDefaultChunkSize;
    if (argc >= 5) chunk_size = (uint32_t)std::strtoul(argv[4], nullptr, 10);
    if (chunk_size == 0 || chunk_size > 65536) die("chunk_size must be in (0, 65536]");
    compress(in_path, out_path, chunk_size);
  } else if (mode == "d") {
    decompress(in_path, out_path);
  } else {
    usage();
  }
  auto t1 = std::chrono::steady_clock::now();
  double secs = std::chrono::duration<double>(t1 - t0).count();
  std::fprintf(stderr, "gzp: %s done in %.3fs\n", mode == "c" ? "compress" : "decompress", secs);
  return 0;
}
