// gzp: a small GPU-accelerated file compressor.
//
// Design: the input is split into fixed-size, independent chunks. Each
// chunk is LZ-compressed (or stored raw if that doesn't help) by one CUDA
// warp. See README.md for the format and the tradeoffs.
//
// Host side, batches of chunks flow through a ring of up to three buffer
// sets, each with its own stream, so batch i+1's file read and upload
// overlap batch i's kernel and batch i-1's download and file write.
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

constexpr int kMaxSets = 3;
// Aim for this many batches per file so the pipeline has something to
// overlap, but keep each batch at least kMinBatchBytes of input (tiny
// launches waste the GPU) unless the file itself is smaller. Pinned and
// device allocations scale with batch size and dominate the fixed setup
// cost on WSL2, so this is deliberately modest; three streams in flight
// still keep the GPU busy.
constexpr uint32_t kTargetBatches = 8;
constexpr size_t kMinBatchBytes = 32u << 20;
constexpr size_t kStdioBuf = 4u << 20;

void die(const std::string& msg) {
  std::fprintf(stderr, "gzp: %s\n", msg.c_str());
  std::exit(1);
}

void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    die(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// GZP_VERBOSE=1 prints per-stage timing so we can tell whether a run is
// bound by file I/O, PCIe copies, or the kernel. GPU stages are measured
// with cudaEvents on their own streams, so they stay meaningful when the
// GPU is shared and stages overlap; their sum can legitimately exceed wall.
struct Stats {
  bool enabled = std::getenv("GZP_VERBOSE") != nullptr;
  double setup_s = 0, fread_s = 0, h2d_s = 0, kernel_s = 0, d2h_s = 0, fwrite_s = 0;
  uint64_t h2d_bytes = 0, d2h_bytes = 0, bytes_in = 0, bytes_out = 0;
  int batches = 0, sets = 0;
  uint32_t batch_chunks = 0;

  void add_span(double& acc, cudaEvent_t a, cudaEvent_t b) {
    if (!enabled) return;
    float ms = 0;
    check_cuda(cudaEventElapsedTime(&ms, a, b), "cudaEventElapsedTime");
    acc += ms / 1000.0;
  }
  void report(const char* mode, double wall_s) const {
    if (!enabled) return;
    auto mbps = [](uint64_t bytes, double s) { return s > 0 ? bytes / 1e6 / s : 0.0; };
    std::fprintf(stderr,
                 "gzp[%s] in=%llu out=%llu wall=%.3fs (%.1f MB/s)  batches=%d x %u chunks, %d buffer sets\n"
                 "  setup  %.3fs  (CUDA context + buffer allocation, fixed cost)\n"
                 "  fread  %.3fs  (%.1f MB/s of input)\n"
                 "  h2d    %.3fs  (%.1f MB/s, %llu bytes)\n"
                 "  kernel %.3fs  (%.1f MB/s of input)\n"
                 "  d2h    %.3fs  (%.1f MB/s, %llu bytes)\n"
                 "  fwrite %.3fs  (%.1f MB/s of output)\n",
                 mode, (unsigned long long)bytes_in, (unsigned long long)bytes_out, wall_s,
                 mbps(bytes_in, wall_s), batches, batch_chunks, sets, setup_s, fread_s,
                 mbps(bytes_in, fread_s), h2d_s, mbps(h2d_bytes, h2d_s), (unsigned long long)h2d_bytes,
                 kernel_s, mbps(bytes_in, kernel_s), d2h_s, mbps(d2h_bytes, d2h_s),
                 (unsigned long long)d2h_bytes, fwrite_s, mbps(bytes_out, fwrite_s));
  }
};
Stats g_stats;

template <typename T>
struct DevBuf {
  T* p = nullptr;
  bool alloc(size_t count) {
    release();
    if (cudaMalloc(&p, count * sizeof(T)) != cudaSuccess) {
      cudaGetLastError();
      p = nullptr;
      return false;
    }
    return true;
  }
  void release() {
    if (p) cudaFree(p);
    p = nullptr;
  }
  ~DevBuf() { release(); }
};

// Pinned host memory: required for cudaMemcpyAsync to actually be
// asynchronous (from pageable memory it silently serialises).
template <typename T>
struct PinBuf {
  T* p = nullptr;
  bool alloc(size_t count) {
    release();
    if (cudaHostAlloc(&p, count * sizeof(T), cudaHostAllocDefault) != cudaSuccess) {
      cudaGetLastError();
      p = nullptr;
      return false;
    }
    return true;
  }
  void release() {
    if (p) cudaFreeHost(p);
    p = nullptr;
  }
  ~PinBuf() { release(); }
};

struct StreamEvents {
  cudaStream_t stream = nullptr;
  cudaEvent_t h2d0 = nullptr, h2d1 = nullptr, k0 = nullptr, k1 = nullptr, d2h0 = nullptr, d2h1 = nullptr;
  cudaEvent_t meta = nullptr;

  void create() {
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    for (cudaEvent_t* e : {&h2d0, &h2d1, &k0, &k1, &d2h0, &d2h1, &meta}) {
      check_cuda(cudaEventCreate(e), "cudaEventCreate");
    }
  }
  ~StreamEvents() {
    for (cudaEvent_t e : {h2d0, h2d1, k0, k1, d2h0, d2h1, meta}) {
      if (e) cudaEventDestroy(e);
    }
    if (stream) cudaStreamDestroy(stream);
  }
};

struct Plan {
  uint32_t batch = 0;
  int sets = 1;
  int batches = 0;
};

// Sizes batches from currently-free VRAM (the GPU may be shared) and the
// file's chunk count. Returns the largest batch we should try; callers
// halve it if allocation still fails.
Plan plan_batches(uint32_t chunk_count, uint32_t chunk_size, size_t dev_bytes_per_chunk) {
  size_t free_bytes = 0, total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  size_t budget = std::min<size_t>(free_bytes / 2, 1ull << 30);

  uint32_t mem_max = (uint32_t)std::max<size_t>(1, budget / (kMaxSets * dev_bytes_per_chunk));
  uint32_t min_batch = (uint32_t)std::max<size_t>(1, kMinBatchBytes / chunk_size);
  uint32_t want = std::max(min_batch, (chunk_count + kTargetBatches - 1) / kTargetBatches);

  Plan p;
  p.batch = std::min({want, mem_max, chunk_count});
  p.batches = (int)((chunk_count + p.batch - 1) / p.batch);
  p.sets = std::min(kMaxSets, p.batches);
  return p;
}

uint64_t file_size(FILE* f) {
  long cur = ftell(f);
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, cur, SEEK_SET);
  return (uint64_t)sz;
}

// ---------------------------------------------------------------------------
// Compress
// ---------------------------------------------------------------------------

struct CompressSet {
  PinBuf<uint8_t> h_in, h_out;
  PinBuf<uint32_t> h_in_lens, h_sizes, h_offsets;
  DevBuf<uint8_t> d_in, d_slots, d_packed, d_temp, d_scratch;
  DevBuf<uint32_t> d_in_lens, d_start, d_sizes, d_offsets;
  StreamEvents ev;
  uint32_t n = 0, first = 0;
  bool in_flight = false, d2h_enqueued = false;

  bool alloc(uint32_t batch, uint32_t chunk_size, uint32_t slot_stride, size_t temp_bytes) {
    return h_in.alloc((size_t)batch * chunk_size) && h_out.alloc((size_t)batch * slot_stride) &&
           h_in_lens.alloc(batch) && h_sizes.alloc(batch) && h_offsets.alloc(batch + 1) &&
           d_in.alloc((size_t)batch * chunk_size) && d_slots.alloc((size_t)batch * slot_stride) &&
           d_packed.alloc((size_t)batch * slot_stride) && d_temp.alloc(temp_bytes) &&
           d_scratch.alloc((size_t)batch * scratch_bytes(chunk_size)) && d_in_lens.alloc(batch) &&
           d_start.alloc(batch) && d_sizes.alloc(batch) && d_offsets.alloc(batch + 1);
  }
  void release() {
    h_in.release(); h_out.release(); h_in_lens.release(); h_sizes.release(); h_offsets.release();
    d_in.release(); d_slots.release(); d_packed.release(); d_temp.release(); d_scratch.release();
    d_in_lens.release(); d_start.release(); d_sizes.release(); d_offsets.release();
  }
};

struct Compressor {
  FILE* in;
  FILE* out;
  uint32_t chunk_size, chunk_count, slot_stride;
  uint64_t total_size;
  Mode mode;
  size_t temp_bytes;
  std::vector<ChunkEntry> entries;
  std::vector<CompressSet> sets;
  Plan plan;
  uint64_t payload_offset = 0;
  uint32_t next_chunk = 0;

  void allocate() {
    size_t dev_per_chunk =
        (size_t)chunk_size + 2 * (size_t)slot_stride + scratch_bytes(chunk_size) + 4 * sizeof(uint32_t);
    plan = plan_batches(chunk_count, chunk_size, dev_per_chunk);
    sets.resize(plan.sets);
    for (auto& s : sets) s.ev.create();
    // Retry with a smaller batch if the GPU (or pinned host memory under
    // WSL2) can't give us what cudaMemGetInfo suggested.
    for (;;) {
      temp_bytes = compaction_temp_bytes(plan.batch);
      bool ok = true;
      for (auto& s : sets) {
        if (!s.alloc(plan.batch, chunk_size, slot_stride, temp_bytes)) {
          ok = false;
          break;
        }
      }
      if (ok) break;
      for (auto& s : sets) s.release();
      if (plan.batch == 1) die("out of GPU or pinned host memory even at 1 chunk per batch");
      plan.batch = std::max<uint32_t>(1, plan.batch / 2);
    }
    plan.batches = (int)((chunk_count + plan.batch - 1) / plan.batch);
    g_stats.batches = plan.batches;
    g_stats.sets = plan.sets;
    g_stats.batch_chunks = plan.batch;
  }

  void enqueue_d2h(CompressSet& s) {
    // Needs the packed total, which only exists once the scan has run.
    check_cuda(cudaEventSynchronize(s.ev.meta), "wait for chunk sizes");
    uint32_t total = s.h_offsets.p[s.n];
    check_cuda(cudaEventRecord(s.ev.d2h0, s.ev.stream), "cudaEventRecord");
    check_cuda(cudaMemcpyAsync(s.h_out.p, s.d_packed.p, total, cudaMemcpyDeviceToHost, s.ev.stream),
               "D2H packed output");
    check_cuda(cudaEventRecord(s.ev.d2h1, s.ev.stream), "cudaEventRecord");
    g_stats.d2h_bytes += total;
    s.d2h_enqueued = true;
  }

  void finish(CompressSet& s) {
    if (!s.in_flight) return;
    if (!s.d2h_enqueued) enqueue_d2h(s);
    check_cuda(cudaEventSynchronize(s.ev.d2h1), "wait for batch");
    g_stats.add_span(g_stats.h2d_s, s.ev.h2d0, s.ev.h2d1);
    g_stats.add_span(g_stats.kernel_s, s.ev.k0, s.ev.k1);
    g_stats.add_span(g_stats.d2h_s, s.ev.d2h0, s.ev.d2h1);

    double t = now_s();
    uint32_t total = s.h_offsets.p[s.n];
    if (std::fwrite(s.h_out.p, 1, total, out) != total) die("write failed");
    for (uint32_t c = 0; c < s.n; ++c) {
      uint32_t csize = s.h_sizes.p[c];
      entries[s.first + c] = ChunkEntry{payload_offset, csize, s.h_in_lens.p[c]};
      payload_offset += csize;
    }
    g_stats.fwrite_s += now_s() - t;
    s.in_flight = false;
    s.d2h_enqueued = false;
  }

  void launch(CompressSet& s) {
    uint32_t n = std::min(plan.batch, chunk_count - next_chunk);
    s.n = n;
    s.first = next_chunk;

    double t = now_s();
    for (uint32_t c = 0; c < n; ++c) {
      size_t want = chunk_size;
      if (s.first + c == chunk_count - 1) want = (size_t)(total_size - (uint64_t)(s.first + c) * chunk_size);
      if (std::fread(s.h_in.p + (size_t)c * chunk_size, 1, want, in) != want) die("short read on input file");
      s.h_in_lens.p[c] = (uint32_t)want;
    }
    g_stats.fread_s += now_s() - t;
    next_chunk += n;

    cudaStream_t st = s.ev.stream;
    check_cuda(cudaEventRecord(s.ev.h2d0, st), "cudaEventRecord");
    check_cuda(cudaMemcpyAsync(s.d_in.p, s.h_in.p, (size_t)n * chunk_size, cudaMemcpyHostToDevice, st),
               "H2D input");
    check_cuda(cudaMemcpyAsync(s.d_in_lens.p, s.h_in_lens.p, n * sizeof(uint32_t), cudaMemcpyHostToDevice, st),
               "H2D lens");
    check_cuda(cudaEventRecord(s.ev.h2d1, st), "cudaEventRecord");
    g_stats.h2d_bytes += (uint64_t)n * chunk_size;

    check_cuda(cudaEventRecord(s.ev.k0, st), "cudaEventRecord");
    launch_compress(s.d_in.p, chunk_size, n, s.d_in_lens.p, s.d_slots.p, slot_stride, s.d_start.p, s.d_sizes.p,
                    s.d_scratch.p, mode, st);
    check_cuda(cudaGetLastError(), "compress_kernel launch");
    check_cuda(launch_compact(s.d_slots.p, slot_stride, s.d_start.p, s.d_sizes.p, n, s.d_offsets.p, s.d_packed.p,
                              s.d_temp.p, temp_bytes, st),
               "compaction");
    check_cuda(cudaEventRecord(s.ev.k1, st), "cudaEventRecord");

    check_cuda(cudaMemcpyAsync(s.h_sizes.p, s.d_sizes.p, n * sizeof(uint32_t), cudaMemcpyDeviceToHost, st),
               "D2H sizes");
    check_cuda(cudaMemcpyAsync(s.h_offsets.p, s.d_offsets.p, (n + 1) * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, st),
               "D2H offsets");
    check_cuda(cudaEventRecord(s.ev.meta, st), "cudaEventRecord");
    s.in_flight = true;
    s.d2h_enqueued = false;
  }

  void run() {
    for (int b = 0; b < plan.batches; ++b) {
      CompressSet& s = sets[b % plan.sets];
      finish(s);
      launch(s);
      // Defer the previous batch's big download until now: its sizes are
      // almost certainly ready, and waiting for them no longer idles the
      // GPU because batch b is already queued.
      if (b > 0) {
        CompressSet& prev = sets[(b - 1) % plan.sets];
        if (prev.in_flight && !prev.d2h_enqueued) enqueue_d2h(prev);
      }
    }
    for (int b = std::max(0, plan.batches - plan.sets); b < plan.batches; ++b) finish(sets[b % plan.sets]);
  }
};

void compress(const std::string& in_path, const std::string& out_path, uint32_t chunk_size, Mode mode) {
  FILE* in = std::fopen(in_path.c_str(), "rb");
  if (!in) die("cannot open input: " + in_path);
  std::setvbuf(in, nullptr, _IOFBF, kStdioBuf);
  uint64_t total_size = file_size(in);

  uint32_t chunk_count = total_size == 0 ? 0 : (uint32_t)((total_size + chunk_size - 1) / chunk_size);

  FILE* out = std::fopen(out_path.c_str(), "wb");
  if (!out) die("cannot open output: " + out_path);
  std::setvbuf(out, nullptr, _IOFBF, kStdioBuf);

  FileHeader header{kMagic, kVersion, chunk_size, total_size, chunk_count};
  std::fwrite(&header, sizeof(header), 1, out);
  long entries_pos = ftell(out);
  std::vector<ChunkEntry> entries(chunk_count);
  // Reserve space for the entry table; filled in for real once we know
  // each chunk's compressed size.
  std::fwrite(entries.data(), sizeof(ChunkEntry), (size_t)chunk_count, out);

  if (chunk_count > 0) {
    Compressor cz;
    cz.in = in;
    cz.out = out;
    cz.chunk_size = chunk_size;
    cz.chunk_count = chunk_count;
    cz.slot_stride = worst_case_size(chunk_size);
    cz.total_size = total_size;
    cz.mode = mode;
    cz.entries.swap(entries);

    double t = now_s();
    cz.allocate();
    g_stats.setup_s += now_s() - t;

    cz.run();

    entries.swap(cz.entries);
    g_stats.bytes_in = total_size;
    g_stats.bytes_out = cz.payload_offset + sizeof(FileHeader) + entries.size() * sizeof(ChunkEntry);
  }

  std::fseek(out, entries_pos, SEEK_SET);
  std::fwrite(entries.data(), sizeof(ChunkEntry), entries.size(), out);
  std::fclose(in);
  if (std::fclose(out) != 0) die("write failed");
}

// ---------------------------------------------------------------------------
// Decompress
// ---------------------------------------------------------------------------

struct DecompressSet {
  PinBuf<uint8_t> h_in, h_out;
  PinBuf<uint32_t> h_in_offsets, h_in_lens, h_out_lens, h_err;
  DevBuf<uint8_t> d_in, d_out, d_scratch;
  DevBuf<uint32_t> d_in_offsets, d_in_lens, d_out_lens, d_err;
  StreamEvents ev;
  uint32_t n = 0, first = 0;
  bool in_flight = false;

  bool alloc(uint32_t batch, uint32_t chunk_size, uint32_t slot_stride) {
    return h_in.alloc((size_t)batch * slot_stride) && h_out.alloc((size_t)batch * chunk_size) &&
           h_in_offsets.alloc(batch) && h_in_lens.alloc(batch) && h_out_lens.alloc(batch) && h_err.alloc(1) &&
           d_in.alloc((size_t)batch * slot_stride) && d_out.alloc((size_t)batch * chunk_size) &&
           d_scratch.alloc((size_t)batch * scratch_bytes(chunk_size)) && d_in_offsets.alloc(batch) &&
           d_in_lens.alloc(batch) && d_out_lens.alloc(batch) && d_err.alloc(1);
  }
  void release() {
    h_in.release(); h_out.release(); h_in_offsets.release(); h_in_lens.release(); h_out_lens.release();
    h_err.release(); d_in.release(); d_out.release(); d_scratch.release(); d_in_offsets.release();
    d_in_lens.release(); d_out_lens.release(); d_err.release();
  }
};

struct Decompressor {
  FILE* in;
  FILE* out;
  FileHeader header;
  uint32_t slot_stride;
  long payload_start;
  std::vector<ChunkEntry> entries;
  std::vector<DecompressSet> sets;
  Plan plan;
  uint32_t next_chunk = 0;

  void allocate() {
    size_t dev_per_chunk =
        (size_t)header.chunk_size + (size_t)slot_stride + scratch_bytes(header.chunk_size) + 3 * sizeof(uint32_t);
    plan = plan_batches(header.chunk_count, header.chunk_size, dev_per_chunk);
    sets.resize(plan.sets);
    for (auto& s : sets) s.ev.create();
    for (;;) {
      bool ok = true;
      for (auto& s : sets) {
        if (!s.alloc(plan.batch, header.chunk_size, slot_stride)) {
          ok = false;
          break;
        }
      }
      if (ok) break;
      for (auto& s : sets) s.release();
      if (plan.batch == 1) die("out of GPU or pinned host memory even at 1 chunk per batch");
      plan.batch = std::max<uint32_t>(1, plan.batch / 2);
    }
    plan.batches = (int)((header.chunk_count + plan.batch - 1) / plan.batch);
    g_stats.batches = plan.batches;
    g_stats.sets = plan.sets;
    g_stats.batch_chunks = plan.batch;
  }

  void finish(DecompressSet& s) {
    if (!s.in_flight) return;
    check_cuda(cudaEventSynchronize(s.ev.d2h1), "wait for batch");
    g_stats.add_span(g_stats.h2d_s, s.ev.h2d0, s.ev.h2d1);
    g_stats.add_span(g_stats.kernel_s, s.ev.k0, s.ev.k1);
    g_stats.add_span(g_stats.d2h_s, s.ev.d2h0, s.ev.d2h1);
    if (*s.h_err.p) die("corrupt input: malformed chunk data");

    double t = now_s();
    // Every chunk but the file's last is full-size, so a batch's output is
    // one contiguous run in h_out; fall back to per-chunk writes otherwise.
    bool contiguous = true;
    uint64_t total = 0;
    for (uint32_t c = 0; c < s.n; ++c) {
      if (c + 1 < s.n && s.h_out_lens.p[c] != header.chunk_size) contiguous = false;
      total += s.h_out_lens.p[c];
    }
    if (contiguous) {
      if (std::fwrite(s.h_out.p, 1, total, out) != total) die("write failed");
    } else {
      for (uint32_t c = 0; c < s.n; ++c) {
        uint32_t len = s.h_out_lens.p[c];
        if (std::fwrite(s.h_out.p + (size_t)c * header.chunk_size, 1, len, out) != len) die("write failed");
      }
    }
    g_stats.fwrite_s += now_s() - t;
    s.in_flight = false;
  }

  void launch(DecompressSet& s) {
    uint32_t n = std::min(plan.batch, header.chunk_count - next_chunk);
    s.n = n;
    s.first = next_chunk;

    // The writer lays chunks out back to back in file order, so a batch's
    // payload is one contiguous range we can read with a single fread.
    uint64_t base = entries[s.first].offset;
    uint64_t expect = base;
    for (uint32_t c = 0; c < n; ++c) {
      const ChunkEntry& e = entries[s.first + c];
      if (e.compressed_size > slot_stride) die("corrupt chunk table: compressed_size too large");
      if (e.original_size > header.chunk_size) die("corrupt chunk table: original_size too large");
      if (e.offset != expect) die("corrupt chunk table: payload not contiguous");
      expect += e.compressed_size;
      s.h_in_offsets.p[c] = (uint32_t)(e.offset - base);
      s.h_in_lens.p[c] = e.compressed_size;
      s.h_out_lens.p[c] = e.original_size;
    }
    uint64_t total_in = expect - base;

    double t = now_s();
    if (std::fseek(in, payload_start + (long)base, SEEK_SET) != 0) die("seek failed");
    if (std::fread(s.h_in.p, 1, total_in, in) != total_in) die("short read on compressed payload");
    g_stats.fread_s += now_s() - t;
    next_chunk += n;

    cudaStream_t st = s.ev.stream;
    check_cuda(cudaEventRecord(s.ev.h2d0, st), "cudaEventRecord");
    check_cuda(cudaMemcpyAsync(s.d_in.p, s.h_in.p, total_in, cudaMemcpyHostToDevice, st), "H2D input");
    check_cuda(cudaMemcpyAsync(s.d_in_offsets.p, s.h_in_offsets.p, n * sizeof(uint32_t), cudaMemcpyHostToDevice, st),
               "H2D offsets");
    check_cuda(cudaMemcpyAsync(s.d_in_lens.p, s.h_in_lens.p, n * sizeof(uint32_t), cudaMemcpyHostToDevice, st),
               "H2D lens");
    check_cuda(cudaMemcpyAsync(s.d_out_lens.p, s.h_out_lens.p, n * sizeof(uint32_t), cudaMemcpyHostToDevice, st),
               "H2D out_lens");
    check_cuda(cudaEventRecord(s.ev.h2d1, st), "cudaEventRecord");
    g_stats.h2d_bytes += total_in;

    check_cuda(cudaMemsetAsync(s.d_err.p, 0, sizeof(uint32_t), st), "clear error flag");
    check_cuda(cudaEventRecord(s.ev.k0, st), "cudaEventRecord");
    launch_decompress(s.d_in.p, s.d_in_offsets.p, n, s.d_in_lens.p, s.d_out.p, header.chunk_size,
                       s.d_out_lens.p, s.d_scratch.p, s.d_err.p, st);
    check_cuda(cudaGetLastError(), "decompress_kernel launch");
    check_cuda(cudaEventRecord(s.ev.k1, st), "cudaEventRecord");

    size_t out_bytes = (size_t)(n - 1) * header.chunk_size + s.h_out_lens.p[n - 1];
    check_cuda(cudaEventRecord(s.ev.d2h0, st), "cudaEventRecord");
    check_cuda(cudaMemcpyAsync(s.h_out.p, s.d_out.p, out_bytes, cudaMemcpyDeviceToHost, st), "D2H output");
    check_cuda(cudaMemcpyAsync(s.h_err.p, s.d_err.p, sizeof(uint32_t), cudaMemcpyDeviceToHost, st), "D2H err");
    check_cuda(cudaEventRecord(s.ev.d2h1, st), "cudaEventRecord");
    g_stats.d2h_bytes += out_bytes;
    s.in_flight = true;
  }

  void run() {
    for (int b = 0; b < plan.batches; ++b) {
      DecompressSet& s = sets[b % plan.sets];
      finish(s);
      launch(s);
    }
    for (int b = std::max(0, plan.batches - plan.sets); b < plan.batches; ++b) finish(sets[b % plan.sets]);
  }
};

void decompress(const std::string& in_path, const std::string& out_path) {
  FILE* in = std::fopen(in_path.c_str(), "rb");
  if (!in) die("cannot open input: " + in_path);
  std::setvbuf(in, nullptr, _IOFBF, kStdioBuf);

  FileHeader header;
  if (std::fread(&header, sizeof(header), 1, in) != 1) die("truncated header");
  if (header.magic != kMagic) die("bad magic (not a gzp file)");
  if (header.version != kVersion) die("unsupported version");
  if (header.chunk_size == 0 || header.chunk_size > kMaxChunkSize) die("corrupt header: bad chunk_size");

  std::vector<ChunkEntry> entries(header.chunk_count);
  if (header.chunk_count > 0 &&
      std::fread(entries.data(), sizeof(ChunkEntry), header.chunk_count, in) != header.chunk_count) {
    die("truncated chunk table");
  }
  long payload_start = ftell(in);

  FILE* out = std::fopen(out_path.c_str(), "wb");
  if (!out) die("cannot open output: " + out_path);
  std::setvbuf(out, nullptr, _IOFBF, kStdioBuf);

  if (header.chunk_count > 0) {
    Decompressor dz;
    dz.in = in;
    dz.out = out;
    dz.header = header;
    dz.slot_stride = worst_case_size(header.chunk_size);
    dz.payload_start = payload_start;
    dz.entries.swap(entries);

    double t = now_s();
    dz.allocate();
    g_stats.setup_s += now_s() - t;

    dz.run();
    g_stats.bytes_in = file_size(in);
    g_stats.bytes_out = header.original_size;
  }

  std::fclose(in);
  if (std::fclose(out) != 0) die("write failed");
}

void usage() {
  std::fprintf(stderr,
               "usage:\n"
               "  gzp c <input> <output> [chunk_size] [--mode lz|lzrans]   compress (default lzrans)\n"
               "  gzp d <input> <output>                                   decompress\n");
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
    Mode codec = Mode::LzRans;
    for (int i = 4; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--mode" && i + 1 < argc) {
        std::string m = argv[++i];
        if (m == "lz") codec = Mode::Lz;
        else if (m == "lzrans") codec = Mode::LzRans;
        else die("unknown mode: " + m);
      } else {
        chunk_size = (uint32_t)std::strtoul(a.c_str(), nullptr, 10);
      }
    }
    if (chunk_size == 0 || chunk_size > kMaxChunkSize) die("chunk_size must be in (0, 65536]");
    compress(in_path, out_path, chunk_size, codec);
  } else if (mode == "d") {
    decompress(in_path, out_path);
  } else {
    usage();
  }
  auto t1 = std::chrono::steady_clock::now();
  double secs = std::chrono::duration<double>(t1 - t0).count();
  std::fprintf(stderr, "gzp: %s done in %.3fs\n", mode == "c" ? "compress" : "decompress", secs);
  g_stats.report(mode == "c" ? "compress" : "decompress", secs);
  return 0;
}
