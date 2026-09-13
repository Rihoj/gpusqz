// gzp: a small GPU-accelerated file compressor.
//
// Design: the input is split into fixed-size, independent chunks. Each
// chunk is LZ-compressed (or stored raw if that doesn't help) by one CUDA
// warp. See README.md for the format and the tradeoffs.
//
// Host side, batches of chunks live only in device memory (a ring of up to
// kMaxSets device buffer sets, each with its own stream). File data moves
// through a small, fixed pool of pinned staging buffers instead: the main
// thread freads into an input stage and copies it up asynchronously, and a
// writer thread drains output stages the main thread fills with
// asynchronous downloads. So batch i+1's read and upload overlap batch i's
// kernels, and batch i-1's download and file write overlap both.
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

#include "format.h"
#include "kernels.h"

using namespace gzp;

namespace {

// Device buffer sets in the ring (see plan_batches). Two measured best: a
// third only buys concurrent D2H/H2D copies, which PCIe at ~13GB/s doesn't
// need, and costs a third of each batch's size.
constexpr int kDefaultSets = 2;
constexpr int kMaxSets = 3; // upper bound for GZP_FORCE_SETS

// Batch sizing (see plan_batches). Every chunk is one warp, and each warp's
// LZ parse is latency-bound: it sustains only a few MB/s on its own, so
// kernel throughput is almost exactly proportional to how many chunks are
// in flight at once. A batch therefore wants at least kMinBatchChunks
// chunks (and at least kMinBatchBytes of input) whenever the file and the
// memory budget allow it; kTargetBatches only matters for files so large
// that even that many chunks would still leave more than kTargetBatches
// batches. The earlier policy (one-eighth of the file per batch, 1GB
// budget) left the 1MB `ratio` profile with 32 chunks per batch -- 32 warps
// on a 36-SM GPU -- and was ~6x slower on that profile for that reason
// alone.
constexpr uint32_t kTargetBatches = 8;
constexpr uint32_t kMinBatchChunks = 1024;
constexpr size_t kMinBatchBytes = 32u << 20;
// GPU memory for batch buffers. By default gzp takes the smaller of half
// the free VRAM (the GPU may be shared) and kDefaultBudgetCap; --gpu-mem /
// GZP_GPU_MEM replace that with an explicit budget, limited only to all but
// kGpuMemReserve of the free VRAM. The cap matters most for the 1MB
// `ratio` profile, whose chunks need ~6MB of device memory each, so 4GB
// holds only ~350 of them per batch at two buffer sets.
constexpr size_t kDefaultBudgetCap = 4ull << 30;
constexpr size_t kGpuMemReserve = 256ull << 20;
size_t g_gpu_mem = 0; // explicit budget in bytes, 0 = automatic

// Pinned staging. Pinning is the expensive part of host allocation under
// WSL2 (cudaHostAlloc measured ~0.3-0.4s per GB, plus ~0.1s per GB to free
// at exit, versus ~3ms per GB for cudaMalloc), so batches are sized by
// device memory alone and host transfers go through this many fixed-size
// stages, whatever the batch size.
constexpr size_t kStageBytes = 8u << 20;
constexpr int kInStages = 4;
constexpr int kOutStages = 8;
constexpr size_t kStdioBuf = 4u << 20;

[[noreturn]] void die(const std::string& msg) {
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
// with per-batch cudaEvents on their own streams, so they stay meaningful
// when stages overlap; their sums can legitimately exceed wall time.
enum Mark { kH2d0, kH2d1, kK0, kK1, kD2h0, kD2h1, kMarks };

struct Stats {
  bool enabled = [] {
    const char* v = std::getenv("GZP_VERBOSE");
    return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
  }();
  double setup_s = 0, fread_s = 0, fwrite_s = 0, in_wait_s = 0, out_wait_s = 0;
  uint64_t h2d_bytes = 0, d2h_bytes = 0, bytes_in = 0, bytes_out = 0;
  int batches = 0, sets = 0;
  uint32_t batch_chunks = 0;
  // `origin` is recorded before any batch is queued; each batch's marks are
  // measured against it so overlapping kernel spans can be unioned.
  cudaEvent_t origin = nullptr;
  std::vector<std::array<cudaEvent_t, kMarks>> marks;

  void start_clock(cudaStream_t stream) {
    if (!enabled) return;
    check_cuda(cudaEventCreate(&origin), "cudaEventCreate");
    check_cuda(cudaEventRecord(origin, stream), "cudaEventRecord");
  }
  void mark(int batch, Mark m, cudaStream_t stream) {
    if (!enabled) return;
    if ((size_t)batch >= marks.size()) marks.resize(batch + 1);
    cudaEvent_t& e = marks[batch][m];
    if (!e) check_cuda(cudaEventCreate(&e), "cudaEventCreate");
    check_cuda(cudaEventRecord(e, stream), "cudaEventRecord");
  }
  // Call once all GPU work has finished.
  void report(const char* mode, double wall_s) {
    if (!enabled) return;
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    double h2d_s = 0, kernel_s = 0, d2h_s = 0;
    std::vector<std::pair<float, float>> spans;
    auto at = [&](cudaEvent_t e) {
      float ms = 0;
      check_cuda(cudaEventElapsedTime(&ms, origin, e), "cudaEventElapsedTime");
      return ms;
    };
    for (auto& m : marks) {
      if (m[kH2d0] && m[kH2d1]) h2d_s += (at(m[kH2d1]) - at(m[kH2d0])) / 1000.0;
      if (m[kK0] && m[kK1]) {
        spans.emplace_back(at(m[kK0]), at(m[kK1]));
        kernel_s += (spans.back().second - spans.back().first) / 1000.0;
      }
      if (m[kD2h0] && m[kD2h1]) d2h_s += (at(m[kD2h1]) - at(m[kD2h0])) / 1000.0;
    }
    // Batches on different streams can run concurrently, so the sum of
    // spans overstates GPU time once they overlap; the union of intervals
    // is the wall-clock time during which some kernel was running.
    std::sort(spans.begin(), spans.end());
    double busy = 0, a = 0, b = -1;
    for (auto [x, y] : spans) {
      if (x > b) {
        if (b > a) busy += b - a;
        a = x;
        b = y;
      } else if (y > b) {
        b = y;
      }
    }
    if (b > a) busy += b - a;
    busy /= 1000.0;

    auto mbps = [](uint64_t bytes, double s) { return s > 0 ? bytes / 1e6 / s : 0.0; };
    std::fprintf(stderr,
                 "gzp[%s] in=%llu out=%llu wall=%.3fs (%.1f MB/s)  batches=%d x %u chunks, %d buffer sets\n"
                 "  setup  %.3fs  (CUDA context + buffer allocation, fixed cost)\n"
                 "  steady %.3fs  (%.1f MB/s of input: wall minus setup)\n"
                 "  fread  %.3fs  (%.1f MB/s of input; main thread)\n"
                 "  h2d    %.3fs  (%.1f MB/s, %llu bytes)\n"
                 "  kernel %.3fs  (%.1f MB/s of input; sum of per-batch spans)\n"
                 "  kbusy  %.3fs  (%.1f MB/s of input; wall-clock time any kernel ran)\n"
                 "  d2h    %.3fs  (%.1f MB/s, %llu bytes)\n"
                 "  fwrite %.3fs  (%.1f MB/s of output; writer thread)\n"
                 "  stall  %.3fs waiting for a free input stage, %.3fs for a free output stage\n",
                 mode, (unsigned long long)bytes_in, (unsigned long long)bytes_out, wall_s,
                 mbps(bytes_in, wall_s), batches, batch_chunks, sets, setup_s, wall_s - setup_s,
                 mbps(bytes_in, wall_s - setup_s), fread_s, mbps(bytes_in, fread_s), h2d_s,
                 mbps(h2d_bytes, h2d_s), (unsigned long long)h2d_bytes, kernel_s, mbps(bytes_in, kernel_s), busy,
                 mbps(bytes_in, busy), d2h_s, mbps(d2h_bytes, d2h_s), (unsigned long long)d2h_bytes, fwrite_s,
                 mbps(bytes_out, fwrite_s), in_wait_s, out_wait_s);
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

// ---------------------------------------------------------------------------
// Pinned staging
// ---------------------------------------------------------------------------

// One pinned staging buffer plus the event of the last async copy using it.
struct Stage {
  PinBuf<uint8_t> buf;
  cudaEvent_t ev = nullptr;
  bool pending = false; // ev recorded and not yet waited on (input stages only)

  bool init() {
    if (!buf.alloc(kStageBytes)) return false;
    check_cuda(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
    return true;
  }
  ~Stage() {
    if (ev) cudaEventDestroy(ev);
  }
};

// Input stages, used only by the main thread: fread into a stage, copy it
// up asynchronously, and reuse the stage once that copy's event has fired.
struct InRing {
  std::unique_ptr<Stage[]> st;
  int n = 0, next = 0;

  bool init(int count) {
    st.reset(new Stage[count]);
    n = count;
    for (int i = 0; i < n; ++i) {
      if (!st[i].init()) return false;
    }
    return true;
  }

  // Copies the next `bytes` of `in` to dev[0..bytes) on `stream`.
  void upload(FILE* in, uint8_t* dev, uint64_t bytes, cudaStream_t stream, const char* short_read_msg) {
    for (uint64_t off = 0; off < bytes; off += kStageBytes) {
      size_t len = (size_t)std::min<uint64_t>(kStageBytes, bytes - off);
      Stage& s = st[next];
      next = (next + 1) % n;
      if (s.pending) {
        double t = now_s();
        check_cuda(cudaEventSynchronize(s.ev), "wait for input stage");
        g_stats.in_wait_s += now_s() - t;
        s.pending = false;
      }
      double t = now_s();
      if (std::fread(s.buf.p, 1, len, in) != len) die(short_read_msg);
      g_stats.fread_s += now_s() - t;
      check_cuda(cudaMemcpyAsync(dev + off, s.buf.p, len, cudaMemcpyHostToDevice, stream), "H2D stage");
      check_cuda(cudaEventRecord(s.ev, stream), "cudaEventRecord");
      s.pending = true;
    }
    g_stats.h2d_bytes += bytes;
  }
};

// Output stages plus the thread that drains them. The main thread queues an
// async download into a free stage and hands (stage, length) over; the
// writer waits for that copy, fwrites it, and frees the stage. Jobs are
// written strictly in the order they were pushed. fwrite (~1.3GB/s on this
// WSL2 setup) is the slowest stage of decompression, so moving it off the
// main thread lets the next batch's read, upload and kernels overlap it.
class Writer {
 public:
  ~Writer() {
    if (th_.joinable()) {
      {
        std::lock_guard<std::mutex> lk(mu_);
        done_ = true;
        cv_.notify_all();
      }
      th_.join();
    }
  }

  bool init(FILE* out, int count) {
    out_ = out;
    n_ = count;
    st_.reset(new Stage[count]);
    busy_.assign(count, 0);
    for (int i = 0; i < n_; ++i) {
      if (!st_[i].init()) return false;
    }
    th_ = std::thread([this] { run(); });
    return true;
  }

  // Blocks until the next stage in ring order is free.
  Stage& acquire(int& idx) {
    std::unique_lock<std::mutex> lk(mu_);
    double t = now_s();
    cv_.wait(lk, [&] { return !busy_[next_] || !error_.empty(); });
    g_stats.out_wait_s += now_s() - t;
    if (!error_.empty()) die(error_);
    idx = next_;
    busy_[idx] = 1;
    next_ = (next_ + 1) % n_;
    return st_[idx];
  }

  // The caller has recorded st[idx].ev after its download. `err`, if not
  // null, is a pinned flag downloaded before that copy on the same stream:
  // nonzero means the batch was corrupt and must not be written.
  void push(int idx, size_t len, const uint32_t* err) {
    std::lock_guard<std::mutex> lk(mu_);
    q_.push_back(Job{idx, len, err});
    cv_.notify_all();
  }

  // Writes everything queued, stops the thread, and dies if any write failed.
  void finish() {
    if (!th_.joinable()) return;
    {
      std::lock_guard<std::mutex> lk(mu_);
      done_ = true;
      cv_.notify_all();
    }
    th_.join();
    g_stats.fwrite_s += fwrite_s_;
    if (!error_.empty()) die(error_);
  }

 private:
  struct Job {
    int idx;
    size_t len;
    const uint32_t* err;
  };

  void run() {
    for (;;) {
      Job j;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return !q_.empty() || done_; });
        if (q_.empty()) return;
        j = q_.front();
        q_.pop_front();
      }
      cudaError_t e = cudaEventSynchronize(st_[j.idx].ev);
      std::string err;
      if (e != cudaSuccess) {
        err = std::string("wait for output stage: ") + cudaGetErrorString(e);
      } else if (j.err && *(const volatile uint32_t*)j.err) {
        err = "corrupt input: malformed chunk data";
      } else {
        double t = now_s();
        bool ok = std::fwrite(st_[j.idx].buf.p, 1, j.len, out_) == j.len;
        fwrite_s_ += now_s() - t;
        if (!ok) err = "write failed";
      }
      std::lock_guard<std::mutex> lk(mu_);
      if (!err.empty() && error_.empty()) error_ = err;
      busy_[j.idx] = 0;
      cv_.notify_all();
      if (!error_.empty()) {
        // Stop writing, but keep freeing stages so the main thread never
        // blocks forever; it dies at its next acquire() or finish().
        for (Job& k : q_) busy_[k.idx] = 0;
        q_.clear();
      }
    }
  }

  FILE* out_ = nullptr;
  int n_ = 0, next_ = 0;
  std::unique_ptr<Stage[]> st_;
  std::vector<char> busy_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Job> q_;
  bool done_ = false;
  std::string error_;
  std::thread th_;
  double fwrite_s_ = 0;
};

// Downloads dev[0..bytes) through the writer's stages on `stream`.
void download(Writer& w, const uint8_t* dev, uint64_t bytes, cudaStream_t stream, const uint32_t* err) {
  for (uint64_t off = 0; off < bytes; off += kStageBytes) {
    size_t len = (size_t)std::min<uint64_t>(kStageBytes, bytes - off);
    int idx;
    Stage& s = w.acquire(idx);
    check_cuda(cudaMemcpyAsync(s.buf.p, dev + off, len, cudaMemcpyDeviceToHost, stream), "D2H stage");
    check_cuda(cudaEventRecord(s.ev, stream), "cudaEventRecord");
    w.push(idx, len, err);
  }
  g_stats.d2h_bytes += bytes;
}

// ---------------------------------------------------------------------------
// Batch planning
// ---------------------------------------------------------------------------

struct Plan {
  uint32_t batch = 0;
  int sets = 1;
  int batches = 0;
};

// Sizes batches from currently-free VRAM (the GPU may be shared) and the
// file's chunk count. Returns the largest batch we should try; callers
// halve it if allocation still fails.
//
// A file that fits in one or two batches gets the whole budget split over
// that many sets instead of being cut into smaller ones: kernel throughput
// follows the chunks in flight (see kMinBatchChunks), and in practice
// batches on different streams barely overlap on the GPU (the next one is
// still being read while this one runs), so a bigger batch beats more sets.
Plan plan_batches(uint32_t chunk_count, uint32_t chunk_size, size_t dev_bytes_per_chunk) {
  size_t free_bytes = 0, total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  size_t budget = std::min<size_t>(free_bytes / 2, kDefaultBudgetCap);
  if (g_gpu_mem) {
    size_t avail = free_bytes > kGpuMemReserve ? free_bytes - kGpuMemReserve : 0;
    budget = std::min(g_gpu_mem, avail);
    if (budget < g_gpu_mem) {
      std::fprintf(stderr, "gzp: --gpu-mem %zu MiB is more than the %zu MiB free; using %zu MiB\n",
                   g_gpu_mem >> 20, free_bytes >> 20, budget >> 20);
    }
  }

  uint32_t min_batch = (uint32_t)std::max<size_t>(1, kMinBatchBytes / chunk_size);
  uint32_t want = std::max({min_batch, kMinBatchChunks, (chunk_count + kTargetBatches - 1) / kTargetBatches});

  // Test-only override: forces compress and decompress to pick different
  // batch sizes (hence different, misaligned TableGroup boundaries on the
  // encode side vs. decode-batch boundaries), which is the one scenario
  // that exercises the per-chunk group_id lookup instead of always
  // hitting the trivial case where a decode batch sits inside one group.
  if (const char* f = std::getenv("GZP_FORCE_BATCH")) {
    uint32_t forced = (uint32_t)std::strtoul(f, nullptr, 10);
    if (forced >= 1) want = forced;
  }
  // Tuning-only override of the ring depth (1..kMaxSets).
  int max_sets = kDefaultSets;
  if (const char* f = std::getenv("GZP_FORCE_SETS")) {
    max_sets = std::clamp((int)std::strtol(f, nullptr, 10), 1, kMaxSets);
  }

  Plan p;
  for (int s = 1; s <= max_sets; ++s) {
    uint32_t mem_max = (uint32_t)std::max<size_t>(1, budget / ((size_t)s * dev_bytes_per_chunk));
    p.batch = std::min({want, mem_max, chunk_count});
    p.batches = (int)((chunk_count + p.batch - 1) / p.batch);
    p.sets = std::min(s, p.batches);
    if (p.batches <= s) break;
  }
  return p;
}

uint64_t file_size(FILE* f) {
  long cur = ftell(f);
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, cur, SEEK_SET);
  return (uint64_t)sz;
}

void create_stream(cudaStream_t& stream, std::initializer_list<cudaEvent_t*> events) {
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
  for (cudaEvent_t* e : events) check_cuda(cudaEventCreateWithFlags(e, cudaEventDisableTiming), "cudaEventCreate");
}

// ---------------------------------------------------------------------------
// Compress
// ---------------------------------------------------------------------------

// Per-chunk device scratch actually allocated for compress: the LzRans
// scratch, which also receives the compacted output once the encode kernel
// is done with it (see Compressor::launch), so it must hold at least one
// output slot per chunk. That only binds for tiny chunk sizes.
size_t compress_scratch_bytes(uint32_t chunk_size, uint32_t slot_stride) {
  return std::max(scratch_bytes(chunk_size), (size_t)slot_stride);
}

struct CompressSet {
  PinBuf<uint32_t> h_in_lens, h_sizes, h_offsets, h_shift;
  PinBuf<uint8_t> h_q; // this batch's quantised tables (up to kMaxQuantBytes)
  DevBuf<uint8_t> d_in, d_slots, d_temp, d_scratch; // d_scratch doubles as the packed output
  DevBuf<uint32_t> d_htab; // LZ parse's match-finding table, one region per chunk; see hash_table_bytes()
  DevBuf<uint32_t> d_in_lens, d_start, d_sizes, d_offsets;
  DevBuf<uint32_t> d_rans_cnt, d_rans_shift, d_rans_n_seq, d_rans_n_lit;
  DevBuf<uint16_t> d_rans_freq, d_rans_cum;
  DevBuf<uint8_t> d_rans_q;
  cudaStream_t stream = nullptr;
  cudaEvent_t h2d_done = nullptr, meta = nullptr;
  uint32_t n = 0, first = 0;
  int batch = -1;
  bool used = false, d2h_pending = false;

  ~CompressSet() {
    for (cudaEvent_t e : {h2d_done, meta}) {
      if (e) cudaEventDestroy(e);
    }
    if (stream) cudaStreamDestroy(stream);
  }
  bool alloc(uint32_t batch, uint32_t chunk_size, uint32_t slot_stride, size_t temp_bytes) {
    return h_in_lens.alloc(batch) && h_sizes.alloc(batch) && h_offsets.alloc(batch + 1) && h_shift.alloc(1) &&
           h_q.alloc(kMaxQuantBytes) && d_in.alloc((size_t)batch * chunk_size) &&
           d_slots.alloc((size_t)batch * slot_stride) && d_temp.alloc(temp_bytes) &&
           d_scratch.alloc((size_t)batch * compress_scratch_bytes(chunk_size, slot_stride)) &&
           d_htab.alloc((size_t)batch * (hash_table_bytes(chunk_size) / sizeof(uint32_t))) &&
           d_in_lens.alloc(batch) && d_start.alloc(batch) && d_sizes.alloc(batch) && d_offsets.alloc(batch + 1) &&
           d_rans_cnt.alloc(kMaxQuantBytes) && d_rans_shift.alloc(1) && d_rans_n_seq.alloc(batch) &&
           d_rans_n_lit.alloc(batch) &&
           d_rans_freq.alloc(kMaxQuantBytes) && d_rans_cum.alloc(kMaxQuantBytes) &&
           d_rans_q.alloc(kMaxQuantBytes);
  }
  void release() {
    h_in_lens.release(); h_sizes.release(); h_offsets.release(); h_shift.release(); h_q.release();
    d_in.release(); d_slots.release(); d_temp.release(); d_scratch.release(); d_htab.release();
    d_in_lens.release(); d_start.release(); d_sizes.release(); d_offsets.release();
    d_rans_cnt.release(); d_rans_shift.release(); d_rans_n_seq.release(); d_rans_n_lit.release();
    d_rans_freq.release();
    d_rans_cum.release(); d_rans_q.release();
  }
  RansBatchBufs rans_bufs() {
    return {d_rans_cnt.p, d_rans_freq.p, d_rans_cum.p, d_rans_q.p, d_rans_shift.p, d_rans_n_seq.p, d_rans_n_lit.p};
  }
};

struct Compressor {
  FILE* in;
  FILE* out;
  uint32_t chunk_size, chunk_count, slot_stride;
  uint64_t total_size;
  size_t temp_bytes;
  std::vector<ChunkEntry> entries;
  std::vector<TableGroup> groups; // one per batch
  std::vector<uint8_t> tables;     // each group's quantised counts, in group order
  int forced_lit_shift = -1;       // GZP_FORCE_LIT_SHIFT, tests and tuning only
  std::unique_ptr<CompressSet[]> sets;
  Plan plan;
  InRing in_ring;
  Writer writer;
  uint64_t payload_offset = 0;
  uint32_t next_chunk = 0;
  FILE* lit_dump = nullptr; // GZP_DUMP_LITS=<path>, see dump_literals

  void allocate() {
    size_t dev_per_chunk = (size_t)chunk_size + (size_t)slot_stride +
                           compress_scratch_bytes(chunk_size, slot_stride) + hash_table_bytes(chunk_size) +
                           6 * sizeof(uint32_t);
    plan = plan_batches(chunk_count, chunk_size, dev_per_chunk);
    sets.reset(new CompressSet[plan.sets]);
    for (int i = 0; i < plan.sets; ++i) create_stream(sets[i].stream, {&sets[i].h2d_done, &sets[i].meta});
    // Retry with a smaller batch if the GPU can't give us what
    // cudaMemGetInfo suggested.
    for (;;) {
      temp_bytes = compaction_temp_bytes(plan.batch);
      bool ok = true;
      for (int i = 0; i < plan.sets && ok; ++i) ok = sets[i].alloc(plan.batch, chunk_size, slot_stride, temp_bytes);
      if (ok) break;
      for (int i = 0; i < plan.sets; ++i) sets[i].release();
      if (plan.batch == 1) die("out of GPU or pinned host memory even at 1 chunk per batch");
      plan.batch = std::max<uint32_t>(1, plan.batch / 2);
    }
    plan.batches = (int)((chunk_count + plan.batch - 1) / plan.batch);
    if (!in_ring.init(kInStages) || !writer.init(out, kOutStages)) die("out of pinned host memory");
    g_stats.batches = plan.batches;
    g_stats.sets = plan.sets;
    g_stats.batch_chunks = plan.batch;
  }

  // Debug aid for evaluating literal models offline: with
  // GZP_DUMP_LITS=<path>, appends each chunk's parsed literal stream to that
  // file as a u32 length followed by the bytes. Called between the encode
  // kernel and the compaction that overwrites scratch; it synchronises the
  // stream, so it serialises the pipeline and must not be used while
  // benchmarking.
  void dump_literals(CompressSet& s) {
    if (!lit_dump) return;
    check_cuda(cudaStreamSynchronize(s.stream), "dump literals sync");
    std::vector<uint32_t> n_lit(s.n);
    check_cuda(cudaMemcpy(n_lit.data(), s.d_rans_n_lit.p, s.n * sizeof(uint32_t), cudaMemcpyDeviceToHost),
               "dump n_lit");
    std::vector<uint8_t> buf;
    for (uint32_t c = 0; c < s.n; ++c) {
      uint32_t k = s.h_in_lens.p[c] <= 1 ? 0 : n_lit[c]; // in_len <= 1 chunks never ran the parse
      buf.resize(k);
      if (k) {
        const uint8_t* src = s.d_scratch.p + (size_t)c * scratch_bytes(chunk_size) + scratch_lits_offset(chunk_size);
        check_cuda(cudaMemcpy(buf.data(), src, k, cudaMemcpyDeviceToHost), "dump lits");
      }
      std::fwrite(&k, sizeof(k), 1, lit_dump);
      std::fwrite(buf.data(), 1, k, lit_dump);
    }
  }

  // Reads batch b's input into s and starts its upload.
  void upload(CompressSet& s, int b) {
    uint32_t n = std::min(plan.batch, chunk_count - next_chunk);
    s.n = n;
    s.first = next_chunk;
    s.batch = b;
    // h_in_lens is about to be rewritten: the previous batch's upload of it
    // (long since finished in practice) must be complete.
    if (s.used) check_cuda(cudaEventSynchronize(s.h2d_done), "wait for set");
    s.used = true;
    for (uint32_t c = 0; c < n; ++c) {
      uint64_t start = (uint64_t)(s.first + c) * chunk_size;
      s.h_in_lens.p[c] = (uint32_t)std::min<uint64_t>(chunk_size, total_size - start);
    }
    // Chunks are contiguous both in the file and in d_in (only the file's
    // last chunk can be short), so the whole batch is one read.
    uint64_t bytes = std::min<uint64_t>((uint64_t)n * chunk_size, total_size - (uint64_t)s.first * chunk_size);
    next_chunk += n;

    g_stats.mark(b, kH2d0, s.stream);
    in_ring.upload(in, s.d_in.p, bytes, s.stream, "short read on input file");
    check_cuda(cudaMemcpyAsync(s.d_in_lens.p, s.h_in_lens.p, n * sizeof(uint32_t), cudaMemcpyHostToDevice, s.stream),
               "H2D lens");
    check_cuda(cudaEventRecord(s.h2d_done, s.stream), "cudaEventRecord");
    g_stats.mark(b, kH2d1, s.stream);
  }

  void launch(CompressSet& s) {
    cudaStream_t st = s.stream;
    uint32_t n = s.n;
    g_stats.mark(s.batch, kK0, st);
    launch_compress(s.d_in.p, chunk_size, n, s.d_in_lens.p, s.d_slots.p, slot_stride, s.d_start.p, s.d_sizes.p,
                    s.d_scratch.p, s.d_htab.p, forced_lit_shift, s.rans_bufs(), st);
    check_cuda(cudaGetLastError(), "compress_kernel launch");
    dump_literals(s);
    // The encode kernel is done with scratch (same stream), so the packed
    // output reuses it; see compress_scratch_bytes.
    check_cuda(launch_compact(s.d_slots.p, slot_stride, s.d_start.p, s.d_sizes.p, n, s.d_offsets.p, s.d_scratch.p,
                              s.d_temp.p, temp_bytes, st),
               "compaction");
    g_stats.mark(s.batch, kK1, st);

    check_cuda(cudaMemcpyAsync(s.h_sizes.p, s.d_sizes.p, n * sizeof(uint32_t), cudaMemcpyDeviceToHost, st),
               "D2H sizes");
    check_cuda(cudaMemcpyAsync(s.h_offsets.p, s.d_offsets.p, (n + 1) * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, st),
               "D2H offsets");
    check_cuda(cudaMemcpyAsync(s.h_shift.p, s.d_rans_shift.p, sizeof(uint32_t), cudaMemcpyDeviceToHost, st),
               "D2H shift");
    check_cuda(cudaMemcpyAsync(s.h_q.p, s.d_rans_q.p, kMaxQuantBytes, cudaMemcpyDeviceToHost, st), "D2H q");
    check_cuda(cudaEventRecord(s.meta, st), "cudaEventRecord");
    s.d2h_pending = true;
  }

  // Waits for s's chunk sizes, records its entries and table group, and
  // queues its packed output for the writer. Batches must come through
  // here in order: the payload is written in the order it is queued.
  void enqueue_d2h(CompressSet& s) {
    check_cuda(cudaEventSynchronize(s.meta), "wait for chunk sizes");
    uint32_t total = s.h_offsets.p[s.n];
    for (uint32_t c = 0; c < s.n; ++c) {
      uint32_t csize = s.h_sizes.p[c];
      entries[s.first + c] = ChunkEntry{payload_offset, csize, s.h_in_lens.p[c]};
      payload_offset += csize;
    }
    TableGroup g{s.first, s.n, *s.h_shift.p};
    groups.push_back(g);
    tables.insert(tables.end(), s.h_q.p, s.h_q.p + group_quant_bytes(g));

    g_stats.mark(s.batch, kD2h0, s.stream);
    download(writer, s.d_scratch.p, total, s.stream, nullptr);
    g_stats.mark(s.batch, kD2h1, s.stream);
    s.d2h_pending = false;
  }

  void run() {
    for (int b = 0; b < plan.batches; ++b) {
      CompressSet& s = sets[b % plan.sets];
      // Only with a single set: its previous batch must be queued for
      // download before this one overwrites it. (Stream order then keeps
      // the download ahead of the new upload and kernels.)
      if (s.d2h_pending) enqueue_d2h(s);
      upload(s, b);
      launch(s);
      // Defer the previous batch's download until now: its sizes are
      // almost certainly ready, and waiting for them no longer idles the
      // GPU because batch b is already queued.
      if (b > 0) {
        CompressSet& prev = sets[(b - 1) % plan.sets];
        if (prev.d2h_pending) enqueue_d2h(prev);
      }
    }
    for (int b = std::max(0, plan.batches - plan.sets); b < plan.batches; ++b) {
      CompressSet& s = sets[b % plan.sets];
      if (s.d2h_pending && s.batch == b) enqueue_d2h(s);
    }
    writer.finish();
    if (groups.size() != (size_t)plan.batches) die("internal error: table group count mismatch");
  }
};

// GZP_FORCE_LIT_SHIFT=0|4|8 forces every batch's literal-context rule
// (rans_codes.h) instead of letting each batch pick; tests and tuning only.
int forced_lit_shift() {
  const char* f = std::getenv("GZP_FORCE_LIT_SHIFT");
  if (!f) return -1;
  uint32_t v = (uint32_t)std::strtoul(f, nullptr, 10);
  if (!lit_shift_valid(v)) die("GZP_FORCE_LIT_SHIFT must be 0, 4 or 8");
  return (int)v;
}

void compress(const std::string& in_path, const std::string& out_path, uint32_t chunk_size) {
  FILE* in = std::fopen(in_path.c_str(), "rb");
  if (!in) die("cannot open input: " + in_path);
  std::setvbuf(in, nullptr, _IOFBF, kStdioBuf);
  uint64_t total_size = file_size(in);

  uint32_t chunk_count = total_size == 0 ? 0 : (uint32_t)((total_size + chunk_size - 1) / chunk_size);

  FILE* out = std::fopen(out_path.c_str(), "wb");
  if (!out) die("cannot open output: " + out_path);
  std::setvbuf(out, nullptr, _IOFBF, kStdioBuf);

  // table_group_count is unknown until Compressor::allocate() picks a
  // batch size, so the header is written with a placeholder and patched
  // at the end, same as the ChunkEntry and TableGroup arrays below.
  FileHeader header{kMagic, kVersion, chunk_size, total_size, chunk_count, 0, 0};
  std::fwrite(&header, sizeof(header), 1, out);
  long entries_pos = ftell(out);
  std::vector<ChunkEntry> entries(chunk_count);
  std::fwrite(entries.data(), sizeof(ChunkEntry), (size_t)chunk_count, out);

  std::vector<TableGroup> groups;
  std::vector<uint8_t> tables;
  long groups_pos = ftell(out);

  if (chunk_count > 0) {
    Compressor cz;
    cz.in = in;
    cz.out = out;
    cz.chunk_size = chunk_size;
    cz.chunk_count = chunk_count;
    cz.slot_stride = worst_case_size(chunk_size);
    cz.total_size = total_size;
    cz.entries.swap(entries);
    cz.forced_lit_shift = forced_lit_shift();
    if (const char* p = std::getenv("GZP_DUMP_LITS")) {
      cz.lit_dump = std::fopen(p, "wb");
      if (!cz.lit_dump) die(std::string("cannot open GZP_DUMP_LITS file: ") + p);
    }

    double t = now_s();
    cz.allocate();
    g_stats.setup_s += now_s() - t;
    g_stats.start_clock(cz.sets[0].stream);

    // Now that allocate() has fixed the batch size, reserve space for the
    // group directory: one TableGroup per batch. The writer thread appends
    // the payload after it.
    header.table_group_count = (uint32_t)cz.plan.batches;
    groups.resize(header.table_group_count);
    std::fwrite(groups.data(), sizeof(TableGroup), groups.size(), out);

    cz.run();

    if (cz.lit_dump) std::fclose(cz.lit_dump);
    entries.swap(cz.entries);
    groups.swap(cz.groups);
    tables.swap(cz.tables);
    g_stats.bytes_in = total_size;
    g_stats.bytes_out = cz.payload_offset + sizeof(FileHeader) + entries.size() * sizeof(ChunkEntry) +
                        groups.size() * sizeof(TableGroup) + tables.size();
  }

  // The table section follows the payload (the writer thread is done, so
  // the stream position is the payload's end) and ends the file.
  header.tables_offset = (uint64_t)ftell(out);
  if (!tables.empty() && std::fwrite(tables.data(), 1, tables.size(), out) != tables.size()) die("write failed");

  std::fseek(out, 0, SEEK_SET);
  std::fwrite(&header, sizeof(header), 1, out);
  std::fseek(out, entries_pos, SEEK_SET);
  std::fwrite(entries.data(), sizeof(ChunkEntry), entries.size(), out);
  std::fseek(out, groups_pos, SEEK_SET);
  std::fwrite(groups.data(), sizeof(TableGroup), groups.size(), out);
  std::fclose(in);
  if (std::fclose(out) != 0) die("write failed");
}

// ---------------------------------------------------------------------------
// Decompress
// ---------------------------------------------------------------------------

struct DecompressSet {
  PinBuf<uint32_t> h_in_offsets, h_in_lens, h_out_lens, h_group_id;
  DevBuf<uint8_t> d_in, d_out, d_scratch;
  DevBuf<uint32_t> d_in_offsets, d_in_lens, d_out_lens, d_group_id, d_err;
  cudaStream_t stream = nullptr;
  cudaEvent_t h2d_done = nullptr;
  uint32_t n = 0, first = 0;
  int batch = -1;
  bool used = false, d2h_pending = false;

  ~DecompressSet() {
    if (h2d_done) cudaEventDestroy(h2d_done);
    if (stream) cudaStreamDestroy(stream);
  }
  bool alloc(uint32_t batch, uint32_t chunk_size, uint32_t slot_stride) {
    return h_in_offsets.alloc(batch) && h_in_lens.alloc(batch) && h_out_lens.alloc(batch) &&
           h_group_id.alloc(batch) && d_in.alloc((size_t)batch * slot_stride) &&
           d_out.alloc((size_t)batch * chunk_size) && d_scratch.alloc((size_t)batch * scratch_bytes(chunk_size)) &&
           d_in_offsets.alloc(batch) && d_in_lens.alloc(batch) && d_out_lens.alloc(batch) &&
           d_group_id.alloc(batch) && d_err.alloc(1);
  }
  void release() {
    h_in_offsets.release(); h_in_lens.release(); h_out_lens.release(); h_group_id.release();
    d_in.release(); d_out.release(); d_scratch.release();
    d_in_offsets.release(); d_in_lens.release(); d_out_lens.release(); d_group_id.release(); d_err.release();
  }
};

struct Decompressor {
  FILE* in;
  FILE* out;
  FileHeader header;
  uint32_t slot_stride;
  long payload_start;
  std::vector<ChunkEntry> entries;
  std::vector<TableGroup> groups;
  std::vector<uint8_t> tables;       // the file's table section: each group's quantised counts in order
  std::vector<uint32_t> chunk_group; // header.chunk_count entries, from `groups`
  DevBuf<uint8_t> d_group_tables;    // every group's expanded tables, see prepare_group_tables
  DevBuf<uint64_t> d_group_off;      // byte offset of group g's region in d_group_tables
  DevBuf<uint32_t> d_group_shift;    // group g's literal-context rule
  std::unique_ptr<DecompressSet[]> sets;
  PinBuf<uint32_t> h_err; // one flag per batch, checked by the writer before writing it
  Plan plan;
  InRing in_ring;
  Writer writer;
  uint32_t next_chunk = 0;

  // Builds chunk_group[] from groups[] and expands every group's table
  // once, up front, so any decode batch (its own boundaries chosen
  // independently of whatever batch size compression used) can look up
  // any chunk's table by a simple index. Must run before the batch loop.
  void prepare_group_tables(cudaStream_t stream) {
    chunk_group.assign(header.chunk_count, 0);
    for (uint32_t g = 0; g < groups.size(); ++g) {
      const TableGroup& tg = groups[g];
      for (uint32_t c = tg.start_chunk; c < tg.start_chunk + tg.chunk_count; ++c) chunk_group[c] = g;
    }
    if (groups.empty()) return;

    // Each group's quantised counts and expanded tables vary in size with
    // its literal-context rule, so both are located by offset.
    std::vector<uint64_t> q_off(groups.size()), table_off(groups.size());
    std::vector<uint32_t> shift(groups.size());
    uint64_t q_total = 0, table_bytes = 0;
    for (uint32_t g = 0; g < groups.size(); ++g) {
      q_off[g] = q_total;
      table_off[g] = table_bytes;
      shift[g] = groups[g].lit_ctx_shift;
      q_total += group_quant_bytes(groups[g]);
      table_bytes += rans_group_table_bytes(shift[g]);
    }
    DevBuf<uint8_t> d_q;
    DevBuf<uint64_t> d_q_off;
    if (!d_q.alloc(tables.size()) || !d_q_off.alloc(groups.size()) || !d_group_off.alloc(groups.size()) ||
        !d_group_shift.alloc(groups.size()) || !d_group_tables.alloc(table_bytes)) {
      die("out of GPU memory expanding table groups");
    }
    check_cuda(cudaMemcpyAsync(d_q.p, tables.data(), tables.size(), cudaMemcpyHostToDevice, stream), "H2D tables");
    check_cuda(cudaMemcpyAsync(d_q_off.p, q_off.data(), q_off.size() * sizeof(uint64_t), cudaMemcpyHostToDevice,
                               stream),
               "H2D q offsets");
    check_cuda(cudaMemcpyAsync(d_group_off.p, table_off.data(), table_off.size() * sizeof(uint64_t),
                               cudaMemcpyHostToDevice, stream),
               "H2D table offsets");
    check_cuda(cudaMemcpyAsync(d_group_shift.p, shift.data(), shift.size() * sizeof(uint32_t), cudaMemcpyHostToDevice,
                               stream),
               "H2D shifts");
    // Unused contexts and groups with no LzRans chunks have all-zero
    // counts, which leave their tables untouched; zeroing first makes those
    // zero frequencies (what flags a corrupt chunk using one) rather than
    // whatever cudaMalloc happened to hand back.
    check_cuda(cudaMemsetAsync(d_group_tables.p, 0, table_bytes, stream), "zero group tables");
    launch_expand_group_tables(d_q.p, d_q_off.p, d_group_off.p, d_group_shift.p, (uint32_t)groups.size(),
                               d_group_tables.p, stream);
    check_cuda(cudaGetLastError(), "expand_group_tables launch");
    check_cuda(cudaStreamSynchronize(stream), "expand_group_tables sync");
  }

  void allocate() {
    size_t dev_per_chunk =
        (size_t)header.chunk_size + (size_t)slot_stride + scratch_bytes(header.chunk_size) + 4 * sizeof(uint32_t);
    plan = plan_batches(header.chunk_count, header.chunk_size, dev_per_chunk);
    sets.reset(new DecompressSet[plan.sets]);
    for (int i = 0; i < plan.sets; ++i) create_stream(sets[i].stream, {&sets[i].h2d_done});
    for (;;) {
      bool ok = true;
      for (int i = 0; i < plan.sets && ok; ++i) ok = sets[i].alloc(plan.batch, header.chunk_size, slot_stride);
      if (ok) break;
      for (int i = 0; i < plan.sets; ++i) sets[i].release();
      if (plan.batch == 1) die("out of GPU or pinned host memory even at 1 chunk per batch");
      plan.batch = std::max<uint32_t>(1, plan.batch / 2);
    }
    plan.batches = (int)((header.chunk_count + plan.batch - 1) / plan.batch);
    if (!h_err.alloc(plan.batches) || !in_ring.init(kInStages) || !writer.init(out, kOutStages)) {
      die("out of pinned host memory");
    }
    g_stats.batches = plan.batches;
    g_stats.sets = plan.sets;
    g_stats.batch_chunks = plan.batch;
  }

  void upload(DecompressSet& s, int b) {
    uint32_t n = std::min(plan.batch, header.chunk_count - next_chunk);
    s.n = n;
    s.first = next_chunk;
    s.batch = b;
    if (s.used) check_cuda(cudaEventSynchronize(s.h2d_done), "wait for set");
    s.used = true;

    // The writer lays chunks out back to back in file order, so a batch's
    // payload is one contiguous range we can read in one pass.
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
      s.h_group_id.p[c] = chunk_group.empty() ? 0 : chunk_group[s.first + c];
    }
    uint64_t total_in = expect - base;
    next_chunk += n;

    cudaStream_t st = s.stream;
    g_stats.mark(b, kH2d0, st);
    if (std::fseek(in, payload_start + (long)base, SEEK_SET) != 0) die("seek failed");
    in_ring.upload(in, s.d_in.p, total_in, st, "short read on compressed payload");
    check_cuda(cudaMemcpyAsync(s.d_in_offsets.p, s.h_in_offsets.p, n * sizeof(uint32_t), cudaMemcpyHostToDevice, st),
               "H2D offsets");
    check_cuda(cudaMemcpyAsync(s.d_in_lens.p, s.h_in_lens.p, n * sizeof(uint32_t), cudaMemcpyHostToDevice, st),
               "H2D lens");
    check_cuda(cudaMemcpyAsync(s.d_out_lens.p, s.h_out_lens.p, n * sizeof(uint32_t), cudaMemcpyHostToDevice, st),
               "H2D out_lens");
    check_cuda(cudaMemcpyAsync(s.d_group_id.p, s.h_group_id.p, n * sizeof(uint32_t), cudaMemcpyHostToDevice, st),
               "H2D group_id");
    check_cuda(cudaEventRecord(s.h2d_done, st), "cudaEventRecord");
    g_stats.mark(b, kH2d1, st);
  }

  void launch(DecompressSet& s) {
    cudaStream_t st = s.stream;
    check_cuda(cudaMemsetAsync(s.d_err.p, 0, sizeof(uint32_t), st), "clear error flag");
    g_stats.mark(s.batch, kK0, st);
    launch_decompress(s.d_in.p, s.d_in_offsets.p, s.n, s.d_in_lens.p, s.d_out.p, header.chunk_size,
                      s.d_out_lens.p, s.d_scratch.p, d_group_tables.p, d_group_off.p, d_group_shift.p,
                      s.d_group_id.p, s.d_err.p, st);
    check_cuda(cudaGetLastError(), "decompress_kernel launch");
    g_stats.mark(s.batch, kK1, st);
    // Downloaded ahead of the output on the same stream, so the writer can
    // check it before writing any of this batch.
    check_cuda(cudaMemcpyAsync(h_err.p + s.batch, s.d_err.p, sizeof(uint32_t), cudaMemcpyDeviceToHost, st),
               "D2H err");
    s.d2h_pending = true;
  }

  // Queues s's output for the writer; batches must come through in order.
  void enqueue_d2h(DecompressSet& s) {
    uint64_t out_bytes = (uint64_t)(s.n - 1) * header.chunk_size + s.h_out_lens.p[s.n - 1];
    g_stats.mark(s.batch, kD2h0, s.stream);
    download(writer, s.d_out.p, out_bytes, s.stream, h_err.p + s.batch);
    g_stats.mark(s.batch, kD2h1, s.stream);
    s.d2h_pending = false;
  }

  // Same shape as Compressor::run: batch b-1's output is queued after batch
  // b is launched, so the writer drains it while b's kernels run.
  void run() {
    for (int b = 0; b < plan.batches; ++b) {
      DecompressSet& s = sets[b % plan.sets];
      if (s.d2h_pending) enqueue_d2h(s);
      upload(s, b);
      launch(s);
      if (b > 0) {
        DecompressSet& prev = sets[(b - 1) % plan.sets];
        if (prev.d2h_pending) enqueue_d2h(prev);
      }
    }
    for (int b = std::max(0, plan.batches - plan.sets); b < plan.batches; ++b) {
      DecompressSet& s = sets[b % plan.sets];
      if (s.d2h_pending && s.batch == b) enqueue_d2h(s);
    }
    writer.finish();
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

  std::vector<TableGroup> groups(header.table_group_count);
  if (header.table_group_count > 0 &&
      std::fread(groups.data(), sizeof(TableGroup), header.table_group_count, in) != header.table_group_count) {
    die("truncated table group directory");
  }
  // Groups must cover [0, chunk_count) contiguously and in order: a
  // corrupt/adversarial directory here would otherwise let a chunk's
  // group_id land outside chunk_group[] or reference an out-of-range slot.
  // An empty group list only ever occurs for an empty file (chunk_count
  // == 0), which is fine as-is: no chunk will ever look one up.
  if (!groups.empty()) {
    uint64_t covered = 0;
    for (const TableGroup& g : groups) {
      if (g.start_chunk != covered || g.chunk_count == 0) die("corrupt table group directory: not contiguous");
      if (!lit_shift_valid(g.lit_ctx_shift)) die("corrupt table group directory: bad lit_ctx_shift");
      covered += g.chunk_count;
    }
    if (covered != header.chunk_count) die("corrupt table group directory: doesn't cover all chunks");
  }

  long payload_start = ftell(in);

  // The payload runs from here to the table section, which ends the file.
  // Checking that the chunk table covers exactly that range keeps any batch
  // read inside the payload.
  uint64_t payload_bytes = entries.empty() ? 0 : entries.back().offset + entries.back().compressed_size;
  uint64_t tables_bytes = 0;
  for (const TableGroup& g : groups) tables_bytes += group_quant_bytes(g);
  if (header.tables_offset != (uint64_t)payload_start + payload_bytes ||
      header.tables_offset + tables_bytes != file_size(in)) {
    die("corrupt header: payload and table section don't match the file");
  }
  std::vector<uint8_t> tables(tables_bytes);
  if (std::fseek(in, (long)header.tables_offset, SEEK_SET) != 0 ||
      (tables_bytes && std::fread(tables.data(), 1, tables_bytes, in) != tables_bytes)) {
    die("truncated table section");
  }

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
    dz.groups.swap(groups);
    dz.tables.swap(tables);

    double t = now_s();
    dz.allocate();
    dz.prepare_group_tables(dz.sets[0].stream);
    g_stats.setup_s += now_s() - t;
    g_stats.start_clock(dz.sets[0].stream);

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
               "  gzp c <input> <output> [chunk_size | --profile speed|balance|ratio] [--gpu-mem SIZE]\n"
               "  gzp d <input> <output> [--gpu-mem SIZE]\n"
               "chunk_size and --profile are mutually exclusive; with neither, chunk_size is %u.\n"
               "--gpu-mem caps the GPU memory used for batch buffers (e.g. 8G, 512M; a bare\n"
               "number is MiB). Default: half the free GPU memory, at most 4G. Also GZP_GPU_MEM.\n",
               kDefaultChunkSize);
  std::exit(1);
}

// Parses a --gpu-mem / GZP_GPU_MEM value: a positive number with an
// optional K, M, G or T suffix (binary units; a trailing "B" or "iB" is
// allowed). A bare number is MiB.
size_t parse_mem_size(const std::string& s) {
  char* end = nullptr;
  double v = std::strtod(s.c_str(), &end);
  if (end == s.c_str() || !(v > 0)) die("bad --gpu-mem value: " + s);
  std::string unit(end);
  for (char& ch : unit) ch = (char)std::tolower((unsigned char)ch);
  if (unit.size() >= 2 && unit.compare(unit.size() - 2, 2, "ib") == 0) unit.resize(unit.size() - 2);
  else if (!unit.empty() && unit.back() == 'b') unit.pop_back();
  double mult = unit == "" || unit == "m" ? 1048576.0
                : unit == "k"             ? 1024.0
                : unit == "g"             ? 1073741824.0
                : unit == "t"             ? 1099511627776.0
                                          : 0.0;
  if (mult == 0.0) die("bad --gpu-mem unit in: " + s + " (use K, M, G or T)");
  double bytes = v * mult;
  if (bytes < 1048576.0) die("--gpu-mem must be at least 1M");
  return (size_t)bytes;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 4) usage();
  std::string mode = argv[1];
  std::string in_path = argv[2];
  std::string out_path = argv[3];

  auto t0 = std::chrono::steady_clock::now();
  if (const char* m = std::getenv("GZP_GPU_MEM")) g_gpu_mem = parse_mem_size(m);
  // --gpu-mem applies to both modes (and wins over GZP_GPU_MEM); the rest
  // of the arguments are mode-specific.
  std::vector<std::string> rest;
  for (int i = 4; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--gpu-mem") {
      if (i + 1 >= argc) die("--gpu-mem needs a value (e.g. 8G)");
      g_gpu_mem = parse_mem_size(argv[++i]);
    } else {
      rest.push_back(a);
    }
  }
  if (mode == "c") {
    bool have_chunk_size = false, have_profile = false;
    uint32_t chunk_size = kDefaultChunkSize;
    for (size_t i = 0; i < rest.size(); ++i) {
      const std::string& a = rest[i];
      if (a == "--profile") {
        if (i + 1 >= rest.size()) die("--profile needs a value (speed, balance, or ratio)");
        std::string p = rest[++i];
        if (p == "speed") chunk_size = kProfileSpeedChunkSize;
        else if (p == "balance") chunk_size = kProfileBalanceChunkSize;
        else if (p == "ratio") chunk_size = kProfileRatioChunkSize;
        else die("unknown --profile: " + p + " (expected speed, balance, or ratio)");
        if (have_profile) die("--profile given more than once");
        have_profile = true;
      } else {
        char* end = nullptr;
        unsigned long v = std::strtoul(a.c_str(), &end, 10);
        if (a.empty() || a[0] == '-' || *end != '\0') die("unrecognised argument: " + a);
        if (have_chunk_size) die("chunk_size given more than once");
        chunk_size = v > kMaxChunkSize ? kMaxChunkSize + 1 : (uint32_t)v;
        have_chunk_size = true;
      }
    }
    if (have_chunk_size && have_profile) die("specify chunk_size or --profile, not both");
    if (chunk_size == 0 || chunk_size > kMaxChunkSize) {
      die("chunk_size must be in (0, " + std::to_string(kMaxChunkSize) + "]");
    }
    compress(in_path, out_path, chunk_size);
  } else if (mode == "d") {
    if (!rest.empty()) die("unrecognised argument: " + rest[0]);
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
