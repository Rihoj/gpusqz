// gpusqz: a small GPU-accelerated file compressor.
//
// Design: the input is split into fixed-size, independent chunks. Each
// chunk is LZ-compressed (or stored raw if that doesn't help) by one group
// of 32 GPU lanes. See docs/design.md for the format and the tradeoffs.
//
// Host side, batches of chunks live only in device memory, in a ring of
// kSets buffer sets with a stream each. File data moves through a small,
// fixed pool of staging buffers: the main thread freads into an input stage
// and uploads it asynchronously, and a writer thread drains the output
// stages the main thread fills with asynchronous downloads. So batch i+1's
// read and upload overlap batch i's kernels, and batch i-1's download and
// file write overlap both. The GPU work itself is behind backend.h.
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

#include "backend.h"
#include "file_io.h"
#include "format.h"

using namespace gpusqz;

namespace {

// Device buffer sets in the ring. A third set would only let uploads and
// downloads overlap each other, which PCIe doesn't need, at the cost of a
// third of each batch's size.
constexpr int kSets = 2;

// Batch sizing (see plan_batches). A chunk's LZ parse is latency-bound at a
// few MB/s, so kernel throughput follows the chunks in flight: a batch
// wants at least kMinBatchChunks chunks and kMinBatchBytes of input when
// the file and the memory budget allow.
constexpr uint32_t kTargetBatches = 8;
constexpr uint32_t kMinBatchChunks = 1024;
constexpr size_t kMinBatchBytes = 32u << 20;
// GPU memory for batch buffers: by default kDefaultBudgetPercent of the
// memory free at startup; --gpu-mem / GPUSQZ_GPU_MEM set an explicit budget,
// up to all but kGpuMemReserve of it. It is allocated once and held.
constexpr size_t kDefaultBudgetPercent = 80;
constexpr size_t kGpuMemReserve = 256ull << 20;
size_t g_gpu_mem = 0; // explicit budget in bytes, 0 = automatic

// Staging, fixed in size whatever the batch: pinning host memory is slow
// under WSL2 (~0.3-0.4s per GB), unlike device allocations.
constexpr size_t kStageBytes = 8u << 20;
constexpr int kInStages = 4;
constexpr int kOutStages = 8;
constexpr size_t kStdioBuf = 4u << 20;

// Opened on first use (open_backend), so an empty file needs no GPU.
Backend* g_backend = nullptr;
std::string g_backend_name = "auto";

[[noreturn]] void die(const std::string& msg) {
  std::fprintf(stderr, "gpusqz: %s\n", msg.c_str());
  std::exit(1);
}

void open_backend();

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// GPUSQZ_VERBOSE=1 prints per-stage timing: whether a run is bound by file
// I/O, PCIe copies or the kernels. GPU stages are timed with per-batch
// events, so stages overlap and their sums can exceed wall time.
enum Mark { kH2d0, kH2d1, kK0, kK1, kD2h0, kD2h1, kMarks };

struct Stats {
  bool enabled = [] {
    const char* v = std::getenv("GPUSQZ_VERBOSE");
    return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
  }();
  double setup_s = 0, fread_s = 0, fwrite_s = 0, in_wait_s = 0, out_wait_s = 0;
  uint64_t h2d_bytes = 0, d2h_bytes = 0, bytes_in = 0, bytes_out = 0;
  int batches = 0, sets = 0;
  uint32_t batch_chunks = 0;
  // `origin` is recorded before any batch is queued; each batch's marks are
  // measured against it so overlapping kernel spans can be unioned.
  std::unique_ptr<Event> origin;
  std::vector<std::array<std::unique_ptr<Event>, kMarks>> marks;

  void start_clock(Stream& stream) {
    if (!enabled) return;
    origin = g_backend->create_event(true);
    g_backend->record(*origin, stream);
  }
  void mark(int batch, Mark m, Stream& stream) {
    if (!enabled) return;
    if ((size_t)batch >= marks.size()) marks.resize(batch + 1);
    auto& e = marks[batch][m];
    if (!e) e = g_backend->create_event(true);
    g_backend->record(*e, stream);
  }
  // Call once all GPU work has finished.
  void report(const char* mode, double wall_s) {
    if (!enabled || !g_backend) return;
    std::string why;
    for (auto& m : marks) {
      for (auto& e : m) {
        if (e && !g_backend->wait(*e, &why)) die("GPU work failed: " + why);
      }
    }
    double h2d_s = 0, kernel_s = 0, d2h_s = 0;
    std::vector<std::pair<double, double>> spans;
    auto at = [&](Event& e) { return g_backend->elapsed_ms(*origin, e); };
    for (auto& m : marks) {
      if (m[kH2d0] && m[kH2d1]) h2d_s += (at(*m[kH2d1]) - at(*m[kH2d0])) / 1000.0;
      if (m[kK0] && m[kK1]) {
        spans.emplace_back(at(*m[kK0]), at(*m[kK1]));
        kernel_s += (spans.back().second - spans.back().first) / 1000.0;
      }
      if (m[kD2h0] && m[kD2h1]) d2h_s += (at(*m[kD2h1]) - at(*m[kD2h0])) / 1000.0;
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
                 "gpusqz[%s] %s\n"
                 "  in=%llu out=%llu wall=%.3fs (%.1f MB/s)  batches=%d x %u chunks, %d buffer sets\n"
                 "  setup  %.3fs  (GPU context + buffer allocation, fixed cost)\n"
                 "  steady %.3fs  (%.1f MB/s of input: wall minus setup)\n"
                 "  fread  %.3fs  (%.1f MB/s of input; main thread)\n"
                 "  h2d    %.3fs  (%.1f MB/s, %llu bytes)\n"
                 "  kernel %.3fs  (%.1f MB/s of input; sum of per-batch spans)\n"
                 "  kbusy  %.3fs  (%.1f MB/s of input; wall-clock time any kernel ran)\n"
                 "  d2h    %.3fs  (%.1f MB/s, %llu bytes)\n"
                 "  fwrite %.3fs  (%.1f MB/s of output; writer thread)\n"
                 "  stall  %.3fs waiting for a free input stage, %.3fs for a free output stage\n",
                 mode, g_backend->name().c_str(), (unsigned long long)bytes_in, (unsigned long long)bytes_out,
                 wall_s, mbps(bytes_in, wall_s), batches, batch_chunks, sets, setup_s, wall_s - setup_s,
                 mbps(bytes_in, wall_s - setup_s), fread_s, mbps(bytes_in, fread_s), h2d_s,
                 mbps(h2d_bytes, h2d_s), (unsigned long long)h2d_bytes, kernel_s, mbps(bytes_in, kernel_s), busy,
                 mbps(bytes_in, busy), d2h_s, mbps(d2h_bytes, d2h_s), (unsigned long long)d2h_bytes, fwrite_s,
                 mbps(bytes_out, fwrite_s), in_wait_s, out_wait_s);
  }
};
Stats g_stats;

// ---------------------------------------------------------------------------
// Staging
// ---------------------------------------------------------------------------

// One staging buffer plus the event of the last async copy using it.
struct Stage {
  std::unique_ptr<HostBuf> buf;
  std::unique_ptr<Event> ev;
  bool pending = false; // ev recorded and not yet waited on (input stages only)

  bool init() {
    buf = g_backend->alloc_host(kStageBytes);
    if (!buf) return false;
    ev = g_backend->create_event(false);
    return true;
  }
};

void wait_or_die(Event& e, const char* what) {
  std::string why;
  if (!g_backend->wait(e, &why)) die(std::string(what) + ": " + why);
}

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

  // Copies the next `bytes` of `in` to the set's input from byte 0.
  template <typename Set>
  void upload(FILE* in, Set& set, uint64_t bytes, const char* short_read_msg) {
    for (uint64_t off = 0; off < bytes; off += kStageBytes) {
      size_t len = (size_t)std::min<uint64_t>(kStageBytes, bytes - off);
      Stage& s = st[next];
      next = (next + 1) % n;
      if (s.pending) {
        double t = now_s();
        wait_or_die(*s.ev, "wait for input stage");
        g_stats.in_wait_s += now_s() - t;
        s.pending = false;
      }
      double t = now_s();
      if (std::fread(s.buf->p, 1, len, in) != len) die(short_read_msg);
      g_stats.fread_s += now_s() - t;
      set.upload_input(off, *s.buf, len);
      g_backend->record(*s.ev, set.stream());
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
  // null, is a host flag downloaded before that copy on the same stream:
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
      std::string why, err;
      if (!g_backend->wait(*st_[j.idx].ev, &why)) {
        err = "wait for output stage: " + why;
      } else if (j.err && *(const volatile uint32_t*)j.err) {
        err = "corrupt input: malformed chunk data";
      } else {
        double t = now_s();
        bool ok = std::fwrite(st_[j.idx].buf->p, 1, j.len, out_) == j.len;
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

// Downloads the set's output [0, bytes) through the writer's stages.
template <typename Set>
void download(Writer& w, Set& set, uint64_t bytes, const uint32_t* err) {
  for (uint64_t off = 0; off < bytes; off += kStageBytes) {
    size_t len = (size_t)std::min<uint64_t>(kStageBytes, bytes - off);
    int idx;
    Stage& s = w.acquire(idx);
    set.download_output(off, *s.buf, len);
    g_backend->record(*s.ev, set.stream());
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

// Sizes batches from free GPU memory (the GPU may be shared) and the file's
// chunk count; callers halve the batch if allocation still fails. A file
// that fits in one batch gets the whole budget as a single set: batches on
// different streams barely overlap on the GPU, so a bigger batch beats a
// second set.
Plan plan_batches(uint32_t chunk_count, uint32_t chunk_size, size_t dev_bytes_per_chunk) {
  size_t free_bytes = g_backend->free_memory();
  size_t budget = free_bytes / 100 * kDefaultBudgetPercent;
  if (g_gpu_mem) {
    size_t avail = free_bytes > kGpuMemReserve ? free_bytes - kGpuMemReserve : 0;
    budget = std::min(g_gpu_mem, avail);
    if (budget < g_gpu_mem) {
      std::fprintf(stderr, "gpusqz: --gpu-mem %zu MiB is more than the %zu MiB free; using %zu MiB\n",
                   g_gpu_mem >> 20, free_bytes >> 20, budget >> 20);
    }
  }

  uint32_t min_batch = (uint32_t)std::max<size_t>(1, kMinBatchBytes / chunk_size);
  uint32_t want = std::max({min_batch, kMinBatchChunks, (chunk_count + kTargetBatches - 1) / kTargetBatches});

  // Tests only: lets compress and decompress use different batch sizes, so
  // decode batches straddle TableGroup boundaries.
  if (const char* f = std::getenv("GPUSQZ_FORCE_BATCH")) {
    uint32_t forced = (uint32_t)std::strtoul(f, nullptr, 10);
    if (forced >= 1) want = forced;
  }
  want = std::min(want, std::max<uint32_t>(1, g_backend->max_batch(chunk_size)));

  Plan p;
  for (int s = 1; s <= kSets; ++s) {
    uint32_t mem_max = (uint32_t)std::max<size_t>(1, budget / ((size_t)s * dev_bytes_per_chunk));
    p.batch = std::min({want, mem_max, chunk_count});
    p.batches = (int)((chunk_count + p.batch - 1) / p.batch);
    p.sets = std::min(s, p.batches);
    if (p.batches <= s) break;
  }
  return p;
}

// Creates plan.sets buffer sets, halving the batch until they fit.
template <typename Set, typename Create>
std::vector<std::unique_ptr<Set>> create_sets(Plan& plan, uint32_t chunk_count, Create create) {
  std::vector<std::unique_ptr<Set>> sets;
  for (;;) {
    sets.clear();
    for (int i = 0; i < plan.sets; ++i) {
      auto s = create(plan.batch);
      if (!s) break;
      sets.push_back(std::move(s));
    }
    if ((int)sets.size() == plan.sets) break;
    sets.clear();
    if (plan.batch == 1) die("out of GPU or pinned host memory even at 1 chunk per batch");
    plan.batch = std::max<uint32_t>(1, plan.batch / 2);
  }
  plan.batches = (int)((chunk_count + plan.batch - 1) / plan.batch);
  g_stats.batches = plan.batches;
  g_stats.sets = plan.sets;
  g_stats.batch_chunks = plan.batch;
  return sets;
}

uint64_t file_size(FILE* f) {
  uint64_t cur = file_tell(f);
  file_seek(f, 0, SEEK_END);
  uint64_t sz = file_tell(f);
  file_seek(f, cur);
  return sz;
}

// ---------------------------------------------------------------------------
// Compress
// ---------------------------------------------------------------------------

struct CompressSlot {
  std::unique_ptr<CompressSet> set;
  uint32_t n = 0, first = 0;
  int batch = -1;
  bool d2h_pending = false;
};

struct Compressor {
  FILE* in;
  FILE* out;
  uint32_t chunk_size, chunk_count;
  uint64_t total_size;
  std::vector<ChunkEntry> entries;
  std::vector<TableGroup> groups; // one per batch
  std::vector<uint8_t> tables;     // each group's quantised counts, in group order
  int forced_lit_shift = -1;       // GPUSQZ_FORCE_LIT_SHIFT, tests and tuning only
  std::vector<CompressSlot> sets;
  Plan plan;
  InRing in_ring;
  Writer writer;
  uint64_t payload_offset = 0;
  uint32_t next_chunk = 0;

  void allocate() {
    plan = plan_batches(chunk_count, chunk_size, g_backend->compress_bytes_per_chunk(chunk_size));
    auto made = create_sets<CompressSet>(
        plan, chunk_count, [&](uint32_t batch) { return g_backend->create_compress_set(batch, chunk_size); });
    sets.resize(made.size());
    for (size_t i = 0; i < made.size(); ++i) sets[i].set = std::move(made[i]);
    if (!in_ring.init(kInStages) || !writer.init(out, kOutStages)) die("out of pinned host memory");
  }

  // Reads batch b's input into s and starts its upload.
  void upload(CompressSlot& s, int b) {
    uint32_t n = std::min(plan.batch, chunk_count - next_chunk);
    s.n = n;
    s.first = next_chunk;
    s.batch = b;
    uint32_t* lens = s.set->begin(n);
    for (uint32_t c = 0; c < n; ++c) {
      uint64_t start = (uint64_t)(s.first + c) * chunk_size;
      lens[c] = (uint32_t)std::min<uint64_t>(chunk_size, total_size - start);
    }
    // Chunks are contiguous both in the file and in the set's input (only
    // the file's last chunk can be short), so the whole batch is one read.
    uint64_t bytes = std::min<uint64_t>((uint64_t)n * chunk_size, total_size - (uint64_t)s.first * chunk_size);
    next_chunk += n;

    g_stats.mark(b, kH2d0, s.set->stream());
    in_ring.upload(in, *s.set, bytes, "short read on input file");
    g_stats.mark(b, kH2d1, s.set->stream());
  }

  void launch(CompressSlot& s) {
    g_stats.mark(s.batch, kK0, s.set->stream());
    s.set->launch(forced_lit_shift);
    g_stats.mark(s.batch, kK1, s.set->stream());
    s.d2h_pending = true;
  }

  // Waits for s's chunk sizes, records its entries and table group, and
  // queues its packed output for the writer. Batches must come through
  // here in order: the payload is written in the order it is queued.
  void enqueue_d2h(CompressSlot& s) {
    CompressSet& set = *s.set;
    set.wait_meta();
    const uint32_t* sizes = set.sizes();
    for (uint32_t c = 0; c < s.n; ++c) {
      uint64_t start = (uint64_t)(s.first + c) * chunk_size;
      uint32_t len = (uint32_t)std::min<uint64_t>(chunk_size, total_size - start);
      entries[s.first + c] = ChunkEntry{payload_offset, sizes[c], len};
      payload_offset += sizes[c];
    }
    TableGroup g{s.first, s.n, set.lit_shift()};
    if (!lit_shift_valid(g.lit_ctx_shift)) die("internal error: bad literal-context rule from the GPU");
    groups.push_back(g);
    tables.insert(tables.end(), set.quant(), set.quant() + group_quant_bytes(g));

    g_stats.mark(s.batch, kD2h0, set.stream());
    download(writer, set, set.packed_bytes(), nullptr);
    g_stats.mark(s.batch, kD2h1, set.stream());
    s.d2h_pending = false;
  }

  void run() {
    for (int b = 0; b < plan.batches; ++b) {
      CompressSlot& s = sets[b % plan.sets];
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
        CompressSlot& prev = sets[(b - 1) % plan.sets];
        if (prev.d2h_pending) enqueue_d2h(prev);
      }
    }
    for (int b = std::max(0, plan.batches - plan.sets); b < plan.batches; ++b) {
      CompressSlot& s = sets[b % plan.sets];
      if (s.d2h_pending && s.batch == b) enqueue_d2h(s);
    }
    writer.finish();
    if (groups.size() != (size_t)plan.batches) die("internal error: table group count mismatch");
  }
};

// GPUSQZ_FORCE_LIT_SHIFT=0|4|8 forces every batch's literal-context rule
// (rans_codes.h) instead of letting each batch pick; tests and tuning only.
int forced_lit_shift() {
  const char* f = std::getenv("GPUSQZ_FORCE_LIT_SHIFT");
  if (!f) return -1;
  uint32_t v = (uint32_t)std::strtoul(f, nullptr, 10);
  if (!lit_shift_valid(v)) die("GPUSQZ_FORCE_LIT_SHIFT must be 0, 4 or 8");
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
  uint64_t entries_pos = file_tell(out);
  std::vector<ChunkEntry> entries(chunk_count);
  std::fwrite(entries.data(), sizeof(ChunkEntry), (size_t)chunk_count, out);

  std::vector<TableGroup> groups;
  std::vector<uint8_t> tables;
  uint64_t groups_pos = file_tell(out);

  if (chunk_count > 0) {
    Compressor cz;
    cz.in = in;
    cz.out = out;
    cz.chunk_size = chunk_size;
    cz.chunk_count = chunk_count;
    cz.total_size = total_size;
    cz.entries.swap(entries);
    cz.forced_lit_shift = forced_lit_shift();

    double t = now_s();
    open_backend();
    cz.allocate();
    g_stats.setup_s += now_s() - t;
    g_stats.start_clock(cz.sets[0].set->stream());

    // Now that allocate() has fixed the batch size, reserve space for the
    // group directory: one TableGroup per batch. The writer thread appends
    // the payload after it.
    header.table_group_count = (uint32_t)cz.plan.batches;
    groups.resize(header.table_group_count);
    std::fwrite(groups.data(), sizeof(TableGroup), groups.size(), out);

    cz.run();

    entries.swap(cz.entries);
    groups.swap(cz.groups);
    tables.swap(cz.tables);
    g_stats.bytes_in = total_size;
    g_stats.bytes_out = cz.payload_offset + sizeof(FileHeader) + entries.size() * sizeof(ChunkEntry) +
                        groups.size() * sizeof(TableGroup) + tables.size();
  }

  // The table section follows the payload (the writer thread is done, so
  // the stream position is the payload's end) and ends the file.
  header.tables_offset = file_tell(out);
  if (!tables.empty() && std::fwrite(tables.data(), 1, tables.size(), out) != tables.size()) die("write failed");

  file_seek(out, 0);
  std::fwrite(&header, sizeof(header), 1, out);
  file_seek(out, entries_pos);
  std::fwrite(entries.data(), sizeof(ChunkEntry), entries.size(), out);
  file_seek(out, groups_pos);
  std::fwrite(groups.data(), sizeof(TableGroup), groups.size(), out);
  std::fclose(in);
  if (std::fclose(out) != 0) die("write failed");
}

// ---------------------------------------------------------------------------
// Decompress
// ---------------------------------------------------------------------------

struct DecompressSlot {
  std::unique_ptr<DecompressSet> set;
  uint32_t n = 0, first = 0, last_len = 0;
  int batch = -1;
  bool d2h_pending = false;
};

struct Decompressor {
  FILE* in;
  FILE* out;
  FileHeader header;
  uint32_t slot_stride;
  uint64_t payload_start;
  std::vector<ChunkEntry> entries;
  std::vector<TableGroup> groups;
  std::vector<uint8_t> tables;       // the file's table section: each group's quantised counts in order
  std::vector<uint32_t> chunk_group; // header.chunk_count entries, from `groups`
  std::unique_ptr<GroupTables> group_tables;
  std::vector<DecompressSlot> sets;
  std::unique_ptr<HostBuf> h_err; // one flag per batch, checked by the writer before writing it
  Plan plan;
  InRing in_ring;
  Writer writer;
  uint32_t next_chunk = 0;

  // Builds chunk_group[] from groups[] and expands every group's table
  // once, up front, so any decode batch (its own boundaries chosen
  // independently of whatever batch size compression used) can look up
  // any chunk's table by a simple index. Must run before the batch loop.
  void prepare_group_tables() {
    chunk_group.assign(header.chunk_count, 0);
    for (uint32_t g = 0; g < groups.size(); ++g) {
      const TableGroup& tg = groups[g];
      for (uint32_t c = tg.start_chunk; c < tg.start_chunk + tg.chunk_count; ++c) chunk_group[c] = g;
    }
    group_tables = g_backend->load_group_tables(groups, tables);
  }

  void allocate() {
    plan = plan_batches(header.chunk_count, header.chunk_size,
                        g_backend->decompress_bytes_per_chunk(header.chunk_size));
    auto made = create_sets<DecompressSet>(plan, header.chunk_count, [&](uint32_t batch) {
      return g_backend->create_decompress_set(batch, header.chunk_size);
    });
    sets.resize(made.size());
    for (size_t i = 0; i < made.size(); ++i) sets[i].set = std::move(made[i]);
    h_err = g_backend->alloc_host((size_t)plan.batches * sizeof(uint32_t));
    if (!h_err || !in_ring.init(kInStages) || !writer.init(out, kOutStages)) die("out of pinned host memory");
  }

  void upload(DecompressSlot& s, int b) {
    uint32_t n = std::min(plan.batch, header.chunk_count - next_chunk);
    s.n = n;
    s.first = next_chunk;
    s.batch = b;
    DecompressSet::Inputs h = s.set->begin(n);

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
      h.in_offsets[c] = (uint32_t)(e.offset - base);
      h.in_lens[c] = e.compressed_size;
      h.out_lens[c] = e.original_size;
      h.group_id[c] = chunk_group.empty() ? 0 : chunk_group[s.first + c];
    }
    s.last_len = entries[s.first + n - 1].original_size;
    uint64_t total_in = expect - base;
    next_chunk += n;

    g_stats.mark(b, kH2d0, s.set->stream());
    if (!file_seek(in, payload_start + base)) die("seek failed");
    in_ring.upload(in, *s.set, total_in, "short read on compressed payload");
    g_stats.mark(b, kH2d1, s.set->stream());
  }

  void launch(DecompressSlot& s) {
    g_stats.mark(s.batch, kK0, s.set->stream());
    // The error flag is downloaded ahead of the output on the same stream,
    // so the writer can check it before writing any of this batch.
    s.set->launch(*group_tables, *h_err, (size_t)s.batch * sizeof(uint32_t));
    g_stats.mark(s.batch, kK1, s.set->stream());
    s.d2h_pending = true;
  }

  // Queues s's output for the writer; batches must come through in order.
  void enqueue_d2h(DecompressSlot& s) {
    uint64_t out_bytes = (uint64_t)(s.n - 1) * header.chunk_size + s.last_len;
    g_stats.mark(s.batch, kD2h0, s.set->stream());
    download(writer, *s.set, out_bytes, reinterpret_cast<const uint32_t*>(h_err->p) + s.batch);
    g_stats.mark(s.batch, kD2h1, s.set->stream());
    s.d2h_pending = false;
  }

  // Same shape as Compressor::run: batch b-1's output is queued after batch
  // b is launched, so the writer drains it while b's kernels run.
  void run() {
    for (int b = 0; b < plan.batches; ++b) {
      DecompressSlot& s = sets[b % plan.sets];
      if (s.d2h_pending) enqueue_d2h(s);
      upload(s, b);
      launch(s);
      if (b > 0) {
        DecompressSlot& prev = sets[(b - 1) % plan.sets];
        if (prev.d2h_pending) enqueue_d2h(prev);
      }
    }
    for (int b = std::max(0, plan.batches - plan.sets); b < plan.batches; ++b) {
      DecompressSlot& s = sets[b % plan.sets];
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
  if (header.magic != kMagic) die("bad magic (not a gpusqz file)");
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

  uint64_t payload_start = file_tell(in);

  // The payload runs from here to the table section, which ends the file.
  // Checking that the chunk table covers exactly that range keeps any batch
  // read inside the payload.
  uint64_t payload_bytes = entries.empty() ? 0 : entries.back().offset + entries.back().compressed_size;
  uint64_t tables_bytes = 0;
  for (const TableGroup& g : groups) tables_bytes += group_quant_bytes(g);
  if (header.tables_offset != payload_start + payload_bytes ||
      header.tables_offset + tables_bytes != file_size(in)) {
    die("corrupt header: payload and table section don't match the file");
  }
  std::vector<uint8_t> tables(tables_bytes);
  if (!file_seek(in, header.tables_offset) ||
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
    open_backend();
    dz.allocate();
    dz.prepare_group_tables();
    g_stats.setup_s += now_s() - t;
    g_stats.start_clock(dz.sets[0].set->stream());

    dz.run();
    g_stats.bytes_in = file_size(in);
    g_stats.bytes_out = header.original_size;
  }

  std::fclose(in);
  if (std::fclose(out) != 0) die("write failed");
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

void usage() {
  std::fprintf(stderr,
               "usage:\n"
               "  gpusqz c <input> <output> [chunk_size | --profile speed|balance|ratio] [options]\n"
               "  gpusqz d <input> <output> [options]\n"
               "  gpusqz devices\n"
               "  gpusqz --version\n"
               "chunk_size and --profile are mutually exclusive; with neither, chunk_size is %u.\n"
               "options:\n"
               "  --gpu-mem SIZE   cap the GPU memory used for batch buffers (e.g. 8G, 512M; a\n"
               "                   bare number is MiB). Default: 80%% of the free GPU memory.\n"
               "                   Also GPUSQZ_GPU_MEM.\n"
               "  --backend NAME   auto, cuda or vulkan. Default auto: CUDA if an NVIDIA GPU\n"
               "                   is present, else Vulkan. Also GPUSQZ_BACKEND.\n",
               kDefaultChunkSize);
  std::exit(1);
}

// Parses a --gpu-mem / GPUSQZ_GPU_MEM value: a positive number with an
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

// Picks the backend: the named one, or with "auto" CUDA first (the tuned
// path on NVIDIA GPUs), then Vulkan.
std::unique_ptr<Backend> make_backend(const std::string& name) {
  std::string why, notes;
  if (name == "auto" || name == "cuda") {
#ifdef GPUSQZ_HAVE_CUDA
    if (auto b = make_cuda_backend(&why)) return b;
    notes += "\n  CUDA: " + why;
#else
    notes += "\n  CUDA: not built in";
#endif
  }
  if (name == "auto" || name == "vulkan") {
#ifdef GPUSQZ_HAVE_VULKAN
    if (auto b = make_vulkan_backend(&why)) return b;
    notes += "\n  Vulkan: " + why;
#else
    notes += "\n  Vulkan: not built in";
#endif
  }
  die("no usable GPU backend:" + notes);
}

// Never destroyed: work still queued when die() exits must not outlive it.
void open_backend() {
  if (!g_backend) g_backend = make_backend(g_backend_name).release();
}

void list_devices() {
#ifdef GPUSQZ_HAVE_CUDA
  list_cuda_devices();
#else
  std::fprintf(stderr, "CUDA: not built in\n");
#endif
#ifdef GPUSQZ_HAVE_VULKAN
  list_vulkan_devices();
#else
  std::fprintf(stderr, "Vulkan: not built in\n");
#endif
}

} // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "devices") == 0) {
    list_devices();
    return 0;
  }
  if (argc == 2 && std::strcmp(argv[1], "--version") == 0) {
    std::printf("gpusqz %s\n", GPUSQZ_VERSION_STRING);
    return 0;
  }
  if (argc < 4) usage();
  std::string mode = argv[1];
  std::string in_path = argv[2];
  std::string out_path = argv[3];
  if (mode != "c" && mode != "d") usage();

  auto t0 = std::chrono::steady_clock::now();
  if (const char* m = std::getenv("GPUSQZ_GPU_MEM")) g_gpu_mem = parse_mem_size(m);
  if (const char* b = std::getenv("GPUSQZ_BACKEND")) g_backend_name = b;
  // --gpu-mem and --backend apply to both modes (and win over the
  // environment); the rest of the arguments are mode-specific.
  std::vector<std::string> rest;
  for (int i = 4; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--gpu-mem") {
      if (i + 1 >= argc) die("--gpu-mem needs a value (e.g. 8G)");
      g_gpu_mem = parse_mem_size(argv[++i]);
    } else if (a == "--backend") {
      if (i + 1 >= argc) die("--backend needs a value (auto, cuda or vulkan)");
      g_backend_name = argv[++i];
    } else {
      rest.push_back(a);
    }
  }
  uint32_t chunk_size = kDefaultChunkSize;
  if (mode == "c") {
    bool have_chunk_size = false, have_profile = false;
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
  } else if (!rest.empty()) {
    die("unrecognised argument: " + rest[0]);
  }

  if (g_backend_name != "auto" && g_backend_name != "cuda" && g_backend_name != "vulkan") {
    die("unknown --backend: " + g_backend_name + " (expected auto, cuda or vulkan)");
  }
  if (mode == "c") compress(in_path, out_path, chunk_size);
  else decompress(in_path, out_path);

  auto t1 = std::chrono::steady_clock::now();
  double secs = std::chrono::duration<double>(t1 - t0).count();
  std::fprintf(stderr, "gpusqz: %s done in %.3fs\n", mode == "c" ? "compress" : "decompress", secs);
  g_stats.report(mode == "c" ? "compress" : "decompress", secs);
  return 0;
}
