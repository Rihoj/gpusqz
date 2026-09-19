// The CUDA backend: NVIDIA GPUs, with the kernels in kernels.cu.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cuda_runtime.h>

#include "backend.h"
#include "kernels.h"

namespace gpusqz {
namespace {

[[noreturn]] void die_cuda(cudaError_t err, const char* what) {
  std::fprintf(stderr, "gpusqz: %s: %s\n", what, cudaGetErrorString(err));
  std::exit(1);
}

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) die_cuda(err, what);
}

template <typename T>
struct DevBuf {
  T* p = nullptr;
  bool alloc(size_t count) {
    if (cudaMalloc(&p, count * sizeof(T)) != cudaSuccess) {
      cudaGetLastError();
      p = nullptr;
      return false;
    }
    return true;
  }
  DevBuf() = default;
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  ~DevBuf() {
    if (p) cudaFree(p);
  }
};

// Pinned host memory: required for cudaMemcpyAsync to actually be
// asynchronous (from pageable memory it silently serialises).
struct CudaHostBuf : HostBuf {
  ~CudaHostBuf() override {
    if (p) cudaFreeHost(p);
  }
};

template <typename T>
struct PinBuf {
  T* p = nullptr;
  bool alloc(size_t count) {
    if (cudaHostAlloc(&p, count * sizeof(T), cudaHostAllocDefault) != cudaSuccess) {
      cudaGetLastError();
      p = nullptr;
      return false;
    }
    return true;
  }
  PinBuf() = default;
  PinBuf(const PinBuf&) = delete;
  PinBuf& operator=(const PinBuf&) = delete;
  ~PinBuf() {
    if (p) cudaFreeHost(p);
  }
};

struct CudaStream : Stream {
  cudaStream_t s = nullptr;
  CudaStream() { check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate"); }
  ~CudaStream() override {
    if (s) cudaStreamDestroy(s);
  }
};

struct CudaEvent : Event {
  cudaEvent_t e = nullptr;
  explicit CudaEvent(bool timing) {
    check(cudaEventCreateWithFlags(&e, timing ? cudaEventDefault : cudaEventDisableTiming), "cudaEventCreate");
  }
  ~CudaEvent() override {
    if (e) cudaEventDestroy(e);
  }
};

cudaStream_t raw(Stream& s) { return static_cast<CudaStream&>(s).s; }

// Per-chunk device scratch actually allocated for compress: the LzRans
// scratch, which also receives the compacted output once the encode kernel
// is done with it (see CudaCompressSet::launch), so it must hold at least
// one output slot per chunk. That only binds for tiny chunk sizes.
size_t compress_scratch_bytes(uint32_t chunk_size) {
  return std::max(scratch_bytes(chunk_size), (size_t)worst_case_size(chunk_size));
}

class CudaCompressSet : public CompressSet {
 public:
  bool alloc(uint32_t batch, uint32_t chunk_size) {
    chunk_size_ = chunk_size;
    slot_stride_ = worst_case_size(chunk_size);
    temp_bytes_ = compaction_temp_bytes(batch);
    group_chunks_ = group_chunks(chunk_size);
    max_groups_ = (batch + group_chunks_ - 1) / group_chunks_;
    size_t tab = (size_t)max_groups_ * kMaxQuantBytes;
    return h_in_lens_.alloc(batch) && h_sizes_.alloc(batch) && h_hashes_.alloc(batch) &&
           h_offsets_.alloc(batch + 1) &&
           h_shift_.alloc(max_groups_) && h_q_.alloc(tab) && d_in_.alloc((size_t)batch * chunk_size) &&
           d_slots_.alloc((size_t)batch * slot_stride_) && d_temp_.alloc(temp_bytes_) &&
           d_scratch_.alloc((size_t)batch * compress_scratch_bytes(chunk_size)) &&
           d_htab_.alloc((size_t)batch * (hash_table_bytes(chunk_size) / sizeof(uint32_t))) &&
           d_in_lens_.alloc(batch) && d_start_.alloc(batch) && d_sizes_.alloc(batch) &&
           d_offsets_.alloc(batch + 1) && d_rans_hash_.alloc(batch) && d_rans_cnt_.alloc(tab) &&
           d_rans_shift_.alloc(max_groups_) &&
           d_rans_n_seq_.alloc(batch) && d_rans_n_lit_.alloc(batch) && d_rans_freq_.alloc(tab) &&
           d_rans_cum_.alloc(tab) && d_rans_q_.alloc(tab);
  }

  Stream& stream() override { return stream_; }

  uint32_t* begin(uint32_t n) override {
    n_ = n;
    // h_in_lens is about to be rewritten: the previous batch's upload of it
    // (long since finished in practice) must be complete.
    if (used_) check(cudaEventSynchronize(h2d_done_.e), "wait for set");
    used_ = true;
    return h_in_lens_.p;
  }

  void upload_input(uint64_t off, HostBuf& src, size_t len) override {
    check(cudaMemcpyAsync(d_in_.p + off, src.p, len, cudaMemcpyHostToDevice, stream_.s), "H2D input");
  }

  void launch(int forced_lit_shift) override {
    cudaStream_t st = stream_.s;
    check(cudaMemcpyAsync(d_in_lens_.p, h_in_lens_.p, n_ * sizeof(uint32_t), cudaMemcpyHostToDevice, st),
          "H2D lens");
    check(cudaEventRecord(h2d_done_.e, st), "cudaEventRecord");
    RansBatchBufs rb{d_rans_hash_.p,  d_rans_cnt_.p,   d_rans_freq_.p,  d_rans_cum_.p,
                     d_rans_q_.p,      d_rans_shift_.p, d_rans_n_seq_.p, d_rans_n_lit_.p};
    groups_ = (n_ + group_chunks_ - 1) / group_chunks_;
    launch_compress(d_in_.p, chunk_size_, n_, d_in_lens_.p, d_slots_.p, slot_stride_, d_start_.p, d_sizes_.p,
                    d_scratch_.p, d_htab_.p, forced_lit_shift, group_chunks_, rb, st);
    check(cudaGetLastError(), "compress_kernel launch");
    // The encode kernel is done with scratch (same stream), so the packed
    // output reuses it; see compress_scratch_bytes.
    check(launch_compact(d_slots_.p, slot_stride_, d_start_.p, d_sizes_.p, n_, d_offsets_.p, d_scratch_.p,
                         d_temp_.p, temp_bytes_, st),
          "compaction");
    check(cudaMemcpyAsync(h_sizes_.p, d_sizes_.p, n_ * sizeof(uint32_t), cudaMemcpyDeviceToHost, st), "D2H sizes");
    check(cudaMemcpyAsync(h_hashes_.p, d_rans_hash_.p, n_ * sizeof(uint32_t), cudaMemcpyDeviceToHost, st),
          "D2H hashes");
    check(cudaMemcpyAsync(h_offsets_.p, d_offsets_.p, (n_ + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost, st),
          "D2H offsets");
    check(cudaMemcpyAsync(h_shift_.p, d_rans_shift_.p, groups_ * sizeof(uint32_t), cudaMemcpyDeviceToHost, st),
          "D2H shift");
    check(cudaMemcpyAsync(h_q_.p, d_rans_q_.p, (size_t)groups_ * kMaxQuantBytes, cudaMemcpyDeviceToHost, st),
          "D2H q");
    check(cudaEventRecord(meta_.e, st), "cudaEventRecord");
  }

  void wait_meta() override { check(cudaEventSynchronize(meta_.e), "wait for chunk sizes"); }
  const uint32_t* sizes() override { return h_sizes_.p; }
  const uint32_t* hashes() override { return h_hashes_.p; }
  uint32_t packed_bytes() override { return h_offsets_.p[n_]; }
  uint32_t group_count() override { return groups_; }
  uint32_t lit_shift(uint32_t g) override { return h_shift_.p[g]; }
  const uint8_t* quant(uint32_t g) override { return h_q_.p + (size_t)g * kMaxQuantBytes; }

  void download_output(uint64_t off, HostBuf& dst, size_t len) override {
    check(cudaMemcpyAsync(dst.p, d_scratch_.p + off, len, cudaMemcpyDeviceToHost, stream_.s), "D2H output");
  }

 private:
  CudaStream stream_;
  CudaEvent h2d_done_{false}, meta_{false};
  uint32_t chunk_size_ = 0, slot_stride_ = 0, n_ = 0;
  uint32_t group_chunks_ = 0, max_groups_ = 0, groups_ = 0;
  size_t temp_bytes_ = 0;
  bool used_ = false;
  PinBuf<uint32_t> h_in_lens_, h_sizes_, h_hashes_, h_offsets_, h_shift_;
  PinBuf<uint8_t> h_q_; // each group's quantised tables, kMaxQuantBytes apart
  DevBuf<uint8_t> d_in_, d_slots_, d_temp_, d_scratch_; // d_scratch_ doubles as the packed output
  DevBuf<uint32_t> d_htab_; // LZ parse's match-finding table, one region per chunk; see hash_table_bytes()
  DevBuf<uint32_t> d_in_lens_, d_start_, d_sizes_, d_offsets_;
  DevBuf<uint32_t> d_rans_hash_, d_rans_cnt_, d_rans_shift_, d_rans_n_seq_, d_rans_n_lit_;
  DevBuf<uint16_t> d_rans_freq_, d_rans_cum_;
  DevBuf<uint8_t> d_rans_q_;
};

class CudaDecompressSet : public DecompressSet {
 public:
  bool alloc(uint32_t batch, uint32_t chunk_size, const TableCapacity& tables) {
    chunk_size_ = chunk_size;
    uint32_t slot_stride = worst_case_size(chunk_size);
    // Quantised counts and expanded tables are both linear in groups and
    // contexts (rans_codes.h's quant_bytes, rans.cuh's group_table_bytes),
    // so the capacity's two maxima bound any batch's total.
    max_groups_ = std::max<uint32_t>(1, tables.groups);
    max_q_ = max_groups_ * (size_t)quant_bytes(0) + tables.contexts * (size_t)kLitSyms;
    size_t per_ctx = (rans_group_table_bytes(4) - rans_group_table_bytes(8)) / 15; // 16 contexts vs 1
    max_tables_ = max_groups_ * (rans_group_table_bytes(8) - per_ctx) + tables.contexts * per_ctx;
    return h_in_offsets_.alloc(batch) && h_in_lens_.alloc(batch) && h_out_lens_.alloc(batch) &&
           h_group_id_.alloc(batch) && h_hash_.alloc(batch) && d_in_.alloc((size_t)batch * slot_stride) &&
           d_out_.alloc((size_t)batch * chunk_size) && d_scratch_.alloc((size_t)batch * scratch_bytes(chunk_size)) &&
           d_in_offsets_.alloc(batch) && d_in_lens_.alloc(batch) && d_out_lens_.alloc(batch) &&
           d_group_id_.alloc(batch) && d_hash_.alloc(batch) && d_err_.alloc(1) && h_q_.alloc(max_q_) &&
           h_q_off_.alloc(max_groups_) &&
           h_table_off_.alloc(max_groups_) && h_shift_.alloc(max_groups_) && d_q_.alloc(max_q_) &&
           d_q_off_.alloc(max_groups_) && d_table_off_.alloc(max_groups_) && d_shift_.alloc(max_groups_) &&
           d_tables_.alloc(max_tables_);
  }

  Stream& stream() override { return stream_; }

  Inputs begin(uint32_t n) override {
    n_ = n;
    if (used_) check(cudaEventSynchronize(h2d_done_.e), "wait for set");
    used_ = true;
    return Inputs{h_in_offsets_.p, h_in_lens_.p, h_out_lens_.p, h_group_id_.p, h_hash_.p};
  }

  void upload_input(uint64_t off, HostBuf& src, size_t len) override {
    check(cudaMemcpyAsync(d_in_.p + off, src.p, len, cudaMemcpyHostToDevice, stream_.s), "H2D input");
  }

  void launch(const TableWindow& tables, HostBuf& err, size_t err_off) override {
    cudaStream_t st = stream_.s;
    size_t bytes = n_ * sizeof(uint32_t);
    check(cudaMemcpyAsync(d_in_offsets_.p, h_in_offsets_.p, bytes, cudaMemcpyHostToDevice, st), "H2D offsets");
    check(cudaMemcpyAsync(d_in_lens_.p, h_in_lens_.p, bytes, cudaMemcpyHostToDevice, st), "H2D lens");
    check(cudaMemcpyAsync(d_out_lens_.p, h_out_lens_.p, bytes, cudaMemcpyHostToDevice, st), "H2D out_lens");
    check(cudaMemcpyAsync(d_group_id_.p, h_group_id_.p, bytes, cudaMemcpyHostToDevice, st), "H2D group_id");
    check(cudaMemcpyAsync(d_hash_.p, h_hash_.p, bytes, cudaMemcpyHostToDevice, st), "H2D hashes");
    size_t table_bytes = upload_tables(tables);
    check(cudaEventRecord(h2d_done_.e, st), "cudaEventRecord");
    // Unused contexts and groups with no LzRans chunks have all-zero
    // counts, which leave their tables untouched; zeroing first makes those
    // zero frequencies (what flags a corrupt chunk using one) rather than
    // whatever the previous batch left.
    check(cudaMemsetAsync(d_tables_.p, 0, table_bytes, st), "zero group tables");
    launch_expand_group_tables(d_q_.p, d_q_off_.p, d_table_off_.p, d_shift_.p, tables.count, d_tables_.p, st);
    check(cudaGetLastError(), "expand_group_tables launch");
    check(cudaMemsetAsync(d_err_.p, 0, sizeof(uint32_t), st), "clear error flag");
    launch_decompress(d_in_.p, d_in_offsets_.p, n_, d_in_lens_.p, d_out_.p, chunk_size_, d_out_lens_.p,
                      d_scratch_.p, d_tables_.p, d_table_off_.p, d_shift_.p, d_group_id_.p, d_hash_.p, d_err_.p,
                      st);
    check(cudaGetLastError(), "decompress_kernel launch");
    check(cudaMemcpyAsync(err.p + err_off, d_err_.p, sizeof(uint32_t), cudaMemcpyDeviceToHost, st), "D2H err");
  }

  void download_output(uint64_t off, HostBuf& dst, size_t len) override {
    check(cudaMemcpyAsync(dst.p, d_out_.p + off, len, cudaMemcpyDeviceToHost, stream_.s), "D2H output");
  }

 private:
  // Stages the window's counts and its groups' offsets and rules, and
  // queues their upload. Each group's counts and expanded tables vary in
  // size with its literal-context rule, so both are located by offset.
  // Returns the window's expanded table bytes.
  size_t upload_tables(const TableWindow& w) {
    uint64_t q = 0, t = 0;
    for (uint32_t g = 0; g < w.count; ++g) {
      h_q_off_.p[g] = q;
      h_table_off_.p[g] = t;
      h_shift_.p[g] = w.groups[g].lit_ctx_shift;
      q += group_quant_bytes(w.groups[g]);
      t += rans_group_table_bytes(w.groups[g].lit_ctx_shift);
    }
    if (w.count > max_groups_ || q > max_q_ || t > max_tables_) {
      std::fprintf(stderr, "gpusqz: internal error: a decode batch's tables exceed its set's capacity\n");
      std::exit(1);
    }
    cudaStream_t st = stream_.s;
    std::memcpy(h_q_.p, w.quant, q);
    check(cudaMemcpyAsync(d_q_.p, h_q_.p, q, cudaMemcpyHostToDevice, st), "H2D tables");
    check(cudaMemcpyAsync(d_q_off_.p, h_q_off_.p, w.count * sizeof(uint64_t), cudaMemcpyHostToDevice, st),
          "H2D q offsets");
    check(cudaMemcpyAsync(d_table_off_.p, h_table_off_.p, w.count * sizeof(uint64_t), cudaMemcpyHostToDevice, st),
          "H2D table offsets");
    check(cudaMemcpyAsync(d_shift_.p, h_shift_.p, w.count * sizeof(uint32_t), cudaMemcpyHostToDevice, st),
          "H2D shifts");
    return t;
  }

  CudaStream stream_;
  CudaEvent h2d_done_{false};
  uint32_t chunk_size_ = 0, n_ = 0, max_groups_ = 0;
  size_t max_q_ = 0, max_tables_ = 0;
  bool used_ = false;
  PinBuf<uint32_t> h_in_offsets_, h_in_lens_, h_out_lens_, h_group_id_, h_hash_, h_shift_;
  PinBuf<uint64_t> h_q_off_, h_table_off_;
  PinBuf<uint8_t> h_q_;
  DevBuf<uint8_t> d_in_, d_out_, d_scratch_, d_q_, d_tables_;
  DevBuf<uint32_t> d_in_offsets_, d_in_lens_, d_out_lens_, d_group_id_, d_hash_, d_err_, d_shift_;
  DevBuf<uint64_t> d_q_off_, d_table_off_;
};

class CudaBackend : public Backend {
 public:
  explicit CudaBackend(int device) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) name_ = std::string("CUDA: ") + prop.name;
  }

  std::string name() override { return name_; }

  size_t free_memory() override {
    size_t free_bytes = 0, total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    return free_bytes;
  }

  uint32_t max_batch(uint32_t) override { return UINT32_MAX; }

  std::unique_ptr<HostBuf> alloc_host(size_t bytes) override {
    auto b = std::make_unique<CudaHostBuf>();
    if (cudaHostAlloc(&b->p, bytes, cudaHostAllocDefault) != cudaSuccess) {
      cudaGetLastError();
      b->p = nullptr;
      return nullptr;
    }
    b->size = bytes;
    return b;
  }

  std::unique_ptr<Event> create_event(bool timing) override { return std::make_unique<CudaEvent>(timing); }

  void record(Event& e, Stream& s) override {
    check(cudaEventRecord(static_cast<CudaEvent&>(e).e, raw(s)), "cudaEventRecord");
  }

  bool wait(Event& e, std::string* why) override {
    cudaError_t err = cudaEventSynchronize(static_cast<CudaEvent&>(e).e);
    if (err == cudaSuccess) return true;
    *why = cudaGetErrorString(err);
    return false;
  }

  double elapsed_ms(Event& from, Event& to) override {
    float ms = 0;
    check(cudaEventElapsedTime(&ms, static_cast<CudaEvent&>(from).e, static_cast<CudaEvent&>(to).e),
          "cudaEventElapsedTime");
    return ms;
  }

  size_t compress_bytes_per_chunk(uint32_t chunk_size) override {
    // Plus each group's histogram and tables, amortised over its chunks
    // (kMaxQuantBytes of u32 counts, u16 freq and cum, and u8 counts).
    size_t per_group = (size_t)kMaxQuantBytes * (4 + 2 + 2 + 1);
    return (size_t)chunk_size + worst_case_size(chunk_size) + compress_scratch_bytes(chunk_size) +
           hash_table_bytes(chunk_size) + 6 * sizeof(uint32_t) + per_group / group_chunks(chunk_size);
  }

  size_t decompress_bytes_per_chunk(uint32_t chunk_size) override {
    return (size_t)chunk_size + worst_case_size(chunk_size) + scratch_bytes(chunk_size) + 4 * sizeof(uint32_t);
  }

  std::unique_ptr<CompressSet> create_compress_set(uint32_t batch, uint32_t chunk_size) override {
    auto s = std::make_unique<CudaCompressSet>();
    if (!s->alloc(batch, chunk_size)) return nullptr;
    return s;
  }

  std::unique_ptr<DecompressSet> create_decompress_set(uint32_t batch, uint32_t chunk_size,
                                                       const TableCapacity& tables) override {
    auto s = std::make_unique<CudaDecompressSet>();
    if (!s->alloc(batch, chunk_size, tables)) return nullptr;
    return s;
  }

 private:
  std::string name_ = "CUDA";
};

} // namespace

std::unique_ptr<Backend> make_cuda_backend(std::string* why) {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count == 0) {
    cudaGetLastError();
    *why = err != cudaSuccess ? cudaGetErrorString(err) : "no CUDA device";
    return nullptr;
  }
  return std::make_unique<CudaBackend>(0);
}

void list_cuda_devices() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    cudaGetLastError();
    std::fprintf(stderr, "CUDA: unavailable (%s)\n", cudaGetErrorString(err));
    return;
  }
  for (int i = 0; i < count; ++i) {
    cudaDeviceProp p{};
    size_t free_bytes = 0, total = 0;
    if (cudaGetDeviceProperties(&p, i) != cudaSuccess) continue;
    if (i == 0) cudaMemGetInfo(&free_bytes, &total);
    std::fprintf(stderr, "CUDA %d: %s, compute %d.%d, %d SMs, %zu MiB%s\n", i, p.name, p.major, p.minor,
                 p.multiProcessorCount, p.totalGlobalMem >> 20, i == 0 ? " (used)" : "");
  }
  if (count == 0) std::fprintf(stderr, "CUDA: no devices\n");
}

} // namespace gpusqz
