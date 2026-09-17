// Compute backends. The host pipeline (main.cpp) is the same for every GPU
// API: it plans batches, streams file data through staging buffers, and
// writes the container. A Backend supplies memory, queues and the kernels:
//   CUDA    NVIDIA GPUs (cuda_backend.cu, kernels.cu)
//   Vulkan  AMD, Apple (through MoltenVK), Intel and NVIDIA (vk_backend.cpp)
// Every backend reads and writes the same .gsz format; only the device-side
// scratch layouts differ.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "format.h"

namespace gpusqz {

// Host memory the device can copy to and from asynchronously (pinned, or
// host-visible on Vulkan).
struct HostBuf {
  virtual ~HostBuf() = default;
  uint8_t* p = nullptr;
  size_t size = 0;
};

// An in-order queue of copies and kernels (a CUDA stream).
struct Stream {
  virtual ~Stream() = default;
};

// A point in a Stream's work: record() it after some commands, then wait()
// for them from any thread.
struct Event {
  virtual ~Event() = default;
};

// The most tables one decompress batch touches: its TableGroups and their
// literal contexts summed (lit_ctx_count). Both backends' expanded tables
// are linear in the two, so these bound a set's table memory.
struct TableCapacity {
  uint32_t groups = 0;
  uint64_t contexts = 0;
};

// One decompress batch's tables: its TableGroups in order (count of them,
// within the set's TableCapacity) and their quantised counts back to back.
// The batch's group_id[] indexes these, 0-based.
struct TableWindow {
  const TableGroup* groups;
  uint32_t count;
  const uint8_t* quant;
};

// Device memory and kernels for one compress batch of up to `batch` chunks.
// A batch goes through: begin() -> fill in_lens -> upload_input() pieces ->
// launch() -> wait_meta() -> read the results -> download_output() pieces.
class CompressSet {
 public:
  virtual ~CompressSet() = default;
  virtual Stream& stream() = 0;
  // Waits until the previous batch no longer needs the host-side inputs and
  // returns the array for this batch's n chunk lengths.
  virtual uint32_t* begin(uint32_t n) = 0;
  // Queues a copy of len bytes from src to the batch input at byte off.
  // Chunk c's input starts at c * chunk_size.
  virtual void upload_input(uint64_t off, HostBuf& src, size_t len) = 0;
  // Queues the kernels (forced_lit_shift < 0 lets the batch pick its
  // literal-context rule) and the download of the results below.
  virtual void launch(int forced_lit_shift) = 0;
  // Waits for launch()'s results.
  virtual void wait_meta() = 0;
  virtual const uint32_t* sizes() = 0;           // n compressed chunk sizes
  virtual uint32_t packed_bytes() = 0;           // their sum: the packed output's size
  // The batch's TableGroups, g of group_count(): chunks [g*group_chunks,
  // (g+1)*group_chunks) of the batch (format.h's group_chunks()).
  virtual uint32_t group_count() = 0;
  virtual uint32_t lit_shift(uint32_t g) = 0;    // group g's literal-context rule
  virtual const uint8_t* quant(uint32_t g) = 0;  // its quantised counts (group_quant_bytes)
  // Queues a copy of len bytes of the packed output, from byte off, to dst.
  virtual void download_output(uint64_t off, HostBuf& dst, size_t len) = 0;
};

// Device memory and kernels for one decompress batch of up to `batch` chunks.
class DecompressSet {
 public:
  struct Inputs {
    uint32_t* in_offsets; // chunk c's data within the batch input
    uint32_t* in_lens;    // its compressed size
    uint32_t* out_lens;   // its original size
    uint32_t* group_id;   // its TableGroup, as an index into launch()'s TableWindow
  };
  virtual ~DecompressSet() = default;
  virtual Stream& stream() = 0;
  virtual Inputs begin(uint32_t n) = 0;
  virtual void upload_input(uint64_t off, HostBuf& src, size_t len) = 0;
  // Queues the expansion of the batch's tables and the kernel, then a
  // download of its error flag (nonzero: some chunk was malformed) to
  // err->p + err_off.
  virtual void launch(const TableWindow& tables, HostBuf& err, size_t err_off) = 0;
  // Chunk c's output is at byte c * chunk_size.
  virtual void download_output(uint64_t off, HostBuf& dst, size_t len) = 0;
};

class Backend {
 public:
  virtual ~Backend() = default;
  virtual std::string name() = 0; // "CUDA: <device>" / "Vulkan: <device>"
  virtual size_t free_memory() = 0;
  // Most chunks one batch may hold, whatever the memory budget (a limit on
  // single buffer sizes).
  virtual uint32_t max_batch(uint32_t chunk_size) = 0;

  virtual std::unique_ptr<HostBuf> alloc_host(size_t bytes) = 0; // null on failure
  virtual std::unique_ptr<Event> create_event(bool timing) = 0;
  virtual void record(Event& e, Stream& s) = 0;
  // Returns false, with the reason in *why, if the work failed. Thread-safe.
  virtual bool wait(Event& e, std::string* why) = 0;
  // Milliseconds between two completed timing events.
  virtual double elapsed_ms(Event& from, Event& to) = 0;

  // Device bytes a set needs per chunk, for batch planning.
  virtual size_t compress_bytes_per_chunk(uint32_t chunk_size) = 0;
  virtual size_t decompress_bytes_per_chunk(uint32_t chunk_size) = 0;
  // Null if the memory isn't available.
  virtual std::unique_ptr<CompressSet> create_compress_set(uint32_t batch, uint32_t chunk_size) = 0;
  // A decompress set also holds the expanded tables of up to `tables` at
  // once, rebuilt for each batch, so table memory doesn't grow with the file.
  virtual std::unique_ptr<DecompressSet> create_decompress_set(uint32_t batch, uint32_t chunk_size,
                                                               const TableCapacity& tables) = 0;
};

// Each returns null, with the reason in *why, if that API or a usable
// device is missing. `list` prints the devices found to stderr.
std::unique_ptr<Backend> make_cuda_backend(std::string* why);
std::unique_ptr<Backend> make_vulkan_backend(std::string* why);
void list_cuda_devices();
void list_vulkan_devices();

} // namespace gpusqz
