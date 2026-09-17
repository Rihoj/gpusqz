// The Vulkan backend: AMD, Apple (through MoltenVK), Intel and NVIDIA GPUs.
// The compute shaders in src/vk/ are ports of the CUDA kernels, compiled to
// SPIR-V at build time and embedded here.
//
// libvulkan (vulkan-1.dll, or MoltenVK on macOS) is loaded at run time, so
// the executable starts without it and falls back to CUDA or reports why.
//
// Each Stream is a queue plus command buffers. Commands are recorded into
// the stream's current command buffer, each after a full memory barrier
// (the batch pipeline is a chain of dependent steps anyway), and an Event
// is recorded by submitting that command buffer with a timeline semaphore
// signal; waiting for the event waits for that value.
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "backend.h"
#include "format.h"
#include "rans_codes.h"

// Generated from src/vk/*.comp (CMakeLists.txt): spv_<shader>_<LG_SUBGROUP>.
#include "spv_build_table_0.h"
#include "spv_compact_0.h"
#include "spv_decompress_0.h"
#include "spv_decompress_1.h"
#include "spv_expand_tables_0.h"
#include "spv_parse_hist_0.h"
#include "spv_parse_hist_1.h"
#include "spv_probe_0.h"
#include "spv_probe_1.h"
#include "spv_rans_encode_0.h"
#include "spv_rans_encode_1.h"
#include "spv_scan_0.h"

namespace gpusqz {
namespace {

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

#define GPUSQZ_VK_GLOBAL(X) X(vkCreateInstance) X(vkEnumerateInstanceExtensionProperties) X(vkEnumerateInstanceVersion)
#define GPUSQZ_VK_INSTANCE(X)                                                                              \
  X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties)                      \
  X(vkGetPhysicalDeviceProperties2) X(vkGetPhysicalDeviceFeatures2) X(vkGetPhysicalDeviceQueueFamilyProperties) \
  X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceMemoryProperties2)                           \
  X(vkEnumerateDeviceExtensionProperties) X(vkCreateDevice) X(vkGetDeviceProcAddr)
#define GPUSQZ_VK_DEVICE(X)                                                                                \
  X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) \
  X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory) X(vkMapMemory) X(vkUnmapMemory)                \
  X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreateDescriptorSetLayout)                          \
  X(vkDestroyDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout)                     \
  X(vkCreateComputePipelines) X(vkDestroyPipeline) X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool)    \
  X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) X(vkCreateCommandPool) X(vkDestroyCommandPool)     \
  X(vkAllocateCommandBuffers) X(vkFreeCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer)        \
  X(vkResetCommandBuffer) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdPushConstants)            \
  X(vkCmdDispatch) X(vkCmdCopyBuffer) X(vkCmdFillBuffer) X(vkCmdPipelineBarrier) X(vkCmdWriteTimestamp)    \
  X(vkCmdResetQueryPool) X(vkCreateQueryPool) X(vkDestroyQueryPool) X(vkGetQueryPoolResults)             \
  X(vkQueueSubmit) X(vkCreateSemaphore) X(vkDestroySemaphore) X(vkWaitSemaphores)                          \
  X(vkGetSemaphoreCounterValue)

#define GPUSQZ_VK_DECLARE(f) PFN_##f f = nullptr;
PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
GPUSQZ_VK_GLOBAL(GPUSQZ_VK_DECLARE)
GPUSQZ_VK_INSTANCE(GPUSQZ_VK_DECLARE)
GPUSQZ_VK_DEVICE(GPUSQZ_VK_DECLARE)

// Finds vkGetInstanceProcAddr in the system's Vulkan loader, or on macOS
// in MoltenVK used directly (installed by Homebrew or shipped next to the
// executable). GPUSQZ_VULKAN_LIB names a library to use instead.
bool load_vulkan(std::string* why) {
  if (vkGetInstanceProcAddr) return true;
  std::vector<std::string> names;
  if (const char* lib = std::getenv("GPUSQZ_VULKAN_LIB")) names.push_back(lib);
#if defined(_WIN32)
  names.push_back("vulkan-1.dll");
#elif defined(__APPLE__)
  names.insert(names.end(), {"libvulkan.1.dylib", "@executable_path/libMoltenVK.dylib",
                             "@executable_path/../lib/libMoltenVK.dylib", "libMoltenVK.dylib",
                             "/opt/homebrew/lib/libvulkan.1.dylib", "/opt/homebrew/lib/libMoltenVK.dylib",
                             "/usr/local/lib/libvulkan.1.dylib", "/usr/local/lib/libMoltenVK.dylib"});
#else
  names.insert(names.end(), {"libvulkan.so.1", "libvulkan.so"});
#endif
  for (const std::string& n : names) {
#ifdef _WIN32
    HMODULE h = LoadLibraryA(n.c_str());
    if (!h) continue;
    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)(void*)GetProcAddress(h, "vkGetInstanceProcAddr");
#else
    void* h = dlopen(n.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) continue;
    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(h, "vkGetInstanceProcAddr");
#endif
    if (vkGetInstanceProcAddr) break;
  }
  if (!vkGetInstanceProcAddr) {
    *why = "no Vulkan library found (install the GPU driver's Vulkan support";
#ifdef __APPLE__
    *why += ", or MoltenVK: brew install molten-vk";
#endif
    *why += ")";
    return false;
  }
#define GPUSQZ_VK_LOAD_GLOBAL(f) f = (PFN_##f)vkGetInstanceProcAddr(nullptr, #f);
  GPUSQZ_VK_GLOBAL(GPUSQZ_VK_LOAD_GLOBAL)
  if (!vkCreateInstance || !vkEnumerateInstanceExtensionProperties) {
    *why = "the Vulkan library is unusable";
    vkGetInstanceProcAddr = nullptr;
    return false;
  }
  return true;
}

[[noreturn]] void vk_die(VkResult r, const char* what) {
  std::fprintf(stderr, "gpusqz: Vulkan %s failed (VkResult %d)\n", what, (int)r);
  std::exit(1);
}

void vk_check(VkResult r, const char* what) {
  if (r != VK_SUCCESS) vk_die(r, what);
}

bool has_ext(const std::vector<VkExtensionProperties>& exts, const char* name) {
  for (const auto& e : exts) {
    if (std::strcmp(e.extensionName, name) == 0) return true;
  }
  return false;
}

// The instance, created once and never destroyed (see Backend lifetime in
// main.cpp).
VkInstance g_instance = VK_NULL_HANDLE;

bool create_instance(std::string* why) {
  if (g_instance) return true;
  if (!load_vulkan(why)) return false;
  uint32_t api = VK_API_VERSION_1_0;
  if (vkEnumerateInstanceVersion) vkEnumerateInstanceVersion(&api);
  if (api < VK_API_VERSION_1_2) {
    *why = "the Vulkan loader is older than Vulkan 1.2";
    return false;
  }
  uint32_t n = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
  std::vector<VkExtensionProperties> exts(n);
  vkEnumerateInstanceExtensionProperties(nullptr, &n, exts.data());

  std::vector<const char*> enable;
  VkInstanceCreateFlags flags = 0;
  // Lists MoltenVK (a "portability" implementation) through the loader.
  if (has_ext(exts, "VK_KHR_portability_enumeration")) {
    enable.push_back("VK_KHR_portability_enumeration");
    flags |= 0x00000001; // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
  }
  if (has_ext(exts, "VK_KHR_get_physical_device_properties2")) {
    enable.push_back("VK_KHR_get_physical_device_properties2");
  }
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "gpusqz";
  app.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ci.flags = flags;
  ci.pApplicationInfo = &app;
  ci.enabledExtensionCount = (uint32_t)enable.size();
  ci.ppEnabledExtensionNames = enable.data();
  VkResult r = vkCreateInstance(&ci, nullptr, &g_instance);
  if (r != VK_SUCCESS) {
    *why = "vkCreateInstance failed (VkResult " + std::to_string((int)r) + ")";
    return false;
  }
#define GPUSQZ_VK_LOAD_INSTANCE(f) f = (PFN_##f)vkGetInstanceProcAddr(g_instance, #f);
  GPUSQZ_VK_INSTANCE(GPUSQZ_VK_LOAD_INSTANCE)
  return true;
}

// ---------------------------------------------------------------------------
// Device selection
// ---------------------------------------------------------------------------

constexpr uint32_t kLanes = 32; // lane-group width: fixed by the format (kRansStates)
constexpr uint32_t kNeededSubgroupOps = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT |
                                        VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT |
                                        VK_SUBGROUP_FEATURE_SHUFFLE_BIT;

// What gpusqz needs to know about a physical device.
struct DevInfo {
  VkPhysicalDevice pd = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties props{};
  VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceSubgroupSizeControlPropertiesEXT sgc{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT};
  VkPhysicalDeviceMaintenance3Properties m3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceSubgroupSizeControlFeaturesEXT fsgc{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT};
  std::vector<VkExtensionProperties> exts;
  bool size_control = false; // VK_EXT_subgroup_size_control or Vulkan 1.3
  uint32_t family = UINT32_MAX, queue_count = 0, timestamp_bits = 0;
  // Lane-group build: subgroup (with or without requiring subgroup size
  // 32) or shared memory; see common.glsl.
  bool subgroup_lanes = false, require32 = false;
  std::string problem; // why the device is unusable, if it is
};

const char* type_name(VkPhysicalDeviceType t) {
  switch (t) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete GPU";
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated GPU";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual GPU";
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU";
    default: return "other";
  }
}

DevInfo query_device(VkPhysicalDevice pd) {
  DevInfo d;
  d.pd = pd;
  vkGetPhysicalDeviceProperties(pd, &d.props);
  uint32_t n = 0;
  vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
  d.exts.resize(n);
  vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, d.exts.data());
  if (d.props.apiVersion < VK_API_VERSION_1_2) {
    d.problem = "needs Vulkan 1.2";
    return d;
  }
  // The instance asks for Vulkan 1.2, so subgroup size control comes from
  // the extension even where it is core (1.3).
  d.size_control = has_ext(d.exts, "VK_EXT_subgroup_size_control");

  VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  p2.pNext = &d.sg;
  d.sg.pNext = &d.m3;
  if (d.size_control) d.m3.pNext = &d.sgc;
  vkGetPhysicalDeviceProperties2(pd, &p2);

  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  f2.pNext = &d.f12;
  if (d.size_control) d.f12.pNext = &d.fsgc;
  vkGetPhysicalDeviceFeatures2(pd, &f2);
  d.f12.pNext = nullptr;
  d.fsgc.pNext = nullptr;
  if (!d.fsgc.subgroupSizeControl) d.size_control = false;

  vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, nullptr);
  std::vector<VkQueueFamilyProperties> fams(n);
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, fams.data());
  for (uint32_t i = 0; i < n; ++i) {
    if (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      d.family = i;
      d.queue_count = fams[i].queueCount;
      d.timestamp_bits = fams[i].timestampValidBits;
      break;
    }
  }

  if (d.family == UINT32_MAX) d.problem = "no compute queue";
  else if (!d.f12.storageBuffer8BitAccess) d.problem = "no 8-bit storage buffer access";
  else if (!d.f12.shaderInt8) d.problem = "no 8-bit integers in shaders";
  else if (!d.f12.timelineSemaphore) d.problem = "no timeline semaphores";
  if (!d.problem.empty()) return d;

  // Lane-group build (common.glsl). GPUSQZ_VK_LANES=shared forces the
  // shared-memory build, for testing.
  const char* force = std::getenv("GPUSQZ_VK_LANES");
  bool ops = (d.sg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
             (d.sg.supportedOperations & kNeededSubgroupOps) == kNeededSubgroupOps;
  if (force && std::strcmp(force, "shared") == 0) {
    ops = false;
  }
  if (ops) {
    if (d.size_control && (d.sgc.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
        d.sgc.minSubgroupSize <= kLanes && kLanes <= d.sgc.maxSubgroupSize) {
      d.subgroup_lanes = true;
      d.require32 = true;
    } else if (d.sg.subgroupSize >= kLanes) {
      d.subgroup_lanes = true;
    }
  }
  return d;
}

std::vector<VkPhysicalDevice> physical_devices() {
  uint32_t n = 0;
  vkEnumeratePhysicalDevices(g_instance, &n, nullptr);
  std::vector<VkPhysicalDevice> pds(n);
  vkEnumeratePhysicalDevices(g_instance, &n, pds.data());
  pds.resize(n);
  return pds;
}

int type_rank(VkPhysicalDeviceType t) {
  switch (t) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 0;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 1;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return 4;
    default: return 3;
  }
}

std::string lanes_desc(const DevInfo& d) {
  if (!d.subgroup_lanes) return "shared-memory lanes (subgroup size " + std::to_string(d.sg.subgroupSize) + ")";
  if (d.require32) return "subgroup lanes (size 32 required)";
  return "subgroup lanes (subgroup size " + std::to_string(d.sg.subgroupSize) + ")";
}

// ---------------------------------------------------------------------------
// Backend objects
// ---------------------------------------------------------------------------

class VkBackend;

struct Buf {
  VkBuffer b = VK_NULL_HANDLE;
  VkDeviceMemory m = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  VkBackend* be = nullptr;
  Buf() = default;
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  ~Buf();
};

struct VkHostBuf : HostBuf {
  Buf buf;
  ~VkHostBuf() override;
};

struct VkStreamImpl;

bool wait_semaphore(VkDevice dev, VkSemaphore sem, uint64_t v) {
  VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  wi.semaphoreCount = 1;
  wi.pSemaphores = &sem;
  wi.pValues = &v;
  return vkWaitSemaphores(dev, &wi, UINT64_MAX) == VK_SUCCESS;
}

// The timeline semaphore value that follows the recorded work. Events may
// outlive their stream (the timing events do), so they keep the semaphore,
// which lives as long as the backend.
struct VkEventImpl : Event {
  VkSemaphore sem = VK_NULL_HANDLE;
  uint64_t value = 0;
  int query = -1; // timestamp slot, timing events only
};

struct Pipeline {
  VkPipeline p = VK_NULL_HANDLE;
};

// Push constants, as declared in compress.glsl / decompress.glsl.
struct CompressPc {
  uint32_t chunk_size, chunk_count, slot_stride, hash_bits, max_seq;
  int32_t forced_shift;
};
struct DecompressPc {
  uint32_t chunk_size, chunk_count, max_seq;
};
constexpr uint32_t kPcBytes = 32;
constexpr uint32_t kCompressBindings = 16;
constexpr uint32_t kDecompressBindings = 13;

uint32_t max_sequences(uint32_t chunk_size) { return chunk_size / kMinMatch + 1; }
int hash_bits_for(uint32_t chunk_size) { // as kernels.h's hash_table_bits
  return chunk_size <= kDefaultChunkSize ? 11 : chunk_size <= 4 * kDefaultChunkSize ? 12 : 15;
}
size_t hash_bytes_for(uint32_t chunk_size) { return ((size_t)4 << hash_bits_for(chunk_size)) * sizeof(uint32_t); }

// Expanded decode tables of one group: literal rows (n_ctx * kProbScale
// bytes) then the three coarse LUTs, in gsym; packed freq/cum, one u32 per
// quantised count, in gfc (see decompress.glsl).
constexpr uint32_t kSmallLutBytes = 3 * 128;
size_t group_sym_bytes(uint32_t shift) {
  return (((size_t)lit_ctx_count(shift) << kProbBits) + kSmallLutBytes + 15) & ~(size_t)15;
}

class VkBackend : public Backend {
 public:
  VkDevice dev = VK_NULL_HANDLE;
  DevInfo info;
  std::vector<VkQueue> queues;
  size_t next_queue = 0;
  VkPhysicalDeviceMemoryProperties mem{};
  VkDescriptorSetLayout compress_dsl = VK_NULL_HANDLE, decompress_dsl = VK_NULL_HANDLE;
  VkPipelineLayout compress_pl = VK_NULL_HANDLE, decompress_pl = VK_NULL_HANDLE;
  Pipeline parse_hist, build_table, rans_encode, scan, compact, decompress, expand_tables;
  // Timestamp queries for timing events (create_event), made on first use.
  VkQueryPool queries = VK_NULL_HANDLE;
  uint32_t query_count = 0, next_query = 0;
  bool queries_tried = false, queries_warned = false;
  bool budget_ext = false;
  // Semaphores of destroyed streams: events may still refer to them.
  std::vector<VkSemaphore> retired;

  bool init(const DevInfo& d, std::string* why);
  // Only for backends with no work in flight (gpusqz never destroys the one
  // it compresses with; see main.cpp).
  ~VkBackend() override;

  std::string name() override {
    return std::string("Vulkan: ") + info.props.deviceName + ", " + lanes_desc(info);
  }

  size_t free_memory() override;
  uint32_t max_batch(uint32_t chunk_size) override;
  std::unique_ptr<HostBuf> alloc_host(size_t bytes) override;
  std::unique_ptr<Event> create_event(bool timing) override;
  void record(Event& e, Stream& s) override;
  bool wait(Event& e, std::string* why) override;
  double elapsed_ms(Event& from, Event& to) override;

  size_t compress_bytes_per_chunk(uint32_t chunk_size) override {
    // in (+1 for the packed output), slot, sequences, repeat codes,
    // literals, hash table, six u32 per-chunk values.
    return 2 * (size_t)chunk_size + worst_case_size(chunk_size) + 9 * (size_t)max_sequences(chunk_size) +
           hash_bytes_for(chunk_size) + 8 * sizeof(uint32_t);
  }
  size_t decompress_bytes_per_chunk(uint32_t chunk_size) override {
    return 2 * (size_t)chunk_size + worst_case_size(chunk_size) + 8 * (size_t)max_sequences(chunk_size) +
           4 * sizeof(uint32_t);
  }
  std::unique_ptr<CompressSet> create_compress_set(uint32_t batch, uint32_t chunk_size) override;
  std::unique_ptr<DecompressSet> create_decompress_set(uint32_t batch, uint32_t chunk_size,
                                                       const TableCapacity& tables) override;

  // Helpers for the sets.
  bool alloc_buf(Buf& b, VkDeviceSize size, bool host);
  VkQueue take_queue() { return queues[next_queue++ % queues.size()]; }
  VkDescriptorPool make_pool(uint32_t bindings);
  VkDescriptorSet make_set(VkDescriptorPool pool, VkDescriptorSetLayout layout);
  void bind(VkDescriptorSet set, uint32_t binding, const Buf& b);

 private:
  VkDescriptorSetLayout make_dsl(uint32_t bindings);
  VkPipelineLayout make_pl(VkDescriptorSetLayout dsl);
  Pipeline make_pipeline(const uint32_t* code, size_t bytes, VkPipelineLayout layout, bool require32);
  bool probe_lanes(bool subgroup);
};

Buf::~Buf() {
  if (!be) return;
  if (b) vkDestroyBuffer(be->dev, b, nullptr);
  if (m) {
    vkFreeMemory(be->dev, m, nullptr);
  }
}

VkHostBuf::~VkHostBuf() {
  if (buf.m && p) vkUnmapMemory(buf.be->dev, buf.m);
}

// A queue's worth of in-order work (see the file comment).
struct VkStreamImpl : Stream {
  VkBackend* be;
  VkQueue queue;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkSemaphore sem = VK_NULL_HANDLE;
  uint64_t submitted = 0; // last value signalled
  VkCommandBuffer cur = VK_NULL_HANDLE;
  std::deque<std::pair<VkCommandBuffer, uint64_t>> inflight;

  explicit VkStreamImpl(VkBackend* b) : be(b), queue(b->take_queue()) {
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = be->info.family;
    vk_check(vkCreateCommandPool(be->dev, &pci, nullptr, &pool), "vkCreateCommandPool");
    VkSemaphoreTypeCreateInfo tci{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    sci.pNext = &tci;
    vk_check(vkCreateSemaphore(be->dev, &sci, nullptr, &sem), "vkCreateSemaphore");
  }

  ~VkStreamImpl() override {
    if (cur) submit(-1);
    wait_value(submitted);
    for (auto& f : inflight) vkFreeCommandBuffers(be->dev, pool, 1, &f.first);
    vkDestroyCommandPool(be->dev, pool, nullptr);
    be->retired.push_back(sem);
  }

  bool wait_value(uint64_t v) { return wait_semaphore(be->dev, sem, v); }

  // The command buffer to append to, after a barrier making every earlier
  // command's writes (on this queue) visible to what comes next.
  VkCommandBuffer cmd() {
    if (!cur) {
      uint64_t done = 0;
      vkGetSemaphoreCounterValue(be->dev, sem, &done);
      if (!inflight.empty() && inflight.front().second <= done) {
        cur = inflight.front().first;
        inflight.pop_front();
        vk_check(vkResetCommandBuffer(cur, 0), "vkResetCommandBuffer");
      } else {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        vk_check(vkAllocateCommandBuffers(be->dev, &ai, &cur), "vkAllocateCommandBuffers");
      }
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vk_check(vkBeginCommandBuffer(cur, &bi), "vkBeginCommandBuffer");
    }
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                       VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cur, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
    return cur;
  }

  // Submits everything recorded (writes made visible to the host, then a
  // timestamp into `query` if >= 0) and returns the value it signals.
  uint64_t submit(int query) {
    VkCommandBuffer cb = cmd();
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr,
                         0, nullptr);
    if (query >= 0) {
      vkCmdResetQueryPool(cb, be->queries, (uint32_t)query, 1);
      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, be->queries, (uint32_t)query);
    }
    vk_check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");
    uint64_t value = ++submitted;
    VkTimelineSemaphoreSubmitInfo ti{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    ti.signalSemaphoreValueCount = 1;
    ti.pSignalSemaphoreValues = &value;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.pNext = &ti;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &sem;
    vk_check(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
    inflight.emplace_back(cb, value);
    cur = VK_NULL_HANDLE;
    return value;
  }

  void copy(const Buf& src, VkDeviceSize src_off, const Buf& dst, VkDeviceSize dst_off, VkDeviceSize len) {
    if (len == 0) return;
    VkBufferCopy region{src_off, dst_off, len};
    vkCmdCopyBuffer(cmd(), src.b, dst.b, 1, &region);
  }

  void zero(const Buf& b) { vkCmdFillBuffer(cmd(), b.b, 0, VK_WHOLE_SIZE, 0); }

  void dispatch(const Pipeline& p, VkPipelineLayout layout, VkDescriptorSet set, const void* pc, uint32_t pc_bytes,
                uint32_t groups) {
    if (groups == 0) return;
    VkCommandBuffer cb = cmd();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_bytes, pc);
    vkCmdDispatch(cb, groups, 1, 1);
  }
};

VkStreamImpl& raw(Stream& s) { return static_cast<VkStreamImpl&>(s); }

// ---------------------------------------------------------------------------
// VkBackend
// ---------------------------------------------------------------------------

bool VkBackend::alloc_buf(Buf& b, VkDeviceSize size, bool host) {
  b.be = this;
  b.size = std::max<VkDeviceSize>(16, (size + 15) & ~(VkDeviceSize)15);
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = b.size;
  bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
              (host ? 0 : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(dev, &bci, nullptr, &b.b) != VK_SUCCESS) return false;
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(dev, b.b, &req);
  // Memory types in order of preference. Every type that qualifies is
  // tried, because a type can refuse a buffer its flags would suggest it
  // takes (MoltenVK's private storage on Apple's paravirtualized VM GPU).
  // Lazily allocated (tile memory) and protected types never back buffers.
  const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  const VkMemoryPropertyFlags never = VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT | VK_MEMORY_PROPERTY_PROTECTED_BIT;
  std::vector<uint32_t> order;
  auto add = [&](VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
      VkMemoryPropertyFlags f = mem.memoryTypes[i].propertyFlags;
      if ((req.memoryTypeBits & (1u << i)) && (f & want) == want && (f & (avoid | never)) == 0 &&
          std::find(order.begin(), order.end(), i) == order.end()) {
        order.push_back(i);
      }
    }
  };
  if (host) {
    add(hv | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
    add(hv, 0);
  } else {
    add(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    add(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    add(0, 0);
  }
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  for (uint32_t type : order) {
    ai.memoryTypeIndex = type;
    VkResult r = vkAllocateMemory(dev, &ai, nullptr, &b.m);
    if (r == VK_SUCCESS) return vkBindBufferMemory(dev, b.b, b.m, 0) == VK_SUCCESS;
    b.m = VK_NULL_HANDLE;
    if (std::getenv("GPUSQZ_VK_DEBUG")) {
      std::fprintf(stderr, "gpusqz: %llu-byte %s buffer: memory type %u refused (VkResult %d)\n",
                   (unsigned long long)req.size, host ? "host" : "device", type, (int)r);
    }
  }
  return false;
}

VkDescriptorSetLayout VkBackend::make_dsl(uint32_t bindings) {
  std::vector<VkDescriptorSetLayoutBinding> bs(bindings);
  for (uint32_t i = 0; i < bindings; ++i) {
    bs[i] = VkDescriptorSetLayoutBinding{i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
                                         nullptr};
  }
  VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  ci.bindingCount = bindings;
  ci.pBindings = bs.data();
  VkDescriptorSetLayout l;
  vk_check(vkCreateDescriptorSetLayout(dev, &ci, nullptr, &l), "vkCreateDescriptorSetLayout");
  return l;
}

VkPipelineLayout VkBackend::make_pl(VkDescriptorSetLayout dsl) {
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, kPcBytes};
  VkPipelineLayoutCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  ci.setLayoutCount = 1;
  ci.pSetLayouts = &dsl;
  ci.pushConstantRangeCount = 1;
  ci.pPushConstantRanges = &pcr;
  VkPipelineLayout l;
  vk_check(vkCreatePipelineLayout(dev, &ci, nullptr, &l), "vkCreatePipelineLayout");
  return l;
}

Pipeline VkBackend::make_pipeline(const uint32_t* code, size_t bytes, VkPipelineLayout layout, bool require32) {
  VkShaderModuleCreateInfo mci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  mci.codeSize = bytes;
  mci.pCode = code;
  VkShaderModule mod;
  vk_check(vkCreateShaderModule(dev, &mci, nullptr, &mod), "vkCreateShaderModule");
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT rss{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT};
  rss.requiredSubgroupSize = kLanes;
  VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  ci.stage.pNext = require32 ? &rss : nullptr;
  ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  ci.stage.module = mod;
  ci.stage.pName = "main";
  ci.layout = layout;
  Pipeline p;
  vk_check(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p.p), "vkCreateComputePipelines");
  vkDestroyShaderModule(dev, mod, nullptr);
  return p;
}

VkDescriptorPool VkBackend::make_pool(uint32_t bindings) {
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bindings};
  VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  ci.maxSets = 1;
  ci.poolSizeCount = 1;
  ci.pPoolSizes = &ps;
  VkDescriptorPool pool;
  vk_check(vkCreateDescriptorPool(dev, &ci, nullptr, &pool), "vkCreateDescriptorPool");
  return pool;
}

VkDescriptorSet VkBackend::make_set(VkDescriptorPool pool, VkDescriptorSetLayout layout) {
  VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  ai.descriptorPool = pool;
  ai.descriptorSetCount = 1;
  ai.pSetLayouts = &layout;
  VkDescriptorSet s;
  vk_check(vkAllocateDescriptorSets(dev, &ai, &s), "vkAllocateDescriptorSets");
  return s;
}

void VkBackend::bind(VkDescriptorSet set, uint32_t binding, const Buf& b) {
  VkDescriptorBufferInfo bi{b.b, 0, VK_WHOLE_SIZE};
  VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  w.dstSet = set;
  w.dstBinding = binding;
  w.descriptorCount = 1;
  w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  w.pBufferInfo = &bi;
  vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
}

// Runs probe.comp in the given lane-group build and checks every lane's
// results against what 32 lock-step lanes produce.
bool VkBackend::probe_lanes(bool subgroup) {
  VkDescriptorSetLayout dsl = make_dsl(1);
  VkPipelineLayout pl = make_pl(dsl);
  Pipeline p = subgroup ? make_pipeline(spv_probe_1, sizeof(spv_probe_1), pl, info.require32)
                        : make_pipeline(spv_probe_0, sizeof(spv_probe_0), pl, false);
  constexpr uint32_t kGroups = 8, kWords = kGroups * kLanes * 4;
  Buf out, host;
  bool ok = alloc_buf(out, kWords * 4, false) && alloc_buf(host, kWords * 4, true);
  if (!ok && std::getenv("GPUSQZ_VK_DEBUG")) {
    std::fprintf(stderr, "gpusqz: %s lane probe could not allocate its buffers\n", subgroup ? "subgroup" : "shared");
  }
  if (ok) {
    VkDescriptorPool pool = make_pool(1);
    VkDescriptorSet set = make_set(pool, dsl);
    bind(set, 0, out);
    std::vector<uint32_t> res(kWords, 0xDEADBEEF);
    {
      VkStreamImpl st(this);
      st.zero(out);
      uint32_t pc[8] = {};
      st.dispatch(p, pl, set, pc, kPcBytes, kGroups);
      st.copy(out, 0, host, 0, kWords * 4);
      ok = st.wait_value(st.submit(-1));
    }
    if (!ok && std::getenv("GPUSQZ_VK_DEBUG")) {
      std::fprintf(stderr, "gpusqz: %s lane probe did not complete on the GPU\n", subgroup ? "subgroup" : "shared");
    }
    void* mapped = nullptr;
    if (ok && vkMapMemory(dev, host.m, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS) {
      std::memcpy(res.data(), mapped, kWords * 4);
      vkUnmapMemory(dev, host.m);
    } else {
      ok = false;
    }
    for (uint32_t g = 0; g < kGroups && ok; ++g) {
      uint32_t seen = 0;
      for (uint32_t i = 0; i < kLanes && ok; ++i) {
        const uint32_t* r = &res[(g * kLanes + i) * 4];
        uint32_t lane = r[0];
        ok = lane < kLanes && !(seen & (1u << lane)) && r[1] == 0xAAAAAAAAu && r[2] == (31 - lane) * 3 + 1 &&
             r[3] == kLanes * (kLanes + 1) / 2;
        seen |= 1u << (lane & 31);
        if (!ok && std::getenv("GPUSQZ_VK_DEBUG")) {
          std::fprintf(stderr, "gpusqz: %s lane probe failed at group %u invocation %u: %u %08x %u %u\n",
                       subgroup ? "subgroup" : "shared", g, i, r[0], r[1], r[2], r[3]);
        }
      }
    }
    vkDestroyDescriptorPool(dev, pool, nullptr);
  }
  vkDestroyPipeline(dev, p.p, nullptr);
  vkDestroyPipelineLayout(dev, pl, nullptr);
  vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
  return ok;
}

bool VkBackend::init(const DevInfo& d, std::string* why) {
  info = d;
  float prio[2] = {1.f, 1.f};
  uint32_t nq = std::min<uint32_t>(2, d.queue_count);
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = d.family;
  qci.queueCount = nq;
  qci.pQueuePriorities = prio;

  std::vector<const char*> exts;
  if (has_ext(d.exts, "VK_KHR_portability_subset")) exts.push_back("VK_KHR_portability_subset");
  if (d.require32) exts.push_back("VK_EXT_subgroup_size_control");
  budget_ext = has_ext(d.exts, "VK_EXT_memory_budget");
  if (budget_ext) exts.push_back("VK_EXT_memory_budget");

  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  f12.storageBuffer8BitAccess = VK_TRUE;
  f12.shaderInt8 = VK_TRUE;
  f12.timelineSemaphore = VK_TRUE;
  VkPhysicalDeviceSubgroupSizeControlFeaturesEXT fsgc{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT};
  fsgc.subgroupSizeControl = VK_TRUE;
  if (d.require32) f12.pNext = &fsgc;
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  f2.pNext = &f12;

  VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  ci.pNext = &f2;
  ci.queueCreateInfoCount = 1;
  ci.pQueueCreateInfos = &qci;
  ci.enabledExtensionCount = (uint32_t)exts.size();
  ci.ppEnabledExtensionNames = exts.data();
  VkResult r = vkCreateDevice(d.pd, &ci, nullptr, &dev);
  if (r != VK_SUCCESS) {
    *why = std::string(d.props.deviceName) + ": vkCreateDevice failed (VkResult " + std::to_string((int)r) + ")";
    return false;
  }
#define GPUSQZ_VK_LOAD_DEVICE(f) f = (PFN_##f)vkGetDeviceProcAddr(dev, #f);
  GPUSQZ_VK_DEVICE(GPUSQZ_VK_LOAD_DEVICE)
  for (uint32_t i = 0; i < nq; ++i) {
    VkQueue q;
    vkGetDeviceQueue(dev, d.family, i, &q);
    queues.push_back(q);
  }
  vkGetPhysicalDeviceMemoryProperties(d.pd, &mem);
  if (std::getenv("GPUSQZ_VK_DEBUG")) {
    for (uint32_t i = 0; i < mem.memoryHeapCount; ++i) {
      std::fprintf(stderr, "gpusqz: memory heap %u: %llu MiB, flags %#x\n", i,
                   (unsigned long long)(mem.memoryHeaps[i].size >> 20), (unsigned)mem.memoryHeaps[i].flags);
    }
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
      std::fprintf(stderr, "gpusqz: memory type %u: heap %u, flags %#x\n", i, mem.memoryTypes[i].heapIndex,
                   (unsigned)mem.memoryTypes[i].propertyFlags);
    }
  }

  // Pick the lane-group build: the subgroup one if the device qualifies
  // and passes the probe, else shared memory (which must pass too).
  if (info.subgroup_lanes && !probe_lanes(true)) {
    info.subgroup_lanes = false;
    info.require32 = false;
  }
  if (!info.subgroup_lanes && !probe_lanes(false)) {
    *why = std::string(d.props.deviceName) + ": the lane-group self-test failed (GPUSQZ_VK_DEBUG=1 shows why)";
    return false;
  }

  compress_dsl = make_dsl(kCompressBindings);
  decompress_dsl = make_dsl(kDecompressBindings);
  compress_pl = make_pl(compress_dsl);
  decompress_pl = make_pl(decompress_dsl);
  bool sg = info.subgroup_lanes, r32 = info.require32;
#define GPUSQZ_SPV(name) spv_##name, sizeof(spv_##name)
  parse_hist = sg ? make_pipeline(GPUSQZ_SPV(parse_hist_1), compress_pl, r32)
                  : make_pipeline(GPUSQZ_SPV(parse_hist_0), compress_pl, false);
  rans_encode = sg ? make_pipeline(GPUSQZ_SPV(rans_encode_1), compress_pl, r32)
                   : make_pipeline(GPUSQZ_SPV(rans_encode_0), compress_pl, false);
  decompress = sg ? make_pipeline(GPUSQZ_SPV(decompress_1), decompress_pl, r32)
                  : make_pipeline(GPUSQZ_SPV(decompress_0), decompress_pl, false);
  build_table = make_pipeline(GPUSQZ_SPV(build_table_0), compress_pl, false);
  scan = make_pipeline(GPUSQZ_SPV(scan_0), compress_pl, false);
  compact = make_pipeline(GPUSQZ_SPV(compact_0), compress_pl, false);
  expand_tables = make_pipeline(GPUSQZ_SPV(expand_tables_0), decompress_pl, false);
#undef GPUSQZ_SPV
  return true;
}

VkBackend::~VkBackend() {
  if (!dev) return;
  for (Pipeline* p : {&parse_hist, &build_table, &rans_encode, &scan, &compact, &decompress, &expand_tables}) {
    if (p->p) vkDestroyPipeline(dev, p->p, nullptr);
  }
  for (VkPipelineLayout l : {compress_pl, decompress_pl}) {
    if (l) vkDestroyPipelineLayout(dev, l, nullptr);
  }
  for (VkDescriptorSetLayout l : {compress_dsl, decompress_dsl}) {
    if (l) vkDestroyDescriptorSetLayout(dev, l, nullptr);
  }
  if (queries) vkDestroyQueryPool(dev, queries, nullptr);
  for (VkSemaphore s : retired) vkDestroySemaphore(dev, s, nullptr);
  vkDestroyDevice(dev, nullptr);
}

size_t VkBackend::free_memory() {
  // The largest device-local heap; with VK_EXT_memory_budget, what is left
  // of this process's budget on it.
  VkPhysicalDeviceMemoryBudgetPropertiesEXT bp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
  VkPhysicalDeviceMemoryProperties2 mp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
  if (budget_ext) mp.pNext = &bp;
  vkGetPhysicalDeviceMemoryProperties2(info.pd, &mp);
  size_t best = 0, heap_size = 0;
  for (uint32_t i = 0; i < mp.memoryProperties.memoryHeapCount; ++i) {
    const VkMemoryHeap& h = mp.memoryProperties.memoryHeaps[i];
    if (!(h.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) continue;
    size_t avail = (size_t)h.size;
    if (budget_ext) avail = bp.heapBudget[i] > bp.heapUsage[i] ? (size_t)(bp.heapBudget[i] - bp.heapUsage[i]) : 0;
    if (avail > best) {
      best = avail;
      heap_size = (size_t)h.size;
    }
  }
  // On integrated GPUs and Apple silicon the "device-local" heap is the
  // system RAM the file buffers and everything else live in too: offer
  // half of it.
  if (info.props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) best = std::min(best, heap_size / 2);
  return best;
}

uint32_t VkBackend::max_batch(uint32_t chunk_size) {
  // Every buffer must stay within one allocation and storage-buffer range,
  // and a batch is one dispatch of one workgroup per chunk. The largest
  // per-chunk buffer is the sequences (8 bytes per possible sequence).
  size_t per_chunk = std::max<size_t>({(size_t)8 * max_sequences(chunk_size), (size_t)worst_case_size(chunk_size),
                                       hash_bytes_for(chunk_size)});
  size_t limit = std::min<size_t>(info.props.limits.maxStorageBufferRange, (size_t)info.m3.maxMemoryAllocationSize);
  size_t by_mem = limit / (per_chunk + 16);
  return (uint32_t)std::min<size_t>({by_mem, info.props.limits.maxComputeWorkGroupCount[0], UINT32_MAX});
}

std::unique_ptr<HostBuf> VkBackend::alloc_host(size_t bytes) {
  auto h = std::make_unique<VkHostBuf>();
  if (!alloc_buf(h->buf, bytes, true)) return nullptr;
  void* p = nullptr;
  if (vkMapMemory(dev, h->buf.m, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS) return nullptr;
  h->p = static_cast<uint8_t*>(p);
  h->size = bytes;
  return h;
}

std::unique_ptr<Event> VkBackend::create_event(bool timing) {
  auto e = std::make_unique<VkEventImpl>();
  if (!timing) return e;
  // Only GPUSQZ_VERBOSE times anything, so the pool is made on first use
  // rather than for every run. 4096 queries: Metal's counter sample buffers
  // hold 4096 samples, and for a bigger pool MoltenVK logs an error and
  // emulates the timestamps. That is the origin plus 6 marks per batch for
  // 682 batches; real runs use far fewer.
  if (!queries_tried) {
    queries_tried = true;
    if (info.timestamp_bits > 0) {
      VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
      qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
      qpi.queryCount = 4096;
      if (vkCreateQueryPool(dev, &qpi, nullptr, &queries) == VK_SUCCESS) query_count = qpi.queryCount;
      else queries = VK_NULL_HANDLE;
    }
  }
  if (queries && next_query < query_count) {
    e->query = (int)next_query++;
  } else if (queries && !queries_warned) {
    queries_warned = true;
    std::fprintf(stderr, "gpusqz: GPU timing ran out of timestamp queries; later batches are not timed\n");
  }
  return e;
}

void VkBackend::record(Event& e, Stream& s) {
  auto& ev = static_cast<VkEventImpl&>(e);
  ev.sem = raw(s).sem;
  ev.value = raw(s).submit(ev.query);
}

bool VkBackend::wait(Event& e, std::string* why) {
  auto& ev = static_cast<VkEventImpl&>(e);
  if (!ev.sem) return true;
  if (wait_semaphore(dev, ev.sem, ev.value)) return true;
  *why = "vkWaitSemaphores failed (device lost?)";
  return false;
}

double VkBackend::elapsed_ms(Event& from, Event& to) {
  auto& a = static_cast<VkEventImpl&>(from);
  auto& b = static_cast<VkEventImpl&>(to);
  if (a.query < 0 || b.query < 0) return 0;
  uint64_t ta = 0, tb = 0;
  VkQueryResultFlags fl = VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT;
  if (vkGetQueryPoolResults(dev, queries, (uint32_t)a.query, 1, 8, &ta, 8, fl) != VK_SUCCESS ||
      vkGetQueryPoolResults(dev, queries, (uint32_t)b.query, 1, 8, &tb, 8, fl) != VK_SUCCESS) {
    return 0;
  }
  uint64_t mask = info.timestamp_bits >= 64 ? ~0ull : (1ull << info.timestamp_bits) - 1;
  uint64_t d = (tb - ta) & mask;
  return (double)d * info.props.limits.timestampPeriod / 1e6;
}

// ---------------------------------------------------------------------------
// Buffer sets
// ---------------------------------------------------------------------------

class VkCompressSet : public CompressSet {
 public:
  explicit VkCompressSet(VkBackend* be) : be_(be), stream_(be) {}
  ~VkCompressSet() override {
    if (pool_) vkDestroyDescriptorPool(be_->dev, pool_, nullptr);
  }

  bool alloc(uint32_t batch, uint32_t chunk_size) {
    chunk_size_ = chunk_size;
    slot_stride_ = worst_case_size(chunk_size);
    max_seq_ = max_sequences(chunk_size);
    size_t b = batch;
    // `in` also receives the packed output (compact.comp), at most one flag
    // byte per chunk more than the input.
    bool ok = be_->alloc_buf(in_, b * chunk_size + b, false) && be_->alloc_buf(in_lens_, b * 4, false) &&
              be_->alloc_buf(slots_, b * slot_stride_, false) && be_->alloc_buf(start_, b * 4, false) &&
              be_->alloc_buf(sizes_, b * 4, false) && be_->alloc_buf(offsets_, (b + 1) * 4, false) &&
              be_->alloc_buf(seqs_, b * max_seq_ * 8, false) && be_->alloc_buf(rep_, b * max_seq_, false) &&
              be_->alloc_buf(lits_, b * chunk_size, false) &&
              be_->alloc_buf(htab_, b * hash_bytes_for(chunk_size), false) &&
              be_->alloc_buf(n_seq_, b * 4, false) && be_->alloc_buf(n_lit_, b * 4, false) &&
              be_->alloc_buf(cnt_, kMaxQuantBytes * 4, false) && be_->alloc_buf(fc_, kMaxQuantBytes * 4, false) &&
              be_->alloc_buf(q_, kMaxQuantBytes, false) && be_->alloc_buf(shift_, 4, false);
    if (!ok) return false;
    h_lens_ = be_->alloc_host(b * 4);
    // Results: sizes, offsets, shift, q.
    meta_sizes_ = 0;
    meta_offsets_ = b * 4;
    meta_shift_ = meta_offsets_ + (b + 1) * 4;
    meta_q_ = meta_shift_ + 4;
    h_meta_ = be_->alloc_host(meta_q_ + kMaxQuantBytes);
    if (!h_lens_ || !h_meta_) return false;
    pool_ = be_->make_pool(kCompressBindings);
    set_ = be_->make_set(pool_, be_->compress_dsl);
    const Buf* bufs[kCompressBindings] = {&in_, &in_lens_, &slots_, &start_, &sizes_, &offsets_, &seqs_, &rep_,
                                          &lits_, &htab_, &n_seq_, &n_lit_, &cnt_, &fc_, &q_, &shift_};
    for (uint32_t i = 0; i < kCompressBindings; ++i) be_->bind(set_, i, *bufs[i]);
    return true;
  }

  Stream& stream() override { return stream_; }

  uint32_t* begin(uint32_t n) override {
    n_ = n;
    // h_lens was last read by the previous batch's launch, which its meta
    // event follows.
    if (meta_.sem) {
      std::string why;
      if (!be_->wait(meta_, &why)) vk_die(VK_ERROR_DEVICE_LOST, "wait for set");
    }
    return reinterpret_cast<uint32_t*>(h_lens_->p);
  }

  void upload_input(uint64_t off, HostBuf& src, size_t len) override {
    stream_.copy(static_cast<VkHostBuf&>(src).buf, 0, in_, off, len);
  }

  void launch(int forced_lit_shift) override {
    VkStreamImpl& st = stream_;
    st.copy(static_cast<VkHostBuf&>(*h_lens_).buf, 0, in_lens_, 0, (VkDeviceSize)n_ * 4);
    st.zero(cnt_);
    CompressPc pc{chunk_size_, n_, slot_stride_, (uint32_t)hash_bits_for(chunk_size_), max_seq_, forced_lit_shift};
    VkPipelineLayout pl = be_->compress_pl;
    st.dispatch(be_->parse_hist, pl, set_, &pc, sizeof(pc), n_);
    st.dispatch(be_->build_table, pl, set_, &pc, sizeof(pc), 1);
    st.dispatch(be_->rans_encode, pl, set_, &pc, sizeof(pc), n_);
    st.dispatch(be_->scan, pl, set_, &pc, sizeof(pc), 1);
    st.dispatch(be_->compact, pl, set_, &pc, sizeof(pc), n_);
    const Buf& h = static_cast<VkHostBuf&>(*h_meta_).buf;
    st.copy(sizes_, 0, h, meta_sizes_, (VkDeviceSize)n_ * 4);
    st.copy(offsets_, 0, h, meta_offsets_, (VkDeviceSize)(n_ + 1) * 4);
    st.copy(shift_, 0, h, meta_shift_, 4);
    st.copy(q_, 0, h, meta_q_, kMaxQuantBytes);
    be_->record(meta_, st);
  }

  void wait_meta() override {
    std::string why;
    if (!be_->wait(meta_, &why)) vk_die(VK_ERROR_DEVICE_LOST, "wait for chunk sizes");
  }
  const uint32_t* sizes() override { return reinterpret_cast<const uint32_t*>(h_meta_->p + meta_sizes_); }
  uint32_t packed_bytes() override { return reinterpret_cast<const uint32_t*>(h_meta_->p + meta_offsets_)[n_]; }
  uint32_t lit_shift() override { return *reinterpret_cast<const uint32_t*>(h_meta_->p + meta_shift_); }
  const uint8_t* quant() override { return h_meta_->p + meta_q_; }

  void download_output(uint64_t off, HostBuf& dst, size_t len) override {
    stream_.copy(in_, off, static_cast<VkHostBuf&>(dst).buf, 0, len);
  }

 private:
  VkBackend* be_;
  VkStreamImpl stream_;
  VkEventImpl meta_;
  uint32_t chunk_size_ = 0, slot_stride_ = 0, max_seq_ = 0, n_ = 0;
  size_t meta_sizes_ = 0, meta_offsets_ = 0, meta_shift_ = 0, meta_q_ = 0;
  Buf in_, in_lens_, slots_, start_, sizes_, offsets_, seqs_, rep_, lits_, htab_, n_seq_, n_lit_, cnt_, fc_, q_,
      shift_;
  std::unique_ptr<HostBuf> h_lens_, h_meta_;
  VkDescriptorPool pool_ = VK_NULL_HANDLE;
  VkDescriptorSet set_ = VK_NULL_HANDLE;
};

class VkDecompressSet : public DecompressSet {
 public:
  explicit VkDecompressSet(VkBackend* be) : be_(be), stream_(be) {}
  ~VkDecompressSet() override {
    if (pool_) vkDestroyDescriptorPool(be_->dev, pool_, nullptr);
  }

  bool alloc(uint32_t batch, uint32_t chunk_size, const TableCapacity& tables) {
    chunk_size_ = chunk_size;
    max_seq_ = max_sequences(chunk_size);
    batch_ = batch;
    size_t b = batch;
    // Quantised counts and both expanded buffers are linear in groups and
    // contexts (quant_bytes, group_sym_bytes), so the capacity's two maxima
    // bound any batch's total.
    max_groups_ = std::max<uint32_t>(1, tables.groups);
    max_q_ = max_groups_ * (size_t)quant_bytes(0) + tables.contexts * (size_t)kLitSyms;
    size_t sym_per_ctx = (group_sym_bytes(4) - group_sym_bytes(8)) / 15; // 16 contexts vs 1
    max_sym_ = max_groups_ * (group_sym_bytes(8) - sym_per_ctx) + tables.contexts * sym_per_ctx;
    // ginfo holds 32-bit offsets; too many tables for one batch makes the
    // caller retry with a smaller one.
    if (max_sym_ > UINT32_MAX || max_q_ * 4 > UINT32_MAX) return false;
    bool ok =be_->alloc_buf(in_, b * worst_case_size(chunk_size), false) &&
              be_->alloc_buf(in_offsets_, b * 4, false) && be_->alloc_buf(in_lens_, b * 4, false) &&
              be_->alloc_buf(out_lens_, b * 4, false) && be_->alloc_buf(group_id_, b * 4, false) &&
              be_->alloc_buf(out_, b * chunk_size, false) && be_->alloc_buf(seqs_, b * max_seq_ * 8, false) &&
              be_->alloc_buf(lits_, b * ((chunk_size + 3) & ~3u), false) && be_->alloc_buf(err_, 4, false) &&
              be_->alloc_buf(ginfo_, (size_t)max_groups_ * 16, false) && be_->alloc_buf(gsym_, max_sym_, false) &&
              be_->alloc_buf(gfc_, max_q_ * 4, false) && be_->alloc_buf(gq_, max_q_, false);
    if (!ok) return false;
    h_in_ = be_->alloc_host(b * 16);
    h_tab_ = be_->alloc_host((size_t)max_groups_ * 16 + max_q_);
    if (!h_in_ || !h_tab_) return false;
    pool_ = be_->make_pool(kDecompressBindings);
    set_ = be_->make_set(pool_, be_->decompress_dsl);
    const Buf* bufs[kDecompressBindings] = {&in_,  &in_offsets_, &in_lens_, &out_lens_, &group_id_, &out_, &seqs_,
                                            &lits_, &err_,       &ginfo_,   &gsym_,     &gfc_,      &gq_};
    for (uint32_t i = 0; i < kDecompressBindings; ++i) be_->bind(set_, i, *bufs[i]);
    return true;
  }

  Stream& stream() override { return stream_; }

  Inputs begin(uint32_t n) override {
    n_ = n;
    if (done_.sem) {
      std::string why;
      if (!be_->wait(done_, &why)) vk_die(VK_ERROR_DEVICE_LOST, "wait for set");
    }
    auto* h = reinterpret_cast<uint32_t*>(h_in_->p);
    return Inputs{h, h + batch_, h + 2 * (size_t)batch_, h + 3 * (size_t)batch_};
  }

  void upload_input(uint64_t off, HostBuf& src, size_t len) override {
    stream_.copy(static_cast<VkHostBuf&>(src).buf, 0, in_, off, len);
  }

  void launch(const TableWindow& tables, HostBuf& err, size_t err_off) override {
    VkStreamImpl& st = stream_;
    const Buf& h = static_cast<VkHostBuf&>(*h_in_).buf;
    VkDeviceSize arr = (VkDeviceSize)batch_ * 4, bytes = (VkDeviceSize)n_ * 4;
    st.copy(h, 0, in_offsets_, 0, bytes);
    st.copy(h, arr, in_lens_, 0, bytes);
    st.copy(h, 2 * arr, out_lens_, 0, bytes);
    st.copy(h, 3 * arr, group_id_, 0, bytes);
    expand_tables(tables);
    st.zero(err_);
    DecompressPc pc{chunk_size_, n_, max_seq_};
    st.dispatch(be_->decompress, be_->decompress_pl, set_, &pc, sizeof(pc), n_);
    st.copy(err_, 0, static_cast<VkHostBuf&>(err).buf, err_off, 4);
    be_->record(done_, st);
  }

  void download_output(uint64_t off, HostBuf& dst, size_t len) override {
    stream_.copy(out_, off, static_cast<VkHostBuf&>(dst).buf, 0, len);
  }

 private:
  // Stages the window's group info and quantised counts, and queues their
  // upload and expansion into this set's table buffers.
  void expand_tables(const TableWindow& w) {
    auto* info = reinterpret_cast<uint32_t*>(h_tab_->p);
    size_t q_off = 0, sym_bytes = 0, fc_words = 0;
    for (uint32_t g = 0; g < w.count; ++g) {
      uint32_t shift = w.groups[g].lit_ctx_shift;
      info[g * 4 + 0] = (uint32_t)q_off;
      info[g * 4 + 1] = (uint32_t)sym_bytes;
      info[g * 4 + 2] = (uint32_t)fc_words;
      info[g * 4 + 3] = shift;
      q_off += group_quant_bytes(w.groups[g]);
      sym_bytes += group_sym_bytes(shift);
      fc_words += (size_t)quant_bytes(lit_ctx_count(shift));
    }
    if (w.count > max_groups_ || q_off > max_q_ || sym_bytes > max_sym_ || fc_words > max_q_) {
      std::fprintf(stderr, "gpusqz: internal error: a decode batch's tables exceed its set's capacity\n");
      std::exit(1);
    }
    size_t info_bytes = (size_t)max_groups_ * 16;
    std::memcpy(h_tab_->p + info_bytes, w.quant, q_off);
    VkStreamImpl& st = stream_;
    const Buf& h = static_cast<VkHostBuf&>(*h_tab_).buf;
    st.copy(h, 0, ginfo_, 0, (VkDeviceSize)w.count * 16);
    st.copy(h, info_bytes, gq_, 0, q_off);
    // Unused contexts and groups with no LzRans chunks have all-zero
    // counts, which leave their tables zero: what flags a corrupt chunk
    // decoding against one.
    st.zero(gsym_);
    st.zero(gfc_);
    DecompressPc pc{0, w.count, 0};
    st.dispatch(be_->expand_tables, be_->decompress_pl, set_, &pc, sizeof(pc), w.count);
  }

  VkBackend* be_;
  VkStreamImpl stream_;
  VkEventImpl done_;
  uint32_t chunk_size_ = 0, max_seq_ = 0, batch_ = 0, n_ = 0, max_groups_ = 0;
  size_t max_q_ = 0, max_sym_ = 0;
  Buf in_, in_offsets_, in_lens_, out_lens_, group_id_, out_, seqs_, lits_, err_;
  Buf ginfo_, gsym_, gfc_, gq_; // the batch's group tables, see decompress.glsl
  std::unique_ptr<HostBuf> h_in_, h_tab_;
  VkDescriptorPool pool_ = VK_NULL_HANDLE;
  VkDescriptorSet set_ = VK_NULL_HANDLE;
};

std::unique_ptr<CompressSet> VkBackend::create_compress_set(uint32_t batch, uint32_t chunk_size) {
  auto s = std::make_unique<VkCompressSet>(this);
  if (!s->alloc(batch, chunk_size)) return nullptr;
  return s;
}

std::unique_ptr<DecompressSet> VkBackend::create_decompress_set(uint32_t batch, uint32_t chunk_size,
                                                               const TableCapacity& tables) {
  auto s = std::make_unique<VkDecompressSet>(this);
  if (!s->alloc(batch, chunk_size, tables)) return nullptr;
  return s;
}

// Every usable device, best first: GPUSQZ_VK_DEVICE=<index into
// `gpusqz devices`'s Vulkan list> picks one explicitly.
std::vector<DevInfo> ranked_devices() {
  std::vector<DevInfo> all;
  for (VkPhysicalDevice pd : physical_devices()) all.push_back(query_device(pd));
  if (const char* e = std::getenv("GPUSQZ_VK_DEVICE")) {
    size_t i = (size_t)std::strtoul(e, nullptr, 10);
    if (i < all.size()) return {all[i]};
    return {};
  }
  std::stable_sort(all.begin(), all.end(), [](const DevInfo& a, const DevInfo& b) {
    return type_rank(a.props.deviceType) < type_rank(b.props.deviceType);
  });
  return all;
}

} // namespace

std::unique_ptr<Backend> make_vulkan_backend(std::string* why) {
  if (!create_instance(why)) return nullptr;
  std::string notes;
  for (const DevInfo& d : ranked_devices()) {
    if (!d.problem.empty()) {
      notes += std::string(notes.empty() ? "" : "; ") + d.props.deviceName + ": " + d.problem;
      continue;
    }
    auto be = std::make_unique<VkBackend>();
    std::string err;
    if (be->init(d, &err)) return be;
    notes += std::string(notes.empty() ? "" : "; ") + err;
  }
  *why = notes.empty() ? "no Vulkan devices" : notes;
  return nullptr;
}

void list_vulkan_devices() {
  std::string why;
  if (!create_instance(&why)) {
    std::fprintf(stderr, "Vulkan: unavailable (%s)\n", why.c_str());
    return;
  }
  auto pds = physical_devices();
  if (pds.empty()) std::fprintf(stderr, "Vulkan: no devices\n");
  for (size_t i = 0; i < pds.size(); ++i) {
    DevInfo d = query_device(pds[i]);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pds[i], &mp);
    uint64_t local = 0;
    for (uint32_t h = 0; h < mp.memoryHeapCount; ++h) {
      if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) local = std::max<uint64_t>(local, mp.memoryHeaps[h].size);
    }
    std::fprintf(stderr, "Vulkan %zu: %s (%s), Vulkan %u.%u, %llu MiB, subgroup %u", i, d.props.deviceName,
                 type_name(d.props.deviceType), VK_API_VERSION_MAJOR(d.props.apiVersion),
                 VK_API_VERSION_MINOR(d.props.apiVersion), (unsigned long long)(local >> 20), d.sg.subgroupSize);
    if (d.size_control) std::fprintf(stderr, " (%u-%u)", d.sgc.minSubgroupSize, d.sgc.maxSubgroupSize);
    if (!d.problem.empty()) {
      std::fprintf(stderr, ": NOT usable, %s\n", d.problem.c_str());
      continue;
    }
    // Opening the device runs the lane-group self-test, which settles the
    // lane mode actually used.
    VkBackend be;
    std::string err;
    if (be.init(d, &err)) std::fprintf(stderr, ": usable, %s\n", lanes_desc(be.info).c_str());
    else std::fprintf(stderr, ": NOT usable, %s\n", err.c_str());
  }
}

} // namespace gpusqz
