// Go/no-go prototype for a context-mixing profile: how fast can a 32-lane
// lockstep integer CM model run on the GPU?
//
//   cm_gpu <file> [chunk_bytes] [hash_bits] [max_chunks]
//
// One warp per chunk, the chunk split into 32 segments coded in lockstep,
// all sharing one model, exactly as docs/predictive-modeling-study.md
// describes. Everything is integer, since both backends would have to
// agree bit for bit (see the floating-point note in
// docs/format-aware-transforms-study.md).
//
// It measures the model, not a codec: it runs predict + update for every
// bit, which is the work decode does and the bulk of what encode does, but
// it writes no bitstream (an arithmetic coder would add a few percent).
// Cross-entropy is accumulated so the ratio stays honest, and the model
// state is per chunk so the VRAM figure is the real one.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

#define CHECK(x)                                                                        \
  do {                                                                                  \
    cudaError_t e_ = (x);                                                               \
    if (e_ != cudaSuccess) {                                                            \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      std::exit(1);                                                                     \
    }                                                                                   \
  } while (0)

constexpr int kOrders = 6;              // orders 0,1,2,3,4,6
constexpr int kInputs = kOrders + 1;    // + bias
constexpr int kSel = 256;               // one weight set per partial byte
constexpr int kProbBits = 12;

__constant__ int c_order[kOrders] = {0, 1, 2, 3, 4, 6};

// lpaq's stretch/squash, integer. stretch(p) = ln(p/(1-p)) in 8.8 fixed
// point, squash is its inverse; both are pure tables so every backend
// would agree.
__device__ __forceinline__ int squash_i(int x) {
  static const int t[33] = {1,    2,    3,    6,    10,   16,   27,   45,   73,   120,  194,
                            310,  488,  747,  1101, 1546, 2047, 2549, 2994, 3348, 3607, 3785,
                            3901, 3975, 4022, 4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094};
  if (x > 2047) return 4095;
  if (x < -2047) return 1;
  int w = x & 127;
  x = (x >> 7) + 16;
  return (t[x] * (128 - w) + t[x + 1] * w + 64) >> 7;
}

// Per-chunk model state, all in global memory.
struct Model {
  uint32_t* tables;  // kOrders * (1 << bits) slots: p in high 22 bits, count in low 10
  int32_t* weights;  // kSel * kInputs, 16.16 fixed point
};

__device__ __forceinline__ int stretch_i(uint32_t p12) {
  // ln(p/(1-p)) * 256, computed from the table-free approximation lpaq uses
  // in reverse: a binary search over squash would be exact; this is the
  // cheap monotone approximation and is good enough for a speed prototype.
  float p = (float)(p12 + 1) / 4098.0f;
  float s = __logf(p / (1.0f - p)) * 256.0f;
  return (int)s;
}

__device__ __forceinline__ uint64_t ctx_hash(const uint8_t* seg, uint32_t pos, int order) {
  if (order == 0) return 0;
  uint64_t x = 0x9E3779B97F4A7C15ull * (uint64_t)(order + 1);
  int k = order < (int)pos ? order : (int)pos;
  for (int j = 0; j < k; ++j) x = (x + seg[pos - 1 - j] + 1) * 0x100000001B3ull;
  return x ^ ((uint64_t)k << 56);
}

__global__ __launch_bounds__(32) void cm_kernel(const uint8_t* in, uint32_t chunk_size, uint32_t chunk_count,
                                                uint32_t* tables_all, int32_t* weights_all, int bits,
                                                unsigned long long* bits_out) {
  uint32_t c = blockIdx.x;
  if (c >= chunk_count) return;
  int lane = threadIdx.x & 31;

  size_t table_words = (size_t)kOrders << bits;
  uint32_t* tables = tables_all + (size_t)c * table_words;
  int32_t* weights = weights_all + (size_t)c * kSel * kInputs;

  // Reset the model for this chunk.
  for (size_t i = lane; i < table_words; i += 32) tables[i] = 1u << 31;
  for (int i = lane; i < kSel * kInputs; i += 32) weights[i] = 1 << 14; // 0.25 in 16.16
  __syncwarp();

  const uint8_t* chunk = in + (size_t)c * chunk_size;
  uint32_t seg_len = chunk_size / 32;
  const uint8_t* seg = chunk + (size_t)lane * seg_len;

  uint32_t mask = (1u << bits) - 1;
  unsigned long long my_bits = 0;

  for (uint32_t pos = 0; pos < seg_len; ++pos) {
    uint64_t h[kOrders];
#pragma unroll
    for (int i = 0; i < kOrders; ++i) h[i] = ctx_hash(seg, pos, c_order[i]);
    uint32_t byte = seg[pos];
    uint32_t c0 = 1;
    for (int b = 7; b >= 0; --b) {
      int y = (byte >> b) & 1;
      uint32_t idx[kOrders];
      int st[kInputs];
#pragma unroll
      for (int i = 0; i < kOrders; ++i) {
        uint64_t k = (h[i] + c0 * 0xD6E8FEB86659FD93ull) * 0x9E3779B97F4A7C15ull;
        idx[i] = (uint32_t)(k >> (64 - bits)) & mask;
        st[i] = stretch_i(tables[((size_t)i << bits) + idx[i]] >> 20);
      }
      st[kOrders] = 256; // bias input

      const int32_t* w = weights + (size_t)c0 * kInputs;
      long long dot = 0;
#pragma unroll
      for (int i = 0; i < kInputs; ++i) dot += (long long)w[i] * st[i];
      int d = (int)(dot >> 16);
      int p = squash_i(d >> 1);

      // Cross-entropy at 12-bit probabilities, as rANS would code it.
      int pq = p < 1 ? 1 : p > 4095 ? 4095 : p;
      float pr = (float)(y ? pq : 4096 - pq) / 4096.0f;
      my_bits += (unsigned long long)(-__log2f(pr) * 1024.0f);

      // Mixer update (integer), then the tables. Lanes share the model, so
      // every cross-lane write happens after a sync, in lane order.
      int err = ((y << 12) - p) * 12;
      __syncwarp();
      for (int l = 0; l < 32; ++l) {
        if (lane == l) {
#pragma unroll
          for (int i = 0; i < kInputs; ++i) {
            int32_t* wp = weights + (size_t)c0 * kInputs + i;
            *wp += (err * st[i]) >> 10;
          }
#pragma unroll
          for (int i = 0; i < kOrders; ++i) {
            uint32_t* sp = &tables[((size_t)i << bits) + idx[i]];
            uint32_t s = *sp;
            uint32_t n = s & 1023;
            int pv = (int)(s >> 10);
            int target = y ? (1 << 22) - 1 : 0;
            pv += (target - pv) / (int)(n + 2);
            if (n < 1023) ++n;
            *sp = ((uint32_t)pv << 10) | n;
          }
        }
        __syncwarp();
      }
      c0 = c0 * 2 + (uint32_t)y;
    }
  }
  atomicAdd(bits_out, my_bits);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: cm_gpu <file> [chunk_bytes] [hash_bits] [max_chunks]\n");
    return 1;
  }
  uint32_t chunk_size = argc > 2 ? (uint32_t)atoi(argv[2]) : 65536;
  int bits = argc > 3 ? atoi(argv[3]) : 16;
  uint32_t max_chunks = argc > 4 ? (uint32_t)atoi(argv[4]) : 256;

  FILE* f = fopen(argv[1], "rb");
  if (!f) return 1;
  fseek(f, 0, SEEK_END);
  size_t n = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint32_t chunks = (uint32_t)(n / chunk_size);
  if (chunks > max_chunks) chunks = max_chunks;
  size_t bytes = (size_t)chunks * chunk_size;
  std::vector<uint8_t> host(bytes);
  if (fread(host.data(), 1, bytes, f) != bytes) return 1;
  fclose(f);

  uint8_t* d_in = nullptr;
  uint32_t* d_tables = nullptr;
  int32_t* d_weights = nullptr;
  unsigned long long* d_bits = nullptr;
  size_t table_words = (size_t)kOrders << bits;
  size_t model_bytes = (size_t)chunks * (table_words * 4 + (size_t)kSel * kInputs * 4);
  CHECK(cudaMalloc(&d_in, bytes));
  CHECK(cudaMalloc(&d_tables, (size_t)chunks * table_words * 4));
  CHECK(cudaMalloc(&d_weights, (size_t)chunks * kSel * kInputs * 4));
  CHECK(cudaMalloc(&d_bits, 8));
  CHECK(cudaMemcpy(d_in, host.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(d_bits, 0, 8));

  cudaEvent_t t0, t1;
  CHECK(cudaEventCreate(&t0));
  CHECK(cudaEventCreate(&t1));
  CHECK(cudaEventRecord(t0));
  cm_kernel<<<chunks, 32>>>(d_in, chunk_size, chunks, d_tables, d_weights, bits, d_bits);
  CHECK(cudaEventRecord(t1));
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
  float ms = 0;
  CHECK(cudaEventElapsedTime(&ms, t0, t1));

  unsigned long long coded = 0;
  CHECK(cudaMemcpy(&coded, d_bits, 8, cudaMemcpyDeviceToHost));
  double out_bytes = (double)coded / 1024.0 / 8.0;
  std::printf("%s chunk=%u hash_bits=%d chunks=%u: %.1f MB in %.1f ms = %.1f MB/s, ratio %.4f, model %.0f MB (%.2f MB/chunk)\n",
              argv[1], chunk_size, bits, chunks, bytes / 1e6, ms, bytes / 1e6 / (ms / 1000.0),
              out_bytes / (double)bytes, model_bytes / 1e6, model_bytes / 1e6 / chunks);
  return 0;
}
