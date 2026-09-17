# Usage

```
gpusqz c <input> <output> [chunk_size]                  # compress (default chunk_size 65536)
gpusqz c <input> <output> --profile speed|balance|ratio # ...or pick a chunk-size preset
gpusqz d <input> <output>                               # decompress
gpusqz c|d ... --gpu-mem 8G                             # GPU memory for batch buffers
gpusqz c|d ... --backend vulkan                         # pick the backend (auto, cuda, vulkan)
gpusqz devices                                          # list GPUs per backend
gpusqz --version
gpusqz_refdec <input.gsz> <output>                      # CPU-only decompressor
```

`chunk_size` and `--profile` are mutually exclusive, and unrecognised
arguments are rejected.

## Profiles

Chunk size trades ratio for speed. Bigger chunks give the match finder
more history to search and larger hash tables, at the cost of fewer,
coarser units of GPU parallelism. The profiles are presets:

| profile | chunk size | match-finder buckets | use it for |
|---|---|---|---|
| `speed` (default) | 64KB | 2048 | throughput |
| `balance` | 256KB | 4096 | a middle ground |
| `ratio` | 1MB (the largest allowed) | 32768 | the smallest output; needs large inputs to fill the GPU |

Decompression reads the chunk size from the file, so it takes no profile.
[Benchmarks](benchmarks.md) has what each profile costs and saves.

## GPU memory

`--gpu-mem SIZE` (or `GPUSQZ_GPU_MEM`) sets how much GPU memory gpusqz may use
for its batch buffers: a number with a K, M, G or T suffix, or a bare
number of MiB. By default gpusqz takes 80% of the GPU memory free when it
starts; an explicit value may use all but 256MiB of the free
memory and is reduced, with a note, if it asks for more. On GPUs that
share system RAM (Apple silicon, integrated GPUs) "free GPU memory" is
counted as at most half of that RAM, so the default stays at 40% of it. Host RAM use
does not depend on it: gpusqz pins a fixed ~96MB of staging buffers.

The memory is allocated once, at startup, and held until gpusqz exits, so
another process can't take it mid-run. See [Host pipeline](design.md#host-pipeline)
for how batches are sized from it.

More memory than the default rarely makes gpusqz faster: once about 350
chunks are in flight the GPU is saturated. On a 16GB GPU, a 1GB file at
the `ratio` profile ran at the same kernel speed with a 4GB budget (three
batches) as with the default on an idle GPU (one batch). Much less does
cost speed: a 2GB budget measured 900 MB/s against 1550 MB/s.

## Backends

`--backend auto|cuda|vulkan` (or `GPUSQZ_BACKEND`) overrides the choice
of backend: by default CUDA when an NVIDIA GPU and driver are present,
Vulkan otherwise. `gpusqz devices` lists what each backend finds. See
[GPU backends](backends.md).

## Environment variables

All optional:

| variable | effect |
|---|---|
| `GPUSQZ_GPU_MEM=<size>` | Same as `--gpu-mem`. |
| `GPUSQZ_BACKEND=auto\|cuda\|vulkan` | Same as `--backend`. |
| `GPUSQZ_VERBOSE=1` | Per-stage timing on stderr: setup, fread, copies, kernel time (summed and wall-clock union), fwrite, staging stalls. |
| `GPUSQZ_FORCE_BATCH=<n>` | Force chunks per batch (testing; see [Testing](testing.md)). |
| `GPUSQZ_FORCE_GROUP_CHUNKS=<n>` | Force chunks per table group (testing; see [Testing](testing.md)). |
| `GPUSQZ_FORCE_LIT_SHIFT=<0\|4\|8>` | Force every group's literal-context rule (testing and tuning). |
| `GPUSQZ_VK_DEVICE=<n>` | Use Vulkan device *n* from `gpusqz devices` (default: the first discrete GPU, then integrated, then others). |
| `GPUSQZ_VK_LANES=shared` | Force shared-memory lanes (testing). |
| `GPUSQZ_VULKAN_LIB=<path>` | Load this Vulkan library instead of the system loader (or bundled MoltenVK). |
| `GPUSQZ_VK_DEBUG=1` | Print why the subgroup-lane probe failed, if it does, and any memory allocation the driver refuses. |
