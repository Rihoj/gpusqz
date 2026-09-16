# gpusqz — a GPU file compressor

`gpusqz` (pronounced "GPU squeeze") compresses and decompresses files on
the GPU; compressed files use the `.gsz` extension. It runs on NVIDIA GPUs
with CUDA and on AMD, Apple silicon and Intel GPUs with Vulkan, and both
backends produce the same files. Each chunk of the input (64KB by default,
up to 1MB) is handled by one group of 32 lanes (a CUDA warp): an LZ parse
where all 32 lanes search for matches together, followed by a 32-way
interleaved rANS entropy coder with order-1 literal contexts. It's a
from-scratch, educational implementation — not a drop-in replacement for
zstd — but at its default setting it compresses 1.4–2.5x faster than
single-threaded `zstd -1` at about the same ratio, and its `ratio`
profile lands within 0.5% of `zstd -3`'s output size while compressing
1.3–2.2x faster. zstd still decompresses faster.

## Results at a glance

Varied 1GB corpus (every distinct `/usr/include` header, no repetition),
CUDA backend on an RTX 5060 Ti (16GB) under WSL2 with the GPU otherwise
idle, CPU tools single-threaded, best of 3–5 runs. MB/s of original data;
ratio is output/input, so **lower is better**.

| codec | compress MB/s | decompress MB/s | ratio |
|---|---|---|---|
| **gpusqz** (default, `speed`, 64KB chunks) | **1217** | 1234 | 0.2512 |
| gpusqz `--profile balance` (256KB) | 946 | 1120 | 0.2373 |
| gpusqz `--profile ratio` (1MB) | 837 | 1058 | **0.2243** |
| gzip -1 | 147 | 251 | 0.2836 |
| zstd -1 (1 thread) | 509 | **1384** | 0.2518 |
| zstd -3 (1 thread) | 400 | 1293 | 0.2245 |

These are whole-process wall-clock figures, including ~0.2s of CUDA
startup. The GPU kernels alone run 1.8–3.6x faster. More corpora, an Apple
M1 Max, and how to reproduce the numbers are in
[docs/benchmarks.md](docs/benchmarks.md). How the numbers got here is in
[docs/performance-history.md](docs/performance-history.md).

## Install

Every [GitHub release](https://github.com/Rihoj/gpusqz/releases) carries
packages for:

| platform | installer | portable archive | backends |
|---|---|---|---|
| Ubuntu 22.04+, Debian 12+ | `.deb` | – | CUDA, Vulkan |
| RHEL/Rocky/Alma 8+, Fedora | `.rpm` | – | CUDA, Vulkan |
| Windows 10/11 x64 | `.msi` | `.zip` | CUDA, Vulkan |
| macOS 11+ (Apple silicon and Intel) | `.pkg` | `.tar.gz` | Vulkan (MoltenVK) |

The Windows and macOS installers are not signed yet. What to click past,
and what each GPU needs installed, is in
[docs/installing.md](docs/installing.md). To build from source, see
[docs/building.md](docs/building.md).

## Use

```
gpusqz c <input> <output.gsz>                   # compress (64KB chunks, the `speed` profile)
gpusqz c <input> <output.gsz> --profile ratio   # smaller output: speed | balance | ratio
gpusqz d <input.gsz> <output>                   # decompress
gpusqz devices                                  # list the GPUs each backend finds
gpusqz_refdec <input.gsz> <output>              # decompress on the CPU, no GPU needed
```

The options, the environment variables and GPU memory use are covered in
[docs/usage.md](docs/usage.md).

## Documentation

| document | what's in it |
|---|---|
| [Installing](docs/installing.md) | packages, installers and uninstalling, per-GPU runtime requirements |
| [Usage](docs/usage.md) | command line, profiles, `--gpu-mem`, environment variables |
| [Design](docs/design.md) | how it works: chunks and lanes, LZ parse, rANS, decoding, host pipeline, file format |
| [GPU backends](docs/backends.md) | CUDA and Vulkan, lane modes on AMD/Apple/Intel, Vulkan speed |
| [Benchmarks](docs/benchmarks.md) | current results against gzip and zstd, and how to measure |
| [Performance history](docs/performance-history.md) | every measured change since the first commit, and the experiments that didn't pay off |
| [Building](docs/building.md) | toolchains, CMake options, packages |
| [Testing](docs/testing.md) | the test suites, the CPU reference decoder, Vulkan without a GPU |
| [Releases and versioning](docs/releasing.md) | semantic versioning from Conventional Commits, how CI releases |
| [Known limitations](docs/limitations.md) | what's missing or slow, and what might come next |
| [Compression survey](docs/compression-survey.md) | how other open-source compressors trade speed for ratio, and what applies here |

## Status

gpusqz is at 0.x: the `.gsz` format and the command line can still change
between minor versions, and older files may stop decoding. It has been run
on NVIDIA (CUDA and Vulkan) and on an Apple M1 Max, and not yet on AMD
hardware. See [docs/limitations.md](docs/limitations.md).
