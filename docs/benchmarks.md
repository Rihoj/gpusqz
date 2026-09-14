# Benchmarks

The current numbers, and how to measure new ones. How they changed over
time, and what was tried and dropped along the way, is in [Performance
history](performance-history.md).

## Current results: RTX 5060 Ti, CUDA

CUDA backend on an RTX 5060 Ti (16GB) under WSL2, CPU tools
single-threaded, best of 3–5 runs per column (`REPEAT=<n>`, see
[Running the benchmark](#running-the-benchmark)). Wall figures are
whole-process (file I/O, PCIe copies, and for gpusqz ~0.2s of CUDA context
creation and allocation); the kernel columns are the wall-clock time any
gpusqz kernel was running. Ratio is output/input, so **lower is better**.

Measured at commit ec7aae3 (2026-09-13), before the 0.1 releases. None of
the changes since then was expected to move these numbers; the one that
touched a kernel measured ~1.2–1.5% faster decompression (see [Performance
history](performance-history.md#1gb-varied-corpus)).

**Measurement conditions.** The GPU was otherwise idle (~1GB used by the
desktop, 0–2% utilisation), and gpusqz's batch budget was 4GB (the default
at the time; it is now 80% of free GPU memory, see [Usage](usage.md#gpu-memory)). On
the 283MB corpus about half of gpusqz's wall time is the ~0.2s fixed startup
cost, which varies by ±15% between runs, so its wall figures there move
by that much from run to run; the kernel figures and the 1GB corpus are
the steadier comparison. An earlier set of runs with another process
holding 11.5GB of VRAM gave kernel figures and ratios within 1% of these.

Varied 1GB corpus (every distinct `/usr/include` header, concatenated
without repetition; see [Corpora](#corpora)):

| codec | compress MB/s (wall) | kernel MB/s | decompress MB/s (wall) | kernel MB/s | ratio |
|---|---|---|---|---|---|
| **gpusqz** (default, `speed`, 64KB) | **1297** | 2983 | 1130 | 4567 | 0.2526 |
| gpusqz `--profile balance` (256KB) | 989 | 1847 | 1267 | 3353 | 0.2387 |
| gpusqz `--profile ratio` (1MB) | 873 | 1552 | 1176 | 1906 | 0.2255 |
| gzip -1 | 145 | – | 250 | – | 0.2836 |
| gzip -6 | 56 | – | 278 | – | 0.2306 |
| zstd -1 (1 thread) | 516 | – | **1434** | – | 0.2518 |
| zstd -3 (1 thread) | 401 | – | 1335 | – | **0.2245** |

Repetitive 283MB corpus (one 5.4MB block of headers repeated 48 times — a
friendlier shape for a match finder, see [Corpora](#corpora)):

| codec | compress MB/s (wall) | kernel MB/s | decompress MB/s (wall) | kernel MB/s | ratio |
|---|---|---|---|---|---|
| **gpusqz** (default, `speed`, 64KB) | **664** | 2390 | 718 | 3805 | 0.2673 |
| gpusqz `--profile balance` (256KB) | 552 | 1746 | 715 | 3414 | 0.2547 |
| gpusqz `--profile ratio` (1MB) | 487 | 1232 | 657 | 1515 | **0.2423** |
| gzip -1 | 140 | – | 249 | – | 0.2985 |
| gzip -6 | 53 | – | 273 | – | 0.2457 |
| zstd -1 (1 thread) | 482 | – | **1328** | – | 0.2724 |
| zstd -3 (1 thread) | 373 | – | 1188 | – | 0.2426 |

Against the CPU tools:

- **Default profile vs `zstd -1`.** gpusqz compresses 1.4x (283MB) to 2.5x
  (1GB) faster. Its output is 1.9% smaller on the repetitive corpus and
  0.3% larger on the varied one. `zstd -1` decompresses faster: 1.85x on
  the smaller file, where gpusqz's fixed startup cost weighs more, and 1.3x
  on the 1GB file.
- **`ratio` profile vs `zstd -3`.** gpusqz compresses 1.3–2.2x faster. Its
  output is 0.1% smaller on the repetitive corpus and 0.5% larger on the
  varied one. `zstd -3` decompresses 1.1–1.8x faster.
- **vs `gzip`.** Every gpusqz profile beats `gzip -1` on both ratio and
  speed. The `ratio` profile also beats `gzip -6`'s ratio on both corpora
  while compressing 9–16x faster; `balance` does not beat `gzip -6`.

The gpusqz wall figures on the 1GB corpus are bounded by file I/O more than
by the GPU: under WSL2, decompression spends most of its time in
`fwrite` (see [Known limitations](limitations.md)), and the kernels run
1.6–4x faster than the wall-clock rate.

## enwik8 on an Apple M1 Max and the RTX 5060 Ti

enwik8 (the first 100MB of a 2006 English Wikipedia dump, the corpus of
the Large Text Compression Benchmark) on two machines, best of 3,
wall-clock MB/s of original data, 2026-09-14:

- **Apple M1 Max** (32GB, macOS): the macOS release package, Vulkan
  backend through MoltenVK, native.
- **RTX 5060 Ti** (16GB): CUDA backend under WSL2, GPU otherwise idle.

| codec | M1 Max compress | M1 Max decompress | RTX 5060 Ti compress | RTX 5060 Ti decompress | ratio |
|---|---|---|---|---|---|
| gpusqz `speed` (default) | 386 | 704 | 347 | 357 | 0.3862 |
| gpusqz `balance` | **442** | **751** | 323 | 344 | 0.3744 |
| gpusqz `ratio` | 221 | 414 | 227 | 308 | 0.3592 |
| gzip -1 | 113 | 621 | 112 | 218 | 0.4226 |
| zstd -1 (1 thread) | 425 | 1041 | 442 | **1298** | 0.4067 |
| zstd -3 (1 thread) | 250 | 892 | 284 | 1098 | **0.3544** |

Both machines wrote `.gsz` files of identical sizes, to the byte, as
the two backends should. On a 100MB input the wall figures are decided by
fixed costs more than by the GPU. `GPUSQZ_VERBOSE=1` splits them out:

| enwik8 | M1 Max | RTX 5060 Ti |
|---|---|---|
| setup (GPU context + allocation) | 0.05–0.085s | 0.18–0.23s |
| `speed` compress kernels | 654 MB/s | 1834 MB/s |
| `speed` decompress kernels | ~2700 MB/s | ~9700 MB/s |
| `ratio` compress kernels | 265 MB/s | 481 MB/s |
| `ratio` decompress kernels | ~650 MB/s | ~1800 MB/s |

- **The Mac wins on wall time because it starts ~3x faster.** CUDA
  context creation under WSL2 costs ~0.2s per run, a native Metal device
  well under 0.1s. That outweighs the RTX's faster kernels on a file this
  size, most of all when decompressing (kernels take 0.04s on the Mac,
  0.01s on the RTX).
- **The RTX kernels are 1.8–3.6x faster.** Part of that gap is the
  backend, not the GPU: on the RTX itself the Vulkan decompress kernel
  runs at 55–70% of CUDA's speed (see [GPU backends](backends.md#vulkan-speed)).
- **The M1 Max's kernel figures are approximate.** The build measured
  asked for a timestamp query pool larger than Metal's 4096 samples, so
  MoltenVK emulated the timestamps (it logged a
  `VK_ERROR_OUT_OF_DEVICE_MEMORY` line about `MTLCounterSampleBuffer`).
  v0.1.1 and later cap the pool at 4096. Setup and wall times are exact.
- **The `ratio` profile needs larger inputs.** 100MB is only 96 of its
  1MB chunks, far too few to fill either GPU.
- **Against zstd:** `balance` on the M1 Max beats `zstd -1` on both
  speed and ratio. `ratio` comes within 1.4% of `zstd -3`'s output size;
  zstd still decompresses faster on both machines.

## Running the benchmark

```
REPEAT=5 bash bench/run_bench.sh corpus_varied.txt                       # gpusqz default vs gzip/zstd
CPU=0 REPEAT=5 bash bench/run_bench.sh corpus_varied.txt --profile ratio # one gpusqz profile only
```

`bench/run_bench.sh` reports wall and kernel MB/s for gpusqz and compares
against `gzip -1/-6` and single-threaded `zstd -1/-3` when available. On
a shared GPU a single wall-clock measurement can be dominated by another
process, so `REPEAT=<n>` runs each codec n times and reports the best run
per column. `GPUSQZ=<path>` benchmarks another gpusqz binary, for A/B
runs against an older build. The script needs GNU `stat` and `date`
(on macOS, GNU coreutils first on the PATH).

Check `nvidia-smi` first: another process's memory shrinks gpusqz's
batches, and its compute slows gpusqz's global-memory-latency-bound
parse. Under ~40% contention from another process, the `ratio` compress
kernel once measured at half its real speed.

### Which numbers to read

- **Kernel MB/s** (`kbusy` in `GPUSQZ_VERBOSE=1` output: the union of
  kernel spans) is what the codec sustains. Wall MB/s adds ~0.2s of CUDA
  context creation and allocation under WSL2, and large decompressions
  are bound by `fwrite`.
- **Throughput follows the chunks in flight.** About 350 1MB chunks
  saturate the RTX 5060 Ti; a 100MB input starves the `ratio` profile.
- **Report the conditions with every number**: GPU, backend, driver, free
  VRAM and utilisation before the run, and the corpus.

### Corpora

```
find /usr/include -name '*.h' | xargs cat > corpus_varied.txt      # all distinct files
find /usr/include -name '*.h' | head -400 | xargs cat > block.txt
for i in $(seq 48); do cat block.txt; done > corpus_283mb.txt      # repetitive
```

Repeat `find`/`xargs cat` against more directories to reach a target size
while keeping the content non-repeating; the 1GB corpus above was built
that way. Because it depends on what's installed, the exact bytes differ
between machines, so compare gpusqz against gzip and zstd on the same
file rather than across machines.

**The repetitive recipe repeats one 5.4MB block 48x, which is not a
neutral choice of large file.** It once hid a real regression: a
match-finding table change that measured as basically free on it nearly
halved compress-kernel throughput on a genuinely varied file (see
[Performance history](performance-history.md#corrected-claims)). Measure
LZ-parse or table changes on a varied file too.

[enwik8](https://mattmahoney.net/dc/textdata.html) is a standard text
corpus that anyone can download, which makes it the easiest to compare
across machines.

### Recording results

When a change moves speed or ratio, or a new GPU is measured, add an
entry to [Performance history](performance-history.md). Update the tables
on this page (and the summary in the top-level README) when the current
numbers change.
