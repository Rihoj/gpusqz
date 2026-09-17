# Performance history

Every measured change to gpusqz's speed or ratio, oldest first, plus the
experiments that were measured and dropped. The current numbers are in
[Benchmarks](benchmarks.md). This page is for comparing a new result with
the past, and for not re-running an experiment that has already failed.

Entries before this page existed (2026-09-14) were collected from the
benchmark tables in earlier versions of the README and from commit
messages. Where those two sources disagree, the later measurement is
given.

## How to read it

- **All measurements so far are on one machine**: an RTX 5060 Ti (16GB)
  under WSL2, CUDA backend, unless a row says otherwise. CPU tools are
  single-threaded.
- **Compare within a corpus, never across corpora.** Even the "same"
  corpus changed once: the 283MB file was rebuilt between 4b943e3 and
  473a54a (`zstd -1`'s ratio on it went from 0.252 to 0.2724), so ratios
  before and after that line don't compare. Each corpus section gives
  `zstd -1`'s ratio on its file as a fingerprint.
- **Conditions matter.** The GPU is shared with other work (often an
  `ollama` model holding 11–15GB of VRAM). Contention halved some kernel
  figures in the past (see [Corrected claims](#corrected-claims)). The
  conditions column says what was running.
- **Kernel MB/s is the steadier number.** Wall MB/s includes ~0.2s of
  CUDA startup under WSL2, which is half the wall time on the 283MB
  corpus. Measurements taken from 2026-09-13 on (the bench script from
  5ae5e60) use `kbusy`, the wall-clock union of kernel spans; older
  rows summed the kernel times. When kernels overlap the union is
  shorter than the sum, so `kbusy` MB/s reads higher.
- Ratio is output/input: **lower is better**. MB/s are of original data.

## Lessons

What the history below taught, in short:

1. **Throughput follows the chunks in flight.** One warp parses one
   chunk at a few MB/s, latency-bound. The biggest single speedup (6x at
   the `ratio` profile) came from sizing batches for occupancy, not from
   touching a kernel. Anything that costs occupancy — shared memory per
   block, registers, small batches — costs speed.
2. **Measure match-finder changes on a varied corpus.** The repetitive
   283MB corpus hid a 2x regression.
3. **Measure on an idle GPU, A/B back to back.** Contention from another
   process made one change look 2x slower than it was and invented a
   regression in another.
4. **Tuning results expire.** Table sizes rejected as "2x slower" with ~32
   warps in flight cost 3–4% once batches held ~1000.

## Milestones

One line per change that moved speed or ratio. "94MB", "283MB" and "1GB"
are the corpora the change was measured on (see the sections below).

| date | commit | release | change | measured effect |
|---|---|---|---|---|
| 2026-09-12 | ea11ade | | First version: one thread per chunk, LZSS, 8KB window | ~98MB of repeated headers, GPU shared: 163 MB/s compress, 209 MB/s decompress (wall), ratio 0.535 (`gzip -1` 0.273) |
| 2026-09-12 | d943678 | | Per-stage timing (`GZP_VERBOSE`) | 94MB: compress kernel ~500 MB/s, decompress kernel ~5 GB/s; ~0.3s of the ~0.55s wall is fixed setup |
| 2026-09-12 | 9715a16 | | Pipelined host: pinned buffers, 3-stream ring, output compacted on the GPU | 94MB: compress wall 0.66 → 0.44s, decompress 0.49 → 0.28s; download 106 → 52.5MB |
| 2026-09-12 | 6761987 | | One warp per chunk, LZ4-style tokens (format v2) | 94MB: ratio 0.535 → 0.408, compress kernel 614 → 1390 MB/s, decompress kernel 4.5 GB/s |
| 2026-09-12 | 871ee24 | | Match finding across all 32 lanes | 94MB: compress kernel 1390 → 4060 MB/s, ratio 0.408 → 0.394 |
| 2026-09-12 | 96857ad | | Interleaved rANS entropy stage | 94MB: ratio 0.394 → 0.295 (32KB chunks), 0.281 (64KB); compress kernel ~1.9 GB/s, decompress ~1.0 GB/s |
| 2026-09-12 | bbd0a9a | | Batches sized by bytes (16MB) instead of chunk count | compress wall 0.38 → 0.31s (94MB), 0.99 → 0.56s (300MB text) |
| 2026-09-12 | 5afb28e | | 64KB default chunks, 32MB batches, 8-byte sequence records | 94MB: ratio 0.295 → 0.281 |
| 2026-09-12 | edcf213 | | Repeats inside one 32-byte window found in that window | 94MB: ratio 0.2808 → 0.2780, no slowdown |
| 2026-09-12 | 86278d8 | | Coarse 128-entry index for rANS length/offset decoding | 94MB: decompress kernel ~1.0–1.3 → 2.26 GB/s |
| 2026-09-12 | 223e92d | | 4-way hash buckets (half as many buckets, same memory) | 94MB: ratio 0.278 → 0.275, same speed |
| 2026-09-12 | 5035ebb | | One rANS table per batch instead of per chunk (format v3) | 94MB: ratio 0.275 → 0.272 (`gzip -1` 0.284, `zstd -1` 0.252). Decompress kernel: no shared memory, 40 registers (was ~25KB, 80) |
| 2026-09-12 | 45b2fcc | | First idle-GPU benchmark | 283MB: see [the table](#283mb-repetitive-corpus) |
| 2026-09-12 | c9f4ebe | | Token-only `--mode lz` removed | 283MB: tokens only gave 0.370 against 0.272 at the same compress speed |
| 2026-09-12 | 4b943e3 | | 1MB chunks, `--profile`, u32 offsets (format v4); hash table grown to fill 48KB of shared memory | 283MB: `speed` 0.272 → 0.263, `ratio` 0.250. Compress kernel 1598 → 766 MB/s: each SM block now held one warp instead of four |
| 2026-09-12 | 473a54a | | Hash table in dynamic shared memory, up to 64KB at `ratio` | 283MB: `ratio` 0.2671 → 0.2572. **Reverted in 80fb50d**: halved compress throughput on varied data |
| 2026-09-12 | 9d189cd | | Repeat-offset codes | 283MB: ~0.3% smaller at every profile; compress kernel ~3–11% slower together with lazy2 (see [Corrected claims](#corrected-claims)) |
| 2026-09-12 | 7dae4ae | | Two-step lazy matching ("lazy2") | 283MB: 0.07–0.16% smaller, no measured cost |
| 2026-09-12 | 80fb50d | | Revert 473a54a | 1GB: `ratio` compress kernel 169 → 297–330 MB/s |
| 2026-09-12 | d5073da | | Hash table moved to global memory; 4x larger at `ratio` | 283MB: `speed` compress kernel 742 → 2458 MB/s, `balance` 477 → 805; `ratio` 0.2662 → 0.2500 |
| 2026-09-13 | 7012791 | | Batches sized for occupancy (≥1024 chunks, 4GB budget); fixed pinned staging; writer thread | 1GB: `ratio` compress kernel 278 → 1525 MB/s, `balance` 994 → 2266 |
| 2026-09-13 | 61f4a78 | | Bigger hash tables for `balance` (4096 buckets) and `ratio` (32768) | 2.9% and 2.2% smaller |
| 2026-09-13 | 8c9ddcc | | Order-1 literal contexts chosen per batch (format v5) | 1–4% smaller; compress kernel 0–5% and decompress kernel 2–8% slower |
| 2026-09-13 | ec7aae3 | | Idle-GPU re-measurement of everything | the [current results](benchmarks.md) |
| 2026-09-13 | 62ebd81 | | Default GPU budget: 80% of free memory (was 4GB) | no throughput change (see [1GB table](#1gb-varied-corpus)) |
| 2026-09-13 | a097d56 | | One repeat-offset helper for encoder and decoder; format version restarts at 1 with the rename | 1GB: decompress kernel ~1.2–1.5% faster, output identical |
| 2026-09-13 | b2871e1 | | Vulkan backend | RTX 5060 Ti: compress kernels 90–96% of CUDA, decompress 55–70% (see [Vulkan](#vulkan-backend)) |
| 2026-09-14 | 35a5ca7 | v0.1.0 | First release | same code paths as above |
| 2026-09-14 | 6dd18e4 | | enwik8 on an Apple M1 Max and the RTX 5060 Ti | see [enwik8](#enwik8) |
| 2026-09-14 | d8f68b8 | v0.1.1 | Vulkan timestamp pool capped at 4096 | no speed change; MoltenVK kernel timings exact from here on |
| 2026-09-14 | f5283bf | | Output written by two threads at known file offsets | 1GB: decompress wall 0.79–0.86s → 0.65–0.81s at `speed`; output identical |
| 2026-09-14 | 0c5280a | | Recent match offsets tried at every parse position | 0.11–0.47% smaller at 2–7% of compress kernel throughput |
| 2026-09-14 | ba9705f | | Positions sampled after 8 windows without a match | random data: compress kernel 3.0–6.7x faster; text unchanged |
| 2026-09-15 | a7eeb42 | | Format 2: coded tables, 4-byte chunk directory, density-aware context choice | 0.004–1.6% smaller; no kernel change |
| 2026-09-15 | a7eeb42 | | Re-measurement of everything on format 2 | the [current results](benchmarks.md) |

## The occupancy round (2026-09-13)

The largest single step so far, from d5073da to ec7aae3: `ratio`
compression 3.3x faster in wall time on the 1GB corpus (265 → 871 MB/s),
and 2–4% smaller output at every profile. The four changes behind it, in
order of impact:

1. **Batches sized for occupancy** (7012791). One warp parses one chunk,
   and a warp's parse is latency-bound at a few MB/s, so throughput
   follows the number of chunks in flight. The old planner split every
   file into eight batches under a 1GB budget, which gave the 1MB `ratio`
   profile 32 chunks — 32 warps on a 36-SM GPU — per batch. Batches now
   aim for at least 1024 chunks under a 4GB budget. This alone made the
   `ratio` kernels ~6x faster.
2. **Small fixed pinned staging instead of pinned batch buffers**, plus a
   writer thread (7012791). Big batches made pinned host memory the new
   bottleneck: `cudaHostAlloc` costs ~0.3–0.4s per GB under WSL2. See
   [Host pipeline](design.md#host-pipeline).
3. **Order-1 literal contexts** chosen per batch (8c9ddcc): 1–4% smaller
   output. See [rANS stage](design.md#rans-stage).
4. **Bigger match-finder tables** for `balance` and `ratio` (61f4a78),
   which had only been measured with ~32 warps in flight: 2–3% smaller at
   those profiles. See [Sizing the match-finder
   table](#sizing-the-match-finder-table).

The before/after measurements are the back-to-back rows in the two
corpus tables below.

## The survey round (2026-09-14 to 2026-09-15)

Four changes measured against the [compression
survey](compression-survey.md), in the order they landed:

1. **Two writer threads** (f5283bf). Decompression was waiting on one
   thread copying cache-cold staging memory into the page cache. Every
   output stage's file offset is known when it is queued, so two threads
   now write them in any order. `speed` decompression of the 1GB corpus
   went from 0.79–0.86s to 0.65–0.81s; the `ratio` profile, whose output
   all arrives after one batch, didn't move.
2. **Recent match offsets tried at every position** (0c5280a). zstd's
   parsers check their repeat offsets before the hash table and gpusqz
   didn't: 0.11–0.47% smaller output for 2–7% of compress kernel
   throughput. `speed` tries one offset, the other profiles two.
3. **Sampling after long runs without a match** (ba9705f), as in LZ4 and
   zstd. Incompressible input was the slowest case; it is now 3.0x
   (`speed`) to 6.7x (`ratio`) faster on the compress kernel, and
   already-compressed files 13–60% faster, with text output unchanged.
4. **Format 2** (a7eeb42): the rANS tables coded with an adaptive binary
   range coder (~10x smaller), the chunk directory cut from 16 bytes to
   4, and the literal-context chooser charging what the coded tables
   actually cost. 0.004% to 1.6% smaller output, most of it on small
   files and at `speed`, where the tables and directory weigh most.

Together, on the 1GB corpus: 0.5% smaller at `speed`, 0.6% at `balance`
and 0.5% at `ratio`, which puts the `ratio` profile just past `zstd -3`
(0.2243 against 0.2245). Compression costs 4–6% of both kernel and wall
throughput, all of it the offset probes. The decompress kernel is within
3% at `speed` and `balance`; `ratio`'s reads much higher than the
ec7aae3 row (3087 against 1906 MB/s), but that is the batch layout the
free VRAM allowed on the day, not a change here — the A/B pairs at a
fixed budget moved decompression by under 1%.

## 283MB repetitive corpus

One 5.4MB block of `/usr/include` headers repeated 48 times (the recipe
is in [Benchmarks](benchmarks.md#corpora)). It flatters the match finder,
so use it alongside the varied corpus, not instead of it.

Default profile (`speed`, 64KB chunks), MB/s:

| date | commit | conditions | compress wall | compress kernel | decompress wall | decompress kernel | ratio | `zstd -1` ratio |
|---|---|---|---|---|---|---|---|---|
| 2026-09-12 | a54c49a | another process using ~90% of the GPU | 498 | 1500 | 614 | 1040 | 0.281 | 0.252 |
| 2026-09-12 | 45b2fcc | idle, best of 5 | 583 | 1598 | 629 | 3759 | 0.272 | 0.252 |
| 2026-09-12 | 8a92750 (4b943e3) | idle, best of 5 | 440 | 766 | 731 | 3474 | 0.263 | 0.252 |
| | | *corpus rebuilt here* | | | | | | |
| 2026-09-12 | afaa7bc (473a54a, 9d189cd, 7dae4ae) | idle, best of 5 | 402 | 719 | 760 | 3009 | 0.2775 | 0.2724 |
| 2026-09-12 | 80fb50d | idle, best of 5 | 417 | 742 | 676 | 3050 | 0.2775 | 0.2724 |
| 2026-09-12 | d5073da | idle, best of 5 | 628 | 2458 | 643 | 2993 | 0.2775 | 0.2724 |
| 2026-09-13 | d5073da | idle, best of 3, back to back with the next row | 614 | 2474 | 707 | 3054 | 0.2775 | 0.2724 |
| 2026-09-13 | ec7aae3 (7012791, 61f4a78, 8c9ddcc) | idle, best of 3, back to back with the previous row | 647 | 2382 | 769 | 3804 | 0.2673 | 0.2724 |
| 2026-09-13 | ec7aae3 | idle, best of 3–5 (the [current results](benchmarks.md)) | 664 | 2390 | 718 | 3805 | 0.2673 | 0.2724 |
| 2026-09-15 | a7eeb42 | idle (0.5GB used, 2–6%), best of 3–5 (the [current results](benchmarks.md)) | 643 | 2319 | 778 | 3700 | 0.2657 | 0.2724 |

`balance` (256KB) and `ratio` (1MB), kernel MB/s:

| commit | `balance` compress | `balance` decompress | `balance` ratio | `ratio` compress | `ratio` decompress | `ratio` ratio |
|---|---|---|---|---|---|---|
| 8a92750 (4b943e3) | 508 | 923 | 0.253 | 252 | 242 | 0.250 |
| | *corpus rebuilt here* | | | | | |
| afaa7bc | 468 | 828 | 0.2687 | 203 | 208 | 0.2561 |
| 80fb50d | 477 | 838 | 0.2687 | 217 | 226 | 0.2662 |
| d5073da | 805 | 829 | 0.2687 | 214 | 205 | 0.2500 |
| d5073da (back to back with the next row) | 809 | 837 | 0.2687 | 216 | 206 | 0.2500 |
| ec7aae3 (back to back with the previous row) | 1737 | 3416 | 0.2547 | 1233 | 1515 | 0.2423 |
| ec7aae3 (current results) | 1746 | 3414 | 0.2547 | 1232 | 1515 | 0.2423 |
| a7eeb42 (current results) | 1699 | 3410 | 0.2535 | 1142 | 1492 | 0.2411 |

Wall MB/s for the back-to-back pair: `balance` compress 409 → 579 and
decompress 679 → 746, `ratio` compress 171 → 499 and decompress 425 →
703.

Chunk size before profiles existed (45b2fcc, idle, best of 5, 64KB was
then the largest chunk the format allowed):

| chunk | coding | ratio | compress kernel | decompress kernel |
|---|---|---|---|---|
| 16KB | LZ + rANS | 0.297 | 2372 | 8106 |
| 32KB | LZ + rANS | 0.281 | 2073 | 6644 |
| 64KB | LZ + rANS | 0.272 | 1585 | 3741 |
| 64KB | LZ tokens only | 0.370 | 1541 | 5286 |

With the GPU idle, smaller chunks decoded much faster (more, smaller
units of work), which made chunk size a real speed/ratio dial and led to
the profiles.

## 1GB varied corpus

Every distinct `/usr/include` header concatenated, grown to 1GB with more
directories and no repetition. In use since 80fb50d, when it exposed a
regression the 283MB corpus hid. On this file `zstd -1` gives 0.2518 and
`zstd -3` 0.2245.

Full measurements, MB/s:

| date | commit | conditions | profile | compress wall | compress kernel | decompress wall | decompress kernel | ratio |
|---|---|---|---|---|---|---|---|---|
| 2026-09-13 | d5073da | idle, best of 3, back to back with ec7aae3 | `speed` | 1295 | 2912 | 1258 | 3573 | 0.2621 |
| | | | `balance` | 721 | 1066 | 1231 | 1283 | 0.2517 |
| | | | `ratio` | 265 | 301 | 858 | 326 | 0.2329 |
| 2026-09-13 | ec7aae3 | idle, best of 3 | `speed` | 1279 | 2980 | 1466 | 4595 | 0.2526 |
| | | | `balance` | 1008 | 1851 | 1314 | 3357 | 0.2387 |
| | | | `ratio` | 871 | 1550 | 1206 | 1917 | 0.2255 |
| 2026-09-13 | ec7aae3 | idle, best of 3–5 (the [current results](benchmarks.md)) | `speed` | 1297 | 2983 | 1130 | 4567 | 0.2526 |
| | | | `balance` | 989 | 1847 | 1267 | 3353 | 0.2387 |
| | | | `ratio` | 873 | 1552 | 1176 | 1906 | 0.2255 |
| 2026-09-15 | a7eeb42 | idle (0.5GB used, 2–6%), best of 3–5 (the [current results](benchmarks.md)) | `speed` | 1217 | 2801 | 1234 | 4440 | 0.2512 |
| | | | `balance` | 946 | 1744 | 1120 | 3283 | 0.2373 |
| | | | `ratio` | 837 | 1484 | 1058 | 3087 | 0.2243 |

The two ec7aae3 runs differ in decompress wall by up to 30%: decompression
on this machine is bound by `fwrite` into the WSL2 page cache, which
varies from run to run. The kernel figures agree within 1%.

Changes measured between those rows:

- **473a54a → 80fb50d** (idle, 3 runs each): the `ratio` compress kernel
  ran at ~312–350 MB/s before 473a54a, 189 with it, and 169 with repeat
  offsets and lazy2 on top. After the revert: 297–330.
- **7012791**, against d5073da, best of 2–3, with an idle `ollama` model
  holding ~11.5GB (batch budget ~2.2GB instead of 4GB). Ratios unchanged:

  | profile | compress wall | compress kernel | decompress wall |
  |---|---|---|---|
  | `speed` | 1263 → 1351 | 2749 → 2923 | 940 → 1288 |
  | `balance` | 733 → 1085 | 994 → 2266 | 1004 → 1070 |
  | `ratio` | 253 → 922 | 278 → 1525 | 839 → 1149 |

- **61f4a78**, on both corpora:

  | profile | buckets | output size | compress kernel | compress wall |
  |---|---|---|---|---|
  | `balance` | 2048 → 4096 | −2.9% | −21 to −25% | −11% |
  | `ratio` | 8192 → 32768 | −2.2% | −3 to −4% | unchanged |

- **8c9ddcc**, against 61f4a78, best of 3, `ollama` idle in the
  background. Wall times unchanged within noise; compress kernel 0–5%
  slower, decompress kernel 2–8% slower:

  | ratio | 283MB repetitive | 1GB varied |
  |---|---|---|
  | `speed` | 0.2775 → 0.2673 | 0.2621 → 0.2526 |
  | `balance` | 0.2611 → 0.2547 | 0.2442 → 0.2387 |
  | `ratio` | 0.2447 → 0.2423 | 0.2277 → 0.2255 |

- **62ebd81** (default budget 80% of free VRAM instead of 4GB), `ratio`
  profile: the old default ran the file as three batches of ~350 chunks,
  the new one as one batch on an idle 16GB GPU. Kernel ~1550 MB/s either
  way (the GPU is saturated); compress wall 851 vs 869 MB/s, within
  noise. A 2G budget measured 900 MB/s kernel.
- **a097d56**, interleaved runs: decompress kernel ~1.2–1.5% faster,
  compress unchanged, output byte-identical.

## enwik8

The first 100MB of a 2006 English Wikipedia dump, the corpus of the Large
Text Compression Benchmark. Measured on 2026-09-14 (6dd18e4), best of 3,
wall MB/s:

| profile | M1 Max compress | M1 Max decompress | RTX 5060 Ti compress | RTX 5060 Ti decompress | ratio |
|---|---|---|---|---|---|
| `speed` | 386 | 704 | 347 | 357 | 0.3862 |
| `balance` | 442 | 751 | 323 | 344 | 0.3744 |
| `ratio` | 221 | 414 | 227 | 308 | 0.3592 |

Apple M1 Max (32GB): macOS package, Vulkan through MoltenVK, subgroup
lanes. The kernel figures ([Benchmarks](benchmarks.md#enwik8-on-an-apple-m1-max-and-the-rtx-5060-ti))
came from emulated timestamps and are approximate; v0.1.1 fixed that.
RTX 5060 Ti: CUDA, WSL2, idle. `zstd -1` gives 0.4067, `zstd -3` 0.3544.

## Vulkan backend

Vulkan against CUDA on the same GPU (RTX 5060 Ti, Windows NVIDIA Vulkan
driver, a Windows build run from WSL2), 283MB corpus, kernel MB/s,
2026-09-13:

| profile | compress, Vulkan / CUDA | decompress, Vulkan / CUDA |
|---|---|---|
| `speed` | 2013 / 2227 | 1874 / 3323 |
| all profiles | 90–96% | 55–70% |

Removing `coherent` from the scratch buffers improved Vulkan
decompression by about 25%, and dropping a redundant memory barrier helped
too. The rest of the gap is not understood yet.

## Survey follow-up measurements (2026-09-14)

Measured at 92901a5 while checking the [compression
survey](compression-survey.md) against gpusqz, to rank what to try next.
An `ollama` model held 10–12.5GB of VRAM at 14–15% utilisation the whole
time, so the I/O and output-size figures below are sound but the kernel
MB/s are not A/B quality.

**Decompression waits on the output file.** 1GB corpus, `speed`:

| output | wall | writer thread busy | stalled waiting for an output stage |
|---|---|---|---|
| `/dev/null` | 0.33–0.39s | 0.00s | 0.09–0.10s |
| a file (WSL2 ext4) | 0.70–0.94s | 0.51–0.75s | 0.44–0.67s |

A standalone `pwrite` test (1000MB in 8MB blocks from a 128MB source
buffer, so the source is out of cache as it is in gpusqz) wrote 1.7–1.9
GB/s from one thread and 2.8–2.9 GB/s from two; four were no faster.
Output offsets are known in advance (chunk *c* goes to *c* × chunk
size), so two writer threads could `pwrite` in parallel. Reading the
same way scaled from 4–7.6 GB/s (one thread, page cache warm) to 11 GB/s
(two).

**Incompressible input is the slowest case.** 1GB of `/dev/urandom`
against the 1GB corpus (every random chunk ends up stored raw):

| profile | random, compress kbusy | text, compress kbusy |
|---|---|---|
| `speed` | 2107 | 2895 |
| `ratio` | 609 | 1507 |

With no matches the parse probes all four candidates at every position
and advances only 32 bytes per step.

**Compression at `speed` is GPU-bound once the input is cached.** Three
warm runs on the 1GB corpus (`ollama` idle, 1% utilisation): 0.54–0.57s
steady against 0.35s of kernel time, with the main thread's reads
(0.16–0.20s) overlapping the kernels. Earlier runs at 0.9–1.0s steady
were reading a cold page cache (`fread` 0.77s).

**Match-finder variants.** Output size with `--gpu-mem 2G` (a fixed batch
split, so every build saw the same batches), change against 92901a5.
Every output decoded correctly with `gpusqz_refdec`:

| variant | 1GB `speed` / `balance` / `ratio` | 283MB | enwik8 | compress kbusy |
|---|---|---|---|---|
| Also probe the last match offset (ties go to it) | −0.31 / −0.33 / −0.30% | −0.23 / −0.24 / −0.25% | −0.11 / −0.11 / −0.09% | 0–4% slower |
| … and the second-last offset | −0.40 / −0.42 / −0.39% | −0.29 / −0.32 / −0.33% | −0.14 / −0.13 / −0.12% | 2–7% slower |
| Hash the last 32 positions of matches longer than a window | −0.11 / −0.13 / −0.14% | −0.10 / −0.11 / −0.12% | −0.02 / −0.02 / −0.03% | 1–13% slower |

**Where the rest of the output goes.** The rANS tables take 0.16–0.20% of
the 1GB corpus's output (0.34% of enwik8's), and zstd shrinks them
4–27x, mostly by exploiting how alike the groups' tables are. The chunk directory, 16 bytes per chunk, takes 0.10% at
`speed`.

## Sizing the match-finder table

The hash table took four rounds to size, and each failure was
informative:

1. **Dynamic shared memory up to 64KB** (473a54a). Requesting more than
   48KB of shared memory at launch changed the SM's cache behaviour for
   the *whole* kernel. That cost was invisible on the repetitive corpus
   and nearly halved `ratio` compress throughput on the varied one. The
   trigger is the bytes requested at launch: raising the kernel's
   shared-memory ceiling without using it cost nothing.
2. **A bigger table within the 48KB static limit.** This still cost
   38–52% of throughput for a 1–4% ratio gain, because more shared memory
   per one-warp block means fewer blocks per SM.
3. **Global memory** (d5073da) sidestepped both and made `speed` and
   `balance` 1.5–3.6x faster by freeing shared memory for occupancy.
4. **Re-tuned under occupancy-sized batches** (61f4a78). The earlier "a
   bigger global table costs ~2x" had been measured with ~32 warps in
   flight, where every extra miss was exposed. With ~1000 in flight,
   `ratio`'s table grew 4x for 2.2% smaller output at 3–4% kernel cost
   and no wall-clock cost, and `balance`'s 2x for 2.9% at ~11% of wall
   throughput. `speed` stays at 2048 buckets.

Global-memory latency is far more sensitive to a *concurrent* GPU process
than shared memory was: under ~40% contention, the `ratio` compress
kernel measured at half its real speed.

## Corrected claims

Claims made at the time that later measurements overturned. They're kept
because each explains a rule in [Benchmarks](benchmarks.md).

- **"The dynamic hash table has negligible kernel-speed cost"** (473a54a).
  True on the 283MB corpus (~13% there); on the varied 1GB file the
  `ratio` compress kernel fell from 349 to 169 MB/s. Found by a
  commit-by-commit bisect on the 1GB file and reverted in 80fb50d. This
  is why match-finder changes must be measured on a varied corpus.
- **"Repeat offsets cost 6–12% of compress speed because their encoder
  pass is serial"** (9d189cd). That measurement included the oversized
  table from 473a54a. Disabling the serial pass alone measured 178.2 vs
  168.6 MB/s, a few percent; with the table reverted, repeat offsets and
  lazy2 together cost ~3–11% across profiles.
- **"gpusqz beats `zstd -1` and `zstd -3` at every profile"** (the README
  at afaa7bc). A misreading of lower-is-better: at that point gpusqz beat
  `zstd -1` only at `balance` and `ratio`, and never beat `zstd -3` or
  `gzip -6`. Corrected in 80fb50d.
- **A `balance` regression and a half-speed `ratio` kernel** while
  developing d5073da. Both came from ~40% contention by another process
  and vanished on an idle GPU.
- **The first benchmark table** (a54c49a) was taken while another
  process used ~90% of the GPU. One run from that time measured 22 MB/s
  wall against a real 250–340 MB/s, which is why `bench/run_bench.sh`
  has `REPEAT` (11b2fcc) and why numbers now come with their conditions.

## Measured and rejected

Tried, measured, and not kept. Check here before re-running one.

| idea | when | result |
|---|---|---|
| 3-byte minimum match (`kMinMatch=3`) | 5afb28e | ~3% worse ratio, and slower |
| 8-way buckets (256 buckets, same memory) | 4b943e3 | worse ratio and slower than more buckets |
| Hash table in dynamic shared memory >48KB | 473a54a | halved `ratio` compress throughput on varied data; reverted |
| Bigger table within 48KB of static shared memory | before d5073da | 38–52% slower for 1–4% smaller |
| `balance` table at 4096 buckets in shared memory | 473a54a | ~2x slower for ~3% smaller (fine later in global memory) |
| `speed` table at 4096 buckets | 61f4a78 | 1.6% smaller for ~12% of wall throughput |
| `ratio` table at 65536 buckets | 61f4a78 | 0.4% smaller than 32768 for ~18% of kernel throughput |
| Third lazy-matching step | 61f4a78 | no effect |
| 64-byte probe cap (instead of 32) | 61f4a78 | 0.1% smaller, 1–3% slower |
| `__restrict__` on kernel pointers | 61f4a78 | no change |
| Prefetching the next window's input bytes | 61f4a78 | no change |
| `GPUSQZ_MIN_BLOCKS_PER_SM=4` occupancy hint | d5073da | no change; neither kernel is register-bound |
| `cudaStreamQuery` flush after each staged copy | 7012791 | within noise |
| Three buffer sets instead of two | 7012791 | within noise at `speed`, slower at `ratio` |
| 14- and 15-bit rANS probabilities (instead of 12) | 8c9ddcc | under 0.05% smaller: the 8-bit quantised counts are the limit |
| 64 frequency-ranked literal contexts | 8c9ddcc | about the same as 256, but needs a stored class map |
| Encoder tables from exact counts instead of 8-bit quantised ones | 92901a5 | 0.06–0.23% smaller before paying for bigger stored tables; 0.07–0.33% with 14-bit probabilities as well. The 8-bit counts did cap the 14-bit test above, but lifting them isn't worth much either |
| Hashing the positions a long match skipped | 92901a5 | 0.02–0.14% smaller for up to 13% of compress kernel throughput |
| Token-only coding (`--mode lz`) | c9f4ebe | 0.370 vs 0.272 at the same compress speed, decompress kernel 5498 vs 3759 MB/s; rANS won on ratio, so the mode was removed |
| Order-2 literal contexts (previous literal + top bits of the one before, in the lane's run) | c777280 | at most 0.61% smaller (enwik8 `speed`), 0.02–0.04% at enwik8 `ratio`, up to 0.37% *larger* on headers at `ratio` once the extra tables are paid for. See [Predictive modeling study](predictive-modeling-study.md) |
| Adaptive (context-mixing) literal models over the lane's literals, reset per lane run or per chunk | c777280 | 0.04–5.2% larger than today's static order-1 tables: a lane's run is too short to learn from. See [Predictive modeling study](predictive-modeling-study.md) |
| Byte planes over a binary STL's raw vertex stream (x, y, z split, then 4 byte planes), normals dropped | 593ae95 | 2.34x *larger* at `ratio` on a grid-ordered mesh than dropping the normals alone (3,700,955 vs 1,581,575 bytes): it breaks the 50-byte record stride the match finder uses. 5% smaller on the same mesh with its triangles shuffled. See [Format-aware transforms study](format-aware-transforms-study.md) |
| Whole-file vertex table for binary STL | 593ae95 | 4.20x at `ratio` on a shuffled mesh against 1.94x for a per-chunk table, but every chunk would reference one shared table, which breaks chunk independence. See [Format-aware transforms study](format-aware-transforms-study.md) |
| Host-side incompressible probe: skip the upload, kernels and download for a batch whose chunks all sample as incompressible (byte-pair collisions near uniform, few repeated 4-byte sequences, 3 × 4KB windows per chunk) | 3169109 | Ceiling too low and not reachable on real data. On 1GB of random or zstd output the kernel was already 0.12–0.24s (4.6–9.3 GB/s) of 1.3–2.2s wall; h2d, d2h and output-stage stalls were the rest, with fwrite underneath (RTX 5060 Ti, CUDA, idle, no ollama). Against the baseline's own raw decisions: random data 100% detected with no false positives, but zstd output isn't flat enough (91% of its `speed` chunks are stored raw, only 23–78% sample as incompressible across thresholds), so **0%** of its batches could be skipped. Loosening the thresholds lets random data repeating every 200KB through: 3.1% larger at `speed`, 4.4x at `ratio`. The GPU kernel is the better detector |
| Parallelising `compute_repeat_codes` (the serial repeat-offset pass on lane 0) | 3169109 | Ceiling too small for a two-backend kernel change. Removing the pass outright (no repeat codes at all) saves 1–6% of compress kernel time: 283MB corpus 2/1/6% at `speed`/`balance`/`ratio`, enwik8 4/5/6%, 1GB of `/usr/lib` binaries 1% (best of 5, RTX 5060 Ti, CUDA, idle, no ollama). A parallel version (lanes resolving their segments' first sequences after a sync) could recover only part of that. The repeat codes themselves are worth keeping: without them the binaries are 3.5% (`speed`) and 4.2% (`ratio`) larger, text 0.15–0.7% |

## Adding an entry

Add an entry when a change moves speed or ratio (usually a `perf:` or
`feat:` commit), when a release changes performance, or when gpusqz is
measured on new hardware. Follow the protocol in
[Benchmarks](benchmarks.md#running-the-benchmark): idle GPU, both
corpora, the change and its baseline run back to back.

1. Add a line to [Milestones](#milestones) with the date, commit, release
   (if any), change and measured effect.
2. Add rows to the table of each corpus you measured, newest last. A new
   machine or backend gets its own table with the same columns. Always
   fill in the conditions: GPU, backend and driver, free VRAM and
   utilisation before the run, best of how many runs.
3. Measure `zstd -1` on the same file and record its ratio (in the
   table's `zstd -1` column where there is one, otherwise in the
   section's text). If you rebuilt a corpus, add a *corpus rebuilt here*
   row so older ratios aren't compared with newer ones.
4. Put experiments that didn't pay off in [Measured and
   rejected](#measured-and-rejected), with the numbers.
5. If a previous entry turns out wrong, don't edit the numbers: add it to
   [Corrected claims](#corrected-claims) with what the new measurement
   showed.

Update the current tables in [Benchmarks](benchmarks.md) and the summary
in the top-level README when the headline numbers change.
