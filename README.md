# gzp — a GPU file compressor

`gzp` compresses and decompresses files on an NVIDIA GPU with CUDA. Each
chunk of the input (64KB by default, up to 1MB) is handled by one warp: an
LZ parse where all 32 lanes search for matches together, followed by a
32-way interleaved rANS entropy coder with order-1 literal contexts. It's
a from-scratch, educational implementation — not a drop-in replacement for
zstd — but at its default setting it compresses 1.4–2.5x faster than
single-threaded `zstd -1` at about the same ratio, and its `ratio`
profile lands within 0.5% of `zstd -3`'s output size while compressing
1.3–2.2x faster (see *Results*; zstd still decompresses faster).

## Results

RTX 5060 Ti (16GB) under WSL2, CPU tools single-threaded, best of 3–5 runs
per column (`REPEAT=<n>`, see *Benchmarking*). Wall figures are
whole-process (file I/O, PCIe copies, and for gzp ~0.2s of CUDA context
creation and allocation); the kernel columns are the wall-clock time any
gzp kernel was running. Ratio is output/input, so **lower is better**.

**Measurement conditions.** The GPU was otherwise idle (~1GB used by the
desktop, 0–2% utilisation), so gzp's batch budget was its full 4GB. On
the 283MB corpus about half of gzp's wall time is the ~0.2s fixed startup
cost, which varies by ±15% between runs, so its wall figures there move
by that much from run to run; the kernel figures and the 1GB corpus are
the steadier comparison. An earlier set of runs with another process
holding 11.5GB of VRAM gave kernel figures and ratios within 1% of these.

Varied 1GB corpus (every distinct `/usr/include` header, concatenated
without repetition; see *Benchmarking*):

| codec | compress MB/s (wall) | kernel MB/s | decompress MB/s (wall) | kernel MB/s | ratio |
|---|---|---|---|---|---|
| **gzp** (default, `speed`, 64KB) | **1297** | 2983 | 1130 | 4567 | 0.2526 |
| gzp `--profile balance` (256KB) | 989 | 1847 | 1267 | 3353 | 0.2387 |
| gzp `--profile ratio` (1MB) | 873 | 1552 | 1176 | 1906 | 0.2255 |
| gzip -1 | 145 | – | 250 | – | 0.2836 |
| gzip -6 | 56 | – | 278 | – | 0.2306 |
| zstd -1 (1 thread) | 516 | – | **1434** | – | 0.2518 |
| zstd -3 (1 thread) | 401 | – | 1335 | – | **0.2245** |

Repetitive 283MB corpus (one 5.4MB block of headers repeated 48 times — a
friendlier shape for a match finder, see the caveat in *Benchmarking*):

| codec | compress MB/s (wall) | kernel MB/s | decompress MB/s (wall) | kernel MB/s | ratio |
|---|---|---|---|---|---|
| **gzp** (default, `speed`, 64KB) | **664** | 2390 | 718 | 3805 | 0.2673 |
| gzp `--profile balance` (256KB) | 552 | 1746 | 715 | 3414 | 0.2547 |
| gzp `--profile ratio` (1MB) | 487 | 1232 | 657 | 1515 | **0.2423** |
| gzip -1 | 140 | – | 249 | – | 0.2985 |
| gzip -6 | 53 | – | 273 | – | 0.2457 |
| zstd -1 (1 thread) | 482 | – | **1328** | – | 0.2724 |
| zstd -3 (1 thread) | 373 | – | 1188 | – | 0.2426 |

Against the CPU tools:

- **Default profile vs `zstd -1`.** gzp compresses 1.4x (283MB) to 2.5x
  (1GB) faster. Its output is 1.9% smaller on the repetitive corpus and
  0.3% larger on the varied one. `zstd -1` decompresses faster: 1.85x on
  the smaller file, where gzp's fixed startup cost weighs more, and 1.3x
  on the 1GB file.
- **`ratio` profile vs `zstd -3`.** gzp compresses 1.3–2.2x faster. Its
  output is 0.1% smaller on the repetitive corpus and 0.5% larger on the
  varied one. `zstd -3` decompresses 1.1–1.8x faster.
- **vs `gzip`.** Every gzp profile beats `gzip -1` on both ratio and
  speed. The `ratio` profile also beats `gzip -6`'s ratio on both corpora
  while compressing 9–16x faster; `balance` does not beat `gzip -6`.

The gzp wall figures on the 1GB corpus are bounded by file I/O more than
by the GPU: under WSL2, decompression spends most of its time in
`fwrite` (see *Known limitations*), and the kernels run 1.6–4x faster
than the wall-clock rate.

### What changed in this round

Same machine, idle GPU, best of 3, previous version (d5073da) against
this one, run back to back:

| corpus | profile | compress wall | compress kernel | decompress wall | decompress kernel | ratio |
|---|---|---|---|---|---|---|
| 1GB varied | speed | 1295 → 1279 | 2912 → 2980 | 1258 → 1466 | 3573 → 4595 | 0.2621 → 0.2526 |
| 1GB varied | balance | 721 → 1008 | 1066 → 1851 | 1231 → 1314 | 1283 → 3357 | 0.2517 → 0.2387 |
| 1GB varied | ratio | 265 → 871 | 301 → 1550 | 858 → 1206 | 326 → 1917 | 0.2329 → 0.2255 |
| 283MB repetitive | speed | 614 → 647 | 2474 → 2382 | 707 → 769 | 3054 → 3804 | 0.2775 → 0.2673 |
| 283MB repetitive | balance | 409 → 579 | 809 → 1737 | 679 → 746 | 837 → 3416 | 0.2687 → 0.2547 |
| 283MB repetitive | ratio | 171 → 499 | 216 → 1233 | 425 → 703 | 206 → 1515 | 0.2500 → 0.2423 |

(MB/s). The four changes behind it, in order of impact:

1. **Batches sized for occupancy.** One warp parses one chunk, and a
   warp's parse is latency-bound at a few MB/s, so throughput follows the
   number of chunks in flight. The old planner split every file into eight
   batches under a 1GB budget, which gave the 1MB `ratio` profile 32
   chunks — 32 warps on a 36-SM GPU — per batch. Batches now aim for at
   least 1024 chunks under a 4GB budget. This alone made the `ratio`
   kernels ~6x faster.
2. **Small fixed pinned staging instead of pinned batch buffers**, plus a
   writer thread. Big batches made pinned host memory the new bottleneck:
   `cudaHostAlloc` costs ~0.3–0.4s per GB under WSL2. See *Host
   pipeline*.
3. **Order-1 literal contexts** chosen per batch: 1–4% smaller output.
   See *rANS stage*.
4. **Bigger match-finder tables** for `balance` and `ratio`, which had
   only been measured with ~32 warps in flight: 2–3% smaller at those
   profiles. See *Known limitations*.

## How it works

### Chunks and warps

The input is split into independent, fixed-size chunks (64KB by default,
up to 1MB — see `--profile` in *Usage*). One warp — 32 lanes — compresses
or decompresses one chunk. A warp's LZ parse is bound by the latency of
its dependent, random hash-table and history reads, not by bandwidth, so
the GPU's throughput is roughly the number of chunks in flight times a
few MB/s each; batches are sized to keep at least ~1000 in flight (see
*Host pipeline*). There are no cross-chunk references: any chunk can be
decoded on its own, and a corrupt chunk cannot damage another.

### LZ parse (`src/lz_warp.cuh`)

The warp walks a chunk in 32-byte windows. Every lane hashes the 4 bytes
at its own position, reads a 4-way bucket from this chunk's hash table
(one u32 chunk-relative position per word) and compares against all four
candidates, capped at 32 bytes so per-lane work is bounded. One
deterministic lane per bucket then inserts its position, evicting the
oldest of the four; lanes that found nothing re-probe once more so
repeats shorter than a window apart are caught immediately.

The table lives in **global memory**, one region per chunk, sized per
profile by `hash_table_bits()` (`kernels.h`): 2048 buckets at `speed`,
4096 at `balance`, 32768 at `ratio`. Two earlier attempts at a bigger
*shared*-memory table both regressed (see *Known limitations*); global
memory avoids those costs, and freeing the shared memory let more of the
kernel's one-warp blocks run per SM.

The window's matches are selected warp-uniformly from a ballot mask with
a lazy lookahead of up to `kLazySteps` (2) positions — take position
*i+1*'s match instead if it's clearly longer, then *i+2* — which zstd calls
"lazy2". Matches that hit the 32-byte probe cap are extended
cooperatively, 32 bytes per step, so long runs never serialise on one
lane.

The parse emits sequences — a literal run followed by a match `(offset,
length)` — into scratch for the entropy stage below, packed 8 bytes per
sequence (three 21-bit fields). Per chunk, the encoder keeps whichever of
the rANS-coded result or a plain LZ4-style token stream comes out
smaller, and falls back to raw storage if neither beats the input.

### rANS stage (`src/rans.cuh`, `src/rans_codes.h`)

Literals, literal-run lengths, match lengths and offsets are coded with
rANS. Length and offset alphabets are zstd-style log2 buckets with raw
extra bits (written straight into the rANS state, so there is no side
stream) — except the offset alphabet's top 3 codes, which are
**repeat-offset codes**: "reuse the 1st/2nd/3rd most-recently-used
distinct match offset", zstd-style. The encoder resolves them in one
forward serial pass over the parsed sequences (`compute_repeat_codes()`),
and the decoder replays the same state machine per group of 32 sequences
with a register-only shuffle walk.

**Literals use order-1 contexts.** Each literal is coded with a table
chosen by the literal before it: its high nibble (16 tables), all of it
(256 tables), or nothing (1 table, order-0). For the decoder to know the
previous literal, each of the 32 lanes owns one contiguous run of the
chunk's literal stream rather than every 32nd literal, and the first
literal of each run uses context 0. Runs are 4-byte aligned, so the
decoder stores 4 literals at a time.

**Tables are shared per batch** (a "table group", see *Container format*):
compression runs three kernels per batch. The first parses every chunk
into scratch while atomically accumulating one full order-1 histogram
for the batch. The second, one 256-thread block, folds that histogram to
16 and 1 contexts and keeps whichever rule minimises the estimated
literal bits under the tables the encoder would actually build, plus 256
table bytes per context. On large text batches that is nearly always all
256 contexts (~66KB of tables per batch), while incompressible or
literal-poor batches keep one table and pay nothing extra. The third
encodes every chunk against that table. A chunk whose plain token stream
is shorter than a rANS header can never end up rANS-coded, so it stays
out of the histogram and skips the rANS attempt.

The GPU-specific part is the interleaving: 32 rANS states, one per lane,
share **one** stream of 16-bit words. All lanes step in lockstep; the
lanes that need to renormalise on a step write (or read) their word
contiguously in lane order, located with a ballot and a popcount. With a
2^16 state floor and 16-bit words, each step moves at most one word per
lane, which is what makes the encoder's and decoder's per-step word
counts line up exactly without storing any per-lane offsets. The encoder
runs in reverse and writes backward from the end of its output slot; the
decoder reads forward.

### Decoding

Token decoding is warp-cooperative: 32 lanes copy each literal run and
each match, with overlapping matches handled by indexing `k mod offset`
into already-written history so no serial path is needed. rANS decoding
runs the same 32-lane lockstep as the encoder into scratch — literals
through a per-context slot-to-symbol table, lengths and offsets through a
coarse 128-entry index plus a short scan — then the same reconstruction
loop rebuilds the chunk. Every table group's tables are expanded once, up
front, into a global buffer that each chunk finds through its group id.
Malformed input sets an error flag that the host turns into an error
rather than garbage output.

### Host pipeline (`src/main.cu`)

Batches live only in device memory, in a ring of two buffer sets, each
with its own CUDA stream. File data moves through twelve fixed 8MB
pinned staging buffers instead of pinned batch-sized ones: the main
thread `fread`s into an input stage and copies it up asynchronously, and
a writer thread drains output stages that the main thread fills with
asynchronous downloads. So batch *i+1*'s read and upload overlap batch
*i*'s kernels, and batch *i−1*'s download and file write overlap both.

Two measurements drove that design:

- **Pinned memory is expensive to create under WSL2.** `cudaHostAlloc`
  measured ~0.3–0.4s per GB, plus ~0.1s per GB to free at exit, against
  ~3ms per GB for `cudaMalloc`. The old ring pinned three sets of
  batch-sized buffers, about 1.5GB for a big `ratio` batch.
- **Batches on different streams barely overlap on the GPU.** The next
  batch is usually still being read while one runs, so the chunks in
  flight are roughly one batch, and a bigger batch beats more sets.

Batch size comes from free VRAM, since the GPU may be shared: at least
1024 chunks and 32MB of input, within half of free VRAM up to 4GB. A
file that fits in one or two batches gets the whole budget. Allocation
retries with a halved batch on failure. Compressed output is compacted
on the GPU (a CUB scan plus a pack kernel, writing into the batch's
now-dead scratch), so the download moves only compressed bytes.

### Container format (`src/format.h`)

```
FileHeader    { magic, version=5, chunk_size, original_size, chunk_count,
                table_group_count, tables_offset }
ChunkEntry[]  { offset, compressed_size, original_size }     -- one per chunk
TableGroup[]  { start_chunk, chunk_count, lit_ctx_shift }    -- one per compression batch
payload       -- each chunk: [flag: Raw | Lz | LzRans] [data]
tables        -- at tables_offset: each group's quantised counts, in group order
```

`TableGroup` entries cover `[0, chunk_count)` contiguously and in order;
chunk *c*'s rANS tables are those of the group whose range contains *c* —
always exactly one host compression batch's worth of chunks, decided at
compress time and independent of whatever batch size decompression later
chooses. Each group's counts are 96 bytes for the three small alphabets
plus 256 per literal context, so their size depends on the group's
`lit_ctx_shift` (8, 4 or 0 for 1, 16 or 256 contexts). That is why they
come after the payload rather than in the directory. The decoder checks
that the payload runs exactly from the end of the directory to
`tables_offset` and that the table section ends the file.

## Building

Requires CUDA 12.8+ and CMake 3.20+. Targets sm_120 (RTX 5060 Ti /
Blackwell) by default; override for other hardware:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release [-DCMAKE_CUDA_ARCHITECTURES=<arch>]
cmake --build build -j
```

(`CMAKE_CUDA_ARCHITECTURES=native` is not used because under WSL it
silently fell back to sm_52 instead of detecting the GPU.)

`GZP_MIN_BLOCKS_PER_SM=<n>` (a CMake cache var, not a runtime flag) sets
`__launch_bounds__`'s `minBlocksPerSM` hint on the per-chunk kernels, for
A/B occupancy testing — see the comment above it in `CMakeLists.txt`. None
of those kernels use shared memory, and at 38 (parse), 64 (encode) and 58
(decode) registers per thread with no spills (`nvcc -Xptxas -v`), their
one-warp blocks are limited by the per-SM block count rather than by
registers, so this knob is mainly a regression check against future
register spilling.

## Usage

```
./build/gzp c <input> <output> [chunk_size]                # compress (default chunk_size 65536)
./build/gzp c <input> <output> --profile speed|balance|ratio  # ...or pick a chunk-size preset
./build/gzp d <input> <output>                              # decompress
```

`chunk_size` and `--profile` are mutually exclusive, and unrecognised
arguments are rejected. See *Results* for what each profile costs and
buys.

Environment variables, all optional:

| variable | effect |
|---|---|
| `GZP_VERBOSE=1` | Per-stage timing on stderr: setup, fread, copies, kernel time (summed and wall-clock union), fwrite, staging stalls. |
| `GZP_FORCE_BATCH=<n>` | Force chunks per batch (testing; see *Testing*). |
| `GZP_FORCE_SETS=<1-3>` | Force the device buffer-set count (tuning). |
| `GZP_FORCE_LIT_SHIFT=<0\|4\|8>` | Force every batch's literal-context rule (testing and tuning). |
| `GZP_DUMP_LITS=<path>` | Dump each chunk's parsed literal stream, for evaluating literal models offline. Serialises the pipeline. |

## Testing

```
bash tests/round_trip.sh                 # default chunk size, literal-context and --profile cases
bash tests/round_trip.sh --extremes      # chunk sizes 1, 16, 4K, 8K, 32K, 65535, 65536,
                                          # 1048575, 1048576 (kMaxChunkSize), mismatched
                                          # batch sizes, and a match-free 1MB chunk
BIG=1 bash tests/round_trip.sh           # + 300MB random and 300MB text (multi-batch)
```

Every case is round-tripped on the GPU *and* decoded by
`tests/ref_decode.cpp`, a CPU decoder that shares no code with the GPU
path apart from the alphabet and table definitions in `rans_codes.h`, so
a symmetric bug in the GPU encoder and decoder can't hide. That matters
here because `compute-sanitizer` in CUDA 12.8 does not support this GPU,
so memcheck was not available during development.

Notable cases:

- **Literal contexts.** Literal-heavy base64 and source-text inputs run
  at every forced literal-context rule, since the automatic choice would
  pick order-0 for most small test files.
- **Mismatched batches.** `GZP_FORCE_BATCH` exists because a
  `TableGroup`'s boundaries are fixed at compress time, but decompression
  picks its own batch size independently. The extremes suite compresses
  and decompresses with deliberately different batch sizes, in both
  directions, including with 256-context tables.
- **A match-free 1MB chunk.** The extremes suite includes a 2^20-byte De
  Bruijn sequence B(32,4), in which no 4-byte string repeats. It once
  produced undecodable output, because a single literal run of exactly
  2^20 is one past what the rANS length alphabet can code; such a chunk
  now falls back to plain tokens.

## Benchmarking

```
find /usr/include -name '*.h' | head -400 | xargs cat > corpus.txt
for i in $(seq 48); do cat corpus.txt; done > corpus_283mb.txt
REPEAT=5 bash bench/run_bench.sh corpus_283mb.txt                     # gzp default vs gzip/zstd
CPU=0 REPEAT=5 bash bench/run_bench.sh corpus_283mb.txt --profile ratio  # one gzp profile only
```

The script reports wall and kernel MB/s for gzp and compares against
`gzip -1/-6` and single-threaded `zstd -1/-3` when available. On a shared
GPU a single wall-clock measurement can be dominated by another process,
so `REPEAT=<n>` runs each codec n times and reports the best run per
column. Check `nvidia-smi` first: another process's memory shrinks gzp's
batches, and its compute slows gzp's global-memory-latency-bound parse.

**The 283MB recipe repeats one 5.4MB block 48x, which is not a neutral
choice of large file.** It once hid a real regression: a match-finding
table change that measured as basically free on it nearly halved
compress-kernel throughput on a genuinely varied file (see *Known
limitations*). Measure LZ-parse or table changes on a varied file too:

```
find /usr/include -name '*.h' | xargs cat > corpus_varied.txt   # all distinct
bash bench/run_bench.sh corpus_varied.txt
```

(Repeat `find`/`xargs cat` against more directories to reach a target
size while keeping the content non-repeating. The 1GB corpus in
*Results* was built that way.)

## Known limitations and next steps

- **Fixed startup cost.** CUDA context creation and allocation take
  ~0.2s on this WSL2 machine, over half of the wall time on the 283MB
  corpus. It amortises on larger inputs and is mostly outside gzp's
  control.
- **Decompression is `fwrite`-bound under WSL2.** Writing 1GB measured
  0.5–1.2s depending on page-cache state, while the decompress kernel
  needs ~0.25s, so the writer thread is almost always the bottleneck.
  gzp would need a faster filesystem path to go further, not a faster
  kernel.
- **Ratio vs zstd.** zstd's parser is more sophisticated than gzp's hash
  match finder with a two-step lazy lookahead: `zstd -3`'s output is ~0.5%
  smaller than gzp's `ratio` profile on the varied corpus. An optimal parser, or
  a match finder with longer chains, would be the next ratio lever. A
  third lazy step and a 64-byte probe cap were measured and did nothing
  useful (see the comments at `kLazySteps` and `kProbe`).
- **The match-finding table took several rounds to size.** The history,
  kept because each failure was informative:
  1. *Dynamic shared memory up to 64KB.* Requesting more than 48KB of
     shared memory at launch changed the SM's cache behaviour for the
     *whole* kernel. That cost was invisible on the repetitive corpus
     and nearly halved `ratio` compress throughput on the varied one.
  2. *A bigger table within the 48KB static limit.* This still cost
     38–52% of throughput for a 1–4% ratio gain, because more shared
     memory per one-warp block means fewer blocks per SM.
  3. *Global memory* sidestepped both and made `speed`/`balance` 1.5–3.6x
     faster by freeing shared memory for occupancy.
  4. *Re-tuned under occupancy-sized batches (this round).* The earlier
     "a bigger global table costs ~2x" had been measured with ~32 warps
     in flight, where every extra miss was exposed. With ~1000 in
     flight, `ratio`'s table grew 4x for 2.2% smaller output at 3–4%
     kernel cost and no wall-clock cost, and `balance`'s 2x for 2.9% at
     ~11% of wall throughput. `speed` stays at 2048 buckets: 4096 would
     save 1.6% for ~12% of wall throughput.
  One gotcha from those measurements: global-memory latency is far more
  sensitive to a *concurrent* GPU process than shared memory was. Under
  ~40% contention, the `ratio` compress kernel measured at half its
  actual speed.
- **The 4GB batch budget cap binds at `ratio`.** Even with 15GB of VRAM
  free, a 1GB file at the 1MB profile gets batches of ~350 chunks,
  because each chunk needs ~6MB of device memory. Kernel throughput
  follows chunks in flight, so raising `kMaxBudgetBytes` (`main.cu`) on
  large GPUs is the next thing to measure for that profile.
- **Literal-context choice is per batch, estimated from the histogram.**
  It is exact about which chunks can't use rANS, but not about which
  chunks will lose to plain tokens later, so a batch can occasionally
  carry 16 or 256 tables that few of its chunks use. The cost is bounded
  by the table bytes (4KB or 66KB per batch).
- **Expanded decode tables use ~1.3MB of device memory per table group**
  at 256 contexts, all expanded up front. That is ~10MB for a 1GB file at
  the default profile, but would grow to gigabytes for a file of many
  terabytes; expanding each decode batch's groups on demand would fix
  it.
- **No integrity check.** Structural corruption (bad offsets, sizes,
  truncation, invalid rANS streams) is detected, but the format has no
  checksum, so a corrupted literal or table byte that still decodes
  consistently produces wrong output silently.
- **`compute_repeat_codes`' encode-side pass is serial**, one lane per
  chunk. Disabling it costs only a few percent of compress-kernel
  throughput on text. Struct-like binary data with recurring strides
  should gain more from repeat offsets than prose does.
- **No multi-GPU, no streaming API** — it's a file-in, file-out CLI, and
  output must be seekable (the header is patched at the end).
- Match offsets are 32-bit, but chunks are capped at 1MB by policy
  (`kMaxChunkSize` in `src/format.h`) rather than by the wire format.
- **NPUs aren't a fit.** Recent CPUs' neural accelerators are dense
  matrix-multiply engines with no data-dependent branching or random
  gathers, which is all LZ matching and rANS consist of, and WSL2 does not
  expose them anyway.
