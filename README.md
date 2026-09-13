# gzp — a GPU file compressor

`gzp` compresses and decompresses files on an NVIDIA GPU with CUDA. Each
chunk of the input (64KB by default, up to 1MB) is handled by one warp: an
LZ parse where all 32 lanes search for matches together, followed by a
32-way interleaved rANS entropy coder. It's a from-scratch, educational
implementation — not a drop-in replacement for zstd — but it beats
single-threaded `gzip -1` on both ratio and speed at every chunk size, and
its `balance`/`ratio` profiles beat `zstd -1`'s ratio too (never `zstd
-3`'s, or `gzip -6`'s — see *Results*).

## Results

283MB of concatenated C headers (`/usr/include`, see *Benchmarking*),
RTX 5060 Ti under WSL2, GPU otherwise idle, CPU tools single-threaded,
best of 5 runs (`REPEAT=5`) per column. Wall figures are whole-process
(file I/O, PCIe copies, and for gzp ~0.15–0.3s of CUDA context
creation); the kernel columns are GPU time only. **This corpus is 48
repeats of one 5.4MB text block — see the caveat in *Benchmarking* about
why validating against a genuinely varied large file matters here.**

| codec | compress MB/s (wall) | kernel MB/s | decompress MB/s (wall) | kernel MB/s | ratio |
|---|---|---|---|---|---|
| **gzp** (`--profile speed`, 64KB) | **628** | 2458 | **643** | 2993 | 0.2775 |
| gzip -1 | 145 | – | 262 | – | 0.2985 |
| gzip -6 | 56 | – | 292 | – | 0.2457 |
| zstd -1 (1 thread) | 538 | – | 1381 | – | 0.2724 |
| zstd -3 (1 thread) | 408 | – | 1320 | – | 0.2426 |

gzp beats `gzip -1` on both ratio and speed at every chunk size, and now
beats `zstd -1`'s compress speed too (its hand-tuned decoder is still
faster). Ratio (lower is better) is a mixed picture against zstd: at the
default `speed` profile gzp is *larger* than `zstd -1` (0.2775 vs
0.2724); `zstd -3` (0.2426) and `gzip -6` (0.2457) beat every gzp
profile, `ratio` included, though `ratio` (0.2500) is now clearly closer
to them than it used to be (see below). Earlier measurements taken while
another process held ~90% of the GPU showed much lower and noisier
numbers purely from contention — `REPEAT=<n>` (see *Benchmarking*)
exists because of exactly that, and it mattered more than usual while
tuning the match-finding table below (see *Known limitations*).

Chunk size trades ratio for speed — bigger chunks give the match finder
more history to search, at the cost of fewer, coarser-grained units of
GPU parallelism. `--profile <name>` (see *Usage*) selects one of three
presets:

| profile | chunk | ratio | compress kernel MB/s | decompress kernel MB/s |
|---|---|---|---|---|
| speed (default) | 64KB | 0.2775 | 2458 | 2993 |
| balance | 256KB | 0.2687 | 805 | 829 |
| ratio | 1MB | **0.2500** | 214 | 205 |

`balance` and `ratio` both beat `zstd -1`'s 0.2724 on this corpus, at a
real compress/decompress-speed cost relative to zstd (not relative to
gzp's own other profiles -- see *Known limitations* for why `ratio`
here is barely slower than it used to be despite compressing noticeably
better than before). Neither beats `zstd -3`'s 0.2426 or `gzip -6`'s
0.2457.

## How it works

### Chunks and warps

The input is split into independent, fixed-size chunks (64KB by default,
up to 1MB — see `--profile` in *Usage*). One warp — 32 lanes — compresses
or decompresses one chunk; a few hundred warps are resident on the GPU at
once, so batches of a few hundred chunks keep it full at the default
chunk size (fewer, larger chunks fill it less well, one reason bigger
chunks cost speed as well as buying ratio). There are no cross-chunk
references: any chunk can be decoded on its own, and a corrupt chunk
cannot damage another.

### LZ parse (`src/lz_warp.cuh`)

The warp walks a chunk in 32-byte windows. Every lane hashes the 4 bytes
at its own position, reads a 4-way bucket from this chunk's hash table
(one u32 chunk-relative position per word, since positions no longer fit
in 16 bits once chunks can exceed 64KB) and compares against all four
candidates, capped at 32 bytes so per-lane work is bounded. One
deterministic lane per bucket then inserts its position, evicting the
oldest of the four; lanes that found nothing re-probe once more so
repeats shorter than a window apart are caught immediately.

The table lives in **global memory**, one region per chunk (sized by
`hash_table_bits()`, `kernels.h`), rather than shared memory — after two
earlier attempts at growing a *shared*-memory version both measured real
regressions instead (see *Known limitations*), moving it to global
memory sidestepped both problems and, unexpectedly, made the smaller
chunk profiles faster too: freeing the shared memory this kernel used to
reserve let more of its (tiny, 32-thread) blocks run concurrently per
SM, more than paying for global memory's higher per-access latency at
the table sizes `speed`/`balance` use. `ratio`'s bigger chunks get a 4x
bigger table (see *Known limitations*) for a real ratio win at a real,
but now much smaller and non-corpus-dependent, speed cost.

The window's matches are selected warp-uniformly from a ballot mask with
a lazy lookahead of up to `kLazySteps` positions (take position *i+1*'s
match instead if it's clearly longer, then *i+2* if that's clearer
longer still, and so on) — zstd calls the 2-step version "lazy2". Matches
that hit the 32-byte probe cap are extended cooperatively, 32 bytes per
step, so long runs never serialise on one lane.

The parse emits sequences — a literal run followed by a match `(offset,
length)` — into scratch for the entropy stage below. Per chunk, the
encoder still keeps whichever of the rANS-coded result or a plain
LZ4-style token stream comes out smaller (see *rANS stage*).

### rANS stage (`src/rans.cuh`, `src/rans_codes.h`)

Literals, literal-run lengths, match lengths and offsets are coded with
rANS. Length and offset alphabets are zstd-style log2 buckets with raw
extra bits (written straight into the rANS state, so there is no side
stream) — except the offset alphabet's top 3 codes, which are
**repeat-offset codes**: "reuse the 1st/2nd/3rd most-recently-used
distinct match offset" instead of coding a fresh magnitude, zstd-style,
for data (especially struct/record-shaped binary formats) that reuses a
handful of strides constantly. That reuse state is a genuine
sequence-order dependency that the interleaved decoder's own per-group
parallel decode doesn't carry across lanes for free: the encoder
precomputes it in one forward serial pass over the already-parsed
sequences (`compute_repeat_codes()`, cheap next to the LZ parse that
produced them), and the decoder replays the same state machine
per-group with a register-only shuffle walk (see the comment above
`rans_decode_warp` for the mechanics) rather than any new rANS step.

The frequency table is **shared by every chunk compressed in
the same host batch** (a "table group" — see *Container format*) rather
than stored per chunk: compression runs as three kernels per batch —
parse every chunk into scratch while atomically accumulating one shared
histogram, quantise+normalise it once, then encode every chunk against
that shared table. This removes the ~352-byte/chunk table overhead an
earlier per-chunk-table design paid.

The GPU-specific part is the interleaving: 32 rANS states, one per lane,
share **one** stream of 16-bit words. All lanes step in lockstep; the
lanes that need to renormalise on a step write (or read) their word
contiguously in lane order, located with a ballot and a popcount. With a
2^16 state floor and 16-bit words, each step moves at most one word per
lane, which is what makes the encoder's and decoder's per-step word
counts line up exactly without storing any per-lane offsets. The encoder
runs in reverse and writes backward from the end of its output slot; the
decoder reads forward.

Each chunk keeps whichever is smaller of the rANS payload and the plain
token stream (highly repetitive chunks are better off without any table
at all), and falls back to raw storage if neither beats the input.

### Decoding

Token decoding is warp-cooperative: 32 lanes copy each literal run and
each match, with overlapping matches handled by indexing `k mod offset`
into already-written history so no serial path is needed. rANS decoding
runs the same 32-lane lockstep as the encoder into scratch (looking up
its symbols in a coarse 128-entry index per alphabet plus a short linear
scan, rather than a binary search), then the same reconstruction loop
rebuilds the chunk. Because the table is shared per batch, decompression
expands every table group's frequencies once up front (into a global
buffer looked up by a per-chunk group id) instead of rebuilding one per
chunk — which also means the decode kernel needs no shared memory for
tables at all anymore. Malformed input sets an error flag that the host
turns into an error rather than garbage output.

### Host pipeline (`src/main.cu`)

Batches of chunks flow through a ring of three pinned buffer sets on
three CUDA streams, so batch *i+1*'s file read and upload overlap batch
*i*'s kernels and batch *i−1*'s download and write. Compressed output is
compacted on the GPU (a CUB scan plus a pack kernel) so the download
moves only compressed bytes, and the sized download is deferred by one
batch so the host never stalls on the launch it just made. Batch size is
planned from free VRAM (the GPU may be shared) with a 32MB minimum, and
allocation retries with a halved batch on failure.

### Container format (`src/format.h`)

```
FileHeader    { magic, version=4, chunk_size, original_size, chunk_count, table_group_count }
ChunkEntry[]  { offset, compressed_size, original_size }         -- one per chunk
TableGroup[]  { start_chunk, chunk_count, q[352] }               -- one per compression batch
payload       -- each chunk: [flag: Raw | Lz | LzRans] [data]
```

`TableGroup` entries cover `[0, chunk_count)` contiguously and in
order; chunk *c*'s rANS table (if it used one) is whichever group's
range contains *c* — always exactly one host compression batch's worth
of chunks, decided at compress time and independent of whatever batch
size decompression later happens to choose. Not every chunk in a group
necessarily uses that table (a chunk can still individually fall back
to `Lz` or `Raw`), but every group is written regardless.

Worst case is the original size plus one byte per chunk plus one table
per batch (~360 bytes), so a file compressed in very few batches (a
small input, or one forced small with `GZP_FORCE_BATCH`) pays that
overhead even when nothing in it uses rANS.

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
`__launch_bounds__`'s `minBlocksPerSM` hint on both kernels, for A/B
occupancy testing — see the comment above it in `CMakeLists.txt` for
why a bare `--maxrregcount` can't do this (both kernels already carry
an explicit `__launch_bounds__`, which takes precedence). The parse
kernel's match-finding table now lives in global memory (see *LZ
parse*), so it uses no shared memory at all; measured (36 registers/
thread for parse, 52 for decompress, no spills per `nvcc -Xptxas -v`)
with `GZP_MIN_BLOCKS_PER_SM=4` and saw no measurable change on either
kernel — both are apparently already at whatever ceiling the hardware's
max-blocks-per-SM limit imposes on small (32-thread) blocks, not a
register or shared-memory one, so this knob is mostly useful as a
regression check that a future change hasn't pushed either kernel into
register spilling.

## Usage

```
./build/gzp c <input> <output> [chunk_size]                # compress (default chunk_size 65536)
./build/gzp c <input> <output> --profile speed|balance|ratio  # ...or pick a chunk-size preset
./build/gzp d <input> <output>                              # decompress
GZP_VERBOSE=1 ./build/gzp ...                               # per-stage timing
GZP_FORCE_BATCH=<n> ./build/gzp ...                         # force chunks/batch (testing only)
```

`chunk_size` and `--profile` are mutually exclusive (specifying both is
an error); the bare default (neither given) is unchanged from before
`--profile` existed. See *Results* for what each profile actually costs
and buys — `balance` and `ratio` trade real compress/decompress speed
for a smaller output.

## Testing

```
bash tests/round_trip.sh                 # default chunk size, plus --profile cases
bash tests/round_trip.sh --extremes      # chunk sizes 1, 16, 4K, 8K, 32K, 65535, 65536,
                                          # 1048575, 1048576 (kMaxChunkSize), plus
                                          # GZP_FORCE_BATCH mismatch cases (see below)
BIG=1 bash tests/round_trip.sh           # + 300MB random and 300MB text (multi-batch)
```

`GZP_FORCE_BATCH` exists because a `TableGroup`'s boundaries are fixed
at compress time to whatever batch size compression used, but
decompression picks its own batch size independently (different free
VRAM, a different machine) — so decompression's per-chunk group-id
lookup must work no matter how the two are misaligned. The extremes
suite compresses and decompresses with deliberately different forced
batch sizes (both directions) to exercise exactly that.

Every case is round-tripped on the GPU *and* decoded by
`tests/ref_decode.cpp`, a CPU decoder that shares no code with the GPU
path, so a symmetric bug in the GPU encoder and decoder can't hide. That
matters here because `compute-sanitizer` in CUDA 12.8 does not support
this GPU, so memcheck was not available during development.

## Benchmarking

```
find /usr/include -name '*.h' | head -400 | xargs cat > corpus.txt
for i in $(seq 48); do cat corpus.txt; done > corpus_283mb.txt
bash bench/run_bench.sh corpus_283mb.txt [chunk_size]
```

The script reports wall and kernel MB/s for gzp and compares against
`gzip -1/-6` and single-threaded `zstd -1/-3` when available. It
runs each codec once by default; on a shared GPU a single wall-clock
measurement can be dominated by another process's contention rather than
by gzp itself, so set `REPEAT=<n>` to run each codec n times and report
the best (highest-throughput) run per column instead:

```
REPEAT=5 bash bench/run_bench.sh corpus_283mb.txt
```

**This recipe repeats one 5.4MB block 48x, which is not a neutral choice
of large file.** It found (and hid) a real bug in this project's history:
a match-finding table size that measured as basically free on this
corpus cost roughly 2x compress-kernel throughput on a genuinely varied
1GB file built from real, non-repeated headers (see *Known limitations*).
Any future change to the LZ parse or table sizing should also be
measured against a file like this before trusting the repeated-corpus
number:

```
find /usr/include -name '*.h' | xargs cat > corpus_varied.txt   # ~ a few hundred MB, all distinct
bash bench/run_bench.sh corpus_varied.txt [chunk_size]
```

(Repeat `find`/`xargs cat` against more directories, or `cat` multiple
such runs together, to reach a specific target size while keeping the
content genuinely non-repeating — do not use the 48x-repeat trick above
for this purpose, since that is exactly the shape that hid the bug.)

## Known limitations and next steps

- **Fixed startup cost.** CUDA context creation alone takes 0.14–0.19s
  on this WSL2 machine; on a 94MB file that is half the wall time. It
  amortises on larger inputs and is outside gzp's control.
- **Ratio vs zstd.** gzp's rANS codes literals with an order-0 model and
  its parser is a hash match finder with a global-memory table plus a
  short lazy lookahead (see *LZ parse*); `zstd -1` beats gzp's `speed`
  profile (0.2724 vs 0.2775 on this corpus) and `zstd -3`/`gzip -6` beat
  every gzp profile, `ratio` included, though `ratio`'s gap to them
  closed noticeably this round (see *Results*). A real optimal parser
  (rather than greedy-plus-lookahead) and a context-mixing literal model
  would improve the ratio further at every profile.
- **The match-finding hash table's capacity took two failed attempts
  and a redesign to actually scale with chunk size**, worth recording in
  full since the failures were informative:
  1. *Dynamic shared memory, up to 64KB.* Requesting more than 48KB of
     shared memory at launch measurably changes something about the SM's
     cache behaviour for the *whole* kernel, not just the table. On the
     repetitive 283MB corpus above that cost was invisible (compress
     kernel MB/s barely moved for a real ratio gain); on the genuinely
     varied 1GB file below, the exact same code and growth path nearly
     halved the `ratio` profile's compress-kernel throughput. Confirmed
     the trigger is the bytes actually requested at launch, not merely
     raising the kernel's ceiling via `cudaFuncSetAttribute`.
  2. *A bigger table fully inside the 48KB static default* (4096
     buckets, 3-way instead of 4-way, no opt-in involved at all). Still
     cost real throughput on both corpora (~38-52%, for only a ~1-4%
     ratio gain) -- growing *any* static shared memory a 1-warp-per-block
     kernel uses reduces how many of its blocks fit per SM, a normal,
     unrelated occupancy cost. A modulo-hashed 3072-bucket variant tried
     first regressed ratio *and* speed, tracked down to the multiplicative
     hash constant being designed for its high bits (used by the
     shift-based hash) rather than its low bits (what `%` on a
     non-power-of-two reads); re-shifting before the modulo recovered the
     ratio but not the speed, since integer modulo itself is expensive
     per lane per window.
  3. **Global memory** (what's shipped): moving the table off shared
     memory entirely sidesteps both costs and, measured cleanly (GPU
     otherwise idle -- see the note on contention below), made `speed`
     and `balance` 1.5-3.6x *faster* at an unchanged ratio (freeing the
     shared memory this kernel used to reserve let more of its blocks
     run per SM), and let `ratio` grow to a 4x bigger table (8192
     buckets) for its best ratio yet (0.2500 on the 283MB corpus, 0.2329
     on the 1GB one -- 6-7% smaller either way) at only a ~3-8% compress-
     kernel cost, consistent across both corpora rather than the
     cliff attempts 1 and 2 hit. `hash_table_bits()` (`kernels.h`) is a
     measured two-tier choice (unchanged up to `balance`'s 256KB, 4x
     bigger above it), not a general formula, since only 3 chunk sizes
     are actually exercised.
  One genuine gotcha hit while measuring attempt 3: global memory's
  higher per-access latency is far more sensitive to a *concurrent* GPU
  process than shared memory was -- the same benchmark run under ~40%
  contention from another process measured `ratio`'s compress kernel at
  roughly half its actual (contention-free) speed, which would have been
  wrongly reported as a regression had `REPEAT=<n>` not been re-run once
  the GPU cleared. Always check `nvidia-smi` before trusting a global-
  memory-table benchmark number.
- **compute_repeat_codes' encode-side pass is serial, not
  warp-parallel.** Resolving repeat-offset codes (see *rANS stage*)
  needs the exact sequence-order state the decoder will reconstruct, so
  one lane walks all of a chunk's sequences before the parallel rANS
  encode step runs. Measured in isolation (disabling the call) this
  costs only a few percent of compress-kernel throughput on this text
  corpus, not the double-digit slowdown once (wrongly) attributed to it
  — that slowdown was actually the hash-table item above, confounding
  the two changes in the same measurement. struct-of-arrays/binary
  formats with recurring strides should see a bigger ratio win from
  repeat offsets than this prose-like text does.
- **rANS decode is inherently more work than a plain token stream's
  direct byte copies** — even with the coarse-LUT lookup and per-batch
  tables (which removed the earlier per-chunk table-rebuild cost
  entirely), decoding still means an integer divide/multiply and a
  table lookup per symbol, versus a token stream's `memcpy`-shaped
  literal and match copies. This is why a chunk that doesn't compress
  much better under rANS is kept as plain tokens instead.
- **No multi-GPU, no streaming API** — it's a file-in, file-out CLI.
- Match offsets are 32-bit, but chunks are capped at 1MB by policy (see
  `kMaxChunkSize` in `src/format.h`) rather than by the wire format.
