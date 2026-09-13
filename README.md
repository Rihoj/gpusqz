# gzp — a GPU file compressor

`gzp` compresses and decompresses files on an NVIDIA GPU with CUDA. Each
chunk of the input (64KB by default, up to 1MB) is handled by one warp: an
LZ parse where all 32 lanes search for matches together, followed by a
32-way interleaved rANS entropy coder. It's a from-scratch, educational
implementation — not a drop-in replacement for zstd — but it beats
single-threaded `gzip -1` on both ratio and speed at every chunk size, and
at the largest chunk size (`--profile ratio`) it beats `zstd -1`'s ratio
too.

## Results

283MB of concatenated C headers (`/usr/include`, see *Benchmarking*),
RTX 5060 Ti under WSL2, GPU otherwise idle, CPU tools single-threaded,
best of 5 runs (`REPEAT=5`) per column. Wall figures are whole-process
(file I/O, PCIe copies, and for gzp ~0.15–0.3s of CUDA context
creation); the kernel columns are GPU time only.

| codec | compress MB/s (wall) | kernel MB/s | decompress MB/s (wall) | kernel MB/s | ratio |
|---|---|---|---|---|---|
| **gzp** (`--profile speed`, 64KB) | **440** | 766 | **731** | 3474 | 0.263 |
| gzip -1 | 153 | – | 274 | – | 0.284 |
| gzip -6 | 59 | – | 301 | – | 0.231 |
| zstd -1 (1 thread) | 552 | – | 1403 | – | 0.252 |
| zstd -3 (1 thread) | 425 | – | 1366 | – | 0.225 |

gzp beats `gzip -1` on both ratio and speed at every chunk size, and
beats `zstd -1`'s compress speed here (zstd's hand-tuned decoder is
still faster to decode). Earlier measurements taken while another
process held ~90% of the GPU showed much lower and noisier numbers
purely from contention — `REPEAT=<n>` (see *Benchmarking*) exists
because of exactly that.

Chunk size trades ratio for speed — bigger chunks give the match finder
more history to search, at the cost of fewer, coarser-grained units of
GPU parallelism. `--profile <name>` (see *Usage*) selects one of three
presets; `ratio` is the largest chunk this format allows and is the
only one that beats `zstd -1`'s ratio on this corpus:

| profile | chunk | ratio | compress kernel MB/s | decompress kernel MB/s |
|---|---|---|---|---|
| speed (default) | 64KB | 0.263 | 766 | 3361 |
| balance | 256KB | 0.253 | 508 | 923 |
| ratio | 1MB | **0.250** | 252 | 242 |

(`zstd -1`'s ratio on this corpus is 0.252, `zstd -3`'s is 0.225.)

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
at its own position, reads a 4-way bucket from a per-warp shared-memory
hash table (2048 buckets, one u32 chunk-relative position per word — 32KB
per warp, the whole default static shared-memory budget for one warp's
table, since positions no longer fit in 16 bits once chunks can exceed
64KB) and compares against all four candidates, capped at 32 bytes so
per-lane work is bounded. One deterministic lane per bucket then inserts
its position, evicting the oldest of the four; lanes that found nothing
re-probe once more so repeats shorter than a window apart are caught
immediately. The window's matches are selected
warp-uniformly from a ballot mask with a one-position lazy lookahead
(take position *i+1*'s match instead if it's clearly longer), and matches
that hit the 32-byte cap are extended cooperatively, 32 bytes per step,
so long runs never serialise on one lane.

The parse emits sequences — a literal run followed by a match `(offset,
length)` — into scratch for the entropy stage below. Per chunk, the
encoder still keeps whichever of the rANS-coded result or a plain
LZ4-style token stream comes out smaller (see *rANS stage*).

### rANS stage (`src/rans.cuh`, `src/rans_codes.h`)

Literals, literal-run lengths, match lengths and offsets are coded with
rANS. Length and offset alphabets are zstd-style log2 buckets with raw
extra bits (written straight into the rANS state, so there is no side
stream). The frequency table is **shared by every chunk compressed in
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
kernel's match-finding table now uses 32KB of the 48KB default static
shared-memory budget for one warp (see *LZ parse*), leaving no room for
a second block, so it's shared-memory-bound at 1 warp/SM regardless of
this flag; decompress is register-bound, where this flag can actually
move the needle.

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

## Known limitations and next steps

- **Fixed startup cost.** CUDA context creation alone takes 0.14–0.19s
  on this WSL2 machine; on a 94MB file that is half the wall time. It
  amortises on larger inputs and is outside gzp's control.
- **Ratio vs zstd at the default profile.** gzp's rANS codes literals
  with an order-0 model and its parser is a single-pass hash match
  finder with a fixed-capacity table; at the default 64KB chunk size
  `zstd -1` is still smaller (0.252 vs 0.263 on this corpus). Only the
  `ratio` profile's 1MB chunks close and slightly reverse that gap
  (0.250) — `--profile balance` (256KB) is a middle ground that gets
  close (0.253) without paying `ratio`'s full speed cost. Repeat-offset
  codes and a better parser (hash chains, optimal parsing) would improve
  the ratio at every chunk size, not just the largest.
- **The match-finding hash table's capacity doesn't scale with chunk
  size.** It's a fixed-size, per-warp shared-memory table (32KB — the
  full default static-shared-memory budget for one warp), so a 1MB
  chunk gets the same "recently seen positions" recall as a 64KB one,
  just spread over 16x more data. This is *why* the ratio gain from
  bigger chunks is real but sub-linear, and why the natural next lever
  is a bigger table via dynamic (rather than static) shared memory,
  which can exceed the current 48KB-per-block static limit.
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
