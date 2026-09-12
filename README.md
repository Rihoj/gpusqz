# gzp — a GPU file compressor

`gzp` compresses and decompresses files on an NVIDIA GPU with CUDA. Each
64KB chunk of the input is handled by one warp: an LZ parse where all 32
lanes search for matches together, followed by a 32-way interleaved rANS
entropy coder. It's a from-scratch, educational implementation — not a
drop-in replacement for zstd — but on text it beats single-threaded
`gzip -1` on both ratio and speed.

## Results

283MB of concatenated C headers (`/usr/include`, see *Benchmarking*),
RTX 5060 Ti under WSL2, GPU otherwise idle, CPU tools single-threaded,
best of 5 runs (`REPEAT=5`) per column. Wall figures are whole-process
(file I/O, PCIe copies, and for gzp ~0.15–0.3s of CUDA context
creation); the kernel columns are GPU time only.

| codec | compress MB/s (wall) | kernel MB/s | decompress MB/s (wall) | kernel MB/s | ratio |
|---|---|---|---|---|---|
| **gzp** (lzrans, 64KB) | **583** | 1598 | **629** | 3759 | **0.272** |
| gzp (lz only, 64KB) | 556 | 1594 | 643 | 5498 | 0.370 |
| gzip -1 | 149 | – | 262 | – | 0.284 |
| gzip -6 | 59 | – | 293 | – | 0.231 |
| zstd -1 (1 thread) | 513 | – | 1400 | – | 0.252 |
| zstd -3 (1 thread) | 390 | – | 1301 | – | 0.225 |

gzp beats `gzip -1` on both ratio and speed, and beats `zstd -1`'s ratio
and compress speed (zstd's hand-tuned decoder is still faster to
decode). Earlier measurements taken while another process held ~90% of
the GPU showed much lower and noisier numbers purely from contention —
`REPEAT=<n>` (see *Benchmarking*) exists because of exactly that.

Chunk size trades ratio for per-chunk latency; 64KB is the default and
the maximum the 16-bit match offsets allow. With the GPU idle, smaller
chunks turn out to decode substantially faster too (more, smaller units
of parallel work), so this is a real ratio-vs-speed dial, not just a
ratio-vs-latency one:

| chunk | mode | ratio | compress kernel MB/s | decompress kernel MB/s |
|---|---|---|---|---|
| 16KB | lzrans | 0.297 | 2372 | 8106 |
| 32KB | lzrans | 0.281 | 2073 | 6644 |
| 64KB | lzrans | 0.272 | 1585 | 3741 |
| 64KB | lz | 0.370 | 1541 | 5286 |

## How it works

### Chunks and warps

The input is split into independent, fixed-size chunks (default 64KB).
One warp — 32 lanes — compresses or decompresses one chunk; a few hundred
warps are resident on the GPU at once, so batches of a few hundred
chunks keep it full. There are no cross-chunk references: any chunk can
be decoded on its own, and a corrupt chunk cannot damage another.

### LZ parse (`src/lz_warp.cuh`)

The warp walks a chunk in 32-byte windows. Every lane hashes the 4 bytes
at its own position, reads a 4-way bucket from a per-warp shared-memory
hash table (1024 buckets, four 16-bit positions each, packed two per u32
word — 8KB per warp total, same footprint as an earlier 2-way/2048-bucket
design that measured worse) and compares against all four candidates,
capped at 32 bytes so per-lane work is bounded. One deterministic lane
per bucket then inserts its position, evicting the oldest of the four;
lanes that found nothing re-probe once more so repeats shorter than a
window apart are caught immediately. The window's matches are selected
warp-uniformly from a ballot mask with a one-position lazy lookahead
(take position *i+1*'s match instead if it's clearly longer), and matches
that hit the 32-byte cap are extended cooperatively, 32 bytes per step,
so long runs never serialise on one lane.

The parse emits sequences — a literal run followed by a match `(offset,
length)` — either as LZ4-style tokens (`--mode lz`) or into scratch for
the entropy stage.

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
FileHeader    { magic, version=3, chunk_size, original_size, chunk_count, table_group_count }
ChunkEntry[]  { offset, compressed_size, original_size }         -- one per chunk
TableGroup[]  { start_chunk, chunk_count, q[352] }               -- one per compression batch (LzRans only)
payload       -- each chunk: [flag: Raw | Lz | LzRans] [data]
```

`TableGroup` entries cover `[0, chunk_count)` contiguously and in
order; chunk *c*'s rANS table (if it used one) is whichever group's
range contains *c* — always exactly one host compression batch's worth
of chunks, decided at compress time and independent of whatever batch
size decompression later happens to choose. `--mode lz` files (no
chunk can ever be `LzRans`) have zero table groups.

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
an explicit `__launch_bounds__`, which takes precedence) and why it's
unlikely to change occupancy today (both kernels are shared-memory-
bound, not register-bound, at their current sizes).

## Usage

```
./build/gzp c <input> <output> [chunk_size] [--mode lz|lzrans]   # compress (default 65536, lzrans)
./build/gzp d <input> <output>                                   # decompress
GZP_VERBOSE=1 ./build/gzp ...                                    # per-stage timing
GZP_FORCE_BATCH=<n> ./build/gzp ...                              # force chunks/batch (testing only)
```

## Testing

```
bash tests/round_trip.sh                 # default chunk size, both modes
bash tests/round_trip.sh --extremes      # chunk sizes 1, 16, 4K, 8K, 32K, 65535, 65536, both modes,
                                          # plus GZP_FORCE_BATCH mismatch cases (see below)
BIG=1 bash tests/round_trip.sh           # + 300MB random and 300MB text (multi-batch), both modes
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

The script reports wall and kernel MB/s for both gzp modes and compares
against `gzip -1/-6` and single-threaded `zstd -1/-3` when available. It
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
- **Ratio vs zstd.** gzp's rANS codes literals with an order-0 model and
  its parser is a single-pass hash match finder; zstd -1 is still ~8%
  smaller on this corpus even after moving to one shared table per batch
  (a per-chunk table cost ~1-2% more). Repeat-offset codes and a better
  parser (hash chains, optimal parsing) would close more of the gap.
- **Decode speed still trails the token-only path** (3.7 GB/s vs 5.3 GB/s
  kernel at 64KB) despite the coarse-LUT decode and per-batch tables
  (which removed the earlier per-chunk table-rebuild cost entirely) —
  the remaining gap is the extra work rANS decode itself does per
  symbol versus a token stream's direct byte copies.
- **No multi-GPU, no streaming API** — it's a file-in, file-out CLI.
- Match offsets are 16-bit, which caps chunks at 64KB.
