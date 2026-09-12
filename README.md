# gzp — a GPU-accelerated file compressor

`gzp` compresses and decompresses files using a CUDA kernel that runs a
simple LZSS scheme, one thread per data chunk. It's a from-scratch,
educational implementation, not a production compressor.

## How it works

The input file is split into fixed-size, independent chunks (8KB by
default). Each chunk is compressed (or decompressed) entirely on its own,
by a single CUDA thread — there is no cooperation between threads and no
cross-chunk references. This means:

- Throughput scales with the number of chunks, not with clever
  cross-thread cooperation. A small file has too few chunks to fill the
  GPU. In the current implementation even a large file doesn't fully
  fill it either: each thread's LZSS hash table is a 512-entry
  `uint32_t` array (2KB of per-thread local memory, confirmed via
  `nvcc -Xptxas -v`), and the batch-size cap in `pick_batch_chunks`
  (256MiB of device buffers) limits any single launch to roughly 15,000
  threads — well under the ~55-70k threads needed to saturate this GPU's
  36 SMs. That cap is the main reason measured throughput (~160MB/s) is
  in CPU-`gzip` territory rather than far above it; raising it (and
  re-measuring) is the natural next step, not attempted here.
- The compression window is the chunk itself (8KB), which caps the ratio
  compared to CPU compressors like `gzip` (32KB window) or `xz` (much
  larger). This is the main ratio-vs-parallelism tradeoff in the design —
  bigger chunks compress better but produce fewer, coarser-grained units
  of parallel work.
- Correctness is easy to reason about: verifying one chunk's round-trip
  verifies the algorithm, and chunks can't corrupt each other's decoded
  output.

This is a much simpler design than production GPU compressors — e.g.
nvCOMP's GPU LZ4 uses a warp or thread-block per chunk, with threads
cooperating on match-finding within the chunk. `gzp` uses one thread per
chunk instead, trading some peak throughput for simplicity.

### Wire format (`format.h`)

```
FileHeader   { magic, version, chunk_size, original_size, chunk_count }
ChunkEntry[] { offset, compressed_size, original_size }   -- one per chunk
payload      -- each chunk: [1 flag byte: Raw|Lzss] [chunk data]
```

Every chunk carries its own Raw/Lzss flag. The encoder always tries LZSS
first; if the result isn't smaller than the original, it stores the chunk
raw instead. This makes the format safe for incompressible or adversarial
input — worst case is original size + 1 byte per chunk, never unbounded
expansion.

### LZSS coding (`lzss_kernels.cuh`)

Standard byte-oriented LZSS: groups of up to 8 items prefixed by a flag
byte (bit = 1 means "match", 0 means "literal"). A match is 3 bytes
(2-byte offset, 1-byte length encoded as length-3, so length 3..258); a
literal is 1 raw byte. Match-finding uses a small per-thread hash table
(512 entries, single-candidate — no chaining) over 3-byte sequences,
which favors speed and simplicity over compression ratio.

### Batching and VRAM

The GPU in this environment is shared with other work (WSL hides the
Windows-side process from `nvidia-smi`, but `cudaMemGetInfo` still reports
real free memory). `gzp` queries free VRAM before each run and picks a
batch size — how many chunks to process per kernel launch — from a
quarter of currently-free memory, capped at 256MiB of device buffers. It
loops over batches, so files much larger than free VRAM still work.

## Building

Requires CUDA 12.8+ and CMake 3.20+. Targets sm_120 (RTX 5060 Ti /
Blackwell) by default; override for other hardware:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release [-DCMAKE_CUDA_ARCHITECTURES=<arch>]
cmake --build build -j
```

(`CMAKE_CUDA_ARCHITECTURES=native` is not used here because it silently
fell back to a very old default (sm_52) instead of detecting the GPU
under WSL — pin the architecture explicitly.)

## Usage

```
./build/gzp c <input> <output> [chunk_size]   # compress (default chunk_size 8192)
./build/gzp d <input> <output>                # decompress
```

## Testing

```
bash tests/round_trip.sh ./build/gzp
```

Covers empty input, 1 byte, sub-chunk, exact chunk multiples, off-by-one,
all-zeros, random (incompressible) data, and mixed text, all round-tripped
and byte-compared. Set `EXTRA_FILE=<path>` to also test a real file, and
`CHUNK=<n>` to test a non-default chunk size (verified at both extremes:
`CHUNK=64` and `CHUNK=65536`, the latter exactly saturating the 2-byte
match-offset field).

## Benchmarking

```
bash bench/run_bench.sh <file>
```

Times compress+decompress end-to-end (including file I/O and host<->device
transfer, not just kernel time) against `gzip -1` and `gzip -6`. Example,
~98MB of concatenated system headers repeated to add redundancy, on an
RTX 5060 Ti (GPU shared with other work) vs single-threaded CPU `gzip`:

| | compress | decompress | ratio |
|---|---|---|---|
| gzp (GPU) | 163 MB/s | 209 MB/s | 0.535 |
| gzip -1 (CPU) | 135 MB/s | 239 MB/s | 0.273 |
| gzip -6 (CPU) | 53 MB/s | 252 MB/s | 0.222 |

`gzp` is faster than `gzip` here but compresses noticeably worse — the
8KB window and single-candidate hash chain leave real ratio on the table
compared to `gzip`'s 32KB window and proper hash chains. That's the
expected shape of this design's tradeoff, not a bug: a bigger window
would close the ratio gap at the cost of fewer, coarser parallel chunks
and more per-thread local memory.

## Known limitations

- Compression ratio trails general-purpose CPU compressors due to the
  small per-chunk window and simplistic (non-chained) match finder.
- Compressed chunk data is transferred to/from the GPU in fixed
  worst-case-sized slots rather than tightly packed, wasting some PCIe
  bandwidth on chunks that compressed well. Fine for a first version;
  packing/compaction would be the next thing to optimize.
- Single GPU stream — batches run sequentially with no overlap between
  a batch's H2D transfer, kernel, and D2H transfer.
