# Design

How gpusqz compresses and decompresses a file. The CUDA sources are named
below; the Vulkan backend implements the same steps in `src/vk/*.comp`
and writes byte-identical files (see [GPU backends](backends.md)).

## Chunks and lanes

The input is split into independent, fixed-size chunks (64KB by default,
up to 1MB — see `--profile` in [Usage](usage.md#profiles)). One warp — 32
lanes — compresses or decompresses one chunk. A warp's LZ parse is bound
by the latency of its dependent, random hash-table and history reads, not
by bandwidth, so the GPU's throughput is roughly the number of chunks in
flight times a few MB/s each; batches are sized to keep at least ~1000 in
flight (see [Host pipeline](#host-pipeline)). There are no cross-chunk
references: any chunk can be decoded on its own, and a corrupt chunk
cannot damage another.

## LZ parse

`src/lz_warp.cuh`. The warp walks a chunk in 32-byte windows. Every lane
hashes the 4 bytes at its own position, reads a 4-way bucket from this
chunk's hash table (one u32 chunk-relative position per word) and
compares against all four candidates, capped at 32 bytes so per-lane work
is bounded. One deterministic lane per bucket then inserts its position,
evicting the oldest of the four; lanes that found nothing re-probe once
more so repeats shorter than a window apart are caught immediately.

The table lives in **global memory**, one region per chunk, sized per
profile by `hash_table_bits()` (`kernels.h`): 2048 buckets at `speed`,
4096 at `balance`, 32768 at `ratio`. Two earlier attempts at a bigger
*shared*-memory table both regressed; global memory avoids those costs,
and freeing the shared memory let more of the kernel's one-warp blocks
run per SM. The four rounds it took to get there are in [Performance
history](performance-history.md#sizing-the-match-finder-table).

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

## rANS stage

`src/rans.cuh`, `src/rans_codes.h`. Literals, literal-run lengths, match
lengths and offsets are coded with rANS. Length and offset alphabets are
zstd-style log2 buckets with raw extra bits (written straight into the
rANS state, so there is no side stream) — except the offset alphabet's
top 3 codes, which are **repeat-offset codes**: "reuse the 1st/2nd/3rd
most-recently-used distinct match offset", zstd-style. The encoder
resolves them in one forward serial pass over the parsed sequences
(`compute_repeat_codes()`), and the decoder replays the same state
machine per group of 32 sequences with a register-only shuffle walk.

**Literals use order-1 contexts.** Each literal is coded with a table
chosen by the literal before it: its high nibble (16 tables), all of it
(256 tables), or nothing (1 table, order-0). For the decoder to know the
previous literal, each of the 32 lanes owns one contiguous run of the
chunk's literal stream rather than every 32nd literal, and the first
literal of each run uses context 0. Runs are 4-byte aligned, so the
decoder stores 4 literals at a time.

**Tables are shared per batch** (a "table group", see [Container
format](#container-format)): compression runs three kernels per batch.
The first parses every chunk into scratch while atomically accumulating
one full order-1 histogram for the batch. The second, one 256-thread
block, folds that histogram to 16 and 1 contexts and keeps whichever rule
minimises the estimated literal bits under the tables the encoder would
actually build, plus 256 table bytes per context. On large text batches
that is nearly always all 256 contexts (~66KB of tables per batch), while
incompressible or literal-poor batches keep one table and pay nothing
extra. The third encodes every chunk against that table. A chunk whose
plain token stream is shorter than a rANS header can never end up
rANS-coded, so it stays out of the histogram and skips the rANS attempt.

The GPU-specific part is the interleaving: 32 rANS states, one per lane,
share **one** stream of 16-bit words. All lanes step in lockstep; the
lanes that need to renormalise on a step write (or read) their word
contiguously in lane order, located with a ballot and a popcount. With a
2^16 state floor and 16-bit words, each step moves at most one word per
lane, which is what makes the encoder's and decoder's per-step word
counts line up exactly without storing any per-lane offsets. The encoder
runs in reverse and writes backward from the end of its output slot; the
decoder reads forward.

## Decoding

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

## Host pipeline

`src/main.cpp`, shared by both backends through `src/backend.h`. Batches
live only in device memory, in a ring of two buffer sets, each with its
own stream (a CUDA stream, or a Vulkan queue with a timeline semaphore).
File data moves through twelve fixed 8MB pinned staging buffers instead
of pinned batch-sized ones: the main thread `fread`s into an input stage
and copies it up asynchronously, and writer threads drain output stages
that the main thread fills with asynchronous downloads. So batch *i+1*'s
read and upload overlap batch *i*'s kernels, and batch *i−1*'s download
and file write overlap both.

Every output stage's place in the file is known when it is queued (a
decompressed chunk *c* starts at *c* × chunk size; compressed batches
follow each other in the payload), so into a regular file two writer
threads write stages at their offsets in any order (`pwrite`, or
`WriteFile` with an offset on Windows). One thread copying out of
cache-cold staging memory was the limit on decompression. A pipe or a
device gets one thread writing in order.

Two measurements drove that design:

- **Pinned memory is expensive to create under WSL2.** `cudaHostAlloc`
  measured ~0.3–0.4s per GB, plus ~0.1s per GB to free at exit, against
  ~3ms per GB for `cudaMalloc`. The old ring pinned three sets of
  batch-sized buffers, about 1.5GB for a big `ratio` batch.
- **Batches on different streams barely overlap on the GPU.** The next
  batch is usually still being read while one runs, so the chunks in
  flight are roughly one batch, and a bigger batch beats more sets.

Batch size comes from free VRAM, since the GPU may be shared: at least
1024 chunks and 32MB of input, within 80% of the free VRAM (or an
explicit `--gpu-mem` budget). That memory is allocated once, at startup,
and held until gpusqz exits, so another process can't take it mid-run. A
file that fits in one or two batches gets the whole budget. Allocation
retries with a halved batch on failure. Compressed output is compacted
on the GPU (a prefix sum plus a pack kernel, writing into memory the
batch no longer needs), so the download moves only compressed bytes.

## Container format

`src/format.h`:

```
FileHeader    { magic="GSQZ", version=1, chunk_size, original_size, chunk_count,
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

The format has no backward compatibility while gpusqz is at 0.x: a change
bumps `version` and older files stop decoding (see
[Releases and versioning](releasing.md)). `tests/ref_decode.cpp`, the CPU
reference decoder, implements this format independently of the GPU code
(see [Testing](testing.md)).
