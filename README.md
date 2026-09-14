# gpusqz — a GPU file compressor

`gpusqz` (pronounced "GPU squeeze") compresses and decompresses files on
the GPU; compressed files use the `.gsz` extension. It runs on NVIDIA GPUs
with CUDA and on AMD, Apple silicon and Intel GPUs with Vulkan (see *GPU
backends*); both produce the same files. Each chunk of the input (64KB by
default, up to 1MB) is handled by one group of 32 lanes (a CUDA warp): an
LZ parse where all 32 lanes search for matches together, followed by a
32-way interleaved rANS entropy coder with order-1 literal contexts. It's
a from-scratch, educational implementation — not a drop-in replacement for
zstd — but at its default setting it compresses 1.4–2.5x faster than
single-threaded `zstd -1` at about the same ratio, and its `ratio`
profile lands within 0.5% of `zstd -3`'s output size while compressing
1.3–2.2x faster (see *Results*; zstd still decompresses faster).

## Results

CUDA backend on an RTX 5060 Ti (16GB) under WSL2, CPU tools
single-threaded, best of 3–5 runs
per column (`REPEAT=<n>`, see *Benchmarking*). Wall figures are
whole-process (file I/O, PCIe copies, and for gpusqz ~0.2s of CUDA context
creation and allocation); the kernel columns are the wall-clock time any
gpusqz kernel was running. Ratio is output/input, so **lower is better**.

**Measurement conditions.** The GPU was otherwise idle (~1GB used by the
desktop, 0–2% utilisation), and gpusqz's batch budget was 4GB (the default
at the time; it is now 80% of free GPU memory, see *Usage*). On
the 283MB corpus about half of gpusqz's wall time is the ~0.2s fixed startup
cost, which varies by ±15% between runs, so its wall figures there move
by that much from run to run; the kernel figures and the 1GB corpus are
the steadier comparison. An earlier set of runs with another process
holding 11.5GB of VRAM gave kernel figures and ratios within 1% of these.

Varied 1GB corpus (every distinct `/usr/include` header, concatenated
without repetition; see *Benchmarking*):

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
friendlier shape for a match finder, see the caveat in *Benchmarking*):

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
`fwrite` (see *Known limitations*), and the kernels run 1.6–4x faster
than the wall-clock rate.

### enwik8 on an Apple M1 Max and the RTX 5060 Ti

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

Both machines wrote identical `.gsz` files (same sizes to the byte), as
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
  runs at 55–70% of CUDA's speed (see *GPU backends*).
- **The M1 Max's kernel figures are approximate.** The build measured
  asked for a timestamp query pool larger than Metal's 4096 samples, so
  MoltenVK emulated the timestamps (it logged a
  `VK_ERROR_OUT_OF_DEVICE_MEMORY` line about `MTLCounterSampleBuffer`).
  Later builds cap the pool at 4096. Setup and wall times are exact.
- **The `ratio` profile needs larger inputs.** 100MB is only 96 of its
  1MB chunks, far too few to fill either GPU.
- **Against zstd:** `balance` on the M1 Max beats `zstd -1` on both
  speed and ratio. `ratio` comes within 1.4% of `zstd -3`'s output size;
  zstd still decompresses faster on both machines.

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

## GPU backends

gpusqz has two backends behind one interface (`src/backend.h`); the host
pipeline, the file format and every file they write are shared.

| backend | GPUs | kernels | chosen when |
|---|---|---|---|
| CUDA | NVIDIA, compute capability 7.0+ | `src/kernels.cu`, `lz_warp.cuh`, `rans.cuh` | an NVIDIA GPU and driver are present |
| Vulkan 1.2 | AMD (e.g. RX 580), Apple M1–M4 through MoltenVK, Intel, NVIDIA | `src/vk/*.comp` (GLSL, compiled to SPIR-V at build time) | otherwise |

`--backend auto|cuda|vulkan` (or `GPUSQZ_BACKEND`) overrides the choice,
and `gpusqz devices` lists what each backend finds, including which lane
mode (below) a Vulkan device ends up in. The two backends' output is
byte-identical: that was checked on lavapipe at subgroup sizes 8, 16, 32
and 64 and on the RTX 5060 Ti through NVIDIA's Vulkan driver, and the
tests decode each backend's files with the other and with the CPU
reference decoder.

**Lanes on non-NVIDIA hardware.** The format depends on exactly 32 lanes
per chunk (32 interleaved rANS states, 32 literal runs), but GPUs group
threads differently: 32 on NVIDIA and Apple, 64 on AMD GCN cards such as
the RX 580, 32 or 64 on newer AMD cards, 8–32 on Intel. The Vulkan
shaders run each chunk as a workgroup of 32 and go through a small
lane-group layer (`src/vk/common.glsl`) with two builds:

- *Subgroup lanes*, the fast one: ballots and shuffles are single
  subgroup operations. Used where a 32-invocation workgroup is one
  subgroup (size 32, or size 32 requested through subgroup size control)
  or half of one (size 64, half the lanes idle).
- *Shared-memory lanes*: ballots and shuffles go through shared memory and
  workgroup barriers. Correct on any subgroup size, several times slower.

At startup the backend runs a probe shader on the subgroup build and
checks every lane's ballot, shuffle and sum; if anything is off, it falls
back to shared-memory lanes rather than risk wrong output. (Mesa's lavapipe
fails the probe at subgroup size 64, for example.) `GPUSQZ_VK_LANES=shared`
forces the fallback.

CUDA's `__match_any_sync`, which picks one lane per hash bucket to
insert, has no Vulkan equivalent. Instead every lane rewrites its bucket's
older ways (lanes sharing a bucket write identical values), and way 0
takes the lowest lane's position through `atomicMin`. That leaves the
table exactly as CUDA does, which is why the output matches byte for
byte.

**Vulkan speed.** On the RTX 5060 Ti (Windows NVIDIA Vulkan driver) the
Vulkan compress kernels ran at 90–96% of CUDA's throughput and the
decompress kernel at 55–70% (283MB corpus; e.g. `speed` profile 2013 vs
2227 MB/s compress, 1874 vs 3323 MB/s decompress). Removing `coherent`
from the scratch buffers and dropping a redundant memory barrier helped;
the remaining decode gap is not understood yet. On an Apple M1 Max the
Vulkan backend runs through MoltenVK in subgroup-lane mode and passes the
round-trip suite; its speed is in *Results* (enwik8). The RX 580 has not
been measured yet.

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

### Host pipeline (`src/main.cpp`)

Batches live only in device memory, in a ring of two buffer sets, each
with its own stream (a CUDA stream, or a Vulkan queue with a timeline
semaphore). File data moves through twelve fixed 8MB
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
1024 chunks and 32MB of input, within 80% of the free VRAM (or an
explicit `--gpu-mem` budget). That memory is allocated once, at startup,
and held until gpusqz exits, so another process can't take it mid-run. A
file that fits in one or two batches gets the whole budget. Allocation
retries with a halved batch on failure. Compressed output is compacted
on the GPU (a prefix sum plus a pack kernel, writing into memory the
batch no longer needs), so the download moves only compressed bytes.

### Container format (`src/format.h`)

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

## Installing

Pre-built packages are attached to each [GitHub
release](https://github.com/Rihoj/gpusqz/releases) (see *Releases and
versioning*). Every other build of the `build` workflow
(`.github/workflows/build.yml`) leaves them as workflow artifacts too.

| platform | installer | portable archive | backends |
|---|---|---|---|
| Ubuntu 22.04+, Debian 12+ | `gpusqz_<version>_amd64.deb` | – | CUDA, Vulkan |
| RHEL/Rocky/Alma 8+, Fedora | `gpusqz-<version>-1.x86_64.rpm` | – | CUDA, Vulkan |
| Windows 10/11 x64 | `gpusqz-<version>-win64.msi` | `gpusqz-<version>-win64.zip` | CUDA, Vulkan |
| macOS 11+ (Apple silicon and Intel) | `gpusqz-<version>-Darwin.pkg` | `gpusqz-<version>-Darwin.tar.gz` | Vulkan (MoltenVK) |

Every package has `gpusqz` and `gpusqz_refdec`; the Windows ones add the
MSVC runtime DLLs and the macOS ones `lib/libMoltenVK.dylib`.

- **Windows `.msi`** installs to `C:\Program Files\gpusqz\bin` and adds
  that to the system PATH (open a new terminal afterwards). Uninstall it
  from Settings → Apps. The installer is not code-signed yet, so
  SmartScreen asks first: *More info* → *Run anyway*. Silent install:
  `msiexec /i gpusqz-<version>-win64.msi /qn`.
- **macOS `.pkg`** installs to `/usr/local/gpusqz` and links `gpusqz` and
  `gpusqz_refdec` into `/usr/local/bin`. It is not signed or notarized
  yet, so Gatekeeper refuses a double-click: allow it under System
  Settings → Privacy & Security → *Open Anyway*, or install from a
  terminal with `sudo installer -pkg gpusqz-<version>-Darwin.pkg -target /`.
  To uninstall: `sudo rm -rf /usr/local/gpusqz /usr/local/bin/gpusqz
  /usr/local/bin/gpusqz_refdec && sudo pkgutil --forget io.github.rihoj.gpusqz`.
- **Archives** (`.zip`, `.tar.gz`) need no installation: unpack and run
  from `bin/`. Keep `bin/` and `lib/` together on macOS, and clear the
  download quarantine there first: `xattr -dr com.apple.quarantine <unpacked dir>`.

What each GPU needs at run time:

- **NVIDIA**: compute capability 7.0 (Volta) or newer and a driver that
  supports CUDA 12. The CUDA runtime is linked in, so no toolkit is
  needed. The packages carry native code for Volta through Blackwell plus
  PTX that newer GPUs compile at load time.
- **AMD** (e.g. Radeon RX 580): the driver's Vulkan support. On Linux
  that is Mesa's RADV (`mesa-vulkan-drivers` on Debian/Ubuntu,
  `mesa-vulkan-drivers` on Fedora/RHEL) with the Vulkan loader
  (`libvulkan1` / `vulkan-loader`). On Windows it is the AMD Adrenalin
  driver. (ROCm/HIP is not used: it dropped Polaris cards like the RX 580
  and doesn't exist for them on Windows.)
- **Apple silicon** (M1, M4 and later): nothing else to install. The macOS
  packages ship MoltenVK, which runs the Vulkan backend on Metal, in
  `lib/` next to `bin/`. A Homebrew `molten-vk` or the Vulkan SDK also
  work.
- **Intel**: the driver's Vulkan support (Mesa ANV on Linux).

Run `gpusqz devices` to see what gpusqz found. `gpusqz_refdec <in.gsz>
<out>` decompresses on the CPU anywhere, with no GPU at all.

## Building

Requires CMake 3.20+, plus for each backend:

- **CUDA** (`-DGPUSQZ_BUILD_CUDA=ON`, the default except on macOS): CUDA
  12.8+. Targets sm_120 (RTX 5060 Ti / Blackwell) by default; override
  with `-DCMAKE_CUDA_ARCHITECTURES=<arch>`.
  (`CMAKE_CUDA_ARCHITECTURES=native` is not used because under WSL it
  silently fell back to sm_52.) Release builds pass every generation:
  see `CUDA_ARCHS` in the workflow.
- **Vulkan** (`-DGPUSQZ_BUILD_VULKAN=ON`, the default): the Vulkan headers
  and `glslangValidator` — the Vulkan SDK, or distro packages
  (`libvulkan-dev glslang-tools` on Debian/Ubuntu; `brew install
  vulkan-headers glslang` on macOS). The Vulkan library itself is loaded
  at run time, not linked. Point CMake at them with
  `-DGPUSQZ_VULKAN_INCLUDE=<dir>` / `-DGPUSQZ_GLSLANG=<path>` if they are
  somewhere unusual.

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

With both backends off only `gpusqz_refdec` is built. On macOS,
`-DGPUSQZ_MOLTENVK_DYLIB=<libMoltenVK.dylib>` (and `_LICENSE`) makes the
package ship MoltenVK. `cpack -G DEB`, `RPM`, `ZIP` or `TGZ` in the build
directory produces the packages above.

`GPUSQZ_MIN_BLOCKS_PER_SM=<n>` (a CMake cache var, not a runtime flag) sets
`__launch_bounds__`'s `minBlocksPerSM` hint on the per-chunk kernels, for
A/B occupancy testing — see the comment above it in `CMakeLists.txt`. None
of those kernels use shared memory, and at 40 (parse), 62 (encode) and 58
(decode) registers per thread with no spills (`nvcc -Xptxas -v`), their
one-warp blocks are limited by the per-SM block count rather than by
registers, so this knob is mainly a regression check against future
register spilling.

## Usage

```
./build/gpusqz c <input> <output> [chunk_size]                # compress (default chunk_size 65536)
./build/gpusqz c <input> <output> --profile speed|balance|ratio  # ...or pick a chunk-size preset
./build/gpusqz d <input> <output>                              # decompress
./build/gpusqz c|d ... --gpu-mem 8G                            # GPU memory for batch buffers
./build/gpusqz c|d ... --backend vulkan                        # pick the backend (auto, cuda, vulkan)
./build/gpusqz devices                                         # list GPUs per backend
```

`chunk_size` and `--profile` are mutually exclusive, and unrecognised
arguments are rejected. See *Results* for what each profile costs and
buys.

`--gpu-mem SIZE` (or `GPUSQZ_GPU_MEM`) sets how much GPU memory gpusqz may use
for its batch buffers: a number with a K, M, G or T suffix, or a bare
number of MiB. By default gpusqz takes 80% of the GPU memory free when it
starts; an explicit value may use all but 256MiB of the free
memory and is reduced, with a note, if it asks for more. On GPUs that
share system RAM (Apple silicon, integrated GPUs) "free GPU memory" is
counted as at most half of that RAM, so the default stays at 40% of it. Host RAM use
does not depend on it: gpusqz pins a fixed ~96MB of staging buffers.

Environment variables, all optional:

| variable | effect |
|---|---|
| `GPUSQZ_VERBOSE=1` | Per-stage timing on stderr: setup, fread, copies, kernel time (summed and wall-clock union), fwrite, staging stalls. |
| `GPUSQZ_FORCE_BATCH=<n>` | Force chunks per batch (testing; see *Testing*). |
| `GPUSQZ_FORCE_LIT_SHIFT=<0\|4\|8>` | Force every batch's literal-context rule (testing and tuning). |
| `GPUSQZ_BACKEND=auto\|cuda\|vulkan` | Same as `--backend`. |
| `GPUSQZ_VK_DEVICE=<n>` | Use Vulkan device *n* from `gpusqz devices` (default: the first discrete GPU, then integrated, then others). |
| `GPUSQZ_VK_LANES=shared` | Force shared-memory lanes (testing). |
| `GPUSQZ_VULKAN_LIB=<path>` | Load this Vulkan library instead of the system loader (or bundled MoltenVK). |
| `GPUSQZ_VK_DEBUG=1` | Print why the subgroup-lane probe failed, if it does. |

## Testing

```
ctest --test-dir build                   # everything below, per backend built
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

`round_trip.sh` uses whatever backend `--backend`'s default picks; set
`GPUSQZ_BACKEND=vulkan` to test the Vulkan one. `ctest` runs the suite
once per backend built (labels `gpu` for CUDA and `vulkan`), plus
committed fixtures in `tests/fixtures` decoded by the CPU decoder and by
each backend: small files compressed by the GPU build that cover raw,
token and rANS chunks, all three literal-context rules and several table
groups, plus corrupt files that must be rejected. After any change to the
file format, regenerate them on a GPU machine with
`tests/fixtures/make_fixtures.sh` and commit the result.

**Vulkan without a GPU.** Mesa's lavapipe is a Vulkan driver that runs on
the CPU, and its subgroup size follows its vector width, so it can stand
in for each kind of GPU:

```
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json GPUSQZ_BACKEND=vulkan
LP_NATIVE_VECTOR_WIDTH=1024 bash tests/round_trip.sh   # subgroup 32 (NVIDIA, Apple)
LP_NATIVE_VECTOR_WIDTH=256  bash tests/round_trip.sh   # subgroup 8: shared-memory lanes
```

(2048 gives subgroup size 64, where lavapipe fails the lane probe and
falls back to shared-memory lanes.) The Linux CI job runs the `vulkan`
tests this way, so the Vulkan backend is exercised on every push.
GitHub's runners have no GPU, so the `gpu` label (CUDA) is skipped there.

Notable cases:

- **Literal contexts.** Literal-heavy base64 and source-text inputs run
  at every forced literal-context rule, since the automatic choice would
  pick order-0 for most small test files.
- **Mismatched batches.** `GPUSQZ_FORCE_BATCH` exists because a
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
REPEAT=5 bash bench/run_bench.sh corpus_283mb.txt                     # gpusqz default vs gzip/zstd
CPU=0 REPEAT=5 bash bench/run_bench.sh corpus_283mb.txt --profile ratio  # one gpusqz profile only
```

The script reports wall and kernel MB/s for gpusqz and compares against
`gzip -1/-6` and single-threaded `zstd -1/-3` when available. On a shared
GPU a single wall-clock measurement can be dominated by another process,
so `REPEAT=<n>` runs each codec n times and reports the best run per
column. Check `nvidia-smi` first: another process's memory shrinks gpusqz's
batches, and its compute slows gpusqz's global-memory-latency-bound parse.

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

## Releases and versioning

gpusqz follows [semantic versioning](https://semver.org) and is released
automatically by [semantic-release](https://semantic-release.gitbook.io)
from the commit messages on `main`, which follow [Conventional
Commits](https://www.conventionalcommits.org):

| commit | example | release |
|---|---|---|
| `fix:` / `perf:` | `fix(vulkan): retry refused allocations` | patch (0.1.0 → 0.1.1) |
| `feat:` | `feat: add --level` | minor (0.1.0 → 0.2.0) |
| breaking: `!` after the type, or a `BREAKING CHANGE:` footer | `feat!: format v2` | minor while on 0.x; major from 1.0 on |
| anything else (`docs:`, `test:`, `ci:`, `chore:`, `refactor:`, …) | `docs: fix typo` | none |

While the version is 0.x, the `.gsz` format and the command line may
still change between minor versions. Going to 1.0.0 is a deliberate step:
remove the `"breaking": true → minor` rule from `.releaserc.json`, then
merge a breaking change.

On every push to `main` the `build` workflow asks semantic-release for the
next version (dry run), builds and tests every package with it, and only
when all of them pass tags the commit `vX.Y.Z` and publishes the GitHub
release with generated notes and the packages attached. A push with
nothing releasable just builds. Pull request titles are checked against
Conventional Commits, because a squash merge turns the title into the
commit message; use squash merges (or write every commit that way).

`gpusqz --version` and `gpusqz_refdec --version` print the version. A
release build prints `X.Y.Z`; any other build prints `git describe`'s
view, e.g. `0.1.0-3-gabc1234` (3 commits after v0.1.0) with `-dirty` for
uncommitted changes, fixed when CMake configures. Configure with
`-DGPUSQZ_VERSION=X.Y.Z` to set it explicitly. The `v0.0.0` tag is not a
release: it marks where the commit history starts to count.

The release tooling is pinned in `release/package.json` (with its
lockfile); `release/next-version.mjs` is the dry run CI uses.

## Known limitations and next steps

- **Vulkan is untested on AMD hardware so far.** It has been verified on
  lavapipe, on an NVIDIA GPU and on an Apple M1 Max (round-trip suite
  passes, subgroup-lane mode; the M4 machines are expected to behave the
  same but have not been run). The first runs on the RX 580 should be
  `gpusqz devices` (which reports the lane mode after the probe) and
  `GPUSQZ_BACKEND=vulkan bash tests/round_trip.sh`. Known risk: GCN
  cards like the RX 580 run 32 of each 64-lane wavefront (half idle; two
  chunks per wavefront would use it fully, but barrier rules make that a
  bigger change).
- **Vulkan decodes slower than CUDA** on NVIDIA, 55–70% of the kernel
  throughput (see *GPU backends*). Compression is within 10%.
- **Fixed startup cost.** CUDA context creation and allocation take
  ~0.2s on this WSL2 machine, over half of the wall time on the 283MB
  corpus. It amortises on larger inputs and is mostly outside gpusqz's
  control.
- **Decompression is `fwrite`-bound under WSL2.** Writing 1GB measured
  0.5–1.2s depending on page-cache state, while the decompress kernel
  needs ~0.25s, so the writer thread is almost always the bottleneck.
  gpusqz would need a faster filesystem path to go further, not a faster
  kernel.
- **Ratio vs zstd.** zstd's parser is more sophisticated than gpusqz's hash
  match finder with a two-step lazy lookahead: `zstd -3`'s output is ~0.5%
  smaller than gpusqz's `ratio` profile on the varied corpus. An optimal parser, or
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
- **A bigger GPU-memory budget doesn't make `ratio` faster.** Each 1MB
  chunk needs ~6MB of device memory. With the old 4GB default, a 1GB file
  ran as three batches of ~350 chunks; the current 80%-of-free default
  fits it in one batch on an idle 16GB GPU. Both measured the same kernel
  throughput (~1550 MB/s, the GPU is saturated) and wall throughput within
  run-to-run noise (851 vs 869 MB/s compress). Going much lower does cost:
  a 2G budget measured 900 MB/s of kernel throughput.
- **No exclusive GPU access.** gpusqz holds its batch memory for the whole
  run, but it can't stop other processes from using the rest of the GPU's
  memory or its compute; exclusive use needs the system-wide compute mode
  (`nvidia-smi -c EXCLUSIVE_PROCESS`, administrator rights, not available
  under WSL2).
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
