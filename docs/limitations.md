# Known limitations and next steps

## Hardware and backends

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
  throughput (see [GPU backends](backends.md#vulkan-speed)). Compression
  is within 10%.
- **No exclusive GPU access.** gpusqz holds its batch memory for the whole
  run, but it can't stop other processes from using the rest of the GPU's
  memory or its compute; exclusive use needs the system-wide compute mode
  (`nvidia-smi -c EXCLUSIVE_PROCESS`, administrator rights, not available
  under WSL2).
- **No multi-GPU.** gpusqz uses one device.
- **NPUs aren't a fit.** Recent CPUs' neural accelerators are dense
  matrix-multiply engines with no data-dependent branching or random
  gathers, which is all LZ matching and rANS consist of, and WSL2 does not
  expose them anyway.

## Speed

- **Fixed startup cost.** CUDA context creation and allocation take
  ~0.2s on this WSL2 machine, over half of the wall time on the 283MB
  corpus. It amortises on larger inputs and is mostly outside gpusqz's
  control. A native Metal device (M1 Max) starts in well under 0.1s.
- **Decompression is bound by writing the file under WSL2.** Decompressing
  1GB to `/dev/null` takes ~0.35s, and to a file 0.6–0.8s, even with two
  writer threads (more measured no faster), while the decompress kernel
  needs ~0.25s. Going further needs a faster filesystem path, not a
  faster kernel.
- **`compute_repeat_codes`' encode-side pass is serial**, one lane per
  chunk. Removing it entirely would save only 1–6% of compress-kernel
  time, too little to parallelise, while its repeat codes make binaries
  3.5–4.2% smaller (see [Measured and
  rejected](performance-history.md#measured-and-rejected)).
- **Small inputs can't fill the GPU at the `ratio` profile.** Each 1MB
  chunk is one warp; a 100MB file is only ~96 of them, while ~350 are
  needed to saturate an RTX 5060 Ti.

## Ratio

- **Ratio vs zstd.** The `ratio` profile now edges `zstd -3` on both
  benchmark corpora (0.1% and 0.6% smaller), but zstd's parser is still
  the more sophisticated one, and `zstd -19` is far out of reach. An
  optimal parser, a second longer-key hash table (zstd's dfast, which is
  what levels 3–4 use), or a match finder with longer chains would be the
  next ratio levers. A third lazy step and a 64-byte probe cap were
  measured and did nothing useful (see [Measured and
  rejected](performance-history.md#measured-and-rejected)).
- **Literal-context choice is per batch, estimated from the histogram.**
  It is exact about which chunks can't use rANS, but not about which
  chunks will lose to plain tokens later, so a batch can occasionally
  carry 16 or 256 tables that few of its chunks use. The cost is bounded
  by the coded table bytes, which are about a tenth of the 4KB or 66KB
  the raw counts would take (see [Design](design.md#container-format)).
- **No format-aware transforms.** gpusqz codes every file as a plain byte
  stream. On binary STL meshes, a bit-exact per-chunk transform (storing
  each normal as its difference from one recomputed from the vertices,
  plus a per-chunk table of distinct vertices) made `ratio` output 1.5–3.3x
  smaller than on the untransformed file, and fits inside chunk
  independence. It
  would be format 3. See the [Format-aware transforms
  study](format-aware-transforms-study.md).

## Format and robustness

- **No integrity check.** Structural corruption (bad offsets, sizes,
  truncation, invalid rANS streams) is detected, but the format has no
  checksum, so a corrupted literal or table byte that still decodes
  consistently produces wrong output silently.
- **Expanded decode tables use ~1.3MB of device memory per table group**
  at 256 contexts, all expanded up front. That is ~10MB for a 1GB file at
  the default profile, but would grow to gigabytes for a file of many
  terabytes; expanding each decode batch's groups on demand would fix
  it.
- **No streaming API** — it's a file-in, file-out CLI, and output must be
  seekable (the header is patched at the end).
- Match offsets are 32-bit, but chunks are capped at 1MB by policy
  (`kMaxChunkSize` in `src/format.h`) rather than by the wire format.
- **No format stability yet.** While on 0.x a format change makes older
  `.gsz` files undecodable by newer builds (see
  [Releases and versioning](releasing.md)).

## Packaging

- The Windows `.msi` and macOS `.pkg` are not code-signed or notarized
  (see [Installing](installing.md)).
- No license has been chosen yet (`packaging/license.txt`).
