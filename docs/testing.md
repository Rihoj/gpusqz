# Testing

```
ctest --test-dir build                   # everything below, per backend built
bash tests/round_trip.sh                 # default chunk size, literal-context and --profile cases
bash tests/round_trip.sh --extremes      # chunk sizes 1, 16, 4K, 8K, 32K, 65535, 65536,
                                          # 1048575, 1048576 (kMaxChunkSize), mismatched
                                          # batch sizes, and a match-free 1MB chunk
BIG=1 bash tests/round_trip.sh           # + 300MB random and 300MB text (multi-batch)
```

## The CPU reference decoder

Every case is round-tripped on the GPU *and* decoded by
`tests/ref_decode.cpp` (`gpusqz_refdec`), a CPU decoder that shares no
code with the GPU path apart from the alphabet and table definitions in
`rans_codes.h`, so a symmetric bug in the GPU encoder and decoder can't
hide. That matters here because `compute-sanitizer` in CUDA 12.8 does not
support this GPU, so memcheck was not available during development.

## Backends and fixtures

`round_trip.sh` uses whatever backend `--backend`'s default picks; set
`GPUSQZ_BACKEND=vulkan` to test the Vulkan one. `ctest` runs the suite
once per backend built (labels `gpu` for CUDA and `vulkan`), plus
committed fixtures in `tests/fixtures` decoded by the CPU decoder and by
each backend: small files compressed by the GPU build that cover raw,
token and rANS chunks, all three literal-context rules and several table
groups, plus corrupt files that must be rejected. After any change to the
file format, regenerate them on a GPU machine with
`tests/fixtures/make_fixtures.sh` and commit the result.

`round_trip.sh` runs on macOS's bash 3.2 and BSD tools, so keep GNU-only
constructs (`stat -c`, `base64 -w`, `sed -i`, `mapfile`, …) out of it.

## Vulkan without a GPU

Mesa's lavapipe is a Vulkan driver that runs on the CPU, and its subgroup
size follows its vector width, so it can stand in for each kind of GPU:

```
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json GPUSQZ_BACKEND=vulkan
LP_NATIVE_VECTOR_WIDTH=1024 bash tests/round_trip.sh   # subgroup 32 (NVIDIA, Apple)
LP_NATIVE_VECTOR_WIDTH=256  bash tests/round_trip.sh   # subgroup 8: shared-memory lanes
```

(2048 gives subgroup size 64, where lavapipe fails the lane probe and
falls back to shared-memory lanes.) The Linux CI job runs the `vulkan`
tests this way, so the Vulkan backend is exercised on every push.
GitHub's runners have no GPU, so the `gpu` label (CUDA) is skipped there.

## Notable cases

- **Literal contexts.** Literal-heavy base64 and source-text inputs run
  at every forced literal-context rule, since the automatic choice would
  pick order-0 for most small test files.
- **Mismatched batches.** `GPUSQZ_FORCE_BATCH` exists because a
  `TableGroup`'s boundaries are fixed at compress time, but decompression
  picks its own batch size independently. The extremes suite compresses
  and decompresses with deliberately different batch sizes, in both
  directions, including with 256-context tables.
- **Same output at any memory budget.** A `TableGroup` covers a fixed
  amount of input (`kGroupBytes`), so the batch size a memory budget
  allows must not change the bytes written. The profile suite compresses
  one file at several `--gpu-mem` values and compares them, with
  `GPUSQZ_FORCE_GROUP_CHUNKS` shrinking groups so a small test file still
  spans several.
- **Checksums.** `text_badchunkhash.gszbad` flips one payload byte and
  leaves the checksum alone: the chunk would still decode, so only the
  checksum can reject it. `text_badtable.gszbad` does the same for a
  group's coded counts. `text_badseq.gszbad` deliberately *recomputes* the
  checksum after its edit, so it still exercises the rANS decoder's own
  rejection rather than stopping at the checksum.
- **A match-free 1MB chunk.** The extremes suite includes a 2^20-byte De
  Bruijn sequence B(32,4), in which no 4-byte string repeats. It once
  produced undecodable output, because a single literal run of exactly
  2^20 is one past what the rANS length alphabet can code; such a chunk
  now falls back to plain tokens.

Lavapipe reports a subgroup size but not a usable subgroup-lane path: at
every `LP_NATIVE_VECTOR_WIDTH` it takes the shared-memory lane build. So
the subgroup-lane build — what NVIDIA and Apple actually run — is not
covered locally, and needs a real GPU (or the Windows cross-build in the
`/vulkan-check` skill).

## CI

The `build` workflow (`.github/workflows/build.yml`) builds and tests the
deb, rpm, Windows and macOS packages on every push and pull request, and
smoke-tests the installers. See [Releases and versioning](releasing.md)
for what happens on `main`.
