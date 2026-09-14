---
name: format-change
description: Checklist for changing the .gsz file format or anything that changes gpusqz's compressed output - format.h, the CPU reference decoder, both GPU backends, regenerated fixtures, tests, and the right Conventional Commit. Use before editing src/format.h, rans_codes.h, the token/sequence encoding, or table layouts.
---

# Changing the .gsz format

Three decoders must agree on every byte: CUDA, Vulkan, and the independent
CPU decoder `tests/ref_decode.cpp` (the oracle; compute-sanitizer can't
check sm_120 kernels). No backward compatibility is kept while on 0.x, so
old files simply stop decoding: bump the version, don't add compat paths.

## Checklist

1. **Spec first**: `src/format.h` (header, `ChunkEntry`, `TableGroup`,
   `kVersion`) and `src/rans_codes.h`. Bump `kVersion` for any change a
   previous decoder would misread. Update `docs/design.md` (*Container format*,
   *rANS stage*) in the same change.
2. **CPU decoder**: `tests/ref_decode.cpp` in lockstep. It must reject
   malformed input (the `*.gszbad` fixtures) rather than crash.
3. **CUDA**: encoder and decoder in `src/kernels.cu`, `src/lz_warp.cuh`,
   `src/rans.cuh`; host side in `src/main.cpp` / `src/cuda_backend.cu`.
4. **Vulkan**: the same logic in `src/vk/*.comp` (`common.glsl` for shared
   helpers, `compress.glsl`/`decompress.glsl` for bindings) and buffer sizes
   in `src/vk_backend.cpp`. Any new per-chunk scratch must be sized in both
   backends (`compress_bytes_per_chunk`, `decompress_bytes_per_chunk`).
5. **Fixtures**: `bash tests/fixtures/make_fixtures.sh ./build/gpusqz`
   regenerates every `.gsz`/`.gszbad` from the committed `.orig` inputs
   (never regenerate the `.orig` files). Check the corrupted fixtures still
   fail for the reason their name says.
6. **Tests**:
   ```
   cmake --build build -j && ctest --test-dir build
   bash tests/round_trip.sh ./build/gpusqz
   bash tests/round_trip.sh --extremes ./build/gpusqz
   BIG=1 bash tests/round_trip.sh ./build/gpusqz
   ```
   then the `/vulkan-check` skill: CUDA and Vulkan output must stay
   byte-identical and cross-decode.
7. **Measure** with the `/benchmark` skill: format changes usually move the
   ratio; report both corpora and all three profiles.

## Commit

A format change is breaking. Use `feat!: ...` or a `BREAKING CHANGE:`
footer that says what no longer decodes, e.g.

```
feat!: 14-bit rANS probabilities

BREAKING CHANGE: .gsz format version 2; files from earlier versions no
longer decode.
```

On 0.x this releases as a minor version (0.1 → 0.2); see CLAUDE.md.
