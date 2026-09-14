---
name: vulkan-check
description: Verify gpusqz's Vulkan backend without AMD or Apple hardware - lavapipe at every subgroup size, byte-identical output vs CUDA, cross-decoding with CUDA and the CPU reference decoder, optional validation layers. Use after any change to src/vk/*, src/vk_backend.cpp, the kernels, or anything both backends share.
---

# Vulkan check

The Vulkan backend must write exactly the same `.gsz` bytes as CUDA, on
every GPU's subgroup size. AMD (RX 580, subgroup 64) and Apple (32) can't be
tested here, so lavapipe stands in: its LLVM vector width sets the subgroup
size (256/512/1024/2048 → 8/16/32/64). Width 2048 fails the lane probe and
exercises the shared-memory lane build; 1024 exercises the subgroup build.

## Run it

```
cmake --build build -j
bash .claude/skills/vulkan-check/vk-check.sh            # all widths, 3 chunk sizes
WIDTHS=1024 CHUNKS=65536 bash .claude/skills/vulkan-check/vk-check.sh   # quick
```

It builds a text and a random input, then per width and chunk size checks:
Vulkan round trip, `gpusqz_refdec` decoding Vulkan's file, and (if a CUDA
GPU is usable) identical bytes to CUDA plus decoding each other's files.
It ends with `ALL VULKAN CHECKS PASSED` or says what differed.

Then run the full suites on lavapipe for the edge cases:

```
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
for w in 1024 2048; do LP_NATIVE_VECTOR_WIDTH=$w GPUSQZ_BACKEND=vulkan bash tests/round_trip.sh ./build/gpusqz; done
LP_NATIVE_VECTOR_WIDTH=1024 GPUSQZ_BACKEND=vulkan bash tests/round_trip.sh --extremes ./build/gpusqz
```

## Validation layers

With the Vulkan SDK installed (`VULKAN_SDK` pointing at its `x86_64` dir),
`VALIDATE=1` runs everything under the Khronos validation layer with
synchronization validation and fails on any `VUID`/`SYNC-HAZARD` message.
Use it for changes to barriers, streams, events, or buffer reuse in
`src/vk_backend.cpp`. It is slow; narrow it with `WIDTHS`/`CHUNKS`.

## Reading failures

- **Different from CUDA but round-trips fine**: a real divergence. Both
  backends must make identical choices; compare the kernel logic line by
  line (`src/kernels.cu`/`lz_warp.cuh`/`rans.cuh` vs `src/vk/*.comp`).
- **Fails only at some widths**: lane-group layer (`src/vk/common.glsl`) or a
  missing `lg_sync()` before cross-lane writes. Width 256 has 4 subgroups per
  32-lane group; 2048 uses shared-memory lanes.
- **Fails only on random data at 8192**: encoder fallback paths (raw/token
  chunks); check the syncs before each fallback in `rans_encode.comp`.
- `GPUSQZ_VK_DEBUG=1` prints memory types, refused allocations and where the
  lane probe stopped; `GPUSQZ_VK_LANES=shared` forces the shared-memory build.

## Real GPUs

- NVIDIA through Vulkan: `GPUSQZ_BACKEND=vulkan` on a machine whose Vulkan
  loader sees the GPU. Under WSL2 the Linux loader doesn't, but a Windows
  build of gpusqz.exe (llvm-mingw cross-compile, `-DGPUSQZ_BUILD_CUDA=OFF`)
  run from WSL uses the Windows NVIDIA driver.
- AMD and Apple: ask the user to run `gpusqz devices` and
  `GPUSQZ_BACKEND=vulkan bash tests/round_trip.sh ./build/gpusqz` there.
