# Building

Requires CMake 3.20+ and a C++17 compiler, plus for each backend:

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

With both backends off only `gpusqz_refdec` is built. The shaders in
`src/vk/` are compiled to SPIR-V at build time and embedded in the
binary.

## Version

The version printed by `--version` is fixed when CMake configures: from
`-DGPUSQZ_VERSION=X.Y.Z` if given (release builds), otherwise from `git
describe` (e.g. `0.1.0-3-gabc1234`, with `-dirty` for uncommitted
changes). See [Releases and versioning](releasing.md).

## Packages

`cpack -G <generator>` in the build directory produces the packages listed
in [Installing](installing.md): `DEB` and `RPM` on Linux, `ZIP` and `WIX`
(the `.msi`, needs WiX Toolset 3) on Windows, `TGZ` and `productbuild`
(the `.pkg`) on macOS. Packaging files live in `packaging/`. On macOS,
`-DGPUSQZ_MOLTENVK_DYLIB=<libMoltenVK.dylib>` (and `_LICENSE`) makes the
package ship MoltenVK. The `build` workflow
(`.github/workflows/build.yml`) shows the exact commands used for each
release package.

## Occupancy knob

`GPUSQZ_MIN_BLOCKS_PER_SM=<n>` (a CMake cache var, not a runtime flag) sets
`__launch_bounds__`'s `minBlocksPerSM` hint on the per-chunk kernels, for
A/B occupancy testing — see the comment above it in `CMakeLists.txt`. None
of those kernels use shared memory, and at 40 (parse), 62 (encode) and 58
(decode) registers per thread with no spills (`nvcc -Xptxas -v`), their
one-warp blocks are limited by the per-SM block count rather than by
registers, so this knob is mainly a regression check against future
register spilling.
