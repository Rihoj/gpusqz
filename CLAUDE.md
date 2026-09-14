# gpusqz — notes for Claude

GPU file compressor (`.gsz`): warp-per-chunk LZ parse + 32-way interleaved
rANS, with a CUDA backend (NVIDIA) and a Vulkan backend (AMD, Apple via
MoltenVK, Intel). The docs are in `docs/` (index: `docs/README.md`);
`docs/design.md` is the design document: read the relevant section before
changing an area.

## Layout
- `src/main.cpp` host pipeline and CLI (backend-agnostic); `src/backend.h` the interface.
- CUDA: `src/cuda_backend.cu`, `src/kernels.cu`, `src/lz_warp.cuh`, `src/rans.cuh`.
- Vulkan: `src/vk_backend.cpp` + GLSL in `src/vk/` (`common.glsl` is the lane-group layer;
  shaders are compiled to SPIR-V at build time and embedded).
- Shared format/coding: `src/format.h`, `src/rans_codes.h`, `src/file_io.h`.
- `tests/ref_decode.cpp` (`gpusqz_refdec`): independent CPU decoder, the oracle.
  `tests/round_trip.sh`, `tests/fixtures/` (committed `.gsz` files + `make_fixtures.sh`).
- `packaging/`, `.github/workflows/build.yml`, `.releaserc.json`, `release/`: packaging and releases.
- `docs/`: user and developer docs; `docs/performance-history.md` logs every measured
  speed/ratio change. README.md is the short front page.

## Build and test
```
cmake -S . -B build && cmake --build build -j
ctest --test-dir build                    # labels: gpu (CUDA), vulkan
bash tests/round_trip.sh ./build/gpusqz   # also: --extremes, BIG=1, GPUSQZ_BACKEND=vulkan
```
No AMD/Apple hardware here: test Vulkan on lavapipe (`/vulkan-check` skill).
Configure options: `-DGPUSQZ_BUILD_CUDA=OFF`, `-DGPUSQZ_BUILD_VULKAN=OFF`,
`-DGPUSQZ_GLSLANG=<glslangValidator>`; `GPUSQZ_VERBOSE=1` prints per-stage timing.

## Invariants — check these on every change
- **CUDA and Vulkan write byte-identical files** and each decodes the other's.
  Any kernel change must land in both backends (`.cu` and `src/vk/*.comp`).
- **32 lanes per chunk** is part of the format (32 rANS states, 32 literal runs).
- **Cross-lane writes need a sync first** (`__syncwarp()` / `lg_sync()`). NVIDIA's
  lockstep hid a real race in the encoder fallbacks that lavapipe exposed.
- **Format changes** (`src/format.h`) need `tests/ref_decode.cpp` updated in lockstep and
  the fixtures regenerated: follow the `/format-change` skill. No backward compatibility
  is kept while on 0.x.
- `tests/round_trip.sh` must run on macOS's bash 3.2 with BSD tools: no `stat -c`,
  `base64 -w`, `truncate`, `sed -i`, `mapfile`, `${x,,}`, or empty `"${arr[@]}"` under `set -u`.

## Commits and releases
- **Conventional Commits drive releases** (semantic-release on push to `main`):
  `fix:`/`perf:` → patch, `feat:` → minor, `!`/`BREAKING CHANGE:` → minor while on 0.x.
  `docs:`, `test:`, `ci:`, `chore:`, `refactor:` release nothing. PR titles are checked.
- Never hand-edit versions or create/push `v*` tags: CI tags releases. The version comes
  from `-DGPUSQZ_VERSION` (CI) or `git describe`. `v0.0.0` is a baseline, not a release.
- Work on a branch and open a PR; `/ship` covers PR → CI → merge → verified release.
- Remotes: `github` (releases, CI) and `origin` (self-hosted Gitea mirror; push needs the
  user's SSH agent unlocked).

## Benchmarks
- The GPU is shared: an `ollama` model often holds 11–15GB of VRAM. **Never stop or kill
  ollama** (a hook blocks it); the user stops it for clean numbers. State VRAM/util
  conditions with every number. Follow the `/benchmark` skill.
- Compare kernel MB/s (`kbusy`) as well as wall: WSL2 adds ~0.2s of CUDA setup per run.
- Measure on a varied corpus too; the repetitive 283MB one once hid a regression.
- Record every measured speed/ratio change in `docs/performance-history.md` (and
  update `docs/benchmarks.md` + the README summary when headline numbers move).
  Check its "Measured and rejected" table before re-trying an idea.

## Environment traps
- WSL2: CUDA `native` arch detection is broken (default pinned to sm_120);
  `compute-sanitizer` doesn't support sm_120 (use `gpusqz_refdec` as the oracle).
- CI: PowerShell splits unquoted `-DVAR=0.5.0` at the dot (quote `-D` args); the macOS
  runner's paravirtual GPU can't run gpusqz (Apple testing happens on real Macs);
  Windows checkouts would CRLF text fixtures (`.gitattributes` marks them binary).
- CPack productbuild (.pkg) needs the explicit `CPACK_COMPONENTS_ALL` component, or the
  product is empty and `installer` crashes.
