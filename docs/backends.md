# GPU backends

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
reference decoder. An Apple M1 Max wrote enwik8 files of exactly the same
sizes as the CUDA backend at every profile.

## Lanes on non-NVIDIA hardware

The format depends on exactly 32 lanes per chunk (32 interleaved rANS
states, 32 literal runs), but GPUs group threads differently: 32 on
NVIDIA and Apple, 64 on AMD GCN cards such as the RX 580, 32 or 64 on
newer AMD cards, 8–32 on Intel. The Vulkan shaders run each chunk as a
workgroup of 32 and go through a small lane-group layer
(`src/vk/common.glsl`) with two builds:

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
forces the fallback, and `GPUSQZ_VK_DEBUG=1` prints why the probe failed.

CUDA's `__match_any_sync`, which picks one lane per hash bucket to
insert, has no Vulkan equivalent. Instead every lane rewrites its bucket's
older ways (lanes sharing a bucket write identical values), and way 0
takes the lowest lane's position through `atomicMin`. That leaves the
table exactly as CUDA does, which is why the output matches byte for
byte.

## Memory on Vulkan devices

The backend tries every memory type that suits a buffer, in order of
preference, and moves on to the next when the driver refuses an
allocation, instead of failing on the first refusal.
`GPUSQZ_VK_DEBUG=1` lists the device's memory heaps and types and each
refused allocation. (The paravirtual Metal GPU on GitHub's macOS runners
refuses every host-visible allocation, so gpusqz can't run there at
all.) On GPUs that share
system RAM (Apple silicon, integrated GPUs), "free GPU memory" is counted
as at most half of that RAM (see [Usage](usage.md#gpu-memory)).

## Vulkan speed

On the RTX 5060 Ti (Windows NVIDIA Vulkan driver) the Vulkan compress
kernels ran at 90–96% of CUDA's throughput and the decompress kernel at
55–70% (283MB corpus; e.g. `speed` profile 2013 vs 2227 MB/s compress,
1874 vs 3323 MB/s decompress). Removing `coherent` from the scratch
buffers and dropping a redundant memory barrier helped.

A profiling pass on 2026-09-17 (same GPU, Windows driver, kernel `kbusy`,
best of 3–4) narrowed where the rest of the gap is, without finding a fix:

| what decodes | Vulkan / CUDA |
|---|---|
| raw chunks only (256MB of random data: a pure copy) | **1.20**, i.e. at least as fast — but 5ms against 6ms, near the timer's resolution |
| rANS decode alone (reconstruction stubbed out) | 0.54–0.55 |
| full decode | 0.51 (283MB `speed`), 0.70 (enwik8 `ratio`) |

So it is not dispatch overhead, PCIe or plain copying: those are at least
as fast as CUDA. The cost is spread evenly over the 32-lane cooperative
code — rANS decoding and LZ reconstruction lose about the same share.
Both backends use one 32-lane group per chunk and the same number of
barriers per match.

Ruled out so far (see [Measured and
rejected](performance-history.md#measured-and-rejected)): a lane-group
layer specialised for subgroup size 32, and `lg_sync()` narrowed to
buffer-only memory semantics with `controlBarrier`. A timing probe that
drops the *execution* barrier entirely (leaving only a buffer memory
barrier, which is not safe to ship) does recover 17–18%, so lane
synchronisation looks like part of the gap. What has not been measured
yet: register use and occupancy, through
`VK_KHR_pipeline_executable_properties`.

On an Apple M1 Max the Vulkan backend runs through MoltenVK in
subgroup-lane mode and passes the round-trip suite; its speed is in
[Benchmarks](benchmarks.md#enwik8-on-an-apple-m1-max-and-the-rtx-5060-ti).
The RX 580 has not been measured yet.

## Hardware that has been run

| GPU | backend | lane mode | status |
|---|---|---|---|
| NVIDIA RTX 5060 Ti | CUDA | – | development machine; all suites |
| NVIDIA RTX 5060 Ti | Vulkan (Windows driver) | subgroup | byte-identical to CUDA; benchmarked |
| Apple M1 Max | Vulkan (MoltenVK) | subgroup | round-trip suite passes; benchmarked |
| Mesa lavapipe (CPU), subgroup 8–64 | Vulkan | subgroup or shared | CI and local testing (see [Testing](testing.md)) |
| AMD RX 580, Apple M4 / M4 Max | Vulkan | – | not run yet |

Because the CI runners can't run it, Apple hardware is only tested on
real Macs.
