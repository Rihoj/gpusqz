# gpusqz documentation

Start with the [project README](../README.md) for what gpusqz is and a
summary of its results.

**Using gpusqz**

- [Installing](installing.md): packages, installers and uninstalling,
  and what each GPU needs at run time.
- [Usage](usage.md): the command line, profiles, GPU memory, environment
  variables.
- [Known limitations](limitations.md): what's missing or slow, and what
  might come next.

**How it works**

- [Design](design.md): chunks and lanes, the LZ parse, the rANS stage,
  decoding, the host pipeline and the `.gsz` file format.
- [GPU backends](backends.md): CUDA and Vulkan, lane modes on AMD, Apple
  and Intel GPUs, which hardware has been run.
- [Compression survey](compression-survey.md): how other open-source
  compressors trade speed for ratio; section 13 covers what applies to
  gpusqz.

**Performance**

- [Benchmarks](benchmarks.md): the current results against gzip and
  zstd, and how to measure new ones.
- [Performance history](performance-history.md): every measured change
  since the first commit, the corpora and conditions behind each number,
  experiments that were dropped, and how to add an entry.
- [Predictive modeling study](predictive-modeling-study.md): where a
  file's bits go, and what context-mixing models (on literals, or instead
  of LZ) would save under gpusqz's chunk and lane constraints.

**Working on gpusqz**

- [Building](building.md): toolchains, CMake options, packages.
- [Testing](testing.md): the test suites, the CPU reference decoder,
  testing Vulkan without a GPU.
- [Releases and versioning](releasing.md): Conventional Commits,
  semantic versioning, and what CI does on `main`.
