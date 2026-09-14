---
name: benchmark
description: Measure gpusqz performance correctly - GPU-idle checks, the shared ollama GPU, both corpora, kernel vs wall MB/s, A/B against a baseline build back to back. Use for any speed or ratio claim, and before and after performance changes.
---

# Benchmark protocol

Numbers without their conditions are misleading here: the GPU is shared,
WSL2 adds fixed startup costs, and one corpus once hid a regression.

## 1. Check the GPU is idle

```
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

An `ollama` model often holds 11–15GB of VRAM. **Never stop or kill ollama**
(the guard hook blocks it). If it is loaded, ask the user to stop it for
clean numbers, or measure anyway and say so. Another process's memory
shrinks gpusqz's batches (default budget: 80% of free VRAM) and its compute
slows the latency-bound parse. Always report the VRAM and utilisation seen.

## 2. Corpora (both, every time)

```
find /usr/include -name '*.h' | xargs cat > corpus_varied.txt          # varied, distinct files
find /usr/include -name '*.h' | head -400 | xargs cat > block.txt
for i in $(seq 48); do cat block.txt; done > corpus_283mb.txt          # repetitive
```

Grow the varied corpus toward ~1GB with more distinct directories, never
by repeating. The repetitive one flatters match-finder changes: a table
change that looked free on it nearly halved compress-kernel throughput on
varied data. Keep corpora outside the repo (scratch dir); `*.txt` inputs
must not be committed.

## 3. Run

```
REPEAT=5 bash bench/run_bench.sh corpus_varied.txt                      # speed profile + gzip/zstd
CPU=0 REPEAT=5 bash bench/run_bench.sh corpus_varied.txt --profile balance
CPU=0 REPEAT=5 bash bench/run_bench.sh corpus_varied.txt --profile ratio
```

`REPEAT` keeps the best run per column. `GPUSQZ_VERBOSE=1` on a single run
shows setup, fread/fwrite, h2d/d2h, `kernel` and `kbusy`, and the batch shape.
`bench/run_bench.sh` needs GNU `stat`/`date` (on macOS, GNU coreutils first
on PATH).

## 4. Read the right column

- **`kbusy` kernel MB/s** is what the codec sustains. Wall MB/s includes
  ~0.2s of CUDA context creation and allocation under WSL2, and on large
  files decompression is bound by `fwrite` (page-cache writes).
- Throughput follows chunks in flight: ~350 1MB chunks saturate the
  RTX 5060 Ti, and 100MB inputs starve the `ratio` profile (1MB chunks).
- Vulkan kernel figures on Apple come from MoltenVK timestamps; on
  NVIDIA, Vulkan decode runs at 55–70% of CUDA.

## 5. A/B a change

Build the baseline from its commit in a separate worktree, then run the two
binaries back to back on the same corpus, alternating, same GPU state:

```
git worktree add /tmp/gpusqz-base <base-commit> && cmake -S /tmp/gpusqz-base -B /tmp/gpusqz-base/build && cmake --build /tmp/gpusqz-base/build -j
for bin in /tmp/gpusqz-base/build/gpusqz ./build/gpusqz; do GPUSQZ=$bin CPU=0 REPEAT=5 bash bench/run_bench.sh corpus_varied.txt; done
```

Keep a change only if it wins on both corpora, and check output sizes: a
speedup that changes the ratio is a trade-off to report, not a free win.
Report per profile, per corpus: kernel and wall MB/s, ratio, conditions.
