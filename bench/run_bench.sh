#!/usr/bin/env bash
# Compares gzp (GPU) against CPU compressors on one file.
#
#   bench/run_bench.sh <file> [chunk_size]
#
# Wall times cover the whole process (file I/O, host<->device copies and,
# for gzp, ~0.15-0.3s of CUDA context creation on WSL2), so they favour
# large inputs. The "kernel" column is GPU time only, from cudaEvents, and
# is what the codec itself sustains once the fixed costs are paid. CPU
# tools run single-threaded.
set -euo pipefail

GZP="${GZP:-./build/gzp}"
FILE="${1:?usage: run_bench.sh <file> [chunk_size]}"
CHUNK="${2:-}"
SIZE=$(stat -c%s "$FILE")
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Input: $FILE ($SIZE bytes)"
printf "%-22s %10s %10s %10s %10s %8s\n" "codec" "comp MB/s" "kern MB/s" "dec MB/s" "kern MB/s" "ratio"

mbps() { awk -v b="$1" -v s="$2" 'BEGIN { if (s > 0) printf "%.0f", b / 1e6 / s; else printf "-" }'; }
kern() { grep -E "^\s*kernel" "$1" | awk '{gsub(/\(/, "", $3); printf "%.0f", $3}'; }

run_gzp() {
  local label="$1"; shift
  local t0 t1 t2 csize
  t0=$(date +%s.%N)
  GZP_VERBOSE=1 "$GZP" c "$FILE" "$TMP/out.gzp" ${CHUNK:+$CHUNK} "$@" 2>"$TMP/c.log"
  t1=$(date +%s.%N)
  GZP_VERBOSE=1 "$GZP" d "$TMP/out.gzp" "$TMP/out.dec" 2>"$TMP/d.log"
  t2=$(date +%s.%N)
  cmp -s "$FILE" "$TMP/out.dec" || { echo "ROUND-TRIP MISMATCH ($label)"; exit 1; }
  csize=$(stat -c%s "$TMP/out.gzp")
  printf "%-22s %10s %10s %10s %10s %8.4f\n" "$label" \
    "$(mbps "$SIZE" "$(echo "$t1 - $t0" | bc)")" "$(kern "$TMP/c.log")" \
    "$(mbps "$SIZE" "$(echo "$t2 - $t1" | bc)")" "$(kern "$TMP/d.log")" \
    "$(echo "scale=4; $csize / $SIZE" | bc)"
}

run_cpu() {
  local label="$1" comp="$2" decomp="$3"
  local t0 t1 t2 csize
  t0=$(date +%s.%N)
  eval "$comp" < "$FILE" > "$TMP/out.cpu"
  t1=$(date +%s.%N)
  eval "$decomp" < "$TMP/out.cpu" > "$TMP/out.dec"
  t2=$(date +%s.%N)
  cmp -s "$FILE" "$TMP/out.dec" || { echo "ROUND-TRIP MISMATCH ($label)"; exit 1; }
  csize=$(stat -c%s "$TMP/out.cpu")
  printf "%-22s %10s %10s %10s %10s %8.4f\n" "$label" \
    "$(mbps "$SIZE" "$(echo "$t1 - $t0" | bc)")" "-" \
    "$(mbps "$SIZE" "$(echo "$t2 - $t1" | bc)")" "-" \
    "$(echo "scale=4; $csize / $SIZE" | bc)"
}

run_gzp "gzp (GPU, lzrans)"
run_gzp "gzp (GPU, lz)" --mode lz
run_cpu "gzip -1" "gzip -1 -c" "gzip -d -c"
run_cpu "gzip -6" "gzip -6 -c" "gzip -d -c"
if command -v zstd >/dev/null; then
  run_cpu "zstd -1 (1 thread)" "zstd -1 -T1 -q -c" "zstd -d -q -c"
  run_cpu "zstd -3 (1 thread)" "zstd -3 -T1 -q -c" "zstd -d -q -c"
fi
if command -v lz4 >/dev/null; then
  run_cpu "lz4 -1" "lz4 -1 -q -c" "lz4 -d -q -c"
fi
