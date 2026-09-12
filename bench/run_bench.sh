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
#
# Env:
#   REPEAT=<n>   run each codec n times (default 1) and report the best
#                (highest-throughput) run per column, since on a shared
#                GPU a single run's wall time can be dominated by another
#                process's contention rather than by gzp itself.
set -euo pipefail

GZP="${GZP:-./build/gzp}"
FILE="${1:?usage: run_bench.sh <file> [chunk_size]}"
CHUNK="${2:-}"
REPEAT="${REPEAT:-1}"
SIZE=$(stat -c%s "$FILE")
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Input: $FILE ($SIZE bytes)$([ "$REPEAT" -gt 1 ] && echo ", best of $REPEAT runs")"
printf "%-22s %10s %10s %10s %10s %8s\n" "codec" "comp MB/s" "kern MB/s" "dec MB/s" "kern MB/s" "ratio"

mbps() { awk -v b="$1" -v s="$2" 'BEGIN { if (s > 0) printf "%.0f", b / 1e6 / s; else printf "0" }'; }
kern() { grep -E "^\s*kernel" "$1" | awk '{gsub(/\(/, "", $3); printf "%.0f", $3}'; }
# Prints the largest of its arguments (highest MB/s = least GPU/host
# contention that run) — "-" is skipped since CPU tools have no kernel MB/s.
best() {
  local m=0 v
  for v in "$@"; do
    [ "$v" = "-" ] && continue
    [ "$v" -gt "$m" ] 2>/dev/null && m="$v"
  done
  echo "$m"
}

run_gzp() {
  local label="$1"; shift
  local cw=() ck=() dw=() dk=() csize=0
  for ((i = 0; i < REPEAT; i++)); do
    local t0 t1 t2
    t0=$(date +%s.%N)
    GZP_VERBOSE=1 "$GZP" c "$FILE" "$TMP/out.gzp" ${CHUNK:+$CHUNK} "$@" 2>"$TMP/c.log"
    t1=$(date +%s.%N)
    GZP_VERBOSE=1 "$GZP" d "$TMP/out.gzp" "$TMP/out.dec" 2>"$TMP/d.log"
    t2=$(date +%s.%N)
    cmp -s "$FILE" "$TMP/out.dec" || { echo "ROUND-TRIP MISMATCH ($label)"; exit 1; }
    csize=$(stat -c%s "$TMP/out.gzp")
    cw+=("$(mbps "$SIZE" "$(echo "$t1 - $t0" | bc)")")
    ck+=("$(kern "$TMP/c.log")")
    dw+=("$(mbps "$SIZE" "$(echo "$t2 - $t1" | bc)")")
    dk+=("$(kern "$TMP/d.log")")
  done
  printf "%-22s %10s %10s %10s %10s %8.4f\n" "$label" \
    "$(best "${cw[@]}")" "$(best "${ck[@]}")" "$(best "${dw[@]}")" "$(best "${dk[@]}")" \
    "$(echo "scale=4; $csize / $SIZE" | bc)"
}

run_cpu() {
  local label="$1" comp="$2" decomp="$3"
  local cw=() dw=() csize=0
  for ((i = 0; i < REPEAT; i++)); do
    local t0 t1 t2
    t0=$(date +%s.%N)
    eval "$comp" < "$FILE" > "$TMP/out.cpu"
    t1=$(date +%s.%N)
    eval "$decomp" < "$TMP/out.cpu" > "$TMP/out.dec"
    t2=$(date +%s.%N)
    cmp -s "$FILE" "$TMP/out.dec" || { echo "ROUND-TRIP MISMATCH ($label)"; exit 1; }
    csize=$(stat -c%s "$TMP/out.cpu")
    cw+=("$(mbps "$SIZE" "$(echo "$t1 - $t0" | bc)")")
    dw+=("$(mbps "$SIZE" "$(echo "$t2 - $t1" | bc)")")
  done
  printf "%-22s %10s %10s %10s %10s %8.4f\n" "$label" \
    "$(best "${cw[@]}")" "-" "$(best "${dw[@]}")" "-" \
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
