#!/usr/bin/env bash
# Compares gpusqz (GPU) against CPU compressors on one file.
#
#   bench/run_bench.sh <file> [chunk_size | --profile speed|balance|ratio]
#
# Wall times cover the whole process (file I/O, host<->device copies and,
# for gpusqz, ~0.2s of CUDA context creation and allocation on WSL2), so they
# favour large inputs. The "kernel" column is GPU time only, from
# cudaEvents -- the wall-clock time during which any gpusqz kernel was running
# (GPUSQZ_VERBOSE's "kbusy") -- and is what the codec itself sustains once
# the fixed costs are paid. CPU tools run single-threaded.
#
# Env:
#   REPEAT=<n>   run each codec n times (default 1) and report the best
#                (highest-throughput) run per column, since on a shared
#                GPU a single run's wall time can be dominated by another
#                process's contention rather than by gpusqz itself.
#   CPU=0        skip the CPU compressors (gpusqz only).
set -euo pipefail

GPUSQZ="${GPUSQZ:-./build/gpusqz}"
FILE="${1:?usage: run_bench.sh <file> [chunk_size | --profile speed|balance|ratio]}"
shift
GPUSQZ_ARGS=("$@")
REPEAT="${REPEAT:-1}"
SIZE=$(stat -c%s "$FILE")
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Input: $FILE ($SIZE bytes)$([ "$REPEAT" -gt 1 ] && echo ", best of $REPEAT runs")"
printf "%-22s %10s %10s %10s %10s %8s\n" "codec" "comp MB/s" "kern MB/s" "dec MB/s" "kern MB/s" "ratio"

mbps() { awk -v b="$1" -v s="$2" 'BEGIN { if (s > 0) printf "%.0f", b / 1e6 / s; else printf "0" }'; }
kern() { grep -E "^\s*kbusy" "$1" | awk '{gsub(/\(/, "", $3); printf "%.0f", $3}'; }
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

run_gpusqz() {
  local label="$1"
  local cw=() ck=() dw=() dk=() csize=0
  for ((i = 0; i < REPEAT; i++)); do
    local t0 t1 t2
    t0=$(date +%s.%N)
    GPUSQZ_VERBOSE=1 "$GPUSQZ" c "$FILE" "$TMP/out.gsz" ${GPUSQZ_ARGS[@]+"${GPUSQZ_ARGS[@]}"} 2>"$TMP/c.log"
    t1=$(date +%s.%N)
    GPUSQZ_VERBOSE=1 "$GPUSQZ" d "$TMP/out.gsz" "$TMP/out.dec" 2>"$TMP/d.log"
    t2=$(date +%s.%N)
    cmp -s "$FILE" "$TMP/out.dec" || { echo "ROUND-TRIP MISMATCH ($label)"; exit 1; }
    csize=$(stat -c%s "$TMP/out.gsz")
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

run_gpusqz "gpusqz ${GPUSQZ_ARGS[*]:-(default)}"
[ "${CPU:-1}" = "0" ] && exit 0
run_cpu "gzip -1" "gzip -1 -c" "gzip -d -c"
run_cpu "gzip -6" "gzip -6 -c" "gzip -d -c"
if command -v zstd >/dev/null; then
  run_cpu "zstd -1 (1 thread)" "zstd -1 -T1 -q -c" "zstd -d -q -c"
  run_cpu "zstd -3 (1 thread)" "zstd -3 -T1 -q -c" "zstd -d -q -c"
fi
if command -v lz4 >/dev/null; then
  run_cpu "lz4 -1" "lz4 -1 -q -c" "lz4 -d -q -c"
fi
