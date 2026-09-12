#!/usr/bin/env bash
# Compares gzp (GPU) against gzip -1/-6 (CPU) on a given file. Timings are
# wall-clock for the whole compress/decompress call, including file I/O and
# host<->device transfers, not just kernel time.
set -euo pipefail

GZP="${GZP:-./build/gzp}"
FILE="${1:?usage: run_bench.sh <file>}"
SIZE=$(stat -c%s "$FILE")
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Input: $FILE ($SIZE bytes)"
echo

echo "--- gzp (GPU, LZSS, 8KB chunks) ---"
t0=$(date +%s.%N)
"$GZP" c "$FILE" "$TMP/out.gzp" >/dev/null
t1=$(date +%s.%N)
"$GZP" d "$TMP/out.gzp" "$TMP/out.dec" >/dev/null
t2=$(date +%s.%N)
cmp -s "$FILE" "$TMP/out.dec" || { echo "ROUND-TRIP MISMATCH"; exit 1; }
csize=$(stat -c%s "$TMP/out.gzp")
awk -v s="$SIZE" -v c="$csize" -v ct="$(echo "$t1 - $t0" | bc)" -v dt="$(echo "$t2 - $t1" | bc)" \
  'BEGIN{printf "compress: %.3fs (%.1f MB/s)  decompress: %.3fs (%.1f MB/s)  ratio: %.4f  size: %d\n", \
         ct, s/1e6/ct, dt, s/1e6/dt, c/s, c}'
echo

for lvl in 1 6; do
  echo "--- gzip -$lvl (CPU, single-threaded) ---"
  t0=$(date +%s.%N)
  gzip -$lvl -c "$FILE" > "$TMP/out.gz"
  t1=$(date +%s.%N)
  gunzip -c "$TMP/out.gz" > "$TMP/out.gz.dec"
  t2=$(date +%s.%N)
  cmp -s "$FILE" "$TMP/out.gz.dec" || { echo "ROUND-TRIP MISMATCH"; exit 1; }
  csize=$(stat -c%s "$TMP/out.gz")
  awk -v s="$SIZE" -v c="$csize" -v ct="$(echo "$t1 - $t0" | bc)" -v dt="$(echo "$t2 - $t1" | bc)" \
    'BEGIN{printf "compress: %.3fs (%.1f MB/s)  decompress: %.3fs (%.1f MB/s)  ratio: %.4f  size: %d\n", \
           ct, s/1e6/ct, dt, s/1e6/dt, c/s, c}'
done
