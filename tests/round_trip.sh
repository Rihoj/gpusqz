#!/usr/bin/env bash
# Round-trip correctness tests for gzp, covering the edge cases most likely
# to hide bugs: empty input, sub-chunk, exact chunk multiples, and
# incompressible data.
set -euo pipefail

GZP="${1:-./build/gzp}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

CHUNK="${CHUNK:-8192}"
fail=0

make_case() {
  local name="$1" size="$2" kind="$3"
  local f="$TMP/$name.in"
  case "$kind" in
    zero) head -c "$size" /dev/zero > "$f" ;;
    random) head -c "$size" /dev/urandom > "$f" ;;
    text) yes "the quick brown fox jumps over the lazy dog" | head -c "$size" > "$f" ;;
    empty) : > "$f" ;;
  esac
  echo "$f"
}

run_case() {
  local name="$1" f="$2"
  local comp="$TMP/$name.gzp" dec="$TMP/$name.out"
  "$GZP" c "$f" "$comp" "$CHUNK" 2>>"$TMP/log"
  "$GZP" d "$comp" "$dec" 2>>"$TMP/log"
  if cmp -s "$f" "$dec"; then
    local in_sz comp_sz
    in_sz=$(stat -c%s "$f")
    comp_sz=$(stat -c%s "$comp")
    printf "PASS %-24s in=%-10d comp=%-10d ratio=%.3f\n" "$name" "$in_sz" "$comp_sz" \
      "$(echo "scale=3; $comp_sz / ($in_sz + 0.0001)" | bc)"
  else
    printf "FAIL %-24s round-trip mismatch\n" "$name"
    fail=1
  fi
}

run_case "empty" "$(make_case empty 0 empty)"
run_case "one_byte" "$(make_case one_byte 1 random)"
SUB_CHUNK_SIZE=$((CHUNK > 100 ? CHUNK - 100 : (CHUNK > 1 ? CHUNK / 2 : 1)))
run_case "sub_chunk" "$(make_case sub_chunk "$SUB_CHUNK_SIZE" text)"
run_case "exact_one_chunk" "$(make_case exact_one_chunk "$CHUNK" text)"
run_case "exact_multi_chunk" "$(make_case exact_multi_chunk $((CHUNK * 5)) text)"
run_case "off_by_one_over" "$(make_case off_by_one_over $((CHUNK + 1)) text)"
run_case "all_zeros_5mb" "$(make_case all_zeros_5mb $((5 * 1024 * 1024)) zero)"
run_case "random_5mb" "$(make_case random_5mb $((5 * 1024 * 1024)) random)"
run_case "mixed_text_5mb" "$(make_case mixed_text_5mb $((5 * 1024 * 1024)) text)"

if [ -n "${EXTRA_FILE:-}" ] && [ -f "$EXTRA_FILE" ]; then
  cp "$EXTRA_FILE" "$TMP/real_file.in"
  run_case "real_file" "$TMP/real_file.in"
fi

if [ "$fail" -ne 0 ]; then
  echo "--- log tail ---"
  tail -40 "$TMP/log" || true
  echo "SOME TESTS FAILED"
  exit 1
fi
echo "ALL TESTS PASSED"
