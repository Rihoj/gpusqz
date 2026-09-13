#!/usr/bin/env bash
# Round-trip correctness tests for gzp, covering the edge cases most likely
# to hide bugs: empty input, sub-chunk, exact chunk multiples, and
# incompressible data.
#
#   tests/round_trip.sh [--extremes] [path/to/gzp]
#
# Env:
#   CHUNK=<n>         chunk size to test (default 8192); ignored with --extremes
#   GZP_PREFIX="..."  command prefix, e.g. "compute-sanitizer --tool memcheck"
#   EXTRA_FILE=<p>    also round-trip this real file
#   BIG=1             also run 300MB random + 300MB text (multi-batch) cases
set -euo pipefail

EXTREMES=0
if [ "${1:-}" = "--extremes" ]; then
  EXTREMES=1
  shift
fi
GZP="${1:-./build/gzp}"
# Independent CPU decoder (built alongside gzp); skipped if absent.
REFDEC="${REFDEC:-$(dirname "$GZP")/gzp_refdec}"
[ -x "$REFDEC" ] || REFDEC=""
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Word-split on purpose so a multi-word prefix works.
# shellcheck disable=SC2206
PREFIX=(${GZP_PREFIX:-})

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
  local chunk="$1" name="$2" f="$3"
  local comp="$TMP/$name.gzp" dec="$TMP/$name.out"
  "${PREFIX[@]}" "$GZP" c "$f" "$comp" "$chunk" 2>>"$TMP/log"
  "${PREFIX[@]}" "$GZP" d "$comp" "$dec" 2>>"$TMP/log"
  local ref_ok=1
  if [ -n "$REFDEC" ]; then
    if ! "$REFDEC" "$comp" "$TMP/$name.ref" 2>>"$TMP/log" || ! cmp -s "$f" "$TMP/$name.ref"; then
      ref_ok=0
    fi
    rm -f "$TMP/$name.ref"
  fi
  if [ "$ref_ok" -eq 0 ]; then
    printf "FAIL chunk=%-6d %-20s CPU reference decoder mismatch\n" "$chunk" "$name"
    fail=1
  elif cmp -s "$f" "$dec"; then
    local in_sz comp_sz
    in_sz=$(stat -c%s "$f")
    comp_sz=$(stat -c%s "$comp")
    printf "PASS chunk=%-6d %-20s in=%-10d comp=%-10d ratio=%.3f\n" "$chunk" "$name" "$in_sz" "$comp_sz" \
      "$(echo "scale=3; $comp_sz / ($in_sz + 0.0001)" | bc)"
  else
    printf "FAIL chunk=%-6d %-20s round-trip mismatch\n" "$chunk" "$name"
    fail=1
  fi
  rm -f "$comp" "$dec"
}

# Compresses and decompresses with different GZP_FORCE_BATCH values, so
# LzRans table groups (fixed at compress time to whatever batch size was
# used then) never align with the decoder's own batches. This is the one
# scenario that exercises the per-chunk group_id lookup instead of always
# hitting the trivial case where a decode batch sits inside one group.
run_case_mismatched_batch() {
  local name="$1" f="$2" enc_batch="$3" dec_batch="$4"
  local comp="$TMP/$name.gzp" dec="$TMP/$name.out"
  GZP_FORCE_BATCH="$enc_batch" "${PREFIX[@]}" "$GZP" c "$f" "$comp" 2>>"$TMP/log"
  GZP_FORCE_BATCH="$dec_batch" "${PREFIX[@]}" "$GZP" d "$comp" "$dec" 2>>"$TMP/log"
  local ref_ok=1
  if [ -n "$REFDEC" ]; then
    if ! "$REFDEC" "$comp" "$TMP/$name.ref" 2>>"$TMP/log" || ! cmp -s "$f" "$TMP/$name.ref"; then
      ref_ok=0
    fi
    rm -f "$TMP/$name.ref"
  fi
  if [ "$ref_ok" -eq 0 ]; then
    printf "FAIL %-28s CPU reference decoder mismatch (enc_batch=%s dec_batch=%s)\n" "$name" "$enc_batch" "$dec_batch"
    fail=1
  elif cmp -s "$f" "$dec"; then
    printf "PASS %-28s enc_batch=%-6s dec_batch=%-6s\n" "$name" "$enc_batch" "$dec_batch"
  else
    printf "FAIL %-28s round-trip mismatch (enc_batch=%s dec_batch=%s)\n" "$name" "$enc_batch" "$dec_batch"
    fail=1
  fi
  rm -f "$comp" "$dec"
}

# Exercises --profile speed|balance|ratio (a convenience over chunk_size,
# see main.cu) round-tripping correctly, plus its two error cases.
run_profile_cases() {
  local f comp dec
  f="$(make_case profile_src $((2 * 1024 * 1024)) text)"
  for prof in speed balance ratio; do
    comp="$TMP/profile_$prof.gzp"
    dec="$TMP/profile_$prof.out"
    "${PREFIX[@]}" "$GZP" c "$f" "$comp" --profile "$prof" 2>>"$TMP/log"
    "${PREFIX[@]}" "$GZP" d "$comp" "$dec" 2>>"$TMP/log"
    if cmp -s "$f" "$dec"; then
      printf "PASS %-28s --profile %s\n" "profile" "$prof"
    else
      printf "FAIL %-28s --profile %s round-trip mismatch\n" "profile" "$prof"
      fail=1
    fi
    rm -f "$comp" "$dec"
  done
  if "${PREFIX[@]}" "$GZP" c "$f" "$TMP/profile_both.gzp" 65536 --profile speed 2>>"$TMP/log"; then
    printf "FAIL %-28s chunk_size + --profile should be rejected\n" "profile"
    fail=1
  else
    printf "PASS %-28s chunk_size + --profile rejected\n" "profile"
  fi
  if "${PREFIX[@]}" "$GZP" c "$f" "$TMP/profile_bogus.gzp" --profile bogus 2>>"$TMP/log"; then
    printf "FAIL %-28s unknown --profile should be rejected\n" "profile"
    fail=1
  else
    printf "PASS %-28s unknown --profile rejected\n" "profile"
  fi
  rm -f "$TMP/profile_both.gzp" "$TMP/profile_bogus.gzp"
}

run_suite() {
  local chunk="$1"
  local sub=$((chunk > 100 ? chunk - 100 : (chunk > 1 ? chunk / 2 : 1)))
  run_case "$chunk" "empty" "$(make_case empty 0 empty)"
  run_case "$chunk" "one_byte" "$(make_case one_byte 1 random)"
  run_case "$chunk" "sub_chunk" "$(make_case sub_chunk "$sub" text)"
  run_case "$chunk" "exact_one_chunk" "$(make_case exact_one_chunk "$chunk" text)"
  run_case "$chunk" "exact_multi_chunk" "$(make_case exact_multi_chunk $((chunk * 5)) text)"
  run_case "$chunk" "off_by_one_over" "$(make_case off_by_one_over $((chunk + 1)) text)"
  run_case "$chunk" "all_zeros_5mb" "$(make_case all_zeros_5mb $((5 * 1024 * 1024)) zero)"
  run_case "$chunk" "random_5mb" "$(make_case random_5mb $((5 * 1024 * 1024)) random)"
  run_case "$chunk" "mixed_text_5mb" "$(make_case mixed_text_5mb $((5 * 1024 * 1024)) text)"
  if [ -n "${EXTRA_FILE:-}" ] && [ -f "$EXTRA_FILE" ]; then
    cp "$EXTRA_FILE" "$TMP/real_file.in"
    run_case "$chunk" "real_file" "$TMP/real_file.in"
  fi
}

if [ "$EXTREMES" -eq 1 ]; then
  # CHUNK=1 makes every byte its own chunk: slow, but it must still work.
  # 65535/65536/1048575/1048576 straddle offsets that were once (before
  # widening to u32) representable-limit boundaries; 1048576 is kMaxChunkSize.
  for chunk in 1 16 4096 8192 32768 65535 65536 1048575 1048576; do
    run_suite "$chunk"
  done
else
  run_suite "${CHUNK:-8192}"
  run_profile_cases
fi

if [ "$EXTREMES" -eq 1 ]; then
  mismatch_file="$(make_case mismatch_batch_src $((5 * 1024 * 1024)) text)"
  run_case_mismatched_batch "small_enc_large_dec" "$mismatch_file" 64 4000
  run_case_mismatched_batch "large_enc_small_dec" "$mismatch_file" 4000 64
fi

if [ "${BIG:-0}" = "1" ]; then
  chunk="${CHUNK:-8192}"
  # >= 5 batches at any sane batch size; exercises multi-batch paths and
  # a stream-ring wraparound with a batch count not divisible by 3.
  run_case "$chunk" "big_random_300mb" "$(make_case big_random_300mb $((300 * 1024 * 1024)) random)"
  run_case "$chunk" "big_text_300mb" "$(make_case big_text_300mb $((300 * 1024 * 1024)) text)"
fi

if [ "$fail" -ne 0 ]; then
  echo "--- log tail ---"
  tail -40 "$TMP/log" || true
  echo "SOME TESTS FAILED"
  exit 1
fi
echo "ALL TESTS PASSED"
