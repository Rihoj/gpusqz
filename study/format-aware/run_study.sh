#!/usr/bin/env bash
# Reproduces docs/format-aware-transforms-study.md.
#
#   bash study/format-aware/run_study.sh <work dir>
#
# Needs a built ./build/gpusqz, a C compiler, zstd and split, and ~100MB free
# in <work dir>. The corpus is generated, nothing is downloaded. Takes about
# ten minutes, mostly zstd -19. Every number it prints is a size, so GPU
# contention doesn't change them.
set -euo pipefail
cd "$(dirname "$0")/../.."
W=$1
S=study/format-aware
G=./build/gpusqz
mkdir -p "$W/bin"
cc -O2 -o "$W/bin/genmesh" $S/genmesh.c -lm
cc -O2 -I$S -o "$W/bin/xform" $S/xform.c -lm
cc -O2 -I$S -o "$W/bin/unxform" $S/unxform.c -lm
"$W/bin/genmesh" "$W"

size() { wc -c < "$1" | tr -d ' '; }

# gpusqz at a profile on a file, printing the output size.
gsz() {
  "$G" c "$1" "$W/tmp.gsz" --profile "$2" > /dev/null 2>&1
  size "$W/tmp.gsz"
}

# zstd -19 on independent 1MB pieces: zstd's best parse, confined to the
# chunks gpusqz's `ratio` profile sees.
zstd_chunked() {
  rm -rf "$W/split" && mkdir "$W/split"
  split -b 1048576 "$1" "$W/split/p"
  local total=0 p
  for p in "$W"/split/p*; do total=$((total + $(zstd -q -19 -c "$p" | wc -c))); done
  echo "$total"
}

echo "== bit-exact round trips"
for input in shuffled ordered; do
  for mode in t2x t4x; do
    for recipe in f64 f32; do
      for chunk in 65536 1048576 0; do
        "$W/bin/xform" "$W/$input.stl" $mode $chunk "$W/t.bin" $recipe 2> /dev/null
        "$W/bin/unxform" $mode "$W/t.bin" $chunk "$W/back.stl" $recipe
        cmp "$W/$input.stl" "$W/back.stl"
      done
    done
  done
  echo "  $input: t2x and t4x, f64 and f32, 64KB/1MB/whole-file chunks: identical"
done

echo "== normal residual words that are exactly zero"
for recipe in f64 f32; do
  printf '  %s: ' $recipe
  "$W/bin/xform" "$W/shuffled.stl" t2x 0 "$W/t.bin" $recipe 2>&1 > /dev/null | sed 's/.*: //'
done

echo "== sizes in bytes (x modes use the f64 recipe; t2x32 is t2x with f32)"
row() { printf '%-9s %-6s %-5s %10s %10s %10s %10s %10s %10s %10s\n' "$@"; }
row input mode chunk transformed zstd-3 zstd-19 zstd-19L zstd-19/1MB gsz-speed gsz-ratio
for input in shuffled ordered; do
  f="$W/$input.stl"
  row $input raw - "$(size "$f")" "$(zstd -q -3 -c "$f" | wc -c)" "$(zstd -q -19 -c "$f" | wc -c)" \
    "$(zstd -q -19 --long -c "$f" | wc -c)" "$(zstd_chunked "$f")" "$(gsz "$f" speed)" "$(gsz "$f" ratio)"
  for mode in t2 t2x t2x32 t3 t4 t4x; do
    for chunk in 65536 1048576 0; do
      case $mode in
        t2x32) "$W/bin/xform" "$f" t2x $chunk "$W/t.bin" f32 2> /dev/null ;;
        *) "$W/bin/xform" "$f" $mode $chunk "$W/t.bin" 2> /dev/null ;;
      esac
      case $chunk in 65536) label=64KB ;; 1048576) label=1MB ;; *) label=file ;; esac
      row $input $mode $label "$(size "$W/t.bin")" "$(zstd -q -3 -c "$W/t.bin" | wc -c)" \
        "$(zstd -q -19 -c "$W/t.bin" | wc -c)" "$(zstd -q -19 --long -c "$W/t.bin" | wc -c)" \
        "$(zstd_chunked "$W/t.bin")" "$(gsz "$W/t.bin" speed)" "$(gsz "$W/t.bin" ratio)"
    done
  done
done
rm -rf "$W/split" "$W/t.bin" "$W/back.stl" "$W/tmp.gsz"
