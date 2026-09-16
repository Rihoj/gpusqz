#!/usr/bin/env bash
# Reproduces docs/predictive-modeling-study.md.
#
#   bash study/predictive/run_study.sh <work dir> [jobs]
#
# Needs a built ./build/gpusqz, g++, curl, python3 (to unzip enwik8), and
# ~2GB free in <work dir>. The context-mixing runs take roughly an hour on
# six cores.
set -euo pipefail
cd "$(dirname "$0")/../.."
W=$1
J=${2:-6}
mkdir -p "$W/bin"
g++ -O2 -std=c++17 -Isrc -o "$W/bin/bit_account" study/predictive/bit_account.cpp
g++ -O3 -march=native -std=c++17 -Isrc -o "$W/bin/lit_models" study/predictive/lit_models.cpp
g++ -O3 -march=native -std=c++17 -o "$W/bin/cm_raw" study/predictive/cm_raw.cpp

E=$W/enwik8
V=$W/headers.txt
if [ ! -f "$E" ]; then
  curl -sSfL -o "$W/enwik8.zip" https://mattmahoney.net/dc/enwik8.zip
  python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$W/enwik8.zip" "$W"
fi
[ -f "$V" ] || find /usr/include -name '*.h' -print0 | sort -z | xargs -0 cat > "$V"

echo "== bit accounting"
for f in "$E" "$V"; do
  for p in speed:65536 balance:262144 ratio:1048576; do
    ./build/gpusqz c "$f" "$f.${p%%:*}.gsz" --profile "${p%%:*}" > /dev/null
    "$W/bin/bit_account" "$f.${p%%:*}.gsz" "$f.${p%%:*}.lits"
  done
done

echo "== literal models"
for f in "$E" "$V"; do
  for p in speed:65536 ratio:1048576; do
    for m in lits1 lits2 lits2b text1 text2 text2b \
             "chunk --src lits --orders 0,1,2,3" \
             "chunk --src text --orders 0,1,2,3,4 --word" \
             "lane --src lits --orders 0,1,2 --bits 14"; do
      echo "$f.${p%%:*}.lits $f ${p##*:} $m"
    done
  done
done | xargs -P "$J" -L 1 "$W/bin/lit_models"

echo "== context mixing on raw bytes"
{
  for f in "$E" "$V"; do
    echo "$f --word --chunk 0 --bits 24"
    echo "$f --word --chunk 65536 --bits 20"
    echo "$f --word --chunk 262144 --bits 21"
    echo "$f --word --chunk 1048576 --bits 22"
    echo "$f --word --chunk 65536 --bits 16 --lanes 32"
    echo "$f --chunk 65536 --bits 16 --lanes 32 --orders 0,1,2,4"
    echo "$f --word --chunk 65536 --bits 18 --lanes 32"
    echo "$f --word --chunk 65536 --bits 20 --lanes 32"
    echo "$f --word --chunk 262144 --bits 21 --lanes 32"
    echo "$f --word --chunk 1048576 --bits 18 --lanes 32"
    echo "$f --word --chunk 1048576 --bits 22 --lanes 32"
  done
  echo "$E --word --chunk 65536 --bits 20 --prior $V"
  echo "$V --word --chunk 65536 --bits 20 --prior $E"
  echo "$E --word --chunk 65536 --bits 20 --from 50000000"
  echo "$E --word --chunk 65536 --bits 20 --from 50000000 --prior $E"
  echo "$E --word --chunk 1048576 --bits 22 --from 50000000"
  echo "$E --word --chunk 1048576 --bits 22 --from 50000000 --prior $E"
  echo "$E --word --chunk 65536 --bits 16 --lanes 32 --from 50000000"
  echo "$E --word --chunk 65536 --bits 16 --lanes 32 --from 50000000 --prior $E"
} | xargs -P "$J" -L 1 "$W/bin/cm_raw"

echo "== CPU references"
for f in "$E" "$V"; do
  for c in "xz -9 -T1" "zstd -19" "zstd -19 --long=27" "bzip2 -9"; do
    echo "$f | $c | $($c -c "$f" | wc -c)"
  done
done
