#!/usr/bin/env bash
# Regenerates the .gzp fixtures in this directory with the GPU compressor.
# Needs an NVIDIA GPU. Run it after any change to the file format, then
# commit the results; ctest (see CMakeLists.txt) checks on every platform,
# GPU or not, that gzp_refdec decodes each <name>__<variant>.gzp back to
# <name>.orig and rejects each *.gzpbad.
#
#   tests/fixtures/make_fixtures.sh [path/to/gzp]
#
# The .orig inputs are committed and never regenerated here, so fixtures
# stay comparable across format versions.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GZP="$(realpath "${1:-$HERE/../../build/gzp}")"
cd "$HERE"
rm -f ./*.gzp ./*.gzpbad

c() { # c <name> <variant> [gzp args...], env passes through
  local name="$1" variant="$2"
  shift 2
  "$GZP" c "$name.orig" "${name}__${variant}.gzp" "$@" 2>/dev/null
}

c text default
c text ratio --profile ratio
GZP_FORCE_LIT_SHIFT=0 c text c4k_lit0 4096
GZP_FORCE_LIT_SHIFT=4 c text c4k_lit4 4096
GZP_FORCE_LIT_SHIFT=8 c text c4k_lit8 4096
GZP_FORCE_BATCH=3 c text c4k_b3 4096 # several table groups
c random default
c random c4k 4096
c zeros c1k 1024
c empty default

# Corrupt files the decoder must reject.
head -c -7 text__default.gzp > text_truncated.gzpbad
printf 'XXXX' | cat - <(tail -c +5 text__default.gzp) > text_badmagic.gzpbad
# A wrong sequence count in the first chunk's rANS header: the decoder then
# consumes a different number of stream words than the chunk holds, which
# it must detect. (A flipped byte inside the rANS stream itself is not
# reliably detectable: the format has no checksum, see README.)
python3 - <<'EOF'
import struct
d = bytearray(open("text__c4k_lit0.gzp", "rb").read())
magic, ver, cs, orig, chunks, groups, tables_offset = struct.unpack_from("<IIIQIIQ", d, 0)
payload = 36 + 16 * chunks + 12 * groups
first = payload + struct.unpack_from("<Q", d, 36)[0]
assert d[first] == 2, "first chunk should be LzRans"
d[first + 1] ^= 0x01  # low byte of n_seq
open("text_badseq.gzpbad", "wb").write(d)
EOF
ls -l ./*.gzp ./*.gzpbad
