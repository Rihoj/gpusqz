#!/usr/bin/env bash
# Regenerates the .gsz fixtures in this directory with the GPU compressor.
# Needs an NVIDIA GPU. Run it after any change to the file format, then
# commit the results; ctest (see CMakeLists.txt) checks on every platform,
# GPU or not, that gpusqz_refdec decodes each <name>__<variant>.gsz back to
# <name>.orig and rejects each *.gszbad.
#
#   tests/fixtures/make_fixtures.sh [path/to/gpusqz]
#
# The .orig inputs are committed and never regenerated here.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUSQZ="$(realpath "${1:-$HERE/../../build/gpusqz}")"
cd "$HERE"
rm -f ./*.gsz ./*.gszbad

c() { # c <name> <variant> [gpusqz args...], env passes through
  local name="$1" variant="$2"
  shift 2
  "$GPUSQZ" c "$name.orig" "${name}__${variant}.gsz" "$@" 2>/dev/null
}

c text default
c text ratio --profile ratio
GPUSQZ_FORCE_LIT_SHIFT=0 c text c4k_lit0 4096
GPUSQZ_FORCE_LIT_SHIFT=4 c text c4k_lit4 4096
GPUSQZ_FORCE_LIT_SHIFT=8 c text c4k_lit8 4096
GPUSQZ_FORCE_BATCH=3 c text c4k_b3 4096 # several table groups
c random default
c random c4k 4096
c zeros c1k 1024
c empty default

# Corrupt files the decoder must reject.
head -c -7 text__default.gsz > text_truncated.gszbad
printf 'XXXX' | cat - <(tail -c +5 text__default.gsz) > text_badmagic.gszbad
# A wrong sequence count in the first chunk's rANS header: the decoder then
# consumes a different number of stream words than the chunk holds, which
# it must detect. (A flipped byte inside the rANS stream itself is not
# reliably detectable: the format has no checksum, see docs/limitations.md.)
python3 - <<'EOF'
import struct
d = bytearray(open("text__c4k_lit0.gsz", "rb").read())
magic, ver, cs, orig, chunks, groups, tables_offset = struct.unpack_from("<IIIQIIQ", d, 0)
first = 36 + 8 * chunks + 20 * groups  # sizes, checksums, directory, then chunk 0
assert d[first] == 2, "first chunk should be LzRans"
d[first + 1] ^= 0x01  # low byte of n_seq
# The chunk's checksum (format.h) would catch this on its own, so it is
# recomputed: this fixture must exercise the rANS decoder's own rejection.
lane = [2166136261] * 32
size = struct.unpack_from("<I", d, 36)[0]
for i in range(size):
    l = i % 32
    lane[l] = ((lane[l] ^ d[first + i]) * 16777619) & 0xFFFFFFFF
h = 2166136261
for l in range(32):
    for b in range(4):
        h = ((h ^ ((lane[l] >> (8 * b)) & 0xFF)) * 16777619) & 0xFFFFFFFF
struct.pack_into("<I", d, 36 + 4 * chunks, h ^ size)
open("text_badseq.gszbad", "wb").write(d)
EOF
# One payload byte flipped and the checksum left alone: the chunk would
# still decode, so only the checksum (format.h) can catch it.
python3 - <<'EOF'
import struct
d = bytearray(open("text__default.gsz", "rb").read())
magic, ver, cs, orig, chunks, groups, tables_offset = struct.unpack_from("<IIIQIIQ", d, 0)
first = 36 + 8 * chunks + 20 * groups
size = struct.unpack_from("<I", d, 36)[0]
d[first + size // 2] ^= 0x20  # inside chunk 0's payload, well past the flag byte
open("text_badchunkhash.gszbad", "wb").write(d)
EOF
# The last table group's coded counts cut short by 3 bytes, with its
# table_bytes shrunk to match, so the directory still adds up but the
# table decoder runs out of data.
python3 - <<'EOF'
import struct
d = bytearray(open("text__c4k_b3.gsz", "rb").read())
magic, ver, cs, orig, chunks, groups, tables_offset = struct.unpack_from("<IIIQIIQ", d, 0)
last = 36 + 8 * chunks + 20 * (groups - 1)
struct.pack_into("<I", d, last + 12, struct.unpack_from("<I", d, last + 12)[0] - 3)
open("text_badtable.gszbad", "wb").write(d[:-3])
EOF
ls -l ./*.gsz ./*.gszbad
