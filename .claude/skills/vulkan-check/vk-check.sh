#!/usr/bin/env bash
# Checks the Vulkan backend on lavapipe (Mesa's CPU Vulkan driver) at every
# subgroup size, against CUDA and the CPU reference decoder:
#   - Vulkan output is byte-identical to CUDA's (when a CUDA GPU is usable)
#   - Vulkan decodes its own files, and gpusqz_refdec decodes them too
#   - CUDA decodes Vulkan's files, Vulkan decodes CUDA's
#
#   vk-check.sh [build dir]        (default: build)
# Env:
#   WIDTHS="256 512 1024 2048"   lavapipe LLVM vector widths = subgroup 8/16/32/64
#   CHUNKS="8192 65536 1048576"  chunk sizes (8192 on random data exercises the
#                                encoder fallbacks, where a lane race once hid)
#   LVP_ICD=<lvp_icd.json>       lavapipe ICD (default: the usual distro paths)
#   VALIDATE=1                   Khronos validation layer with sync validation
#                                (needs $VULKAN_SDK with the layers)
set -u
root="$(cd "$(dirname "$0")/../../.." && pwd)"
build="${1:-$root/build}"
g="$build/gpusqz"
ref="$build/gpusqz_refdec"
[ -x "$g" ] && [ -x "$ref" ] || { echo "build first: $g and $ref not found" >&2; exit 1; }

icd="${LVP_ICD:-}"
for c in /usr/share/vulkan/icd.d/lvp_icd.json /usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
         /usr/share/vulkan/icd.d/lvp_icd.aarch64.json; do
  [ -z "$icd" ] && [ -f "$c" ] && icd=$c
done
[ -n "$icd" ] || { echo "lavapipe not found (Debian/Ubuntu: mesa-vulkan-drivers); set LVP_ICD" >&2; exit 1; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
# Inputs: this repo's sources (ordinary text mix) and random bytes.
: > "$tmp/text"
while [ "$(wc -c < "$tmp/text")" -lt 4000000 ]; do
  find "$root/src" "$root/tests" -type f \( -name '*.cpp' -o -name '*.cu*' -o -name '*.h' -o -name '*.comp' -o -name '*.glsl' \) \
    -exec cat {} + >> "$tmp/text"
done
head -c 3000000 /dev/urandom > "$tmp/random"

have_cuda=0
if GPUSQZ_BACKEND=cuda "$g" c "$tmp/random" "$tmp/probe.gsz" 65536 2> /dev/null; then have_cuda=1; fi
[ $have_cuda = 1 ] || echo "note: no usable CUDA GPU; skipping the CUDA comparisons"

lvp() {
  local w=$1; shift
  if [ "${VALIDATE:-0}" = 1 ]; then
    VK_ICD_FILENAMES=$icd LP_NATIVE_VECTOR_WIDTH=$w \
      LD_LIBRARY_PATH="${VULKAN_SDK:?VALIDATE=1 needs VULKAN_SDK}/lib" \
      VK_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layer.d" VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
      VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true GPUSQZ_BACKEND=vulkan "$@"
  else
    VK_ICD_FILENAMES=$icd LP_NATIVE_VECTOR_WIDTH=$w GPUSQZ_BACKEND=vulkan "$@"
  fi
}

fail=0
for w in ${WIDTHS:-256 512 1024 2048}; do
  mode=$(lvp "$w" "$g" devices 2>&1 | grep -oE '(usable|NOT usable)[^,]*, [a-z-]+ lanes|NOT usable.*' | head -1)
  echo "== lavapipe width $w (subgroup $((w / 32))): ${mode:-?}"
  for f in text random; do
    for cs in ${CHUNKS:-8192 65536 1048576}; do
      in="$tmp/$f" vk="$tmp/vk.gsz" cu="$tmp/cu.gsz"
      res="$f chunk=$cs:"
      if ! lvp "$w" "$g" c "$in" "$vk" "$cs" 2> "$tmp/log"; then
        echo "  $res vulkan compress FAILED: $(tail -1 "$tmp/log")"; fail=1; continue
      fi
      lvp "$w" "$g" d "$vk" "$tmp/out" 2>> "$tmp/log" && cmp -s "$in" "$tmp/out" && res="$res vk->vk ok" || { res="$res vk->vk BAD"; fail=1; }
      "$ref" "$vk" "$tmp/out" 2>> "$tmp/log" && cmp -s "$in" "$tmp/out" && res="$res vk->refdec ok" || { res="$res vk->refdec BAD"; fail=1; }
      if [ $have_cuda = 1 ]; then
        GPUSQZ_BACKEND=cuda "$g" c "$in" "$cu" "$cs" 2>> "$tmp/log"
        cmp -s "$vk" "$cu" && res="$res same-as-cuda" || { res="$res DIFFERENT-FROM-CUDA"; fail=1; }
        GPUSQZ_BACKEND=cuda "$g" d "$vk" "$tmp/out" 2>> "$tmp/log" && cmp -s "$in" "$tmp/out" && res="$res vk->cuda ok" || { res="$res vk->cuda BAD"; fail=1; }
        lvp "$w" "$g" d "$cu" "$tmp/out" 2>> "$tmp/log" && cmp -s "$in" "$tmp/out" && res="$res cuda->vk ok" || { res="$res cuda->vk BAD"; fail=1; }
      fi
      if [ "${VALIDATE:-0}" = 1 ] && grep -q 'VUID\|SYNC-HAZARD\|Validation Error' "$tmp/log"; then
        res="$res VALIDATION-ERRORS"; fail=1; grep -m3 'VUID\|SYNC-HAZARD\|Validation Error' "$tmp/log" | sed 's/^/    /'
      fi
      echo "  $res"
    done
  done
done
[ $fail = 0 ] && echo "ALL VULKAN CHECKS PASSED" || { echo "SOME VULKAN CHECKS FAILED"; exit 1; }
