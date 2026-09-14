#!/usr/bin/env bash
# Compiles Vulkan shaders the way CMakeLists.txt does, to catch GLSL errors
# right after an edit.  check-shaders.sh <repo root> <src/vk/... path>
# An edited .comp compiles alone; an edited .glsl include recompiles every
# shader. Shaders using the lane-group layer build with LG_SUBGROUP=0 and 1
# (keep the list in sync with the foreach in CMakeLists.txt).
root=$1 rel=$2
lane_shaders=" parse_hist rans_encode decompress probe "

# The compiler: $GPUSQZ_GLSLANG, the one a build directory was configured
# with, then PATH and $VULKAN_SDK. None found: skip.
glslang="${GPUSQZ_GLSLANG:-}"
if [ -z "$glslang" ]; then
  for cache in "$root"/build*/CMakeCache.txt; do
    [ -f "$cache" ] || continue
    g=$(sed -n 's/^GPUSQZ_GLSLANG:[A-Z]*=//p' "$cache")
    if [ -x "$g" ]; then glslang=$g; break; fi
  done
fi
[ -n "$glslang" ] || glslang=$(command -v glslangValidator || command -v glslang || true)
if [ -z "$glslang" ] && [ -x "${VULKAN_SDK:-/nonexistent}/bin/glslangValidator" ]; then
  glslang="$VULKAN_SDK/bin/glslangValidator"
fi
[ -n "$glslang" ] || exit 0

case "$rel" in
  *.comp) shaders=("$root/$rel") ;;
  *) shaders=("$root"/src/vk/*.comp) ;;
esac

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
status=0
for s in "${shaders[@]}"; do
  name=$(basename "$s" .comp)
  variants=0
  case "$lane_shaders" in *" $name "*) variants="0 1" ;; esac
  for lg in $variants; do
    if ! out=$("$glslang" -V --target-env vulkan1.2 -DLG_SUBGROUP="$lg" -o "$tmp/$name.spv" "$s" 2>&1); then
      printf '%s.comp (LG_SUBGROUP=%s) does not compile:\n%s\n\n' "$name" "$lg" "$out" >&2
      status=2
    fi
  done
done
exit $status
