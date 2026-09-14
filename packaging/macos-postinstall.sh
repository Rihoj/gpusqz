#!/bin/sh
# postinstall script of the macOS .pkg. The package installs into
# /usr/local/gpusqz (gpusqz loads MoltenVK from ../lib next to itself, and
# a libMoltenVK.dylib in /usr/local/lib could clash with Homebrew's), then
# links the programs into /usr/local/bin, which is on every shell's PATH.
# $3 is the target volume's mount point ("/" for the startup disk).
root="${3%/}"
mkdir -p "$root/usr/local/bin"
for p in gpusqz gpusqz_refdec; do
  if [ -e "$root/usr/local/gpusqz/bin/$p" ]; then
    ln -sf "/usr/local/gpusqz/bin/$p" "$root/usr/local/bin/$p"
  fi
done
exit 0
