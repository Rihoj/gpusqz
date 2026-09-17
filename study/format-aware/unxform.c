// Inverse of the bit-exact xform modes: rebuilds the original binary STL.
//
//   unxform t2x|t4x <in> <chunk bytes, 0 = whole file> <out.stl> [f64|f32]
//
// The mode, chunk size and normal recipe must match the xform run.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "normal.h"

static void read_or_die(void *p, size_t size, size_t n, FILE *f) {
  if (fread(p, size, n, f) != n) { fprintf(stderr, "truncated input\n"); exit(1); }
}

// Inverse of xform's byte_planes: 4 byte planes at src -> n 4-byte words at dst.
static void unplane(const unsigned char *src, size_t n, unsigned char *dst) {
  for (size_t i = 0; i < n; i++)
    for (int b = 0; b < 4; b++) dst[i * 4 + b] = src[b * n + i];
}

int main(int argc, char **argv) {
  if (argc < 5) { fprintf(stderr, "usage: unxform t2x|t4x <in> <chunk bytes, 0 = whole file> <out.stl> [f64|f32]\n"); return 1; }
  int dedup = !strcmp(argv[1], "t4x");
  int recipe32 = argc > 5 && !strcmp(argv[5], "f32");
  FILE *f = fopen(argv[2], "rb");
  if (!f) { perror(argv[2]); return 1; }
  unsigned char hdr[84];
  read_or_die(hdr, 1, 84, f);
  uint32_t nt;
  memcpy(&nt, hdr + 80, 4);
  size_t per = (size_t)atoll(argv[3]) / 50;
  if (per == 0 || per > nt) per = nt;
  unsigned char *vert = malloc(per * 36), *resid = malloc(per * 12), *attr = malloc(per * 2);
  unsigned char *planes = malloc(per * 36), *words = malloc(per * 36);
  uint32_t *index = malloc(sizeof(uint32_t) * per * 3);
  FILE *o = fopen(argv[4], "wb");
  if (!o) { perror(argv[4]); return 1; }
  fwrite(hdr, 1, 84, o);
  for (size_t base = 0; base < nt; base += per) {
    size_t cn = nt - base < per ? nt - base : per;
    if (!dedup) {
      read_or_die(vert, 36, cn, f);
    } else {
      uint32_t nv;
      read_or_die(&nv, 4, 1, f);
      read_or_die(planes, 12, nv, f);
      unplane(planes, (size_t)nv * 3, words);  // x of every distinct vertex, then y, then z
      read_or_die(planes, 12, cn, f);
      unplane(planes, cn * 3, (unsigned char *)index);
      for (size_t i = 1; i < cn * 3; i++) index[i] += index[i - 1];
      for (size_t i = 0; i < cn * 3; i++)
        for (int c = 0; c < 3; c++) memcpy(vert + i * 12 + c * 4, words + (c * (size_t)nv + index[i]) * 4, 4);
    }
    read_or_die(resid, 12, cn, f);
    read_or_die(attr, 2, cn, f);
    for (size_t i = 0; i < cn; i++) {
      unsigned char rec[50];
      recompute_normal(vert + i * 36, recipe32, rec);
      for (int w = 0; w < 12; w++) rec[w] ^= resid[i * 12 + w];
      memcpy(rec + 12, vert + i * 36, 36);
      memcpy(rec + 48, attr + i * 2, 2);
      fwrite(rec, 1, 50, o);
    }
  }
  fclose(o);
  return 0;
}
