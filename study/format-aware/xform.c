// Reversible (or geometry-only) transforms of a binary STL, applied chunk by
// chunk with every table reset at each chunk boundary, as gpusqz's chunk
// independence would require. The 84-byte header is copied first.
//
//   xform <in.stl> <mode> <chunk bytes, 0 = whole file> <out> [f64|f32]
//
// Modes (records per chunk = chunk bytes / 50):
//   t2   vertices only: normals and attribute bytes dropped (geometry-only)
//   t2x  vertices, normal residuals, attribute bytes (bit-exact, see unxform)
//   t3   t2, with x/y/z de-interleaved and split into byte planes (geometry-only)
//   t4   t3 over a per-chunk table of distinct vertices, plus delta-coded
//        vertex indices in byte planes (geometry-only)
//   t4x  t4, plus normal residuals and attribute bytes (bit-exact)
//
// A normal residual is the stored normal XOR the normal recomputed from the
// triangle's vertices, computed in float64 then cast (f64, the default) or in
// float32 (f32). The x modes print the share of residual words that are zero.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "normal.h"

typedef struct { unsigned char b[12]; } Vert;

static int32_t *table;
static Vert *verts;
static uint32_t nverts, table_bits;

static uint32_t hash12(const Vert *v) {
  uint32_t h = 2166136261u;
  for (int i = 0; i < 12; i++) { h ^= v->b[i]; h *= 16777619u; }
  return h;
}

static uint32_t intern(const Vert *v) {
  uint32_t mask = (1u << table_bits) - 1, h = hash12(v) & mask;
  while (table[h] >= 0) {
    if (!memcmp(verts[table[h]].b, v->b, 12)) return (uint32_t)table[h];
    h = (h + 1) & mask;
  }
  table[h] = (int32_t)nverts;
  verts[nverts] = *v;
  return nverts++;
}

// n 4-byte words at src (already grouped by field) -> 4 byte planes at dst.
static void byte_planes(const unsigned char *src, size_t n, unsigned char *dst) {
  for (size_t i = 0; i < n; i++)
    for (int b = 0; b < 4; b++) dst[b * n + i] = src[i * 4 + b];
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: xform <in.stl> t2|t2x|t3|t4|t4x <chunk bytes, 0 = whole file> <out> [f64|f32]\n");
    return 1;
  }
  const char *mode = argv[2];
  int exact = mode[strlen(mode) - 1] == 'x';
  int dedup = mode[1] == '4';
  int planes = mode[1] != '2';
  int recipe32 = argc > 5 && !strcmp(argv[5], "f32");

  FILE *f = fopen(argv[1], "rb");
  if (!f) { perror(argv[1]); return 1; }
  unsigned char hdr[84];
  if (fread(hdr, 1, 84, f) != 84) { fprintf(stderr, "short header\n"); return 1; }
  uint32_t nt;
  memcpy(&nt, hdr + 80, 4);
  unsigned char *recs = malloc((size_t)nt * 50);
  if (fread(recs, 50, nt, f) != nt) { fprintf(stderr, "not a binary STL of %u triangles\n", nt); return 1; }
  fclose(f);

  size_t per = (size_t)atoll(argv[3]) / 50;
  if (per == 0 || per > nt) per = nt;
  table_bits = 1;
  while ((1u << table_bits) < per * 6) table_bits++;
  table = malloc(sizeof(int32_t) << table_bits);
  verts = malloc(sizeof(Vert) * per * 3);
  unsigned char *grouped = malloc(per * 36), *out = malloc(per * 36);
  uint32_t *index = malloc(sizeof(uint32_t) * per * 3);
  unsigned char *resid = malloc(per * 12), *attr = malloc(per * 2);
  size_t zero_words = 0;

  FILE *o = fopen(argv[4], "wb");
  if (!o) { perror(argv[4]); return 1; }
  fwrite(hdr, 1, 84, o);
  for (size_t base = 0; base < nt; base += per) {
    size_t cn = nt - base < per ? nt - base : per;
    const unsigned char *r = recs + base * 50;
    if (exact)
      for (size_t i = 0; i < cn; i++) {
        const unsigned char *t = r + i * 50;
        unsigned char rn[12];
        recompute_normal(t + 12, recipe32, rn);
        for (int w = 0; w < 12; w++) resid[i * 12 + w] = t[w] ^ rn[w];
        for (int w = 0; w < 12; w += 4)
          if (!memcmp(resid + i * 12 + w, "\0\0\0\0", 4)) zero_words++;
        memcpy(attr + i * 2, t + 48, 2);
      }
    if (!planes) {
      for (size_t i = 0; i < cn; i++) fwrite(r + i * 50 + 12, 1, 36, o);
    } else if (!dedup) {
      // x of every vertex, then y, then z; then byte planes.
      for (int c = 0; c < 3; c++)
        for (size_t i = 0; i < cn * 3; i++) memcpy(grouped + (c * cn * 3 + i) * 4, r + (i / 3) * 50 + 12 + (i % 3) * 12 + c * 4, 4);
      byte_planes(grouped, cn * 9, out);
      fwrite(out, 1, cn * 36, o);
    } else {
      memset(table, 0xff, sizeof(int32_t) << table_bits);
      nverts = 0;
      for (size_t i = 0; i < cn * 3; i++) {
        Vert v;
        memcpy(v.b, r + (i / 3) * 50 + 12 + (i % 3) * 12, 12);
        index[i] = intern(&v);
      }
      for (int c = 0; c < 3; c++)
        for (uint32_t i = 0; i < nverts; i++) memcpy(grouped + (c * (size_t)nverts + i) * 4, verts[i].b + c * 4, 4);
      byte_planes(grouped, (size_t)nverts * 3, out);
      fwrite(&nverts, 4, 1, o);
      fwrite(out, 1, (size_t)nverts * 12, o);
      for (size_t i = cn * 3; i-- > 1;) index[i] -= index[i - 1];
      byte_planes((unsigned char *)index, cn * 3, out);
      fwrite(out, 1, cn * 12, o);
    }
    if (exact) {
      fwrite(resid, 1, cn * 12, o);
      fwrite(attr, 1, cn * 2, o);
    }
  }
  fclose(o);
  if (exact) fprintf(stderr, "normal residual words exactly zero: %.1f%%\n", 100.0 * zero_words / ((double)nt * 3));
  return 0;
}
