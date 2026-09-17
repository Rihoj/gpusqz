// Deterministic binary-STL test meshes for the format-aware transforms study.
//
//   genmesh <dir>   writes <dir>/ordered.stl and <dir>/shuffled.stl (9MB each)
//
// A parametric closed surface (a lumpy sphere, about 100mm across) with small
// per-vertex noise, 180,000 triangles. ordered.stl emits triangles in grid
// order, so neighbours share vertices within a few records; shuffled.stl holds
// the same triangles in a random order, standing in for meshes whose triangle
// order carries no spatial locality. Normals are computed in float32 from the
// float32 vertices, and the attribute bytes are zero.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng = 88172645463325252ULL;
static uint64_t next(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static double unit(void) { return (double)(next() >> 11) / 9007199254740992.0; }

static void tri(unsigned char *out, const float *a, const float *b, const float *c) {
  float u[3], v[3], n[3];
  for (int i = 0; i < 3; i++) { u[i] = b[i] - a[i]; v[i] = c[i] - a[i]; }
  n[0] = u[1] * v[2] - u[2] * v[1];
  n[1] = u[2] * v[0] - u[0] * v[2];
  n[2] = u[0] * v[1] - u[1] * v[0];
  float l = sqrtf(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (l > 0) { n[0] /= l; n[1] /= l; n[2] /= l; }
  memcpy(out, n, 12);
  memcpy(out + 12, a, 12);
  memcpy(out + 24, b, 12);
  memcpy(out + 36, c, 12);
  memset(out + 48, 0, 2);
}

static void write_stl(const char *path, const unsigned char *recs, uint32_t n) {
  FILE *f = fopen(path, "wb");
  if (!f) { perror(path); exit(1); }
  char hdr[80] = {0};
  snprintf(hdr, sizeof hdr, "gpusqz format-aware study mesh");
  fwrite(hdr, 1, 80, f);
  fwrite(&n, 4, 1, f);
  fwrite(recs, 50, n, f);
  fclose(f);
}

int main(int argc, char **argv) {
  if (argc != 2) { fprintf(stderr, "usage: genmesh <dir>\n"); return 1; }
  const int nu = 300, nv = 300;
  double *r = malloc(sizeof(double) * (nu + 1) * (nv + 1));
  for (int i = 0; i <= nu; i++)
    for (int j = 0; j <= nv; j++) {
      double th = M_PI * i / nu, ph = 2 * M_PI * j / nv;
      r[i * (nv + 1) + j] = 40.0 + 8.0 * sin(3 * th) * cos(2 * ph) + 4.0 * cos(5 * th + ph) +
                            0.35 * (unit() - 0.5);
    }
  uint32_t n = (uint32_t)(nu * nv * 2);
  unsigned char *recs = malloc((size_t)n * 50);
  size_t k = 0;
  for (int i = 0; i < nu; i++)
    for (int j = 0; j < nv; j++) {
      float p[4][3];
      int ii[4] = {i, i + 1, i + 1, i}, jj[4] = {j, j, j + 1, j + 1};
      for (int q = 0; q < 4; q++) {
        double th = M_PI * ii[q] / nu, ph = 2 * M_PI * jj[q] / nv, rr = r[ii[q] * (nv + 1) + jj[q]];
        p[q][0] = (float)(rr * sin(th) * cos(ph));
        p[q][1] = (float)(rr * sin(th) * sin(ph));
        p[q][2] = (float)(rr * cos(th) + 60.0);
      }
      tri(recs + 50 * k++, p[0], p[1], p[2]);
      tri(recs + 50 * k++, p[0], p[2], p[3]);
    }
  char path[4096];
  snprintf(path, sizeof path, "%s/ordered.stl", argv[1]);
  write_stl(path, recs, n);
  unsigned char t[50];
  for (uint32_t i = n; i-- > 1;) {
    uint32_t j = (uint32_t)(next() % (i + 1));
    memcpy(t, recs + 50 * (size_t)i, 50);
    memcpy(recs + 50 * (size_t)i, recs + 50 * (size_t)j, 50);
    memcpy(recs + 50 * (size_t)j, t, 50);
  }
  snprintf(path, sizeof path, "%s/shuffled.stl", argv[1]);
  write_stl(path, recs, n);
  return 0;
}
