// The triangle normal an STL writer would store, recomputed from the three
// float32 vertices at v (36 bytes): in float64 then cast, or in float32.
#pragma once
#include <math.h>
#include <string.h>

static void recompute_normal(const unsigned char *v, int float32, unsigned char *out) {
  float a[3], b[3], c[3], n32[3];
  memcpy(a, v, 12);
  memcpy(b, v + 12, 12);
  memcpy(c, v + 24, 12);
  if (float32) {
    float u[3], w[3];
    for (int i = 0; i < 3; i++) { u[i] = b[i] - a[i]; w[i] = c[i] - a[i]; }
    n32[0] = u[1] * w[2] - u[2] * w[1];
    n32[1] = u[2] * w[0] - u[0] * w[2];
    n32[2] = u[0] * w[1] - u[1] * w[0];
    float l = sqrtf(n32[0] * n32[0] + n32[1] * n32[1] + n32[2] * n32[2]);
    if (l > 0) { n32[0] /= l; n32[1] /= l; n32[2] /= l; }
  } else {
    double u[3], w[3], n[3];
    for (int i = 0; i < 3; i++) { u[i] = (double)b[i] - a[i]; w[i] = (double)c[i] - a[i]; }
    n[0] = u[1] * w[2] - u[2] * w[1];
    n[1] = u[2] * w[0] - u[0] * w[2];
    n[2] = u[0] * w[1] - u[1] * w[0];
    double l = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (l > 0) { n[0] /= l; n[1] /= l; n[2] /= l; }
    for (int i = 0; i < 3; i++) n32[i] = (float)n[i];
  }
  memcpy(out, n32, 12);
}
