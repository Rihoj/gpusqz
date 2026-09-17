# Format-aware transforms study

gpusqz codes every file as a plain byte stream. Some file types carry
structure a byte-oriented match finder can't see: a binary STL mesh stores,
for every triangle, a normal that follows from its three vertices, and each
vertex again in every triangle that shares it. A reversible transform run
before the LZ parse can remove that redundancy, the way xz's BCJ filters or
Blosc's shuffle do for their data (see [Compression
survey](compression-survey.md) 3.6, 8.3). This study measures what such
transforms would gain on binary STL, whether they fit gpusqz's chunk
independence, and where the remaining gap to `zstd -19` comes from. It
changes nothing in gpusqz itself.

Measured 2026-09-17 on 593ae95 (format 2). All numbers are sizes, not
speeds, so GPU contention doesn't affect them (another process held 12GB of
VRAM at the time). The tools are in `study/format-aware/` and
`study/format-aware/run_study.sh` reproduces every number.

## Short answer

- **A bit-exact per-chunk transform shrinks mesh output 1.5–3.3x.**
  Storing each normal as its XOR against one recomputed from the vertices,
  plus a per-chunk table of distinct vertices, takes the `ratio` profile
  from 1.16x to 1.79x on a mesh whose triangles are in random order, and
  from 2.43x to 7.98x on the same mesh in grid order. Both steps are
  reversible to the byte and verified by round trip.
- **The normal residual alone is the simple first step.** It is
  stateless, per triangle and size-preserving, and takes `ratio` to 1.57x
  (random order) and 4.41x (grid order). When the recomputation matches
  the writer's arithmetic every residual is zero, which gives 1.69x and
  5.68x.
- **Triangle order decides most of the result, not the transform.** The
  same triangles compress to 1.16x in random order and 2.43x in grid
  order before any transform. Order changes how much every transform
  helps; none measured here helps on one order and hurts on the other,
  except byte planes.
- **Byte planes over the vertex stream are rejected.** They made grid-
  ordered output 2.34x larger at `ratio` by breaking the 50-byte record
  stride the match finder was using.
- **Most of `zstd -19`'s lead is chunk independence, not its parser.** On
  random-order meshes `zstd -19` over the whole file is 1.6–2.1x
  smaller than `zstd -19` run on independent 1MB pieces. Within the same pieces,
  gpusqz's `ratio` output is 7–11% larger than `zstd -19`'s. On
  grid-ordered meshes chunking costs almost nothing, and gpusqz on the raw
  file is 0.6% smaller than chunked `zstd -19`.

## Corpus

`genmesh` writes one deterministic binary STL of 180,000 triangles
(9,000,084 bytes): a closed, lumpy, sphere-like surface about 100mm across,
the scale of a 3D-printing model, with 0.35mm of per-vertex noise. It comes
in two orders:

- **ordered**: triangles in the surface grid's order, so a vertex's
  neighbouring uses sit within a few records.
- **shuffled**: the same triangles in random order, the worst case for
  locality. Real exporters fall somewhere between, depending on how they
  traverse the mesh.

Normals are computed in float32 from the float32 vertices, and the
attribute bytes are zero. Real meshes from photogrammetry or CAD exporters
were not available for this study; see the recommendation before relying on
the normal-residual numbers.

## Transforms

Each transform runs on the triangles of one chunk at a time, with every
table reset at the chunk boundary. Chunks are counted in input triangles
(64KB or 1MB of records), and the 84-byte STL header is copied through.

| name | per chunk, in order | exact? |
|---|---|---|
| `t2` | vertices only (36 bytes per triangle) | no: normals and attribute bytes dropped |
| `t2x` | vertices; normal residuals (12 bytes); attribute bytes (2) | yes, and size-preserving |
| `t3` | `t2` with x, y and z de-interleaved, then split into 4 byte planes | no |
| `t4` | table of distinct vertices (x, y, z grouped, byte planes); per-triangle vertex indices, delta-coded, byte planes | no |
| `t4x` | `t4`, then normal residuals and attribute bytes | yes |

A normal residual is the stored normal XOR a normal recomputed from the
triangle's vertices. The default recipe recomputes in float64 and casts to
float32 (`f64`). The `f32` recipe recomputes in float32, which is what this
corpus's writer did. `unxform` inverts `t2x` and `t4x`, and the study
checks both round trips on both orders, both recipes, and 64KB, 1MB and
whole-file chunks: every output is identical to its input.

`t2`, `t3` and `t4` are not admissible in gpusqz, whose contract is a
byte-exact round trip. They are measured to separate what each step is
worth.

## Results

Compression ratio (original size / output size) of the whole 9,000,084-byte
file. gpusqz's `speed` column uses the 64KB transform, `ratio` the 1MB one.

| transform | shuffled `speed` | shuffled `ratio` | ordered `speed` | ordered `ratio` |
|---|---|---|---|---|
| none | 1.11x | 1.16x | 2.06x | 2.43x |
| `t2x` (`f64` recipe) | 1.45x | 1.57x | 3.44x | 4.41x |
| `t2x` (`f32`, matching the writer) | 1.55x | 1.69x | 4.18x | 5.68x |
| `t4x` (`f64` recipe) | 1.64x | 1.79x | 5.49x | 7.98x |
| *`t2`, normals dropped* | *1.55x* | *1.70x* | *4.31x* | *5.69x* |
| *`t3`, byte planes* | *1.79x* | *1.79x* | *2.30x* | *2.43x* |
| *`t4`, vertex table, normals dropped* | *1.77x* | *1.94x* | *7.45x* | *11.86x* |

For comparison, on the untransformed file:

| codec | shuffled | ordered |
|---|---|---|
| `zstd -3` | 1.18x | 2.38x |
| `zstd -19` | 1.96x | 2.41x |
| `zstd -19 --long` | 1.96x | 2.41x |

### Normal residuals

With the `f64` recipe 51.9% of residual words are zero. The rest differ in
the low mantissa bits and cost 441KB at `ratio` on the shuffled mesh
(5,730,134 bytes against 5,288,725 for dropping normals outright). With the
`f32` recipe every residual is zero and the residual stream costs 25KB, so
the bit-exact transform keeps 99% of the gain of discarding normals.

How often the residual is zero depends on matching the arithmetic of the
program that wrote the file, which isn't recorded in an STL. A real
implementation would try a few recipes per chunk (float32; float64 then
cast; normals left as zero, which some exporters write) and record the one
used.

### Vertex table

The per-chunk table stores each distinct vertex once and refers to it by
index. The mesh reuses each vertex 5.96 times, but the table can only find
repeats inside a chunk. On the ordered mesh almost all of them are: at
`ratio` a per-chunk table (`t4`, 758,589 bytes) is only 5% larger
than a whole-file one (721,999). On the shuffled mesh most repeats cross a
boundary: 4,641,414 bytes per chunk against 2,143,575 for the whole file. A
whole-file table would break chunk independence, since every chunk would
reference it. It is in [Measured and
rejected](performance-history.md#measured-and-rejected).

### Byte planes

Splitting the raw vertex stream into byte planes (`t3`) is 5% smaller than
`t2` at `ratio` on the shuffled mesh, but 2.34x larger on the ordered one
(3,700,955 against 1,581,575 bytes). In record order, identical vertices
recur 12, 24 or 36 bytes apart. Planes spread those repeats across
different planes, where the match finder can no longer see them. Planes over
the distinct-vertex table in `t4` and `t4x` don't have that problem, as the
table holds no repeats. Byte planes over the raw stream are in [Measured and
rejected](performance-history.md#measured-and-rejected).

## Where `zstd -19`'s lead comes from

On the shuffled mesh `zstd -19` compresses the raw file 1.96x, against
1.16x for gpusqz's `ratio`. `--long` changes nothing: `zstd -19`'s 8MB
window already covers the 9MB file, and the long-distance matcher only looks
for matches of 64 bytes or more, while these repeats are 12-byte vertices.

To separate chunking from parsing, the study runs `zstd -19` on independent
1MB pieces of each stream, the same pieces gpusqz's `ratio` profile sees.

| stream | `zstd -19`, whole file | `zstd -19`, 1MB pieces | gpusqz `ratio` |
|---|---|---|---|
| shuffled, none | 4,587,830 | 7,232,605 | 7,757,956 |
| shuffled, `t2x` | 2,501,092 | 5,182,357 | 5,730,134 |
| shuffled, `t4x` (whole-file table for column 1) | 2,197,484 | 4,631,728 | 5,030,661 |
| ordered, none | 3,741,064 | 3,718,374 | 3,697,498 |
| ordered, `t2x` | 1,670,417 | 1,692,127 | 2,042,108 |
| ordered, `t4x` (whole-file table for column 1) | 922,109 | 978,293 | 1,127,883 |

- **Chunk independence costs 1.6–2.1x on shuffled meshes**, where most
  repeated vertices sit in other chunks, and 0–6% on ordered ones.
- **gpusqz's parser costs 7–11% on shuffled meshes** against `zstd -19`
  within the same pieces. On the raw ordered mesh gpusqz is 0.6% smaller.
  Once transformed, the ordered stream is 15–21% larger than chunked
  `zstd -19`: the dense vertex streams reward a stronger parser more than
  the raw records did.

A better parser (the levers in [Known limitations](limitations.md#ratio))
would recover at most that 7–21%. The rest of the gap on randomly ordered
meshes is the price of decoding chunks independently.

## What it would take in gpusqz

None of this is built.

- **Format 3.** The transform needs a per-chunk marker, such as new
  `ChunkFlag` values beside `Raw`, `Lz` and `LzRans`, or a transform byte
  after the flag. `src/format.h` and `tests/ref_decode.cpp` change in
  lockstep, with fixtures regenerated (the `/format-change` skill).
- **Detection per chunk, by trial.** Binary STL has no magic number, but
  `file size == 84 + 50 * triangle count` rejects nearly everything else. A
  wrong guess costs ratio, never correctness, because the transform is
  reversible and recorded per chunk. Trying transformed and plain for each
  chunk and keeping the smaller output is the same choice gpusqz already
  makes between raw and LZ.
- **Record alignment.** gpusqz chunks are byte ranges. With the 84-byte
  header, 50-byte records don't start on chunk boundaries, and 64KB isn't a
  multiple of 50. Each chunk would pass its partial leading and trailing
  records through untransformed, or the STL mode would choose a chunk size
  that is a whole number of records.
- **`t2x` fits today's buffers.** It is size-preserving (50 bytes in, 50
  out per triangle), so the decoder's per-chunk output size is unchanged.
  The inverse is one normal recomputation and an XOR per triangle, run
  after LZ decoding, with no state shared between lanes, so it is simple in
  both `src/kernels.cu` and `src/vk/*.comp`. The one hazard is floating
  point: both backends and `ref_decode.cpp` must produce the same normal
  bit for bit, so the recomputation has to forbid fused multiply-add and
  other contractions (`precise` in GLSL, `-ffp-contract=off` for the C++
  and CUDA compilers). Add, multiply, divide and square root are then
  exact IEEE 754 operations on every backend.
- **`t4x` needs a stored length and a table.** Its transformed size varies
  per chunk, so the decoder must be told how many bytes the LZ stage
  produces. The encoder needs a per-chunk hash table of vertices (the
  match finder's tables are the model). The decoder only gathers vertices
  by index.

## Recommendation

1. **Measure real meshes before building.** This corpus is synthetic.
   Before committing to format 3, take `xform` and gpusqz to a few real
   binary STLs from photogrammetry and CAD exporters, and record the
   triangle order they use and how often each normal recipe gives zero
   residuals. Those two facts decide between the low (1.5x) and high
   (3.3x) ends of the measured range.
2. **Build `t2x` first.** It is stateless, size-preserving and gets most
   of the dropped-normal gain once recipes are tried per chunk. It is the
   smallest format change that tests detection, per-chunk trial and record
   alignment end to end.
3. **Add the per-chunk vertex table (`t4x`) second.** It adds 14–81% on
   top of `t2x` at `ratio` (both with the `f64` recipe), but needs a variable transformed length and an
   encoder hash table.
4. **Don't pursue byte planes over raw records or whole-file tables.** The
   first breaks the stride the match finder relies on. The second breaks
   chunk independence.
5. **Treat STL as the first instance of fixed-stride float records.** The
   same structure (derived fields, repeated values, a fixed record size)
   occurs in glTF buffers, `.npy` arrays and point clouds. Detecting the
   stride automatically is a separate question not measured here.

## Tools

In `study/format-aware/`, built with plain `cc` (see `run_study.sh`), not
part of the CMake build or the release:

- `genmesh.c`: the deterministic ordered and shuffled test meshes.
- `xform.c`: the `t2`, `t2x`, `t3`, `t4` and `t4x` transforms at any chunk
  size, and the zero share of normal residuals.
- `unxform.c`: the inverse of `t2x` and `t4x`, for the round-trip check.
- `normal.h`: the `f64` and `f32` normal recipes shared by both.
