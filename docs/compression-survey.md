# How open-source compression tools optimize for speed and ratio

A survey of what open-source compressors, archivers, and backup and
filesystem tools do to compress faster, decompress faster, or compress
smaller. It covers single files and streams, multi-file archives, whole
directories, and GPU and hardware offload. Every finding lists its pros and
cons.

Researched September 2026.

## Contents

1. [Scope and conventions](#1-scope-and-conventions)
2. [Single-stream codecs: entropy coding](#2-single-stream-codecs-entropy-coding)
3. [Single-stream codecs: LZ modeling](#3-single-stream-codecs-lz-modeling)
4. [Single-stream codecs: match finding and parsing](#4-single-stream-codecs-match-finding-and-parsing)
5. [Single-stream codecs: implementation engineering](#5-single-stream-codecs-implementation-engineering)
6. [Other codec families: BWT, context mixing, PPM](#6-other-codec-families-bwt-context-mixing-ppm)
7. [Parallelism inside one file](#7-parallelism-inside-one-file)
8. [GPU and hardware offload](#8-gpu-and-hardware-offload)
9. [Archive formats: packaging many files](#9-archive-formats-packaging-many-files)
10. [Multi-file and directory techniques](#10-multi-file-and-directory-techniques)
11. [Cross-cutting trade-offs](#11-cross-cutting-trade-offs)
12. [Summary table](#12-summary-table)
13. [Relevance to gpusqz](#13-relevance-to-gpusqz)
14. [Unverified items and caveats](#14-unverified-items-and-caveats)

---

## 1. Scope and conventions

**Scope.** Only open-source tools and formats. There are two exceptions. RAR
and RAD's Oodle get a one-line mention where their design is the obvious
comparison. nvCOMP is included even though it stopped releasing source after
version 2.3, because it is the reference point for GPU compression.

**How findings were gathered.** Every finding comes from a primary source:
a format spec or RFC, a man page or manual, project source code, a README or
changelog, a design post by the author, or a paper. Version numbers and
throughput figures appear only where a fetched source states them, and the
source is linked next to each finding. Claims that could not be confirmed
are marked **[unverified]** and collected in section 14.

**Benchmark figures.** Unless a finding says otherwise, figures are from
lzbench 2.0.1 on one thread of an AMD EPYC 9554 (3.1 GHz), gcc 14.2, on
`silesia.tar` (<https://github.com/inikep/lzbench>). They are written as
`compress/decompress MB/s, ratio`, where ratio is compressed size as a
percentage of the original, so **lower is better**. The zstd README uses a
different machine and the opposite convention (original ÷ compressed, higher
is better). Numbers from the two sources are never mixed.

**Finding format.** Each finding follows the same template:

- **What:** the technique
- **Used by:** the tools that use it
- **Pros**
- **Cons**
- **Sources**

---

## 2. Single-stream codecs: entropy coding

Most general-purpose compressors have two stages: a modeling stage (usually
LZ77 matches plus literals) and an entropy coder that turns the symbols into
bits. The choice of entropy coder sets the ceiling on decode speed more than
anything else.

### 2.1 Huffman coding with table-driven decode

- **What:** Each symbol gets a code with a whole number of bits, and decoding
  is a table lookup on the next N bits. zlib's inflate uses a first-level
  table of about 9 bits, with overflow entries pointing to second-level
  tables. The first-level size balances table-fill cost against how often
  the second level is needed.
- **Used by:** Deflate (zlib, zlib-ng, libdeflate, igzip), bzip2, zstd
  literals, Brotli prefix codes.
- **Pros:**
  - Very fast decode: one lookup and a shift per symbol.
  - Tables are static per block, so the decoder never updates a model.
- **Cons:**
  - Code lengths are whole bits. A symbol with probability 0.9 still costs
    at least 1 bit, so ratio suffers on skewed data such as LZ match
    lengths or highly predictable literals.
- **Sources:** <https://raw.githubusercontent.com/madler/zlib/develop/doc/algorithm.txt>,
  <https://mattmahoney.net/dc/dce.html> §3.1

### 2.2 Multi-stream Huffman (zstd 4-stream literals, Huff0, Oodle 3/6-stream)

- **What:** The literal section is split into several independent bitstreams
  that are decoded in an interleaved loop, so an out-of-order CPU can overlap
  their dependency chains.
  - zstd uses 1 or 4 streams (RFC 8878) with a 6-byte jump table giving the
    sizes of streams 1–3. Codes are capped at 11 bits. The decoder picks
    single-symbol (X1) or double-symbol (X2) tables using a timing model.
  - zstd 1.5.1 added a BMI2 assembly Huffman decoder. Its release notes
    report about 40% faster decode.
  - Oodle Kraken (closed source; public design notes) uses 3 streams on
    Jaguar and 6 on x86-64. The 6 are two 3-stream groups, one read from the
    front and one from the back. Codes are limited to 11 bits so the table
    is 2048 × 2 bytes.
- **Used by:** zstd, Huff0/FiniteStateEntropy; Oodle for comparison.
- **Pros:**
  - Cheap instruction-level parallelism with almost no size cost. For
    Oodle on Zen 4, Giesen reports 1.37 vs 2.08 cycles per symbol, about
    3 GB/s per core.
  - Multi-symbol tables decode two symbols per lookup without branching.
- **Cons:**
  - A small header for the stream sizes.
  - The code-length cap costs a little optimality.
  - 4–6 streams is far too narrow for SIMD or GPU widths (see 8.9).
  - Register pressure grows with stream count.
- **Sources:** <https://www.rfc-editor.org/rfc/rfc8878.html>,
  <https://github.com/facebook/zstd/releases/tag/v1.5.1>,
  <https://github.com/Cyan4973/FiniteStateEntropy>,
  <https://fastcompression.blogspot.com/2015/10/huffman-revisited-part-4-multi-bytes.html>,
  <https://fgiesen.wordpress.com/2023/10/29/entropy-decoding-in-oodle-data-x86-64-6-stream-huffman-decoders/>

### 2.3 tANS / FSE (table-based asymmetric numeral systems)

- **What:** An ANS coder in which one integer state indexes a precomputed
  table. It gives fractional-bit precision using only adds, masks, and
  shifts, with no multiply or divide.
  - zstd codes literal lengths, match lengths, and offsets with FSE, and
    reads the bitstream backwards.
  - Each block chooses a mode per table: Predefined, RLE, FSE_Compressed, or
    Repeat (reuse the previous block's table).
- **Used by:** zstd; Oodle (closed source) and others.
- **Pros:**
  - Close to arithmetic-coding ratio on skewed distributions, at speed
    close to Huffman.
  - The FSE README (i7-5600U, 32 KB blocks, a highly skewed "Proba80"
    sample) shows FSE at ratio 8.84 vs Huff0 at 6.38. The same table shows
    the speed cost: 325/440 MB/s vs 600/1350 MB/s.
- **Cons:**
  - Building the table costs time and cache, so the coder suits static
    per-block statistics, not per-symbol adaptation.
  - Encoding runs in reverse (LIFO), so the encoder has to buffer.
  - Slower than Huffman when the distribution is flat.
- **Sources:** <https://fastcompression.blogspot.com/2013/12/finite-state-entropy-new-breed-of.html>,
  <https://github.com/Cyan4973/FiniteStateEntropy>, RFC 8878

### 2.4 rANS and interleaved rANS

- **What:** The range variant of ANS: one state, one multiply per symbol,
  and no large tables. Giesen's point is that the encoder and decoder states
  are symmetric. As a result, N coders can be *interleaved into one
  bitstream with no metadata*: each lane's renormalization reads and writes
  land in a predictable order.
  - `rans_word_sse41` decodes 4+ streams with SIMD at about 6 cycles/symbol,
    vs about 19–22 for scalar.
  - The "rANS in practice" post adds a two-state swap (about 1.4× faster), a
    32-bit state with 16-bit renormalization (which caps probabilities at
    14 bits), and a flush that reduces across GPU lanes in O(log N).
- **Used by:** ryg_rans (public domain), Oodle/BitKnit (closed source),
  DietGPU and Recoil (§8), and gpusqz.
- **Pros:**
  - Parallel and SIMD decode at almost no ratio cost.
  - Switching models is cheap, since there is no big table, which helps
    context modeling.
  - Giesen: with static probabilities it is "about ~2x faster than a
    typical arithmetic coder with comparable compression".
- **Cons:**
  - Needs a multiply per symbol.
  - Encoding is LIFO and must be buffered.
  - Each lane still decodes serially.
  - Precision is bounded by the state and word sizes.
- **Sources:** <https://fgiesen.wordpress.com/2014/02/02/rans-notes/>,
  <https://arxiv.org/abs/1402.3392>, <https://github.com/rygorous/ryg_rans>,
  <https://fgiesen.wordpress.com/2015/12/21/rans-in-practice/>

### 2.5 Adaptive binary range coding

- **What:** Every decision is coded as a bit with an adaptive probability.
  In LZMA each probability is 11 bits and moves toward the observed bit
  after every bit (`kNumMoveBits 5`). Multi-bit symbols are coded through
  bit trees.
- **Used by:** LZMA/LZMA2 (xz, 7z, lzip); the PAQ, cmix, and zpaq family;
  partly LZHAM.
- **Pros:**
  - The model adapts continuously and no tables are transmitted, so it
    compresses well, especially with rich contexts (see 3.2).
  - Can spend less than one bit per symbol.
- **Cons:**
  - Every bit is a data-dependent serial step, so decode is an order of
    magnitude slower than table-based coders. xz -6 decodes at 127 MB/s vs
    1347 MB/s for zstd -1.
  - lzip's manual notes that range-coded data "can't be split in pieces
    that could be described individually", so parallelism has to come from
    independent blocks (see 7.5).
- **Sources:** LZMA spec (mirror:
  <https://raw.githubusercontent.com/jljusten/LZMA-SDK/master/DOC/lzma-specification.txt>),
  <https://www.nongnu.org/lzip/manual/lzip_manual.html>, lzbench

### 2.6 No entropy stage: byte-aligned LZ tokens

- **What:** The LZ output is written as byte-aligned tokens with no entropy
  coding at all.
  - LZ4: each sequence is a 1-byte token (4-bit literal length, 4-bit match
    length, where the value 15 means extension bytes follow), then the
    literals, then a 2-byte offset (maximum 65535). The minimum match is 4.
  - Snappy uses tagged literal and copy elements with 1-, 2-, or 4-byte
    offsets.
- **Used by:** LZ4/LZ4HC, Snappy, LZO, Lizard's fastLZ4 modes, and the
  kernel's zram/zswap defaults.
- **Pros:**
  - Decoding is almost entirely memcpy and runs at RAM speed. lz4 1.10:
    577/3716 MB/s. snappy 1.2.1: 401/1077 MB/s.
  - Simple enough to decode one warp per chunk on a GPU (8.2).
- **Cons:**
  - Much weaker ratio: lz4 47.6% and snappy 47.9%, vs zstd -1 at 34.6%.
    Snappy's own README says it is 20–100% bigger than zlib's fastest
    mode.
- **Sources:** <https://raw.githubusercontent.com/lz4/lz4/dev/doc/lz4_Block_format.md>,
  <https://raw.githubusercontent.com/google/snappy/main/format_description.txt>,
  lzbench

---

## 3. Single-stream codecs: LZ modeling

These techniques change what the LZ stage emits or what it can refer to.
They affect ratio more than speed.

### 3.1 Repeat offsets (repcodes)

- **What:** The coder keeps a small list of recently used match distances,
  and a match can refer to a list slot instead of coding the distance
  again.
  - zstd keeps 3, in recency order.
  - LZMA keeps rep0–rep3 plus a "short rep" (a 1-byte match at rep0), each
    with its own context bits.
  - Brotli keeps a ring of 4 last distances plus 16 short codes for small
    variations on them.
  - zstd's match finders check the repcodes before doing hash lookups.
- **Used by:** zstd, LZMA/xz/lzip, Brotli, gpusqz.
- **Pros:**
  - Structured data (tables, fixed-size records, columns in text) reuses the
    same distance, which then costs a few bits instead of a full offset.
  - Checking a repcode is an almost free match candidate.
- **Cons:**
  - The encoder has to carry rep state through its parse, which makes
    optimal parsing harder.
  - The decoder gains a small serial dependency, since the current offset
    depends on earlier sequences.
- **Sources:** RFC 8878, LZMA spec, <https://www.rfc-editor.org/rfc/rfc7932.html>

### 3.2 Literal context modeling

- **What:** The coding of a literal is conditioned on the bytes around it.
  - LZMA selects the literal probability set from the top `lc` bits of the
    previous byte and the low `lp` bits of the position (xz defaults:
    lc=3, lp=0, pb=2). Right after a match it codes a "matched literal",
    using the byte at rep0 as extra context.
  - Brotli: each literal block type picks a context mode (LSB6, MSB6, UTF8,
    or Signed), which produces a context ID from 0 to 63. A context map then
    sends each ID to one of several Huffman codes. This is order-2-style
    modeling with static codes. It turns on at quality ≥5.
- **Used by:** LZMA family, Brotli, gpusqz (order-1 contexts chosen per
  batch).
- **Pros:**
  - Large ratio gains on text and structured binary data.
  - Brotli gets them while keeping Huffman-speed decode, because the codes
    are static.
- **Cons:**
  - More tables to send (Brotli) or more contexts to warm up (LZMA).
    Context dilution hurts small inputs.
  - Brotli has a `DISABLE_LITERAL_CONTEXT_MODELING` switch because the
    feature costs encoder time.
- **Sources:** LZMA spec, <https://raw.githubusercontent.com/tukaani-project/xz/master/src/xz/xz.1>,
  RFC 7932, <https://raw.githubusercontent.com/google/brotli/master/c/enc/quality.h>

### 3.3 Built-in static dictionary

- **What:** Brotli's format includes a fixed 122,784-byte dictionary of
  words and phrases (lengths 4–24), with 121 transforms per base word
  (capitalization, suffixes, and so on). A distance beyond the current
  window refers into this dictionary.
- **Used by:** Brotli.
- **Pros:**
  - Small web payloads compress well from the first byte, with nothing
    extra to transmit.
- **Cons:**
  - Tuned for web text and little help on other data.
  - Fixed by the format forever.
  - Adds 120 KB to every decoder.
- **Sources:** RFC 7932

### 3.4 Trained external dictionaries

- **What:** The compressor and decompressor both preload a dictionary built
  from sample data.
  - zstd's `--train` uses the COVER or fastCover algorithms. The default is
    fastCover with d=8 and steps=4; the default maximum dictionary size is
    112,640 bytes. The recommendation is at least 100 samples totalling
    about 100× the dictionary size.
  - A zstd dictionary contains entropy tables as well as content, which
    matters for tiny inputs.
  - Frames carry a dictionary ID so a mismatch can be detected.
- **Used by:** zstd, LZ4 (official since 1.10.0), and Deflate preset
  dictionaries. Section 10.3 covers system-level use: RocksDB, and HTTP
  Compression Dictionary Transport.
- **Pros:**
  - Large gains on small, similar records. Meta reports halving one
    service's storage and up to 40% more cache capacity.
  - Costs nothing at decode time beyond loading the dictionary.
- **Cons:**
  - zdict.h says gains are "mostly effective in the first few KB" and not
    expected past about 100 KB.
  - Dictionaries have to be distributed and versioned, and they go stale as
    the data drifts.
  - There is "no universal dictionary".
- **Sources:** <https://raw.githubusercontent.com/facebook/zstd/dev/lib/zdict.h>,
  <https://engineering.fb.com/2018/12/19/core-infra/zstandard/>,
  <https://github.com/lz4/lz4/releases/tag/v1.10.0>

### 3.5 Large windows and long-distance matching (LDM)

- **What:** A separate, sparse match finder with its own hash table (in
  zstd: ldmHashLog, and ldmMinMatch defaulting to 64) finds long matches far
  back in the input.
  - zstd `--long` defaults to windowLog 27 (128 MB); the maximum is 31
    (2 GB). LDM also switches on automatically at level 16+ with a window
    ≥128 MB.
  - Brotli's large-window mode allows lgwin up to 30 but is not RFC 7932
    compatible.
  - xz -9 uses a 64 MiB dictionary; lzip allows up to 512 MiB.
- **Used by:** zstd, Brotli, xz, lzip.
- **Pros:**
  - Catches redundancy between widely separated parts of an input, such as
    repeated files in a tarball, without any external dedup.
  - Meta reports 16% smaller full backups and 27% smaller diff backups.
- **Cons:**
  - Memory on both sides. zstd windows above 27 need `--long` or `--memory`
    passed to the *decompressor* as well.
  - With multithreading, memory is roughly window × threads.
  - Interacts badly with `--rsyncable`.
- **Sources:** <https://raw.githubusercontent.com/facebook/zstd/dev/programs/zstd.1.md>,
  <https://raw.githubusercontent.com/facebook/zstd/dev/lib/zstd.h>,
  <https://github.com/facebook/zstd/releases/tag/v1.3.2>

### 3.6 Reversible pre-filters (BCJ, delta)

- **What:** A reversible transform applied before LZ.
  - BCJ filters convert relative branch and call targets in machine code to
    absolute addresses, so repeated calls to one function become repeated
    byte strings. xz's man page says this makes output "0–15% smaller".
    Filters exist for x86, ARM, ARM-Thumb, ARM64, PowerPC, IA-64, SPARC,
    and RISC-V.
  - The delta filter subtracts the byte N positions back, for PCM audio and
    uncompressed bitmaps.
  - xz allows at most 4 filters in a chain.
- **Used by:** xz/liblzma, 7-Zip (it picks filters by extension or by parsing
  the file; see 9.8), SquashFS (`-Xbcj`), zpaq (E8E9). lzip deliberately has
  none.
- **Pros:**
  - Cheap, often large gains on the data type each filter targets.
  - Output size is unchanged, so the cost is one linear pass.
- **Cons:**
  - Only helps the matching data type and can hurt other data.
  - Needs file-type detection or user knowledge.
  - Adds format complexity: xz's filter chains, 7z's four-stream BCJ2.
- **Sources:** xz.1, <https://raw.githubusercontent.com/tukaani-project/xz/master/doc/xz-file-format.txt>,
  <https://www.7-zip.org/history.txt>

---

## 4. Single-stream codecs: match finding and parsing

This is where compression *speed* is mostly decided. Decode speed hardly
depends on how hard the encoder searched.

### 4.1 One format, a ladder of strategies (zstd)

- **What:** zstd has nine strategies, each tuned by windowLog, chainLog,
  hashLog, searchLog, minMatch, and targetLength. For inputs over 256 KB the
  levels map to strategies as follows. The window grows from 2^19 bytes at
  level 1 to 2^27 at level 22.

  | Level | Strategy |
  |---|---|
  | 1–2 | fast |
  | 3–4 | dfast |
  | 5 | greedy |
  | 6–7 | lazy |
  | 8–12 | lazy2 |
  | 13–15 | btlazy2 |
  | 16–17 | btopt |
  | 18 | btultra |
  | 19–22 | btultra2 |

- **Used by:** zstd. LZ4 (fast, HC, opt) and libdeflate (levels 1–12) use
  the same idea.
- **Pros:**
  - One decoder covers the whole speed/ratio curve, and decode speed is
    "roughly the same at all settings".
  - zstd 1.5.6 at -1: 422/1347 MB/s, 34.6%. At -22: 2.08/1073 MB/s, 24.7%.
- **Cons:**
  - The top levels are 200× slower to compress and need a lot of memory.
    Levels 20+ require `--ultra`, and the decoder also needs more memory
    for them.
- **Sources:** <https://raw.githubusercontent.com/facebook/zstd/dev/lib/compress/clevels.h>,
  zstd.1.md, lzbench

### 4.2 Sparse sampling and "acceleration"

- **What:** Below the normal fast level, the match finder looks at fewer
  positions.
  - zstd `--fast=N` (negative levels): only some positions are inserted
    into the hash table, and the step size grows the longer no match is
    found.
  - LZ4's `acceleration` parameter skips positions in the same way, "each
    successive value providing roughly +~3% to speed". Its default hash
    table is 16 KB.
- **Used by:** zstd, LZ4, and ZFS `zstd-fast-N`.
- **Pros:**
  - Faster than level 1. Decode also speeds up, because there are fewer,
    longer literal runs.
  - zstd README (i7-9700K): `--fast=1` runs at 545/1850 MB/s vs 510/1550 for
    -1.
  - Matters most on incompressible input, where the skip grows quickly.
- **Cons:**
  - Ratio drops fast. In the same README, ratio (higher is better) falls
    from 2.896 at -1 to 2.439 at `--fast=1`.
- **Sources:** zstd.h, <https://raw.githubusercontent.com/facebook/zstd/dev/lib/compress/zstd_fast.c>,
  <https://raw.githubusercontent.com/lz4/lz4/dev/lib/lz4.h>

### 4.3 SIMD row-based match finder

- **What:** Used by zstd's greedy, lazy, and lazy2 strategies since v1.5.0.
  The hash table is split into rows of 16, 32, or 64 entries, each with an
  8-bit tag. One SIMD compare (SSE2, NEON, or RVV) of the tag row produces a
  bitmask of candidates, so the finder checks only tag hits instead of
  chasing pointers.
- **Used by:** zstd levels 5–12.
- **Pros:**
  - The v1.5.0 notes report, on silesia.tar, +25% compression speed at
    level 5, +50% at 6, and up to +110% at 12. On loaded machines they
    report 2–3× at levels 5–7.
- **Cons:**
  - Needs SSE2-class SIMD for the full gain.
  - Bounded rows can miss candidates a deep chain would find, which costs a
    little ratio.
- **Sources:** <https://github.com/facebook/zstd/releases/tag/v1.5.0>,
  <https://raw.githubusercontent.com/facebook/zstd/dev/lib/compress/zstd_lazy.c>

### 4.4 Hash chains vs binary trees

- **What:** Two ways to find earlier occurrences of the current string.
  - A hash chain links earlier positions that share a hash of the first
    3–4 bytes and walks the chain up to a depth limit. zlib's depth is 4 at
    level 1, 128 at level 6, and 4096 at level 9.
  - A binary tree keeps earlier positions sorted by suffix, so it finds all
    longer matches in about log time. Optimal parsers need that complete
    set.
  - xz uses hc3/hc4 at -0 to -3 and bt4 above. bt4 needs about 11.5× the
    dictionary size in memory, hc4 about 7.5×.
- **Used by:** zlib, LZ4HC, libdeflate levels 2–9 (chains) and 10–12
  (trees), xz, zstd lazy (chains) and bt* (trees).
- **Pros:**
  - Hash chains are cheap to insert into and bounded by depth.
  - Binary trees give complete, sorted match sets.
- **Cons:**
  - Hash chains degrade on highly repetitive data (long chains, same
    prefix).
  - Binary trees cost more memory and more work per insert.
- **Sources:** xz.1, <https://raw.githubusercontent.com/madler/zlib/develop/deflate.c>

### 4.5 Lazy matching

- **What:** After finding a match at position p, the compressor also checks
  p+1 (and p+2 for lazy2). If that match is longer, it emits a literal and
  takes the later match. zlib tunes this per level with `max_lazy`,
  `good_length`, `nice_length`, and `max_chain`.
- **Used by:** zlib levels 4–9, zlib-ng, libdeflate 5–7, zstd lazy/lazy2.
- **Pros:**
  - A cheap ratio gain over greedy parsing.
- **Cons:**
  - Still a heuristic.
  - In Deflate, the 32 KB window and 258-byte match cap limit ratio no
    matter how hard the search works.
- **Sources:** algorithm.txt, deflate.c

### 4.6 Price-based optimal parsing

- **What:** The encoder builds a graph of literal and match choices over a
  window of positions, then finds the cheapest path through it using bit
  costs estimated from the current entropy statistics. Variants:
  - zstd btopt uses whole-bit prices; btultra uses fractional prices.
  - btultra2 makes an extra first pass over the first block just to seed
    the statistics.
  - libdeflate repeats the minimum-cost-path pass, re-estimating costs each
    time.
  - Brotli uses "zopfli-fication" at q10–11.
- **Used by:** xz/LZMA "normal" mode, lzip, zstd 16–22, libdeflate 8–12,
  LZ4HC 10–12 (`lz4opt`, with an optional `favorDecSpeed`), Brotli q10–11,
  LZHAM.
- **Pros:**
  - The best ratio a given format can express, with no decode cost.
  - libdeflate -12 gets 30.5% vs zlib -9's 31.9%, producing standard
    Deflate that any decoder reads.
- **Cons:**
  - Compression is 10–100× slower. libdeflate -12: 5.1 MB/s. lz4hc -12:
    10.5 MB/s. brotli -11: 0.58 MB/s.
  - More memory.
  - Hard to parallelize within a block.
- **Sources:** <https://raw.githubusercontent.com/tukaani-project/xz/master/src/liblzma/lzma/lzma_encoder_optimum_normal.c>,
  <https://raw.githubusercontent.com/facebook/zstd/dev/lib/compress/zstd_opt.c>,
  <https://raw.githubusercontent.com/ebiggers/libdeflate/master/lib/deflate_compress.c>,
  quality.h, lzbench

### 4.7 Block splitting and entropy table reuse

- **What:** The encoder starts a new block with fresh Huffman or FSE tables
  when the symbol statistics shift, and reuses the previous tables when
  they have not.
  - libdeflate compares the observed and expected distributions every N
    symbols.
  - Brotli uses block types, with splitting from q4.
  - zstd 1.5.0+ splits blocks at level 16+ (+0.49% ratio on silesia at
    level 22). zstd's "Repeat" (FSE) and "Treeless" (Huffman) modes reuse
    the previous block's tables.
  - bzip2 keeps 2–6 Huffman tables and chooses one every 50 symbols.
- **Used by:** libdeflate, Brotli, zstd, bzip2.
- **Pros:**
  - Adapts to mixed content without per-symbol adaptive coding, so decode
    stays fast.
  - Table reuse saves header bytes on small blocks.
- **Cons:**
  - The split search costs encoder time.
  - Gains are small on homogeneous data.
- **Sources:** libdeflate deflate_compress.c, quality.h, zstd v1.5.0 notes,
  RFC 8878

---

## 5. Single-stream codecs: implementation engineering

Same formats, faster code. These gains come without any format change.

### 5.1 zlib-ng: SIMD kernels, runtime dispatch, new strategies

- **What:** zlib-ng picks the best available code path for the CPU at
  runtime.
  - SIMD kernels for the hot loops: Adler-32 (SSSE3 through
    AVX-512-VNNI, NEON), CRC-32 ((V)PCLMULQDQ, PMULL), `slide_hash`,
    `compare256` (the heart of longest_match), and inflate's chunked copy.
  - Level 1 is `deflate_quick`: fixed Huffman tables and a single hash
    probe.
  - Levels 3–6 use `deflate_medium`, which came from Intel.
  - IBM Z DFLTCC hardware is also supported.
- **Used by:** zlib-ng, which many distros ship as a zlib replacement.
- **Pros:**
  - At level 6: 62.1/509 MB/s vs stock zlib 1.3.1's 25.3/344, at
    32.5% vs 32.2%.
  - A drop-in API (in compat mode).
- **Cons:**
  - Level 1 compresses much worse (44.4% vs 36.5%) because of the fixed
    Huffman tables.
  - Output is not byte-identical to zlib, which breaks tests and caches
    that hash compressed output.
- **Sources:** <https://github.com/zlib-ng/zlib-ng/blob/develop/README.md>, lzbench

### 5.2 libdeflate: whole-buffer API

- **What:** libdeflate drops streaming. "Only full-buffer decompression is
  supported", so the decoder never has to save and resume state. That lets
  it use word-sized reads and copies, a larger bit buffer, fewer branches,
  and a runtime-selected BMI2 build. Compression levels go up to 12
  (near-optimal parsing, §4.6).
- **Used by:** libdeflate, which is embedded in many tools.
- **Pros:**
  - Decode at 860–919 MB/s vs zlib's 323–348.
  - -6 compresses at 84.3 MB/s and 31.9%, vs zlib -6 at 25.3 MB/s and
    32.2%: faster and smaller.
- **Cons:**
  - No streaming at all. You need the whole input and an output bound.
  - The README pitches it for chunks under about 1 MB.
- **Sources:** <https://github.com/ebiggers/libdeflate/blob/master/README.md>, lzbench

### 5.3 Intel ISA-L igzip

- **What:** Deflate written in assembly for x86, with only four levels
  (0–3). Features by release:
  - v2.23: multi-symbol decode tables that hold up to three symbols per
    entry.
  - Custom Huffman tables built from histograms, and a stateless one-shot
    API.
  - v2.27: multithreaded compression in the CLI.
- **Used by:** ISA-L, python-isal, and some storage stacks.
- **Pros:**
  - Very high compression throughput on x86.
  - Standard Deflate output.
- **Cons:**
  - Ratio tops out around zlib -3.
  - The level-3 fast path is reportedly AVX-512-only **[unverified]**.
  - x86-centric.
- **Sources:** <https://raw.githubusercontent.com/intel/isa-l/master/Release_notes.txt>,
  <https://raw.githubusercontent.com/intel/isa-l/master/include/igzip_lib.h>

### 5.4 Cloudflare's zlib fork

- **What:** A zlib fork tuned for modern x86:
  - 64-bit types
  - the SSE4.2 CRC32 instruction as the match hash
  - a 4-byte minimum match
  - SIMD window sliding
  - PCLMULQDQ for the CRC-32 checksum
  - an optimized longest_match
- **Used by:** cloudflare/zlib.
- **Pros:**
  - The author reports about 2.4× faster at level 9 on Silesia.
  - AWS measured level-6 compression about 113% faster on x86 and about 90%
    faster on Arm.
- **Cons:**
  - Needs SSE4.2/PCLMUL-class CPUs.
  - Output differs from zlib.
  - Its current maintenance status was not confirmed.
- **Sources:** <https://blog.cloudflare.com/cloudflare-fights-cancer/>,
  <https://aws.amazon.com/blogs/opensource/improving-zlib-cloudflare-and-comparing-performance-with-other-zlib-forks>

### 5.5 Decode-speed-first LZ variants

- **What:** Codecs that give up ratio for decode speed.
  - **LZO:** "extremely fast decompression", with 64 KiB of compressor
    work memory. LZO1X-999 compresses slowly but decodes just as fast, and
    supports in-place decompression. -999: 7.13/658 MB/s, 35.5%.
  - **LZ4HC:** the LZ4 format with harder searches: level 2 uses two hash
    tables, 3–9 hash chains, and 10–12 optimal parsing. It decodes with the
    same fast decoder (lz4hc -12: 10.5/3616 MB/s, 36.5%). v1.10.0 added
    multithreading, cutting level-12 time on silesia from 16.2 s to 3.05 s.
  - **LZHAM:** "ratio similar to LZMA but with 1.5x–8x faster
    decompression" (3.08/309 MB/s, 25.8%).
  - **Lizard (formerly LZ5):** its own README now calls it "outdated and
    effectively obsolete".
- **Used by:** embedded systems, game assets, kernels.
- **Pros:**
  - You can compress once, slowly, and decode very fast everywhere.
- **Cons:**
  - Format fragmentation.
  - Several of these (LZO, Lizard, LZHAM) are no longer actively developed
    or have been superseded by zstd and LZ4.
- **Sources:** <https://www.oberhumer.com/opensource/lzo/>,
  <https://github.com/lz4/lz4/releases/tag/v1.10.0>,
  <https://github.com/richgel999/lzham_codec>,
  <https://github.com/inikep/lizard>, lzbench

---

## 6. Other codec families: BWT, context mixing, PPM

### 6.1 BWT block sorting (bzip2)

- **What:** The pipeline is Burrows–Wheeler transform, then move-to-front,
  then run-length coding, then Huffman with multiple tables.
  - Blocks are 100–900 KB and fully independent.
  - Each block starts with a 48-bit magic number (0x314159265359).
- **Used by:** bzip2, pbzip2, lbzip2.
- **Pros:**
  - Good ratio for its speed on text: bzip2 -9 gets 25.8% vs zlib -9's
    31.9%.
  - Blocks are independent, so compression parallelizes trivially, and
    decompression can too (see 7.6).
- **Cons:**
  - Slow and roughly symmetric: 13.1/37.5 MB/s.
  - The 900 KB block caps how much context BWT can see.
  - The sort is a bad fit for GPUs. A GPU bzip2 pipeline (Patel et al.
    2012) ended up slower than CPU bzip2 (8.8).
- **Sources:** <https://sourceware.org/bzip2/manual/manual.html>, lzbench

### 6.2 Context mixing (PAQ, cmix, zpaq high methods)

- **What:** The coder predicts one *bit* at a time. Tens to thousands of
  models each make a prediction, a neural-network mixer combines them, SSE
  or APM stages refine the result, and an arithmetic coder codes the bit.
  cmix v21 uses 2,077 models plus an LSTM mixer.
- **Used by:** PAQ8 variants, cmix, zpaq (methods 4–5).
- **Pros:**
  - The best ratios available. cmix compresses enwik9 to 107,963,380 bytes.
- **Cons:**
  - Fully symmetric and extremely slow: cmix took 622,950 s (about 7 days)
    for enwik9 and recommends at least 32 GB of RAM.
  - The decoder has to rerun every model on every bit, so there is no cheap
    decode path.
- **Sources:** <https://mattmahoney.net/dc/dce.html> §4.3,
  <https://www.byronknoll.com/cmix.html>

### 6.3 PPM (PPMd)

- **What:** Byte-wise prediction by partial matching. The coder predicts the
  next byte from the longest matching context it has seen, and falls back
  to shorter contexts through escape symbols.
- **Used by:** 7-Zip (PPMd, about variant H per 7-Zip's page), ZIP method
  98, RAR.
- **Pros:**
  - Strong on text (ppmd8 -4: 24.2%).
  - Mahoney: often faster than equivalent bit-wise context-mixing models.
- **Cons:**
  - Symmetric and slow (13.2/12.0 MB/s).
  - Memory-hungry.
  - Poor on binary data.
- **Sources:** dce.html, <https://www.7-zip.org/7z.html>, lzbench

---

## 7. Parallelism inside one file

The central tension: **every independence boundary you add for parallelism
throws away the context (window, entropy statistics, model state) built up
before it.** Tools differ in which direction they parallelize (compress,
decompress, or both) and in how much context they give up.

### 7.1 Parallel compression with a primed window (pigz)

- **What:** pigz cuts the input into 128 KB chunks and compresses them in
  parallel.
  - Each chunk is primed with "the last 32K of the previous block" as a
    preset dictionary.
  - Each piece ends with an empty stored block, so the pieces join on byte
    boundaries.
  - CRCs are computed per chunk and combined.
  - The result is one ordinary gzip stream.
- **Used by:** pigz. `-i/--independent` turns off the priming.
- **Pros:**
  - Near-linear compression scaling with almost no ratio loss.
  - Standard output that any gunzip reads.
- **Cons:**
  - "Decompression can't be parallelized." A Deflate block's start position
    and its 32 KB window depend on everything before it, so pigz decodes on
    one thread (with helper threads for reading, writing, and the check).
  - Output matches pigz's single-threaded output only at level 4 and above
    (see 10.10).
- **Sources:** <https://raw.githubusercontent.com/madler/pigz/master/pigz.c>

### 7.2 Dependent jobs with overlap in one frame (zstd -T)

- **What:** zstd's multithreaded mode splits the input into jobs, by default
  about 4× the window size.
  - Each job reloads a tail of the previous job as its prefix. `overlapLog`
    runs from 1 to 9, where 9 means the full window.
  - The output is a single frame.
- **Used by:** zstd `-T`.
- **Pros:**
  - Meta: "nearly linearly speed up compression per core, with almost no
    loss of ratio".
  - Output does not depend on the thread count (10.10).
- **Cons:**
  - Decode is single-threaded. The maintainer's position is that one-thread
    decode is fast enough (usually faster than an SSD). Later commenters
    point out that PCIe 4/5 SSDs now outpace it.
  - Small files get no parallelism, because jobs have a 512 KB minimum.
- **Sources:** zstd.h, zstd.1.md,
  <https://raw.githubusercontent.com/facebook/zstd/dev/lib/compress/zstdmt_compress.c>,
  <https://github.com/facebook/zstd/issues/2470>

### 7.3 Independent frames plus a size index (pzstd, seekable zstd)

- **What:** The input is compressed as independent zstd frames.
  - pzstd puts a 12-byte skippable frame holding each frame's size before
    the frame, so a decoder can split the work without decompressing
    anything.
  - The seekable format puts a seek table in a skippable frame at the end
    (magic 0x184D2A5E; footer magic 0x8F92EAB1), which also gives random
    access.
- **Used by:** pzstd, `contrib/seekable_format`, and many tools that build
  on them.
- **Pros:**
  - Parallel decode and random access.
  - Still valid zstd: ordinary decoders skip the skippable frames.
- **Cons:**
  - Each frame starts with an empty window, so ratio drops as frames
    shrink.
  - Only files written this way decode in parallel.
- **Sources:** <https://raw.githubusercontent.com/facebook/zstd/dev/contrib/pzstd/README.md>,
  <https://github.com/facebook/zstd/blob/dev/contrib/seekable_format/zstd_seekable_compression_format.md>

### 7.4 Independent members (plzip, pbzip2) and independent blocks (LZ4 frame)

- **What:** Each piece is a complete, self-contained stream.
  - plzip splits input into members, by default twice the dictionary size
    (1 MiB at -0), and decodes one member per thread.
  - pbzip2 writes multiple bzip2 streams.
  - The LZ4 frame format has a Block Independence flag (block sizes
    64 KB–4 MB). When set, blocks drop the 64 KB history between them.
- **Used by:** plzip, pbzip2, LZ4 frame.
- **Pros:**
  - Parallel in both directions, in a standard format.
- **Cons:**
  - plzip files are "0.4 to 2 percent larger"; pbzip2 "typically under
    0.2%".
  - Files made by the serial tools (single-member lzip, standard bzip2 for
    pbzip2) get no speedup.
  - Below about 1 MiB of input there is nothing to parallelize.
- **Sources:** <https://www.nongnu.org/lzip/manual/plzip_manual.html>,
  <https://linux.die.net/man/1/pbzip2>,
  <https://raw.githubusercontent.com/lz4/lz4/dev/doc/lz4_Frame_format.md>

### 7.5 Independent blocks with sizes in the headers (xz -T, LZMA2 chunks)

- **What:** xz's threaded mode compresses independent blocks. The default
  block size is 3× the dictionary size or 1 MiB, whichever is larger.
  - Block sizes are written into the block headers, which is what lets the
    threaded decoder added in xz 5.4.0 work. Threaded mode became the
    default in 5.6.0.
  - The `.xz` index also allows random access.
  - Inside a block, LZMA2 wraps LZMA in chunks of at most 64 KiB
    compressed and 2 MiB uncompressed. Each chunk can be stored raw or can
    reset the dictionary, state, or properties.
- **Used by:** xz/liblzma, 7-Zip.
- **Pros:**
  - Parallel compression and decompression in a standard format.
  - LZMA2 stores incompressible chunks raw instead of expanding them.
- **Cons:**
  - The man page: "Single-threaded compressor will give the smallest file
    size but only the output from the multi-threaded compressor can be
    decompressed using multiple threads". `-T1` files never decode in
    parallel.
  - Memory use rises significantly.
  - tukaani.org reports a bug in the threaded decoder in 5.3.3alpha–5.8.0.
- **Sources:** <https://raw.githubusercontent.com/tukaani-project/xz/master/NEWS>, xz.1,
  xz-file-format.txt, <https://tukaani.org/xz/>

### 7.6 Parallel decode of files written by serial tools (lbzip2, rapidgzip)

- **What:** Parallel decompression of ordinary files that were never written
  for it.
  - **lbzip2:** bzip2 blocks share no state, so scanner threads search for
    the 48-bit block magic, and a sequential step confirms each candidate
    (false positives are "usually very small (below 1e-14)").
  - **rapidgzip:** decodes arbitrary gzip in parallel by *guessing* where
    Deflate blocks start. It fills the unknown 32 KB window with unique
    15-bit markers and resolves them once the earlier chunk has been
    decoded. It builds on pugz, which only handled bytes in the range
    9–126.
- **Used by:** lbzip2, rapidgzip / indexed_gzip-style tools.
- **Pros:**
  - Works on the files that already exist.
  - rapidgzip reports 5.6 GB/s on Silesia with 128 cores, 33× GNU gzip.
- **Cons:**
  - Speculative work is wasted.
  - Needs many cores to beat a good serial decoder.
  - The worst case falls back to serial speed.
- **Sources:** <https://github.com/kjn/lbzip2/blob/master/src/parse.c>,
  <https://arxiv.org/abs/2308.08955>

### 7.7 Blocked gzip with virtual offsets (BGZF)

- **What:** A series of gzip members, each at most 64 KiB both compressed
  and uncompressed.
  - A "BC" extra field stores each block's size.
  - A virtual offset `coffset<<16 | uoffset` addresses any byte, which lets
    `.bai`, `.gzi`, and tabix indexes point into the file.
  - A fixed 28-byte empty block marks EOF, so truncation can be detected.
- **Used by:** htslib/bgzip, BAM, VCF, tabix (bioinformatics).
- **Pros:**
  - Any gzip reader can decompress it.
  - Fine-grained random access.
  - Parallel in both directions (`bgzip -@`, `bgzf_mt`).
- **Cons:**
  - The 64 KiB blocks with no shared history cost ratio compared with plain
    gzip.
- **Sources:** <https://raw.githubusercontent.com/samtools/hts-specs/master/SAMv1.tex>,
  <https://www.htslib.org/doc/bgzip.html>

### 7.8 Two-phase decode: parallel entropy, serial LZ (Oodle, closed source)

- **What:** Kraken processes data in 128 KiB chunks. Each chunk has
  self-contained entropy streams for literals, commands, offsets, and
  lengths, so decoding happens in two phases.
  - **Phase 1:** entropy-decode into temporary arrays. This is independent
    per chunk and can run on other threads.
  - **Phase 2:** execute the LZ sequences. Matches can reach any earlier
    byte, so this runs in order.
- **Used by:** Oodle (closed source; public design notes). It is the
  clearest description of this split.
- **Pros:**
  - 1.4–1.9× decode speedup with two threads (1.7× on Silesia), with no
    ratio loss and no format change.
- **Cons:**
  - The LZ phase stays serial, so scaling stops at about 2×.
  - For more parallelism, customers "chop data into fixed-size blocks,
    typically between 64KiB and 512KiB", which does cost ratio.
- **Sources:** <http://cbloomrants.blogspot.com/2016/05/oodle-kraken-thread-phased-decoding.html>,
  <https://fgiesen.wordpress.com/2021/07/09/entropy-coding-in-oodle-data-the-big-picture/>

### 7.9 Adaptive level and rsync-friendly output

- **What:** Two zstd options that change how output is produced.
  - `--adapt` changes the compression level on the fly to match observed
    I/O speed.
  - `--rsyncable` uses a 32-byte rolling hash to force flush points at
    content-defined positions. A local change then only alters nearby
    output, so rsync can still send deltas. pigz and gzip have the same
    option (see 10.9).
- **Used by:** zstd, pigz, gzip.
- **Pros:**
  - `--adapt` keeps a pipeline saturated. Meta reports about 10% better
    ratio for the same transfer time.
  - `--rsyncable` has a "negligible" ratio cost in zstd.
- **Cons:**
  - `--adapt` output is not reproducible.
  - `--rsyncable` weakens long-distance matching.
- **Sources:** zstd.1.md, zstdmt_compress.c, pigz.c

---

## 8. GPU and hardware offload

GPUs need tens of thousands of independent work items. Every GPU codec
below is a different answer to the same question: *where does the
independence come from, and what does it cost in ratio?*

### 8.1 nvCOMP: batched chunk API vs manager API

- **What:** NVIDIA's GPU compression library.
  - **Formats:** LZ4, Snappy, Deflate, GDeflate, Gzip, and zstd for
    compatibility, plus NVIDIA's own Cascaded, and Bitcomp and ANS (both
    labelled proprietary).
  - **Low-level batched C API:** takes device arrays of chunk pointers and
    chunk sizes, and processes "many independent chunks simultaneously".
  - **High-level manager API:** takes one buffer and does the chunking
    itself.
  - **Chunk sizes:** the recommended starting point is 64 KB. ANS, Bitcomp,
    and Cascaded allow up to 16 MB per chunk.
- **Used by:** RAPIDS cuDF (Parquet/ORC I/O), NVIDIA Spark, gzstd.
- **Pros:**
  - Mature, broad format coverage.
  - Batched kernels keep occupancy high.
  - Blackwell GPUs add a hardware Decompression Engine (8.4).
- **Cons:**
  - **No longer open source.** The branch-2.3 README: "From version 2.3
    onwards, the compression / decompression source code will not be
    released." It now ships under the NVIDIA SDK license, and the GitHub
    repo was archived in July 2026.
  - Independent chunks cost ratio. NVIDIA measured 3–8% larger files from
    nvCOMP 3.0.6's zstd than from libzstd.
  - The manager API is slower on small batches.
- **Sources:** <https://docs.nvidia.com/cuda/nvcomp/>,
  <https://github.com/NVIDIA/nvcomp/blob/branch-2.3/README.md>,
  <https://docs.nvidia.com/cuda/nvcomp/license.html>,
  <https://developer.nvidia.com/blog/encoding-and-compression-guide-for-parquet-string-data-using-rapids/>

### 8.2 Warp-per-chunk LZ4 (last open nvCOMP, v2.2.0, BSD-3)

- **What:** The last open-source nvCOMP gives each chunk one warp: 32
  threads per chunk, 2 chunks per block.
  - Compressed input is staged through a shared-memory window.
  - LZ4's variable-length length bytes (runs of 0xFF) are parsed by the
    warp at once: `__ballot_sync` finds the terminating byte and
    `__shfl_sync` fetches it.
  - Match copies use cooperative copy routines, with a separate path for
    overlapping matches (distance < length).
  - The compressor also uses one warp per chunk, with a hash table of at
    most 2^14 entries.
- **Used by:** nvCOMP ≤2.2. The source is still at the old git tags.
- **Pros:**
  - All 32 lanes work on parsing and copying.
  - No format change: it reads and writes standard LZ4 blocks.
- **Cons:**
  - Sequences within a chunk are still decoded in order, so parallelism
    comes only from the number of chunks.
  - LZ4's ratio.
- **Sources:** <https://github.com/NVIDIA/nvcomp/blob/v2.2.0/src/LZ4Kernels.cuh>,
  <https://github.com/NVIDIA/nvcomp/blob/v2.2.0/src/lowlevel/LZ4CompressionKernels.cu>

### 8.3 Columnar pre-transforms (Cascaded: RLE, delta, bit-packing)

- **What:** Stackable stages, each a parallel scan or map: RLE, delta
  coding, and bit-packing (a base value plus minimum-width offsets). An
  example pipeline is RLE → Delta → RLE → Bit-pack.
- **Used by:** nvCOMP Cascaded; the same ideas appear in Parquet and ORC
  encodings and Blosc's shuffle filters.
- **Pros:**
  - Massive gains on integer columns. NVIDIA reports 80:1 at 56 GB/s on a
    T4 for one Fannie Mae column, "nearly 4× LZ4".
  - Every stage is trivially parallel.
- **Cons:**
  - Only useful for typed numeric data.
  - The best stage order depends on the data, so it has to be tuned or
    auto-selected.
- **Sources:** <https://developer.nvidia.com/blog/optimizing-data-transfer-using-lossless-compression-with-nvcomp/>,
  <https://www.blosc.org/pages/blosc-in-depth/>

### 8.4 Fixed-function decompression (Blackwell DE, Intel IAA/QAT)

- **What:** Dedicated hardware decompresses standard formats.
  - **Blackwell Decompression Engine** (B200/GB200-class, nvCOMP 4.2+):
    decodes Snappy, LZ4, and Deflate-based streams as data moves over PCIe
    or C2C. Buffers must be allocated with a hardware-decompress usage
    flag, and on B200 buffers over 4 MB fall back to the SM kernels.
  - **Intel IAA** (via QPL, MIT) and **QAT** (via QATzip, BSD-3) offload
    Deflate. QAT Gen4+ adds LZ4, and Gen6 adds zstd frames.
  - The Linux `iaa_crypto` driver exposes `deflate-iaa` to zswap. Its limits
    are a 4 KB history window and fixed Huffman only.
- **Used by:** nvCOMP, QPL, QATzip, the Linux crypto API.
- **Pros:**
  - Leaves SMs or CPU cores free.
  - Page-sized units are trivially parallel.
- **Cons:**
  - Standard formats only, with size limits.
  - Weak ratio settings: IAA's 4 KB history, fixed Huffman.
  - Hardware lock-in.
  - Can break reproducibility. gzip disables IBM Z DFLTCC when
    `SOURCE_DATE_EPOCH` is set.
- **Sources:** <https://developer.nvidia.com/blog/speeding-up-data-decompression-with-nvcomp-and-the-nvidia-blackwell-decompression-engine/>,
  <https://github.com/intel/qpl>, <https://github.com/intel/QATzip>,
  <https://docs.kernel.org/driver-api/crypto/iaa/iaa-crypto.html>

### 8.5 GDeflate: Deflate spread across 32 sub-streams

- **What:** Deflate reorganized for SIMD and GPU decode.
  - **Tiles:** the input is split into 64 KiB tiles, compressed
    independently. This gives the coarse parallelism.
  - **Sub-streams:** within a tile, the Deflate bitstream is permuted
    across 32 sub-streams. This gives the fine parallelism.
  - **Symbol assignment:** symbols are dealt round-robin in groups of up to
    32. A literal, or a length+distance pair, stays on one lane. A lane that
    received a length code skips the next round so it can decode the
    distance, which keeps the lanes balanced.
  - **Refills:** each lane keeps a 64-bit bit buffer and refills 32 bits at
    a time, in lane order, so the reads land next to each other in memory.
- **Used by:** Microsoft DirectStorage GPU decompression, nvCOMP.
- **Pros:**
  - 32-way Huffman decode with ratio "exactly" equal to Deflate's.
  - Any Deflate encoder can be used before the permutation step.
  - DirectStorage uses it to multiply effective PCIe/NVMe bandwidth by the
    compression ratio and to remove the CPU decode bottleneck.
- **Cons:**
  - Only the Huffman decode is parallel. LZ copies inside a tile still
    depend on each other.
  - Deflate's 32 KB window caps ratio.
  - Assets must be packed as 64 KiB tiles.
- **Sources:** <https://github.com/microsoft/DirectStorage/blob/main/GDeflate/GDeflate/README.md>,
  <https://devblogs.microsoft.com/directx/directstorage-1-1-now-available/>

### 8.6 DietGPU: warp-per-4 KiB-segment interleaved rANS

- **What:** A GPU rANS coder from Meta (MIT licensed; archived June 2026).
  - One warp per 4 KiB segment. Each lane has its own 32-bit state, and
    lane *i* codes bytes *i*, *i*+32, and so on. Output words are 16 bits.
  - Probabilities use 9–11 bits. Decode uses a 2^probBits-entry table
    (symbol, pdf, cdf) held in shared memory.
  - At renormalization, each lane finds its read offset in the shared,
    backward-read word stream with a warp ballot plus popcount.
  - A companion float codec entropy-codes only the exponent bytes of
    bfloat16/float16 data and stores sign and mantissa raw.
- **Used by:** DietGPU (aimed at ML collectives and checkpoints).
- **Pros:**
  - Very fast: 250–410 GB/s for ANS on A100, and 250–600 GB/s for the float
    codec.
  - Simple, byte-oriented, and usable as the entropy stage behind LZ.
- **Cons:**
  - Order-0 bytes only, with a low target ratio (0.6–0.9× of the input).
  - Not worth using below about 512 KiB of input. Needs enough segments to
    fill the GPU.
  - Batches with more than one array are slower because of load imbalance.
- **Sources:** <https://github.com/facebookresearch/dietgpu>,
  <https://github.com/facebookresearch/dietgpu/blob/main/dietgpu/ans/GpuANSDecode.cuh>

### 8.7 Recoil: split one interleaved rANS stream using metadata

- **What:** Encode *one* 32-way interleaved rANS bitstream (so there is no
  table or state reset), then record "split points" so decoding can start
  in the middle.
  - At each split it stores every lane's state at its last renormalization
    (< 2^16, so 16 bits each), plus symbol and bitstream offsets, all
    variable-length coded.
  - Decode has three phases: synchronize, decode normally, then decode the
    section that crosses into the previous split.
  - A decoder with less parallelism simply ignores some split points.
- **Used by:** Lin et al., ICPP 2023 (research code).
- **Pros:**
  - Parallelism without resetting state or tables.
  - Tiny overhead: +0.02% on enwik9 with 2176 partitions, vs +0.03% for
    independent partitions.
  - The split count can be chosen at decode time.
  - About 90 GB/s on an RTX 2080 Ti, "similar performance" to DietGPU.
- **Cons:**
  - The encoder needs an extra pass.
  - Covers only the entropy stage. It does nothing for LZ dependencies.
- **Sources:** <https://arxiv.org/abs/2306.12141>

### 8.8 GPU LZ77: the dependency problem and its answers

**Why this is hard.** An LZ match copies from earlier *output*, so decoding
is serial within a block. Published answers fall into four groups:
independent blocks, restricted references, multi-round resolution, and
abandoning LZ.

- **Independent blocks (CULZSS, nvCOMP, most practical tools):** one chunk
  per thread, warp, or block, with no references across chunks.
  - Pros: simple, and scales perfectly.
  - Cons: ratio falls as chunks shrink. CULZSS's reported speedups could
    not be confirmed.
- **Gompresso (Sitaridi et al., ICPP 2016):** 256 KB blocks split into
  sub-blocks of 16 sequences. The header stores sub-block bit offsets so
  each thread Huffman-decodes its own sub-block.
  - One warp handles 32 sequences, using warp prefix sums to find literal
    and output positions.
  - *Multi-Round Resolution:* each round, a warp copies every match whose
    source lies below a high-water mark of finished output. Deeply nested
    matches can make this serial.
  - *Dependency Elimination:* the encoder refuses any match whose source is
    still being produced by the same warp.
  - Pros: 2–3× faster decode from Dependency Elimination; about 2× a
    12-core zlib on a K40.
  - Cons: Dependency Elimination costs up to 19% ratio and 13% compression
    speed. Length-limited Huffman costs another ~9%.
- **Two-pass and multi-byte-symbol LZ (GPULZ, 2023):** LZSS over 2-, 4-, or
  8-byte symbols for scientific data, with a two-pass prefix sum and fused
  kernels.
  - Pros: claims up to 272× speedup and 1.4× better ratio than earlier GPU
    work.
  - Cons: the baselines are unclear.
- **CODAG (2023):** argues that decoders with specialized thread roles (a
  few decoding threads fed by data-movement threads) leave most threads
  idle. It runs many plain independent decode streams instead.
  - Pros: 13.5× (RLE v1), 5.7× (RLE v2), and 1.18× (Deflate) over RAPIDS.
  - Cons: needs many chunks.
- **BWT on GPU (Patel et al., 2012):** parallel BWT sort, MTF, and Huffman.
  - Cons: slower than CPU bzip2, because string sorting is the bottleneck.
    Widely cited as a negative result **[figures unverified]**.
- **Sources:** <https://arxiv.org/abs/1606.00519>, <https://arxiv.org/abs/2304.07342>,
  <https://arxiv.org/abs/2307.03760>, <https://github.com/adnanozsoy/CUDA_Compression>,
  <https://escholarship.org/uc/item/7bs1x3qn>

### 8.9 Parallel Huffman decode on GPUs

Three approaches to the same problem:

- **Self-synchronization (Weißenberger & Schmidt, ICPP 2018, "gpuhd"):**
  threads start at arbitrary bit offsets. Huffman codes tend to fall back
  into sync, so wrong starts converge on the true symbol boundaries, and a
  sync-detection pass fixes each thread's output position.
  - Pros: works on *standard* Huffman streams; "over one order of
    magnitude" faster than zstd's CPU Huffman decoder.
  - Cons: wasted work during sync, and the sync distance depends on the
    data.
- **Gap arrays (Yamamoto et al., ICPP 2020):** store a small per-segment bit
  offset of the first codeword boundary, so each segment decodes on its own.
  - Pros: deterministic, with small metadata. Gap arrays add 1.67–6450× on
    top of the other optimizations.
  - Cons: a format change. This is the Huffman counterpart of Recoil.
- **multians (ICPP 2019):** the self-synchronization idea applied to tANS.
  - Pros: zero storage overhead.
  - Cons: per Recoil, sync is expensive and quantization is limited
    **[characterized second-hand]**.
- **Also:** cuSZ's decoder tuning (IPDPS 2022) got 3.64× from shared-memory
  and divergence work alone.
- **Sources:** <https://github.com/weissenberger/gpuhd>,
  <https://github.com/daisuke-takafuji/Huffman_coding_Gap_arrays>,
  <https://github.com/weissenberger/multians>, <https://arxiv.org/abs/2201.09118>

### 8.10 GPU zstd

- **What:** Attempts to bring zstd to GPUs.
  - nvCOMP's zstd is closed source and produces 3–8% larger files than
    libzstd.
  - The open "Gstd" work in `elasota/zstdhl` (MIT/Apache-2.0, in progress,
    last pushed 2024) transcodes a zstd stream into a lane-interleaved
    layout. It rebuilds the FSE tables as rANS tables and caps the accuracy
    log at 9 and Huffman code lengths at 11 **[design read from code
    only]**.
- **Pros:**
  - Would keep zstd's parse and ratio while decoding on the GPU.
- **Cons:**
  - Not a standard format, and immature.
  - No open, production-quality GPU zstd exists.
- **Sources:** <https://github.com/elasota/zstdhl>, <https://github.com/rtcheek/gzstd>

### 8.11 GPU-friendly columnar layout (cuDF, Parquet on GPU)

- **What:** Size the data layout so the GPU gets enough independent work.
  - cuDF uses nvCOMP for Snappy, ZSTD, and LZ4 in Parquet and ORC, plus
    Deflate for ORC.
  - A 2026 study recommends at least 100 pages per file, because pages set
    the kernel grid, and million-row row groups. It reports up to
    125 GB/s on an A100 with GPUDirect Storage.
- **Pros:**
  - The format's own page and row-group structure supplies the parallelism.
- **Cons:**
  - Writers have to be tuned for GPU readers, and the best page size for a
    GPU is not the best for a CPU.
- **Sources:** <https://docs.nvidia.com/cudf/latest/cudf/io/io/index.html>,
  <https://arxiv.org/html/2602.17335v1>

---

## 9. Archive formats: packaging many files

### 9.1 tar: fixed 512-byte records

- **What:** Each entry is a 512-byte header followed by data padded to 512
  bytes. The archive ends with two zero records. Long names and extended
  attributes need extra pax `x` header entries.
- **Used by:** GNU tar, bsdtar/libarchive.
- **Pros:**
  - Trivially streamable: no seeking in either direction.
  - Can wrap any byte stream.
- **Cons:**
  - At least 512 bytes of overhead per file, plus padding.
  - No index, so listing requires reading the whole archive.
- **Sources:** <https://man.freebsd.org/cgi/man.cgi?query=tar&sektion=5>

### 9.2 tar plus an external codec (solid by construction)

- **What:** tar produces one byte stream and an external codec compresses
  it (`-z/-j/-J/--zstd/-I prog`). The result is automatically "solid":
  redundancy between files is exploited up to the codec's window.
- **Used by:** GNU tar, bsdtar (`--options zstd:threads=N`).
- **Pros:**
  - Maximum cross-file redundancy for a given codec.
  - Codec multithreading applies unchanged.
- **Cons:**
  - No random access: extracting one file means decompressing everything
    before it.
  - Compressed archives can't be appended to.
  - How well similar files are grouped depends on traversal order.
    `--sort=name` is for reproducibility, not ratio.
- **Sources:** <https://manpages.debian.org/testing/tar/tar.1.en.html>,
  <https://man.freebsd.org/cgi/man.cgi?query=bsdtar&sektion=1>

### 9.3 ZIP: per-entry compression, central directory at the end

- **What:** Each entry is compressed on its own. A central directory at the
  end of the file gives the offset of every entry.
- **Used by:** ZIP (PKWARE APPNOTE 6.3.10), Info-ZIP, 7-Zip, libarchive, JAR,
  OOXML, and many others.
- **Pros:**
  - True per-file random access.
  - Entries can be decompressed in parallel.
  - Entries can be added or replaced by rewriting only the tail.
- **Cons:**
  - Nothing is shared between entries, so many small, similar files
    compress badly.
  - Headers are never compressed, and each file name is stored twice (local
    and central headers).
- **Sources:** <https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT>

### 9.4 Per-entry method selection, including "store"

- **What:** Each ZIP entry records its own method. The APPNOTE lists 0
  store, 8 deflate, 12 bzip2, 14 LZMA, 93 zstd, 95 xz, and 98 PPMd. (Method
  20 was an early zstd ID and is now deprecated in favour of 93.)
  - Info-ZIP stores rather than compresses files with extensions such as
    `.zip`, `.Z`, and `.arj`, and falls back to store when compression
    doesn't help.
  - libarchive can write store, deflate, bzip2, LZMA, xz, or zstd entries.
- **Used by:** ZIP writers.
- **Pros:**
  - No CPU wasted on data that is already compressed.
  - Each entry can use the best codec for its content.
- **Cons:**
  - Many readers only support method 8 (Deflate). Using zstd or xz entries
    hurts portability.
- **Sources:** APPNOTE, <https://manpages.debian.org/testing/zip/zip.1.en.html>

### 9.5 Streaming ZIP writes (data descriptors) and Zip64

- **What:** Two ZIP extensions.
  - With flag bit 3 set, the local header's CRC and size fields are zero,
    and the real values go in a data descriptor after the data. The writer
    never has to seek back, so it can write to a pipe.
  - Zip64 adds 8-byte size and offset fields and lifts the 4 GB and
    65,535-entry limits.
- **Used by:** all modern ZIP writers.
- **Pros:**
  - Streamable output.
  - Large archives.
- **Cons:**
  - A reader that only scans local headers can't find where an entry's
    data ends without decompressing it.
  - Zip64 breaks old unzippers.
  - Streaming and threading interact badly: 7-Zip 23.00 fixed a bug where
    multithreaded ZIP output to stdout left out the descriptors.
- **Sources:** APPNOTE, <https://www.7-zip.org/history.txt>

### 9.6 Parallel ZIP creation

- **What:** Implementations differ.
  - libarchive threads the codec inside one entry, and only for xz and
    zstd.
  - Apache Commons Compress `ParallelScatterZipCreator` compresses entries
    concurrently into thread-local streams, then merges them.
  - 7-Zip uses multiple threads when adding several files to a ZIP
    (32 MB of buffer per thread). Exactly how it schedules them is
    **[unverified]**.
  - Info-ZIP has no threading.
- **Pros:**
  - Independent entries make parallelism across files straightforward.
- **Cons:**
  - Entries must be buffered in memory or temp files before they are
    written in order.
  - The result is still non-solid.
- **Sources:** <https://commons.apache.org/proper/commons-compress/apidocs/org/apache/commons/compress/archivers/zip/ParallelScatterZipCreator.html>,
  <https://man.freebsd.org/cgi/man.cgi?query=archive_write_set_options&sektion=3>

### 9.7 7z solid blocks with a size cap

- **What:** A 7z "folder" is one compressed stream holding several files.
  Solid mode is on by default. `-ms` caps a block by file count or byte
  size, or starts a new block for each extension (`e`).
  - Default caps by level: Fastest 16 MB, Fast 128 MB, Normal 2 GB,
    Maximum/Ultra 4 GB.
- **Used by:** 7-Zip, libarchive (writing).
- **Pros:**
  - Exploits redundancy between files, while the cap bounds how much must
    be decoded to reach one file.
- **Cons:**
  - Extracting one file costs up to one whole block.
  - "Updating of solid .7z archives can be slow."
  - Damage in one block loses every file in it.
- **Sources:** <https://github.com/ip7z/7zip/blob/main/DOC/7zFormat.txt>,
  7-Zip method switches (mirror: <https://7-zip.opensource.jp/chm/cmdline/switches/method.htm>)

### 9.8 Sorting by type (`-mqs`) and automatic executable filters

- **What:** Two ways 7-Zip adapts to file content.
  - `-mqs` sorts files by extension so similar files sit next to each other
    inside a solid block. It has been off by default since 15.06.
  - 7-Zip applies BCJ, BCJ2, or ARM64 filters to executables automatically.
    Since 23.00 it *parses* `.exe` and `.dll` files to pick the right
    filter. It also analyzes WAV, PE, ELF, and Mach-O headers to choose
    Delta or BCJ filters.
- **Used by:** 7-Zip.
- **Pros:**
  - Better LZ matches, and a better-matched filter for each file.
  - The help says the gain can be large when the dictionary is smaller than
    the total input.
- **Cons:**
  - Results depend on order and type.
  - The default moved away from type sorting, likely because of file
    locality on disk (NTFS stores directory entries sorted by name).
- **Sources:** <https://www.7-zip.org/history.txt>, method.htm mirror

### 9.9 LZMA2 chunk multithreading and compressed headers (7z)

- **What:** Two more 7z features.
  - LZMA2 splits a large file into chunks and compresses them in parallel.
    It uses 1 thread per chunk at x1 and x3, and 2 per chunk at x5–x9
    (the second thread runs the match finder).
  - All metadata (names, times, sizes) lives in one header at the end of
    the archive, compressed with LZMA by default (`hc=on`).
- **Used by:** 7-Zip.
- **Pros:**
  - Parallel speedup even on a single large file.
  - Per-file metadata costs almost nothing, even for millions of small
    files.
- **Cons:**
  - Chunk boundaries cost ratio.
  - Memory grows with thread count.
  - The whole header must be decompressed before listing, and rewritten on
    every update.
- **Sources:** method.htm mirror, 7zFormat.txt

> **RAR (proprietary, for comparison).** RAR5 has a solid flag that keeps
> the dictionary between files, an optional recovery record for error
> correction, and a "quick open" header cache. Source:
> <https://www.rarlab.com/technote.htm>.

### 9.10 SquashFS: fixed-size input blocks, fragments, dedup

- **What:** A read-only, mountable filesystem image.
  - Each file is split into fixed-size input blocks (4 KB–1 MB, default
    128 KB), compressed independently with gzip, lzo, lz4, xz, zstd, or
    lzma.
  - Small files and file tails are packed together into shared *fragment*
    blocks.
  - Inodes and directories are compressed in 8 KB metadata blocks.
  - Duplicate files are detected.
- **Used by:** Linux live images, Snap, AppImage, firmware.
- **Pros:**
  - A random read costs at most one block of decompression.
  - Small files share compression context through fragments.
  - Metadata overhead is low.
- **Cons:**
  - Nothing is shared across blocks.
  - Fixed-size input produces variable-size compressed blocks that don't
    line up with I/O units (the problem EROFS solves).
  - Large blocks raise read latency.
- **Sources:** <https://docs.kernel.org/filesystems/squashfs.html>,
  <https://manpages.debian.org/testing/squashfs-tools/mksquashfs.1.en.html>

### 9.11 EROFS: fixed-size output compression

- **What:** Instead of cutting fixed-size input, the compressor fills
  fixed-size *physical clusters* exactly with as much input as fits.
  - Decompression happens in place, with no bounce buffer.
  - Other features: big physical clusters, tail packing, fragments, and
    dedup. Algorithms can be chosen per inode (LZ4, MicroLZMA, DEFLATE,
    zstd).
- **Used by:** Android system images, container images.
- **Pros:**
  - Every byte of every I/O is usable.
  - About 5% smaller at small (4–8 KiB) cluster sizes.
  - Low read amplification.
- **Cons:**
  - A more complex encoder: it has to stop at an output boundary.
  - Features depend on the kernel version (big clusters in 5.13, tail
    packing in 5.17, fragments and dedup in 6.1).
- **Sources:** <https://docs.kernel.org/filesystems/erofs.html>,
  <https://erofs.docs.kernel.org/en/latest/design.html>

### 9.12 DwarFS: similarity ordering, segment dedup, big blocks

- **What:** A pipeline in four steps:
  1. Deduplicate whole files by hash.
  2. Order files by similarity (nilsimsa clustering plus nearest-neighbour
     search).
  3. Run a "segmenter": a cyclic hash plus a Bloom filter finds repeated
     segments across the last few blocks.
  4. Compress the rest in large blocks (2^20–2^26) with zstd, LZMA, or
     Brotli.

  Categorizers send PCM audio to FLAC, FITS images to Rice coding, and
  data judged incompressible to storage without compression.
- **Used by:** DwarFS (mkdwarfs plus a FUSE driver).
- **Pros:**
  - Very high ratios on redundant trees. The author's example: 47.65 GiB of
    Perl installations shrinks to "less than 0.9%", more than 10× smaller
    than SquashFS, and builds 6× faster than mksquashfs. These are the
    project's own figures.
- **Cons:**
  - One file can reference several blocks. A read can require
    decompressing up to (lookback × block size), for example 64 MiB.
  - The nearest-neighbour ordering is O(n²) and "does not scale well"
    beyond a few hundred thousand files per cluster.
  - Getting reproducible images across machines requires pinning worker
    counts.
- **Sources:** <https://github.com/mhx/dwarfs/blob/main/doc/mkdwarfs.md>,
  <https://github.com/mhx/dwarfs>

### 9.13 Indexed tarballs (pixz, tarlz)

- **What:** Combine an archive index with independent blocks so single
  files can be extracted.
  - pixz writes blocked, parallel xz and indexes the tar members during
    compression.
  - tarlz aligns lzip members with tar members and lets you choose the
    granularity: per file, per block (the default, 16 MiB), per directory,
    or fully solid.
- **Used by:** pixz (`.tpxz`), tarlz.
- **Pros:**
  - Output is still valid xz or lzip.
  - One file can be listed or extracted without a full decode.
  - Parallel in both directions.
  - tarlz makes the solid-vs-random-access trade-off an explicit setting.
- **Cons:**
  - Independent blocks lower the ratio compared with one solid stream.
- **Sources:** <https://github.com/vasi/pixz>,
  <https://manpages.debian.org/testing/tarlz/tarlz.1.en.html>

### 9.14 Recompressing existing Deflate data (precomp, preflate, grittibanzli)

- **What:** Files often contain Deflate streams (zip, PNG, PDF, docx) that a
  better compressor can't see into. These tools decompress the stream and
  store just enough information to rebuild the original compressed bytes
  exactly.
  - Old precomp brute-forced zlib's parameters, and gave up on anything
    made by 7-Zip or kzip.
  - preflate predicts each decision the encoder made and stores only the
    corrections.
  - grittibanzli stores the Deflate choices separately and compresses them
    with Brotli.
- **Used by:** precomp-cpp (with preflate), grittibanzli (archived 2022).
- **Pros:**
  - Unlocks the redundancy hidden inside containers that are already
    compressed.
  - preflate works with any Deflate encoder.
- **Cons:**
  - Slow ("it's quite slow").
  - For about 20–30% of zlib streams, preflate's correction data exceeds
    3 bytes.
  - The whole scheme depends on reconstructing the stream bit-exactly.
- **Sources:** <https://github.com/schnaader/precomp-cpp>,
  <https://github.com/deus-libri/preflate>, <https://github.com/google/grittibanzli>

### 9.15 Git packfiles: delta chains with sorted candidate windows

- **What:** How git packs objects:
  - Objects are sorted by type, then by a hash of the file name (the
    basename weighs most), then by size, largest first.
  - Each object is tried as a delta against the others inside a sliding
    window (`--window`, default 10). Chains are capped by `--depth`
    (default 50).
  - Deltas go from larger to smaller objects, because "removing data is
    cheaper".
  - Every object, whether full or delta, is zlib-compressed on its own.
  - Existing deltas are reused.
  - Newer heuristics: `--name-hash-version=2` salts the name hash with the
    parent directories, and `--path-walk` groups objects by path first.
- **Used by:** git.
- **Pros:**
  - Excellent ratios on version histories.
  - An index gives fast random access by object ID.
  - The delta search runs in parallel.
- **Cons:**
  - Deep chains slow down reads.
  - Name-hash collisions pair up unrelated files.
  - Per-object zlib shares no context between objects.
- **Sources:** <https://git-scm.com/docs/pack-heuristics>,
  <https://git-scm.com/docs/git-pack-objects>

### 9.16 WIM (wimlib): single-instance storage plus solid LZMS

- **What:** Each distinct file content is stored once, keyed by SHA-1, even
  across multiple images in one archive.
  - Non-solid resources are compressed in chunks: XPRESS, LZX (the
    default), or LZMS.
  - `--solid` uses LZMS with 64 MiB solid chunks (the ESD format).
  - Images can be mounted with FUSE.
- **Used by:** wimlib, Windows installation media.
- **Pros:**
  - Dedup plus a choice of chunk size.
  - Solid mode gives a "significantly better compression ratio".
- **Cons:**
  - Solid mode degrades random access and costs speed and memory.
  - Microsoft's tools reject LZMS chunks over 64 MiB.
- **Sources:** <https://wimlib.net/>, <https://wimlib.net/man1/wimcapture.html>

### 9.17 The metadata cost of many small files

With millions of small files, per-file metadata can outweigh the data.

| Format | Per-file metadata handling |
|---|---|
| tar | 512-byte header + padding + pax records; only an outer solid codec hides it |
| ZIP | ~76 bytes of uncompressed headers + the name twice; never compressed |
| 7z | One LZMA-compressed header for the whole archive |
| SquashFS | Inodes/dirs compressed in 8 KiB blocks; small files packed in fragments |
| EROFS | Compact 32/64-byte inodes; inline or fragment tail packing |
| DwarFS | Bit-packed, delta-coded tables; FSST-compressed names ("50%" smaller) |
| zpaq, WIM | Content hashing dedups identical small files entirely |

- **Pattern:** random-access formats either compress metadata in blocks
  (7z, SquashFS, DwarFS) or keep it tiny and packed (EROFS). ZIP pays full
  price for every entry.
- **Pros (compressed or packed metadata):**
  - For huge small-file trees, per-file overhead drops from hundreds of
    bytes to a few.
  - Listing an archive reads one small region instead of scanning the whole
    file.
- **Cons:**
  - Compressed headers must be decompressed before any file can be found,
    and rewritten on update (7z).
  - Uncompressed per-entry headers (tar, ZIP) are simpler, streamable, and
    easier to recover from damage.
- **Sources:** as in 9.1–9.16, plus
  <https://github.com/mhx/dwarfs/blob/main/doc/dwarfs-format.md>

---

## 10. Multi-file and directory techniques

These operate above the codec: they find redundancy *between* files,
decide *what* to compress, and decide *in what order*.

### 10.1 Content-defined chunking (CDC) and dedup before compression

- **What:** A rolling hash (Rabin, buzhash, or Gear) runs over the data, and
  a chunk ends wherever the masked hash matches a pattern.
  - Because boundaries depend on content, an insertion only changes the
    chunks around it.
  - Chunks are identified by a strong hash, only new chunks are stored, and
    compression is applied per chunk *after* dedup.
  - **FastCDC** speeds up Gear hashing by skipping the region below the
    minimum chunk size and by using normalized masks, which pull chunk
    sizes toward the target.
  - Tools and their parameters:

    | Tool | Hash | Chunk size | Notes |
    |---|---|---|---|
    | borg | buzhash, per-repo secret seed | 512 KiB–8 MiB, ~2 MiB target | |
    | restic | Rabin, random per-repo polynomial | 512 KiB–8 MiB, ~1 MiB average | zstd in repo v2; blobs packed into 16 MiB pack files |
    | casync | buzhash | 64 KiB average | xz per chunk |
    | duplicacy | buzhash | 4 MB average | packs files in name order first |
    | kopia | selectable | | default `DYNAMIC-4M-BUZHASH` |
    | bupstash | Gear | < 8 MiB | keyed BLAKE3 addresses |
    | zpaq | rolling hash | ~64 KB average | SHA-1 fragment identity |

- **Pros:**
  - FastCDC reports "about 10× faster than … Rabin-based CDC" and 3× faster
    than earlier Gear or AE chunkers, with nearly the same dedup ratio.
  - The literature credits CDC with 10–20% more redundancy found than
    fixed-size chunking.
  - borg notes that compression settings never affect dedup, because dedup
    runs first.
  - Repository-secret seeds prevent fingerprinting attacks that infer file
    contents from chunk sizes.
- **Cons:**
  - Every byte is hashed.
  - The chunk index and RAM grow with chunk count. Small chunks cost
    resources and, on remote stores, one request per chunk (casync notes
    the HTTP GET load).
  - Each chunk is compressed on its own, so there is no match-finding
    across chunks. For restic that means a 512 KiB window.
- **Sources:** <https://csyhua.github.io/csyhua/hua-atc2016.pdf>,
  <https://borgbackup.readthedocs.io/en/stable/internals/data-structures.html>,
  <https://restic.readthedocs.io/en/stable/100_references.html>,
  <https://0pointer.net/blog/casync-a-tool-for-distributing-file-system-images.html>,
  <https://github.com/gilbertchen/duplicacy/wiki/Chunk-Size>,
  <https://raw.githubusercontent.com/kopia/kopia/master/repo/splitter/splitter.go>,
  <https://github.com/andrewchambers/bupstash/blob/master/doc/technical_overview.md>,
  <https://mattmahoney.net/dc/zpaq.html>

### 10.2 Fixed-size chunking, chosen on purpose

- **What:** Cut at fixed offsets, for example every 4 MiB.
  - borg recommends it for block devices and raw disk images.
  - duplicacy switches to it when min = avg = max, which suits VM disks,
    databases, and encrypted containers.
- **Used by:** borg `fixed`, duplicacy, kopia `FIXED-*`.
- **Pros:**
  - No hashing cost.
  - Matches data that is updated in place at aligned offsets exactly.
- **Cons:**
  - An insertion or deletion shifts every later boundary, and dedup
    collapses.
- **Sources:** <https://borgbackup.readthedocs.io/en/stable/usage/notes.html>, duplicacy wiki

### 10.3 Skip unchanged files using metadata (files cache)

- **What:** borg keeps a cache keyed by path, holding each file's inode,
  size, mtime_ns, and chunk IDs. If none of those have changed, the file is
  not read, chunked, or hashed again.
- **Used by:** borg; restic and others have equivalents.
- **Pros:**
  - This, not the codec, is what makes re-backups of huge small-file trees
    fast.
- **Cons:**
  - Memory: borg estimates 240 bytes per file plus 80 per chunk.
  - Trusts mtime, which can be wrong on some filesystems or after certain
    restores.
- **Sources:** borg data-structures doc

### 10.4 Long-range matching as an alternative to dedup

- **What:** Instead of a chunk index, a huge match window finds repeats
  directly.
  - zstd `--long` (3.5).
  - rzip and lrzip first run a long-range pass (rzip's effective history is
    900 MB; lrzip uses the largest window that fits in RAM, or a sliding
    mmap). Then a normal codec runs: LZMA, ZPAQ, bzip2, gzip, or LZO.
  - SREP is reportedly a similar preprocessor that keeps hashes instead of
    a RAM dictionary **[not verified from a primary source]**.
- **Used by:** zstd, rzip, lrzip.
- **Pros:**
  - Catches duplicates hundreds of MB apart in a single stream, with no
    repository or index to maintain.
- **Cons:**
  - Memory-hungry.
  - lrzip only compresses single files, so it needs tar.
  - rzip can't be used in a pipe, and lrzip's stdin mode loses ratio on
    inputs above about 25% of RAM.
  - No incremental or versioned storage.
- **Sources:** <https://rzip.samba.org/>,
  <https://raw.githubusercontent.com/ckolivas/lrzip/master/README.md>

### 10.5 Shared dictionaries across many small objects

Section 3.4 covers the dictionary codec itself. This is about deploying
dictionaries across many objects.

- **What:**
  - **RocksDB** trains a zstd dictionary for each SST file by sampling
    its data blocks, and stores it in the file's meta-block.
  - **HTTP Compression Dictionary Transport (RFC 9842, 2025):** a response
    can mark itself as a dictionary with `Use-As-Dictionary`. The client
    advertises the SHA-256 of the dictionary it holds, and the server sends
    `dcb` (Shared Brotli) or `dcz` (zstd) responses. It is HTTPS-only and
    same-origin, and needs `Vary`.
  - **zstd dictionary IDs** detect a mismatched dictionary at decode time.
- **Used by:** RocksDB, Chrome and CDNs, zstd.
- **Pros:**
  - Chrome's example: Angular 1.8.3 delta-compressed against 1.7.9 is
    "just over 4 KiB", vs about 53 KiB with plain Brotli.
  - RocksDB's dictionaries travel with the data, so versioning is
    automatic.
- **Cons:**
  - Every dictionary version must be kept as long as any data uses it.
  - HTTP dictionaries expire with the cache and complicate CDN caching. The
    earlier SDCH attempt "was challenging to implement safely".
  - A dictionary controlled by an attacker can make the decoder produce
    arbitrary bytes.
  - RocksDB has to buffer a whole file's blocks to train.
- **Sources:** <https://github.com/facebook/rocksdb/wiki/Dictionary-Compression>,
  <https://www.rfc-editor.org/rfc/rfc9842.html>,
  <https://developer.chrome.com/blog/shared-dictionary-compression>, zdict.h

### 10.6 File ordering and clustering before solid compression

- **What:** Put similar files next to each other so that one compression
  window sees them together.
  - 7-Zip sorts by extension (`-mqs`) or starts one block per extension
    (`s=e`).
  - DwarFS clusters files by similarity (9.12).
  - git sorts by name hash and size (9.15).
  - zpaq sorts fragments by extension and then size before packing them
    into blocks.
  - duplicacy packs files in name order.
- **Pros:**
  - Cheap, often large ratio gains, especially when the total input is
    bigger than the window.
- **Cons:**
  - Only heuristics: file names and sizes are weak signals, and true
    similarity ordering scales poorly.
  - Order changes can break reproducibility and hurt disk locality.
  - Name order (tar `--sort=name`) gives no ratio gain.
- **Sources:** 7-Zip method.htm mirror, mkdwarfs.md, pack-heuristics, zpaq page,
  <https://reproducible-builds.org/docs/archives/>

### 10.7 Solid vs per-file compression, and small-file packing

- **What:** Choose whether files share one compression context or each get
  their own.
  - Solid (tar+codec, 7z solid blocks) shares one context across files.
  - Per-file (ZIP) compresses each file alone.
  - Middle grounds: capped solid blocks (7z), SquashFS and EROFS fragments
    for small files, tarlz's selectable granularity, and zpaq's 16/64 MB
    blocks.
- **Pros:**
  - Solid: the best ratio on many small, similar files.
  - Per-file: random access, cheap updates, parallelism across entries, and
    damage stays contained to one file.
- **Cons:**
  - Solid: slow extraction of single files, slow updates, and a wider blast
    radius when data is damaged.
  - Per-file: poor ratio on small files. zstd's README: small data
    compresses poorly because "there is no 'past' to build upon".
- **Sources:** 7-Zip method.htm mirror,
  <https://raw.githubusercontent.com/plougher/squashfs-tools/master/Documentation/4.7.6/USAGE-MKSQUASHFS>,
  tarlz man page

### 10.8 Detecting and skipping incompressible data

- **What:** Spend (almost) no CPU on data that won't shrink.

  | Tool | Method |
  |---|---|
  | borg `auto` | Compress with LZ4. If it doesn't shrink, store raw. If LZ4 reaches below 97% of the size, run the expensive codec, and keep its output only if it's below 99% of the LZ4 size. |
  | lrzip | A pass of LZ4 first. If nothing compresses, the slow backend is skipped. |
  | OpenZFS zstd early abort | For zstd level ≥3 on records ≥128 KiB: try LZ4, then zstd-1, then store raw. LZ4 alone was "losing up to 8.5%" of savings on highly compressible data. Blocks must also save a sector and 12.5%. |
  | btrfs | If the first part of a file doesn't compress, sets a sticky NOCOMPRESS flag for the whole file. A read-only pre-check looks at entropy, byte frequencies, and repeated patterns. `compress-force` overrides. |
  | zstd | Emits Raw or RLE blocks when compression doesn't pay (it needs at least about 1/64 of the block in savings). The fast strategies skip ahead faster when matches stop appearing. |
  | ZIP / kopia | Skip by extension: `zip -n`, never-compress policies. |
  | DwarFS | Test-compresses with negative-level zstd. |

- **Pros:**
  - Media and archive files cost about one LZ4 pass or less.
  - Worst-case expansion is bounded.
- **Cons:**
  - Probes can misclassify data that only a strong or specialized codec
    compresses. DwarFS warns about audio.
  - btrfs's sticky flag gives up on a whole file based on its first part.
    The docs themselves call this "not optimal".
  - Extension lists go stale.
- **Sources:** <https://raw.githubusercontent.com/borgbackup/borg/1.4-maint/src/borg/compress.pyx>,
  <https://raw.githubusercontent.com/openzfs/zfs/master/module/zstd/zfs_zstd.c>,
  <https://raw.githubusercontent.com/openzfs/zfs/master/man/man4/zfs.4>,
  <https://raw.githubusercontent.com/kdave/btrfs-progs/master/Documentation/ch-compression.rst>,
  <https://raw.githubusercontent.com/facebook/zstd/dev/doc/zstd_compression_format.md>

### 10.9 Delta compression between versions

- **What:** Store a new version as instructions relative to an old one.
  - **VCDIFF / xdelta3 (RFC 3284):** ADD, COPY, and RUN instructions
    against a source window. xdelta3's source buffer defaults to 64 MB.
  - **bsdiff:** suffix sorting, byte-wise differences, then bzip2. It claims
    patches 50–80% smaller than xdelta for executables.
  - **zstd `--patch-from` (1.4.5+):** dictionary compression with the old
    file as the dictionary. The reference is limited to 128 MiB unless
    `-M` raises it.
  - **rsync:** matches fixed-size blocks at any offset using a weak rolling
    checksum plus a strong hash. `--rsyncable` gzip, pigz, and zstd reset
    at content-defined points so compressed files still rsync well. The
    cost is about 1% for gzip, 1.5–3% for pigz, and "negligible" for zstd.
- **Pros:**
  - Tiny updates.
- **Cons:**
  - bsdiff needs about 17× the input size in memory and O(n log n) time.
  - xdelta only finds copies within half its source buffer.
  - The old version must be kept to apply the patch.
- **Sources:** <https://www.rfc-editor.org/rfc/rfc3284.html>,
  <https://www.daemonology.net/bsdiff/>, zstd.1.md,
  <https://rsync.samba.org/tech_report/node2.html>,
  <https://www.gnu.org/software/gzip/manual/gzip.html>

### 10.10 Parallelism across files vs within a file, and determinism

- **What:** Tools split work at different levels.
  - GNU parallel, pigz, or gzip run one process per file.
  - The zstd CLI walks multiple input files one after another and threads
    only within each file, so many small files get no parallelism at all.
  - 7-Zip threads LZMA2 chunks.
  - btrfs compresses its 128 KiB extents in parallel.
- **Determinism** varies by tool:
  - zstd guarantees identical output for any thread count, for a given
    version and parameters. `--single-thread`, `--jobsize`, and `--adapt`
    break that.
  - pigz matches single-threaded output only at level 4 and above (and
    zlib ≥1.2.4).
  - xz's single-threaded and multithreaded outputs differ by design.
  - mksquashfs defaults to reproducible output. DwarFS needs pinned
    options to get it.
- **Pros:**
  - Parallelism across files is free and has no ratio cost when there are
    many files.
  - Deterministic output enables caching and verification.
- **Cons:**
  - Parallelism across files doesn't help one huge file.
  - Parallelism within a file doesn't help many tiny files.
  - Few tools do both.
- **Sources:** <https://www.gnu.org/software/parallel/parallel_examples.html>,
  <https://raw.githubusercontent.com/facebook/zstd/dev/programs/fileio.c>,
  <https://github.com/facebook/zstd/issues/2079>, pigz.c, xz.1

### 10.11 Filesystem-level transparent compression

- **What:** The filesystem compresses each record or extent as it is
  written.
  - ZFS compresses each record (recordsize up to 128 KiB, or 16 MiB with
    `large_blocks`) with lz4, zstd, or zstd-fast.
  - btrfs compresses 128 KiB chunks with zstd, lzo, or zlib, in parallel.
- **Used by:** ZFS, btrfs.
- **Pros:**
  - Every application gets compression with no changes.
  - Random rewrites stay cheap.
- **Cons:**
  - Small units cap the ratio, and redundancy between files is invisible.
  - ZFS rounds each block to whole sectors, so small records lose savings.
  - btrfs uses more metadata, direct I/O falls back to buffered, and
    nodatacow disables compression.
- **Sources:** <https://raw.githubusercontent.com/openzfs/zfs/master/man/man7/zfsprops.7>,
  btrfs compression doc

---

## 11. Cross-cutting trade-offs

The same few trade-offs recur throughout the findings above.

### 11.1 Independence vs context

Every unit of parallelism or random access is a place where context is
thrown away. Tools lose less ratio when they keep some context across the
boundary cheaply:

| Technique | Keeps | Cost | Example |
|---|---|---|---|
| Priming / overlap | the previous block's window | serial decode | pigz, zstd -T |
| Table reuse | entropy tables | tables can't adapt per chunk | zstd Repeat/Treeless, gpusqz table groups |
| Trained dictionary | history, *across* files | dictionary logistics | zstd `--train`, RocksDB |
| Split metadata | coder state at split points | encoder pass, format change | Recoil, gap arrays |
| Restricted references | matches from finished output only | up to 19% ratio | Gompresso DE |
| Plain independence | nothing | full ratio loss, simplest | pzstd, BGZF, nvCOMP, plzip |

As a rule, ratio loss shrinks as chunks grow. The loss reported for
independent chunks ranges from 0.2% (pbzip2's 900 KB blocks) to 0.4–2%
(plzip's multi-MB members) and 3–8% (nvCOMP zstd vs libzstd). Small
chunks (BGZF's 64 KiB, DietGPU's 4 KiB) pay much more, and need a context-
keeping technique from the table to compensate.

### 11.2 Where decode parallelism comes from

| Format property | Who can decode in parallel |
|---|---|
| Independent units with sizes recorded (pzstd, seekable, xz -T, plzip, BGZF, LZ4 independent blocks, nvCOMP chunks) | Anyone |
| Independent units, sizes not recorded (bzip2) | Only with a magic-scan (lbzip2) |
| One dependent stream (gzip, zstd -T, xz -T1, lzip) | Only with speculation (rapidgzip) or not at all |
| Separate entropy and LZ phases (Kraken) | Entropy phase in parallel, LZ serial |
| Interleaved entropy lanes (GDeflate, interleaved rANS) | SIMD/GPU parallel within a unit, at no ratio cost |

### 11.3 The compression-speed ladder

At the same decode speed, ratio is bought with compression time. The
match-finder steps, from cheapest to most expensive:

1. sparse sampling
2. single hash probe
3. SIMD row tags
4. hash chains of growing depth
5. lazy evaluation
6. binary trees
7. price-based optimal parsing
8. multi-pass statistics seeding

Each step costs roughly 2–10× more compression time for a few percent of
ratio.

### 11.4 Decode speed is set by the entropy coder and the token format

In rough order of decode speed, fastest first:

1. byte-aligned tokens (LZ4)
2. Huffman, especially multi-stream (zstd literals, Deflate)
3. tANS/rANS
4. adaptive binary range coding (LZMA)
5. context mixing, which is as slow as its encoder

Context modeling can be added without slowing decode as long as the codes
stay static (Brotli's context maps, order-1 rANS tables).

### 11.5 The biggest ratio gains come before the codec

Across multi-file tools, the largest gains come from stages that run before
any codec:

- deduplication (CDC, single-instance storage)
- long-range matching
- ordering similar files together
- type-specific transforms (BCJ, delta, Cascaded, FLAC routing in DwarFS)
- recompressing embedded Deflate streams (precomp)

The codec's own level setting is usually a second-order effect by
comparison.

### 11.6 The cheapest speed win is not compressing

Probes for incompressible data (borg auto, ZFS early abort, btrfs), stored
and raw fallbacks (ZIP store, zstd raw blocks, LZMA2 raw chunks), and file
caches keyed on metadata (borg) save more time on real directory trees than
any match-finder tuning.

---

## 12. Summary table

**Legend:** ↑ better, ↓ worse, ≈ neutral. "Speed" means compression (C) or
decompression (D).

| Technique | Speed effect | Ratio effect | Main users | Main downside |
|---|---|---|---|---|
| Byte-aligned tokens, no entropy stage | C↑ D↑↑ | ↓↓ | LZ4, Snappy, LZO | weak ratio |
| Multi-stream Huffman | D↑ | ≈ | zstd, Oodle | narrow parallelism |
| tANS/FSE | D≈ | ↑ vs Huffman | zstd | static tables, LIFO encode |
| Interleaved rANS | D↑↑ (SIMD/GPU) | ≈ | DietGPU, Recoil, gpusqz | multiply, LIFO encode |
| Adaptive range coding | D↓↓ | ↑ | LZMA, xz, 7z | serial per bit |
| Context mixing | C↓↓↓ D↓↓↓ | ↑↑↑ | PAQ, cmix | symmetric, days per GB |
| Repcodes | ≈ | ↑ | zstd, LZMA, Brotli | complicates parsing |
| Literal contexts | D≈ (static) / ↓ (adaptive) | ↑ | LZMA, Brotli | table/context overhead |
| Static built-in dictionary | ≈ | ↑ on web text | Brotli | domain-specific |
| Trained dictionary | ≈ | ↑↑ on small data | zstd, RocksDB, HTTP CDT | distribution/versioning |
| Long-range matching | C↓ | ↑↑ on redundant data | zstd --long, lrzip | memory on both sides |
| BCJ/delta filters | ≈ | ↑ on exe/audio | xz, 7z, SquashFS | type detection needed |
| Sparse sampling / --fast | C↑↑ D↑ | ↓↓ | zstd, LZ4 | ratio falls fast |
| SIMD row match finder | C↑ | ≈ | zstd 5–12 | SIMD required |
| Optimal parsing | C↓↓ | ↑ | xz, zstd 16+, libdeflate 8+ | 10–100× slower C |
| Block splitting / table reuse | C↓ | ↑ | libdeflate, Brotli, zstd | encoder search cost |
| SIMD kernels / whole-buffer API | C↑ D↑ | ≈ | zlib-ng, libdeflate, ISA-L | non-identical output / no streaming |
| Primed parallel blocks | C↑↑ | ≈ | pigz, zstd -T | serial decode |
| Independent frames + index | C↑↑ D↑↑ | ↓ (small) | pzstd, xz -T, plzip, BGZF | ratio loss grows as blocks shrink |
| Speculative parallel decode | D↑ (many cores) | n/a | rapidgzip, lbzip2 | wasted work |
| Two-phase decode | D↑ (~1.7×) | ≈ | Oodle | LZ stays serial |
| GPU warp-per-chunk LZ | C↑↑ D↑↑ | ↓ | nvCOMP, gpusqz | chunk independence |
| GDeflate 32-lane layout | D↑↑ | ≈ vs Deflate | DirectStorage | Deflate ceiling |
| LZ dependency elimination | D↑↑ | ↓ (≤19%) | Gompresso | ratio |
| Solid archive | C≈ | ↑↑ many small files | tar+codec, 7z | no random access |
| Per-entry archive | C↑ (parallel) | ↓ small files | ZIP | headers, no sharing |
| Fixed-output blocks | D↑ (I/O) | ↑ ~5% | EROFS | encoder complexity |
| CDC dedup | C↑ on re-runs | ↑↑ on redundant data | borg, restic, casync, zpaq | index RAM, per-chunk windows |
| Similarity ordering | C↓ | ↑↑ | DwarFS, 7z -mqs, git | O(n²) / heuristics |
| Incompressible probing | C↑↑ on media | ≈ (small risk) | borg, ZFS, btrfs | misclassification |
| Deflate recompression | C↓↓ | ↑↑ on zip/png/pdf | precomp | slow, bit-exactness |

---

## 13. Relevance to gpusqz

This section is separate from the survey above. gpusqz is a warp-per-chunk
LZ plus 32-way interleaved rANS compressor, with order-1 literal contexts
chosen per batch, rANS tables shared across a group of chunks, repeat-offset
codes, and a per-chunk raw fallback. On the axes above, it sits with
nvCOMP, DietGPU, and pzstd: **fully independent chunks with recorded sizes,
parallel in both directions**. It already uses several of the techniques
surveyed: interleaved rANS (2.4), repcodes (3.1), static literal contexts
(3.2), table reuse (4.7 and 11.1), and stored fallback (10.8).

Findings from this survey that bear most directly on its trade-offs:

- **Ratio lost to independence (11.1).** The listed ways to recover it
  without giving up GPU decode are:
  - trained or per-batch dictionaries (3.4)
  - Recoil-style split metadata instead of state resets (8.7)
  - Gompresso-style references restricted to already-decoded chunks
    (8.8), which trades some decode parallelism for cross-chunk matches

  Pigz/zstd-style priming (7.1, 7.2) would make decode serial across
  chunks, which removes gpusqz's decode parallelism.
- **Decode speed vs zstd.** Oodle's two-phase design (7.8) and GDeflate's
  lane interleaving (8.5) both separate the entropy phase, which
  parallelizes, from LZ execution, which doesn't. CODAG (8.8) argues for
  many plain streams over specialized thread roles.
- **Multi-file input.** If gpusqz grows beyond single files, the cheap,
  proven wins are:
  - solid concatenation with type ordering (10.6)
  - an incompressible-data probe before the GPU pass (10.8)
  - a seek index for chunks (7.3, 9.13), which its fixed chunking already
    almost provides
- **Type-specific transforms (3.6, 8.3, 11.5).** These are the largest
  pre-codec gains that keep chunks independent. The [Format-aware transforms
  study](format-aware-transforms-study.md) measures them on binary STL
  meshes.

---

## 14. Unverified items and caveats

Collected from the research. None of these is relied on above without a
flag.

- **ISA-L level 3 being AVX-512-only:** the release notes say level 3 is
  "currently only optimized for" AVX-512. Whether that makes the fast path
  AVX-512-only was not confirmed. No igzip MB/s figures were obtained.
- **Blackwell Decompression Engine peak throughput:** an "up to 600 GB/s"
  figure appeared only in a search snippet, so it is omitted.
- **CULZSS and Patel et al. figures:** not confirmed from the papers.
- **multians details:** second-hand, from the Recoil paper.
- **Gstd design intent:** read from code only.
- **GDeflate license:** the repo says MIT, while Microsoft's blog promised
  Apache 2.0.
- **7-Zip ZIP multithreading:** how it schedules entries across threads is
  unconfirmed. Only the 32 MB-per-thread figure is primary.
- **7-Zip help:** read from a Japanese mirror of the official help, since
  the help ships only inside the download.
- **DwarFS:** the default `--order` value was not confirmed.
- **SquashFS:** whether large-file tails go to fragments by default is
  ambiguous in the man page.
- **RAR5:** the recovery-record algorithm (Reed–Solomon) is unconfirmed.
- **SREP:** open-source status and mechanism were seen only in search
  results.
- **bupstash:** the compressor was not identified.
- **FastCDC:** mask and normalization details were not extracted from the
  paper, only the abstract-level claims.
- **zswap IAA batching:** whether the Intel patch series has been merged is
  unknown.
- **PPMd variant:** 7-Zip's page says it uses "PPMdH with small changes",
  while Mahoney's DCE says 7-Zip uses variant I. This survey follows 7-Zip's
  page.
- **Project-reported figures:** DwarFS vs SquashFS, wimlib vs Microsoft,
  and the Cloudflare zlib speedups come from the projects themselves and
  were not independently re-measured.
- **Benchmarks:** lzbench figures are single-threaded on one EPYC machine
  with one corpus (silesia.tar). Relative ordering is more reliable than
  the absolute numbers.
