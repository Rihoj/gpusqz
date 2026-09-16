# Predictive modeling study

Could gpusqz compress better by *predicting* bytes, the way a language
model predicts tokens, and still run on the GPU? Any model that assigns a
probability to the next symbol becomes a lossless compressor once an
entropy coder (such as gpusqz's rANS) codes the real symbol against that
probability. The output size is the model's cross-entropy on the data, so
a better predictor is a smaller file. This study measures how much there
is to gain under gpusqz's constraints, before any kernel is written. It
changes nothing in gpusqz itself.

Measured 2026-09-16 on c777280 (format 2). All numbers are sizes, not
speeds, so GPU contention doesn't affect them (ollama held 12GB of VRAM at
the time). The tools are in `study/predictive/` and
`study/predictive/run_study.sh` reproduces every table.

## Short answer

- **Pure prediction beats today's files by a lot.** A small
  context-mixing model coding raw bytes, reset every chunk, with each chunk
  split into 32 lanes that share one model in lockstep, makes enwik8 20%
  (64KB chunks) to 28% (1MB) smaller than `gpusqz`, and the headers corpus
  22% to 27% smaller. With under 2MB of model state per chunk the gain is
  still 15–17% at 64KB.
- **That lands near zstd -19 and xz -9, not beyond them.** The best
  GPU-shaped configuration (1MB chunks, 85MB of model per chunk) sits
  between `xz -9` and `zstd -19` on both corpora. The small one that fits
  many chunks in flight is worse than `bzip2 -9`. The case for it is those
  ratios at GPU speed, where xz and zstd -19 compress at 1–3 MB/s.
- **Better literal modeling on top of LZ is small.** Literals are 11–31%
  of a gpusqz file. The best literal model found saves 1.8–4.1% of the
  file, and only by conditioning on output bytes that the two-phase decoder
  doesn't have yet. What today's decoder could use saves at most 0.6%.
- **Priors help only when they match the data.** A model pre-trained on
  the same kind of data cuts 64KB chunks by a further 6–18%. One trained
  on the other corpus made files 6–7% *larger* (as implemented; see below).
- **Speed is the open question.** The model runs at about 1–3 MB/s per
  CPU core, and coding is symmetric. Whether a 32-lane GPU version reaches
  tens of MB/s has not been measured, and it decides whether any of this
  is useful.

## Where a gpusqz file's bits go

`bit_account` decodes a `.gsz` exactly as `gpusqz_refdec` does and prices
every rANS symbol at its exact code length, log2(4096/freq), and every
bypass bit at one bit. What is left of each chunk's payload (86–104 bytes
per chunk, against a 136-byte state header whose initial states carry some
information) is the rANS framing. That remainder being close to the header
size confirms the pricing.

Share of the whole file:

| file | literals (share of input) | literals | lit_len | match_len | offset codes | offset extra bits | framing, tables |
|---|---|---|---|---|---|---|---|
| enwik8 `speed` (0.3846) | 19.5% | 31.1% | 9.1% | 11.7% | 12.6% | 35.0% | 0.5% |
| enwik8 `balance` (0.3734) | 13.6% | 23.3% | 8.1% | 13.0% | 14.4% | 41.1% | 0.1% |
| enwik8 `ratio` (0.3583) | 6.5% | 12.5% | 5.9% | 14.6% | 16.7% | 50.3% | 0.1% |
| headers `speed` (0.2041) | 8.4% | 25.9% | 8.1% | 17.8% | 14.2% | 33.2% | 0.9% |
| headers `balance` (0.1907) | 5.4% | 18.8% | 7.2% | 19.3% | 16.0% | 38.4% | 0.3% |
| headers `ratio` (0.1808) | 3.0% | 11.4% | 5.6% | 20.6% | 17.8% | 44.5% | 0.2% |

("lit_len" and "match_len" include their extra bits.)

On text, a gpusqz file is mostly the description of its matches: at
`ratio`, enwik8's 12.3M matches average 7.6 bytes and cost about 20 bits
each, half of it raw offset bits. A better literal model can only reach
the literal column. A model that replaces the LZ parse competes with all
of it.

## Better literals on top of LZ

`lit_models` re-prices the literals `bit_account` dumped. Static models use
one table per context per table group, priced exactly as gpusqz prices
them (8-bit quantised counts, 12-bit frequencies, tables coded by
`table_codec.h`). The calibration model (`lits1`, today's rule) reproduces
`bit_account`'s literal bytes exactly on all six files.

Change in *whole-file* size against today's literal coding:

| literal model | decodable today? | enwik8 `speed` | enwik8 `ratio` | headers `speed` | headers `ratio` |
|---|---|---|---|---|---|
| order-2 on the lane's literals, 1024 contexts | yes | −0.28% | −0.04% | −0.15% | +0.05% |
| order-2 on the lane's literals, 4096 contexts | yes | −0.61% | −0.02% | −0.29% | +0.36% |
| order-1 on the *output* byte before the literal | no | −2.5% | −1.3% | −1.4% | −0.8% |
| order-2 on output bytes, 1024 contexts | no | −3.5% | −1.6% | −2.2% | −1.0% |
| order-2 on output bytes, 4096 contexts | no | −4.1% | −1.8% | −2.9% | −1.1% |
| adaptive CM (orders 0–4 + word) on output bytes, per chunk | no | −2.2% | −1.8% | −1.6% | −1.8% |
| adaptive CM (orders 0–3) on the lane's literals, per chunk | yes | +1.5% | +0.2% | +1.2% | +0.04% |
| adaptive CM (orders 0–2) on the lane's literals, per lane run | yes | +5.2% | +0.9% | +4.8% | +0.6% |

- **The context source matters more than the model.** The byte just
  before a literal in the output is a far better context than the previous
  literal in the lane's run: after a match, the previous literal is
  unrelated text. Static order-2 tables on output bytes do as well as
  adaptive context mixing.
- **Output-byte contexts are not decodable by today's decoder.** It
  decodes all literals before rebuilding any output. Using them would mean
  decoding literals during reconstruction, which gives up the two-phase
  decode.
- **Adaptive models inside one lane lose.** At 64KB, each lane's run holds
  at most about 2KB of literals, too few to learn from.
- What is decodable today saves at most 0.6%. That is not worth a format
  change.

## Drop LZ: context mixing on raw bytes

`cm_raw` runs a small lpaq-style model over the original bytes. It has
context orders 0, 1, 2, 3, 4 and 6, a word context, and a match model,
mixed by a logistic mixer, with probabilities clamped to 12 bits as rANS
would need. It is reset at every chunk, or never. With `--lanes 32` a chunk
is split into 32 segments coded in lockstep, as gpusqz's lanes step. All
32 share one model. Each segment's contexts come from its own bytes, and
matches may only point at bytes already decoded. Update order within a
step is lane order, with no delay, which is slightly optimistic. The match
model's hash index has one slot per chunk byte (2^16 entries at 64KB).
The mixer learning rate (0.01) and adaptation cap (60) were picked from a
small grid on enwik8's first 10MB, part of the same corpus reported here.
The differences across that grid were about 1%.

Sanity check: with no resets, enwik8 codes to 0.2156 (1.72 bpc), in the
range of lpaq-class models, which is what this model is.

| corpus, chunk | gpusqz | CM, one stream per chunk | CM, 32 lanes in lockstep |
|---|---|---|---|
| enwik8, 64KB | 0.3846 | 0.3002 (−22%) | 0.3059 (−20%) |
| enwik8, 256KB | 0.3734 | 0.2724 (−27%) | 0.2792 (−25%) |
| enwik8, 1MB | 0.3583 | 0.2516 (−30%) | 0.2594 (−28%) |
| enwik8, never reset | – | 0.2156 | – |
| headers, 64KB | 0.2041 | 0.1513 (−26%) | 0.1594 (−22%) |
| headers, 256KB | 0.1907 | 0.1313 (−31%) | 0.1411 (−26%) |
| headers, 1MB | 0.1808 | 0.1201 (−34%) | 0.1318 (−27%) |
| headers, never reset | – | 0.1065 | – |

Splitting a chunk into 32 lockstep lanes makes files 2–10% larger than one
stream per chunk, but most of the gain survives the layout the GPU needs.

For scale, CPU compressors on the same files (whole file, one stream;
speeds are rough, measured with eight runs sharing six cores):

| compressor | enwik8 | headers | compress |
|---|---|---|---|
| `xz -9 -T1` | 0.2486 | 0.1261 | ~1–3 MB/s |
| `zstd -19 --long=27` | 0.2650 | 0.1304 | ~1–3 MB/s |
| `zstd -19` | 0.2695 | 0.1337 | ~2 MB/s |
| `bzip2 -9` | 0.2900 | 0.1481 | ~18 MB/s |

The 32-lane model at 1MB (0.2594, 0.1318) falls between `xz -9` and
`zstd -19`.

### Model size

Model state per chunk decides how many chunks fit in flight
([Lesson 1](performance-history.md#lessons)). Sizes are estimated from the
tables: one hashed table of 2^slots × 4 bytes per hashed context (five
with the word context), about 0.4MB of fixed tables (order 1, match
probabilities, mixer weights), and a match index of about 4 bytes per
chunk byte. All runs below use 32 lanes.

| chunk, hash slots | state per chunk | chunks in 4GB | enwik8 | headers |
|---|---|---|---|---|
| 64KB, 2^16 | ~1.9MB | ~2100 | 0.3251 (−15%) | 0.1688 (−17%) |
| 64KB, 2^16, orders 0,1,2,4, no word | ~1.4MB | ~2900 | 0.3452 (−10%) | 0.1802 (−12%) |
| 64KB, 2^18 | ~5.6MB | ~700 | 0.3114 (−19%) | 0.1619 (−21%) |
| 64KB, 2^20 | ~21MB | ~190 | 0.3059 (−20%) | 0.1594 (−22%) |
| 1MB, 2^18 | ~9.4MB | ~430 | 0.2776 (−23%) | 0.1405 (−22%) |
| 1MB, 2^22 | ~85MB | ~48 | 0.2594 (−28%) | 0.1318 (−27%) |

The 64KB, 2^16 configuration fits the ~1000-chunks-in-flight target that
gpusqz's throughput depends on, and keeps most of the gain over gpusqz.
But its ratio is behind `bzip2 -9`. Getting near xz takes big chunks and
big models, and so few chunks in flight.

### Priors

A prior is a model trained ahead of time. Each chunk starts from a copy of
it instead of from nothing, as a model shipped inside the binary would.
"Same domain" trains on enwik8's first 16MB and evaluates chunks in its
second half ([50MB, 100MB)), disjoint from the training data. "Cross
domain" trains on the other corpus.

| run | no prior | with prior |
|---|---|---|
| enwik8 2nd half, 64KB, 2^20, 1 stream | 0.2999 | 0.2472 (−18%) |
| enwik8 2nd half, 1MB, 2^22, 1 stream | 0.2516 | 0.2299 (−9%) |
| enwik8 2nd half, 64KB, 2^16, 32 lanes | 0.3248 | 0.3057 (−6%) |
| enwik8, 64KB, prior from headers | 0.3002 | 0.3191 (+6%) |
| headers, 64KB, prior from enwik8 | 0.1513 | 0.1622 (+7%) |

A matched prior gives 64KB chunks most of what 1MB chunks get, which
matters because small chunks are what keep the GPU busy. A mismatched one
hurts. The prior's counters arrive with their adaptation counts saturated,
so they unlearn slowly. Capping those counts when copying the prior would
probably remove most of the loss, but that wasn't measured. A single
generic prior shipped for all data (the "ship a small language model"
idea) is not supported by these numbers. Priors per data type, chosen per
file or per batch, would be the version to test.

## What it would take on the GPU

None of this is measured. It is the reasoning for what to test next.

- **Determinism.** Both backends must write identical files, so the
  model's arithmetic has to be integer: lpaq's fixed-point stretch/squash
  tables and integer mixer weights. The float mixer used here would give
  different results on different GPUs. How much ratio an integer mixer
  costs hasn't been measured here; measure it before building anything.
- **Shared model across lanes.** All 32 lanes read the model for their
  current bit, then all update it after a sync. Integer adds are the same
  in any order, but two lanes updating the same slot in one step must be
  combined deterministically (e.g. applied in lane order after the sync).
  This is the cross-lane-write invariant in CLAUDE.md, applied on every
  step.
- **Steps.** Coding bits means 8 lockstep steps per byte. A 64KB chunk
  over 32 lanes is about 16K steps, each doing about 7 table reads and
  writes per lane plus the mixer, against about 1 table lookup per step
  in today's rANS decoder. Decode would be orders of magnitude slower
  than today's multi-GB/s kernel, and compression would be as slow as
  decode. Coding nibbles (2 steps per byte, 16-symbol tables built from
  the binary nodes) is the obvious mitigation to measure.
- **The encoder runs backwards.** rANS encodes in reverse, but the model
  must predict forwards. The encoder would run the model forward, store
  each bit's 12-bit probability (a byte and a half per bit, so about 12
  bytes per input byte in scratch), then rANS-code in reverse.
- **Match finding moves into the model.** The match model needs a hash
  index per chunk, like today's match finder, but only for predictions,
  not a parse.

## Recommendation

1. **Don't pursue better literal models on top of LZ.** What today's
   decoder can use is worth at most 0.6%. The rest needs a decoder
   restructure for 1–4%.
2. **The prize is a context-mixing profile.** Expect 15–28% smaller than
   today's profiles on text, reaching roughly `zstd -19` / `xz -9` ratios
   only with big chunks and models. It is worth building only if the GPU
   runs it well above the 1–3 MB/s those compress at. Its decode speed
   would be the same as its compression speed, where zstd decodes at
   hundreds of MB/s. The go/no-go experiment is a CUDA prototype of the
   32-lane lockstep coder with an integer mixer, run on a full batch, at
   two points: 64KB with 2^16 slots (occupancy-friendly) and 1MB with
   2^18 (ratio). Measure MB/s and VRAM. If neither clearly beats CPU
   `zstd -19` on compression time at a comparable ratio, record the idea
   in "Measured and rejected".
3. **Test matched priors only after the prototype.** They are the
   cheapest way to make small chunks compress like big ones, but only
   matter if step 2 succeeds.

## Tools

In `study/predictive/`, built with plain `g++` (see `run_study.sh`), not
part of the CMake build or the release:

- `bit_account.cpp`: bit accounting for a `.gsz` file, and the literal
  dump.
- `lit_models.cpp`: static and adaptive literal models over the dump.
- `cm.h`, `cm_raw.cpp`: the context-mixing model and the raw-byte runs
  (resets, 32-lane lockstep, priors).

Corpora: enwik8 (the first 100MB of the 2006 English Wikipedia dump) and
"headers", every `*.h` under this machine's `/usr/include`, concatenated
(43MB, smaller than the 1GB corpus in [Benchmarks](benchmarks.md)).
