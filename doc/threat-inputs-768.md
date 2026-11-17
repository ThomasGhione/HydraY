# Threat inputs, 768-wide first layer (branch `threat-inputs-768`)

Branched from `threat-inputs` at `7329139` (63,554 threat features). The only
architectural change is the width of the feature transformer:

    (768x4hm + 63,554 threats -> 768)x2 -> pairwise -> 768 -> 16 -> 1, 8 or 16 output buckets

Pairwise halves are 384 (`PAIRWISE_OUT`), l1's input is 768.

| | 1024 | 768 |
|---|---:|---:|
| `l0w` + `l0b` + l1 + tails, 8 buckets | 6,425,632 B | 4,819,488 B |
| `l0t` (63,554 x width, i8) | 65,079,296 B | 48,809,472 B |
| file, 8 buckets, threats | 71,504,960 B | 53,628,992 B |
| file, 16 buckets, threats | 71,637,056 B | 53,728,320 B |

## What changed

- One width constant on each side: `HIDDEN` in `nnue/network.hpp` (engine) and
  `HIDDEN` in `nnue/trainer/src/bin/sanity_deep.rs` (oracle; `trainer_deep.rs` reads
  it from there). The accumulator, the threat kernels, the forward (scalar, AVX2,
  VNNI), `reorder`, `l1census`, the Finny table and the file layout follow it; the
  payload `static_assert`s in `network_deep.hpp` pin the sizes above.
  `static_assert(PAIRWISE_OUT % 32 == 0)` guards the vectorised pairwise pass.
- `nnue/tools/holdoutloss.cpp`: the single-layer format is 1024-wide; its size assert now
  covers both widths (no 768 single-layer net exists).
- `nnue/trainer/colab_v10_threats.ipynb`: branch `threat-inputs-768`, net ids
  `hydray-threats768-v10-*` (Drive never mixes them with the 1024 or 60k nets),
  `NET_FILE_BYTES = 53_628_992`, trainer banner `... threats -> 768 -> 16 -> 1`, asserts that
  the clone is the 768 build.
- `nnue/tools/cut_width.py`: cuts a 1024-wide net without threats to 768, keeping the
  pairs (j, j + 512), j < 384.

## The embedded net

`nnue/net/hydray.nnue.sha256` names `hydray-threats768-v10-ft-antidead-steps-qat-80sb-reordered.nnue`
(sha256 `bc3c5971...`), the first trained 768 threat net, fetched from the Hugging Face
bucket by `make`. Until 2026-11-17 the embedded net was a placeholder: **the 4.4.0 net cut
to 768 with `cut_width.py`** (sha256 `fdf0eaba...`, also in the bucket), not a trained net
(a 1024-wide file cannot be loaded by this build); compare nothing against it.

## Verified (2026-11-15)

- `make test` (77 asserts), `perft suite`, `nnue-selftest` (incremental == scratch) on the
  placeholder and on a synthetic 768 threat net (placeholder + random non-zero i8
  threat weights in [-16, 16]), also an `x86-64-v2` build.
- engine == oracle (`sanity_deep`) on the 2,048 FENs of `nnue/scale_ref.txt`, for the
  placeholder and for the synthetic threat net.
- `reorder` on the synthetic net (`.bin` pool): the reordered net evaluates the same 2,048 FENs
  identically; `l1census`, `datagen` run.
- `cargo test --features train --bin trainer_deep`, `cargo build -r --bin sanity_deep`.

**Not verified: the trainer graph on a GPU** (same as the 1024 build): without `cuda`,
bullet compiles a mock device. The first superbatch on Colab is the test.
