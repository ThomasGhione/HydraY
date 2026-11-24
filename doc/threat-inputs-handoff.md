# Threat inputs: handoff (2026-11-13, updated 2026-11-14, branch `threat-inputs`)

For Andrea. Thomas asked for threat inputs as the next NNUE lever, after weeks in which
the audits and the architecture changes (more output buckets, more king buckets, more
layers) gave nothing. The code is complete and verified for correctness; **no net has
been trained** and **nothing was SPRT'd**. The open question is whether the eval gain
pays for the cost per node: x2.15 at first, x1.76 after your six speed patches,
x1.67 after the engine campaign of section 4, and **x1.60** (bench6, P-core), **x1.76**
(bench6, E-core) and **x1.51** (datagen) with one accumulator for pieces and threats, as
Stockfish and PlentyChess have it. What is left is set by the feature set and the width
of the first layer, that is by training: Stockfish and PlentyChess both made the first
layer narrower when they added threats (section 4, "Other engines"), and section 5 gives
the speed of each training-side option, widths included.

On `threat-inputs`, branched from `dev` at `fdf4756`: `64fcf83` (threat inputs),
`beb787d`..`b909626` (your six patches, applied with `git am`), `4ca1354` (helpers they
left unused, shorter comments; `.text` byte-identical), `3dc0dff` (deferred-decode
kernel), `9b94adb` (no cancellation in one-move updates), `4cd4179` (eval cache slot
prefetched in `doMove`), `8d45227` (one accumulator for piece and threat features).
`dev` is untouched.

## 1. What a threat feature is

"Piece a on square s attacks piece v on square t", with the real occupancy (a slider
stops at its first blocker), for both colours of victim: attacks and defences. Seen from
each perspective like the piece features: pieces 0-5 its own, 6-11 the opponent's,
squares in its view with the king's file mirror. Not bucketed by the king.

Thomas's choices, from the options offered:

| question | chosen | alternatives offered |
|---|---|---|
| feature set | full set, attacks + defences (Stockfish's Full_Threats) | attacks on enemy pieces only (34,990 features); per-square "attacked/defended" flags inside the king buckets |
| threat weight precision | i8 at QA (as Stockfish, PlentyChess) | i16 (net ~144 MB) |
| first training | from scratch, 4.3.0 recipe | warm start from 4.4.0 with zero threat weights; both |

Excluded pairs, as in Stockfish: king on king; the pairs where the victim always attacks
back (P->B, P->Q, P->K, B->Q, R->Q, K->Q, the reverse feature is kept); two pieces of one
type that attack each other (pawns: only of opposite colours) are one feature, the one
with the lower attacker square. Kings stay in as attackers and victims (Alexandria and
bullet's example drop them; see option E). **70,050 features.**

## 2. What is in the branch

| file | content |
|---|---|
| `nnue/trainer/src/threats.rs` | the reference: tables, `index`, `map` over a perspective's view of the board. Pure std. |
| `nnue/threats.hpp` | its C++ port: `TABLE` of `{mask, first}` per (pair, attacker square) with an empty block for the other pairs, `ThreatPerspective::tryFeature` on Board pieces and squares, `threatFeaturePair` for both perspectives in one lookup |
| `nnue/trainer/src/bin/trainer_deep.rs` | `Inputs`: piece features, then threat features from `768*4` on; `l0t` weights `(1024, 70050)`, clip ±127/255, QAT rounding, zero init (or from `HYDRAY_INIT_NET`); concatenated to `l0w` in the graph; saved as i8 after `l2b`. Loader threads now `available_parallelism` clamped to 2..8. Test `inputs_match_the_oracle`. |
| `nnue/trainer/src/bin/sanity_deep.rs` | oracle: reads 4 layouts (8/16 buckets, with/without threats), `threat_features(fen)`, threats in `evaluate` and `evaluate_bullet`, zero `l0t` in `to_bullet` for an old net |
| `board/boardapi.cpp` | `recordThreats<Add>(piece, sq)` in `doMoveByKind`: the piece's attacks, the attacks on it, and the slider lines it cuts (`BEYOND` table), with the never-feature pairs masked out before the pops; `recordReplacedThreats` for the square of a capture; the child's key and TT prefetch come before all of it |
| `board/board.cpp` | `computeThreats` (Finny entry + every active threat, by piece type); in `ensureNnueAccumulatorClean` a perspective is incremental when the ancestor has the same basis (bucket and flip), its piece and threat rows in one pass (`applyMixed`), threat rows cancelled across a walk (`cancelThreats`, AVX2) and `applyFramePair1` for one move on both perspectives; a change of bucket or flip rebuilds from the Finny table and the threats from nothing |
| `nnue/accumulator.hpp` | `v[2][1024]` holds pieces + threats; `ThreatDelta threatDelta[64]` (4 bytes each) per frame; `applyMixedRows`: the deferred-decode kernel (`applyThreatList`) for the i8 threat rows with the source row and up to two i16 piece rows of each sign folded into the same tile pass; `applyRows` for piece rows alone (Finny sync) |
| `engine/sort/movepicker.hpp`, `engine/search/searcher.cpp`, `tt/tt.hpp` | AVX2 max in `nextMove`; `doMove(m, st, HashPrefetch)` in search, qsearch and ProbCut |
| `nnue/network_deep.hpp`, `nnue_deep.cpp` | `l0t` on huge pages (zero for an old net), re-laid out at load for the AVX2 kernel (the file keeps the bullet layout), `layoutForFileSize`; the forward reads one accumulator per perspective. The sum is modular i16 (Stockfish's too): it differs from an exact sum only where pieces + threats overflow i16, and `sanity_deep.rs` sums the same way |
| `huge_pages.hpp` | the huge-page allocator, moved out of `tt/tt.hpp` unchanged |
| `nnue/tools/*` | one decoder and accumulator for all (`position.hpp`, the engine's attack tables); reorder and l1census add the threat rows, deepcheck, holdoutloss and qatgap build piece-only accumulators and **refuse** threat nets instead of computing something wrong |

File layout: the old payload unchanged, then `l0t [70050][1024] i8`. 8 output buckets:
78,156,832 B payload, 78,156,864 B file; 16 buckets: 78,288,960 B. The trainer prints
`trainer_deep: 768x4hm + threats -> 1024 -> 16 -> 1, <OB> output buckets`, so the
`TRAINER_LINE` check of the existing notebooks fails on purpose.

Invariants worth knowing:
- every `recordThreats` call sees a consistent board with the recorded piece on its
  square: an en passant victim first, then the mover off `from`, then the mover (or the
  promoted piece) on `to`; on a capture that last step is `recordReplacedThreats`, the
  captured piece becoming the new one on a square that stays occupied, with the board
  after the move; castling records king off, king on, rook off, rook on;
- a frame whose move logged more than 64 threat deltas is marked
  `ACC_THREATS_OVERFLOW` and no later frame starts from before it. Never seen in tests;
- a frame's row holds the piece and the threat values together, under one `dirty` flag.

## 3. What was verified

- Engine == oracle **to the centipawn** on the 2,048 positions of `nnue/scale_ref.txt`,
  with a synthetic net (4.4.0 + random i8 threat weights in [-12, 12]): 0 differences.
- `cargo test --features train --bin trainer_deep`: bullet's inputs (from bulletformat
  records) == the oracle's features (from FENs), both perspectives, 9 positions.
- `nnue-selftest` on the synthetic net: incremental == scratch over 41,577 positions
  (threat values compared too). Mirrored pair equal.
- perft suite, `make test`, x86-64-v2 build (scalar forward == SIMD on every valid
  position), Windows cross build.
- With the 4.4.0 net (zero threat weights): bench6 448,769, identical per position.
- Your six patches, each against the commit before it (2026-11-14): bench6 per position
  with the 4.4.0 net, with the synthetic net above (580,995) and with yours (734,233);
  seeded datagen SHA (40 games, 8,000 nodes) with all three nets; raweval == oracle on
  the 2,048 FENs with both synthetic nets; selftest; perft. All identical, so no SPRT.
  The same checks for `3dc0dff`, `9b94adb` and `4cd4179`, and for every prototype of
  section 5 the 4.4.0 bench6 (448,769). Final tree (`4cd4179`, and before it `9b94adb` and
  `4ca1354`): x86-64-v2 build
  (scalar paths: bench6 with both nets, raweval == oracle),
  Windows build under wine (bench6 with both nets), `make test`, and GCC ASan+UBSan on
  bench6, selftest, perft, seeded datagen and the 37 UCI corner cases: clean but for the
  first issue of section 7.
- `8d45227` (one accumulator) against `4cd4179`: bench6 per position with the 4.4.0 net
  and with your synthetic net, seeded datagen SHA (4.4.0 and both synthetic nets),
  raweval == oracle on the 2,048 FENs, selftest, perft, `make test`; x86-64-v2 and
  Windows (wine) bench6 identical; ASan+UBSan clean (bench, perft, uci_edge 37/37). The
  oracle with the new wrapping sum gives the same 2,048 values as before.

**Not verified: the trainer graph on a GPU.** Without `cuda` bullet compiles a mock
device, so the `reshape`/`concat` of `l0w` and `l0t` is only type-checked. See option A.

## 4. Measurements

Machine loaded (Docker datagen at ~1500% CPU) during all of this; A/Bs are interleaved and
pinned to P-core 2 (`share/scripts/timing_ab.py`), so the ratios hold.

**Speed:** bench6:12, CPU time 0.355 s -> 0.75 s, **x2.1**, same nodes. Startup is
negligible (memset of 72 MB). RSS +72 MB.

**After your patches (2026-11-14, datagen paused, P-core 2, interleaved pairs,
`-march=sierraforest`):** each patch against the one before it, CPU time.

| patch | bench6:13, 4.4.0 net (20 pairs) | bench6:13, your net (12) | datagen, 6 games (10) |
|---|---|---|---|
| 0001 `nextMove` AVX2 max | +1.71% | +0.92% | +1.30% |
| 0002 compact offset table | +10.77% | +10.89% | +8.55% |
| 0003 shared tiled kernels | +2.36% | +2.24% | +1.74% |
| 0004 packed deltas, masks, `computeThreats` by type | +1.63% | +1.50% | +1.33% |
| 0005 cancel/apply out of line | +3.05% | +3.14% | +2.51% |
| 0006 TT prefetch in `doMove` | +0.73% | +0.79% | +0.83% |
| all six | **+21.69%** | +21.15% | +16.82% |

Every row is faster in every column (the 95% CI of each median ratio excludes 1).
Against `dev` (no threat code, same nodes): bench6:13 x2.148 -> **x1.757**, datagen
x1.933 -> **x1.635**.

**Engine campaign after the patches (2026-11-14).** The 4.4.0 net has zero threat
weights, so any part of the threat code can be switched off without changing an eval:
the tree stays identical and the time saved is that part's exact cost. Against `dev`
= 1.00, on `4ca1354`:

| switched off | x dev | its cost |
|---|---|---|
| nothing | 1.747 | |
| the row kernel (dst = src instead) | 1.311 | 0.436 (58% of the overhead) |
| + index lookup and cancellation | 1.163 | 0.148 (20%) |
| + recording in `doMove` | 1.067 | 0.096 (13%) |
| + threat reads in the forward | 1.038 | 0.029 (4%) |
| rest (bigger frames) | | 0.038 (5%) |

The kernel is **not** memory-bound: folding every feature onto 32 rows (32 KB, in L1)
saves 3.5% of the total, onto 1,024 rows 1.6%, against 30.7% for removing it. Row
accesses are concentrated (bench6 depth 17: 50% in the top 201 rows, 90% in 2,054,
95% in 3,604), 9.3 rows per perspective and update after cancellation, 7.7% of the rows
from rebuilds, 81% of the frames `doMove` pushes get computed, no overflow.

Kept (each identical as in section 3, bench6:13 CPU time against the commit before):
- `3dc0dff` **deferred decode, +1.93%** (24/24 pairs): `l0t` is re-laid out at load
  (AVX2 builds) so that a 16-bit lane holds weight i + 128 and weight 16 + i; the kernel
  sums raw lanes and their `vpsraw 8` and decodes once per tile. No row is widened:
  3 vector ops per 32 weights and row instead of 2 `vpmovsxbw` + 2 adds.
- `9b94adb` **no cancellation in one-move updates, +1.39%** (19/20).
- `4cd4179` **eval cache slot prefetched in `doMove` with the TT bucket, +1.00%** (16/16;
  +0.35% on datagen, +0.15% on an E-core). Not threat code, but an evaluation now costs
  1.7x and the probe was 5.8% of the samples.
Together +5.1% on bench6: x1.757 -> **x1.672** on a P-core; datagen x1.635 -> x1.594.

**On an E-core** (cpu 6, where `share/scripts/sprt.sh` plays: `CPUS=4-11`) threats cost
more: `4ca1354` x1.893, `4cd4179` **x1.844**. The same oracles there: kernel 0.552, index
0.119, recording 0.078, forward 0.026, rest 0.118. The rest is the copy of the threat
accumulator: without the dst = src copy as well, the engine with all threat work off is
as fast as `dev` (x0.995 E-core, x0.992 P-core). Reading src and writing dst is 4 KiB
per update; on a P-core it is 0.05 of `dev`'s time.

Also measured and dropped in a second round: the check features (a piece on the other
side's king, 6,496 of the 70,050) are never active in training data, because the
datagen records no position in check (`datagen.cpp:579`), so their rows stay zero in a
trained net; skipping them in the engine when the rows are zero is exact but -0.27%
(they only appear in frames in check, rare in search; patch kept outside the branch).
A Finny table for threat rebuilds (the diff against the last same-flip rebuild is 18.3
rows against 39.7) -0.35%, datagen -1.09%: the rebuild cost is the enumeration. Both
perspectives in one kernel call -0.2%. Eval cache of 8, 32, 64, 128 MiB: +-0.3% on
replayed games. clang PGO: +0.65% on the threat branch but +1.9% on `dev` too, so it
does not change the ratio (a build option for both, not a threat fix). Ceiling of
removing threat rebuilds entirely: +2.3% (P-core), +3.0% (E-core).

Measured and dropped (against the deferred kernel unless said): `vpmaddubsw`/shift
interleaves (+1.2% against the old kernel, below deferred); i16 rows (in-cache equal,
2x slower outside); tiles of 64/32 lanes -1.9%/-4.5%; row-major groups of 2/4/8 rows
-24%/-15%/-10%; fall-through switch per row count (microbench, worse); 4-row unroll and
interleaved sub/add 0.0%; row prefetch in the kernel -2.4%, on 4 lines 0.0%, on all 16
-4.1%, none at all 0.0% (0005's prefetch is neutral); L1-sized index tables -2.7%;
branchless push -0.4%; branchless `recordThreats` masks -0.4%; reusing the
"piece leaves `from`" deltas across siblings (only 24% of updates have such a sibling,
at most 11.9% of rows: not built).

The profile and the counts below are from before the patches.

Profile (SIGPROF sampler, bench6 positions at depth 17), where the extra time goes:

| part | share of the extra |
|---|---|
| `applyThreatRows` (the row kernel) | 47% |
| `threatFeature` (index per delta and perspective) | 16% |
| `recordThreats` in `doMove` | 10% |
| `computeThreats` (rebuilds from scratch) | 7% |
| forward (extra loads) | 5% |

Counts (kiwipete, depth 15): 11.5 threat rows per perspective per computed frame
(piece rows: 2-4); rebuilds from scratch were 28% of all threat rows before the
"same flip is enough" change; ~15% of incremental rows cancel (same feature added and
removed). Neither of those two reductions changed the time measurably, and the row counts
after them were not re-taken.

**It is not memory.** The weights are on huge pages (verified: AnonHugePages 151 MB =
TT + eval cache + l0t). A microbenchmark of 12 rows per call
(`vpmovsxbw` + `vpaddw`, 4 ymm accumulators per 64-wide tile) costs ~21 ns per 1 KiB row
with only 20 hot rows (all in cache), ~32 ns with 3,000. Alternatives, all slower on this
Panther Lake P-core: `vpmaddubsw` even/odd widening with interleaved rows (+5-20%),
row-major order (+80%), software prefetch (+12%).

Rows by attacker (row) and victim (column), share of all incremental rows:

|   | P | N | B | R | Q | K |
|---|---|---|---|---|---|---|
| P | 4.5 | 4.7 | 0.0 | 0.1 | 0.0 | 0.0 |
| N | 10.8 | 1.1 | 2.9 | 0.7 | 2.1 | 2.5 |
| B | 9.9 | 2.9 | 0.5 | 1.2 | 0.0 | 0.7 |
| R | 5.3 | 0.8 | 2.4 | 0.9 | 0.0 | 5.6 |
| Q | 15.1 | 4.7 | 3.5 | 3.7 | 0.4 | 3.3 |
| K | 4.9 | 0.7 | 1.5 | 2.7 | 0.0 | 0.0 |

The king as attacker or victim is ~22%, P->P 4.5%, pawn victims ~50%.

**Other engines (read from their sources and PRs, 2026-11-14).** None of them states its
slowdown, but the two that brought threats in also shrank the first layer at the same
time, so that the cost per node did not grow by much:

| engine | first layer before -> with threats | threat features | accumulators |
|---|---|---|---|
| Stockfish (SFNNv10, PR #6411, Nov 2025) | 3072 -> **1024** | 79,856 then; today 59,808 (no king as attacker or victim, P->P moved to a pawn-pair set of 4,560) | one, pieces + threats + pawn pairs |
| PlentyChess (PR #400, Oct 2025) | 1792 -> **640** (today 1024, 12 king buckets) | 79,856 then; today 59,808, the same as Stockfish | one |
| Reckless | 768 | 66,864: ours without the king as victim | two, as ours was |
| HydraY | 1024 -> 1024 | 70,050 | one since `8d45227` |

PlentyChess's PR #400 (1792 without threats against 640 with): +45.5 at fixed nodes,
-3.1 at 8+0.08 (H0), +1.8 at 40+0.4; merged on the LTC. Stockfish: STC, LTC and VVLTC
passed, +13.9 on ARM LTC. Alexandria +15.53 ±5.50 at 8+0.08 (PR #630). Our engine also
starts from a cheaper node (HydraY is about 2x Stockfish 18's nps per thread), so the same
absolute cost per update is a larger fraction of our time.

**Third round (2026-11-14), from what they do.**
- Kept: `8d45227` **one accumulator for pieces and threats** (Stockfish, PlentyChess):
  one row loaded and stored per update instead of two, frames 4 KiB smaller, the forward
  reads one row. +4.24% P-core (16/16), +4.49% E-core (12/12), datagen x1.566 -> x1.508.
  With it, a change of king bucket under the same flip rebuilds the threats too (before,
  they carried over): 66k rebuilds in bench6:13, 40 threat rows each, 20% of all threat
  rows (incremental updates: 8.9 threat rows and 2.5 piece rows).
- Stockfish's hybrid update (ancestor - old Finny entry + new entry, so the threats carry
  over a bucket change): -1.2% on an E-core, +-0.1% when limited to old entries within 2
  or 6 pieces of the ancestor. The old entry is rarely close.
- Finny entries that hold the threats too (bitset diff of the features): -0.79% P-core,
  +0.31% E-core.
- i8 piece weights in the same kernel (Stockfish's quantisation): on a copy of 4.4.0
  clipped to ±127, same tree, +0.06% P-core, -0.43% E-core. Nothing for us.
- ACC_MAX_WALK 4 -1.85%, 16 -0.35%; computing the top frame only -13.9%.
- Recording the threats lazily (at evaluation, not in `doMove`): 27% of the pushed frames
  are never computed, so at most ~1%: not built.

## 5. Options to continue

**A. Validate the trainer graph on Colab: required before any of the others.** About 15
minutes. Build a synthetic net from the shipped one (any net with non-zero threat weights
works; the script below is the one used for section 3) and start a run with
`HYDRAY_INIT_NET` on it: the trainer's init check compares bullet's forward with
`evaluate_bullet` on 5 positions and panics if they differ by 1 cp or more, so it tests
the concat order, the input mapping and the threat columns in bullet's real graph.
Then let it train 1 superbatch and run `sanity_deep` on the checkpoint's `quantised.bin`
(size 78,156,864 with 8 buckets). Also watch superbatch time and GPU memory: the
threat matrix adds 72M parameters (~1.2 GB of AdamW state) and a concat of ~300 MB per
step; if the GPU waits on batch preparation, the loader threads are the knob.

```python
# yours: 4.4.0 + random non-zero i8 threat weights in [-16, 16] -> synth.nnue
# (sha256 f5e33fae79ff96c2... from the 4.4.0 net)
import pathlib, random
body = pathlib.Path("nnue/net/hydray.nnue").read_bytes()[:6425632]
vals = [v for v in range(-16, 17) if v != 0]
tbl = bytes((vals[i & 31] & 0xFF) for i in range(256))
l0t = random.Random(42).randbytes(70050 * 1024).translate(tbl)
pathlib.Path("synth.nnue").write_bytes(body + l0t + (b"bullet" * 6)[:32])
```

No weight is zero, so a wrong feature index cannot hide behind one (the first synthetic
net, in [-12, 12] with numpy, had 4% zeros).

**B. Short probe, then decide (recommended).** Two arms with one
short schedule on v10 (e.g. 40 SB from scratch, the antidead phase settings), with and
without threats. For the arm without threats use `dev`'s trainer (same recipe), since this
branch's trainer always trains threats. Then:
1. SPRT at fixed nodes: does the threat net evaluate better, and by how much?
2. SPRT at 4+0.04, Threads=1: does it pay the x1.60 (x1.76 on the E-cores the SPRT uses)?
If (1) is large and (2) is positive or close, go to C. If (1) is small, stop or try E.
Costs ~40% of a full run.

**C. Full run directly.** The 4.3.0 recipe as in `colab_v10.ipynb` (160 SB antidead +
40 SB ft-antidead-qat, eval scale 356), with the new `TRAINER_LINE`. Highest Elo if
threats pay at all, highest cost if they do not. Before shipping such a net, `reorder`
must learn to permute the columns of `l0t` with the FT neurons (today it refuses threat
nets), and `holdoutloss` needs threat accumulators to compare losses.

**D. Cut the engine overhead (no retraining, identity-checkable): done, 2026-11-14.**
Your six patches took bench6 from x2.15 to x1.76 against `dev`, the campaign of section 4
to x1.67 and the single accumulator to x1.60; what is left in the engine is listed there
with what was tried.

**E/F/H. Fewer features or a narrower first layer (retraining needed): measured.**
Feature sets: each option built as the real engine change on `8d45227` and timed with
the 4.4.0 net, whose zero threat weights keep the tree identical (bench6 448,769 for
all; the reduced sets pass the selftest with your synthetic net), so the times are those
a net trained that way would have. bench6:13 CPU time against `dev`:

| feature set | features | P-core (10 pairs) | E-core (8 pairs) |
|---|---:|---:|---:|
| full (today) | 70,050 | 1.603 | 1.757 |
| no king as victim (Reckless) | 57,058 | 1.575 | 1.722 |
| no P->P | 69,840 | 1.570 | |
| no king as attacker or victim | 50,626 | 1.539 | 1.669 |
| **no king, no P->P (E; Stockfish's and PlentyChess's set)** | 50,416 | **1.507** | **1.631** |
| H: attacks only, victims of the other colour | 34,990 | 1.266 | 1.311 |
| H + no king + no P->P | 25,208 | 1.251 | |

Widths (what Stockfish and PlentyChess did): the engine built with `HIDDEN` = 768, 640,
512 and the 4.4.0 net cut to those widths (pairs k and k + 512 kept, l1 columns with them),
with and without threat code on the same cut net: same tree, so the threat overhead at
each width is exact. The cut nets search other trees than 4.4.0 (1.5-2.8M nodes against
0.88M at depth 13), so across widths the comparison is per node, not exact:

| first layer | dev, ns/node vs dev-1024 (P / E) | full set (P / E) | no king, no P->P (P / E) |
|---|---|---|---|
| 1024 | 1.000 / 1.000 | 1.61 / 1.76 | 1.51 / 1.64 |
| 768 | 0.88 / 0.86 | 1.37 / 1.50 | 1.29 / 1.39 |
| 640 | 0.86 / 0.82 | 1.32 / 1.44 | 1.24 / 1.33 |
| 512 | 0.80 / 0.75 | 1.21 / 1.27 | 1.13 / 1.18 |

The engine side of a width change is `HIDDEN` in `network.hpp` (every kernel and the
forward build at 768/640/512; the payload `static_assert`s must follow); the trainer and
`threats.rs`/`sanity_deep.rs` need the same width. For a feature set change `threats.rs`
and `threats.hpp` change together (the C++ static_assert pins the count) and the checks of
section 3 are re-run. Strength is the open question: PlentyChess's 640 threat net beat its
1792 net by 45 Elo at fixed nodes, which suggests a narrower threat net can pay at TC
where ours at 1024 may not. B with a 640 or 768 arm (set E) next to the 1024 full set
would answer it.

**G. Warm start from 4.4.0 (quick signal, lower ceiling).** `HYDRAY_INIT_NET` on the
shipped net works as is: `l0t` starts at zero and the start is the same function. A 40 SB
fine-tune gives a threat net within a day, but it is likely below what a run from scratch
reaches; it can serve as the threat arm of B if time is short.

Suggested order: A, then B (with a narrower arm, 640 or 768 with set E, if time allows),
then C on the shape B favours.

## 6. How to check a change

```sh
# engine vs oracle, raw evals (no normalisation), on the 2,048 reference FENs
grep -v '^#' nnue/scale_ref.txt | cut -d' ' -f2- > /tmp/fens.txt
mapfile -t F < /tmp/fens.txt
nnue/trainer/target/release/sanity_deep synth.nnue "${F[@]}" | tail -n +3 \
    | sed 's/.*: \(-\?[0-9]*\) cp (stm)/\1/' > /tmp/oracle.txt
/tmp/raweval synth.nnue < /tmp/fens.txt > /tmp/engine.txt
cmp /tmp/oracle.txt /tmp/engine.txt && echo identical
```

`raweval` is not in the repo: compile this with every engine source but `main.cpp`
(output to a scratch directory, not the repo root), plus `tbprobe.c`, linked with `-lgomp`:

```cpp
#include <iostream>
#include <memory>
#include <string>
#include "board/board.hpp"
#include "nnue/network_deep.hpp"
#include "nnue/nnue.hpp"
int main(int, char** argv) {
    auto net = std::make_unique<NNUE::Deep::NetworkDeep>();
    if (!NNUE::Deep::loadFromFile(argv[1], *net)) return 1;
    NNUE::activeNetwork = net.get();   // no normalisation: outputScale stays 400
    for (std::string line; std::getline(std::cin, line); ) std::cout << NNUE::evaluate(chess::Board(line)) << "\n";
}
```

Also: `./chess nnue-selftest synth.nnue 200`, `./chess perft suite`,
`script/engine_driver.sh bench6` (448,769 with the 4.4.0 net),
`cargo test --features train --bin trainer_deep` in `nnue/trainer`.

## 7. Known issues

- The selftest's "wN e4 extra" FEN has 33 pieces; `fenToBoard` rejects more than 32, so
  the engine evaluates an empty board there (the oracle does not). Older than this branch.
  UBSan shows it is also an out-of-bounds read: with `BUCKET_DIVISOR` 2 the empty board
  gets output bucket (0 - 2) / 2 = -1. The search never gets there (the missing-king
  guard returns first); only this selftest line does.
- Until a threat net exists, this branch is ~x1.67 slower than `dev` with the same evals
  (x1.84 on an E-core, x1.59 in datagen): do not merge it for speed-neutral reasons.
- Threat nets are ~78 MB: the embedded binary grows by the same amount, and the release
  workflow and the lichess bot copy should be checked for size limits.
