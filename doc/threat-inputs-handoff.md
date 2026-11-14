# Threat inputs: handoff (2026-11-13, updated 2026-11-14, branch `threat-inputs`)

For Andrea. Thomas asked for threat inputs as the next NNUE lever, after weeks in which
the audits and the architecture changes (more output buckets, more king buckets, more
layers) gave nothing. The code is complete and verified for correctness; **no net has
been trained** and **nothing was SPRT'd**. The open question is whether the eval gain
pays for the cost per node: x2.15 at first, x1.76 after your six speed patches,
**x1.67** (bench6, P-core), **x1.84** (bench6, E-core) and **x1.59** (datagen) after the
engine campaign of section 4. What is
left is set by the feature set and the threat width, that is by training: section 5
gives the exact speed of each training-side option.

On `threat-inputs`, branched from `dev` at `fdf4756`: `64fcf83` (threat inputs),
`beb787d`..`b909626` (your six patches, applied with `git am`), `4ca1354` (helpers they
left unused, shorter comments; `.text` byte-identical), `3dc0dff` (deferred-decode
kernel), `9b94adb` (no cancellation in one-move updates), `4cd4179` (eval cache slot
prefetched in `doMove`). `dev` is untouched.

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
| `board/board.cpp` | `computeThreats` (from scratch, by piece type); in `ensureNnueAccumulatorClean` the threats of a perspective are incremental whenever the ancestor has the same flip (even if the king bucket changed), with add/sub cancellation (`cancelAndApplyThreats`, AVX2) and `applyThreatPair1` for one move on both perspectives |
| `nnue/accumulator.hpp` | `threat[2][1024]` and `ThreatDelta threatDelta[64]` (4 bytes each) per frame; `applyWeightRows`: register-tiled AVX2 kernels, the deferred-decode one (`applyThreatList`) for i8 threat rows and the plain one for i16 piece rows |
| `engine/sort/movepicker.hpp`, `engine/search/searcher.cpp`, `tt/tt.hpp` | AVX2 max in `nextMove`; `doMove(m, st, HashPrefetch)` in search, qsearch and ProbCut |
| `nnue/network_deep.hpp`, `nnue_deep.cpp` | `l0t` on huge pages (zero for an old net), re-laid out at load for the AVX2 kernel (the file keeps the bullet layout), `layoutForFileSize`, forward takes the two threat accumulators and adds them with `adds_epi16` (scalar: int32 sum, same result) |
| `huge_pages.hpp` | the huge-page allocator, moved out of `tt/tt.hpp` unchanged |
| `nnue/tools/*` | deepcheck, holdoutloss, qatgap, reorder still build piece-only accumulators: they **refuse** threat nets instead of computing something wrong |

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
- the psq and threat values of a frame share the `dirty` flag.

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

Other engines (gains net of their own slowdown, which none of them states): Alexandria
+15.53 ±5.50 at 8+0.08 and +13.43 ±4.96 at 40+0.4 (PR #630); Stockfish SFNNv10 passed
STC, LTC and VVLTC, ARM LTC estimate +13.9; PlentyChess was first. Their piece
accumulators are wider than ours, so the relative cost of threats is likely smaller there.

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
2. SPRT at 4+0.04, Threads=1: does it pay the x1.67 (x1.84 on the E-cores the SPRT uses)?
If (1) is large and (2) is positive or close, go to C. If (1) is small, stop or try E.
Costs ~40% of a full run.

**C. Full run directly.** The 4.3.0 recipe as in `colab_v10.ipynb` (160 SB antidead +
40 SB ft-antidead-qat, eval scale 356), with the new `TRAINER_LINE`. Highest Elo if
threats pay at all, highest cost if they do not. Before shipping such a net, `reorder`
must learn to permute the columns of `l0t` with the FT neurons (today it refuses threat
nets), and `holdoutloss` needs threat accumulators to compare losses.

**D. Cut the engine overhead (no retraining, identity-checkable): done, 2026-11-14.**
Your six patches took bench6 from x2.15 to x1.76 against `dev`, the campaign of section 4
to x1.67; what is left in the engine is listed there with what was tried.

**E/F/H. Fewer features or a narrower threat layer (retraining needed): measured.**
Each option below was built as the real engine change (on `9b94adb`) and timed with the 4.4.0 net,
whose zero threat weights keep the tree identical (bench6 448,769 for all; the reduced
sets pass the selftest with your synthetic net), so the times are those a net trained
that way would have. bench6:13 CPU time against `dev`, 12 pairs, same session:

| option | x dev |
|---|---|
| full set (today, `9b94adb`) | 1.707 |
| no king as attacker or victim | 1.627 |
| no P->P | 1.665 |
| no king, no P->P (E; close to bullet's example) | 1.598 |
| threats on half the accumulator, 512 of 1024 (F) | 1.528 |
| half + no king + no P->P | 1.455 |
| **H: attacks only, victims of the other colour (34,990 features)** | **1.363** |
| H + no king + no P->P | 1.354 |
| H + half | 1.317 |
| H + half + no king + no P->P | 1.303 |
| threat weights clipped to ±31 / ±15, 4 / 8 rows per `vpsraw` | 1.715 / 1.747 (worse) |

Defences are about half of the threat cost; the king and P->P are small. H is the set
that was offered at the start next to the full one. For any of them change `threats.rs`
and `threats.hpp` together (the C++ static_assert pins the count) and re-run the checks
of section 3; F needs `l0t` of 512 columns in the trainer and the engine. Strength is
the open question: B with H as a third arm would answer it.

**G. Warm start from 4.4.0 (quick signal, lower ceiling).** `HYDRAY_INIT_NET` on the
shipped net works as is: `l0t` starts at zero and the start is the same function. A 40 SB
fine-tune gives a threat net within a day, but it is likely below what a run from scratch
reaches; it can serve as the threat arm of B if time is short.

Suggested order: A, then B (with an H arm if time allows), then C on the set B favours.

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
