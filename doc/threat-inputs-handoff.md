# Threat inputs: handoff (2026-11-13, branch `threat-inputs`)

For Andrea. Thomas asked for threat inputs as the next NNUE lever, after weeks in which
the audits and the architecture changes (more output buckets, more king buckets, more
layers) gave nothing. The code is complete and verified for correctness; **no net has
been trained** and **nothing was SPRT'd**. The open question is whether the eval gain
pays for a measured 2.1x cost per node. Section 5 lists the ways to continue.

Commit `64fcf83` on `threat-inputs`, branched from `dev` at `fdf4756`. `dev` is untouched.

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
| `nnue/threats.hpp` | its C++ port; `threatFeature(p, flip, attacker, from, victim, to)` maps Board pieces and squares |
| `nnue/trainer/src/bin/trainer_deep.rs` | `Inputs`: piece features, then threat features from `768*4` on; `l0t` weights `(1024, 70050)`, clip ±127/255, QAT rounding, zero init (or from `HYDRAY_INIT_NET`); concatenated to `l0w` in the graph; saved as i8 after `l2b`. Loader threads now `available_parallelism` clamped to 2..8. Test `inputs_match_the_oracle`. |
| `nnue/trainer/src/bin/sanity_deep.rs` | oracle: reads 4 layouts (8/16 buckets, with/without threats), `threat_features(fen)`, threats in `evaluate` and `evaluate_bullet`, zero `l0t` in `to_bullet` for an old net |
| `board/boardapi.cpp` | `recordThreats<Add>(piece, sq)` in `doMoveByKind`: the piece's attacks, the attacks on it, and the slider lines it cuts (`BEYOND` table) |
| `board/board.cpp` | `computeThreats` (from scratch); in `ensureNnueAccumulatorClean` the threats of a perspective are incremental whenever the ancestor has the same flip (even if the king bucket changed), with add/sub cancellation |
| `nnue/accumulator.hpp` | `threat[2][1024]` and `ThreatDelta threatDelta[64]` per frame; `applyThreatRows` (tiled, i8 rows widened in registers) |
| `nnue/network_deep.hpp`, `nnue_deep.cpp` | `l0t` on huge pages (zero for an old net), `layoutForFileSize`, forward takes the two threat accumulators and adds them with `adds_epi16` (scalar: int32 sum, same result) |
| `huge_pages.hpp` | the huge-page allocator, moved out of `tt/tt.hpp` unchanged |
| `nnue/tools/*` | deepcheck, holdoutloss, qatgap, reorder still build piece-only accumulators: they **refuse** threat nets instead of computing something wrong |

File layout: the old payload unchanged, then `l0t [70050][1024] i8`. 8 output buckets:
78,156,832 B payload, 78,156,864 B file; 16 buckets: 78,288,960 B. The trainer prints
`trainer_deep: 768x4hm + threats -> 1024 -> 16 -> 1, <OB> output buckets`, so the
`TRAINER_LINE` check of the existing notebooks fails on purpose.

Invariants worth knowing:
- every `recordThreats` call sees a consistent board with the recorded piece on its
  square: captured piece first (and `occupancy` cleared on `to`), then the mover off
  `from`, then the mover on `to`; castling records king off, king on, rook off, rook on;
  a promotion records the promoted piece, never the pawn on the last rank;
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

**Not verified: the trainer graph on a GPU.** Without `cuda` bullet compiles a mock
device, so the `reshape`/`concat` of `l0w` and `l0t` is only type-checked. See option A.

## 4. Measurements

Machine loaded (Docker datagen at ~1500% CPU) during all of this; A/Bs are interleaved and
pinned to P-core 2 (`share/scripts/timing_ab.py`), so the ratios hold.

**Speed:** bench6:12, CPU time 0.355 s -> 0.75 s, **x2.1**, same nodes. Startup is
negligible (memset of 72 MB). RSS +72 MB.

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
# 4.4.0 + random i8 threat weights -> synth.nnue
import numpy as np
raw = open('nnue/net/hydray.nnue', 'rb').read()
body = raw[:6425632 if len(raw) == 6425664 else 6557760]
t = np.random.default_rng(1).integers(-12, 13, size=70050 * 1024).astype(np.int8).tobytes()
out = body + t
out += (b'bullet' * 11)[:(-len(out)) % 64]
open('synth.nnue', 'wb').write(out)
```

**B. Short probe, then decide (recommended).** Two arms with one
short schedule on v10 (e.g. 40 SB from scratch, the antidead phase settings), with and
without threats. For the arm without threats use `dev`'s trainer (same recipe), since this
branch's trainer always trains threats. Then:
1. SPRT at fixed nodes: does the threat net evaluate better, and by how much?
2. SPRT at 4+0.04, Threads=1: does it pay the x2.1?
If (1) is large and (2) is positive or close, go to C. If (1) is small, stop or try E.
Costs ~40% of a full run.

**C. Full run directly.** The 4.3.0 recipe as in `colab_v10.ipynb` (160 SB antidead +
40 SB ft-antidead-qat, eval scale 356), with the new `TRAINER_LINE`. Highest Elo if
threats pay at all, highest cost if they do not. Before shipping such a net, `reorder`
must learn to permute the columns of `l0t` with the FT neurons (today it refuses threat
nets), and `holdoutloss` needs threat accumulators to compare losses.

**D. Cut the engine overhead (no retraining, identity-checkable).** About a third of the
extra time is not the kernel: `threatFeature` (16%), `recordThreats` (10%), rebuilds
(7%). Ideas: one table `{mask, first}` per (attacker, victim, square) so that the index
is a load, a bit test and a popcount without branches; filtering the pair types once per
piece in `recordThreats` instead of per delta. Check each step with bench6 identical, the
oracle diff on the 2,048 FENs (section 6) and the selftest; then interleaved timing.
Realistic target: around x1.85. Worth doing whatever B says, if threats survive.

**E. Smaller feature set (retraining needed).** Dropping the king as attacker and victim
and P->P removes ~27% of the rows (x2.1 -> ~x1.8, more with D). This is close to bullet's
own example (`examples/advanced/inputs.rs` at the pinned rev: no king, pawns attack only
N and R). Change `threats.rs` and `threats.hpp` together (the C++ static_assert pins
the count), then re-run the checks of section 3.

**F. Narrower threat weights (architecture change, speculative).** Threats feeding only
half of the 1024 accumulator entries would halve the kernel cost and the file. Untested
anywhere I know of; in bullet it needs `l0t` padded with fixed zero rows or a split
accumulator. Only if B shows a good gain at fixed nodes and a bad one at time control.

**G. Warm start from 4.4.0 (quick signal, lower ceiling).** `HYDRAY_INIT_NET` on the
shipped net works as is: `l0t` starts at zero and the start is the same function. A 40 SB
fine-tune gives a threat net within a day, but it is likely below what a run from scratch
reaches; it can serve as the threat arm of B if time is short.

Suggested order: A, then B (with D in parallel on the engine side), then C or E.

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
- Until a threat net exists, this branch is ~x2.1 slower than `dev` with the same evals:
  do not merge it for speed-neutral reasons.
- Threat nets are ~78 MB: the embedded binary grows by the same amount, and the release
  workflow and the lichess bot copy should be checked for size limits.
