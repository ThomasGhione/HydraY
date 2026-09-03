# HydraY — Elo Roadmap and Hot-Path Audit

**Date:** 2026-09-01 · **Tree:** `a6e3bf3` (dev) · **External strength:** 3132.4 ±14.9
on the Stockfish `UCI_Elo` bracket · **bench6 @ d12:** 1,953,820

This audit was written by reading every Tier-1 and Tier-2 file end to end, not by
pattern-matching. Every claim below is tagged with how it is known:

| tag | meaning |
|---|---|
| **[M]** | Measured today, numbers reproduced in this document |
| **[C]** | Read directly from the code; the mechanism is verifiable by inspection |
| **[H]** | Hypothesis. Plausible, unmeasured, could be wrong |
| **[X]** | Already tested and settled — do not redo |

Files read in full: `searcher.cpp` (1503), `sorter.cpp`, `movepicker.hpp`,
`searchruntime.hpp`, `movelist.hpp`, `corrhist.hpp`, `search_constants.hpp`,
`accumulator.hpp`, `tt.hpp` (layout/probe/store), `boardapi.inl` (doMove path),
`time_manager.{hpp,cpp}`, `move_generator.cpp` (entry points).

---

## 1. The one theme that matters: material and evaluation are different units

Today's `+82.3` external Elo came from a single realisation: **`PIECE_VALUES` and
SEE are handcrafted-era centipawns; the NNUE output is on a hotter scale.** Any
expression that adds one to the other is dimensionally wrong.

Measured ratio, this build, 376 piece-removal pairs at depth 8 **[M]**:

| phase (non-pawn material) | n | material→eval ratio |
|---|---|---|
| opening / middlegame (>5000) | 134 | 1.99 |
| late middlegame (3000–5000) | 50 | 2.89 |
| early endgame (1500–3000) | 69 | 2.77 |
| endgame (<1500) | 123 | 3.20 |
| **global** | 376 | **2.62** |

Against Stockfish on identical positions at equal depth: **2.256×, with 92.4% sign
agreement** — the unit differs, the judgement does not.

### Sites fixed

| site | commit | result |
|---|---|---|
| qsearch node delta gate | `85e7c2e` | part of +105.86 |
| qsearch per-move captured-value filter | `85e7c2e` | " |
| qsearch per-move SEE filter | `85e7c2e` | " |
| ProbCut SEE-vs-margin filter | `73b877d` | **unvalidated** — see §6 |
| factor 220 → 260 | `a6e3bf3` | +15.42 ±7.66, H1 accepted |

### Sites still wrong — the highest-value work left

**1.1 `REPETITION_DRAW_ADVANTAGE_THRESHOLD` — DONE, and it was worth nothing [M]**

> **Resolved 2026-09-03 (`5291b28`). The mechanism was real; my ranking was wrong.**
> Instrumented before committing, over 200 searches on late-game positions
> replayed with full move history:
>
> ```
> repetition scans        136,860,059
> twofold  -> score 0       9,307,319
> threefold path                    1      <-- the branch this threshold governs
> outcome changed by fix            0
> ```
>
> The threefold branch runs about once per 200M nodes. Committed as correctness,
> **no SPRT** — it would measure nothing. The lesson generalises: *a dimensional
> error in a cold branch is still worth nothing.* Rank by
> mechanism **× firing rate**, and measure the second factor first. This is the
> third time in two days that instrumentation killed a plausible idea for ~20
> minutes instead of a 3-hour SPRT (after the corrhist caps and the data-gap
> hypothesis).
>
> **The same measurement found something better — see §2.5.**

*Original analysis, kept for the record:*

```cpp
// searcher.cpp:18
constexpr int32_t REPETITION_DRAW_ADVANTAGE_THRESHOLD = PAWN_VALUE / 2;   // material: 50

// searcher.cpp:156 — compared against an EVAL-scale quantity
const int32_t staticEval = Evaluator::evaluate(b);
outScore = (std::abs(staticEval) <= REPETITION_DRAW_ADVANTAGE_THRESHOLD) ? 0 : ...
```

This decides "is the position level enough that a repetition is acceptable?" The
threshold is half a pawn of *material*; `staticEval` is *eval*. At 2.6× the test
fires at roughly 0.19 pawns instead of 0.5 — so the engine treats far more
positions as "level" and accepts repetitions in positions where it is meaningfully
better. **This is exactly the bug class that just paid +82, in a code path that
decides whole half-points.** Fix: `materialToEval(PAWN_VALUE / 2)`.
*Highest-confidence untested item in this document.*

**1.2 Aspiration windows — TESTED AND DEAD [M]. And it reclassifies the rest of §1.**

> **Resolved 2026-09-03. Rescaled by the measured 2.6x. The mechanism worked
> exactly as predicted and the engine still got weaker.**
>
> | | before | after 2.6x |
> |---|---|---|
> | first window OK | 50.2% | **76.7%** |
> | re-searches / iteration | 0.787 | **0.309** |
> | fail low / high | 868 / 850 | 322 / 341 |
> | bench6 nodes | 1,953,820 | **3,344,881 (+71%)** |
>
> SPRT: **−7.72 ±10.78, LOS 7.30%, LLR −1.04 @ 1800** (stopped on trend). A wider
> root window prunes less at *every* node, and that cost buried the 61% cut in
> re-searches.
>
> Then the deciding measurement — sweeping the factor both ways:
>
> | factor | 50% | 70% | 85% | **100%** | 130% | 160% | 200% | 260% |
> |---|---|---|---|---|---|---|---|---|
> | bench6 vs base | +21.7% | +10.6% | +26.3% | **0.0%** | +15.4% | +41.6% | +55.5% | +71.2% |
>
> **The current values are a local minimum in both directions.** They are not
> mis-set; the search evolved around them.
>
> ### ⚠️ The lesson, which invalidates part of this document's premise
>
> **"Never rescaled after the eval scale changed" is a hypothesis, not a defect.**
> Two different things were being conflated:
>
> 1. **Dimensional error** — an expression that *adds material to eval*
>    (`standPat + capturedValue`, `see < PROBCUT_MARGIN`). Objectively wrong at
>    any magnitude. This is what paid **+82**.
> 2. **Magnitude drift** — a constant that is dimensionally *consistent* (eval
>    compared against eval) whose nominal chess meaning shifted. Here the value
>    may be empirically fine regardless of what the comment claims it means.
>
> The aspiration window is type 2, and so are §1.3 and §1.4 below — **downgrade
> them accordingly.** The type-1 seam is now fully mined: four sites fixed, the
> fifth (§1.1) was a dead branch. *Do not go looking for more Elo in §1.*

*Original analysis, kept for the record:*

**1.2b Aspiration window magnitudes were never rescaled [C]**

```cpp
// searcher.cpp:1412
int32_t windowDelta = std::clamp<int32_t>(15 + (scoreSwing / 4), 25, 100);
// searcher.cpp:1441
windowDelta = std::min<int32_t>(WINDOW_HARD_CAP, windowDelta * 2 + 10);   // cap 1500
```

`15`, `25`, `100`, `/4`, `*2+10` and `WINDOW_HARD_CAP` are all eval-scale
magnitudes, all hardcoded, and **none appears in any `tuning/groups/*.json`**
(verified: those files contain only `LMP_SCALE_*`, `LMR_C_PERCENT`, `SE_*`,
`SEE_CAPTURE_MARGIN`, `HISTORY_PRUNE_D*`, `RFP_MARGIN_PER_DEPTH`, `NMP_EVAL_*`,
`FUTILITY_MID_STEP`, `PROBCUT_MARGIN`). They were chosen when a pawn was 100. A
pawn is now ~260, so **the aspiration window is effectively 2.6× tighter than
designed** — more fail-highs, more fail-lows, more full re-searches per iteration,
every iteration. Note `15` is already dead code: `clamp(15 + x/4, 25, 100)` can
never return 15.

**1.3 The NMP eval gate `+100` [C]** — `searcher.cpp:888`, comment says "within
~100cp of beta". Now ~38 real cp. Hardcoded; `NMP_EVAL_DIV`/`NMP_EVAL_MAX` are
tuned but this gate is not.

**1.4 `REPETITION_CONTEMPT = 80`** — commented `~0.8 pawn`, delivers ~0.31.
**Caveat [C]:** contempt is the one parameter self-play SPRT *cannot* measure —
both sides are the same engine, so draw-avoidance against an equal opponent is
structurally penalised. Validate on the SF bracket or not at all.

**1.5 Phase-dependent factor [H]** — the 1.99→3.20 spread says no scalar is right
everywhere. But two cautions: the endgame figure is inflated by the *same
saturation* that made queens read 1.42 (in sparse positions every removal is
nearer decisive), and the middle buckets (2.89 / 2.77, n=50/69) are one value
within noise. The data supports **at most a two-level split**, not a curve.
Settle with a saturation control — restrict to removals where the post-removal
eval stays inside a moderate band — before implementing anything.

---

## 2. Hot-loop structure — branches and work that can be removed

**2.1 Loop-invariant test inside the hottest loop [C]**

```cpp
// searcher.cpp:518, inside `while (movePicker.hasNext())`
if (chess::isValidSquare(ctx.excludedMove.from) && m.sameFromTo(ctx.excludedMove))
```

`ctx.excludedMove` never changes inside the loop. This evaluates a load and a
compare on **every move of every node**. Hoist to a `const bool hasExcluded` above
the loop and the inner test collapses to a predictable, almost-always-false
branch. Pure win, node-identical — exactly the "fewer branches" class you asked
about. Verify with bench6 node-identity, then measure NPS interleaved.

**2.2 The qsearch delta prune returns a fail-hard bound in a fail-soft searcher [C]**

```cpp
// searcher.cpp:1073
if (shouldDeltaPrune(standPat, materialToEval(QSEARCH_DELTA_MARGIN), alpha)) {
    return alpha;   // <-- fail-HARD
}
```

Everything else in this searcher is fail-soft, deliberately (`return standPat`
above it is commented "fail-soft: tighter bound than a flat beta"). Here we know
`standPat + margin < alpha`, so `standPat + margin` is a *strictly tighter and
still valid* upper bound than `alpha`. Returning `alpha` hands the parent a
less informative value. No TT pollution risk — this path returns before the store.

**2.3 ProbCut builds a full `MoveList` per qualifying node [C/M]**

`searcher.cpp:912` — `MoveList captures = generateTacticalMoves(b);` constructs a
660-byte object **[M]** at every interior non-PV node with `depth >= 3`, then runs
SEE over it. Now that `73b877d` made the filter more permissive, more of those
SEEs also lead to a child search. Worth instrumenting the firing rate before
touching, per the rule that saved two SPRTs on thread voting.

**2.5 `countRepetitions` — found by instrumenting §1.1, DONE (`3a68534`) [M]**

Not in CLAUDE.md's Tier-1/Tier-2 lists, and it should be. Measured on the
late-game workload: **136,860,059 calls, mean scan length 41.86, 5.73 billion
uint64 comparisons.** It ran `std::count` over *every* history slot — but only a
position with the same side to move can repeat, and the Zobrist key carries a
side-to-move term, so half the slots could never match. Stepping by two is
node-identical by construction.

⚠️ The parity is **not** simply "every other slot". `doNullMove` flips the side
without pushing an entry, so inside a null-move subtree the current position
sits on the opposite parity. `Board::nullPly` tracks it. The first attempt put
the increment in `doMove` instead of `doNullMove` and bench6 diverged
immediately (1,922,755 vs 1,953,820) — **the node-identity check earned its
keep.** Do not touch this code without it.

Result: bench6 exactly neutral (0.945s both — opening positions have short
clocks), **+1.58% on the late-game workload** (median 46.120s → 45.390s, new
binary won all 5 interleaved pairs). Kept for being node-identical and never
slower; the honest Elo expectation is well under 1, below SPRT resolution.

**2.4 `nonPawnMajors` is computed unconditionally [C]** — `searcher.cpp:849`, four
bitboard ORs plus a popcount at every node, consumed only by `canNullMove`
(line 890). **[X]** Moving it behind the short-circuit was measured NPS-neutral on
2026-07-28 — but that was several nets and a different tree ago. Low priority,
re-measurable.

---

## 3. Memory and cache — the largest unexamined surface

Measured struct sizes on this build **[M]**:

```
SearchRuntime      1,028,056 B   (0.98 MB PER THREAD)
  contHist           802,816 B   <-- 78% of it
  corrHist           196,608 B
  history             16,384 B
  captureHistory       3,584 B
MovePicker             1,768 B
Board                  6,400 B
Board::MoveState          32 B
chess::Move                3 B
```

**3.1 `contHist` is 785 KiB and every probe is a cache miss [C/H]**

`int16_t contHist[2][7][64][7][64]` = 401,408 cells. That is larger than a typical
per-core L2, so essentially every continuation-history read in move ordering
(`sorter.cpp:91`) and every update (`searcher.cpp:467`, `678`) is an L3-or-DRAM
access, in the hottest ordering path. The design comment justifies it with "a
24-position fixed-depth bench dropped ~6% nodes" — a *node* argument that never
priced the *cache* cost. Both directions are worth measuring:
narrowing the trailing index, or splitting into a smaller hot table.
⚠️ `perf` overstates memory-bound code ~3× here — trust interleaved wall-clock.

**3.2 `softResetHistory` streams ~820 KiB per search per thread [C/M]**

```cpp
// searchruntime.hpp:100-105 — runs at the start of EVERY search, on every thread
for (int i = 0; i < CONT_HIST_CELLS; ++i) contHistFlat[i] >>= 1;   // 401,408 cells
```

At 4+0.04 a search is ~0.1 s, so this is ~1 MB read + 1 MB written before any node
is searched, multiplied by thread count at the same instant. Cheap to test: gate
the halving to run every N searches, or skip `contHist` specifically.

**3.3 `MovePicker` is 1,768 B and qsearch default-constructs then move-assigns [C]**

`searcher.cpp:1043` declares `MovePicker movePicker;` and `1077` move-assigns the
generator's result, so the live prefix is memcpy'd instead of being constructed in
place. Restructuring to direct-initialise both branches removes a copy from
50–80% of all nodes.

---

## 4. NNUE

**4.1 A benchmark comment that says the optimisation is a pessimisation [C]**

```
// accumulator.hpp:143-144
// Il guadagno sta negli accessi in memoria, non nelle somme: [...]
// Misurato in isolamento, 145 ns contro 98 per le due separate.
```

Read literally, this says the fused `updateMove` costs **145 ns against 98 ns for
the two separate calls it replaces** — i.e. the fast path is *slower*. Either the
numbers are transposed or the optimisation is inverted. This is in the single
hottest function in the engine (two 512-wide int16 rows per piece event). **Settle
it with a direct measurement.** If the comment is right as written, deleting
`updateMove` is both a speedup and a branch removal.

**4.2 Italian comments violate the project rule [C]** — `accumulator.hpp:136-149`
and `152-154`. `CLAUDE.md` and the standing instruction require English in code.
Convert when this file is next touched.

**4.3 Architecture is the lever with the best recent evidence [X/H]** — 512→1024
was +3.37; one layer → two (`deep16`) was +7.89. The L2 is only **16 wide**.
L2=32 or a third layer is the natural next step. ⚠️ The *first* deep16 attempt
lost outright (H0, LLR −2.29); only the reordered/sparse variants won — depth
alone is not sufficient, weight layout matters.

---

## 5. Verified sound — no action

Genuinely good code; listed so nobody re-audits it.

- **TT** (`tt.hpp`) — lockless XOR hashing is correct; payload-then-key store order
  is safe in both interleavings; bucket replacement scores age over depth
  sensibly; prefetch is already issued at both `doMove` sites. **[X]** Smaller TT
  was tested twice and is dead (16 MiB = −1.44 ±3.80 over 15,662 games).
- **`doMove`/`undoMove`** (`boardapi.cpp` → `boardapi.inl`) — one runtime switch
  dispatching to `if constexpr` templated specialisations per `MoveKind`. This is
  the right shape already; there is no branch left to remove.
- **SEE** (`sorter.cpp:124`) — the return *order* encodes the value ladder
  P<N<B<R<Q<K. ⚠️ Do not "simplify" the queen test by merging it with
  bishops/rooks: it would return QUEEN where a rook also attacks, and that wrong
  value feeds the swap list. **[X]** The PEXT-gating micro-opt was measured
  neutral and rejected.
- **Lazy SMP** (`searcher.cpp:225-328`) — depth diversification (+6.40) and thread
  voting (+4.29) both won and are correctly implemented.
- **`evalStack`** is `static thread_local` — correct, and load-bearing.

---

## 6. Open debts

**6.1 ProbCut (`73b877d`) is in the shipped build, unvalidated.** Its SPRT stopped
at +5.39 ±12.21, LOS 80.7%, LLR 0.32 — indistinguishable from zero. The +82.3
calibration measured it *together with* the qsearch fix and cannot separate them.
Finish that SPRT. If negative, `git revert 73b877d` — and note that would also
invalidate the reasoning behind re-tuning `PROBCUT_MARGIN`.

**6.2 The latent-stalls bug is still live [M].** The 260 SPRT logged
`base: 1 timeout, 1 crash / new: 2 timeouts, 2 crashes` — **6 in 3472 games ≈
0.17%**, inside the long-recorded 0.1–0.2% band, roughly symmetric so it biases
nothing. But unlike an Elo tweak it costs *whole games* in real play. This is a
debugging problem, not a measurement one, and it is the best-value non-Elo target
left.

---

## 7. Ranked plan

Ordered by (expected Elo × confidence) ÷ cost.

| # | action | evidence | cost |
|---|---|---|---|
| ~~1~~ | ~~`REPETITION_DRAW_ADVANTAGE_THRESHOLD`~~ | **DONE `5291b28` — dead branch, 0 Elo. Ranking wrong.** | — |
| ~~2~~ | ~~Rescale aspiration windows~~ | **DEAD — −7.72, and 100% is a local minimum both ways. Premise wrong.** | — |
| 3 | Finish the ProbCut SPRT | **[M]** unvalidated code is shipped | machine time only |
| 4 | Settle the `updateMove` benchmark contradiction | **[C]** hottest function | ~1 h |
| 5 | Hoist the `excludedMove` invariant; fail-soft delta return | **[C]** free, node-identical | ~1 h |
| 6 | Re-tune `search_evalmargins` on the fixed tree | **[C]** those margins were compensating for the bug | SMAC3 run |
| 7 | Saturation control → decide the phase-split question | **[M]** 1.99 vs 3.20 unexplained | ~1 h |
| 8 | `contHist` cache-cost experiment | **[M]** 785 KiB, never priced | ~half day |
| 9 | Hunt the stalls bug | **[M]** 0.17%, whole games | open-ended |
| 10 | NNUE L2 width 16 → 32 | **[X]** best-evidenced architecture lever | training run |

**Scoreboard on this document's own predictions (2026-09-03).** My top two
ranked items were both wrong: #1 was a dead branch (0 Elo), #2 measured −7.72
and its premise turned out to be backwards. What *did* pay was
`countRepetitions` (§2.5) — which was not on the list at all and surfaced only
from instrumenting #1. **Treat the ranking below as hypotheses to be measured,
not a queue to be worked.** Cost so far of following the discipline instead of
the list: ~2 hours, versus the ~8 an unmeasured march down the ranking would
have spent to reach the same place.

**Method note.** Today's result did not come from a list like this one. It came
from analysing 2400 games that had *already been played*. Of the candidates I
reasoned out from first principles, the data-gap hypothesis was falsified and the
corrhist-cap idea was a provable no-op (`CORR_TOTAL_CAP` never binds: 0 hits in
~1.7M calls **[M]**) — while the finding that mattered came from interrogating a
PGN. **Mine the games before trusting the list.**

---

## 8. Graveyard — do not retry

From SPRT logs and the project record. Re-testing these costs hours and returns
nothing.

| change | result |
|---|---|
| RFP depth gate widened | **37.03%, LOS 0.00%** — lost hard |
| Node-effort time management | H0 accepted, −4.0 ±5.2 @8682 |
| IIR `depth>=6`→`>=4` | +1.37 ±3.37 @20,000 — null (was LOS 92% at 17k!) |
| LMR `moveIndex>=4`→`>=3` | −1.80 ±6.89 |
| LMR `depth>=2` + guard | −6.84 ±11.89 |
| Check-extension cap removed | −10.28 ±14.33 |
| History divisor 8192→4096 | +1.48 ±4.85 — flat, +19.7% nodes |
| TT 16 MiB / 8 MiB | −1.44 ±3.80 / −1.97 ±9.62 |
| PGO | −4% / −9%, twice |
| Razoring (re-added) | +1.1 ±6.4 — still redundant |
| Multi-ply contHist in ordering | −7 Elo |
| NNUE training budget past 320 SB | +29.4 → +9.9 → +2.7, exhausted |
| More same-distribution data | −7.30 ±8.61 |
| SEE PEXT-gating micro-opt | node-identical, NPS-neutral |

**Two measurement laws earned the hard way.** (1) The fixed-depth node profile
does **not** predict the sign — the LMR gate won +13.36 *with* tactical +26%.
(2) LLR is a random walk with drift: today it went 2.09 → 1.91 → 2.30 → 2.98 and
crossed. Neither a dip nor a plateau is diagnostic; compare LLR at equal N before
citing a precedent.
