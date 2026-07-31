# HydraY Chess Engine

HydraY is a C++23 chess engine with NNUE evaluation. It has a terminal
interface and full UCI support for GUIs, bots, and automated testing.

It is built around bitboards with magic sliding-piece attacks, an iterative
deepening alpha-beta/PVS search with Lazy SMP parallelism, a cache-friendly
transposition table, Syzygy tablebase probing, and a neural network evaluator
(HalfKA king-bucketed, dual-perspective, embedded in the binary). The handcrafted evaluator
was removed in 2.0.0 — evaluation strength now improves by training better
nets, not by editing C++. Development and testing happen mainly on Linux/WSL;
a MinGW target exists for Windows builds.

## Requirements

- `g++` with C++23 support
- GNU `make`
- OpenMP (`-fopenmp`)
- x86-64 CPU with **BMI2 (PEXT)** — a hard requirement: sliding-piece attacks
  use PEXT indexing and the build fails outright without it (`-march=native` or
  `-march=x86-64-v3` provide it). AVX2 is strongly recommended on top: the NNUE
  forward pass has a scalar fallback, but it is much slower
- Optional analysis tools: `valgrind`, `cppcheck`, `clang-tidy`, `scan-build`,
  `include-what-you-use`, `cppclean`, `lizard` (`pip install lizard`), `perf`
- Optional for Windows builds: `mingw-w64`
- Optional for tuning/testing: `chess-tuning-tools`, `cutechess-cli`,
  `fastchess`, and `ordo`

## Quick Start

```sh
make prod        # build the optimized engine -> ./chess
./chess          # interactive terminal menu
./chess uci      # UCI mode
```

Example UCI session:

```txt
uci
isready
position startpos moves e2e4 e7e5
go depth 10
quit
```

The engine also switches to UCI mode automatically when stdin is piped, which
is what GUIs and bot runners do.

## Run Modes

```sh
./chess              # interactive menu
./chess uci          # UCI mode (-uci and --uci also work)
./chess -pvp         # human vs human
./chess -pvb w       # human (White) vs engine
./chess -pvb b       # human (Black) vs engine
./chess -bvb         # engine vs engine
```

In terminal game mode, enter moves in coordinate notation without a space,
e.g. `e2e4` (promotions: `e7e8q`).

NNUE-specific modes (see the NNUE section below):

```sh
./chess datagen [prefix] [threads] [nodes] [netPath] [tbPath]   # self-play training data
./chess datagen-dump <file.bin> [count]                         # inspect a data file
./chess nnue-selftest <net.nnue>                                # verify accumulator correctness
```

Move-generation diagnostics (see Testing below):

```sh
./chess perft <depth> [fen]          # leaf count of the legal-move tree
./chess perft divide <depth> [fen]   # per-root-move breakdown, for bisecting
./chess perft suite [maxdepth]       # standard positions vs published counts
```

## Build Targets

```sh
make prod            # optimized production build -> ./chess
make debug           # debug build with symbols and profiling
make prod_windows    # cross-compile -> ./chess.exe (needs mingw-w64)
make cls             # remove binaries, object files, and temp files
make help            # show build-system help
```

Production builds use `-O3 -march=native`, OpenMP, and LTO. Object files go to
`output/` so incremental builds stay fast.

## Testing

```sh
make test            # functional tests -> ./tests/test
make perf            # performance tests -> ./tests/perf
make all-tests       # build and run functional + performance tests
make test-valgrind   # run functional tests under valgrind
```

`make perf` reports **times**, not node counts. To compare node counts before
and after a change to a hot path, run a fixed-depth search from the same
position on both builds and read the `nodes` field of the last `info` line:

```sh
printf 'position startpos\ngo depth 12\nquit\n' | ./chess uci
```

Functional tests alone do **not** catch search/eval strength regressions —
use SPRT for that.

### Perft (move-generation correctness)

Perft ("performance test") counts the leaf nodes of the legal-move tree at a
fixed depth. The counts for the standard positions are published, so a single
wrong number proves a defect in move generation, legality, or make/unmake —
with no evaluation or search involved.

```sh
./chess perft suite              # 7 standard positions, depths 1-4 (~11M nodes)
./chess perft suite 6            # every published depth (~626M nodes, seconds)
./chess perft 5 "<fen>"          # one position
./chess perft divide 3 "<fen>"   # per-root-move counts
```

`suite` exits non-zero on any mismatch and also runs inside `make test` at
depth 4. When a count is wrong, `divide` narrows it down: compare each root
move against a reference, descend into the branch that differs, and repeat
until the depth-1 mismatch names the offending move.

Unlike the search benches, perft is deterministic and load-independent, so it
stays trustworthy on a busy machine. Run it after any change to move
generation, `doMove`/`undoMove`, or the square/bitboard conventions.

### SPRT (is the change really stronger?)

`tuning/run_sprt.sh` plays the current build against a frozen baseline under a
time control and runs a sequential test (SPRT) that stops as soon as the change
is accepted (H1) or rejected (H0). Search changes must be tested under a time
control, never at fixed depth.

```sh
make prod && ./tuning/run_sprt.sh --snapshot   # freeze baseline BEFORE editing
# ...make your change...
make prod && ./tuning/run_sprt.sh              # new vs baseline
```

The backend is **fastchess** when it is on your PATH (pentanomial model, needs
fewer games for the same decision); it falls back to `cutechess-cli`. Force one
with `SPRT_BACKEND=fastchess|cutechess`.

Env knobs (defaults): `TC=4+0.04`, `ELO0=0 ELO1=5` (gain test; use
`ELO0=-3 ELO1=3` to prove a cleanup is not a regression), `CONCURRENCY`,
`HASH`, `THREADS`. To test a candidate net without rebuilding:
`NEW_OPTS="EvalFile=/abs/candidate.nnue" ./tuning/run_sprt.sh`.

Watch a running test live (the SPRT verdict comes from fastchess itself; ordo
just shows Elo and error bars from the PGN):

```sh
watch -n 5 'ordo -q -s 1000 -J -p "$(ls -t tuning/sprt_*.pgn | head -1)"'
```

### Gauntlet (absolute Elo on a fixed scale)

`tuning/run_gauntlet.sh` answers "how strong are we overall?" instead of "better
than the last baseline?". The current build plays frozen release tags, then
[ordo](https://github.com/michiguel/Ordo) turns the PGN into ratings with one
old release pinned at a constant Elo, so runs stay comparable over time.

```sh
make prod && REF_TAGS="2.0.0" ./tuning/run_gauntlet.sh
REF_TAGS="2.0.0" GAMES=1000 ./tuning/run_gauntlet.sh
```

Reference binaries are built on demand in a throwaway `git worktree` (your
checkout is never touched) and cached as `tuning/chess_ref_<tag>`. `ordo` must
be on your PATH (the script also checks `~/.local/bin`).

Env knobs (defaults): `REF_TAGS=1.2.0`, `ANCHOR_TAG` / `ANCHOR_ELO=2000` (the
fixed yardstick — an internal value, not a CCRL rating), `GAMES=400` per
opponent, `TC=4+0.04`, `CONCURRENCY`, `THREADS`.

> Since the NNUE switch, pre-2.0.0 tags are saturated (the current engine
> scores near 100% against them, which makes ratings meaningless) — anchor on
> `2.0.0` or newer. Also, tag 1.1.0 and older have broken time management and
> forfeit at any real TC.

## NNUE Evaluation

The evaluator is a quantised neural network: (768x4kb_hm -> 512)*2 dual-perspective
accumulator with SCReLU activation and 8 output buckets. Inputs are king-bucketed
and horizontally mirrored, so each perspective is indexed by its own king square;
a king move crossing a bucket boundary triggers a lazy refresh of that
perspective, absorbed by a thread-local Finny table. Trained with
[bullet](https://github.com/jw1912/bullet) on self-play data. The first net
(768->256, 121M self-play positions) measured **+662 Elo** against the old
handcrafted evaluator (SPRT, LOS 100%); later nets are validated against the
previous release instead.

- The net ships **embedded in the binary** (`nnue/net/hydray.nnue` via
  `.incbin`) and is activated at startup. To ship a new net, replace that file
  and rebuild.
- The UCI option `EvalFile` loads an external `.nnue` file instead — useful
  for A/B-testing candidate nets without rebuilding.
- The accumulator is updated incrementally in the board's piece add/remove
  hooks; `./chess nnue-selftest <net>` verifies that the incremental
  accumulator matches a from-scratch rebuild over a game corpus.

### Generating training data

```sh
./chess datagen [prefix] [threads] [nodes] [netPath] [tbPath]
# defaults: nnue/data/hydray, machine threads, 8000 nodes/move,
#           embedded net for labeling, engine/syzygy/files
```

Self-play games are written in bulletformat. When Syzygy tablebases are
present (default path `engine/syzygy/files`, see below), games are adjudicated
at <6 pieces with the exact WDL result; without tablebases the adjudication
disables itself (the startup banner says which mode is active). Stop/resume
with Ctrl+C is safe. Inspect a file with
`./chess datagen-dump <file.bin> [count]`.

### Training

The trainer lives in `nnue/trainer/` (Rust, bullet-based), with a Colab
notebook for GPU runs. `nnue/trainer/src/bin/sanity.rs` is the reference
reader for the quantised net layout — run it on any new net before swapping it
in. See `NNUE_PLAN.md` for the current roadmap (dataset targets, architecture
steps, validation gates).

## Syzygy Tablebases

The search probes Syzygy WDL tablebases at internal nodes and filters root
moves with them. Set the UCI options:

```txt
SyzygyPath        path to the tablebase directory
SyzygyProbeDepth  minimum depth to probe at internal nodes (default 1)
```

`engine/syzygy/download_syzygy.sh` downloads the 3-4-5 piece files (~1 GB,
not tracked in git) into `engine/syzygy/files`. The same files power datagen
adjudication.

## Transposition Table and Huge Pages

The transposition table stores 4 entries per 64-byte bucket (one cache line).
Size it with the UCI `Hash` option (1–4096 MB, default 64). On Linux the
allocation tries explicit huge pages first, then transparent huge pages, then
normal heap.

Control huge-page behavior with `CHESS_TT_HUGEPAGE`:

```sh
CHESS_TT_HUGEPAGE=auto ./chess   # default
CHESS_TT_HUGEPAGE=on   ./chess   # force/try huge pages
CHESS_TT_HUGEPAGE=off  ./chess   # disable huge pages
```

Enabled values: `on`, `1`, `true`, `force`. Disabled values: `off`, `0`,
`false`. A huge-page benchmark is available with `make tt-huge-bench`
(`./tests/tt_hugepage_bench --depth 10 --repeats 3`).

## UCI Options

```txt
Threads              spin,   default = hardware threads (Lazy SMP)
Hash                 spin,   default 64 (MB, 1–4096)
EvalFile             string, external .nnue net (default: embedded net)
SyzygyPath           string, tablebase directory
SyzygyProbeDepth     spin,   default 1
SearchApiMutexGuard  check,  default true
```

`SearchApiMutexGuard` can also be set at startup with
`CHESS_ENGINE_SEARCH_MUTEX_GUARD=false ./chess uci`.

The tunable search constants from `engine/search/search_constants.hpp` (RFP,
NMP, ProbCut, SEE, futility, LMP, LMR margins…) are also exposed as `spin`
options for the tuner.

## Search-Constant Tuning

HydraY tunes search constants through self-play with `chess-tuning-tools` and
`cutechess-cli`. The workflow lives in `tuning/`:

```txt
tuning/base_config.json    shared engine, depth, rounds, and book settings
tuning/groups/*.json       tracked parameter groups (search_pruning, search_shape)
tuning/run_tune_local.sh   local launcher
tuning/chess_uci.sh        starts the engine in UCI mode for the tuner
```

The `tune` CLI must be on your PATH (it lives in a dedicated Python
environment, not in `tuning/.venv`). Verify your tools:

```sh
tune --help
cutechess-cli --version
```

Run a tracked parameter group (build first with `make prod`):

```sh
cd tuning
./run_tune_local.sh search_pruning
./run_tune_local.sh search_shape
```

`run_tune_local.sh` merges the chosen group with `base_config.json`, writes the
active `tuning_config.json`, and tracks each dataset under
`tuning/.tuning_state/` so unchanged runs resume and changed configs start
fresh. Keep each run focused: tune at most about 8 related parameters at once.

### Reading Results

Watch progress with `tail -f tuning/log.txt`. Key lines:

```txt
Testing {...}          parameter point being tested right now
Current optimum: {...} best point the Bayesian model has found
Estimated Elo: X +- Y  model estimate for that optimum
```

Apply `Current optimum`, not a single lucky `Got Elo`. If its 90% confidence
interval includes zero, the result is still uncertain — confirm with more
games. To apply: stop with `Ctrl+C`, copy the optimum into
`engine/search/search_constants.hpp`, rebuild with `make prod`, then
re-validate with SPRT or a gauntlet.

## How the Engine Works

The board uses compact square storage plus piece bitboards, kept in sync at
all times. Sliding attacks use magic bitboard tables built at startup. Zobrist
hashing tracks positions for the transposition table and repetition detection.

Search is iterative deepening over alpha-beta/PVS with: quiescence search,
transposition-table probing/storing, aspiration windows, null-move pruning,
reverse futility pruning, ProbCut, singular extensions, futility pruning,
late move reductions and pruning, SEE and history pruning, Syzygy WDL probes,
killer/history/countermove/capture-history/continuation-history move ordering,
correction history, draw handling (stalemate, repetition, fifty-move,
insufficient material), and Lazy SMP parallelism (helper threads share the
transposition table, each with private heuristics).

Evaluation is NNUE-only: the dual-perspective accumulator is maintained
incrementally on every make/unmake, and the AVX2 forward pass produces the
score at leaves and for pruning decisions. All that remains of the old
handcrafted evaluator's constants (`engine/eval_constants.hpp`) are the piece
values used by SEE, capture ordering, and stalemate scoring.

## Static Analysis

```sh
make analyze       # cppcheck, clang-tidy, iwyu, scan-build, GCC analyzer, cppclean, lizard
make complexity    # lizard complexity only
```

Reports are written to files such as `analysis.log`, `scan-build-report/`, and
`complexity-report.csv` (depending on which tools are installed).

## Project Layout

```txt
board/         board representation, FEN, move execution, legality, bitboards
engine/        engine runtime, evaluation seam, perft, time management
engine/search/ iterative deepening, alpha-beta/PVS, pruning, search constants
engine/sort/   move generation, move ordering, SEE
engine/syzygy/ Syzygy tablebase probing (fathom-based)
nnue/          NNUE net, accumulator, datagen, selftest, trainer
tt/            Zobrist hashing and transposition table
uci/           UCI protocol interface
driver/        terminal UI and game modes
tests/         functional, performance, and benchmark programs
tuning/        self-play tuning, SPRT, and gauntlet scripts
script/        benchmark and analysis helpers
```

## License

See `LICENSE`.
