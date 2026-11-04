# HydraY Chess Engine

HydraY is a C++23 UCI chess engine with NNUE evaluation, for GUIs, bots, and
automated testing.

It is built around bitboards with PEXT-indexed sliding-piece attacks, an
iterative deepening alpha-beta/PVS search with Lazy SMP parallelism, a
cache-friendly transposition table, Syzygy tablebase probing, and a neural
network evaluator (HalfKA king-bucketed, dual-perspective, embedded in the
binary). The handcrafted evaluator was removed in 2.0.0 - evaluation strength
now improves by training better nets, not by editing C++. Development and
testing happen mainly on Linux/WSL; `make release` also cross-compiles the
Windows binaries.

## Binaries

Releases carry static binaries for Linux and Windows, with the net embedded.
Pick the first line that describes your CPU:

| Build | CPU | Speed |
|---|---|---|
| `x86-64-v3-vnni` | AVX-VNNI: Intel 12th gen and later, AMD Zen 5 | 1.30 |
| `x86-64-v3` | AVX2 + BMI2: Intel Haswell and later, AMD Zen 3 and Zen 4 | 1.00 |
| `x86-64-v3-nobmi2` | AVX2: AMD Zen 1 and Zen 2, whose PEXT is very slow | 0.88 |
| `x86-64-v2` | SSE4.2 + POPCNT, for older CPUs | 0.35 |

Speed is nodes per second relative to `x86-64-v3`, measured on one Intel core.
All four search the same tree: a fixed-depth search visits the same nodes on
every build.

## Requirements

- `clang++` (preferred when installed) or `g++` with C++23 support
- GNU `make`
- OpenMP (`-fopenmp`)
- x86-64 CPU. Without BMI2 the sliding-piece index falls back to a bit loop
  (12% slower overall), and without AVX2 the NNUE forward pass falls back to a
  scalar path (about 3x slower)
- Optional analysis tools: `valgrind`, `cppcheck`, `clang-tidy`, `scan-build`,
  `include-what-you-use`, `cppclean`, `lizard` (`pip install lizard`), `perf`
- Optional for Windows builds: `mingw-w64`, and `wine` to run them on Linux
- Optional for tuning/testing: `fastchess`, `cutechess-cli`, and `ordo`;
  `matplotlib` for the live SPSA plot

## Quick Start

```sh
make prod        # build the optimized engine -> ./chess
./chess          # UCI mode (./chess uci, -uci and --uci also work)
```

Example UCI session:

```txt
uci
isready
position startpos moves e2e4 e7e5
go depth 10
quit
```

## Run Modes

NNUE-specific modes (see the NNUE section below):

```sh
./chess datagen [--out PREFIX] [--threads N] [--nodes N] ...    # self-play training data
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
make release         # every release binary -> release/ (Linux and Windows)
make cls             # remove binaries, object files, and temp files
make help            # show build-system help
```

Production builds use `-O3 -march=native`, OpenMP, and LTO, and a `native`
binary may not start on another CPU: `make prod ARCH=-march=x86-64-v3` builds a
portable one. Object files go to `output/` so incremental builds stay fast.
`make release` hides the tuning options (`-DHYDRAY_RELEASE`); every other build
keeps them for the scripts in `tuning/`.

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

Functional tests alone do **not** catch search/eval strength regressions -
use SPRT for that.

### Perft (move-generation correctness)

Perft ("performance test") counts the leaf nodes of the legal-move tree at a
fixed depth. The counts for the standard positions are published, so a single
wrong number proves a defect in move generation, legality, or make/unmake -
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

Or in a window that redraws every minute (needs matplotlib): Elo per game with
its 95% band, the pentanomial, LOS, the LLR against the SPRT bounds and the
expected end. Without a pgn it takes the newest one in the current directory.

```sh
./tuning/match_watch.py tuning/sprt_20261030_152225.pgn
```

### Gauntlet and calibration (absolute Elo)

`tuning/runfastchess.sh` answers "how strong are we overall?" instead of
"better than the last baseline?". The current build plays fixed opponents with
[fastchess](https://github.com/Disservin/fastchess), then
[ordo](https://github.com/michiguel/Ordo) turns the PGN into ratings with the
opponents pinned at known Elo values. Both must be on your PATH (the scripts
also look in `~/.local/bin`). It does not rebuild `./chess`: run `make prod`
first. Relative paths given to it are relative to `tuning/`.

```sh
make prod && ./tuning/runfastchess.sh            # Stockfish calibration, default levels
./tuning/runfastchess.sh --tag 2.0.0 --tag 2.1.0 --tag 3.0.0 --tag 3.1.0 \
                         --anchor HydraY_2.0.0=3000   # old releases
./tuning/runfastchess.sh --sf-levels ../stockfish/stockfish-18:3000,3190 \
                         --opponent ../stockfish/stockfish-11 --anchor stockfish-11=-
./tuning/elo_watch.sh --loop                     # live dashboard, another terminal
./tuning/match_watch.py tuning/<run>.pgn --anchor stockfish-11=3350   # live plot
```

- **Stockfish calibration** (`--sf-levels PATH:ELO,...`): one opponent per
  level, Stockfish limited with `UCI_LimitStrength`/`UCI_Elo` and pinned at
  that level, so the result is on the Stockfish UCI_Elo scale (not CCRL). The
  binary needs `UCI_Elo` (Stockfish 11+). With no opponents at all it runs
  `../stockfish/stockfish-18:2800,3000,3190` when that binary exists, otherwise
  every UCI engine in `tuning/`.
- **Old releases** (`--tag TAG`): built once from `git archive` (the checkout
  is never touched) and cached in `tuning/.refs/`. Pre-2.0.0 tags are
  saturated (the current engine scores near 100% against them) and tag 1.1.0
  and older ignore the clock; every engine gets a clock check before the games.
- **Any engine** (`--opponent PATH`), with `--name` and per-engine UCI options
  (`--opt NAME=VALUE`, e.g. two nets on one binary with `EvalFile`).
- **Anchors**: `--anchor NAME=ELO` pins an opponent, `--anchor NAME=-` leaves
  it free (rated through the games). ordo pins all anchors together, so their
  differences enter the estimate. `--extra-pgn` adds games from other runs,
  e.g. the opponents playing each other.

Cores and hash are picked automatically: one thread per physical core and, on
a hybrid CPU, the largest group below the fastest one (the fastest stays free
for the system, fastchess and ordo); `--show-cpus` shows the choice. Resign is
two-sided by default: HydraY's eval scale is 2.3-3.2x Stockfish's, so the
one-sided rule cuts its losses short and lets its wins run long.

At the end, Ctrl+C included, `elo_watch.sh --save` writes the ratings, head to
head and anchors next to the PGN (`*_ratings.txt`, `*_h2h.txt`,
`*_anchors.txt`) and checks that every anchor came out exactly at its value.
The dashboard also shows, per opponent, the implied Elo (anchor plus our Elo
against that opponent alone): if it differs a lot between opponents, the
anchors are not consistent. `--help` on either script lists every option.

`match_watch.py` draws a gauntlet the same way: every engine on one rating
scale fixed by `--anchor NAME=ELO` (by default the first opponent at 0), with
its 95% band, a ladder and the standings, each engine's row against ours.

## NNUE Evaluation

The evaluator is a quantised neural network: a `(768x4kb_hm -> 1024)*2`
dual-perspective accumulator, pairwise-multiplied and run through a
`1024 -> 16 -> 1` hidden layer with SCReLU, 8 output buckets. Inputs are
king-bucketed and horizontally mirrored, so each perspective is indexed by its
own king square;
a king move crossing a bucket boundary triggers a lazy refresh of that
perspective, absorbed by a thread-local Finny table. Trained with
[bullet](https://github.com/jw1912/bullet) on self-play data. The first net
(768->256, 121M self-play positions) measured **+662 Elo** against the old
handcrafted evaluator (SPRT, LOS 100%); later nets are validated against the
previous release instead.

- The net ships **embedded in the binary** (`nnue/net/hydray.nnue` via
  `.incbin`) and is activated at startup. To ship a new net, replace that file
  and rebuild.
- The UCI option `EvalFile` loads an external `.nnue` file instead - useful
  for A/B-testing candidate nets without rebuilding.
- The accumulator is updated incrementally in the board's piece add/remove
  hooks; `./chess nnue-selftest <net>` verifies that the incremental
  accumulator matches a from-scratch rebuild over a game corpus.

### Generating training data

```sh
./chess datagen --out nnue/data/pc1 --threads 8 --nodes 8000
./chess datagen --help        # every option with its default
```

Options: `--out` (prefix, default `nnue/data/hydray`), `--threads` (1..32,
default 3), `--nodes` (per move, default 8000), `--net` (labeler, default the
embedded net), `--tb` (Syzygy, default `engine/syzygy/files`), `--shard`,
`--seal-on-exit`, `--start standard|chess324`, `--eg-every`, `--games` (games started: about
1 in 4 is discarded at the first search, see below),
`--target` (ETA line only), `--seed` (benchmark mode), `--first-shard` and
`--machine`/`--build` (recorded in the manifests). Both `--flag value` and
`--flag=value` work; a wrong value stops the run instead of being guessed.

Self-play games are written in bulletformat. All threads append whole games
to one shard, `<prefix>.sNNNN.bin.part`; every `--shard` positions (default
50M, 1.6 GB) it is sealed, renamed to `<prefix>.sNNNN.bin`, and the next one
starts. A sealed shard never changes again, so it can be uploaded or trained
on while the run goes on; `--shard 0` writes one unsealed `<prefix>.bin`, for
short experiments. Stop/resume with Ctrl+C or SIGTERM is safe: a restart
appends to the open `.part`, and `--seal-on-exit` seals it at the stop
instead. Every sealed shard gets a manifest, `<prefix>.sNNNN.json`: positions,
seal time, machine, build, labeler net (FNV-1a fingerprint), nodes, start,
endgame seeding, tablebases and seed. A restart with other settings seals the
open shard first, so a shard never mixes two configurations. A new shard
takes the number after every local shard and manifest, and after
`--first-shard`; `<prefix>.lock` stops a second datagen on the same prefix.
When Syzygy tablebases are present (see below), games are
adjudicated at <6 pieces with the exact WDL result; without tablebases the
adjudication disables itself (the startup banner says which mode is active).
A game whose random opening leaves one side more than 400 cp ahead at the
first search is discarded (about 25% of the non-seeded games, at the cost of
a single search each); the progress line counts them as `discarded`.
Inspect a file with `./chess datagen-dump <file.bin> [count]`.

#### In Docker

The image runs the datagen on any machine with Docker and an AVX2 + BMI2 CPU,
and can upload the sealed shards to a Hugging Face dataset as they come.

**Start (Docker Compose).** The settings are in `docker/compose.yaml`: the
`command:` list holds the `./chess datagen` flags (`--nodes`, `--threads`,
...), `user:` the owner of the files (`id -u`:`id -g`), the volumes the data
folder (`data/` at the repository root) and the optional tablebases.
`docker/.env`, ignored by git, holds only the token. From the repository root:

```sh
cp docker/.env.example docker/.env     # HF_TOKEN=..., only for the upload
docker compose -f docker/compose.yaml up -d --build
docker compose -f docker/compose.yaml logs -f      # progress, seals, uploads
docker compose -f docker/compose.yaml down         # stop: shard sealed and uploaded
```

The prefix needs no setting: it is `m` plus 8 hex digits of the hash of the
host's `/etc/machine-id`, stable across restarts and different on every
machine, and the engine version, as in `m956ee98b-v4.3.1.s0000.bin`: a new
version starts new names (`--out=/data/NAME` overrides it). Data that changes
without a version bump (a retuned search, a new net) is told apart by the
manifests. The host name and the build
(version, a hash of the sources, the ISA; the commit when the build gets
`GIT_COMMIT=$(git rev-parse --short HEAD)` in the environment) go in every
manifest.

The service restarts by itself after a crash or a reboot and resumes where it
stopped (`down` stops it for good). The threads default to every CPU of the
machine. `--games` does not fit it, since the run would restart when it ends:
for a fixed batch use `docker compose -f docker/compose.yaml run --rm datagen
--games=1000` (the arguments of `run` replace `command:`).

**Upload.** Uncomment `--hf-repo=USER/NAME` in `command:` (the dataset is
created private if missing) and, to free the disk, `--delete-after-upload`.
Every sealed shard goes with its manifest, in one commit, to the machine's
folder in the dataset (`m956ee98b/`, every version inside), and counts as uploaded only once the Hub reports its
SHA-256; only then is the local shard deleted. The manifest stays (a few
hundred bytes), and at the start the datagen continues the numbering after
the last shard in the dataset, so even an emptied `data/` never reuses a
name. Nothing on the Hub is overwritten: a name already there with other
content is reported as `CONFLICT` and left alone on both sides.

`OWNER` in the folder names the machine that writes it: a second machine on
the same folder is refused at the start, before it generates anything. Run
one datagen per machine: a second one, from another `data/`, needs its own
`--out` and `--hf-subdir`. After each upload `index.jsonl` lists the folder's
manifests and
`README.md`, the dataset card, sums up every folder: machine, shards,
positions, nodes, start, net, build and dates. The token, the dataset and
the owner are checked before the datagen starts. At the stop a last pass
uploads the shard sealed at exit (`stop_grace_period` gives it 10 minutes);
anything left goes up at the next start.

**Plain `docker run`.** The container takes the flags of `./chess datagen`
after its defaults, the machine-id prefix (mount `/etc/machine-id` at
`/host/machine-id`, or give `--out`) and `--threads` = the CPUs it may use
(`--cpus` included), plus the upload flags `--hf-repo USER/NAME`,
`--hf-subdir DIR` (default: the machine hash, else the `--out` name), `--hf-every SECONDS` (600) and
`--delete-after-upload`:

```sh
docker build -f docker/Dockerfile -t hydray-datagen .
docker run -d --name datagen --init --user "$(id -u):$(id -g)" -e HF_TOKEN \
    -v "$PWD/data:/data" \
    -v "$PWD/engine/syzygy/files:/hydray/engine/syzygy/files:ro" \
    -v /etc/machine-id:/host/machine-id:ro -v /etc/hostname:/host/hostname:ro \
    hydray-datagen --nodes 8000 --seal-on-exit \
    --hf-repo user/hydray-data --delete-after-upload
docker stop -t 600 datagen
docker run --rm -v "$PWD/data:/data" hydray-datagen dump /data/<prefix>.s0000.bin 5
```

The uploader also runs without Docker:
`nnue/tools/hf_upload.py --data data --repo user/hydray-data --once` (needs
`pip install huggingface_hub`).

### Training

The trainer lives in `nnue/trainer/` (Rust, bullet-based), with Colab notebooks
for GPU runs; `colab_recipe.ipynb` holds the recipe of the current net.
`nnue/trainer/src/bin/sanity_deep.rs` is the reference reader for the quantised
net layout - run it on any new net before swapping it in.

### Where the net comes from

The net is HydraY's own work, end to end:

- **Data**: self-play games of HydraY only, generated with `./chess datagen` at
  8,000 nodes per move from random 8-9 ply opening walks. The positions are
  labelled with HydraY's own search score and the game result; Syzygy
  tablebases adjudicate endings. No Stockfish, Leela or other engine's games,
  evaluations or nets were used, as data or as labels.
- **Lineage**: the first net (2.0.0) was trained on games of HydraY's old
  handcrafted evaluator. Every later net was trained on games played by an
  earlier HydraY net.
- **Training**: [bullet](https://github.com/jw1912/bullet) with our recipe, from
  random weights followed by a fine-tune; the 4.3.0 net saw 1.9 billion
  positions.

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

## Transposition Table

The transposition table stores 4 entries per 64-byte bucket (one cache line).
Size it with the UCI `Hash` option (1–4096 MB, default 64). On Linux the table
is aligned to 2 MiB and advised for transparent huge pages, which the kernel
grants when `/sys/kernel/mm/transparent_hugepage/enabled` is `always` or
`madvise`.

## UCI Options

```txt
Threads              spin,   default 1 (Lazy SMP)
Hash                 spin,   default 64 (MB, 1–4096)
Move Overhead        spin,   default 30 (ms kept in reserve for GUI and network lag)
Ponder               check,  default false (the GUI may send `go ponder`)
EvalFile             string, external .nnue net (default: embedded net)
SyzygyPath           string, tablebase directory
SyzygyProbeDepth     spin,   default 1
```

Outside release builds the tunable search constants from
`engine/search/search_constants.hpp` (RFP, NMP, ProbCut, SEE, futility, LMP, LMR
margins…) are also exposed as `spin` options for the tuner.

## Search-Constant Tuning

HydraY tunes search constants with SPSA, as Fishtest and OpenBench do:
`tuning/spsa.py` plays game pairs between theta+delta and theta-delta with
`fastchess` and moves theta along the perturbation by the score difference.
The constants are set through the engine's tuning spins, so build first with
`make prod` (release builds do not offer them).

```txt
tuning/spsa.py             SPSA tuner
tuning/spsa_watch.py       live plot of a running SPSA
tuning/groups/*.json       parameter groups (search_corrhist, search_evalmargins, ...)
tuning/run_sprt.sh         SPRT of the tuned values against a frozen baseline
```

A group lists its parameters under `parameter_ranges`, in either form:

```json
"CORR_WEIGHT_PAWN": "Integer(4000, 20000)",
"CORR_WEIGHT_PAWN": {"start": 10347, "min": 4000, "max": 20000, "c_end": 800, "r_end": 0.002}
```

`Integer(min, max)` starts from the engine's default, with `c_end` a
twentieth of the range and `r_end` 0.002. Run a group:

```sh
cd tuning
./spsa.py search_corrhist                       # 1500 iterations x 16 games, 8+0.08
./spsa.py search_corrhist --iterations 3000 --pairs 8 --tc 8+0.08 --concurrency 11
```

Games run at 1 thread and 64 MB, from `books/openings.pgn`, with as many
parallel games as physical cores minus one. Theta and the iteration are
saved to `spsa_<group>_state.json` after every iteration: stop with `Ctrl+C`
and rerun the same command to resume. To start over, move the state file
away.

SPSA suits numeric constants. Switches and small depth limits, where a step
of 1 changes behaviour, are decided with an SPRT each.

### Reading Results

`spsa_<group>.log` has one line per iteration: the score of theta+ against
theta- and theta after the update. The score is not an Elo gain, since
theta+ and theta- move in a random direction each time; what matters is
where theta drifts. `./spsa_watch.py <group>` plots it in a window that
redraws every minute: one panel per parameter, the start with a band of
+-c_end/2 around it, the drift in units of `c_end`, and the expected end
from the last 50 iterations. To apply the result, freeze the current build as the
baseline, write the final theta, rounded, into
`engine/search/search_constants.hpp`, rebuild and run an SPRT:

```sh
./tuning/run_sprt.sh --snapshot                 # before changing the constants
make prod
TC=8+0.08 MAXGAMES=8000 ./tuning/run_sprt.sh
```

## How the Engine Works

The board uses compact square storage plus piece bitboards, kept in sync at
all times. Sliding attacks read tables built at compile time through a PEXT
index. Zobrist hashing tracks positions for the transposition table and repetition detection.

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
handcrafted evaluator's constants are the piece values (`PIECE_VALUES` in
`engine/search/search_constants.hpp`) used by SEE, capture ordering and the
qsearch delta margins.

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
docker/        datagen image, Compose service, Hugging Face upload
tt/            Zobrist hashing and transposition table
uci/           UCI protocol interface
tests/         functional, performance, and benchmark programs
tuning/        self-play tuning, SPRT, and gauntlet scripts
script/        benchmark and analysis helpers
```

## Credits

- The search techniques come from the
  [Chess Programming Wiki](https://www.chessprogramming.org) and the published
  literature on them.
- Stockfish is a calibration opponent in `tuning/` and is neither shipped nor
  linked. Its source was read for individual ideas: the correction-history
  weights and gates follow Stockfish 18, and a few UCI behaviours match it.
- `engine/syzygy/` is [Pyrrhic](https://github.com/AndyGrant/Pyrrhic), by Ronald
  de Man, Basil, Jon Dart and Andrew Grant, under its own MIT-style license.
- Nets are trained with [bullet](https://github.com/jw1912/bullet).
- `tests/ut.hpp` is [boost-ext/ut](https://github.com/boost-ext/ut).

## License

MIT, see `LICENSE`.
