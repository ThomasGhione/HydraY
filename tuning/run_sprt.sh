#!/usr/bin/env bash
#
# run_sprt.sh - Sequential Probability Ratio Test for HydraY search/eval changes.
#
# Plays the current build (./chess) against a frozen baseline (./chess_baseline)
# under a time control, and lets cutechess-cli's native -sprt decide merge/no-merge:
# it stops automatically as soon as an H0/H1 bound is crossed.
#
# WHY TIME CONTROL (not fixed depth): a search/pruning change that cuts nodes shows
# no benefit at fixed depth - the win is reaching greater depth in the same time.
# Always SPRT search changes under tc=, never under fixed depth.
#
# ---------------------------------------------------------------------------
# WORKFLOW
#   1. Freeze a baseline BEFORE you start changing code:
#        make prod && cp ./chess ./tuning/chess_baseline
#      (or:  ./tuning/run_sprt.sh --snapshot   to freeze the current ./chess)
#   2. Make your change, then:  make prod
#   3. Run:  ./tuning/run_sprt.sh
#
# ---------------------------------------------------------------------------
# TUNABLE KNOBS (env vars; sensible defaults below)
#   TC          time control "moves/sec+inc" or "sec+inc"   (default 4+0.04)
#   NODES       fixed nodes per move INSTEAD of TC (fastchess only). Sends a bare
#               `go nodes N`: no clock, so the engine's time manager is off and
#               the result is independent of NPS. For isolating eval quality
#               from eval cost; it is a diagnostic, not a merge gate.
#   ELO0 ELO1   SPRT hypotheses, ELO                          (default 0 and 5)
#   ALPHA BETA  SPRT error rates                              (default 0.05 0.05)
#   CONCURRENCY parallel games                                (default: nproc/2)
#   HASH        TT size MiB per engine                        (default 64)
#   THREADS     search threads per engine                     (default 1)
#   BOOK        opening book PGN                              (default books/openings.pgn)
#   MAXGAMES    hard cap on games (safety net)                (default 4000)
#   NEW_OPTS    extra UCI options for the NEW engine, space-separated
#               "Name=Value" pairs (e.g. "EvalFile=/abs/net.bin")
#   BASE_OPTS   same, for the BASELINE engine
#
# Examples:
#   ./tuning/run_sprt.sh                          # default [0,5] gain test
#   ELO0=-3 ELO1=3 ./tuning/run_sprt.sh           # non-regression test (for cleanups)
#   TC=10+0.1 CONCURRENCY=4 ./tuning/run_sprt.sh
#   NEW_OPTS="EvalFile=/abs/new.nnue" ./tuning/run_sprt.sh  # test a candidate net
#   NODES=150000 ./tuning/run_sprt.sh             # equal work per move, speed removed
# ---------------------------------------------------------------------------

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${script_dir}"

new_bin="${repo_root}/chess"
base_bin="${script_dir}/chess_baseline"

# --- --snapshot: freeze the current ./chess as the baseline and exit ----------
if [[ "${1:-}" == "--snapshot" ]]; then
    if [[ ! -x "${new_bin}" ]]; then
        echo "error: ${new_bin} not found - run 'make prod' first." >&2
        exit 1
    fi
    cp -- "${new_bin}" "${base_bin}"
    echo "Baseline frozen: ${base_bin}"
    echo "  (from $(cd "${repo_root}" && git rev-parse --short HEAD 2>/dev/null || echo '?') / current ./chess)"
    exit 0
fi

# --- preflight ----------------------------------------------------------------
if [[ ! -x "${new_bin}" ]]; then
    echo "error: ${new_bin} not found - run 'make prod' first." >&2
    exit 1
fi
if [[ ! -x "${base_bin}" ]]; then
    echo "error: baseline ${base_bin} not found." >&2
    echo "  Freeze one first:  ./tuning/run_sprt.sh --snapshot" >&2
    echo "  (do this on the OLD code, before applying your change)." >&2
    exit 1
fi

# Resolve the real cutechess-cli, skipping the tuning/ forwarding wrapper.
real_cutechess=""
IFS=':' read -ra path_dirs <<< "${PATH}:/usr/local/bin:/usr/bin"
for d in "${path_dirs[@]}"; do
    c="${d}/cutechess-cli"
    if [[ -x "${c}" && "$(readlink -f -- "${c}")" != "$(readlink -f -- "${script_dir}/cutechess-cli")" ]]; then
        real_cutechess="${c}"; break
    fi
done
[[ -n "${real_cutechess}" ]] || { echo "error: real cutechess-cli not found in PATH." >&2; exit 127; }

# --- knobs --------------------------------------------------------------------
TC="${TC:-4+0.04}"
ELO0="${ELO0:-0}"
ELO1="${ELO1:-5}"
ALPHA="${ALPHA:-0.05}"
BETA="${BETA:-0.05}"
HASH="${HASH:-64}"
THREADS="${THREADS:-1}"
# The engine honours the UCI "Threads" option (Lazy SMP); its auto default is
# omp_get_max_threads(). Pin the env too, as a belt-and-braces cap in case a
# tool ever launches the engine without sending setoption Threads.
export OMP_NUM_THREADS="${THREADS}"
BOOK="${BOOK:-books/openings.pgn}"
MAXGAMES="${MAXGAMES:-4000}"
NODES="${NODES:-}"
if [[ -n "${NODES}" ]]; then
    [[ "${NODES}" =~ ^[0-9]+$ && "${NODES}" -gt 0 ]] || { echo "error: NODES must be a positive integer." >&2; exit 1; }
    limit_arg="nodes=${NODES}"
    limit_desc="NODES=${NODES} (fixed per move, no clock)"
else
    limit_arg="tc=${TC}"
    limit_desc="TC=${TC}"
fi
# Per-engine UCI options ("Name=Value" pairs) -> backend option.Name=Value args.
NEW_OPTS="${NEW_OPTS:-}"
BASE_OPTS="${BASE_OPTS:-}"
new_opt_args=()
for kv in ${NEW_OPTS}; do new_opt_args+=("option.${kv}"); done
base_opt_args=()
for kv in ${BASE_OPTS}; do base_opt_args+=("option.${kv}"); done
# Time-controlled games need 1 searcher per *physical* core (HT siblings sharing
# a core distort the clock), leaving one core for the OS/cutechess. Fall back to
# nproc/2 if lscpu is unavailable.
phys_cores="$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)"
[[ "${phys_cores}" =~ ^[0-9]+$ && "${phys_cores}" -ge 1 ]] || phys_cores="$(( $(nproc 2>/dev/null || echo 2) / 2 ))"
default_conc="$(( phys_cores - 1 ))"
[[ "${default_conc}" -ge 1 ]] || default_conc=1
CONCURRENCY="${CONCURRENCY:-${default_conc}}"

[[ -f "${BOOK}" ]] || { echo "error: opening book ${BOOK} not found." >&2; exit 1; }

stamp="$(date +%Y%m%d_%H%M%S)"
pgn_out="sprt_${stamp}.pgn"

echo "=============================================================="
echo " HydraY SPRT"
echo "   new      : ${new_bin}"
echo "   baseline : ${base_bin}"
echo "   ${limit_desc}  H0=${ELO0} H1=${ELO1}  alpha=${ALPHA} beta=${BETA}"
echo "   threads=${THREADS} hash=${HASH}MiB concurrency=${CONCURRENCY}"
echo "   book=${BOOK}  pgn=${pgn_out}  cap=${MAXGAMES} games"
echo "=============================================================="
echo " H1 accepted (LLR>=upper)  -> the change is a GAIN: keep it."
echo " H0 accepted (LLR<=lower)  -> no measurable gain: discard it."
echo "=============================================================="

# Backend: fastchess computes the pentanomial SPRT (~1.5-2x fewer games for the
# same decision) and is preferred when present; cutechess-cli is the fallback.
# Force either with SPRT_BACKEND=fastchess|cutechess.
fastchess_bin=""
command -v fastchess >/dev/null 2>&1 && fastchess_bin="$(command -v fastchess)"
SPRT_BACKEND="${SPRT_BACKEND:-$([[ -n "${fastchess_bin}" ]] && echo fastchess || echo cutechess)}"

if [[ "${SPRT_BACKEND}" == "fastchess" ]]; then
    [[ -n "${fastchess_bin}" ]] || { echo "error: SPRT_BACKEND=fastchess but fastchess not on PATH (~/.local/bin)." >&2; exit 127; }
    echo " backend: fastchess (pentanomial, model=normalized)"
    echo "=============================================================="
    exec "${fastchess_bin}" \
        -engine name=new  cmd="${new_bin}"  args="-uci" proto=uci option.Threads="${THREADS}" ${new_opt_args[@]+"${new_opt_args[@]}"} \
        -engine name=base cmd="${base_bin}" args="-uci" proto=uci option.Threads="${THREADS}" ${base_opt_args[@]+"${base_opt_args[@]}"} \
        -each "${limit_arg}" \
        -openings file="${BOOK}" format=pgn order=random plies=16 \
        -repeat -games 2 -rounds "$(( MAXGAMES / 2 ))" \
        -sprt elo0="${ELO0}" elo1="${ELO1}" alpha="${ALPHA}" beta="${BETA}" model=normalized \
        -resign movecount=3 score=500 \
        -draw movenumber=40 movecount=8 score=8 \
        -concurrency "${CONCURRENCY}" \
        -ratinginterval 10 \
        -recover \
        -pgnout file="${pgn_out}"
fi

# cutechess-cli would need a clock alongside the node limit, which puts the time
# manager back in play; refuse rather than silently run a different experiment.
[[ -z "${NODES}" ]] || { echo "error: NODES requires the fastchess backend." >&2; exit 1; }
echo " backend: cutechess-cli"
echo "=============================================================="
# -repeat: each opening played from both sides for fairness.
# adjudication speeds up decided games without biasing Elo.
exec "${real_cutechess}" \
    -engine name=new  cmd="${new_bin}"  arg=-uci ${new_opt_args[@]+"${new_opt_args[@]}"} \
    -engine name=base cmd="${base_bin}" arg=-uci ${base_opt_args[@]+"${base_opt_args[@]}"} \
    -each proto=uci tc="${TC}" \
        option.Threads="${THREADS}" option.Hash="${HASH}" \
    -openings file="${BOOK}" format=pgn order=random plies=16 \
    -repeat -games 2 -rounds "$(( MAXGAMES / 2 ))" \
    -sprt elo0="${ELO0}" elo1="${ELO1}" alpha="${ALPHA}" beta="${BETA}" \
    -resign movecount=3 score=500 \
    -draw movenumber=40 movecount=8 score=8 \
    -concurrency "${CONCURRENCY}" \
    -ratinginterval 10 \
    -pgnout "${pgn_out}"
