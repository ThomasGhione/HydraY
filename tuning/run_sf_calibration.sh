#!/usr/bin/env bash
#
# run_sf_calibration.sh — absolute-strength calibration for HydraY against
# real Stockfish, using ordo with MULTIPLE fixed anchors.
#
# WHY THIS EXISTS (vs run_gauntlet.sh): run_gauntlet.sh pins an old HydraY tag
# at an arbitrary internal Elo (ANCHOR_ELO=3000 for 2.0.0) — a self-consistent
# scale, but not tied to any external, independently-known rating. This script
# anchors on Stockfish's own UCI_Elo strength limiter instead, at several
# levels bracketing HydraY's estimated strength, so the fit is well-conditioned
# (no single lopsided score against one distant opponent) and internally
# checkable: if ordo's fitted gaps between SF levels come out close to the
# nominal UCI_Elo gaps we set, the calibration is trustworthy.
#
# CAVEAT: Stockfish's UCI_Elo is the engine's own self-reported strength
# target (calibrated by the SF team against a pool of opponents), not a CCRL/
# CEGT rating. Treat the result as "Elo on the Stockfish UCI_Elo scale",
# not as a CCRL number. It also assumes the limiter is reasonably TC-stable;
# sanity-check by comparing the fitted gap between two SF anchors against
# their nominal gap in the output.
#
# NOTE: this script does NOT rebuild ./chess — run `make prod` yourself first.
#
# ---------------------------------------------------------------------------
# TUNABLE KNOBS (env vars; sensible defaults below)
#   ENGINE_BIN  path to the HydraY binary to test      (default ./chess, i.e. dev)
#   ENGINE_NAME name this build shows as in the gauntlet/ordo output (default HydraY-dev)
#               use these two to calibrate an alternate build (e.g. a different
#               branch/net) without touching the repo's own ./chess.
#   SF_BIN      path to the Stockfish binary          (default stockfish/stockfish-18)
#   SF_ELOS     space-separated UCI_Elo anchor levels  (default "2600 2800 3000 3190")
#               must be within the binary's advertised UCI_Elo min/max.
#   GAMES       games played PER Stockfish level        (default 200)
#   TC          time control "moves/sec+inc" or "sec+inc" (default 4+0.04)
#   CONCURRENCY parallel games                           (default: phys_cores-1)
#   THREADS     search threads per engine                (default 1)
#   HASH        hash size in MB per engine                (default 16)
#   BOOK        opening book PGN                          (default books/openings.pgn)
#
# Examples:
#   ./tuning/run_sf_calibration.sh                                # default bracket, 200 games/level
#   SF_ELOS="2800 3000 3190" GAMES=400 ./tuning/run_sf_calibration.sh
#   TC=10+0.1 ./tuning/run_sf_calibration.sh
# ---------------------------------------------------------------------------

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${script_dir}"

export PATH="${HOME}/.local/bin:${PATH}"

# --- knobs --------------------------------------------------------------------
new_bin="${ENGINE_BIN:-${repo_root}/chess}"
engine_name="${ENGINE_NAME:-HydraY-dev}"
SF_BIN="${SF_BIN:-${repo_root}/stockfish/stockfish-18}"
SF_ELOS="${SF_ELOS:-2600 2800 3000 3190}"
read -ra sf_elo_arr <<< "${SF_ELOS}"
GAMES="${GAMES:-200}"
TC="${TC:-4+0.04}"
THREADS="${THREADS:-1}"
HASH="${HASH:-16}"
BOOK="${BOOK:-books/openings.pgn}"

export OMP_NUM_THREADS="${THREADS}"

# --- preflight ----------------------------------------------------------------
if [[ ! -x "${new_bin}" ]]; then
    echo "error: ${new_bin} not found — run 'make prod' first." >&2
    exit 1
fi
if [[ ! -x "${SF_BIN}" ]]; then
    echo "error: Stockfish binary not found or not executable: ${SF_BIN}" >&2
    exit 1
fi
if ! command -v ordo >/dev/null 2>&1; then
    echo "error: ordo not found on PATH (looked in ~/.local/bin too)." >&2
    exit 127
fi
[[ -f "${BOOK}" ]] || { echo "error: opening book ${BOOK} not found." >&2; exit 1; }

# Sanity-check every requested level against the binary's advertised range.
sf_uci_out="$(printf 'uci\nquit\n' | "${SF_BIN}")"
elo_line="$(grep -E '^option name UCI_Elo ' <<< "${sf_uci_out}")"
elo_min="$(grep -oE 'min [0-9]+' <<< "${elo_line}" | awk '{print $2}')"
elo_max="$(grep -oE 'max [0-9]+' <<< "${elo_line}" | awk '{print $2}')"
for elo in "${sf_elo_arr[@]}"; do
    if (( elo < elo_min || elo > elo_max )); then
        echo "error: SF_ELOS level ${elo} outside this binary's UCI_Elo range [${elo_min}, ${elo_max}]." >&2
        exit 1
    fi
done

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

# --- concurrency: one searcher's worth of cores per game, leave one for the OS
# Only one engine per game is actively searching at a time (no pondering), so
# each concurrent game costs ~THREADS cores, not 2x.
phys_cores="$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)"
[[ "${phys_cores}" =~ ^[0-9]+$ && "${phys_cores}" -ge 1 ]] || phys_cores="$(( $(nproc 2>/dev/null || echo 2) / 2 ))"
default_conc="$(( (phys_cores - 1) / THREADS ))"
[[ "${default_conc}" -ge 1 ]] || default_conc=1
CONCURRENCY="${CONCURRENCY:-${default_conc}}"

echo "=============================================================="
echo " HydraY vs STOCKFISH calibration (absolute Elo via ordo, multi-anchor)"
echo "   gauntlet engine : ${engine_name} = ${new_bin}"
echo "   stockfish       : ${SF_BIN}  (UCI_Elo range [${elo_min}, ${elo_max}])"
echo "   anchor levels   : ${SF_ELOS}"
echo "   TC=${TC}  games/level=${GAMES}  threads=${THREADS} (OMP) conc=${CONCURRENCY}"
echo "=============================================================="

# --- assemble cutechess engine args -------------------------------------------
# -tournament gauntlet: the FIRST engine plays every other engine.
engine_args=( -engine name="${engine_name}" cmd="${new_bin}" arg=-uci )
for elo in "${sf_elo_arr[@]}"; do
    # Unlike HydraY, Stockfish auto-detects UCI over stdin — no "-uci" arg
    # (passing one makes it choke trying to execute "-uci" as a UCI command).
    engine_args+=(
        -engine name="SF-${elo}" cmd="${SF_BIN}"
        option.UCI_LimitStrength=true option.UCI_Elo="${elo}"
        option.Threads="${THREADS}" option.Hash="${HASH}"
    )
done

stamp="$(date +%Y%m%d_%H%M%S)"
pgn_out="sfcal_${stamp}.pgn"
ratings_out="sfcal_${stamp}.txt"
anchors_file="sfcal_${stamp}_anchors.txt"

for elo in "${sf_elo_arr[@]}"; do
    printf '"SF-%s",%s\n' "${elo}" "${elo}" >> "${anchors_file}"
done

# -repeat: each opening played from both colors for fairness.
rounds="$(( GAMES / 2 ))"
[[ "${rounds}" -ge 1 ]] || rounds=1

echo
echo "Watch live absolute Elo in another terminal (updates as games finish;"
echo "handles anchors with zero finished games yet, e.g. right at startup;"
echo "-s is omitted for speed — the final pass below computes real error bars):"
echo "  watch -n 15 ${script_dir}/watch_calibration.sh"
echo

"${real_cutechess}" \
    "${engine_args[@]}" \
    -each proto=uci tc="${TC}" \
    -tournament gauntlet \
    -openings file="${BOOK}" format=pgn order=random plies=16 \
    -repeat -games 2 -rounds "${rounds}" \
    -draw movenumber=40 movecount=8 score=8 \
    -resign movecount=3 score=500 \
    -concurrency "${CONCURRENCY}" \
    -ratinginterval 20 \
    -recover \
    -pgnout "${pgn_out}"

echo
echo "=============================================================="
echo " Computing absolute ratings with ordo (anchor: ${SF_ELOS})"
echo "=============================================================="

# -W/-D: auto-fit white advantage and draw rate. -s: bootstrap error bars.
# ordo 1.2.6's -m (multi-anchor) only reliably PINS an anchor when there are
# 2+ anchors in the file; with exactly one anchor + one floating player it
# silently applies a partial/wrong shift instead of an exact pin (verified
# empirically: SF-3000 came out at 2681, not 3000, with -m and a single row).
# The classic single-anchor -a/-A pair does not have this bug, so use it
# whenever there is exactly one SF level.
if [[ "${#sf_elo_arr[@]}" -eq 1 ]]; then
    ordo -q -W -D \
        -a "${sf_elo_arr[0]}" -A "SF-${sf_elo_arr[0]}" \
        -s 1000 -n "${CONCURRENCY}" \
        -p "${pgn_out}" \
        -o "${ratings_out}" \
        -j "sfcal_${stamp}_h2h.txt"
else
    ordo -q -W -D \
        -m "${anchors_file}" \
        -s 1000 -n "${CONCURRENCY}" \
        -p "${pgn_out}" \
        -o "${ratings_out}" \
        -j "sfcal_${stamp}_h2h.txt"
fi

echo
cat "${ratings_out}"
echo
echo "PGN     : ${pgn_out}"
echo "Ratings : ${ratings_out}"
echo "H2H     : sfcal_${stamp}_h2h.txt"
echo "Anchors : ${anchors_file}"
echo
echo "Sanity check: every SF-<n> row above must show EXACTLY its nominal Elo"
echo "(${SF_ELOS}). If one doesn't, ordo's anchor mechanism silently failed"
echo "(see the -m single-anchor caveat in this script's ordo-invocation"
echo "comment) — don't trust HydraY-dev's number until that's fixed."
