#!/usr/bin/env bash
#
# run_gauntlet.sh — absolute-strength gauntlet for HydraY using ordo.
#
# Plays the current build (./chess) as the gauntlet engine against one or more
# FROZEN reference versions (old git tags), then feeds the PGN to `ordo` with a
# fixed anchor so the reported Elo tracks ABSOLUTE progress across dev cycles.
#
# WHY A GAUNTLET (vs run_sprt.sh): SPRT only answers "is the new build better
# than the immediately-frozen baseline?". It cannot tell you if you are globally
# stronger or weaker, and it has no fixed yardstick. The gauntlet pins an old
# release at a constant Elo (ANCHOR_ELO) so every run is comparable on one scale
# — the right tool for measuring the march toward 3000.
#
# NOTE: this script does NOT rebuild ./chess — run `make prod` yourself first.
# Reference binaries for old tags are built ON DEMAND in a throwaway git
# worktree (your working tree / current checkout is never touched) and cached as
# tuning/chess_ref_<tag>.
#
# ---------------------------------------------------------------------------
# TUNABLE KNOBS (env vars; sensible defaults below)
#   REF_TAGS    space-separated git tags to use as fixed anchors  (default "2.0.0")
#               NOTE: tag 1.1.0 (and older) has BROKEN time management — it
#               ignores wtime/btime/movetime and dumps ~3s on every move, so it
#               flags instantly at any real TC. Always time-sanity-check a new
#               anchor tag first:  printf 'position startpos\ngo wtime 4000 winc 40\n'
#               | ./tuning/chess_ref_<tag> -uci   should return in ~100ms, not seconds.
#   ANCHOR_TAG  which REF_TAG ordo pins to ANCHOR_ELO             (default: first of REF_TAGS)
#   ANCHOR_ELO  fixed internal Elo for the anchor (yardstick, NOT (default 3000)
#               a CCRL rating — only consistency across runs matters)
#   GAMES       games played PER reference opponent               (default 400)
#   TC          time control "moves/sec+inc" or "sec+inc"         (default 4+0.04)
#   CONCURRENCY parallel games                                    (default: phys_cores-1)
#   THREADS     search threads per engine (via OMP_NUM_THREADS)   (default 1)
#   BOOK        opening book PGN                                  (default books/openings.pgn)
#
# Examples:
#   ./tuning/run_gauntlet.sh                              # dev vs 1.2.0, 400 games
#   REF_TAGS="1.2.0" GAMES=1000 ./tuning/run_gauntlet.sh  # more games, one anchor
#   GAMES=1000 TC=10+0.1 ./tuning/run_gauntlet.sh
# ---------------------------------------------------------------------------

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${script_dir}"

# ordo is built into ~/.local/bin by setup; make sure it is reachable.
export PATH="${HOME}/.local/bin:${PATH}"

new_bin="${repo_root}/chess"

# --- knobs --------------------------------------------------------------------
# Scale: 2.0.0 = 3000 (since 2026-07-10; chained to the old 1.2.0=2000 scale
# via 2.0.0's SPRT +662 vs HCE ≈ 1.3.0 ≈ 2366). 1.2.0/1.3.0 are saturated —
# only anchor on them to rate pre-2.0.0 tags.
REF_TAGS="${REF_TAGS:-2.0.0}"
read -ra ref_tag_arr <<< "${REF_TAGS}"
ANCHOR_TAG="${ANCHOR_TAG:-${ref_tag_arr[0]}}"
ANCHOR_ELO="${ANCHOR_ELO:-3000}"
GAMES="${GAMES:-400}"
TC="${TC:-4+0.04}"
THREADS="${THREADS:-1}"
BOOK="${BOOK:-books/openings.pgn}"

# The engine honours the UCI "Threads" option (Lazy SMP), but its auto default
# is omp_get_max_threads(); pin the env so engines launched without a setoption
# (this script sends none) stay at THREADS and concurrency never oversubscribes.
export OMP_NUM_THREADS="${THREADS}"

# --- preflight ----------------------------------------------------------------
if [[ ! -x "${new_bin}" ]]; then
    echo "error: ${new_bin} not found — run 'make prod' first." >&2
    exit 1
fi
if ! command -v ordo >/dev/null 2>&1; then
    echo "error: ordo not found on PATH (looked in ~/.local/bin too)." >&2
    echo "  Build it:  git clone https://github.com/michiguel/Ordo && cd Ordo && make && cp ordo ~/.local/bin/" >&2
    exit 127
fi
[[ -f "${BOOK}" ]] || { echo "error: opening book ${BOOK} not found." >&2; exit 1; }

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

# --- concurrency: one searcher per physical core, leave one for the OS --------
phys_cores="$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)"
[[ "${phys_cores}" =~ ^[0-9]+$ && "${phys_cores}" -ge 1 ]] || phys_cores="$(( $(nproc 2>/dev/null || echo 2) / 2 ))"
default_conc="$(( phys_cores - 1 ))"
[[ "${default_conc}" -ge 1 ]] || default_conc=1
CONCURRENCY="${CONCURRENCY:-${default_conc}}"

# --- build (on demand) the frozen reference binary for each tag ---------------
# Built in a throwaway git worktree so the current checkout is never disturbed.
build_ref() {
    local tag="$1"
    local out="${script_dir}/chess_ref_${tag}"
    if [[ -x "${out}" ]]; then
        echo "  ref ${tag}: cached (${out})"
        return 0
    fi
    if ! git -C "${repo_root}" rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
        echo "error: git tag '${tag}' does not exist." >&2
        exit 1
    fi
    echo "  ref ${tag}: building in throwaway worktree..."
    local wt
    wt="$(mktemp -d)"
    git -C "${repo_root}" worktree add --detach --quiet "${wt}" "${tag}"
    # Build optimized; do not let a build failure leave a dangling worktree.
    if ! make -C "${wt}" prod >/dev/null 2>"${script_dir}/.gauntlet_build_${tag}.log"; then
        echo "error: 'make prod' failed for tag ${tag} (see .gauntlet_build_${tag}.log)" >&2
        git -C "${repo_root}" worktree remove --force "${wt}" || true
        exit 1
    fi
    cp -- "${wt}/chess" "${out}"
    git -C "${repo_root}" worktree remove --force "${wt}"
    echo "  ref ${tag}: built -> ${out}"
}

echo "=============================================================="
echo " HydraY GAUNTLET (absolute Elo via ordo)"
echo "   gauntlet engine : ${new_bin}  ($(cd "${repo_root}" && git rev-parse --short HEAD))"
echo "   references      : ${REF_TAGS}"
echo "   anchor          : ${ANCHOR_TAG} pinned at ${ANCHOR_ELO} Elo"
echo "   TC=${TC}  games/opp=${GAMES}  threads=${THREADS} (OMP) conc=${CONCURRENCY}"
echo "=============================================================="

for tag in "${ref_tag_arr[@]}"; do
    build_ref "${tag}"
done

# --- assemble cutechess engine args -------------------------------------------
# -tournament gauntlet: the FIRST engine plays every other engine.
engine_args=( -engine name="HydraY-dev" cmd="${new_bin}" arg=-uci )
for tag in "${ref_tag_arr[@]}"; do
    engine_args+=( -engine name="HydraY-${tag}" cmd="${script_dir}/chess_ref_${tag}" arg=-uci )
done

stamp="$(date +%Y%m%d_%H%M%S)"
pgn_out="gauntlet_${stamp}.pgn"
ratings_out="gauntlet_${stamp}.txt"

# -repeat: each opening played from both colors for fairness.
# rounds*games_per_round total per pairing; with -games 2 we want GAMES total.
rounds="$(( GAMES / 2 ))"
[[ "${rounds}" -ge 1 ]] || rounds=1

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
echo " Computing absolute ratings with ordo"
echo "   (${ANCHOR_TAG} fixed at ${ANCHOR_ELO}; white-advantage & draw-rate auto)"
echo "=============================================================="

# -W/-D: auto-fit white advantage and draw rate. -s: bootstrap error bars.
# -A anchors the reference release to ANCHOR_ELO so the scale is stable run-to-run.
ordo -q -W -D \
    -a "${ANCHOR_ELO}" -A "HydraY-${ANCHOR_TAG}" \
    -s 1000 -n "${CONCURRENCY}" \
    -p "${pgn_out}" \
    -o "${ratings_out}" \
    -j "gauntlet_${stamp}_h2h.txt"

echo
cat "${ratings_out}"
echo
echo "PGN     : ${pgn_out}"
echo "Ratings : ${ratings_out}"
echo "H2H     : gauntlet_${stamp}_h2h.txt"
