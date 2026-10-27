#!/usr/bin/env bash
#
# watch_sprt.sh - compact live view of a run_sprt.sh log.
#
# run_sprt.sh's log is dominated by per-game PV dumps; this pulls out just the
# latest result block (Elo / LOS / LLR / game count) and says who is winning.
#
# Usage:
#   ./tuning/watch_sprt.sh                          # newest log_sprt_*.txt
#   ./tuning/watch_sprt.sh tuning/log_sprt_foo.txt  # a specific log
#   WATCH=1 ./tuning/watch_sprt.sh                  # refresh every 30s

set -uo pipefail
# Resolve a path argument against the CALLER's directory, not this one: the cd
# below used to make the invocation in the usage block above fail with "no SPRT
# log found", because tuning/log_sprt_foo.txt was looked up inside tuning/.
orig_pwd="${PWD}"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

log="${1:-}"
if [[ -n "${log}" ]]; then
    if [[ "${log}" != /* && -f "${orig_pwd}/${log}" ]]; then
        log="${orig_pwd}/${log}"
    fi
    if [[ ! -f "${log}" ]]; then
        echo "no such SPRT log: ${1}" >&2
        exit 1
    fi
else
    log="$(ls -t log_sprt_*.txt 2>/dev/null | head -1)"
    if [[ -z "${log}" || ! -f "${log}" ]]; then
        echo "no SPRT log found (looked for tuning/log_sprt_*.txt)" >&2
        exit 1
    fi
fi

show() {
    local block
    block=$(grep -nE "^Results of|^Elo:|^LOS:|^Games:|^LLR:" "${log}" | tail -5)
    if [[ -z "${block}" ]]; then
        echo "$(date +%H:%M:%S)  ${log##*/}: no result block yet ($(grep -c '^Started game' "${log}" 2>/dev/null || echo 0) games started)"
        return
    fi

    local elo nelo los llr games ptnml
    # The result line is "Elo: X +/- Y, nElo: Z +/- W" -- take the field BEFORE
    # the comma. Trailing-colon matching silently reports nElo as Elo, which
    # reads ~1.6x larger and has already caused one misreported result.
    elo=$(grep '^Elo:'   "${log}" | tail -1 | sed 's/^Elo: //; s/,.*//')
    nelo=$(grep '^Elo:'  "${log}" | tail -1 | sed 's/.*nElo: //')
    los=$(grep '^LOS:'   "${log}" | tail -1 | sed 's/LOS: //; s/,.*//')
    games=$(grep '^Games:' "${log}" | tail -1 | sed 's/Games: //')
    llr=$(grep '^LLR:'   "${log}" | tail -1 | sed 's/LLR: //')
    ptnml=$(grep '^Ptnml' "${log}" | tail -1)

    echo "=== ${log##*/}  @ $(date +%H:%M:%S) ==="
    echo "  Elo   : ${elo}       <- this is the Elo"
    echo "  nElo  : ${nelo}   (normalised; larger by construction, not the Elo)"
    echo "  LOS   : ${los}"
    echo "  LLR   : ${llr}"
    echo "  Games : ${games}"
    [[ -n "${ptnml}" ]] && echo "  ${ptnml}"

    # Verdict: LLR crossing a bound is the decision; Elo sign is only a hint.
    local llrval lo hi
    llrval=$(awk '{print $1}' <<<"${llr}")
    lo=$(sed -E 's/.*\(([-0-9.]+), ([-0-9.]+)\).*/\1/' <<<"${llr}")
    hi=$(sed -E 's/.*\(([-0-9.]+), ([-0-9.]+)\).*/\2/' <<<"${llr}")
    if [[ -n "${llrval}" && -n "${hi}" ]]; then
        awk -v v="${llrval}" -v lo="${lo}" -v hi="${hi}" 'BEGIN{
            if (v >= hi)      print "  VERDICT: H1 accepted - the change is a GAIN, keep it.";
            else if (v <= lo) print "  VERDICT: H0 accepted - no measurable gain, discard it.";
            else {
                pct = (hi != 0) ? 100*v/hi : 0;
                printf "  VERDICT: still running (LLR %.0f%% of the way to H1)\n", pct;
            }
        }'
    fi
    if pgrep -x fastchess >/dev/null 2>&1 || pgrep -x cutechess-cli >/dev/null 2>&1; then
        echo "  engine: RUNNING"
    else
        echo "  engine: not running (finished or stopped)"
    fi
}

if [[ "${WATCH:-0}" == "1" ]]; then
    while true; do clear; show; sleep 30; done
else
    show
fi
