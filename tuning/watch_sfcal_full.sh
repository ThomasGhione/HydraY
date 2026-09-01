#!/usr/bin/env bash
# Full live view of an SF calibration run: ordo's fitted ratings plus the raw
# per-level score, which is what actually moves between runs. The aggregate
# rating hides where the strength changed; the per-level score does not.
#
#   watch -n 30 tuning/watch_sfcal_full.sh [logfile]
set -uo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
# Default to the most recently modified calibration log, NOT a hardcoded name:
# ordo (above) always fits the newest PGN, so a stale default here silently
# pairs a live rating with a previous run's per-level table — which reads as a
# real result and is not obviously wrong. Pass a filename to pin one run.
log="${1:-$(ls -t log_sfcal_*.txt 2>/dev/null | head -1)}"
if [[ -z "${log}" || ! -f "${log}" ]]; then
    echo "no calibration log found (looked for log_sfcal_*.txt)" >&2
    exit 1
fi
echo "=== source: ${log} ======================================"

echo "=== fitted ratings (ordo) ==============================="
./watch_calibration.sh 2>/dev/null | sed -n '/PLAYER/,/^$/p'

echo "=== score per level (HydraY point of view) =============="
grep -oE "Finished game [0-9]+ \((HydraY-dev|SF-[0-9]+) vs (HydraY-dev|SF-[0-9]+)\): [01/2-]+" "$log" \
| awk '
{
    white = $4; sub(/^\(/, "", white)
    black = $6; sub(/\)/, "", black); sub(/:$/, "", black)
    res   = $7
    if (white ~ /^SF/) { lvl = white; hyWhite = 0 } else { lvl = black; hyWhite = 1 }
    sub(/\)/, "", lvl); sub(/:$/, "", lvl)
    n[lvl]++
    if (res == "1/2-1/2")      { d[lvl]++; pts[lvl] += 0.5 }
    else if (res == "1-0")     { if (hyWhite) { w[lvl]++; pts[lvl] += 1 } else l[lvl]++ }
    else if (res == "0-1")     { if (hyWhite) l[lvl]++; else { w[lvl]++; pts[lvl] += 1 } }
}
END {
    printf "  %-10s %5s %5s %5s %6s %8s %10s\n", "opponent","W","D","L","games","score%","implied"
    for (lvl in n) {
        s = pts[lvl] / n[lvl]
        if (s <= 0 || s >= 1) imp = "n/a"
        else { sfelo = lvl; sub(/SF-/, "", sfelo); imp = sprintf("%.0f", sfelo + 400*log(s/(1-s))/log(10)) }
        printf "  %-10s %5d %5d %5d %6d %8.1f %10s\n", lvl, w[lvl], d[lvl], l[lvl], n[lvl], 100*s, imp
    }
}' | sort -k1,1

echo "  (implied = that level's UCI_Elo + logistic conversion of our score"
echo "   against it alone; spread across levels = bracket incoherence)"
