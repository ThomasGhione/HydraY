#!/usr/bin/env bash
#
# watch_calibration.sh — live absolute-Elo viewer for an in-progress
# run_sf_calibration.sh gauntlet.
#
# ordo's -m (multi-anchor) hard-fails if ANY named anchor has zero finished
# games in the PGN yet — which is normal for the first minute or two of a
# run, before every SF-<elo> level has completed at least one game. This
# filters the anchors file down to whichever anchors already appear in the
# PGN, so early polls degrade gracefully instead of erroring out.
#
# Also works around an ordo 1.2.6 bug: -m only reliably PINS an anchor when
# there are 2+ anchors in the file. With exactly one (e.g. a solo run vs one
# SF level, or early in a bracket run before a 2nd level has finished a
# game), it silently applies a partial/wrong shift instead of an exact pin
# (verified: a lone SF-3000 anchor came out at 2681, not 3000). The classic
# single-anchor -a/-A pair pins correctly in that case, so this script
# switches to it whenever exactly one anchor currently has data.
#
# Usage: watch -n 15 ./watch_calibration.sh
# (auto-finds the most recently started sfcal_*.pgn/anchors pair in this dir)

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"
export PATH="${HOME}/.local/bin:${PATH}"

pgn="$(ls -t sfcal_*.pgn 2>/dev/null | head -1)"
[[ -n "${pgn}" ]] || { echo "no sfcal_*.pgn found in ${script_dir}"; exit 1; }
anchors="${pgn%.pgn}_anchors.txt"
[[ -f "${anchors}" ]] || { echo "no matching anchors file: ${anchors}"; exit 1; }

live_anchors="$(mktemp)"
trap 'rm -f "${live_anchors}"' EXIT
while IFS=, read -r name elo; do
    bare="${name//\"/}"
    grep -qF "\"${bare}\"" "${pgn}" && printf '%s,%s\n' "${name}" "${elo}" >> "${live_anchors}"
done < "${anchors}"

live_count="$(wc -l < "${live_anchors}")"

if [[ "${live_count}" -eq 0 ]]; then
    echo "watching ${pgn}: no anchor has a finished game yet"
    exit 0
elif [[ "${live_count}" -eq 1 ]]; then
    IFS=, read -r name elo < "${live_anchors}"
    bare="${name//\"/}"
    ordo -q -G -W -D -a "${elo}" -A "${bare}" -p "${pgn}" -o /dev/stdout
else
    ordo -q -G -W -D -m "${live_anchors}" -p "${pgn}" -o /dev/stdout
fi
