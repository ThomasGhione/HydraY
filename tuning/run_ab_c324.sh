#!/usr/bin/env bash
# Verdict of the Chess324 A/B (ELO_ROADMAP 4.4): the two nets trained by
# nnue/trainer/colab_ab_c324.ipynb, head to head on one build. Fixed nodes, so
# the datagen running alongside changes how long it takes, not the result.
# Chess324 joins v8 only on H1.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

for arm in std c324; do
    net="$repo/nnue/net/ab_$arm.nnue"
    [[ "$(stat -c %s "$net" 2>/dev/null)" == 6425664 ]] \
        || { echo "$net: missing, or not a 1024 -> 16 -> 1 net" >&2; exit 1; }
done
! cmp -s "$repo/nnue/net/ab_std.nnue" "$repo/nnue/net/ab_c324.nnue" \
    || { echo "the two nets are the same file" >&2; exit 1; }

export BASE_BIN="$repo/chess" NODES=150000 ELO0=0 ELO1=5 CONCURRENCY="${CONCURRENCY:-8}"
# run_sprt.sh caps at 4000 games by default: the first run of this test hit it.
export MAXGAMES="${MAXGAMES:-40000}"
export NEW_OPTS="EvalFile=$repo/nnue/net/ab_c324.nnue"
export BASE_OPTS="EvalFile=$repo/nnue/net/ab_std.nnue"
exec "$repo/tuning/run_sprt.sh"
