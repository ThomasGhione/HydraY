#!/usr/bin/env bash
# HydraY interaction driver — talks UCI to ./chess over a coproc.
#
# Why this exists: piping a fixed command list (`printf '...go depth 12\nquit' | ./chess`)
# is RACY — the UCI loop handles `quit` mid-search by aborting (uci.cpp `stopSearch(true)`),
# so under CPU load the search dies before printing `bestmove`. This driver waits for the
# engine's replies (uciok / readyok / bestmove) before sending the next command.
#
# Usage (run from repo root):
#   script/engine_driver.sh smoke                # handshake + short search
#   script/engine_driver.sh bench [depth]        # startpos node count (default: 12)
#   script/engine_driver.sh bench6 [depth]       # canonical 6-position node-count bench
#   script/engine_driver.sh search "<uci moves>" [depth]
#   script/engine_driver.sh tui [outfile]        # tmux-driven terminal game smoke
#
# Eval is always NNUE (embedded net) — the HCE evaluator was removed (2.0.0).
set -u
BIN="${HYDRAY_BIN:-./chess}"
DIE() { echo "FAIL: $*" >&2; exit 1; }

[[ -x $BIN ]] || DIE "$BIN not found or not executable — run 'make prod' first"

# --- UCI session over a coproc -------------------------------------------
ENG_IN= ENG_OUT= ENG_PID_=

uci_open() { # <setup-commands>  — spawn engine, handshake, apply options
    local line ok=0
    coproc ENG { "$BIN" 2>/dev/null; }
    ENG_IN=${ENG[1]} ENG_OUT=${ENG[0]} ENG_PID_=$ENG_PID
    printf 'uci\n' >&"$ENG_IN"
    while IFS= read -r -t 20 line <&"$ENG_OUT"; do [[ $line == uciok ]] && { ok=1; break; }; done
    [[ $ok == 1 ]] || DIE "no uciok within 20s"
    [[ -n $1 ]] && printf '%s\n' "$1" >&"$ENG_IN"
    printf 'isready\n' >&"$ENG_IN"
    ok=0
    while IFS= read -r -t 20 line <&"$ENG_OUT"; do [[ $line == readyok ]] && { ok=1; break; }; done
    [[ $ok == 1 ]] || DIE "no readyok within 20s"
}

uci_go() { # <position-cmd> <go-cmd> <timeout-s>  — one fresh-TT search; prints output up to bestmove
    local line ok=0
    printf 'ucinewgame\n%s\n%s\n' "$1" "$2" >&"$ENG_IN"
    while IFS= read -r -t "$3" line <&"$ENG_OUT"; do
        printf '%s\n' "$line"
        [[ $line == bestmove* ]] && { ok=1; break; }
    done
    [[ $ok == 1 ]] || DIE "no bestmove within ${3}s for: $1"
}

uci_close() {
    printf 'quit\n' >&"$ENG_IN" 2>/dev/null
    wait "$ENG_PID_" 2>/dev/null
}

setup_common() {
    # Threads=1 keeps node counts deterministic (Lazy SMP is not reproducible).
    printf 'setoption name Threads value 1'
}

nodes_of() { # <info-line>
    sed -n 's/.* nodes \([0-9]*\).*/\1/p' <<<"$1"
}

# --- commands --------------------------------------------------------------
cmd_smoke() {
    local out
    uci_open "$(setup_common)"
    out=$(uci_go 'position startpos' 'go depth 8' 60) || exit 1
    uci_close
    grep -q '^bestmove [a-h][1-8][a-h][1-8]' <<<"$out" || DIE "malformed bestmove: $(tail -1 <<<"$out")"
    echo "OK: $(grep '^bestmove' <<<"$out")"
    echo "SMOKE PASS"
}

cmd_bench() {
    local depth=${1:-12} out info
    uci_open "$(setup_common)"
    out=$(uci_go 'position startpos' "go depth $depth" 300) || exit 1
    uci_close
    info=$(grep "^info depth $depth " <<<"$out" | tail -1)
    [[ -n $info ]] || DIE "no 'info depth $depth' line"
    echo "$info"
    echo "nodes=$(nodes_of "$info") depth=$depth"
}

# Canonical 6-position set. The total is a behaviour fingerprint: a pure refactor
# must leave it identical, a pruning/ordering change moves it.
# Baseline @ depth 12 depends on the embedded net:
# 4,595,112 (v3, current) · 3,943,540 (v2) · 4,735,578 (v1, release 2.0.0).
# NB: the usage banner is built from lines matching '^#   ' — keep other
# comments off that exact indentation.
BENCH6_NAMES=(startpos kiwipete kp-endgame midgame tactical open)
BENCH6_POS=(
    'position startpos'
    'position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1'
    'position fen 8/2k5/3p4/p2P1p2/P2P1P2/8/8/4K3 w - - 0 1'
    'position fen r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 1'
    'position fen r1bqk2r/pp2bppp/2n1pn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 b kq - 0 1'
    'position fen r2qkbnr/ppp2ppp/2np4/4p3/2B1P1b1/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 1'
)

cmd_bench6() {
    local depth=${1:-12} i out info n total=0
    uci_open "$(setup_common)"
    for i in "${!BENCH6_POS[@]}"; do
        out=$(uci_go "${BENCH6_POS[$i]}" "go depth $depth" 300) || exit 1
        info=$(grep "^info depth $depth " <<<"$out" | tail -1)
        [[ -n $info ]] || DIE "no 'info depth $depth' line for ${BENCH6_NAMES[$i]}"
        n=$(nodes_of "$info")
        printf '%-12s nodes=%s\n' "${BENCH6_NAMES[$i]}" "$n"
        total=$((total + n))
    done
    uci_close
    echo "TOTAL nodes=$total depth=$depth"
}

cmd_search() {
    local moves=$1 depth=${2:-12}
    uci_open "$(setup_common)"
    uci_go "position startpos moves $moves" "go depth $depth" 300
    uci_close
}

cmd_tui() {
    local out=${1:-/tmp/hydray-tui-capture.txt} ses="hydray-tui-$$"
    command -v tmux >/dev/null || DIE "tmux not installed"
    tmux new-session -d -s "$ses" -x 100 -y 40 "$BIN -pvb w" || DIE "tmux session failed"
    sleep 2                                  # engine init (magic tables, TT)
    # NB: move format is 'e2e4' (no space)
    tmux send-keys -t "$ses" "e2e4" Enter
    # engine replies on its own clock; poll until the board shows Black moved (max ~30s)
    for _ in $(seq 30); do
        sleep 1
        tmux capture-pane -t "$ses" -p > "$out" 2>/dev/null || break
        # after Black's reply it's White's prompt again with a non-start board
        grep -q '4 . . . . P' "$out" && tail -20 "$out" | grep -q 'Enter your move' && break
    done
    tmux capture-pane -t "$ses" -p > "$out" 2>/dev/null
    tmux kill-session -t "$ses" 2>/dev/null
    [[ -s $out ]] || DIE "empty TUI capture"
    grep -q 8 "$out" || DIE "no board rank markers in capture"
    echo "TUI capture written to $out ($(wc -l < "$out") lines)"
}

case ${1:-} in
    smoke)  cmd_smoke ;;
    bench)  shift; cmd_bench "$@" ;;
    bench6) shift; cmd_bench6 "$@" ;;
    search) shift; [[ $# -ge 1 ]] || DIE "usage: engine_driver.sh search \"<uci moves>\" [depth]"; cmd_search "$@" ;;
    tui)    shift; cmd_tui "$@" ;;
    *)      grep '^#   ' "$0" | sed 's/^#   //'; exit 2 ;;
esac
