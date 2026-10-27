#!/usr/bin/env bash
# ==============================================================
#  runfastchess.sh - fastchess gauntlet or calibration between UCI engines
#
#  Usage: ./tuning/runfastchess.sh                Stockfish calibration (default levels)
#         ./tuning/runfastchess.sh --tag 3.1.0 --tag 3.0.0
#         ./tuning/runfastchess.sh --sf-levels ../stockfish/stockfish-18:2800,3000,3190
#         ./tuning/runfastchess.sh --opponent ../stockfish/stockfish-7
#         ./tuning/runfastchess.sh --show-cpus
#         ./tuning/runfastchess.sh --help
#
#  Relative paths are relative to tuning/. It does not rebuild ../chess:
#  run make prod first.
#
#  No menus: everything is set by arguments, the rest has a sensible
#  default. Cores and hash are worked out automatically (see --help).
# ==============================================================
set -euo pipefail

# awk prints decimals in the current locale: under it_IT "0.6" comes back
# as "0,6" and the next step reads it as zero. Numbers stay in C, the rest
# of the locale does not.
export LC_NUMERIC=C
# fastchess and ordo are often built into ~/.local/bin.
export PATH="$HOME/.local/bin:$PATH"

# ---------------- constants -----------------------------------
readonly PGN_DEFAULT="elo_estimate.pgn"
readonly CONF=".elo_run.conf"
readonly STAMP=".elo_watch_start"
readonly CACHE=".elo_engines.cache"
readonly REFS=".refs"
readonly ENGINE_DEFAULT="../chess"
# Opponents when none is given, as for --sf-levels; empty or a missing
# binary means every UCI engine in this folder instead.
readonly DEFAULT_SF_LEVELS="../stockfish/stockfish-18:2800,3000,3190"

# For the automatic hash: estimated nodes per second per engine (on the
# high side, so the table is never too small) and bytes per TT entry.
readonly NPS_ESTIMATE=2000000
readonly BYTES_PER_NODE=16
readonly HASH_MIN=16
readonly HASH_MAX=1024

# Clock check: an engine has 4 s + 0.04 s and must answer well within this.
readonly CLOCK_LIMIT_MS=1500

# ---------------- defaults ------------------------------------
ENGINE_BIN=""
ENGINE_FIXNAME=""
ENGINE_OPTS=""                  # newline-separated Name=Value
# One entry per opponent, all arrays in step.
OPP_KIND=()                     # path | tag | sf
OPP_SRC=()                      # path, tag, or path of the Stockfish binary
OPP_LEVEL=()                    # UCI_Elo for sf, empty otherwise
OPP_FIXNAME=()                  # --name, empty = from the UCI name
OPP_OPTS=()                     # newline-separated Name=Value
LAST=""                         # target of --name/--opt: engine or an index
declare -A ANCHOR_FOR=()        # --anchor NAME=ELO or NAME=-
EXTRA_PGNS=()
ROUNDS=500
TC=10+0.1
THREADS=1
CORES_WANTED=""                 # empty = all the chosen ones
CPUS_FORCED=""                  # empty = automatic choice
HASH=auto
ANCHOR_ELO=""                   # --anchor N, single opponent only
BOOK=""
ORDER=sequential
PLIES=8
PGN="$PGN_DEFAULT"
PGN_MODE=archive                # archive | delete | append
RESIGN_TWOSIDED=true
DRY_RUN=0
REPORT=1
LIST=0
SHOW_CPUS=0

# ---------------- looks ---------------------------------------
if [ -t 1 ]; then
  B=$'\e[1m'; DIM=$'\e[2m'
  GR=$'\e[32m'; YE=$'\e[33m'; RE=$'\e[31m'; CY=$'\e[36m'; N=$'\e[0m'
else
  B=''; DIM=''; GR=''; YE=''; RE=''; CY=''; N=''
fi

# ---------------- utilities -----------------------------------
die()  { printf '%s  error: %s%s\n' "$RE" "$1" "$N" >&2; exit "${2:-1}"; }
warn() { printf '  %s%s%s\n' "$YE" "$1" "$N"; }
hms()  { printf '%02d:%02d:%02d' $(( $1 / 3600 )) $(( $1 % 3600 / 60 )) $(( $1 % 60 )); }

# "0-3,8,10-11" -> "0 1 2 3 8 10 11"
expand_list() {
  local part a b out=() parts=()
  IFS=',' read -ra parts <<<"$1"
  for part in "${parts[@]}"; do
    [ -n "$part" ] || continue
    if [[ $part == *-* ]]; then
      a=${part%-*}; b=${part#*-}
      for (( ; a <= b; a++ )); do out+=("$a"); done
    else
      out+=("$part")
    fi
  done
  echo "${out[*]}"
}

# "4 5 6 7 9" -> "4-7,9"
compress_list() {
  local out="" first="" prev="" c
  for c in $(printf '%s\n' "$@" | sort -n); do
    if [ -z "$first" ]; then first=$c; prev=$c; continue; fi
    if [ "$c" -eq $(( prev + 1 )) ]; then prev=$c; continue; fi
    if [ "$first" = "$prev" ]; then out+="$first,"; else out+="$first-$prev,"; fi
    first=$c; prev=$c
  done
  if [ -n "$first" ]; then
    if [ "$first" = "$prev" ]; then out+="$first"; else out+="$first-$prev"; fi
  fi
  echo "$out"
}

# Base seconds, increment and moves per period from a cutechess-style
# time control: [moves/]minutes:seconds[+increment] or seconds[+inc].
split_tc() {
  awk -v tc="$1" 'BEGIN {
    moves = 0
    if (match(tc, /^[0-9]+\//)) {
      moves = substr(tc, 1, RLENGTH - 1) + 0
      tc = substr(tc, RLENGTH + 1)
    }
    inc = 0
    p = index(tc, "+")
    if (p > 0) { inc = substr(tc, p + 1) + 0; tc = substr(tc, 1, p - 1) }
    c = index(tc, ":")
    if (c > 0) base = substr(tc, 1, c - 1) * 60 + substr(tc, c + 1)
    else       base = tc + 0
    printf "%.3f %.3f %d", base, inc, moves
  }'
}

tc_valid() {
  printf '%s' "$1" | grep -qE '^([0-9]+/)?[0-9]+(:[0-9]+)?(\.[0-9]+)?(\+[0-9]+(\.[0-9]+)?)?$'
}

# Upper bound on the duration: each game spends both sides' budget, with
# about 60 moves each.
estimate_duration() { # $1 = games, $2 = tc, $3 = concurrency
  local base inc moves
  read -r base inc moves <<<"$(split_tc "$2")"
  awk -v g="$1" -v b="$base" -v i="$inc" -v m="$moves" -v c="$3" 'BEGIN {
    typical = 60
    periods = (m > 0) ? int((typical + m - 1) / m) : 1
    printf "%d", g * 2 * (periods * b + typical * i) / c
  }'
}

# ---------------- cores ---------------------------------------
# One logical thread per physical core, cores grouped by max frequency.
# A single group (homogeneous CPU) means all of them. On a hybrid CPU the
# fastest group is left to the system, fastchess and ordo, and the games
# go on the largest of the other groups: on recent Intel parts that is
# the E-cores, while the few slow LP-E stay out.
declare -A CPU_GROUP=()
SORTED_GROUPS=()

detect_groups() {
  local c n sib first freq kind
  local p_cores=" $(expand_list "$(cat /sys/devices/cpu_core/cpus 2>/dev/null || true)") "
  for c in /sys/devices/system/cpu/cpu[0-9]*; do
    n=${c##*cpu}
    [ -r "$c/online" ] && [ "$(cat "$c/online")" = 0 ] && continue
    sib=$(cat "$c/topology/thread_siblings_list" 2>/dev/null || echo "$n")
    first=${sib%%[,-]*}
    [ "$first" = "$n" ] || continue
    freq=$(cat "$c/cpufreq/cpuinfo_max_freq" 2>/dev/null || echo 0)
    if [ "$p_cores" = "  " ]; then kind=-
    elif [[ $p_cores == *" $n "* ]]; then kind=P
    else kind=E; fi
    CPU_GROUP["$freq:$kind"]+="$n "
  done
  [ "${#CPU_GROUP[@]}" -gt 0 ] || die "cannot read the CPU topology from /sys"
  mapfile -t SORTED_GROUPS < <(printf '%s\n' "${!CPU_GROUP[@]}" | sort -t: -k1,1nr)
}

choose_cpus() {
  local g best="" n_best=0 n
  if [ "${#SORTED_GROUPS[@]}" -eq 1 ]; then
    echo "${CPU_GROUP[${SORTED_GROUPS[0]}]}"; return
  fi
  for g in "${SORTED_GROUPS[@]:1}"; do
    n=$(wc -w <<<"${CPU_GROUP[$g]}")
    if [ "$n" -gt "$n_best" ]; then best=$g; n_best=$n; fi
  done
  echo "${CPU_GROUP[$best]}"
}

describe_groups() { # $1 = chosen cpus
  local g freq kind cpus name ghz state c used chosen=" $1 "
  for g in "${SORTED_GROUPS[@]}"; do
    freq=${g%%:*}; kind=${g##*:}; cpus=${CPU_GROUP[$g]}
    case $kind in P) name=P-core ;; E) name=E-core ;; *) name=core ;; esac
    # An E group slower than another E group: the SoC's LP-E cores.
    if [ "$kind" = E ]; then
      for c in "${SORTED_GROUPS[@]}"; do
        [ "${c##*:}" = E ] && [ "${c%%:*}" -gt "$freq" ] && { name=LP-E; break; }
      done
    fi
    ghz=$(awk -v f="$freq" 'BEGIN { if (f > 0) printf "%.1f GHz", f / 1e6; else printf "? GHz" }')
    used=0
    for c in $cpus; do [[ $chosen == *" $c "* ]] && used=$(( used + 1 )); done
    if [ "$used" -gt 0 ]; then state="${GR}games ($used)${N}"; else state="${DIM}free${N}"; fi
    printf '    %-7s %-8s %2d cores  cpu %-8s %s\n' "$name" "$ghz" "$(wc -w <<<"$cpus")" \
           "$(compress_list $cpus)" "$state"
  done
}

# ---------------- hash ----------------------------------------
# Enough entries for the nodes of an average move, rounded up to a power of
# two, then halved until all the hash tables together fit in half of the
# free RAM.
auto_hash() { # $1 = tc, $2 = concurrency
  local base inc moves mb free
  read -r base inc moves <<<"$(split_tc "$1")"
  mb=$(awk -v b="$base" -v i="$inc" -v m="$moves" -v nps="$NPS_ESTIMATE" -v bpn="$BYTES_PER_NODE" \
           -v lo="$HASH_MIN" -v hi="$HASH_MAX" 'BEGIN {
    per_move = (m > 0 ? b / m : b / 40) + i
    need = per_move * nps * bpn / 1048576
    h = lo; while (h < need && h < hi) h *= 2
    printf "%d", h
  }')
  free=$(awk '/MemAvailable/ { printf "%d", $2 / 1024 }' /proc/meminfo 2>/dev/null || echo 0)
  if [ "$free" -gt 0 ]; then
    while [ "$mb" -gt 1 ] && [ $(( 2 * $2 * mb )) -gt $(( free / 2 )) ]; do mb=$(( mb / 2 )); done
  fi
  echo "$mb"
}

# ---------------- engines -------------------------------------
# An engine is an executable ELF that answers "id name" to uci. Names are
# cached with the binary's mtime: rebuild it and the entry is redone.
is_elf() {
  [ -f "$1" ] && [ -x "$1" ] || return 1
  [ "$(od -An -tx1 -N4 "$1" 2>/dev/null | tr -d ' \n')" = "7f454c46" ]
}

uci_output() { printf 'uci\nquit\n' 2>/dev/null | timeout 5 "$1" 2>/dev/null | tr -d '\r'; }

probe_uci() { # $1 = path -> UCI name, empty if not an engine
  uci_output "$1" | sed -n 's/^id name //p' | head -1
}

# Milliseconds the engine takes to answer "go" with 4 s + 0.04 s on the
# clock, empty if it does not answer within 3 s. An engine that ignores the
# clock (old HydraY tags before 2.0.0 did) loses every game on time.
probe_clock() { # $1 = path; result in CLOCK_MS
  local line t0 in out pid ready=0
  CLOCK_MS=""
  # An engine that dies early must not take this script down with SIGPIPE.
  trap '' PIPE
  coproc CLOCK { timeout 8 "$1" 2>/dev/null; }
  pid=$CLOCK_PID
  exec {in}<&"${CLOCK[0]}" {out}>&"${CLOCK[1]}"
  printf 'uci\nisready\nucinewgame\nposition startpos\nisready\n' >&"$out" 2>/dev/null || true
  while IFS= read -r -t 5 line <&"$in"; do
    [[ $line == readyok* ]] && { ready=$(( ready + 1 )); [ "$ready" -eq 2 ] && break; }
  done
  t0=$(date +%s%N)
  printf 'go wtime 4000 btime 4000 winc 40 binc 40\n' >&"$out" 2>/dev/null || true
  while IFS= read -r -t 3 line <&"$in"; do
    if [[ $line == bestmove* ]]; then CLOCK_MS=$(( ($(date +%s%N) - t0) / 1000000 )); break; fi
  done
  printf 'quit\n' >&"$out" 2>/dev/null || true
  exec {in}<&- {out}>&-
  wait "$pid" 2>/dev/null || true
  trap - PIPE
}

list_engines() {
  local -A cache_name=() cache_mtime=()
  local path mtime name found=0
  if [ -r "$CACHE" ]; then
    while IFS=$'\t' read -r path mtime name; do
      [ -n "${path:-}" ] || continue
      cache_name["$path"]="$name"; cache_mtime["$path"]="$mtime"
    done < "$CACHE"
  fi
  : > "$CACHE.tmp"
  printf '\n  %sUCI engines found%s\n' "$B$CY" "$N"
  while IFS= read -r path; do
    is_elf "$path" || continue
    mtime=$(stat -c %Y "$path" 2>/dev/null || echo 0)
    if [ "${cache_mtime[$path]:-}" = "$mtime" ]; then name="${cache_name[$path]}"
    else name=$(probe_uci "$path"); fi
    printf '%s\t%s\t%s\n' "$path" "$mtime" "$name" >> "$CACHE.tmp"
    [ -n "$name" ] || continue
    printf '  %-36s %s%s%s\n' "$path" "$B" "$name" "$N"; found=1
  done < <(find . -maxdepth 3 -type f -executable \
             -not -path '*/.git/*' -not -path '*/output/*' -not -path '*/target/*' \
             -not -path '*/trainer/*' -not -path '*/books/*' \
             -not -name '*.sh' -not -name '*.py' -not -name '*.o' -not -name '*.d' \
             2>/dev/null | sort)
  mv -f "$CACHE.tmp" "$CACHE"
  [ "$found" -eq 1 ] || printf '  %snone%s\n' "$DIM" "$N"
  echo
}

# Rough Elo of a few known opponents, used as anchors when none is given.
# Empty for anything else.
suggested_anchor() {
  case "$1" in
    *"Stockfish 7"*)  echo 3250 ;;
    *"Stockfish 8"*)  echo 3300 ;;
    *"Stockfish 9"*)  echo 3350 ;;
    *"Stockfish 10"*) echo 3400 ;;
  esac
}

label() { printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_' | sed 's/_\+/_/g; s/_$//'; }

# Build a git tag of the engine's repository into $REFS/chess_<tag>, once.
# The tree comes from git archive, so the checkout is never touched.
build_tag() { # $1 = repo, $2 = tag -> path in BUILT
  local repo=$1 tag=$2 tmp log
  BUILT="$REFS/chess_$(label "$tag")"
  [ -x "$BUILT" ] && return 0
  git -C "$repo" rev-parse -q --verify "refs/tags/$tag" >/dev/null \
    || die "tag $tag does not exist in $repo" 2
  mkdir -p "$REFS"
  tmp=$(mktemp -d); log="$REFS/build_$(label "$tag").log"
  printf '  building tag %s ... ' "$tag"
  git -C "$repo" archive "$tag" | tar -x -C "$tmp"
  if ! make -C "$tmp" -j"$(nproc)" prod >"$log" 2>&1 || [ ! -x "$tmp/chess" ]; then
    rm -rf "$tmp"; printf 'failed\n'
    die "make prod failed for tag $tag (see $log)"
  fi
  cp "$tmp/chess" "$BUILT"; rm -rf "$tmp"
  printf 'done\n'
}

# If the engine lives in a git repo: branch, commit and the three cases that
# skew an estimate without any error (stale binary, dirty tree, branch behind).
BRANCH=""; COMMIT=""; REPO=""; PROVENANCE_NOTES=()
provenance() {
  local dir t_bin t_commit dirty behind
  dir=$(cd "$(dirname "$1")" 2>/dev/null && pwd) || return 0
  git -C "$dir" rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 0
  REPO=$(git -C "$dir" rev-parse --show-toplevel)
  BRANCH=$(git -C "$dir" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')
  COMMIT=$(git -C "$dir" rev-parse --short HEAD 2>/dev/null || echo '?')
  t_bin=$(stat -c %Y "$1" 2>/dev/null || echo 0)
  t_commit=$(git -C "$dir" log -1 --format=%ct 2>/dev/null || echo 0)
  [ "$t_bin" -lt "$t_commit" ] && PROVENANCE_NOTES+=(
    "the binary is older than the last commit: rebuild before measuring")
  dirty=$(git -C "$dir" status --porcelain --untracked-files=no 2>/dev/null | wc -l)
  [ "$dirty" -gt 0 ] && PROVENANCE_NOTES+=(
    "$dirty modified files not committed: the commit does not describe the binary")
  behind=$(git -C "$dir" rev-list --count 'HEAD..@{upstream}' 2>/dev/null || echo 0)
  [ "${behind:-0}" -gt 0 ] && PROVENANCE_NOTES+=(
    "the branch is $behind commits behind its remote: maybe you need a pull")
  return 0
}

# ---------------- help ----------------------------------------
usage() {
  cat <<HELP
${B}runfastchess.sh${N} - fastchess gauntlet or calibration between UCI engines

${B}USAGE${N}
  ./runfastchess.sh [opponents] [options]

${B}ENGINES${N}
  --engine PATH      engine under test (default $ENGINE_DEFAULT)
  --opponent PATH    an opponent binary; repeatable
  --tag TAG          an old version of the engine under test, built once from
                     that git tag of its repository into $REFS/; repeatable
  --sf-levels PATH:ELO,ELO,...
                     one opponent per level, the Stockfish binary at PATH
                     limited with UCI_LimitStrength/UCI_Elo, each anchored at
                     its level. The binary must have UCI_Elo (Stockfish 11+)
  --name NAME        name of the last --engine, --opponent or --tag in the
                     games and in ordo
  --opt NAME=VALUE   UCI option of the last --engine, --opponent or --tag;
                     repeatable (e.g. --opt EvalFile=/abs/net.nnue)
  --list-engines     list the UCI engines found below this folder and exit

$(if [ -n "$DEFAULT_SF_LEVELS" ]; then
    printf '  With no opponents at all: --sf-levels %s\n' "$DEFAULT_SF_LEVELS"
    printf '  if that binary exists, otherwise every UCI engine in this folder (not\n'
    printf '  in subfolders) except the engine under test.'
  else
    printf '  With no opponents at all: every UCI engine in this folder (not in\n'
    printf '  subfolders) except the engine under test.'
  fi)

  One opponent is a match, more are a gauntlet: the engine under test
  plays each, they do not play each other.

${B}GAMES${N}
  --tc STR           time control [moves/]min:sec[+inc] or sec[+inc] (default $TC)
  --rounds N         rounds per opponent, 2 games each with colours
                     reversed (default $ROUNDS)
  --book FILE        opening book, pgn or epd (default: the first in */book*)
  --plies N          book plies (default $PLIES)
  --order STR        sequential or random (default $ORDER)
  --resign-onesided  resignation decided by one engine alone. The default is
                     two-sided: with different eval scales (HydraY ~2.3-3.2x
                     Stockfish) the one-sided rule cuts losses short and lets
                     wins run long. Only useful to compare with old runs

${B}RESOURCES${N}
  --threads N        threads per engine (default $THREADS); concurrency = cores / threads
  --cores N          use only the first N of the chosen cores
  --cpus LIST        force the cpus, e.g. 4-11 or 4,6,8-11 (no automatic choice)
  --hash MB|auto     hash per engine (default auto); --opt Hash=N overrides
                     it for one engine
  --show-cpus        show the core groups, the choice and the hash, and exit

  Automatic cores: one logical thread per physical core, grouped by max
  frequency. Homogeneous CPU: all of them. Hybrid CPU: the fastest group
  stays free for the system, fastchess and ordo; the games go on the
  largest of the other groups (here the E-cores; not the slow LP-E).

  Automatic hash: enough for the nodes of an average move
  (${NPS_ESTIMATE} nodes/s x ${BYTES_PER_NODE} bytes), as a power of two between ${HASH_MIN} and
  ${HASH_MAX} MB, then halved until all the hash tables together fit in
  half of the free RAM.

${B}ESTIMATE${N}
  --anchor NAME=ELO  pin an opponent at ELO in ordo; repeatable. NAME is the
                     --name, the path, the file name or the UCI name
  --anchor NAME=-    leave that opponent free (rated through the games)
  --anchor N         with a single opponent, its Elo
  --extra-pgn FILE   more games for ordo, e.g. the opponents playing each
                     other in another run; repeatable

  Anchors: Stockfish levels at their UCI_Elo; known Stockfish versions at a
  rough value, only if nothing else is anchored; if still nothing is, the
  first opponent at 3000. ordo pins all anchors together, so their
  differences enter the estimate: keep them consistent.

${B}OUTPUT${N}
  --pgn FILE         where to write the games (default $PGN_DEFAULT)
  --pgn-mode MODE    if the pgn already exists: archive (default, renamed with
                     date and time), delete, append
  --no-report        no ordo report at the end
  --dry-run          print the summary and the fastchess command, run nothing
  --help             this message

  Writes $CONF, which elo_watch.sh reads to know what it is watching.
  At the end (also after Ctrl+C) elo_watch.sh --save writes next to the
  pgn: _ratings.txt, _h2h.txt and _anchors.txt.

${B}EXAMPLES${N}  (relative paths are relative to tuning/)
  ./tuning/runfastchess.sh
  ./tuning/runfastchess.sh --tag 2.0.0 --tag 2.1.0 --tag 3.0.0 --tag 3.1.0 --anchor HydraY_2.0.0=3000
  ./tuning/runfastchess.sh --sf-levels ../stockfish/stockfish-18:2800,3000,3190 --rounds 100
  ./tuning/runfastchess.sh --sf-levels ../stockfish/stockfish-18:3000,3190 \\
                           --opponent ../stockfish/stockfish-11 --anchor stockfish-11=-
  ./tuning/runfastchess.sh --opponent ../chess --name newnet --opt EvalFile=/abs/new.nnue
HELP
}

# ---------------- arguments -----------------------------------
need_value() { [ "$2" -ge 2 ] || die "$1 needs a value (try --help)" 2; }

add_opp() { # $1 = kind, $2 = source, $3 = level
  OPP_KIND+=("$1"); OPP_SRC+=("$2"); OPP_LEVEL+=("${3:-}"); OPP_FIXNAME+=(""); OPP_OPTS+=("")
  LAST=$(( ${#OPP_KIND[@]} - 1 ))
}

while [ $# -gt 0 ]; do
  case "$1" in
    --engine)          need_value "$1" $#; ENGINE_BIN="$2"; LAST=engine; shift 2 ;;
    --opponent)        need_value "$1" $#; add_opp path "$2"; shift 2 ;;
    --tag)             need_value "$1" $#; add_opp tag "$2"; shift 2 ;;
    --sf-levels)       need_value "$1" $#
                       [[ $2 == *:* ]] || die "--sf-levels wants PATH:ELO,ELO,..." 2
                       IFS=',' read -ra levels <<<"${2##*:}"
                       [ "${#levels[@]}" -gt 0 ] || die "--sf-levels: no levels in '$2'" 2
                       for l in "${levels[@]}"; do add_opp sf "${2%:*}" "$l"; done
                       LAST=""; shift 2 ;;
    --name)            need_value "$1" $#
                       case "$LAST" in
                         "")     die "--name must follow --engine, --opponent or --tag" 2 ;;
                         engine) ENGINE_FIXNAME=$(label "$2") ;;
                         *)      OPP_FIXNAME[$LAST]=$(label "$2") ;;
                       esac
                       shift 2 ;;
    --opt)             need_value "$1" $#
                       [[ $2 == ?*=* ]] || die "--opt wants NAME=VALUE, got '$2'" 2
                       case "$LAST" in
                         "")     die "--opt must follow --engine, --opponent or --tag" 2 ;;
                         engine) ENGINE_OPTS+="$2"$'\n' ;;
                         *)      OPP_OPTS[$LAST]+="$2"$'\n' ;;
                       esac
                       shift 2 ;;
    --list-engines)    LIST=1; shift ;;
    --tc)              need_value "$1" $#; TC="$2"; shift 2 ;;
    --rounds)          need_value "$1" $#; ROUNDS="$2"; shift 2 ;;
    --book)            need_value "$1" $#; BOOK="$2"; shift 2 ;;
    --plies)           need_value "$1" $#; PLIES="$2"; shift 2 ;;
    --order)           need_value "$1" $#; ORDER="$2"; shift 2 ;;
    --resign-onesided) RESIGN_TWOSIDED=false; shift ;;
    --threads)         need_value "$1" $#; THREADS="$2"; shift 2 ;;
    --cores)           need_value "$1" $#; CORES_WANTED="$2"; shift 2 ;;
    --cpus)            need_value "$1" $#; CPUS_FORCED="$2"; shift 2 ;;
    --hash)            need_value "$1" $#; HASH="$2"; shift 2 ;;
    --show-cpus)       SHOW_CPUS=1; shift ;;
    --anchor)          need_value "$1" $#
                       if [[ $2 == *=* ]]; then ANCHOR_FOR["${2%=*}"]="${2##*=}"
                       else ANCHOR_ELO="$2"; fi
                       shift 2 ;;
    --extra-pgn)       need_value "$1" $#; EXTRA_PGNS+=("$2"); shift 2 ;;
    --pgn)             need_value "$1" $#; PGN="$2"; shift 2 ;;
    --pgn-mode)        need_value "$1" $#; PGN_MODE="$2"; shift 2 ;;
    --no-report)       REPORT=0; shift ;;
    --dry-run)         DRY_RUN=1; shift ;;
    --help|-h)         usage; exit 0 ;;
    *)                 die "unknown argument: $1 (try --help)" 2 ;;
  esac
done

cd "$(dirname "$0")"

[ "$LIST" -eq 1 ] && { list_engines; exit 0; }

# ---------------- validation ----------------------------------
integer() { case "$2" in ''|*[!0-9]*) die "invalid $1: '$2'" 2 ;; esac; }
integer --rounds  "$ROUNDS";  [ "$ROUNDS" -gt 0 ]  || die "--rounds must be > 0" 2
integer --threads "$THREADS"; [ "$THREADS" -ge 1 ] || die "--threads must be >= 1" 2
integer --plies   "$PLIES"
tc_valid "$TC" || die "invalid time control: '$TC' (expected [moves/]min:sec[+inc])" 2
case "$ORDER"    in sequential|random) ;; *) die "invalid --order: $ORDER" 2 ;; esac
case "$PGN_MODE" in archive|delete|append) ;; *) die "invalid --pgn-mode: $PGN_MODE" 2 ;; esac
if [ "$HASH" != auto ]; then
  integer --hash "$HASH"; [ "$HASH" -ge 1 ] || die "--hash must be >= 1" 2
fi
[ -z "$ANCHOR_ELO" ] || integer --anchor "$ANCHOR_ELO"
for k in "${!ANCHOR_FOR[@]}"; do
  [ "${ANCHOR_FOR[$k]}" = - ] || integer "--anchor $k" "${ANCHOR_FOR[$k]}"
done
for i in "${!OPP_LEVEL[@]}"; do
  [ -z "${OPP_LEVEL[$i]}" ] || integer "--sf-levels level" "${OPP_LEVEL[$i]}"
done
if [ -n "$CORES_WANTED" ]; then
  integer --cores "$CORES_WANTED"; [ "$CORES_WANTED" -ge 1 ] || die "--cores must be >= 1" 2
fi
for i in "${!EXTRA_PGNS[@]}"; do
  [ -s "${EXTRA_PGNS[$i]}" ] || die "--extra-pgn: '${EXTRA_PGNS[$i]}' missing or empty" 2
  EXTRA_PGNS[$i]=$(readlink -f "${EXTRA_PGNS[$i]}")
done

# ---------------- cores and concurrency -----------------------
detect_groups
if [ -n "$CPUS_FORCED" ]; then
  printf '%s' "$CPUS_FORCED" | grep -qE '^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$' \
    || die "invalid --cpus: '$CPUS_FORCED' (e.g. 4-11 or 4,6,8-11)" 2
  CHOSEN_CPUS=$(expand_list "$CPUS_FORCED")
  for c in $CHOSEN_CPUS; do
    [ -d "/sys/devices/system/cpu/cpu$c" ] || die "cpu $c does not exist" 2
  done
  CPU_SOURCE="forced with --cpus"
elif [ "${#SORTED_GROUPS[@]}" -eq 1 ]; then
  CHOSEN_CPUS=$(choose_cpus); CPU_SOURCE="homogeneous CPU, all cores"
else
  CHOSEN_CPUS=$(choose_cpus); CPU_SOURCE="hybrid CPU, automatic choice"
fi
# /sys lists cpus alphabetically (cpu10 before cpu2): sort them numerically.
mapfile -t cpu_list < <(printf '%s\n' $CHOSEN_CPUS | sort -nu)
if [ -n "$CORES_WANTED" ]; then
  [ "$CORES_WANTED" -le "${#cpu_list[@]}" ] \
    || die "--cores $CORES_WANTED: only ${#cpu_list[@]} available (cpu $(compress_list "${cpu_list[@]}"))" 2
  cpu_list=("${cpu_list[@]:0:CORES_WANTED}")
fi
CORES=${#cpu_list[@]}
[ "$THREADS" -le "$CORES" ] || die "$THREADS threads per engine do not fit in $CORES cores" 2
[ $(( CORES % THREADS )) -eq 0 ] \
  || die "threads ($THREADS) must divide cores ($CORES), otherwise cores are left unpaired" 2
CONCURRENCY=$(( CORES / THREADS ))
AFFINITY=$(compress_list "${cpu_list[@]}")

if [ "$HASH" = auto ]; then HASH=$(auto_hash "$TC" "$CONCURRENCY"); HASH_SOURCE="automatic"
else HASH_SOURCE="forced"; fi
RAM_MB=$(( 2 * CONCURRENCY * HASH ))
if [ "$THREADS" -eq 1 ]; then THREAD_WORD=thread; else THREAD_WORD=threads; fi

if [ "$SHOW_CPUS" -eq 1 ]; then
  printf '\n  %sCores%s\n' "$B$CY" "$N"
  describe_groups "${cpu_list[*]}"
  printf '\n  games on cpu %s%s%s (%s), concurrency %d with %d %s per engine\n' \
         "$B" "$AFFINITY" "$N" "$CPU_SOURCE" "$CONCURRENCY" "$THREADS" "$THREAD_WORD"
  printf '  hash %s%d MB%s per engine (%s, tc %s), %d MB in total\n\n' \
         "$B" "$HASH" "$N" "$HASH_SOURCE" "$TC" "$RAM_MB"
  exit 0
fi

# ---------------- engine under test and book ------------------
for cmd in fastchess timeout; do
  command -v "$cmd" >/dev/null 2>&1 || die "command '$cmd' not found in PATH"
done
[ "$THREADS" -eq 1 ] || command -v taskset >/dev/null 2>&1 || die "--threads > 1 needs taskset"

if [ -z "$ENGINE_BIN" ]; then
  [ -x "$ENGINE_DEFAULT" ] || die "no --engine given and $ENGINE_DEFAULT not found" 2
  ENGINE_BIN="$ENGINE_DEFAULT"
fi
[ -x "$ENGINE_BIN" ] || die "engine not executable: $ENGINE_BIN"
ENGINE_REAL=$(readlink -f "$ENGINE_BIN")
provenance "$ENGINE_BIN"

if [ -z "$BOOK" ]; then
  BOOK=$(find . -maxdepth 4 -type f \( -name '*.pgn' -o -name '*.epd' \) -path '*book*' 2>/dev/null | sort | head -1)
  [ -n "$BOOK" ] || BOOK=$(find . -maxdepth 4 -type f -name '*.pgn' -path '*opening*' 2>/dev/null | sort | head -1)
fi
[ -s "$BOOK" ] || die "opening book missing or empty: '${BOOK:-none found}' (use --book)"
case "$BOOK" in *.epd) BOOK_FORMAT=epd ;; *) BOOK_FORMAT=pgn ;; esac

# An engine that does not answer uci is a broken binary: better now than
# after a thousand games lost by disconnection.
ENGINE_NAME=$(probe_uci "$ENGINE_BIN")
[ -n "$ENGINE_NAME" ] || die "$ENGINE_BIN does not answer uci: broken or not a UCI engine"
ENGINE=${ENGINE_FIXNAME:-$(label "$ENGINE_NAME")}

# ---------------- opponents -----------------------------------
# Without any: the default calibration, or every executable ELF next to
# this script that answers uci, minus the engine under test and duplicates
# (links to the same file).
if [ "${#OPP_KIND[@]}" -eq 0 ] && [ -n "$DEFAULT_SF_LEVELS" ] && [ -x "${DEFAULT_SF_LEVELS%:*}" ]; then
  OPP_SOURCE="default calibration"
  IFS=',' read -ra levels <<<"${DEFAULT_SF_LEVELS##*:}"
  for l in "${levels[@]}"; do add_opp sf "${DEFAULT_SF_LEVELS%:*}" "$l"; done
elif [ "${#OPP_KIND[@]}" -eq 0 ]; then
  OPP_SOURCE="found in this folder"
  declare -A seen=(["$ENGINE_REAL"]=1)
  while IFS= read -r path; do
    is_elf "$path" || continue
    real=$(readlink -f "$path")
    [ -z "${seen[$real]:-}" ] || continue
    seen["$real"]=1
    [ -n "$(probe_uci "$path")" ] && add_opp path "$path"
  done < <(find . -maxdepth 1 \( -type f -o -type l \) -executable 2>/dev/null | sort -V)
  [ "${#OPP_KIND[@]}" -gt 0 ] \
    || die "no opponents: $(pwd) has no UCI engine besides $ENGINE_BIN (use --opponent)" 2
else
  OPP_SOURCE="chosen on the command line"
fi
N_OPP=${#OPP_KIND[@]}
[ "$N_OPP" -eq 1 ] || [ -z "$ANCHOR_ELO" ] \
  || die "with several opponents give anchors by name: --anchor NAME=ELO" 2

# Binaries, names and per-engine checks.
OPP_BIN=(); OPP_NAME=(); OPP=(); OPP_DESC=()
declare -A taken=(["$ENGINE"]=1) sf_range=()
for i in "${!OPP_KIND[@]}"; do
  src=${OPP_SRC[$i]}
  case ${OPP_KIND[$i]} in
    path)
      [ -x "$src" ] || die "opponent not executable: $src"
      bin=$src; desc=$src ;;
    tag)
      [ -n "$REPO" ] || die "--tag $src: $ENGINE_BIN is not in a git repository" 2
      if [ "$DRY_RUN" -eq 1 ] && [ ! -x "$REFS/chess_$(label "$src")" ]; then
        git -C "$REPO" rev-parse -q --verify "refs/tags/$src" >/dev/null \
          || die "tag $src does not exist in $REPO" 2
        bin="$REFS/chess_$(label "$src")"; desc="tag $src (not built yet)"
      else
        build_tag "$REPO" "$src"; bin=$BUILT; desc="tag $src, $BUILT"
      fi ;;
    sf)
      [ -x "$src" ] || die "Stockfish binary not executable: $src"
      if [ -z "${sf_range[$src]:-}" ]; then
        line=$(uci_output "$src" | grep -E '^option name UCI_Elo ' || true)
        [ -n "$line" ] || die "$src has no UCI_Elo option: --sf-levels needs Stockfish 11 or later" 2
        sf_range[$src]="$(grep -oE 'min [0-9]+' <<<"$line" | cut -d' ' -f2) $(grep -oE 'max [0-9]+' <<<"$line" | cut -d' ' -f2)"
      fi
      read -r lo hi <<<"${sf_range[$src]}"
      lvl=${OPP_LEVEL[$i]}
      [ "$lvl" -ge "$lo" ] && [ "$lvl" -le "$hi" ] \
        || die "Stockfish level $lvl outside the UCI_Elo range [$lo, $hi] of $src" 2
      bin=$src; desc="$src at UCI_Elo $lvl"
      OPP_OPTS[$i]="UCI_LimitStrength=true"$'\n'"UCI_Elo=$lvl"$'\n'"${OPP_OPTS[$i]}" ;;
  esac
  # The same file as the engine under test only makes sense with other options.
  if [ -x "$bin" ] && [ "$(readlink -f "$bin")" = "$ENGINE_REAL" ] \
     && [ -z "${OPP_OPTS[$i]}" ] && [ -z "$ENGINE_OPTS" ]; then
    die "$src is the same file as the engine under test (use --opt to make them differ)" 2
  fi
  if [ -x "$bin" ]; then
    name=$(probe_uci "$bin")
    [ -n "$name" ] || die "$bin does not answer uci: broken or not a UCI engine"
  else
    name="HydraY ${OPP_SRC[$i]}"
  fi
  if [ -n "${OPP_FIXNAME[$i]}" ]; then base=${OPP_FIXNAME[$i]}
  elif [ "${OPP_KIND[$i]}" = sf ]; then base="SF-${OPP_LEVEL[$i]}"
  else base=$(label "$name"); fi
  # Unique labels: fastchess and ordo tell players apart by name.
  lbl=$base; k=2
  while [ -n "${taken[$lbl]:-}" ]; do lbl="${base}_$k"; k=$(( k + 1 )); done
  taken["$lbl"]=1
  OPP_BIN+=("$bin"); OPP_NAME+=("$name"); OPP+=("$lbl"); OPP_DESC+=("$desc")
done

# ---------------- anchors -------------------------------------
# Explicit first (--anchor, then the Stockfish level), then the rough value
# of a known Stockfish only if nothing is explicit, then the first opponent
# at 3000 if nothing at all is anchored.
OPP_ANCHOR=(); OPP_ANCHOR_SOURCE=()
declare -A anchor_used=()
explicit=0
for i in "${!OPP[@]}"; do
  a=""; s=""
  for key in "${OPP[$i]}" "${OPP_SRC[$i]}" "${OPP_SRC[$i]#./}" "$(basename "${OPP_SRC[$i]}")" "${OPP_NAME[$i]}"; do
    [ "${OPP_KIND[$i]}" = sf ] && [ "$key" != "${OPP[$i]}" ] && continue
    if [ -n "${ANCHOR_FOR[$key]:-}" ]; then
      a=${ANCHOR_FOR[$key]}; s=forced; anchor_used["$key"]=1; break
    fi
  done
  if [ -z "$a" ] && [ -n "$ANCHOR_ELO" ]; then a=$ANCHOR_ELO; s=forced; fi
  if [ -z "$a" ] && [ "${OPP_KIND[$i]}" = sf ]; then a=${OPP_LEVEL[$i]}; s=UCI_Elo; fi
  [ "$a" = - ] && { a=""; s="free, by request"; }
  [ -n "$a" ] && explicit=1
  OPP_ANCHOR+=("$a"); OPP_ANCHOR_SOURCE+=("$s")
done
for k in "${!ANCHOR_FOR[@]}"; do
  [ -n "${anchor_used[$k]:-}" ] || die "--anchor $k=...: no opponent is called $k" 2
done
for i in "${!OPP[@]}"; do
  [ -z "${OPP_ANCHOR_SOURCE[$i]}" ] || continue
  g=""; [ "$explicit" -eq 1 ] || g=$(suggested_anchor "${OPP_NAME[$i]}")
  if [ -n "$g" ]; then OPP_ANCHOR[$i]=$g; OPP_ANCHOR_SOURCE[$i]="rough, from the name"
  else OPP_ANCHOR_SOURCE[$i]=free; fi
done
anchored=0
for a in "${OPP_ANCHOR[@]}"; do [ -n "$a" ] && anchored=$(( anchored + 1 )); done
if [ "$anchored" -eq 0 ]; then
  for i in "${!OPP[@]}"; do
    [ "${OPP_ANCHOR_SOURCE[$i]}" = free ] || continue
    OPP_ANCHOR[$i]=3000; OPP_ANCHOR_SOURCE[$i]="arbitrary scale, only differences count"
    anchored=1; break
  done
  [ "$anchored" -eq 1 ] || die "every opponent is free: anchor at least one with --anchor NAME=ELO" 2
fi

# The first anchored opponent also fills the single-opponent fields.
OPPONENT=${OPP[0]}; OPPONENT_BIN=${OPP_BIN[0]}; OPPONENT_NAME=${OPP_NAME[0]}
ANCHOR_ELO=""
for i in "${!OPP[@]}"; do
  [ -n "${OPP_ANCHOR[$i]}" ] && { ANCHOR_ELO=${OPP_ANCHOR[$i]}; break; }
done
if [ "$N_OPP" -gt 1 ]; then MODE=gauntlet; else MODE=match; fi

# ---------------- clock check ---------------------------------
CLOCK_NOTES=()
declare -A clocked=()
for bin in "$ENGINE_BIN" "${OPP_BIN[@]}"; do
  [ -x "$bin" ] || continue
  real=$(readlink -f "$bin")
  [ -z "${clocked[$real]:-}" ] || continue
  clocked["$real"]=1
  probe_clock "$bin"
  if [ -z "$CLOCK_MS" ]; then
    CLOCK_NOTES+=("$bin did not move within 3 s with 4 s on the clock: it would lose on time")
  elif [ "$CLOCK_MS" -gt "$CLOCK_LIMIT_MS" ]; then
    CLOCK_NOTES+=("$bin took $CLOCK_MS ms with 4 s on the clock: it seems to ignore the clock")
  fi
done

TOTAL_GAMES=$(( ROUNDS * 2 * N_OPP ))
DURATION=$(estimate_duration "$TOTAL_GAMES" "$TC" "$CONCURRENCY")
if [ "$RESIGN_TWOSIDED" = true ]; then RESIGN_TEXT=two-sided; else RESIGN_TEXT="${YE}one-sided${N}"; fi

# ---------------- summary -------------------------------------
opts_text() { # UCI options worth showing; the Stockfish level is already in the description
  local t
  t=$(printf '%s' "$1" | grep -v '^UCI_' | paste -sd' ' - || true)
  [ -z "$t" ] || printf ', %s' "$t"
}
row() { printf '  %-15s %s\n' "$1" "$2"; }
case $MODE in gauntlet) TITLE="GAUNTLET ($N_OPP opponents)" ;; *) TITLE=MATCH ;; esac
printf '\n%s  %s%s\n' "$B$CY" "$TITLE" "$N"
printf '  %s---------------------------------------------------------%s\n' "$DIM" "$N"
row "Under test"     "$B$ENGINE$N   $DIM$ENGINE_NAME$(opts_text "$ENGINE_OPTS")$N"
row ""               "$DIM$ENGINE_BIN${BRANCH:+   branch $BRANCH, commit $COMMIT}$N"
for i in "${!OPP[@]}"; do
  if [ "$i" -eq 0 ]; then
    if [ "$MODE" = gauntlet ]; then t=Opponents; else t=Opponent; fi
  else t=""; fi
  if [ -n "${OPP_ANCHOR[$i]}" ]; then
    an="anchor ${OPP_ANCHOR[$i]} (${OPP_ANCHOR_SOURCE[$i]})"
  else an="free"; fi
  row "$t" "${OPP[$i]}   ${DIM}${OPP_DESC[$i]}$(opts_text "${OPP_OPTS[$i]}"), $an$N"
done
row ""               "$DIM$OPP_SOURCE$N"
for p in "${EXTRA_PGNS[@]}"; do row "Extra pgn" "$p"; done
row "Time control"   "$B$TC$N"
if [ "$MODE" = gauntlet ]; then
  row "Rounds"       "$B$ROUNDS$N per opponent   ($TOTAL_GAMES games in total)"
else
  row "Rounds"       "$B$ROUNDS$N   ($TOTAL_GAMES games)"
fi
row "Cores"          "$B$CORES$N   cpu $AFFINITY   $DIM$CPU_SOURCE$N"
row "Concurrency"    "$CONCURRENCY   ($THREADS $THREAD_WORD per engine)"
row "Hash"           "$HASH MB per engine   $DIM$HASH_SOURCE, $RAM_MB MB in total$N"
row "Book"           "$BOOK   $DIM$PLIES plies, $ORDER$N"
row "Resign"         "600 cp for 3 moves, $RESIGN_TEXT"
row "Duration"       "at most $(hms "$DURATION")"
printf '  %s---------------------------------------------------------%s\n' "$DIM" "$N"

for note in "${PROVENANCE_NOTES[@]:-}" "${CLOCK_NOTES[@]:-}"; do [ -n "$note" ] && warn "$note"; done
FREE=$(awk '/MemAvailable/ { printf "%d", $2 / 1024 }' /proc/meminfo 2>/dev/null || echo 0)
if [ "$FREE" -gt 0 ] && [ "$RAM_MB" -gt $(( FREE * 8 / 10 )) ]; then
  warn "the hash tables need $RAM_MB MB but only $FREE are free: lower --hash or --cores"
fi
read -r TC_BASE _ <<<"$(split_tc "$TC")"
if awk -v b="$TC_BASE" 'BEGIN { exit !(b < 5) }'; then
  warn "below 5 seconds of base time, time management weighs more than playing strength"
fi
[ "$THREADS" -gt 1 ] && warn "with several threads the search is not deterministic: more games are needed"
[ "$RESIGN_TWOSIDED" = true ] || warn "one-sided resign: against an engine with another eval scale the estimate is skewed"
echo

# ---------------- fastchess command ---------------------------
# Options per engine: Hash and Threads unless --opt sets them, then --opt.
engine_args() { # $1 = name, $2 = binary, $3 = options
  local o args=(-engine cmd="$2" name="$1")
  printf '%s' "$3" | grep -q '^Hash='    || args+=(option.Hash="$HASH")
  printf '%s' "$3" | grep -q '^Threads=' || args+=(option.Threads="$THREADS")
  while IFS= read -r o; do [ -n "$o" ] && args+=("option.$o"); done <<<"$3"
  COMMAND+=("${args[@]}")
}
COMMAND=(fastchess)
engine_args "$ENGINE" "$ENGINE_BIN" "$ENGINE_OPTS"
for i in "${!OPP[@]}"; do engine_args "${OPP[$i]}" "${OPP_BIN[$i]}" "${OPP_OPTS[$i]}"; done
# Gauntlet: the first engine (seeds 1) plays everyone, the others not each other.
[ "$MODE" = gauntlet ] && COMMAND+=(-tournament gauntlet -seeds 1)
COMMAND+=(
  -each tc="$TC"
  -openings file="$BOOK" format="$BOOK_FORMAT" order="$ORDER" plies="$PLIES"
  -rounds "$ROUNDS" -repeat
  -resign movecount=3 score=600 twosided="$RESIGN_TWOSIDED"
  -draw movenumber=40 movecount=8 score=10
  -concurrency "$CONCURRENCY")

# With one thread per engine fastchess's affinity puts both engines of a
# game on the same core, and with ponder off only one thinks at a time.
# With more threads it would ignore them and squeeze them onto one core:
# use taskset on the whole process instead.
if [ "$THREADS" -eq 1 ]; then
  COMMAND+=(-use-affinity "$AFFINITY")
else
  COMMAND=(taskset -c "$AFFINITY" "${COMMAND[@]}")
fi
COMMAND+=(-recover -ratinginterval 50 -report penta=true
  -log file=fastchess.log level=warn
  -pgnout file="$PGN")

if [ "$DRY_RUN" -eq 1 ]; then
  printf '  %scommand that would run:%s\n\n' "$DIM" "$N"
  printf '%q ' "${COMMAND[@]}"; printf '\n\n'
  exit 0
fi

# ---------------- previous PGN --------------------------------
if [ -s "$PGN" ]; then
  case "$PGN_MODE" in
    delete)  rm -f "$PGN"; printf '  %s existed: deleted\n' "$PGN" ;;
    append)  warn "$PGN exists: new games are appended and the estimate includes the old ones" ;;
    archive) archived="${PGN%.pgn}_$(date +%Y%m%d_%H%M%S).pgn"
             mv "$PGN" "$archived"; printf '  %s existed: archived as %s\n' "$PGN" "$archived" ;;
  esac
fi

# ---------------- configuration for elo_watch -----------------
date +%s > "$STAMP"
cat > "$CONF" <<CONFIG
# written by runfastchess.sh on $(date '+%Y-%m-%d %H:%M:%S')
ENGINE="$ENGINE"
ENGINE_BIN="$ENGINE_BIN"
ENGINE_NAME="$ENGINE_NAME"
ENGINE_BRANCH="$BRANCH"
ENGINE_COMMIT="$COMMIT"
OPPONENT="$OPPONENT"
OPPONENT_BIN="$OPPONENT_BIN"
OPPONENT_NAME="$OPPONENT_NAME"
PGN="$PGN"
BOOK="$BOOK"
ROUNDS="$ROUNDS"
TOTAL_GAMES="$TOTAL_GAMES"
TC="$TC"
CORES="$CORES"
THREADS="$THREADS"
HASH="$HASH"
CONCURRENCY="$CONCURRENCY"
AFFINITY="$AFFINITY"
ANCHOR_ELO="$ANCHOR_ELO"
RESIGN_TWOSIDED="$RESIGN_TWOSIDED"
START_EPOCH="$(cat "$STAMP")"
MODE="$MODE"
CONFIG
conf_array() { # $1 = name, then the elements; printf alone would turn none into ''
  printf '%s=(' "$1"; shift
  [ $# -eq 0 ] || printf ' %q' "$@"
  printf ' )\n'
}
{
  conf_array OPPONENTS        "${OPP[@]}"
  conf_array OPPONENTS_BIN    "${OPP_BIN[@]}"
  conf_array OPPONENTS_NAME   "${OPP_NAME[@]}"
  conf_array OPPONENTS_ANCHOR "${OPP_ANCHOR[@]}"
  conf_array EXTRA_PGNS       "${EXTRA_PGNS[@]}"
} >> "$CONF"

printf '  %sconfiguration in %s, live dashboard: ./elo_watch.sh --loop%s\n\n' "$DIM" "$CONF" "$N"

# ---------------- games ---------------------------------------
# Ctrl+C stops fastchess (it gets the signal too) but not this script, so
# the report below still runs on the games played so far.
trap ':' INT TERM
set +e
"${COMMAND[@]}"
rc=$?
set -e
trap - INT TERM

# ---------------- report --------------------------------------
if [ "$REPORT" -eq 1 ] && [ -s "$PGN" ]; then
  echo
  ./elo_watch.sh --sims 1000 --save
fi
exit "$rc"
