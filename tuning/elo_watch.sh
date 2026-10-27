#!/usr/bin/env bash
# ==============================================================
#  elo_watch.sh - live Elo estimate dashboard (fastchess + ordo),
#                 for a match or a gauntlet
#
#  Usage: ./elo_watch.sh                  a single snapshot
#         ./elo_watch.sh --loop           refresh every 60s
#         ./elo_watch.sh --loop 30        every 30s
#         ./elo_watch.sh --pgn other.pgn  watch another file
#         ./elo_watch.sh --engine HydraY_4.0.0    force the point of view
#         watch -n 60 -c ./elo_watch.sh   (the classic way, still works)
#
#  Reads .elo_run.conf written by runfastchess.sh: engines, expected games,
#  time control, cores and anchors come from there. Without that
#  file it manages on its own: it takes the newest pgn and gets the players
#  from the game tags, so no engine is hard-coded here.
# ==============================================================
set -uo pipefail

# awk prints decimals in the current locale: under it_IT "0.6" comes back
# as "0,6" and the next step reads it as zero.
export LC_NUMERIC=C
# ordo is often built into ~/.local/bin.
export PATH="$HOME/.local/bin:$PATH"

# ---------------- defaults (used when .elo_run.conf is missing)
PGN=""
ENGINE=""
OPPONENT=""
ENGINE_NAME=""
OPPONENT_NAME=""
ENGINE_BRANCH=""
ENGINE_COMMIT=""
TOTAL_GAMES=0
TC="?"
CORES="?"
THREADS="?"
HASH="?"
CONCURRENCY="?"
START_EPOCH=""
MODE=match
OPPONENTS=()
OPPONENTS_NAME=()
OPPONENTS_ANCHOR=()
EXTRA_PGNS=()

# ---------------- constants -----------------------------------
readonly CONF=".elo_run.conf"
readonly STAMP=".elo_watch_start"
readonly ORDO_MAX_THREADS=4   # ordo goes on up to 4 cpus the games leave free
readonly WIDTH=40              # progress bar width
readonly ORDO_TIMEOUT=45       # ordo is dropped after this: with the machine
                               # busy it can take longer than the loop

SIMULATIONS=2000               # ordo simulations for the error bars

ANCHOR_ELO=3350                # opponent anchor, if nobody says otherwise
ANCHOR_FORCED=""               # the one given by hand beats the conf
declare -A ANCHOR_FOR=()       # --anchor NAME=ELO, for the gauntlet
INTERVAL=0                     # 0 = single snapshot
USE_ORDO=1
PGN_FORCED=""
ENGINE_FORCED=""
EXTRA_FORCED=()                # --extra-pgn
SAVE=0                         # --save: keep ratings, head to head, anchors

# ---------------- looks ---------------------------------------
B=$'\e[1m'; DIM=$'\e[2m'; GR=$'\e[32m'; YE=$'\e[33m'
RE=$'\e[31m'; CY=$'\e[36m'; N=$'\e[0m'

case "${LC_ALL:-${LC_CTYPE:-${LANG:-}}}" in
  *UTF-8*|*utf8*|*UTF8*)
    FULL=$'█'; EMPTY=$'░'
    PARTIAL=($'▏' $'▎' $'▍' $'▌' $'▋' $'▊' $'▉')
    ;;
  *)
    FULL='#'; EMPTY='.'
    PARTIAL=('-' '-' '-' '=' '=' '=' '=')
    ;;
esac

# ---------------- arguments -----------------------------------
usage() {
  cat <<HELP
  elo_watch.sh - live Elo estimate dashboard

    --loop [SEC]   refresh every SEC seconds (default 60)
    --pgn FILE     watch this pgn instead of the one in $CONF
    --engine NAME  treat NAME as the engine under test, the others are opponents
    --anchor N     Elo given to the opponent in ordo. Without this flag the
                   one chosen at launch and written to $CONF is used,
                   otherwise $ANCHOR_ELO
    --anchor NAME=ELO
                   anchor for one player (name as in the pgn); repeatable.
                   NAME=- leaves that player free
    --extra-pgn FILE
                   more games for ordo, on top of those in $CONF; repeatable
    --save         write the ordo ratings, head to head and anchors next to
                   the pgn (_ratings.txt, _h2h.txt, _anchors.txt)
    --sims N       ordo simulations for the error bars (default $SIMULATIONS)
    --no-ordo      skip the ordo table, show only the built-in statistics
    --help         this message

  Without flags it prints once and exits, so it still works with
  'watch -n 60 -c ./elo_watch.sh'. Reads $CONF if present.
HELP
}

need() { [ "$2" -ge 2 ] || { printf '%s  %s needs a value%s\n' "$RE" "$1" "$N" >&2; exit 2; }; }

while [ $# -gt 0 ]; do
  case "$1" in
    --loop|-l)
      INTERVAL=60
      case "${2:-}" in
        ''|-*)    shift ;;
        *[!0-9]*) shift ;;
        *)        INTERVAL="$2"; shift 2 ;;
      esac
      ;;
    --pgn)     need "$1" $#; PGN_FORCED="$2"; shift 2 ;;
    --engine)  need "$1" $#; ENGINE_FORCED="$2"; shift 2 ;;
    --anchor)  need "$1" $#
               if [[ $2 == *=* ]]; then ANCHOR_FOR["${2%=*}"]="${2##*=}"
               else ANCHOR_FORCED="$2"; fi
               shift 2 ;;
    --sims)    need "$1" $#; SIMULATIONS="$2"; shift 2 ;;
    --extra-pgn) need "$1" $#; EXTRA_FORCED+=("$2"); shift 2 ;;
    --save)    SAVE=1; shift ;;
    --no-ordo) USE_ORDO=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf '%s  unknown argument: %s (try --help)%s\n' "$RE" "$1" "$N" >&2; exit 2 ;;
  esac
done
[ "$INTERVAL" -gt 0 ] 2>/dev/null || INTERVAL=0

cd "$(dirname "$0")"

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

# ---------------- configuration -------------------------------
CONF_FOUND=0
if [ -r "$CONF" ]; then
  # shellcheck disable=SC1090
  . "./$CONF" && CONF_FOUND=1
fi

# ordo runs on the first cpus the games do not use (all of them if the
# games use every cpu or the conf does not say).
ORDO_CPUS=""
for c in $(expand_list "$(cat /sys/devices/system/cpu/online 2>/dev/null || echo 0)"); do
  [[ " $(expand_list "${AFFINITY:-}") " == *" $c "* ]] && continue
  ORDO_CPUS+="$c,"
done
ORDO_CPUS=$(printf '%s' "${ORDO_CPUS%,}" | cut -d, -f1-"$ORDO_MAX_THREADS")
[ -n "$ORDO_CPUS" ] || ORDO_CPUS=$(cat /sys/devices/system/cpu/online 2>/dev/null || echo 0)
ORDO_THREADS=$(expand_list "$ORDO_CPUS" | wc -w)
[ "$ORDO_THREADS" -le "$ORDO_MAX_THREADS" ] || ORDO_THREADS=$ORDO_MAX_THREADS

# Anchor precedence: flag, then conf, then default.
[ -n "$ANCHOR_FORCED" ] && ANCHOR_ELO="$ANCHOR_FORCED"

# Scratch files for ordo, gone on exit.
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# A pgn other than the conf's one is another match: the conf fields would
# describe something else, so start again from the pgn.
if [ -n "$PGN_FORCED" ]; then
  if [ "$(readlink -f "$PGN_FORCED" 2>/dev/null)" != "$(readlink -f "${PGN:-}" 2>/dev/null)" ]; then
    CONF_FOUND=0
    ENGINE=""; OPPONENT=""; ENGINE_NAME=""; OPPONENT_NAME=""
    ENGINE_BRANCH=""; ENGINE_COMMIT=""; TOTAL_GAMES=0; START_EPOCH=""
    TC="?"; CORES="?"; THREADS="?"; HASH="?"; CONCURRENCY="?"
    MODE=match
    OPPONENTS=(); OPPONENTS_NAME=(); OPPONENTS_ANCHOR=(); EXTRA_PGNS=()
    [ -n "$ANCHOR_FORCED" ] || ANCHOR_ELO=3350
  fi
  PGN="$PGN_FORCED"
fi

for f in "${EXTRA_FORCED[@]}"; do
  [ -s "$f" ] || { printf '%s  --extra-pgn: %s missing or empty%s\n' "$RE" "$f" "$N" >&2; exit 2; }
  EXTRA_PGNS+=("$f")
done

# Without a conf, or with a pgn given by hand, the newest file is the
# right choice nine times out of ten.
if [ -z "$PGN" ] || [ ! -e "$PGN" ]; then
  PGN=$(ls -1t ./*.pgn 2>/dev/null | head -1)
fi

# The players come from the tags of the first game, so the dashboard does
# not need to know in advance who is playing.
if [ -n "${PGN:-}" ] && [ -s "$PGN" ]; then
  read -r P_WHITE P_BLACK E_WHITE E_BLACK TC_PGN START_PGN <<<"$(awk -F'"' '
    /^\[White "/           && !w  { w  = $2 }
    /^\[Black "/           && !b  { b  = $2 }
    /^\[EngineWhiteName "/ && !ew { ew = $2 }
    /^\[EngineBlackName "/ && !eb { eb = $2 }
    /^\[TimeControl "/     && !tc { tc = $2 }
    /^\[GameStartTime "/   && !gs { gs = $2 }
    (w && b && ew && eb && tc && gs) { exit }
    END { gsub(/ /, "_", ew); gsub(/ /, "_", eb); gsub(/ /, "_", gs)
          printf "%s %s %s %s %s %s", (w?w:"?"), (b?b:"?"), (ew?ew:"-"), (eb?eb:"-"),
                                      (tc?tc:"?"), (gs?gs:"-") }' "$PGN")"

  if [ -n "$ENGINE_FORCED" ]; then
    ENGINE="$ENGINE_FORCED"
    OPPONENT=$([ "$ENGINE_FORCED" = "$P_WHITE" ] && echo "$P_BLACK" || echo "$P_WHITE")
  elif [ "$CONF_FOUND" -eq 0 ]; then
    # The engine under test is the one with the most games: in a match they
    # tie and the first white wins, in a gauntlet it is the seed.
    ENGINE=$(awk -F'"' '/^\[(White|Black) "/ { n[$2]++; if (!($2 in o)) o[$2] = k++ }
      END { for (g in n) if (n[g] > m || (n[g] == m && o[g] < o[best])) { m = n[g]; best = g }
            print best }' "$PGN")
    OPPONENT=$([ "$ENGINE" = "$P_WHITE" ] && echo "$P_BLACK" || echo "$P_WHITE")
  fi
  # Opponents from the pgn when the conf does not list them: everyone who
  # played the engine under test, in order of appearance.
  if [ "${#OPPONENTS[@]}" -eq 0 ] || [ -n "$ENGINE_FORCED" ]; then
    mapfile -t OPPONENTS < <(awk -F'"' -v me="$ENGINE" '
      /^\[White "/ { w = $2 } /^\[Black "/ { b = $2
        o = (w == me) ? b : ((b == me) ? w : "")
        if (o != "" && !(o in seen)) { seen[o] = 1; print o } }' "$PGN")
    OPPONENTS_NAME=(); OPPONENTS_ANCHOR=()
  fi
  [ "${#OPPONENTS[@]}" -gt 1 ] && MODE=gauntlet
  # Long names with the version, for the header. If the player is not in
  # the pgn leave it empty: nothing beats the other player's name.
  long_name() { # $1 = player name
    case "$1" in
      "$P_WHITE") printf '%s' "${E_WHITE//_/ }" ;;
      "$P_BLACK") printf '%s' "${E_BLACK//_/ }" ;;
    esac
  }
  [ -n "$ENGINE_NAME" ]   || ENGINE_NAME=$(long_name "$ENGINE")
  [ -n "$OPPONENT_NAME" ] || OPPONENT_NAME=$(long_name "$OPPONENT")
  [ "$TC" = "?" ] && TC="$TC_PGN"

  # Without a conf the start is in the first game: an archived pgn must be
  # read with its own times, not with the time you open it.
  if [ -z "$START_EPOCH" ] && [ "$START_PGN" != "-" ]; then
    START_EPOCH=$(date -d "${START_PGN//_/ }" +%s 2>/dev/null || true)
  fi
fi

# ---------------- functions -----------------------------------
hms() { printf '%02d:%02d:%02d' $(( $1 / 3600 )) $(( $1 % 3600 / 60 )) $(( $1 % 60 )); }

bar() { # $1 = done, $2 = total -- one-eighth-of-a-character resolution
  local done_="$1" tot="$2"
  local eighths full rest empty fill out=''
  [ "$tot" -gt 0 ] 2>/dev/null || tot=1
  eighths=$(( done_ * WIDTH * 8 / tot ))
  [ "$eighths" -lt 0 ] && eighths=0
  [ "$eighths" -gt $(( WIDTH * 8 )) ] && eighths=$(( WIDTH * 8 ))
  full=$(( eighths / 8 ))
  rest=$(( eighths % 8 ))

  if [ "$full" -gt 0 ]; then
    printf -v fill '%*s' "$full" ''
    out="${fill// /$FULL}"
  fi
  if [ "$rest" -gt 0 ] && [ "$full" -lt "$WIDTH" ]; then
    out+="${PARTIAL[$(( rest - 1 ))]}"
    empty=$(( WIDTH - full - 1 ))
  else
    empty=$(( WIDTH - full ))
  fi
  if [ "$empty" -gt 0 ]; then
    printf -v fill '%*s' "$empty" ''
    out+="${fill// /$EMPTY}"
  fi
  printf '[%s]' "$out"
}

# A single pass over the pgn: overall score, score by colour, how games
# ended. On a 15 MB pgn it matters.
scan() { # $1 = pgn, $2 = name of the engine under test
  awk -F'"' -v me="$2" '
    /^\[White "/       { w = $2 }
    /^\[Termination "/ { t = $2 }
    /^\[Result "/ {
        r = $2
        if (r == "*") next
        n++
        my_white = (w == me)
        if (r == "1/2-1/2")                     { d++; if (my_white) dw++; else db++ }
        else if ((r == "1-0") == my_white)      { v++; if (my_white) vw++; else vb++ }
        else                                    { l++; if (my_white) lw++; else lb++ }
        if (t == "time forfeit")      tf++
        else if (t == "adjudication") ad++
        else if (t == "normal")       nm++
        else if (t != "")             ot++
        t = ""
    }
    END { printf "%d %d %d %d %d %d %d %d %d %d %d %d %d %d",
                 n+0, v+0, d+0, l+0, vw+0, dw+0, lw+0, vb+0, db+0, lb+0,
                 tf+0, ad+0, nm+0, ot+0 }
  ' "$1"
}

# Score per opponent: one line "name games wins draws losses".
# fastchess player names have no spaces.
per_opponent() { # $1 = pgn, $2 = name of the engine under test
  awk -F'"' -v me="$2" '
    /^\[White "/ { w = $2 } /^\[Black "/ { b = $2 }
    /^\[Result "/ {
        r = $2
        if (r == "*" || (w != me && b != me)) next
        o = (w == me) ? b : w
        n[o]++
        if (r == "1/2-1/2") d[o]++
        else if ((r == "1-0") == (w == me)) v[o]++
        else l[o]++
    }
    END { for (o in n) printf "%s %d %d %d %d\n", o, n[o], v[o]+0, d[o]+0, l[o]+0 }
  ' "$1"
}

# Elo from the score, LOS and 95% interval.
# erf from Abramowitz-Stegun 7.1.26, awk has no native one.
stats() { # $1 = wins, $2 = draws, $3 = losses
  awk -v v="$1" -v d="$2" -v l="$3" '
    function erf(x,   t, y, sg) {
      sg = (x < 0) ? -1 : 1
      if (x < 0) x = -x
      t = 1 / (1 + 0.3275911 * x)
      y = 1 - ((((( 1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * exp(-x * x)
      return sg * y
    }
    function elo(sc,   e) {
      if (sc <= 0) return -800
      if (sc >= 1) return  800
      e = -400 * log(1 / sc - 1) / log(10)
      return (e > -0.5 && e < 0.5) ? 0 : e     # no "-0" on the dashboard
    }
    BEGIN {
      n = v + d + l
      if (n <= 0) { print "0.0 --- --- --- ---"; exit }
      sc = (v + 0.5 * d) / n
      # sample variance of the per-game score
      var = (v * (1 - sc)^2 + d * (0.5 - sc)^2 + l * sc^2) / n
      se  = sqrt(var / n)
      printf "%.1f ", 100 * sc
      if (sc <= 0)      printf "-inf "
      else if (sc >= 1) printf "+inf "
      else              printf "%+.0f ", elo(sc)
      lo = sc - 1.96 * se; hi = sc + 1.96 * se
      if (lo <= 0 || hi >= 1) printf "--- --- "
      else                    printf "%+.0f %+.0f ", elo(lo), elo(hi)
      dec = v + l
      if (dec <= 0) printf "---\n"
      else printf "%.1f\n", 100 * 0.5 * (1 + erf((v - l) / sqrt(2 * dec)))
    }'
}

# Elo of every anchored player: --anchor NAME=ELO, then --anchor N for the
# opponent, then the conf; the default goes to the first opponent only when
# nothing else is anchored.
declare -A ANCHOR_OF=()
build_anchors() {
  local i a elo
  ANCHOR_OF=()
  for i in "${!OPPONENTS[@]}"; do
    a=${OPPONENTS[$i]}
    if [ -n "${ANCHOR_FOR[$a]:-}" ]; then elo=${ANCHOR_FOR[$a]}
    elif [ -n "$ANCHOR_FORCED" ] && [ "$a" = "$OPPONENT" ]; then elo=$ANCHOR_FORCED
    elif [ "${#OPPONENTS_ANCHOR[@]}" -gt 0 ]; then elo=${OPPONENTS_ANCHOR[$i]:-}
    else elo=""; fi
    [ "$elo" = - ] && elo=""
    [ -n "$elo" ] && ANCHOR_OF["$a"]=$elo
  done
  # players that are not opponents, e.g. from an extra pgn
  for a in "${!ANCHOR_FOR[@]}"; do
    [ -n "${ANCHOR_OF[$a]:-}" ] || [ "${ANCHOR_FOR[$a]}" = - ] || ANCHOR_OF["$a"]=${ANCHOR_FOR[$a]}
  done
  if [ "${#ANCHOR_OF[@]}" -eq 0 ] && [ -n "$OPPONENT" ] && [ "${ANCHOR_FOR[$OPPONENT]:-}" != - ]; then
    ANCHOR_OF["$OPPONENT"]=$ANCHOR_ELO
  fi
}

# Anchor plus our Elo against that opponent alone, empty if either is missing.
implied() { # $1 = anchor, $2 = Elo like +120
  [ -n "$1" ] || return 0
  case "$2" in *inf*|---|'') return 0 ;; esac
  awk -v a="$1" -v e="$2" 'BEGIN { printf "%.0f", a + e }'
}

# ---------------- one dashboard frame -------------------------
draw() {
  build_anchors
  local TOT WIN DRW LOS_ WW DW LW WB DB LB TF AD NM OT
  local PCT ELO_NOW ELO_LO ELO_HI LOS
  local ELAPSED REMAINING ETA RATE PCT_TO COL_TO OUT END_EPOCH LAST

  if [ "$MODE" = gauntlet ]; then
    printf '%s  GAUNTLET - %s vs %s%s\n' "$B$CY" "${ENGINE:-?}" "$(IFS=,; echo "${OPPONENTS[*]}" | sed 's/,/, /g')" "$N"
  else
    printf '%s  ELO ESTIMATE - %s vs %s%s\n' "$B$CY" "${ENGINE:-?}" "${OPPONENT:-?}" "$N"
  fi
  local RUNNING f
  if pgrep -x fastchess >/dev/null 2>&1; then RUNNING="${GR}fastchess running${N}"
  else RUNNING="${DIM}fastchess not running${N}"; fi
  printf '%s  %s   file: %s%s   %s\n' "$DIM" "$(date '+%Y-%m-%d %H:%M:%S')" "${PGN:-none}" "$N" "$RUNNING"
  for f in "${EXTRA_PGNS[@]}"; do [ -n "$f" ] && printf '%s  extra pgn for ordo: %s%s\n' "$DIM" "$f" "$N"; done
  if [ "$MODE" = match ] && { [ -n "$ENGINE_NAME" ] || [ -n "$OPPONENT_NAME" ]; }; then
    printf '%s  %s  against  %s%s\n' "$DIM" "${ENGINE_NAME:-?}" "${OPPONENT_NAME:-?}" "$N"
  fi
  if [ -n "$ENGINE_BRANCH" ]; then
    printf '%s  binary under test: branch %s, commit %s%s\n' "$DIM" "$ENGINE_BRANCH" "$ENGINE_COMMIT" "$N"
  fi
  if [ "$CONF_FOUND" -eq 1 ]; then
    printf '%s  tc %s   %s cores, %s per engine, concurrency %s, hash %s MB%s\n' \
           "$DIM" "$TC" "$CORES" "$THREADS thread$([ "$THREADS" = 1 ] || echo s)" "$CONCURRENCY" "$HASH" "$N"
  else
    printf '%s  %s not found: engines, tc and total taken from the pgn%s\n' "$DIM" "$CONF" "$N"
    [ "$TC" != "?" ] && printf '%s  tc %s%s\n' "$DIM" "$TC" "$N"
  fi
  echo

  if [ -z "${PGN:-}" ] || [ ! -s "$PGN" ]; then
    printf '  %sWaiting for the first games...%s\n' "$YE" "$N"
    return 0
  fi

  read -r TOT WIN DRW LOS_ WW DW LW WB DB LB TF AD NM OT \
       <<<"$(scan "$PGN" "$ENGINE")"

  if [ "${TOT:-0}" -eq 0 ]; then
    printf '  %sNo finished games yet.%s\n' "$YE" "$N"
    return 0
  fi

  read -r PCT ELO_NOW ELO_LO ELO_HI LOS <<<"$(stats "$WIN" "$DRW" "$LOS_")"

  # ---------------- times -------------------------------------
  if [ -n "$START_EPOCH" ]; then
    END_EPOCH=$(date +%s)
    # if the pgn has not grown for a minute the match is over: the clock
    # stops there, otherwise the rate would drop at every refresh
    LAST=$(stat -c %Y "$PGN" 2>/dev/null || echo "$END_EPOCH")
    [ $(( END_EPOCH - LAST )) -gt 60 ] && END_EPOCH="$LAST"
    ELAPSED=$(( END_EPOCH - START_EPOCH ))
  else
    [ -f "$STAMP" ] || date +%s > "$STAMP"
    ELAPSED=$(( $(date +%s) - $(cat "$STAMP") ))
  fi
  [ "$ELAPSED" -lt 1 ] && ELAPSED=1

  if [ "${TOTAL_GAMES:-0}" -gt "$TOT" ]; then
    REMAINING=$(( ELAPSED * (TOTAL_GAMES - TOT) / TOT ))
    ETA="$(hms "$REMAINING")"
  elif [ "${TOTAL_GAMES:-0}" -eq 0 ]; then
    ETA="?"
  else
    ETA="finished"
  fi
  RATE=$(awk -v t="$TOT" -v e="$ELAPSED" 'BEGIN { printf "%.0f", t * 3600 / e }')

  # ---------------- summary -----------------------------------
  if [ "${TOTAL_GAMES:-0}" -gt 0 ]; then
    printf '  Progress     %s %s%d/%d%s  (%d%%)\n' \
           "$(bar "$TOT" "$TOTAL_GAMES")" "$B" "$TOT" "$TOTAL_GAMES" "$N" \
           $(( TOT * 100 / TOTAL_GAMES ))
  else
    printf '  Progress     %s%d games%s  (expected total unknown)\n' "$B" "$TOT" "$N"
  fi
  printf '  Time         elapsed %s   remaining %s   %s%s games/hour%s\n' \
         "$(hms "$ELAPSED")" "$ETA" "$DIM" "$RATE" "$N"
  printf '  Score        %s+%d%s  %s=%d%s  %s-%d%s   ->  %s%s%%%s\n' \
         "$GR" "$WIN" "$N" "$YE" "$DRW" "$N" "$RE" "$LOS_" "$N" "$B" "$PCT" "$N"
  printf '  By colour    %swhite%s +%d =%d -%d   %sblack%s +%d =%d -%d\n' \
         "$DIM" "$N" "$WW" "$DW" "$LW" "$DIM" "$N" "$WB" "$DB" "$LB"
  if [ "$MODE" = gauntlet ]; then
    printf '  Quick Elo    %s%s%s  (95%%: %s .. %s)   %sagainst the whole field, opponents mixed%s\n' \
           "$B" "$ELO_NOW" "$N" "$ELO_LO" "$ELO_HI" "$DIM" "$N"
  else
    local IMP; IMP=$(implied "${ANCHOR_OF[$OPPONENT]:-}" "$ELO_NOW")
    printf '  Quick Elo    %s%s%s  (95%%: %s .. %s)   %sLOS %s%%%s%s\n' \
           "$B" "$ELO_NOW" "$N" "$ELO_LO" "$ELO_HI" "$DIM" "$LOS" "$N" \
           "${IMP:+   implied $B$IMP$N $DIM($OPPONENT = ${ANCHOR_OF[$OPPONENT]:-})$N}"
  fi
  printf '  Game ends    %sadjudicated %d, on the board %d, on time %d%s\n\n' \
         "$DIM" "$AD" "$NM" "$TF" "$N"

  # ---------------- per opponent ------------------------------
  if [ "$MODE" = gauntlet ]; then
    local -A opp_row=()
    local name n v d l pc e lo hi los a imp
    while read -r name n v d l; do
      [ -n "$name" ] && opp_row["$name"]="$n $v $d $l"
    done < <(per_opponent "$PGN" "$ENGINE")
    printf '  %sPer opponent%s\n' "$B" "$N"
    for a in "${OPPONENTS[@]}"; do
      if [ -z "${opp_row[$a]:-}" ]; then
        printf '    %-24s %sno games yet%s\n' "$a" "$DIM" "$N"; continue
      fi
      read -r n v d l <<<"${opp_row[$a]}"
      read -r pc e lo hi los <<<"$(stats "$v" "$d" "$l")"
      imp=$(implied "${ANCHOR_OF[$a]:-}" "$e")
      printf '    %-24s %4d  %s+%d%s %s=%d%s %s-%d%s  %5s%%  %sElo %5s%s  (%s .. %s)  %sLOS %s%%%s%s\n' \
             "$a" "$n" "$GR" "$v" "$N" "$YE" "$d" "$N" "$RE" "$l" "$N" "$pc" \
             "$B" "$e" "$N" "$lo" "$hi" "$DIM" "$los" "$N" \
             "${imp:+   implied $B$imp$N $DIM(anchor ${ANCHOR_OF[$a]})$N}"
    done
    printf '  %simplied = anchor + our Elo against that opponent alone: if they differ\n' "$DIM"
    printf '  much between opponents, the anchors are not consistent with each other%s\n\n' "$N"
  fi

  # ---------------- time losses -------------------------------
  if [ "${TF:-0}" -gt 0 ]; then
    # percentage and severity in the same awk: sending back an already
    # formatted number would have it re-read with the Italian decimal comma
    read -r PCT_TO COL_TO <<<"$(awk -v t="$TF" -v n="$TOT" 'BEGIN {
        p = 100 * t / n
        printf "%.2f %s", p, (p < 0.5 ? "g" : (p <= 1 ? "y" : "r"))
      }')"
    case "$COL_TO" in
      g) printf '  %sLost on time: %d (%s%%) - normal%s\n\n' "$GR" "$TF" "$PCT_TO" "$N" ;;
      y) printf '  %sLost on time: %d (%s%%) - keep an eye on time management%s\n\n' \
                "$YE" "$TF" "$PCT_TO" "$N" ;;
      r) printf '  %sWARNING: %d games lost on time (%s%%)%s\n' "$RE" "$TF" "$PCT_TO" "$N"
         printf '  %sAbove 1%% the estimate is not reliable: check time management%s\n\n' "$DIM" "$N" ;;
    esac
  fi

  # ---------------- ordo table --------------------------------
  [ "$USE_ORDO" -eq 1 ] || return 0
  if ! command -v ordo >/dev/null 2>&1; then
    printf '  %sordo is not installed: showing only the built-in statistics.%s\n' "$DIM" "$N"
    printf '  %sInstall it to get Elo with error bars and CFS.%s\n' "$DIM" "$N"
    return 0
  fi

  # Only anchors that already have games: ordo -m refuses a name that is not
  # in the input. -G lets ordo go on when a player with all wins or all
  # losses splits the games into groups; that player is left out. And ordo 1.2.6 does not pin a lone -m anchor (it came out
  # at 2685 instead of 3250), so a single anchor goes through -a/-A.
  local inputs=("$PGN") live=() waiting=() anchor_text="" i
  for f in "${EXTRA_PGNS[@]}"; do [ -n "$f" ] && inputs+=("$f"); done
  for a in "${!ANCHOR_OF[@]}"; do
    if grep -qF -e "[White \"$a\"]" -e "[Black \"$a\"]" "${inputs[@]}" 2>/dev/null
    then live+=("$a"); else waiting+=("$a"); fi
  done
  if [ "${#live[@]}" -eq 0 ]; then
    printf '  %sRanking (ordo): no anchored player has a finished game yet%s\n' "$DIM" "$N"
    return 0
  fi
  mapfile -t live < <(printf '%s\n' "${live[@]}" | sort)

  local base="$WORK/ordo" ratings anchors h2h=() ordo_anchors=()
  [ "$SAVE" -eq 1 ] && base="${PGN%.pgn}"
  ratings="${base}_ratings.txt"; anchors="${base}_anchors.txt"
  [ "$SAVE" -eq 1 ] && h2h=(-j "${base}_h2h.txt")
  : > "$anchors"
  for a in "${live[@]}"; do
    printf '"%s",%s\n' "$a" "${ANCHOR_OF[$a]}" >> "$anchors"
    anchor_text+="$a = ${ANCHOR_OF[$a]}, "
  done
  if [ "${#live[@]}" -eq 1 ]; then ordo_anchors=(-a "${ANCHOR_OF[${live[0]}]}" -A "${live[0]}")
  else ordo_anchors=(-m "$anchors"); fi

  printf '  %sRanking (ordo, anchors: %s)%s\n' "$B" "${anchor_text%, }" "$N"
  [ "${#waiting[@]}" -eq 0 ] \
    || printf '  %sanchors with no games yet, left out: %s%s\n' "$DIM" "${waiting[*]}" "$N"
  printf '  %s-------------------------------------------------------------%s\n' "$DIM" "$N"

  local limit=$ORDO_TIMEOUT
  [ "$SAVE" -eq 1 ] && limit=$(( ORDO_TIMEOUT * 6 ))
  if [ "${#inputs[@]}" -gt 1 ]; then
    OUT=$(timeout "$limit" taskset -c "$ORDO_CPUS" ordo -q -G "${ordo_anchors[@]}" -W -D \
            -s "$SIMULATIONS" -n "$ORDO_THREADS" -J -o "$ratings" "${h2h[@]}" \
            -p <(cat "${inputs[@]}") 2>&1)
  else
    OUT=$(timeout "$limit" taskset -c "$ORDO_CPUS" ordo -q -G "${ordo_anchors[@]}" -W -D \
            -s "$SIMULATIONS" -n "$ORDO_THREADS" -J -o "$ratings" "${h2h[@]}" \
            -p "$PGN" 2>&1)
  fi
  local rc=$?
  if [ "$rc" -ne 0 ] || [ ! -s "$ratings" ]; then
    if [ "$rc" -eq 124 ]; then
      printf '  %sordo took more than %d seconds and was dropped.%s\n' "$RE" "$limit" "$N"
      printf '  %sWith the machine busy lower the simulations: --sims 200%s\n' "$DIM" "$N"
    else
      printf '  %sordo does not have enough data yet:%s\n' "$RE" "$N"
      printf '%s\n' "$OUT" | sed 's/^/    /'
    fi
    return 0
  fi
  sed 's/^/  /' "$ratings"

  # Every anchor must come out exactly at its value, or the scale is off.
  local bad="" dropped="" got
  for a in "${live[@]}"; do
    got=$(awk -v n="$a" '$2 == n && $3 == ":" { print $4; exit }' "$ratings")
    [ -n "$got" ] || { dropped+="$a, "; continue; }
    if awk -v g="$got" -v e="${ANCHOR_OF[$a]}" 'BEGIN { exit !(g - e > 0.5 || e - g > 0.5) }'
    then bad+="$a at ${got:-?} instead of ${ANCHOR_OF[$a]}, "; fi
  done
  if [ -n "$bad" ]; then
    printf '  %sWARNING: anchor not pinned: %s: do not trust these ratings%s\n' "$RE" "${bad%, }" "$N"
  else
    printf '  %sanchors pinned at their values: ok%s\n' "$GR" "$N"
  fi
  [ -z "$dropped" ] || printf '  %sleft out by ordo (all wins or all losses so far): %s%s\n' \
                              "$DIM" "${dropped%, }" "$N"

  echo
  printf '  %sRATING = estimated Elo | ERROR = +/- margin at 95%% | POINTS = points\n' "$DIM"
  printf '  PLAYED = games | (%%) = score | CFS = confidence of superiority\n'
  if [ "${#live[@]}" -gt 1 ]; then
    printf '  The anchors are pinned together: their difference enters the estimate.%s\n' "$N"
  else
    printf '  The anchor sets the scale: differences count, not the number.%s\n' "$N"
  fi
  if [ "$SAVE" -eq 1 ]; then
    printf '\n  saved: %s, %s, %s\n' "$ratings" "${base}_h2h.txt" "$anchors"
  fi
}

# ---------------- run -----------------------------------------
if [ "$INTERVAL" -eq 0 ]; then
  draw
  exit 0
fi

trap 'printf "\e[?25h\n"; exit 0' INT TERM
printf '\e[?25l'
while true; do
  printf '\e[H\e[2J'
  draw
  printf '\n  %srefresh every %ds - Ctrl+C to quit%s\n' "$DIM" "$INTERVAL" "$N"
  sleep "$INTERVAL"
done
