#!/usr/bin/env python3
"""Live plot of a fastchess match or gauntlet from its PGN. Redraws every 60 s.

Match (two players): Elo of the engine under test per game with its 95%
band, the pentanomial, a summary with LOS, and a thin LLR strip for SPRTs.
Gauntlet (one engine against several): every engine on one rating scale,
the engine under test included, fixed by an anchor (--anchor NAME=ELO, by
default the first opponent at 0), with 95% bands, a ladder and a table.

Games are paired by Round and players (fastchess -repeat plays both colours
of an opening under one round). Paired games use the pentanomial statistics
of fastchess with model=normalized, so Elo, nElo and LLR match what it
prints; unpaired games fall back to win/draw/loss.

SPRT bounds and the expected number of games come from the fastchess config
saved next to the pgn (-config outname=...) or from .elo_run.conf; the LLR
shows when that config runs an SPRT or the pgn is named sprt_*.

Usage:  match_watch.py [file.pgn] [--engine NAME] [--anchor NAME=ELO]
                      [--llr | --no-llr] [--elo0 0 --elo1 5 --alpha 0.05 --beta 0.05]
                      [--total N] [--every 60] [--save out.png]
Without a pgn it takes the newest one in the current directory.
"""
import argparse
import json
import math
import os
import re
import signal
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
from matplotlib.patches import FancyBboxPatch

SURFACE, INK, MUTED, FAINT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#ecebe7"
GOOD, BAD = "#008300", "#e34948"
# fixed categorical order: opponent i always gets colour i
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300", "#e34948")
NELO_PER_T = 800 / math.log(10)
PAIR_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)
TAG_RE = re.compile(rb'^\[(Event|Round|White|Black|Result|Termination|GameStartTime|GameEndTime) '
                    rb'"([^"]*)"\]', re.M)


# ---------------------------------------------------------------- reading

def read_games(pgn):
    """Header tags of every finished game, in file order."""
    games = []
    for tag, value in TAG_RE.findall(pgn.read_bytes()):
        if tag == b"Event":
            games.append({})
        elif games:
            games[-1][tag.decode()] = value.decode()
    return [g for g in games
            if {"Round", "White", "Black", "Result"} <= g.keys() and g["Result"] != "*"]


def points(game, me):
    if game["Result"] == "1/2-1/2":
        return 0.5
    return 1.0 if (game["Result"] == "1-0") == (game["White"] == me) else 0.0


def engine_under_test(games, forced):
    if forced:
        return forced
    count = Counter(p for g in games for p in (g["White"], g["Black"]))
    players = list(dict.fromkeys(p for g in games for p in (g["White"], g["Black"])))
    if len(players) == 2:
        # the baseline or the reference engine is never the one under test
        reference = re.compile(r"base|stockfish|^sf[-_\d]|^sf$", re.I)
        return next((p for p in players if not reference.search(p)), players[0])
    return max(players, key=lambda p: (count[p], -players.index(p)))


# ---------------------------------------------------------------- statistics

class Tally:
    """Running score of one engine against one opponent."""

    def __init__(self):
        self.penta = [0] * 5
        self.wdl = [0, 0, 0]          # wins, draws, losses
        self.waiting = {}             # first game of a pair, by round
        self.singles = []             # points of games left unpaired

    def add(self, game, me):
        p = points(game, me)
        self.wdl[0 if p == 1 else 1 if p == 0.5 else 2] += 1
        first = self.waiting.pop(game["Round"], None)
        if first is None:
            self.waiting[game["Round"]] = p
        else:
            self.penta[int(round((first + p) * 2))] += 1

    @property
    def games(self):
        return sum(self.wdl)

    def summary(self):
        """Elo, its 95% interval, nElo, normalized t and LOS; None if too few games."""
        pairs = sum(self.penta)
        if pairs >= 2 and 2 * pairs >= 0.9 * self.games:
            n, scores, counts = pairs, PAIR_SCORES, self.penta
            per_game_var = 2          # var of a pair mean is half that of a game
        elif self.games >= 2:
            n, scores, counts = self.games, (1.0, 0.5, 0.0), self.wdl
            per_game_var = 1
        else:
            return None
        mu = sum(s * c for s, c in zip(scores, counts)) / n
        var = sum(c * (s - mu) ** 2 for s, c in zip(scores, counts)) / n
        se = math.sqrt(var / n)
        sigma = math.sqrt(per_game_var * var)
        nt = (mu - 0.5) / sigma if sigma > 0 else 0.0
        los = 0.5 * (1 + math.erf((mu - 0.5) / (se * math.sqrt(2)))) if se > 0 else 0.5
        return {"elo": elo(mu), "lo": elo(mu - 1.96 * se), "hi": elo(mu + 1.96 * se),
                "nelo": nt * NELO_PER_T, "nt": nt, "los": los, "pairs": pairs,
                "score": mu}


def elo(score):
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / score - 1)


def llr(pairs, nt, nelo0, nelo1):
    """Fishtest's normalized-Elo GSPRT, which fastchess uses for model=normalized."""
    t0, t1 = nelo0 / NELO_PER_T, nelo1 / NELO_PER_T
    return pairs * math.log((1 + (nt - t0) ** 2) / (1 + (nt - t1) ** 2))


def replay(games, me, opponents, args):
    """Tallies per opponent, with the Elo series after every finished pair."""
    tallies = {o: Tally() for o in opponents}
    curves = {o: {"games": [], "elo": [], "lo": [], "hi": [], "llr": []} for o in opponents}
    for g in games:
        if me not in (g["White"], g["Black"]):
            continue
        opp = g["Black"] if g["White"] == me else g["White"]
        t = tallies[opp]
        pairs_before = sum(t.penta)
        t.add(g, me)
        if sum(t.penta) == pairs_before and t.penta != [0] * 5:
            continue                 # half a pair: the curve moves on whole pairs
        s = t.summary()
        if s is None:
            continue
        c = curves[opp]
        c["games"].append(t.games)
        for k in ("elo", "lo", "hi"):
            c[k].append(s[k])
        c["llr"].append(llr(s["pairs"], s["nt"], args.elo0, args.elo1))
    return tallies, curves


# ---------------------------------------------------------------- drawing

def style(ax, grid_axis="y"):
    ax.clear()
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)


def zoomed_ylim(ax, curves, from_games=200, keep=0.0):
    """The first games swing by hundreds of Elo: scale on the rest, with keep in view."""
    lo, hi = [], []
    for c in curves:
        late = [i for i, g in enumerate(c["games"]) if g >= min(from_games, c["games"][-1])]
        lo += [c["lo"][i] for i in late]
        hi += [c["hi"][i] for i in late]
    if lo:
        pad = 0.1 * max(hi[-1] - lo[-1], max(hi) - min(lo), 4)
        low, high = min(lo), max(hi)
        if keep is not None:
            low, high = min(low, keep), max(high, keep)
        ax.set_ylim(low - pad, high + pad)


def elo_panel(ax, curve, color, label=None, end_label=True):
    """Elo per game with the 95% band drawn as a filled region with edges."""
    g = curve["games"]
    ax.fill_between(g, curve["lo"], curve["hi"], color=color, alpha=0.18, lw=0)
    ax.plot(g, curve["lo"], color=color, lw=0.8, alpha=0.6)
    ax.plot(g, curve["hi"], color=color, lw=0.8, alpha=0.6)
    ax.plot(g, curve["elo"], color=color, lw=2, label=label)
    ax.plot(g[-1], curve["elo"][-1], "o", color=color, ms=7, mec=SURFACE, mew=2, zorder=5)
    if end_label:
        half = (curve["hi"][-1] - curve["lo"][-1]) / 2
        ax.annotate(f"{curve['elo'][-1]:+.1f} ± {half:.1f}", (g[-1], curve["elo"][-1]),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK, weight="bold")


def los_gauge(ax, x, y, width, los):
    """LOS as a horizontal bar from 0 to 100%, with the 5% and 95% marks."""
    h = 0.035
    ax.add_patch(FancyBboxPatch((x, y), width, h, boxstyle="round,pad=0,rounding_size=0.01",
                                                   transform=ax.transAxes, color=GRID, lw=0))
    col = GOOD if los >= 0.95 else BAD if los <= 0.05 else PALETTE[0]
    ax.add_patch(FancyBboxPatch((x, y), width * los, h, boxstyle="round,pad=0,rounding_size=0.01",
                                                   transform=ax.transAxes, color=col, lw=0))
    for mark in (0.05, 0.5, 0.95):
        ax.plot([x + width * mark] * 2, [y - 0.008, y + h + 0.008], color=FAINT, lw=0.8,
                transform=ax.transAxes)
    ax.text(x + width + 0.02, y + h / 2, f"{100 * los:.1f}%", va="center", fontsize=11,
            weight="bold", color=INK, transform=ax.transAxes)


def eta_text(t):
    """English weekday whatever the locale (GTK sets LC_TIME from the desktop)."""
    days = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    return f"{days[t.tm_wday]} {t.tm_hour:02d}:{t.tm_min:02d}"


def run_info(pgn):
    """SPRT settings and expected games of the run that writes this pgn.

    From the fastchess config saved with -config outname (newest first, the
    archived .json.bak too), else from the .elo_run.conf of runfastchess.sh.
    """
    configs = sorted([*pgn.parent.glob("*.json"), *pgn.parent.glob("*.json.bak")],
                     key=lambda f: f.stat().st_mtime, reverse=True)
    for f in configs:
        try:
            conf = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(conf, dict) or Path(conf.get("pgn", {}).get("file", "")).name != pgn.name:
            continue
        sprt = conf.get("sprt") or {}
        # alpha and beta near zero never stop the run: fastchess used as a plain match
        if min(sprt.get("alpha", 1), sprt.get("beta", 1)) < 1e-6:
            sprt = {}
        total = conf.get("rounds", 0) * conf.get("games", 0) or None
        return {"sprt": sprt if sprt.get("enabled") else None, "total": total, "source": f.name}
    conf = pgn.parent / ".elo_run.conf"
    if conf.exists():
        values = dict(re.findall(r'^(\w+)="?([^"\n]*)"?$', conf.read_text(), re.M))
        if Path(values.get("PGN", "")).name == pgn.name and values.get("TOTAL_GAMES", "").isdigit():
            return {"sprt": None, "total": int(values["TOTAL_GAMES"]), "source": conf.name}
    return {"sprt": None, "total": None, "source": None}


def stamp(text):
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S %z").timestamp()
    except ValueError:
        return None


def timing(games, pgn):
    """Games per hour, whether the run is going, time elapsed, time losses."""
    start = stamp(games[0].get("GameStartTime", "")) or stamp(games[0].get("GameEndTime", ""))
    end = stamp(games[-1].get("GameEndTime", ""))
    running = time.time() - pgn.stat().st_mtime < 120
    rate = len(games) * 3600 / (end - start) if start and end and end > start else 0
    elapsed = ((time.time() if running else end) - start) if start and end else None
    forfeits = sum(g.get("Termination") == "time forfeit" for g in games)
    return {"rate": rate, "running": running, "elapsed": elapsed, "forfeits": forfeits}


def hms(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def forfeit_text(forfeits, n):
    pct = 100 * forfeits / n if n else 0
    return f"{forfeits} ({pct:.2f}%)", (BAD if pct > 1 else INK)


def draw_match(fig, games, me, opp, args, pgn, run):
    tallies, curves = replay(games, me, [opp], args)
    t, c = tallies[opp], curves[opp]
    s = t.summary()
    show_llr = run["llr"] and s and s["pairs"] >= 2
    cap = run["cap"]
    grid = fig.add_gridspec(3 if show_llr else 2, 3, height_ratios=([3, 3, 1.2] if show_llr else [3, 3]),
                            width_ratios=[2.2, 2.2, 1.6])
    ax_elo = fig.add_subplot(grid[0:2, 0:2])
    ax_penta = fig.add_subplot(grid[0, 2])
    ax_card = fig.add_subplot(grid[1:, 2])
    if s is None or not c["games"]:
        ax_elo.set_title("waiting for games", color=MUTED)
        return

    style(ax_elo)
    ax_elo.axhline(0, color=FAINT, lw=1, ls="--")
    elo_panel(ax_elo, c, PALETTE[0])
    zoomed_ylim(ax_elo, [c])
    ax_elo.set_title(f"Elo of {me} against {opp}, with its 95% interval",
                     loc="left", color=INK, fontsize=10)
    ax_elo.set_ylabel("Elo", color=MUTED, fontsize=9)
    xmax = max(cap or 0, c["games"][-1]) * 1.04
    ax_elo.set_xlim(0, xmax)
    if not show_llr:
        ax_elo.set_xlabel("games", color=MUTED, fontsize=9)

    lower = math.log(args.beta / (1 - args.alpha))
    upper = math.log((1 - args.beta) / args.alpha)
    cur_llr = llr(s["pairs"], s["nt"], args.elo0, args.elo1)
    if show_llr:
        ax_llr = fig.add_subplot(grid[2, 0:2], sharex=ax_elo)
        style(ax_llr)
        ax_llr.axhspan(lower, upper, color=GRID, lw=0)
        ax_llr.axhline(upper, color=GOOD, lw=1, ls="--")
        ax_llr.axhline(lower, color=BAD, lw=1, ls="--")
        ax_llr.plot(c["games"], c["llr"], color=FAINT, lw=1.4)
        ax_llr.plot(c["games"][-1], c["llr"][-1], "o", color=MUTED, ms=5, mec=SURFACE, mew=1.5)
        ax_llr.set_ylim(lower * 1.15, upper * 1.15)
        ax_llr.set_yticks([lower, 0, upper])
        ax_llr.set_title(f"LLR {cur_llr:+.2f}   SPRT [{args.elo0:g}, {args.elo1:g}] normalized",
                         loc="left", color=MUTED, fontsize=9)
        ax_llr.set_xlabel("games", color=MUTED, fontsize=9)
        plt_setp(ax_elo.get_xticklabels(), visible=False)

    style(ax_penta)
    labels = ("LL", "LD", "DD+WL", "WD", "WW")
    colors = (BAD, BAD, FAINT, GOOD, GOOD)
    if s["pairs"]:
        bars = ax_penta.bar(labels, t.penta, color=colors, width=0.7, edgecolor=SURFACE, linewidth=2)
        for b, n in zip(bars, t.penta):
            ax_penta.annotate(f"{100 * n / s['pairs']:.1f}%", (b.get_x() + b.get_width() / 2, n),
                              xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                              fontsize=8, color=INK)
        ax_penta.set_ylim(0, max(t.penta) * 1.2)
    ax_penta.set_title(f"pentanomial, {s['pairs']} pairs", loc="left", color=INK, fontsize=10)

    tm = timing(games, pgn)
    rate, running = tm["rate"], tm["running"]
    n = t.games
    if show_llr and cur_llr >= upper:
        verdict, vcol = "H1 accepted: passed", GOOD
    elif show_llr and cur_llr <= lower:
        verdict, vcol = "H0 accepted: rejected", BAD
    elif cap and n >= cap:
        verdict, vcol = (f"inconclusive at {cap} games" if show_llr else "finished"), MUTED
    else:
        verdict, vcol = ("running" if running else "stopped"), (PALETTE[0] if running else MUTED)

    ax = ax_card
    ax.clear()
    ax.axis("off")
    tr = ax.transAxes
    half = (s["hi"] - s["lo"]) / 2
    ax.text(0.02, 0.97, f"{me}  vs  {opp}", fontsize=10, color=MUTED, va="top", transform=tr)
    ax.text(0.02, 0.88, f"{s['elo']:+.1f}", fontsize=30, weight="bold", color=INK, va="top", transform=tr)
    ax.text(0.50, 0.85, f"± {half:.1f} Elo\n95%: {s['lo']:+.1f} .. {s['hi']:+.1f}",
            fontsize=10, color=INK, va="top", transform=tr, linespacing=1.5)
    ax.text(0.02, 0.62, "LOS", fontsize=9, color=MUTED, transform=tr)
    los_gauge(ax, 0.14, 0.60, 0.58, s["los"])
    w, d, lo_ = t.wdl
    rows = [("games", f"{n}" + (f" / {cap}" if cap else "")),
            ("W / D / L", f"{w} / {d} / {lo_}"),
            ("score", f"{100 * (w + d / 2) / n:.1f}%"),
            ("nElo", f"{s['nelo']:+.1f}")]
    if show_llr:
        rows.append(("LLR", f"{cur_llr:+.2f}  ({lower:+.2f}, {upper:+.2f})"))
    rows.append(("time", (hms(tm["elapsed"]) if tm["elapsed"] else "?")
                 + (f",  {rate:.0f} games/h" if rate else "")))
    rows.append(("time losses", *forfeit_text(tm["forfeits"], n)))
    if running and rate and cap and n < cap:
        rows.append(("end", f"{eta_text(time.localtime(time.time() + (cap - n) / rate * 3600))}"
                            f"  ({hms((cap - n) / rate * 3600)} left)"))
    if show_llr and abs(cur_llr) < upper and n >= 200:
        # where the LLR is heading at its average drift so far
        slope = cur_llr / n
        bound, name = (upper, "H1") if slope > 0 else (lower, "H0")
        need = (bound - cur_llr) / slope if slope else math.inf
        if cap and n + need > cap:
            rows.append(("trend", "no verdict before the cap"))
        elif rate and running:
            rows.append(("trend", f"{name} in ~{need:.0f} games, "
                                  f"{eta_text(time.localtime(time.time() + need / rate * 3600))}"))
        else:
            rows.append(("trend", f"{name} in ~{need:.0f} games"))
    for i, row in enumerate(rows):
        k, v, col = (*row, INK) if len(row) == 2 else row
        y = 0.52 - i * 0.052
        ax.text(0.02, y, k, fontsize=9, color=MUTED, transform=tr)
        ax.text(0.30, y, v, fontsize=9.5, color=col, family="monospace", transform=tr)
    ax.text(0.02, 0.0, verdict, fontsize=13, weight="bold", color=vcol, transform=tr)


def ladder(games, me, opponents, anchor, anchor_elo):
    """Ratings of every player on one scale, after every game of the gauntlet.

    The opponents never meet, so each is placed through the engine under test:
    me = anchor + (me against the anchor), opponent = me - (me against it),
    its error the anchor's and its own added in quadrature.
    """
    tallies = {o: Tally() for o in opponents}
    curves = {p: {"games": [], "elo": [], "lo": [], "hi": []} for p in [me, *opponents]}
    played = 0
    for g in games:
        if me not in (g["White"], g["Black"]):
            continue
        opp = g["Black"] if g["White"] == me else g["White"]
        tallies[opp].add(g, me)
        played += 1
        rows = rate_all(tallies, me, anchor, anchor_elo)
        for p, (r, h) in rows.items():
            c = curves[p]
            c["games"].append(played)
            c["elo"].append(r)
            c["lo"].append(r - h)
            c["hi"].append(r + h)
    return tallies, curves, rate_all(tallies, me, anchor, anchor_elo)


def rate_all(tallies, me, anchor, anchor_elo):
    """{player: (rating, 95% half-width)} for those that can be placed yet."""
    sums = {o: t.summary() for o, t in tallies.items()}
    half = {o: (s["hi"] - s["lo"]) / 2 for o, s in sums.items() if s}
    if anchor == me:
        mine = (anchor_elo, 0.0)
    elif sums.get(anchor):
        mine = (anchor_elo + sums[anchor]["elo"], half[anchor])
    else:
        return {}
    out = {me: mine}
    for o, s in sums.items():
        if not s:
            continue
        if o == anchor:
            out[o] = (anchor_elo, 0.0)
        else:
            out[o] = (mine[0] - s["elo"], math.hypot(mine[1], half[o]))
    return out


def draw_gauntlet(fig, games, me, opponents, args, pgn, run):
    anchor, anchor_elo = opponents[0], 0.0
    if args.anchor:
        name, _, value = args.anchor.partition("=")
        anchor, anchor_elo = name, float(value or 0)
    if anchor not in (me, *opponents):
        fig.suptitle(f"anchor {anchor} is not in this pgn", x=0.01, ha="left", color=BAD)
        return
    tallies, curves, final = ladder(games, me, opponents, anchor, anchor_elo)
    grid = fig.add_gridspec(2, 3, height_ratios=[3, 0.8 + 0.25 * (len(opponents) + 1)], width_ratios=[2.2, 2.2, 1.6])
    ax_elo = fig.add_subplot(grid[0, 0:2])
    ax_cmp = fig.add_subplot(grid[0, 2])
    ax_tab = fig.add_subplot(grid[1, :])
    if not final:
        ax_elo.set_title(f"waiting for games against the anchor {anchor}", color=MUTED)
        return
    rf = "{:+.0f}" if anchor_elo == 0 else "{:.0f}"   # signed only on a relative scale
    color = {o: PALETTE[i % len(PALETTE)] for i, o in enumerate(opponents)}
    color[me] = INK
    players = sorted(final, key=lambda p: -final[p][0])
    anchor_text = f"{anchor} = {anchor_elo:g}" + ("" if args.anchor else " (default anchor)")

    style(ax_elo)
    for p in players:
        c = curves[p]
        if not c["games"]:
            continue
        if p == anchor:
            ax_elo.axhline(anchor_elo, color=color[p], lw=1.6, ls="--",
                           label=f"{p}   {rf.format(anchor_elo)}  anchor")
            continue
        r, h = final[p]
        elo_panel(ax_elo, c, color[p], label=f"{p}   {rf.format(r)} ± {h:.0f}", end_label=False)
    zoomed_ylim(ax_elo, [curves[p] for p in players if curves[p]["games"] and p != anchor], keep=anchor_elo)
    ax_elo.set_title(f"rating of every engine, with its 95% interval   ·   {anchor_text}",
                     loc="left", color=INK, fontsize=10)
    ax_elo.set_xlabel("games played in the gauntlet", color=MUTED, fontsize=9)
    ax_elo.set_ylabel("Elo", color=MUTED, fontsize=9)
    ax_elo.set_xlim(0, max(curves[me]["games"][-1], 1) * 1.02)
    # legend in ladder order, so it reads like the panel on the right
    ax_elo.legend(loc="best", frameon=True, framealpha=0.9, edgecolor=GRID, fontsize=9)

    # the ladder: strongest at the top, the engine under test in bold
    style(ax_cmp, grid_axis="x")
    ys = list(range(len(players)))[::-1]
    for y, p in zip(ys, players):
        r, h = final[p]
        if h:
            ax_cmp.plot([r - h, r + h], [y, y], color=color[p], lw=7, alpha=0.3, solid_capstyle="round")
        ax_cmp.plot(r, y, "D" if p == anchor else "o", color=color[p], ms=9, mec=SURFACE, mew=2)
        ax_cmp.annotate(rf.format(r) + (f" ± {h:.0f}" if h else "  anchor"), (r, y), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=9, weight="bold", color=INK)
    ax_cmp.set_yticks(ys)
    ax_cmp.set_yticklabels(players, fontsize=9)
    for lab in ax_cmp.get_yticklabels():
        if lab.get_text() == me:
            lab.set_weight("bold")
    ax_cmp.set_ylim(-0.7, len(players) - 0.3)
    ax_cmp.set_title("ladder, 95% intervals", loc="left", color=INK, fontsize=10)
    ax_cmp.set_xlabel("Elo", color=MUTED, fontsize=9)

    ax_tab.clear()
    ax_tab.axis("off")
    # standings: every engine by rating, each row from that engine's side of the board
    header = ("#", "engine", "rating", "games", "W / D / L", "score", f"vs {me}", "95% interval", f"LOS vs {me}")
    cols = (0.0, 0.03, 0.24, 0.35, 0.42, 0.55, 0.62, 0.71, 0.86)
    tr = ax_tab.transAxes
    for x, h in zip(cols, header):
        ax_tab.text(x, 0.92, h, fontsize=9, color=MUTED, transform=tr)
    ax_tab.plot([0, 1], [0.88, 0.88], color=GRID, lw=1, transform=tr)
    live = [o for o in opponents if tallies[o].summary()]
    total = [0, 0, 0]
    for o in live:
        total = [a + b for a, b in zip(total, tallies[o].wdl)]
    rows = [p for p in players if p == me or p in live]
    for i, p in enumerate(rows):
        y = 0.88 - (i + 1) * 0.8 / (len(rows) + 1)
        r, h = final[p]
        rating = f"{rf.format(r)}  anchor" if p == anchor else f"{rf.format(r)} ± {h:.0f}"
        if p == me:
            w, d, l = total
            games_n, vs = sum(total), ("-", "-", "-")
        else:
            t, s = tallies[p], tallies[p].summary()
            l, d, w = t.wdl          # the tally counts from the side of the engine under test
            games_n = t.games
            vs = (f"{-s['elo']:+.1f}", f"{-s['hi']:+.1f} .. {-s['lo']:+.1f}", f"{100 * (1 - s['los']):.1f}%")
        cells = (f"{i + 1}", p, rating, f"{games_n}", f"{w} / {d} / {l}",
                 f"{100 * (w + d / 2) / games_n:.1f}%", *vs)
        ax_tab.plot(cols[1] - 0.012, y + 0.02, "s", color=color[p], ms=8, transform=tr, clip_on=False)
        for x, v in zip(cols, cells):
            ax_tab.text(x, y, v, fontsize=10, color=INK, transform=tr,
                        family="monospace" if x > cols[1] else None,
                        weight="bold" if p == me or x == cols[2] else None)
    tm = timing(games, pgn)
    n = sum(total)
    w, d, l = total
    r, h = final[me]
    cap = run["cap"]
    parts = [f"{me}: rating {rf.format(r)} ± {h:.0f}",
             f"{n}" + (f" / {cap}" if cap else "") + f" games, +{w} ={d} -{l}, score {100 * (w + d / 2) / n:.1f}%",
             (hms(tm["elapsed"]) if tm["elapsed"] else "?") + (f", {tm['rate']:.0f} games/h" if tm["rate"] else "")]
    if tm["running"] and tm["rate"] and cap and n < cap:
        parts.append(f"end {eta_text(time.localtime(time.time() + (cap - n) / tm['rate'] * 3600))}")
    parts.append("running" if tm["running"] else ("finished" if cap and n >= cap else "stopped"))
    ax_tab.text(0.0, 0.02, "   ·   ".join(parts), fontsize=9, color=MUTED, transform=tr)
    text, col = forfeit_text(tm["forfeits"], len(games))
    ax_tab.text(1.0, 0.02, f"time losses {text}", fontsize=9, ha="right",
                color=col if col == BAD else MUTED, transform=tr)


def resolve(args, gauntlet):
    """LLR on or off, SPRT bounds and the game total: flags first, then the run's
    config, then the defaults (an SPRT caps at 8000 games)."""
    info = run_info(args.pgn)
    sprt = info["sprt"] or {}
    llr_on = False if gauntlet else (args.llr if args.llr is not None
                                     else bool(sprt) or args.pgn.name.startswith("sprt"))
    for key, default in (("elo0", 0.0), ("elo1", 5.0), ("alpha", 0.05), ("beta", 0.05)):
        if getattr(args, key) is None:
            setattr(args, key, float(sprt.get(key, default)))
    cap = args.total if args.total is not None else (info["total"] or (8000 if llr_on else None))
    return {"llr": llr_on, "cap": cap or None, "source": info["source"]}


def plt_setp(items, **kw):
    for it in items:
        it.set(**kw)


def draw(fig, args):
    fig.clear()
    games = read_games(args.pgn)
    if not games:
        fig.suptitle(f"{args.pgn.name}: waiting for games", x=0.01, ha="left", color=INK)
        return
    me = engine_under_test(games, args.engine)
    opponents = list(dict.fromkeys(g["Black"] if g["White"] == me else g["White"]
                                   for g in games if me in (g["White"], g["Black"])))
    run = resolve(args, len(opponents) > 1)
    if len(opponents) == 1:
        draw_match(fig, games, me, opponents[0], args, args.pgn, run)
        kind = "SPRT" if run["llr"] else "match"
    else:
        draw_gauntlet(fig, games, me, opponents, args, args.pgn, run)
        kind = "gauntlet"
    fig.suptitle(f"{kind}  {args.pgn.name}   (updated {time.strftime('%H:%M:%S')})",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pgn", nargs="?", type=Path)
    ap.add_argument("--engine", help="engine under test (default: the one not named base, "
                                     "or the one with the most games)")
    ap.add_argument("--elo0", type=float, help="default: the run's config, else 0")
    ap.add_argument("--elo1", type=float, help="default: the run's config, else 5")
    ap.add_argument("--alpha", type=float, help="default: the run's config, else 0.05")
    ap.add_argument("--beta", type=float, help="default: the run's config, else 0.05")
    ap.add_argument("--total", type=int, help="expected games, 0 for unknown (default: the run's "
                                              "config, else 8000 for an SPRT)")
    llr = ap.add_mutually_exclusive_group()
    llr.add_argument("--llr", dest="llr", action="store_true", default=None,
                     help="show the LLR (default: when the config has an SPRT or the pgn is sprt_*)")
    llr.add_argument("--no-llr", dest="llr", action="store_false")
    ap.add_argument("--anchor", help="gauntlet: NAME=ELO fixes that player's rating "
                                     "(default: the first opponent at 0)")
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--save", type=Path, help="write one frame to this png and exit")
    args = ap.parse_args()
    if args.pgn is None:
        found = sorted(Path.cwd().glob("*.pgn"), key=lambda p: p.stat().st_mtime)
        if not found:
            ap.error("no pgn here: pass one")
        args.pgn = found[-1]

    # a watcher must not take cpu from the games it watches
    os.nice(10)
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig = plt.figure(figsize=(15, 8.5), facecolor=SURFACE)
    if args.save:
        draw(fig, args)
        fig.savefig(args.save, dpi=110, facecolor=SURFACE)
        return
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Ctrl-C closes the window too
    fig.canvas.manager.set_window_title(f"{args.pgn.name}")
    anim = FuncAnimation(fig, lambda _: draw(fig, args), interval=args.every * 1000,
                         cache_frame_data=False)
    plt.show()
    return anim


if __name__ == "__main__":
    main()
