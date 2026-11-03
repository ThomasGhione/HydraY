#!/usr/bin/env python3
"""Live plot of an SPSA run: theta of every parameter per iteration, one panel
each, with the start and a +-c_end/2 band as reference. Redraws every 60 s.

Usage:  ./spsa_watch.py search_stale [--every 60]      (needs matplotlib)
"""
import argparse
import json
import re
import signal
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

SCRIPT_DIR = Path(__file__).resolve().parent
LINE_RE = re.compile(r"^(\S+ \S+) it (\d+)/(\d+) \+(\d+) -(\d+) =(\d+) \((\d+)s\) (.*)$")

SURFACE, INK, MUTED, GRID, BAND, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#ecebe7", "#e9e8e3", "#2a78d6"


def read(group):
    ranges = json.loads((SCRIPT_DIR / "groups" / f"{group}.json").read_text())["parameter_ranges"]
    log = SCRIPT_DIR / f"spsa_{group}.log"
    rows = []
    for line in (log.read_text().splitlines() if log.exists() else []):
        m = LINE_RE.match(line)
        if m:
            rows.append((m, {k: float(v) for k, v in (kv.split("=") for kv in m.group(8).split())}))
    return ranges, rows


def eta_text(t):
    """English weekday whatever the locale (GTK sets LC_TIME from the desktop)."""
    days = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    return f"{days[t.tm_wday]} {t.tm_hour:02d}:{t.tm_min:02d}"


def draw(fig, axes, group):
    ranges, rows = read(group)
    if not rows:
        return
    last = rows[-1][0]
    its = [int(m.group(2)) for m, _ in rows]
    secs = [int(m.group(7)) for m, _ in rows[-50:]]
    left = (int(last.group(3)) - int(last.group(2))) * sum(secs) / len(secs)

    for ax, key in zip(axes, ranges):
        spec = ranges[key]
        values = [t[key] for _, t in rows]
        ax.clear()
        ax.set_facecolor(SURFACE)
        title = f"{key}   {values[-1]:.1f}"
        if isinstance(spec, dict):
            start, c_end = spec["start"], spec["c_end"]
            ax.axhspan(start - c_end / 2, start + c_end / 2, color=BAND, lw=0)
            ax.axhline(start, color="#8a8985", lw=1, ls="--")
            title += f"   ({(values[-1] - start) / c_end:+.2f} c_end)"
        ax.plot(its, values, color=SERIES, lw=1.6)
        ax.plot(its[-1], values[-1], "o", color=SERIES, ms=5, mec=SURFACE, mew=1.5)
        ax.set_title(title, loc="left", color=INK, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    for ax in axes[len(ranges):]:
        ax.set_visible(False)

    fig.suptitle(f"SPSA {group}   it {last.group(2)}/{last.group(3)}   "
                 f"{sum(secs) / len(secs):.0f} s/it   ETA "
                 f"{eta_text(time.localtime(time.time() + left))}   "
                 f"(updated {time.strftime('%H:%M:%S')})",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group")
    ap.add_argument("--every", type=int, default=60)
    args = ap.parse_args()
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Ctrl-C closes the GTK/Qt window too

    n = len(read(args.group)[0])
    cols = 3
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(15, 9), facecolor=SURFACE)
    fig.canvas.manager.set_window_title(f"SPSA {args.group}")
    axes = axes.flatten()
    anim = FuncAnimation(fig, lambda _: draw(fig, axes, args.group),
                         interval=args.every * 1000, cache_frame_data=False)
    plt.show()
    return anim


if __name__ == "__main__":
    main()
