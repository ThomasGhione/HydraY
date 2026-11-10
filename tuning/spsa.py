#!/usr/bin/env python3
"""SPSA tuner for HydraY search constants, in the style of Fishtest/OpenBench.

Each iteration perturbs every parameter by +-c_k at random, plays game pairs
theta+ vs theta- with fastchess, and moves theta along the perturbation in
proportion to the score difference:

    theta_i += R_k * c_k,i * (W - L) * delta_i,   R_k = a_k / c_k^2

with the Fishtest schedules c_k = c / k^gamma and a_k = a / (A + k)^alpha,
where c and a come from each parameter's c_end and r_end (a_end = r_end * c_end^2).

Groups live in groups/<name>.json. "parameter_ranges" accepts both the
chess-tuning-tools form, "Integer(min, max)" (start = engine default,
c_end = (max - min) / 20, r_end = 0.002), and the explicit form
{"start": .., "min": .., "max": .., "c_end": .., "r_end": ..}, where a
missing "start" is the engine default too.

State (theta and the iteration) is saved after every iteration to
spsa_<group>_state.json, so a run interrupted with Ctrl-C resumes where it
stopped; the log of theta per iteration goes to spsa_<group>.log.

Usage:  ./spsa.py search_corrhist [--iterations 1500] [--pairs 8] [--tc 8+0.08]
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ENGINE = SCRIPT_DIR.parent / "chess"
BOOK = SCRIPT_DIR / "books" / "openings.pgn"

ALPHA, GAMMA = 0.602, 0.101
INTEGER_RE = re.compile(r"Integer\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
RESULT_RE = re.compile(r"Games: (\d+), Wins: (\d+), Losses: (\d+), Draws: (\d+)")


def option_name(key):
    """RFP_MARGIN_PER_DEPTH -> RfpMarginPerDepth, as the engine prints it."""
    return "".join(part.capitalize() for part in key.split("_"))


def engine_defaults():
    out = subprocess.run([str(ENGINE), "-uci"], input="uci\nquit\n",
                         capture_output=True, text=True, timeout=30).stdout
    defaults = {}
    for m in re.finditer(r"option name (\S+) type spin default (-?\d+)", out):
        defaults[m.group(1).lower()] = int(m.group(2))
    return defaults


def load_params(group_file):
    ranges = json.loads(group_file.read_text())["parameter_ranges"]
    defaults = None
    params = []
    for key, spec in ranges.items():
        if isinstance(spec, str):
            m = INTEGER_RE.fullmatch(spec.strip())
            if not m:
                sys.exit(f"{key}: cannot parse range {spec!r}")
            lo, hi = int(m.group(1)), int(m.group(2))
            if defaults is None:
                defaults = engine_defaults()
            start = defaults.get(option_name(key).lower())
            if start is None:
                sys.exit(f"{key}: the engine offers no spin option {option_name(key)}")
            spec = {"start": start, "min": lo, "max": hi,
                    "c_end": (hi - lo) / 20, "r_end": 0.002}
        elif "start" not in spec:
            if defaults is None:
                defaults = engine_defaults()
            start = defaults.get(option_name(key).lower())
            if start is None:
                sys.exit(f"{key}: the engine offers no spin option {option_name(key)}")
            spec = {"start": start, **spec}
        params.append({"key": key, "option": option_name(key), **spec})
    return params


def play(theta_plus, theta_minus, params, args, seed, pgn):
    def engine(name, theta):
        opts = [f"option.{p['option']}={v}" for p, v in zip(params, theta)]
        return ["-engine", f"name={name}", f"cmd={ENGINE}", "args=-uci", *opts]

    cmd = ["fastchess",
           *engine("plus", theta_plus), *engine("minus", theta_minus),
           "-each", f"tc={args.tc}", "option.Threads=1", f"option.Hash={args.hash}",
           "-openings", f"file={BOOK}", "format=pgn", "order=random", "plies=16",
           "-srand", str(seed),
           "-repeat", "-games", "2", "-rounds", str(args.pairs),
           "-resign", "movecount=3", "score=500",
           "-draw", "movenumber=40", "movecount=8", "score=8",
           "-concurrency", str(args.concurrency),
           "-recover", "-pgnout", f"file={pgn}"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    matches = RESULT_RE.findall(out)
    if not matches:
        raise RuntimeError("fastchess printed no result:\n" + out[-2000:])
    games, wins, losses, draws = map(int, matches[-1])
    return wins, losses, draws


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("group")
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--pairs", type=int, default=8, help="game pairs per iteration")
    ap.add_argument("--tc", default="8+0.08")
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=0, help="default: physical cores - 1")
    args = ap.parse_args()

    if not args.concurrency:
        out = subprocess.run(["lscpu", "-p=Core,Socket"], capture_output=True, text=True).stdout
        cores = len({l for l in out.splitlines() if not l.startswith("#")}) or 2
        args.concurrency = max(1, cores - 1)

    os.chdir(SCRIPT_DIR)
    group_file = Path("groups") / f"{args.group}.json"
    params = load_params(group_file)
    state_file = Path(f"spsa_{args.group}_state.json")
    log_file = Path(f"spsa_{args.group}.log")
    pgn = Path(f"spsa_{args.group}.pgn")

    if state_file.exists():
        state = json.loads(state_file.read_text())
        if [p["key"] for p in params] != list(state["theta"]):
            sys.exit(f"{state_file} belongs to other parameters: move it away to start over")
        print(f"Resuming {args.group} at iteration {state['k'] + 1}")
    else:
        state = {"k": 0, "theta": {p["key"]: float(p["start"]) for p in params},
                 "wins": 0, "losses": 0, "draws": 0}
        print(f"Starting {args.group} from {[p['start'] for p in params]}")

    N = args.iterations
    A = 0.1 * N
    for p in params:
        p["c"] = p["c_end"] * N ** GAMMA
        p["a"] = p["r_end"] * p["c_end"] ** 2 * (A + N) ** ALPHA

    print(f"{N} iterations x {2 * args.pairs} games at {args.tc}, concurrency {args.concurrency}")
    while state["k"] < N:
        k = state["k"] + 1
        theta = [state["theta"][p["key"]] for p in params]
        ck = [p["c"] / k ** GAMMA for p in params]
        delta = [random.choice((-1, 1)) for _ in params]
        clamp = lambda v, p: min(p["max"], max(p["min"], v))
        plus = [round(clamp(t + c * d, p)) for t, c, d, p in zip(theta, ck, delta, params)]
        minus = [round(clamp(t - c * d, p)) for t, c, d, p in zip(theta, ck, delta, params)]

        started = time.time()
        wins, losses, draws = play(plus, minus, params, args, seed=k, pgn=pgn)
        result = wins - losses

        for i, p in enumerate(params):
            ak = p["a"] / (A + k) ** ALPHA
            theta[i] = clamp(theta[i] + ak / ck[i] * result * delta[i], p)

        state["k"] = k
        state["theta"] = {p["key"]: t for p, t in zip(params, theta)}
        state["wins"] += wins
        state["losses"] += losses
        state["draws"] += draws
        tmp = state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(state_file)

        shown = " ".join(f"{p['key']}={t:.1f}" for p, t in zip(params, theta))
        line = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} it {k}/{N} "
                f"+{wins} -{losses} ={draws} ({time.time() - started:.0f}s) {shown}")
        print(line, flush=True)
        with log_file.open("a") as f:
            f.write(line + "\n")

    print("Done. Final:", {p["key"]: round(state["theta"][p["key"]]) for p in params})


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted: the last finished iteration is saved, rerun to resume.")
        sys.exit(130)
