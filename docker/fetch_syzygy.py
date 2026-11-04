#!/usr/bin/env python3
"""Makes DIR hold the Syzygy 3-4-5 tables, every file checked against the
SHA-256 list: a file missing or different is fetched from Lichess (HTTPS),
checked, then renamed into place, so an interrupted run never leaves a bad
table behind and the next one resumes from the files already there.

    fetch_syzygy.py --dir DIR --sums syzygy-3-4-5.sha256 [--check]

--check only reports: exit 0 if DIR is complete and sound, 1 if not.
"""
import argparse
import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://tablebase.lichess.ovh/tables/standard/"
FOLDER = {".rtbw": "3-4-5-wdl", ".rtbz": "3-4-5-dtz"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, want, dest):
    url = BASE + FOLDER[Path(name).suffix] + "/" + name
    tmp = dest / f".{name}.{os.getpid()}.tmp"
    for attempt in range(1, 6):
        try:
            h = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    h.update(chunk)
                    f.write(chunk)
            if h.hexdigest() != want:
                raise ValueError("SHA-256 differs from the list")
            os.replace(tmp, dest / name)
            return
        except Exception as e:
            tmp.unlink(missing_ok=True)
            print(f"fetch_syzygy: {name}: {e} (attempt {attempt}/5)", flush=True)
            time.sleep(5 * attempt)
    sys.exit(f"fetch_syzygy: cannot fetch {url}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--sums", required=True, type=Path)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    sums = [line.split() for line in args.sums.read_text().splitlines() if line.strip()]
    bad = [(name, want) for want, name in sums
           if not (args.dir / name).is_file() or sha256(args.dir / name) != want]
    if not bad:
        print(f"fetch_syzygy: {args.dir} holds the {len(sums)} tables, all checked", flush=True)
        return
    if args.check:
        print(f"fetch_syzygy: {len(bad)} of {len(sums)} tables missing or damaged in {args.dir}",
              flush=True)
        sys.exit(1)
    print(f"fetch_syzygy: fetching {len(bad)} of {len(sums)} tables (all 290: ~1 GB) "
          f"into {args.dir}", flush=True)
    for i, (name, want) in enumerate(bad, 1):
        fetch(name, want, args.dir)
        if i % 20 == 0 or i == len(bad):
            print(f"fetch_syzygy: {i}/{len(bad)}", flush=True)


if __name__ == "__main__":
    main()
