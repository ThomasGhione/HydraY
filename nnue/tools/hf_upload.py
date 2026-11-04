#!/usr/bin/env python3
"""Uploads the sealed datagen shards to a Hugging Face dataset.

Every --every seconds it lists the sealed shards in DATA (<prefix>.sNNNN.bin,
only those of --prefix when given:
the open .part, the unsealed --shard 0 files and the old .t<N>.bin files are
never touched), uploads the ones the dataset lacks to <subdir>/<name>, and
checks each upload by size on the Hub. --delete-after-upload then removes the
local copy, and only then. A shard already on the Hub with the same size counts
as uploaded, so a restart picks up where it stopped.

The token comes from HF_TOKEN (or `hf auth login`); the dataset is created
private if missing.

Usage:  hf_upload.py --data data --repo user/hydray-data [--prefix pc1] [--subdir pc1]
                     [--every 600] [--once] [--delete-after-upload] [--dry-run]
"""
import argparse
import re
import signal
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

SEALED_RE = re.compile(r".+\.s\d{4,}\.bin")

stopping = False


def on_stop(signum, frame):
    global stopping
    stopping = True


def sealed_shards(data, prefix):
    return sorted(p for p in data.iterdir()
                  if p.is_file() and SEALED_RE.fullmatch(p.name)
                  and (not prefix or p.name.startswith(prefix + ".s")))


def remote_sizes(api, repo, paths):
    if not paths:
        return {}
    infos = api.get_paths_info(repo, paths, repo_type="dataset")
    return {i.path: i.size for i in infos if hasattr(i, "size")}


def one_pass(api, args):
    shards = sealed_shards(args.data, args.prefix)
    if not shards:
        return
    in_repo = {p: f"{args.subdir}/{p.name}" if args.subdir else p.name for p in shards}
    present = set(api.list_repo_files(args.repo, repo_type="dataset")) if not args.dry_run else set()
    known = remote_sizes(api, args.repo, [r for r in in_repo.values() if r in present])

    for local, remote in in_repo.items():
        if stopping:
            return
        size = local.stat().st_size
        if known.get(remote) != size:
            if args.dry_run:
                print(f"would upload {local} -> {remote} ({size} bytes)", flush=True)
                continue
            print(f"uploading {local} -> {args.repo}/{remote} ({size / 1e9:.2f} GB)", flush=True)
            api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                            repo_id=args.repo, repo_type="dataset",
                            commit_message=f"Add {remote}")
            known.update(remote_sizes(api, args.repo, [remote]))
            if known.get(remote) != size:
                print(f"{remote}: the Hub reports {known.get(remote)} bytes, not {size}: "
                      "kept locally, retried next pass", flush=True)
                continue
            print(f"uploaded {remote}", flush=True)
        if args.delete_after_upload and not args.dry_run:
            local.unlink()
            print(f"deleted {local}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True, help="directory with the shards")
    ap.add_argument("--repo", required=True, help="dataset id, user/name")
    ap.add_argument("--prefix", default="", help="only the shards of this datagen --out name")
    ap.add_argument("--subdir", default="", help="folder in the dataset (default: its root)")
    ap.add_argument("--every", type=int, default=600, help="seconds between passes")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--delete-after-upload", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list, upload nothing, no token needed")
    args = ap.parse_args()
    args.subdir = args.subdir.strip("/")
    if not args.data.is_dir():
        sys.exit(f"hf_upload: {args.data} is not a directory")

    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)
    api = HfApi()
    if not args.dry_run:
        user = api.whoami()["name"]  # fails fast on a missing or bad token
        api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
        print(f"hf_upload: {args.data} -> datasets/{args.repo}/{args.subdir} as {user}, "
              f"{'one pass' if args.once else f'every {args.every}s'}"
              f"{', deleting after upload' if args.delete_after_upload else ''}", flush=True)

    while True:
        try:
            one_pass(api, args)
        except Exception as e:  # network or Hub errors: the next pass retries
            print(f"hf_upload: pass failed: {e}", flush=True)
        if args.once or stopping:
            return
        for _ in range(args.every):
            if stopping:
                return
            time.sleep(1)


if __name__ == "__main__":
    main()
