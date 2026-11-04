#!/usr/bin/env python3
"""Uploads the sealed datagen shards to a Hugging Face dataset.

Every --every seconds it lists the sealed shards in DATA (<prefix>.sNNNN.bin,
only those of --prefix when given: the open .part, the unsealed --shard 0
files and the old .t<N>.bin files are never touched) and uploads each one the
dataset lacks, with its manifest <prefix>.sNNNN.json, in one commit to
<subdir>/. An upload counts only once the Hub reports the local SHA-256;
--delete-after-upload then removes the local shard, and only then. Its
manifest stays: a few hundred bytes that keep the datagen from reusing the
shard's number after a restart (--next-shard covers a wiped folder).

Nothing on the Hub is ever overwritten. A shard already there with the same
SHA-256 counts as uploaded, so a restart picks up where it stopped; one with
other content is reported and left alone on both sides. <subdir>/OWNER names
the machine that writes the folder: another owner stops the upload, so two
machines with the same prefix cannot mix their shards.

After a pass that uploaded something, <subdir>/index.jsonl lists the manifests
of the folder and README.md, the dataset card, sums up every folder.

The token comes from HF_TOKEN (or `hf auth login`); the dataset is created
private if missing.

Usage:  hf_upload.py --data data --repo user/hydray-data [--prefix pc1] [--subdir pc1]
                     [--owner ID] [--every 600] [--once] [--delete-after-upload] [--dry-run]
                     [--check [--next-shard]]
"""
import argparse
import hashlib
import json
import re
import shutil
import signal
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

SEALED_RE = re.compile(r".+\.s\d{4,}\.bin")
MANIFEST_RE = re.compile(r".+\.s\d{4,}\.json")
REPO_TYPE = "dataset"

stopping = False


def on_stop(signum, frame):
    global stopping
    stopping = True


def default_owner():
    try:
        mid = Path("/etc/machine-id").read_text().strip()
        return "m" + hashlib.sha256(mid.encode()).hexdigest()[:8]
    except OSError:
        return socket.gethostname()


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def same_content(info, local):
    """True when the Hub file `info` holds exactly the local file."""
    if info.lfs is not None:
        return info.size == local.stat().st_size and info.lfs.sha256 == sha256_of(local)
    return info.blob_id == git_blob_sha1(local)


class Uploader:
    def __init__(self, api, args):
        self.api, self.args = api, args
        self.cache = args.data / ".hf_manifests" / (args.subdir or "_root")
        self.done = set()   # (name, size, mtime) of the shards already checked on the Hub

    def remote(self, name):
        return f"{self.args.subdir}/{name}" if self.args.subdir else name

    def infos(self, paths):
        if not paths:
            return {}
        return {i.path: i for i in self.api.get_paths_info(self.args.repo, list(paths), repo_type=REPO_TYPE)
                if hasattr(i, "blob_id")}

    def commit(self, files, message):
        ops = [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
               for remote, local in files.items()]
        self.api.create_commit(self.args.repo, operations=ops, commit_message=message,
                               repo_type=REPO_TYPE)

    def check_owner(self):
        """Claims <subdir>/OWNER, or stops when another machine holds it."""
        remote = self.remote("OWNER")
        if self.infos([remote]):
            path = self.api.hf_hub_download(self.args.repo, remote, repo_type=REPO_TYPE,
                                            force_download=True)
            holder = Path(path).read_text().split()
            if holder[:1] != [self.args.owner]:
                sys.exit(f"hf_upload: datasets/{self.args.repo}/{self.args.subdir} belongs to "
                         f"{' '.join(holder)}, not {self.args.owner}: give this machine another prefix")
            return
        local = self.args.data / ".hf_owner"
        local.write_text(f"{self.args.owner} {self.args.host}\n")
        self.commit({remote: local}, f"Claim {self.args.subdir or 'the root'} for {self.args.owner}")
        local.unlink()

    def next_shard(self):
        """First shard index past every one of this prefix in the dataset folder."""
        pattern = re.compile(re.escape(self.remote(self.args.prefix)) + r"\.s(\d{4,})\.(bin|json)")
        indices = [int(m.group(1)) for f in self.api.list_repo_files(self.args.repo, repo_type=REPO_TYPE)
                   if (m := pattern.fullmatch(f))]
        return max(indices) + 1 if indices else 0

    def one_pass(self):
        args = self.args
        shards = sorted(p for p in args.data.iterdir()
                        if p.is_file() and SEALED_RE.fullmatch(p.name)
                        and (not args.prefix or p.name.startswith(args.prefix + ".s")))
        if not shards:
            return False
        if args.dry_run:
            for p in shards:
                print(f"would upload {p} -> {self.remote(p.name)} ({p.stat().st_size} bytes)", flush=True)
            return False
        self.check_owner()

        uploaded = False
        for shard in shards:
            if stopping:
                break
            st = shard.stat()
            if (shard.name, st.st_size, st.st_mtime_ns) in self.done:
                continue
            manifest = shard.with_suffix(".json")
            pair = {self.remote(shard.name): shard}
            if manifest.exists():
                pair[self.remote(manifest.name)] = manifest
            known = self.infos(pair)
            clash = [r for r, info in known.items() if not same_content(info, pair[r])]
            if clash:
                print(f"hf_upload: CONFLICT {', '.join(clash)} is on the Hub with other content: "
                      "nothing uploaded or deleted, check the prefix of each machine", flush=True)
                continue
            missing = {r: l for r, l in pair.items() if r not in known}
            if missing:
                size = shard.stat().st_size
                print(f"uploading {shard.name} -> {args.repo}/{self.remote(shard.name)} "
                      f"({size / 1e9:.2f} GB)", flush=True)
                self.commit(missing, f"Add {self.remote(shard.name)}")
                known = self.infos(pair)
                bad = [r for r in pair if r not in known or not same_content(known[r], pair[r])]
                if bad:
                    print(f"hf_upload: {', '.join(bad)} differs on the Hub after the upload: "
                          "kept locally", flush=True)
                    continue
                print(f"uploaded {self.remote(shard.name)} (SHA-256 checked)", flush=True)
                uploaded = True
            if manifest.exists():
                self.cache.mkdir(parents=True, exist_ok=True)
                shutil.copy(manifest, self.cache / manifest.name)
            if args.delete_after_upload:
                shard.unlink()
                print(f"deleted {shard} (its manifest stays)", flush=True)
            else:
                self.done.add((shard.name, st.st_size, st.st_mtime_ns))
        if uploaded:
            self.publish_index()
        return uploaded

    def publish_index(self):
        """<subdir>/index.jsonl from every manifest of the folder, then the card."""
        files = self.api.list_repo_files(self.args.repo, repo_type=REPO_TYPE)
        prefix = self.remote("")
        self.cache.mkdir(parents=True, exist_ok=True)
        for f in files:
            name = f[len(prefix):]
            if f.startswith(prefix) and "/" not in name and MANIFEST_RE.fullmatch(name) \
                    and not (self.cache / name).exists():
                shutil.copy(self.api.hf_hub_download(self.args.repo, f, repo_type=REPO_TYPE),
                            self.cache / name)
        rows = [json.loads(p.read_text()) for p in sorted(self.cache.glob("*.json"))]
        index = self.args.data / ".hf_index.jsonl"
        index.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        self.commit({self.remote("index.jsonl"): index}, f"Index of {self.args.subdir or 'the root'}")
        index.unlink()
        self.publish_card()

    def publish_card(self):
        files = self.api.list_repo_files(self.args.repo, repo_type=REPO_TYPE)
        folders = {}
        for f in sorted(files):
            if f == "index.jsonl" or f.endswith("/index.jsonl"):
                path = self.api.hf_hub_download(self.args.repo, f, repo_type=REPO_TYPE,
                                                force_download=True)
                folders[f.rpartition("/")[0] or "."] = [json.loads(l) for l in open(path) if l.strip()]
        card = self.args.data / ".hf_README.md"
        card.write_text(render_card(self.args.repo, folders))
        self.commit({"README.md": card}, "Dataset card")
        card.unlink()


def render_card(repo, folders):
    def values(rows, key):
        return ", ".join(sorted({str(r.get(key, "?")) for r in rows}))

    total = sum(r.get("positions", 0) for rows in folders.values() for r in rows)
    lines = ["---", "tags: [chess, nnue, hydray, bulletformat]", f"pretty_name: {repo.split('/')[-1]}",
             "---", "", f"# {repo}", "",
             "HydraY self-play positions in bulletformat (32 bytes per record), labeled by "
             "an NNUE search. Each folder is one machine; each shard `<prefix>.sNNNN.bin` "
             "has its manifest `<prefix>.sNNNN.json` with the settings it was generated with, "
             "and `index.jsonl` lists the manifests of the folder.", "",
             f"**{total:,} positions** in {sum(len(r) for r in folders.values())} shards.", "",
             "| folder | machine | shards | positions | nodes/move | start | eg-every | syzygy | net | build | sealed |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for folder, rows in folders.items():
        sealed = sorted(r.get("sealed_utc", "") for r in rows)
        lines.append(f"| {folder} | {values(rows, 'machine')} | {len(rows)} | "
                     f"{sum(r.get('positions', 0) for r in rows):,} | {values(rows, 'nodes_per_move')} | "
                     f"{values(rows, 'start')} | {values(rows, 'eg_every')} | {values(rows, 'syzygy_men')} | "
                     f"{values(rows, 'net_fnv1a64')} | {values(rows, 'build')} | "
                     f"{sealed[0][:10] if sealed else ''} - {sealed[-1][:10] if sealed else ''} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True, help="directory with the shards")
    ap.add_argument("--repo", required=True, help="dataset id, user/name")
    ap.add_argument("--prefix", default="", help="only the shards of this datagen --out name")
    ap.add_argument("--subdir", default="", help="folder in the dataset (default: its root)")
    ap.add_argument("--owner", default=default_owner(),
                    help="machine id written to <subdir>/OWNER (default: from /etc/machine-id)")
    ap.add_argument("--host", default=socket.gethostname(), help="host name written next to the owner")
    ap.add_argument("--every", type=int, default=600, help="seconds between passes")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--delete-after-upload", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list, upload nothing, no token needed")
    ap.add_argument("--check", action="store_true", help="token, dataset and OWNER only, then exit")
    ap.add_argument("--next-shard", action="store_true",
                    help="with --check: print the first shard index free in the dataset")
    args = ap.parse_args()
    args.subdir = args.subdir.strip("/")
    if not args.data.is_dir():
        sys.exit(f"hf_upload: {args.data} is not a directory")

    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)
    api = HfApi()
    up = Uploader(api, args)
    if not args.dry_run:
        user = api.whoami()["name"]  # fails fast on a missing or bad token
        api.create_repo(args.repo, repo_type=REPO_TYPE, private=True, exist_ok=True)
        up.check_owner()
        if args.check:
            print(f"hf_upload: datasets/{args.repo}/{args.subdir} ready for {args.owner} ({user})",
                  file=sys.stderr)
            if args.next_shard:
                print(up.next_shard())
            return
        print(f"hf_upload: {args.data} -> datasets/{args.repo}/{args.subdir} as {user}, "
              f"owner {args.owner}, {'one pass' if args.once else f'every {args.every}s'}"
              f"{', deleting after upload' if args.delete_after_upload else ''}", flush=True)

    while True:
        try:
            up.one_pass()
        except Exception as e:  # network or Hub errors: the next pass retries
            print(f"hf_upload: pass failed: {type(e).__name__}: {e}", flush=True)
        if args.once or stopping:
            return
        for _ in range(args.every):
            if stopping:
                return
            time.sleep(1)


if __name__ == "__main__":
    main()
