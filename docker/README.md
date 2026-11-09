# HydraY datagen in Docker

The image runs `./chess datagen` on any machine with Docker and an AVX2 + BMI2
CPU, and can upload the sealed shards to a Hugging Face dataset as they come.
The datagen itself (flags, shards, manifests) is described in the main
README, "Generating training data".

## Requirements

- A Linux host: the prefix and the dataset owner come from `/etc/machine-id`.
  On macOS or Windows remove the `/host/machine-id` mount from `compose.yaml`
  and give `--out=/data/NAME` (and `--hf-subdir`) by hand.
- Docker with Compose 2.24 or newer (`docker compose version`), for the
  optional `.env`.
- An x86-64 CPU with AVX2 + BMI2. The image holds two builds: AVX-VNNI
  (Intel 12th gen and later, AMD Zen 5, ~30% faster) and plain AVX2 + BMI2;
  at the start the container tries the first and, where it does not run,
  says `the ... build does not run on this CPU, using ...` and takes the
  second. `ARCH` and `FALLBACK_ARCH` in `compose.yaml` choose them.
- About 1 GB of disk and an internet connection at the first start, for the
  Syzygy tables (next section).

## Tablebases

Every shard is made with the Syzygy 3-4-5 tables: games under 6 pieces end
with the exact result, and the search scores endgames with them. At every
start the container checks the 290 files in `engine/syzygy/files` against
the SHA-256 list `engine/syzygy/syzygy-3-4-5.sha256` (under a second) and
fetches from Lichess (HTTPS) any that is missing or damaged, ~1 GB the first
time: the log says `fetch_syzygy: fetching ...`, then `N/290`. A fetch that
stops halfway resumes at the next start, and the files serve the engine on
the host too.

`--no-tb` in `command:` skips them and plays the endgames out: only for
tests, since those shards label endgames worse.

## Quick start

From the repository root:

```sh
mkdir -p data engine/syzygy/files      # before the first start, see below
id -u; id -g                           # into user: in compose.yaml if not 1000:1000
cp docker/.env.example docker/.env     # HF_TOKEN=..., only for the upload
docker compose -f docker/compose.yaml up -d --build
docker compose -f docker/compose.yaml logs -f      # progress, seals, uploads
docker compose -f docker/compose.yaml down         # stop: shard sealed and uploaded
```

The shards land in `data/` at the repository root. The service restarts by
itself after a crash or a reboot and resumes where it stopped; `down` stops it
for good.

`data/` and `engine/syzygy/files` must exist and belong to the `user:` of
`compose.yaml` before the first start: a missing folder is created by Docker
as root, and the container cannot write it. The container then stops at once
with `... is not writable by uid ...`; the fix is
`sudo chown "$(id -u):$(id -g)" data engine/syzygy/files` or the right
`user:`. A folder with all the tables already in it may be read-only.

The datagen uses every CPU of the container; `--threads=N` in `command:`, or
`cpus:` in the service, leaves room for other work.

## Updating

```sh
git pull
docker compose -f docker/compose.yaml up -d --build
```

The rebuilt image stops the old container (shard sealed and uploaded) and
starts the new one. A new engine version, or another `--nodes`, writes under a
new prefix; the old shards stay where they are.

## Settings

Everything is in `compose.yaml`; `.env`, ignored by git, holds only the token.

| Where | What |
|---|---|
| `command:` | the `./chess datagen` flags (`--nodes`, `--threads`, `--start`, ...) and the upload flags below |
| `user:` | owner of the files written: `id -u`:`id -g` |
| `volumes:` | `data/`, the tablebases, the host's machine-id |
| `build.args` | `ARCH` (ISA, as in the makefile), `FALLBACK_ARCH` (the build for CPUs where `ARCH` does not run, `""` for none), `JOBS` (compile jobs), `GIT_COMMIT` |
| `.env` | `HF_TOKEN`, a Hugging Face write token |

The container puts its defaults before `command:`, and a later flag wins:
`--threads` = the CPUs it may use (`--cpus` included), `--nodes=8000` (the
default of `./chess datagen`, which the prefix names), and `--out`,
`--machine` and `--build` as below. `command:` adds `--seal-on-exit` and
`--min-free-gb=4`.

`--games` does not fit a service that restarts when it ends: for a fixed batch
use `docker compose -f docker/compose.yaml run --rm datagen --games=1000` (the
arguments of `run` replace `command:`).

## Names and folders

The prefix needs no setting: `m` plus 8 hex digits of the hash of the host's
`/etc/machine-id` (stable across restarts, different on every machine), the
engine version and the nodes per move (`--nodes`):

```txt
data/m956ee98b-v4.3.1-n8000.s0000.bin    sealed shard
data/m956ee98b-v4.3.1-n8000.s0000.json   its manifest
data/m956ee98b-v4.3.1-n8000.s0001.bin.part   the shard being written
```

A new engine version or another `--nodes` starts new names, and every new net
comes with a new version, so a series never mixes two labeler nets or two node
counts: an 8000-node and a 24000-node run of the same machine are two series,
side by side in its folder. Series made before the nodes were in the name
(`m956ee98b-v4.3.1.s0000.bin`) keep their names. Smaller changes without a
version bump (a retuned search) are told apart by the manifests, which hold
the machine hash and the build: version, a hash of the sources and net, the ISA,
and the commit when the build gets `GIT_COMMIT=$(git rev-parse --short HEAD)`
in the environment. `--out=/data/NAME` overrides the prefix.

## Upload to Hugging Face

The shards go to the public dataset
[ThomasGhione/hydray](https://huggingface.co/datasets/ThomasGhione/hydray),
set in `command:` with `--delete-after-upload` (a shard leaves the disk once
the Hub holds it). The token goes in `docker/.env` as `HF_TOKEN=hf_...`, one
with write access to that dataset (ask for it); without a token or a
connection comment the two flags out, and the shards stay in `data/`.
Everything uploaded can be read by anyone: shards, manifests, index and card.
They name a machine only by its hash (`m956ee98b`), never by its host name.
Another dataset: `--hf-repo=USER/NAME` (`ORG/NAME` for an organization), created
public if missing; a private one counts against the account's quota (100 GB
on a free account, ~60 shards). Upload
flags: `--hf-repo`, `--hf-subdir` (default: the machine hash), `--hf-every`
(seconds between passes, default 600), `--delete-after-upload`.

- **What goes up.** Every sealed shard with its manifest, in one commit, to
  the machine's folder (`m956ee98b/`, every version inside). The open `.part`
  never does.
- **Checked.** An upload counts only once the Hub reports the local SHA-256;
  only then is the local shard deleted. The manifest stays (a few hundred
  bytes).
- **Never overwritten.** A name already on the Hub with other content is
  reported as `CONFLICT` and left alone on both sides.
- **Numbering.** At the start the datagen continues after the last shard of
  its prefix in the dataset, so even an emptied `data/` never reuses a name.
- **One owner per folder.** `OWNER` names the machine that writes the folder:
  a second machine on it is refused at the start, before it generates
  anything. Run one datagen per machine; a second one, from another `data/`,
  needs its own `--out` and `--hf-subdir`.
- **Index and card.** After each upload `index.jsonl` lists the folder's
  manifests, and `README.md`, the dataset card, sums up every folder: machine,
  shards, positions, nodes, start, net, build, dates.
- **Start and stop.** Token, dataset and owner are checked before the datagen
  starts. At the stop a last pass uploads the shard sealed at exit
  (`stop_grace_period` gives it 10 minutes); anything left goes up at the next
  start.

### First run with a new token

Before the long run, a short one on a throwaway dataset checks token, upload
and deletion end to end in a couple of minutes:

```sh
docker compose -f docker/compose.yaml run --rm datagen --games=200 --seal-on-exit \
    --shard=2000 --out=/data/test --hf-repo=USER/hydray-test \
    --hf-subdir=test --hf-every=30 --delete-after-upload
```

The log must show `sealed`, `uploaded ... (SHA-256 checked)` and `deleted`; the dataset must hold
`test/` with shards, manifests, `index.jsonl` and `OWNER`, and `data/` only
the manifests. Then delete the test dataset on the Hub and `data/test*`.

### Downloading for training

```sh
pip install huggingface_hub
hf download ThomasGhione/hydray --repo-type dataset --local-dir ~/hydray-data
```

`--include "m956ee98b/*-v4.3.1.*"` takes one machine and version only;
`index.jsonl` and the manifests tell which shards were made with which net
and settings.

## Disk space

Below `--min-free-gb` free (4 GB in `compose.yaml`) the datagen finishes its
games and pauses, and resumes past the threshold plus 1 GB; the progress line
says `PAUSED`. With `--delete-after-upload` this sorts itself out: the 4 GB
leave the uploader room, and every uploaded shard it deletes frees 1.6 GB.
Without it the datagen stays paused until space is freed by hand. A full disk
never corrupts a shard, and the container log is capped at 30 MB.

## Machines without internet

Three things need the network: building the image, the tablebases and the
upload. Each one is done on a connected machine and carried over (USB disk,
`scp` on the local network).

On a connected machine, from the repository root, with the tables already in
`engine/syzygy/files` (`engine/syzygy/download_syzygy.sh`):

```sh
docker compose -f docker/compose.yaml build
docker save hydray-datagen | gzip > hydray-datagen.tar.gz     # ~70 MB
```

Copy to the offline machine the repository folder, `engine/syzygy/files`
included, and `hydray-datagen.tar.gz`. There, from the repository root:

```sh
docker load -i hydray-datagen.tar.gz
mkdir -p data
docker compose -f docker/compose.yaml up -d      # no --build: it uses the loaded image
```

`--hf-repo` stays commented out. The tables are checked at the start as
usual, with no network needed when they are all there; if any is missing the
container stops with `... unreachable ...: without internet copy the tables`.

To bring the shards back, copy the sealed ones with their manifests (never
the `.part`) from the offline machine:

```sh
mkdir -p /media/usb/shards
cp data/m*.s[0-9][0-9][0-9][0-9].bin data/m*.s[0-9][0-9][0-9][0-9].json /media/usb/shards/
```

On a connected machine with this repository and the token in `docker/.env`,
put them under `data/` (say `data/offline/`) and upload them:

```sh
docker compose -f docker/compose.yaml run --rm datagen upload /data/offline \
    --hf-repo=ThomasGhione/hydray --delete-after-upload
```

Each shard goes to the folder of the machine that made it (`m<hash>/`, from
its name), with the same checks as the live upload. Once it says `uploaded`,
delete the `.bin` on the offline machine and keep the `.json`: the manifests
are what lets that machine number its next shards past the ones uploaded.

## Plain `docker run`

```sh
docker build -f docker/Dockerfile -t hydray-datagen .
docker run -d --name datagen --init --user "$(id -u):$(id -g)" -e HF_TOKEN \
    -v "$PWD/data:/data" \
    -v "$PWD/engine/syzygy/files:/hydray/engine/syzygy/files" \
    -v /etc/machine-id:/host/machine-id:ro \
    hydray-datagen --nodes 8000 --seal-on-exit \
    --hf-repo ThomasGhione/hydray --delete-after-upload
docker stop -t 600 datagen
docker run --rm -v "$PWD/data:/data" hydray-datagen dump /data/<prefix>.s0000.bin 5
```

Without the machine-id mount the container needs `--out`.

The uploader also runs without Docker:
`nnue/tools/hf_upload.py --data data --repo ThomasGhione/hydray --once` (needs
`pip install huggingface_hub`; `--help` for the rest). Outside Docker
datagen writes the host name in the manifests unless it gets `--machine`.

## Messages

| Message | Meaning |
|---|---|
| `fetch_syzygy: ... unreachable` | no internet and tables missing: copy `engine/syzygy/files` from another machine ("Machines without internet") |
| `... is not writable by uid ...` | `data/` or `engine/syzygy/files` missing when Docker started (so created as root) or owned by another user: `mkdir -p`, `chown`, or fix `user:` |
| `fetch_syzygy: ... (attempt N/5)` | a table download failed and is retried; after 5 the container stops and retries at the restart |
| `PAUSED: ... MB free` | under `--min-free-gb`; it resumes by itself once there is room |
| `CONFLICT ... on the Hub with other content` | a name taken upstream with other data: nothing was uploaded or deleted; check the prefixes |
| `... belongs to mXXXXXXXX ...` | the dataset folder is another machine's: give this one another `--out` and `--hf-subdir` |
| `the upload cannot start` | missing or invalid `HF_TOKEN`, a dataset it cannot write, or an `OWNER` clash |
| `is in use by another datagen` | a second datagen on the same prefix and `data/` |
| `cut a partial record ... off the end` | a crash left half a record; it was removed, the shard is sound |

## Files

| File | Role |
|---|---|
| `compose.yaml` | the service and its settings |
| `.env.example` | template of `.env` (token only) |
| `Dockerfile` | two static builds (`make prod`, `HYDRAY_RELEASE`; `ARCH` and `FALLBACK_ARCH`) into a `python:3.13-slim` image with `huggingface_hub` |
| `Dockerfile.dockerignore` | only the sources reach the build |
| `datagen.sh` | entry point: defaults, upload flags, datagen and uploader side by side |
| `../nnue/tools/hf_upload.py` | the uploader |
| `../engine/syzygy/fetch_syzygy.py` | fetches the tablebases and checks them against `syzygy-3-4-5.sha256` |
