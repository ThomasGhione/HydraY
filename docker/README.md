# HydraY datagen in Docker

The image runs `./chess datagen` on any machine with Docker and an AVX2 + BMI2
CPU, and can upload the sealed shards to a Hugging Face dataset as they come.
The datagen itself (flags, shards, manifests) is described in the main
README, "Generating training data".

## Quick start

From the repository root:

```sh
cp docker/.env.example docker/.env     # HF_TOKEN=..., only for the upload
docker compose -f docker/compose.yaml up -d --build
docker compose -f docker/compose.yaml logs -f      # progress, seals, uploads
docker compose -f docker/compose.yaml down         # stop: shard sealed and uploaded
```

The shards land in `data/` at the repository root. The service restarts by
itself after a crash or a reboot and resumes where it stopped; `down` stops it
for good.

## Settings

Everything is in `compose.yaml`; `.env`, ignored by git, holds only the token.

| Where | What |
|---|---|
| `command:` | the `./chess datagen` flags (`--nodes`, `--threads`, `--start`, ...) and the upload flags below |
| `user:` | owner of the files written: `id -u`:`id -g` |
| `volumes:` | `data/`, the optional tablebases, the host's machine-id and host name |
| `build.args` | `ARCH` (ISA, as in the makefile), `JOBS` (compile jobs), `GIT_COMMIT` |
| `.env` | `HF_TOKEN`, a Hugging Face write token |

The container puts its defaults before `command:`, and a later flag wins:
`--threads` = the CPUs it may use (`--cpus` included), and `--out`,
`--machine` and `--build` as below. `command:` adds `--seal-on-exit` and
`--min-free-gb=4`.

`--games` does not fit a service that restarts when it ends: for a fixed batch
use `docker compose -f docker/compose.yaml run --rm datagen --games=1000` (the
arguments of `run` replace `command:`).

## Names and folders

The prefix needs no setting: `m` plus 8 hex digits of the hash of the host's
`/etc/machine-id` (stable across restarts, different on every machine) and the
engine version:

```txt
data/m956ee98b-v4.3.1.s0000.bin    sealed shard
data/m956ee98b-v4.3.1.s0000.json   its manifest
data/m956ee98b-v4.3.1.s0001.bin.part   the shard being written
```

A new engine version starts new names, and every new net comes with a new
version, so a series never mixes two labeler nets. Smaller changes without a
version bump (a retuned search) are told apart by the manifests, which hold
the host name and the build: version, a hash of the sources and net, the ISA,
and the commit when the build gets `GIT_COMMIT=$(git rev-parse --short HEAD)`
in the environment. `--out=/data/NAME` overrides the prefix.

## Upload to Hugging Face

Uncomment `--hf-repo=USER/NAME` in `command:` (the dataset is created private
if missing) and, to free the disk as it goes, `--delete-after-upload`. Upload
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

## Disk space

Below `--min-free-gb` free (4 GB in `compose.yaml`) the datagen finishes its
games and pauses, and resumes past the threshold plus 1 GB; the progress line
says `PAUSED`. With `--delete-after-upload` this sorts itself out: the 4 GB
leave the uploader room, and every uploaded shard it deletes frees 1.6 GB.
Without it the datagen stays paused until space is freed by hand. A full disk
never corrupts a shard, and the container log is capped at 30 MB.

## Plain `docker run`

```sh
docker build -f docker/Dockerfile -t hydray-datagen .
docker run -d --name datagen --init --user "$(id -u):$(id -g)" -e HF_TOKEN \
    -v "$PWD/data:/data" \
    -v "$PWD/engine/syzygy/files:/hydray/engine/syzygy/files:ro" \
    -v /etc/machine-id:/host/machine-id:ro -v /etc/hostname:/host/hostname:ro \
    hydray-datagen --nodes 8000 --seal-on-exit \
    --hf-repo user/hydray-data --delete-after-upload
docker stop -t 600 datagen
docker run --rm -v "$PWD/data:/data" hydray-datagen dump /data/<prefix>.s0000.bin 5
```

Without the machine-id mount the container needs `--out`.

The uploader also runs without Docker:
`nnue/tools/hf_upload.py --data data --repo user/hydray-data --once` (needs
`pip install huggingface_hub`; `--help` for the rest).

## Messages

| Message | Meaning |
|---|---|
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
| `Dockerfile` | static build (`make prod`, `HYDRAY_RELEASE`) into a `python:3.13-slim` image with `huggingface_hub` |
| `Dockerfile.dockerignore` | only the sources reach the build |
| `datagen.sh` | entry point: defaults, upload flags, datagen and uploader side by side |
| `../nnue/tools/hf_upload.py` | the uploader |
