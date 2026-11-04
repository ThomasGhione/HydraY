#!/bin/sh
# Container entry point. The arguments are those of ./chess datagen (see
# --help), after the container's defaults; a later flag wins, so each can be
# overridden:
#   --out /data/m<8 hex>-v<version>
#                          the hash of the host's /etc/machine-id, mounted at
#                          /host/machine-id (one stable prefix per machine),
#                          and the engine version: a new version, new names
#   --threads              the CPUs the container may use
#   --machine, --build     the host name (/host/hostname) and the image's build
#
#   --tb                   the Syzygy 3-4-5 tables, mounted at
#                          /hydray/engine/syzygy/files: missing or damaged
#                          ones are fetched and checked before the start
#   --no-tb                plays the endgames out instead (tests only: such
#                          shards label endgames worse)
#
# Upload of the sealed shards to a Hugging Face dataset (HF_TOKEN in the
# environment), handled here and not passed to datagen:
#   --hf-repo USER/NAME     turns the upload on
#   --hf-subdir DIR         folder in the dataset (default: the machine hash,
#                           else the --out name)
#   --hf-every SECONDS      time between passes (default 600)
#   --delete-after-upload   remove a shard once the Hub holds it
# The dataset folder belongs to the machine (OWNER, the machine-id hash): a
# second machine with the same prefix is refused instead of mixing shards.
#
#   docker run ... hydray-datagen dump /data/<file.bin> [count]
set -e

if [ "$1" = "dump" ]; then
    shift
    exec ./chess datagen-dump "$@"
fi

# nproc sees --cpuset-cpus but not --cpus, which is a cgroup quota.
cpus=$(nproc)
if read -r quota period 2>/dev/null < /sys/fs/cgroup/cpu.max && [ "$quota" != max ]; then
    quota_cpus=$(( (quota + period - 1) / period ))
    [ "$quota_cpus" -lt "$cpus" ] && cpus=$quota_cpus
fi
[ "$cpus" -gt 32 ] && cpus=32

# Never the container's own machine-id: every container of the image shares it.
machine_hash=
if [ -s /host/machine-id ]; then
    machine_hash=m$(tr -d '[:space:]' < /host/machine-id | sha256sum | cut -c1-8)
fi
host=$(cat /host/hostname 2>/dev/null || true)
build=$(cat /hydray/BUILD_ID)
version=${build%% *}
version=${version#hydray-}

# Splits the upload flags off and rebuilds the rest as datagen's arguments,
# noting --out to know where the shards land.
out= repo= subdir= every=600 delete= notb=
n=$#
while [ "$n" -gt 0 ]; do
    a=$1; shift; n=$((n - 1))
    case $a in
        --hf-repo|--hf-subdir|--hf-every)
            [ "$n" -gt 0 ] || { echo "datagen.sh: $a needs a value" >&2; exit 1; }
            v=$1; shift; n=$((n - 1)) ;;
        --hf-*=*) v=${a#*=}; a=${a%%=*} ;;
    esac
    case $a in
        --hf-repo) repo=$v ;;
        --hf-subdir) subdir=$v ;;
        --hf-every) every=$v ;;
        --delete-after-upload) delete=--delete-after-upload ;;
        --no-tb) notb=1 ;;
        --hf-*) echo "datagen.sh: unknown option $a" >&2; exit 1 ;;
        --out) out=$1; set -- "$@" "$a" "$1"; shift; n=$((n - 1)) ;;
        --out=*) out=${a#*=}; set -- "$@" "$a" ;;
        *) set -- "$@" "$a" ;;
    esac
done
if [ -z "$out" ]; then
    [ -n "$machine_hash" ] || {
        echo "datagen.sh: mount the host's /etc/machine-id at /host/machine-id, or give --out" >&2
        exit 1
    }
    out=/data/$machine_hash-v$version
fi
set -- --out "$out" --threads "$cpus" --machine "${host:-${machine_hash:-unknown}}" \
    --build "$build" "$@"
case $out in
    /data/*) ;;
    *) echo "datagen.sh: warning: --out $out is outside the /data volume" >&2 ;;
esac
# The usual first-run trap: a data/ that Docker created as root, or a user:
# that is not the owner of data/.
outdir=$(dirname "$out")
if ! { mkdir -p "$outdir" && [ -w "$outdir" ]; } 2>/dev/null; then
    echo "datagen.sh: $outdir is not writable by uid $(id -u): create data/ before the" \
         "first start (mkdir -p data) and set user: in compose.yaml to \`id -u\`:\`id -g\`" >&2
    exit 1
fi

# Shards labelled without tables would mix silently with the others.
tb=/hydray/engine/syzygy/files
sums=/hydray/syzygy-3-4-5.sha256
if [ -n "$notb" ]; then
    echo "datagen.sh: --no-tb: no tablebases, endgames played out" >&2
    set -- "$@" --tb /nonexistent
elif { mkdir -p "$tb" && [ -w "$tb" ]; } 2>/dev/null; then
    python3 /hydray/fetch_syzygy.py --dir "$tb" --sums "$sums" || exit 1
elif ! python3 /hydray/fetch_syzygy.py --dir "$tb" --sums "$sums" --check; then
    echo "datagen.sh: engine/syzygy/files is not writable by uid $(id -u), so the" \
         "tables cannot be fetched: mkdir -p engine/syzygy/files before the first start," \
         "or chown it to \`id -u\`:\`id -g\`" >&2
    exit 1
fi

if [ -z "$repo" ]; then
    exec ./chess datagen "$@"
fi

upload() {
    python3 /hydray/hf_upload.py --data "$(dirname "$out")" --repo "$repo" \
        --prefix "$(basename "$out")" --subdir "${subdir:-${machine_hash:-$(basename "$out")}}" \
        --owner "${machine_hash:-$(basename "$out")}" --host "${host:-unknown}" $delete "$@"
}
# Fails here, before hours of datagen, on a bad token, repo or OWNER. The
# first free shard number upstream keeps a wiped /data from reusing names.
first=$(upload --check --next-shard) \
    || { echo "datagen.sh: the upload cannot start (HF_TOKEN? dataset? OWNER?)" >&2; exit 1; }
set -- "$@" --first-shard "$first"

upload --every "$every" &
uploader=$!
./chess datagen "$@" &
gen=$!
trap 'kill -TERM "$gen" "$uploader" 2>/dev/null' TERM INT

# A trapped signal interrupts wait: keep waiting until datagen has exited.
status=0
wait "$gen" || status=$?
while kill -0 "$gen" 2>/dev/null; do
    status=0
    wait "$gen" || status=$?
done
kill -TERM "$uploader" 2>/dev/null || true
wait "$uploader" || true
# Shards sealed since the last pass, the one sealed at exit included.
upload --once || true
exit "$status"
