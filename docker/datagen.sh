#!/bin/sh
# Container entry point. The arguments are those of ./chess datagen (see
# --help), after the container's defaults: --out /data/hydray, --threads = the
# CPUs the container may use. A later flag wins, so both can be overridden.
#
# Upload of the sealed shards to a Hugging Face dataset (HF_TOKEN in the
# environment), handled here and not passed to datagen:
#   --hf-repo USER/NAME     turns the upload on
#   --hf-subdir DIR         folder in the dataset (default: the --out name)
#   --hf-every SECONDS      time between passes (default 600)
#   --delete-after-upload   remove a shard once the Hub holds it
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

# Splits the upload flags off and rebuilds the rest as datagen's arguments,
# noting --out to know where the shards land.
out=/data/hydray repo= subdir= every=600 delete=
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
        --hf-*) echo "datagen.sh: unknown option $a" >&2; exit 1 ;;
        --out) out=$1; set -- "$@" "$a" "$1"; shift; n=$((n - 1)) ;;
        --out=*) out=${a#*=}; set -- "$@" "$a" ;;
        *) set -- "$@" "$a" ;;
    esac
done
set -- --out /data/hydray --threads "$cpus" "$@"
case $out in
    /data/*) ;;
    *) echo "datagen.sh: warning: --out $out is outside the /data volume" >&2 ;;
esac

if [ -z "$repo" ]; then
    exec ./chess datagen "$@"
fi

upload() {
    python3 /hydray/hf_upload.py --data "$(dirname "$out")" --repo "$repo" \
        --prefix "$(basename "$out")" --subdir "${subdir:-$(basename "$out")}" $delete "$@"
}
# Fails here, before hours of datagen, on a bad token or repo.
python3 - <<'EOF' || exit 1
import sys
from huggingface_hub import HfApi
try:
    HfApi().whoami()
except Exception as e:
    sys.exit(f"datagen.sh: Hugging Face login failed (HF_TOKEN?): {type(e).__name__}")
EOF

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
