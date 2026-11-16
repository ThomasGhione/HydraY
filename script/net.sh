#!/usr/bin/env bash
# The embedded net lives in the HF bucket ThomasGhione/hydray-net as <sha256>.nnue;
# git tracks only nnue/net/hydray.nnue.sha256 (the sha of the shipped net).
#   script/net.sh fetch          make nnue/net/hydray.nnue match the pointer (downloads if needed)
#   script/net.sh push <file>    upload <file> to the bucket and repoint the pointer to it
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
net=$root/nnue/net/hydray.nnue
ptr=$net.sha256
bucket=ThomasGhione/hydray-net

sha_of() { sha256sum "$1" | cut -d' ' -f1; }

case "${1:-}" in
fetch)
    want=$(tr -d '[:space:]' < "$ptr")
    if [[ -f $net && $(sha_of "$net") == "$want" ]]; then
        touch "$net"   # keeps make from re-running this rule
        exit 0
    fi
    echo "Downloading net $want from hf://buckets/$bucket" >&2
    tmp=$(mktemp "$net.XXXXXX")
    trap 'rm -f "$tmp"' EXIT
    curl -fsSL --retry 3 -o "$tmp" "https://huggingface.co/buckets/$bucket/resolve/$want.nnue"
    got=$(sha_of "$tmp")
    [[ $got == "$want" ]] || { echo "sha256 mismatch: got $got, want $want" >&2; exit 1; }
    mv "$tmp" "$net"
    ;;
push)
    src=${2:?usage: script/net.sh push <file.nnue>}
    [[ -f $root/.env && -z ${HF_TOKEN:-} ]] && { set -a; . "$root/.env"; set +a; }
    [[ -n ${HF_TOKEN:-} ]] || { echo "HF_TOKEN not set (env or .env)" >&2; exit 1; }
    command -v hf >/dev/null || { echo "hf CLI missing: pip install -U huggingface_hub" >&2; exit 1; }
    sha=$(sha_of "$src")
    hf buckets cp "$src" "hf://buckets/$bucket/$sha.nnue"
    echo "$sha" > "$ptr"
    [[ $src -ef $net ]] || cp "$src" "$net"
    echo "Uploaded $sha. Commit nnue/net/hydray.nnue.sha256 to ship it." >&2
    ;;
*)
    echo "usage: script/net.sh fetch | push <file.nnue>" >&2
    exit 2
    ;;
esac
