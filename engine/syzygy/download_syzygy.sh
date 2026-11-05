#!/bin/sh
# Downloads the Syzygy 3-4-5 tables (~1 GB) into engine/syzygy/files, each
# checked by SHA-256. Interrupted, it resumes where it stopped; run again, it
# only checks the files and replaces any that is missing or damaged.
dir=$(dirname "$(realpath "$0")")
exec python3 "$dir/fetch_syzygy.py" --dir "$dir/files" --sums "$dir/syzygy-3-4-5.sha256" "$@"
