#!/bin/sh
# Build tools are needed only here; the resulting executable needs none of them.
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$script_dir/linux/build.py" "$@"
