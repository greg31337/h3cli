#!/bin/sh
set -eu

sample_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -z "${H3CLI_BIN:-}" ] && [ -x "$sample_dir/../../bin/h3cli" ]; then
  H3CLI_BIN="$sample_dir/../../bin/h3cli"
  export H3CLI_BIN
fi

exec python3 "$sample_dir/test.py" "$sample_dir/panda.txt" "$sample_dir/panda.png" \
  --steps 50 --no-overlay -o panda.mp4 "$@"
