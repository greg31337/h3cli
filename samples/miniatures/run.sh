#!/bin/sh
set -eu

sample_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$sample_dir/test.py" "$sample_dir/miniatures.txt" \
  --steps 50 --upscale-refine-steps 4 --no-overlay -o miniatures.mp4 "$@"
