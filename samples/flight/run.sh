#!/bin/sh
set -eu

sample_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$sample_dir/test.py" "$sample_dir/flight.txt" \
  --steps 50 --no-overlay -o flight.mp4 "$@"
