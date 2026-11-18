#!/bin/sh
set -eu

sample_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -n "${H3CLI_BIN:-}" ]; then
  sample_binary=$H3CLI_BIN
elif [ -x "$sample_dir/../../bin/h3cli" ]; then
  sample_binary="$sample_dir/../../bin/h3cli"
else
  sample_binary=h3cli
fi

exec "$sample_binary" --render "$sample_dir/flight.txt" \
  --output-dir outputs/flight-render "$@"
