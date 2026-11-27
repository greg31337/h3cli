#!/usr/bin/env bash
set -euo pipefail
h3_release_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$h3_release_dir/python.sh"
h3_release_require_python
exec python3 "$h3_release_dir/flow.py" "$@"
