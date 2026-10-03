#!/usr/bin/env bash
set -euo pipefail
h3_release_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$h3_release_dir/../driver.py" source-tests --platform macos "$@"
