#!/usr/bin/env bash
set -euo pipefail
h3_release_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$h3_release_dir/../prerequisites.sh" --platform macos "$@"
