#!/usr/bin/env bash
# Bootstrap can run before Python or a release configuration exists.
set -euo pipefail
h3_prereq_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
h3_prereq_platform=''
h3_prereq_install=false
h3_prereq_dry=false
while (($#)); do
  case "$1" in
    --platform|--config)
      if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
      if [[ "$1" == --platform ]]; then h3_prereq_platform="$2"; fi
      shift 2 ;;
    --install) h3_prereq_install=true; shift ;;
    --dry-run) h3_prereq_dry=true; shift ;;
    --help|-h)
      echo 'Usage: step1-prerequisites.sh [--install] [--dry-run] [--config PATH]'
      echo 'Checks build tools without needing Python or configuration first. --config is accepted but not needed by this step.'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$h3_prereq_platform" == linux || "$h3_prereq_platform" == macos ]] || { echo 'Specify a platform' >&2; exit 2; }
h3_prereq_run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "$h3_prereq_dry" == false ]]; then "$@"; fi
}
if [[ "$h3_prereq_dry" == false ]]; then
  h3_prereq_host="$(uname -s):$(uname -m)"
  [[ "$h3_prereq_platform:$h3_prereq_host" == linux:Linux:x86_64 || "$h3_prereq_platform:$h3_prereq_host" == macos:Darwin:arm64 ]] || { echo 'Wrong host OS/architecture' >&2; exit 2; }
fi
if [[ "$h3_prereq_install" == true ]]; then
  if [[ "$h3_prereq_platform" == linux ]]; then
    h3_prereq_sudo=()
    if ((EUID != 0)); then h3_prereq_sudo=(sudo); fi
    h3_prereq_run "${h3_prereq_sudo[@]}" apt-get update
    h3_prereq_run "${h3_prereq_sudo[@]}" apt-get install -y git ca-certificates python3 skopeo umoci coreutils
  elif [[ "$h3_prereq_dry" == true ]]; then
    echo 'Would install only missing Homebrew helpers (including Python if too old).'
    h3_prereq_run brew install python pkg-config gh minisign
  else
    h3_prereq_missing=()
    if ! command -v python3 >/dev/null || ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
      h3_prereq_missing+=(python)
    fi
    for h3_prereq_tool in pkg-config gh minisign; do
      if ! command -v "$h3_prereq_tool" >/dev/null; then h3_prereq_missing+=("$h3_prereq_tool"); fi
    done
    if ((${#h3_prereq_missing[@]})); then h3_prereq_run brew install "${h3_prereq_missing[@]}"; fi
    hash -r
  fi
fi
if [[ "$h3_prereq_dry" == false ]]; then
  command -v python3 >/dev/null || { echo 'Python missing; run this step with --install first' >&2; exit 2; }
  python3 -c 'import sys; sys.exit("Python 3.10+ is required" if sys.version_info < (3, 10) else 0)'
  h3_prereq_tools=(git)
  if [[ "$h3_prereq_platform" == linux ]]; then h3_prereq_tools+=(skopeo umoci cp); else h3_prereq_tools+=(pkg-config xcrun); fi
  for h3_prereq_tool in "${h3_prereq_tools[@]}"; do
    command -v "$h3_prereq_tool" >/dev/null || { echo "Missing $h3_prereq_tool; run with --install (Apple tools must be installed separately)" >&2; exit 2; }
  done
fi
h3_prereq_run python3 --version
if [[ "$h3_prereq_platform" == macos ]]; then
  h3_prereq_run bash "$h3_prereq_root/scripts/build_macos.sh" --doctor
fi
