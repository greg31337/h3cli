#!/bin/bash
# Prepare an Apple Silicon Mac for this repository's Metal build.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/setup_macos.sh [--dry-run]

Install Homebrew (if needed), FFmpeg, and Python for an Apple Silicon macOS
26+ build. Requires Apple's Command Line Tools / SDK 26+. If the tools are
missing, opens Apple's installer; finish it and rerun this script.

For standalone packaging, use scripts/build_macos.sh --doctor instead.
A packaged executable needs none of these source-build prerequisites at runtime.

Run as your normal user, not with sudo. Homebrew may request administrator
access. Does not build the project, download models, or edit shell profiles.
  --dry-run  Print the setup commands without changes or host validation.
  --help     Show this help.
EOF
}

dry_run=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=1 ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$arg" >&2; usage >&2; exit 2 ;;
    esac
done

fail() { printf 'setup_macos: %s\n' "$*" >&2; exit 1; }
run() {
    printf '+'; printf ' %q' "$@"; printf '\n'
    if (( ! dry_run )); then "$@"; fi
}

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
env_file="$repo_dir/outputs/setup/macos-env.sh"
temp_dir=''
trap 'if [[ -n "$temp_dir" ]]; then rm -rf -- "$temp_dir"; fi' EXIT

if (( ! dry_run )); then
    [[ $(uname -s) == Darwin && $(uname -m) == arm64 ]] ||
        fail 'Requires Apple Silicon macOS; use an ARM-native Terminal (no Rosetta).'
    [[ $(id -u) != 0 ]] || fail 'Run as your normal user, without sudo.'
    os_version=$(sw_vers -productVersion)
    [[ ${os_version%%.*} -ge 26 ]] || fail 'Requires macOS 26 or newer.'
    if ! xcode-select -p >/dev/null 2>&1; then
        xcode-select --install || true
        fail "Finish Apple's Command Line Tools installer, then rerun this script."
    fi
    sdk_version=$(xcrun --sdk macosx --show-sdk-version) ||
        fail 'Cannot find the macOS SDK. Check xcode-select -p and your Command Line Tools installation.'
    [[ ${sdk_version%%.*} -ge 26 ]] ||
        fail 'Requires SDK 26+. Update the Command Line Tools or select a newer Xcode using xcode-select.'
    run xcrun clang --version
else
    printf 'Dry run for Apple Silicon macOS 26+; requires Command Line Tools / SDK 26+.\n'
fi

brew_bin=/opt/homebrew/bin/brew
if (( dry_run )); then
    printf 'If Homebrew is missing:\n'
    installer='<temporary-directory>/homebrew-install.sh'
elif command -v brew >/dev/null 2>&1; then
    brew_bin=$(command -v brew)
elif [[ ! -x "$brew_bin" ]]; then
    temp_dir=$(mktemp -d)
    installer="$temp_dir/homebrew-install.sh"
fi
if [[ -n ${installer:-} ]]; then
    run curl -fsSL --retry 3 https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh -o "$installer"
    run /bin/bash "$installer"
fi
if (( ! dry_run )); then
    [[ -x "$brew_bin" ]] || fail "Homebrew was not installed at $brew_bin."
    brew_env=$("$brew_bin" shellenv)
    eval "$brew_env"
fi
run "$brew_bin" install x264 pkg-config python
run python3 "$repo_dir/scripts/setup_ffmpeg.py"

if (( ! dry_run )); then
    run make --version
    run git --version
    run "$repo_dir/outputs/setup/ffmpeg/bin/ffmpeg" -version
    run "$repo_dir/outputs/setup/ffmpeg/bin/ffprobe" -version
    run python3 -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ is required"; print(sys.version)'
    mkdir -p "$(dirname "$env_file")"
    {
        printf 'eval "$(%q shellenv)"\n' "$brew_bin"
        printf 'export PATH=%q":$PATH"\n' "$repo_dir/outputs/setup/ffmpeg/bin"
        printf 'export H3_FFMPEG=%q\nexport H3_SGLANG_INPUT_FFMPEG=%q\nexport H3_FFPROBE=%q\n' \
            "$repo_dir/outputs/setup/ffmpeg/bin/ffmpeg" "$repo_dir/outputs/setup/ffmpeg/bin/ffmpeg" "$repo_dir/outputs/setup/ffmpeg/bin/ffprobe"
    } > "$env_file"
else
    printf 'Would verify Make, Git, FFmpeg, FFprobe, and Python 3.11+, then write %s\n' "$env_file"
fi
printf '\n%s From the repository root, run:\n' "$(if (( dry_run )); then printf 'After setup completes:'; else printf 'Setup complete.'; fi)"
printf '  source %q\n' "$env_file"
printf '  make -j8\n  ./bin/h3cli --help\n  make test\n'
