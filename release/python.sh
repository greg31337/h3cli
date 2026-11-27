#!/usr/bin/env bash
# Sourced by release entry points before Python is needed.
h3_release_use_python() {
  local h3_python_candidate="$1" h3_python_directory
  if [[ -x "$h3_python_candidate" ]] &&
      "$h3_python_candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' >/dev/null 2>&1; then
    h3_python_directory="$(cd -- "$(dirname -- "$h3_python_candidate")" && pwd)" || return 1
    # Workers and build scripts also invoke python3 by name. Keep them on the
    # same interpreter even when they change working directories.
    # If this is already the selected interpreter, keep earlier directories
    # containing other qualified helpers in place.
    if [[ "$(command -v python3)" != "$h3_python_directory/python3" ]]; then
      export PATH="$h3_python_directory:$PATH"
    fi
    hash -r
    return 0
  fi
  return 1
}

h3_release_select_python() {
  local h3_python_candidate h3_python_prefix
  # Check every PATH entry: Apple's older Python can shadow Homebrew's Python.
  while IFS= read -r h3_python_candidate; do
    if h3_release_use_python "$h3_python_candidate"; then return 0; fi
  done < <(type -aP python3 || true)
  # A newly installed Homebrew Python may not be on PATH at all yet.
  if [[ "$(uname -s)" == Darwin ]] && command -v brew >/dev/null 2>&1; then
    h3_python_prefix="$(brew --prefix)" || return 1
    h3_release_use_python "$h3_python_prefix/bin/python3" && return 0
  fi
  return 1
}

h3_release_require_python() {
  h3_release_select_python && return 0
  echo 'Python 3.10+ is required; no compatible python3 was found on PATH or in Homebrew.' >&2
  echo 'Run ./release/macos/step1-prerequisites.sh --install on macOS, or ./release/linux/step1-prerequisites.sh --install on Linux, then retry.' >&2
  return 2
}
