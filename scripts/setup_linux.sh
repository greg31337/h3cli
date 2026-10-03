#!/bin/bash
# Install the qualified native CUDA toolchain/runtime and build h3cli.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: scripts/setup_linux.sh [--no-build] [--install-driver] [--dry-run]

Prepare Ubuntu 24.04 x86-64 and build bin/h3cli and bin/libh3.a.
Compile with CUDA 13.0 Update 3; use pinned CUDA 13 cuBLAS/NVRTC and
cuDNN 9.20 for both encoding and decoding in the main process.
GPU execution requires an NVIDIA driver >=580.126.20 supporting the GPU.
Run as root or a user with sudo. Models are not downloaded.

  --no-build        Install and verify dependencies; leave compilation to you.
  --with-cudnn      Accepted for compatibility; cuDNN is now always installed.
  --install-driver  Install Ubuntu's recommended driver on a bare host if none
                    works. Never replaces a working driver or installs one in
                    a container. Reboot and rerun if needed.
  --dry-run         Print commands without changes or host validation.
  --help            Show this help.

Dependencies and linux-env.sh live in outputs/setup/. Source that environment
before subsequent builds, tests, or inference. Shell profiles are not modified.
H3_BUILD_JOBS sets build parallelism (default 8). Without a visible GPU, builds
use CUDA_ARCH=fat; GPU runtime validation still requires the deployment device.
HELP
}

dry_run=0
build=1
install_driver=0
for arg in "$@"; do
    case "$arg" in
        --no-build) build=0 ;;
        --with-cudnn) ;; # The default video pipeline requires cuDNN.
        --install-driver) install_driver=1 ;;
        --dry-run) dry_run=1 ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$arg" >&2; usage >&2; exit 2 ;;
    esac
done
fail() { printf 'setup_linux: %s\n' "$*" >&2; exit 1; }
run() {
    printf '+'; printf ' %q' "$@"; printf '\n'
    if (( ! dry_run )); then "$@"; fi
}
as_root() {
    if (( dry_run )) || [[ $(id -u) != 0 ]]; then run sudo "$@"; else run "$@"; fi
}

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
setup_dir="$repo_dir/outputs/setup"
env_file="$setup_dir/linux-env.sh"
env_temp="$setup_dir/.linux-env.sh.$$"
frontend_dir="$setup_dir/cudnn-frontend"
cutlass_dir="$setup_dir/sglang-cutlass"
cutlass_revision=da5e086dab31d63815acafdac9a9c5893b1c69e2
reference_env="$setup_dir/sglang-runtime"
cuda_dir=/usr/local/cuda-13.0
jobs=${H3_BUILD_JOBS:-8}
[[ $jobs =~ ^[1-9][0-9]*$ ]] || fail 'H3_BUILD_JOBS must be a positive integer.'
temp_dir=''
driver_ready=0
trap 'if [[ -n "$temp_dir" ]]; then rm -rf -- "$temp_dir"; fi; if (( ! dry_run )); then rm -f -- "$env_temp"; fi' EXIT

if (( ! dry_run )); then
    [[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || fail 'Requires Ubuntu 24.04 x86-64.'
    . /etc/os-release
    [[ ${ID:-} == ubuntu && ${VERSION_ID:-} == 24.04 ]] ||
        fail 'Requires Ubuntu 24.04; other distributions need separately qualified dependencies.'
    if [[ $(id -u) != 0 ]]; then
        command -v sudo >/dev/null 2>&1 || fail 'Run as root or install sudo first.'
    fi
    if nvidia-smi >/dev/null 2>&1; then
        driver_ready=1
        driver=$(nvidia-smi -i "${H3_CUDA_DEVICE:-0}" --query-gpu=driver_version --format=csv,noheader)
        [[ $(printf '%s\n' 580.126.20 "$driver" | sort -V | head -n1) == 580.126.20 ]] ||
            fail "Driver $driver is too old for the pinned CUDA 13 runtime. Update the host driver to >=580.126.20."
    fi
    if (( install_driver && ! driver_ready )); then
        if [[ -f /.dockerenv || -f /run/.containerenv ]] ||
           systemd-detect-virt --container --quiet 2>/dev/null; then
            fail 'Install the NVIDIA driver on the container host, then expose its GPU. Omit --install-driver for compilation only.'
        fi
    fi
else
    printf 'Dry run for Ubuntu 24.04 x86-64; commands shown with sudo (omitted when root).\n'
fi

as_root apt-get update
as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential git curl ca-certificates pkg-config libicu-dev libjson-c-dev \
    libssl-dev libsqlite3-dev libcurl4-openssl-dev zlib1g-dev libturbojpeg libx264-dev nasm python3 python3-venv openssh-client rsync
if (( install_driver && ! driver_ready )); then
    as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y ubuntu-drivers-common
    as_root ubuntu-drivers install
elif (( driver_ready )); then
    printf 'Keeping the working NVIDIA driver.\n'
fi

# Reuse provider repositories to avoid duplicate Signed-By settings.
cuda_repo=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64
if (( ! dry_run )) && apt-cache policy | grep -F "$cuda_repo" >/dev/null; then
    printf 'Keeping the existing NVIDIA CUDA package repository.\n'
else
    if (( dry_run )); then keyring='<temporary-directory>/cuda-keyring.deb'
    else temp_dir=$(mktemp -d); keyring="$temp_dir/cuda-keyring.deb"; fi
    run curl -fsSL --retry 3 "$cuda_repo/cuda-keyring_1.1-1_all.deb" -o "$keyring"
    as_root dpkg -i "$keyring"
    as_root apt-get update
fi
if (( ! dry_run )) && [[ -x "$cuda_dir/bin/nvcc" &&
        -f "$cuda_dir/include/cuda_runtime.h" && -f "$cuda_dir/include/cublasLt.h" &&
        -e "$cuda_dir/lib64/libcudart.so" && -e "$cuda_dir/lib64/libcublas.so" &&
        -e "$cuda_dir/lib64/libcublasLt.so" ]] &&
        "$cuda_dir/bin/nvcc" --version | grep -F 'V13.0.88' >/dev/null &&
        python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["cuda"]["version"] == "13.0.3"' "$cuda_dir/version.json" &&
        bash "$repo_dir/scripts/cuda_arch.sh" fat "$cuda_dir/bin/nvcc" >/dev/null 2>&1; then
    printf 'Keeping the existing complete CUDA 13.0 Update 3 toolkit.\n'
else
    as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-13-0=13.0.3-1
fi

run mkdir -p "$setup_dir"
run python3 "$repo_dir/scripts/setup_ffmpeg.py" --prefix "$setup_dir/ffmpeg" --jobs "$jobs"
if [[ ! -e "$frontend_dir" ]]; then
    run git clone --depth 1 --branch v1.11.0 https://github.com/NVIDIA/cudnn-frontend.git "$frontend_dir"
fi
if [[ ! -e "$cutlass_dir" ]]; then
    run git init "$cutlass_dir"
    run git -C "$cutlass_dir" remote add origin https://github.com/NVIDIA/cutlass.git
    run git -C "$cutlass_dir" fetch --depth 1 origin "$cutlass_revision"
    run git -C "$cutlass_dir" sparse-checkout set include
    run git -C "$cutlass_dir" checkout --detach FETCH_HEAD
fi
if (( ! dry_run )); then
    expected=$(git -C "$frontend_dir" rev-parse 'refs/tags/v1.11.0^{commit}') ||
        fail "Frontend at $frontend_dir must be a v1.11.0 checkout."
    [[ $(git -C "$frontend_dir" rev-parse HEAD) == "$expected" &&
       -z $(git -C "$frontend_dir" status --porcelain) &&
       -f "$frontend_dir/include/cudnn_frontend.h" ]] ||
        fail "Frontend at $frontend_dir must be an unmodified v1.11.0 checkout. Move it aside and rerun."
    [[ $(git -C "$cutlass_dir" rev-parse HEAD) == "$cutlass_revision" &&
       -z $(git -C "$cutlass_dir" status --porcelain) ]] ||
        fail "CUTLASS at $cutlass_dir must be an unmodified $cutlass_revision checkout."
fi
run python3 "$repo_dir/scripts/verify_sglang_dependencies.py" "$cutlass_dir"

# One private native runtime preserves provider packages.
if [[ ! -x "$reference_env/bin/python" ]]; then run python3 -m venv "$reference_env"; fi
run "$reference_env/bin/python" -m pip install --disable-pip-version-check \
    --only-binary=:all: --no-deps --require-hashes -r "$repo_dir/scripts/requirements-sglang-runtime.txt"

cuda_arch=fat
if (( ! dry_run )); then
    "$cuda_dir/bin/nvcc" --version | grep -F 'V13.0.88' >/dev/null || fail 'Expected CUDA 13.0 Update 3 (nvcc V13.0.88).'
    run python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["cuda"]["version"] == "13.0.3"' "$cuda_dir/version.json"
    run "$cuda_dir/bin/nvcc" --version
    run gcc --version
    run pkg-config --modversion icu-uc json-c libcrypto
    run python3 -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ is required"'
    if arch_result=$(bash "$repo_dir/scripts/cuda_arch.sh" auto "$cuda_dir/bin/nvcc" 2>&1); then cuda_arch=auto
    else
        if (( driver_ready )); then
            printf 'Automatic selection failed for the visible NVIDIA GPU: %s\n' "$arch_result"
            printf 'Using CUDA_ARCH=fat for compilation; execution on this GPU is not qualified by that fallback.\n'
        else printf 'No working visible NVIDIA GPU detected; using CUDA_ARCH=fat for compilation.\n'; fi
        bash "$repo_dir/scripts/cuda_arch.sh" fat "$cuda_dir/bin/nvcc" >/dev/null
    fi
    reference_site=$("$reference_env/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
    cudnn_lib="$reference_site/nvidia/cudnn/lib"
    reference_lib="$reference_site/nvidia/cu13/lib"
    run ln -sfn libcudnn.so.9 "$cudnn_lib/libcudnn.so"
    {
        printf '# Generated by scripts/setup_linux.sh; source before build, tests, or inference.\n'
        printf 'export CUDA_PATH=%q\nexport NVCC=%q\n' "$cuda_dir" "$cuda_dir/bin/nvcc"
        printf 'export PATH=%q":$CUDA_PATH/bin:$PATH"\n' "$setup_dir/ffmpeg/bin"
        printf 'export CUDA_ARCH=%s\nexport CUDA_SGLANG=1\nexport CUDA_CUDNN=1\n' "$cuda_arch"
        printf 'export CUDNN_FRONTEND_PATH=%q\nexport SGLANG_CUTLASS_PATH=%q\n' "$frontend_dir/include" "$cutlass_dir"
        printf 'export CPATH=%q"${CPATH:+:$CPATH}"\n' "$reference_site/nvidia/cudnn/include"
        printf 'export LIBRARY_PATH=%q"${LIBRARY_PATH:+:$LIBRARY_PATH}"\n' "$cudnn_lib"
        printf 'export LD_LIBRARY_PATH=%q"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n' "$cudnn_lib:$cuda_dir/lib64:$reference_lib"
        printf 'export H3_SGLANG_CUBLAS_LIBRARY=%q\n' "$reference_lib/libcublas.so.13"
        printf 'export H3_SGLANG_CUDNN_LIBRARY=%q\n' "$reference_site/nvidia/cudnn/lib/libcudnn.so.9"
        printf 'export H3_SGLANG_JPEG_LIBRARY=/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0\n'
        printf 'export H3_SGLANG_INPUT_FFMPEG=%q\nexport H3_FFPROBE=%q\n' "$setup_dir/ffmpeg/bin/ffmpeg" "$setup_dir/ffmpeg/bin/ffprobe"
        printf 'export H3_FFMPEG=%q\n' "$setup_dir/ffmpeg/bin/ffmpeg"
    } > "$env_temp"
    # shellcheck disable=SC1090
    source "$env_temp"
    check_args=(--out "$setup_dir/runtime.json")
    if (( driver_ready )); then check_args+=(--gpu); fi
    run python3 "$repo_dir/scripts/verify_cuda_runtime.py" "${check_args[@]}"
    if (( build )); then run make -C "$repo_dir" -j"$jobs" all; run "$repo_dir/bin/h3cli" --help; fi
    mv "$env_temp" "$env_file"
else
    printf 'Would verify library versions, GPU runtime (when available), and write %s\n' "$env_file"
    if (( build )); then run make -C "$repo_dir" -j"$jobs" all; run "$repo_dir/bin/h3cli" --help; fi
fi
printf '\nSource the environment before subsequent builds, tests, or inference:\n  source %q\n' "$env_file"
if (( ! build )); then printf '  make -j%s\n' "$jobs"; fi
printf '  ./bin/h3cli --help\n  make test  # requires a working NVIDIA GPU\n'
printf 'Install test-only codecs with python3 scripts/setup_reference_media.py.\n'
printf 'For exact recorded parity: set H3_REFERENCE_MODEL and H3_REFERENCE_REGRESSION_OUT,\nthen run make test-cuda-reference-regression (see CONTRIBUTING.md).\n'
if (( install_driver && ! driver_ready )); then
    printf '\nReboot the host if needed, check nvidia-smi, and rerun setup for GPU validation.\n'
elif (( ! driver_ready )); then
    printf '\nGPU runtime was not validated. Expose a GPU with driver >=580.126.20 and rerun setup.\n'
fi
