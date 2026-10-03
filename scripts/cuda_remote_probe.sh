#!/usr/bin/env bash
source "$(dirname "$0")/cuda_remote_common.sh"
mkdir -p "$H3_CUDA_LOG_DIR"
remote "mkdir -p $remote_dir/outputs/cuda-validation $remote_storage/tmp && cd $remote_dir && bash -s -- $remote_storage" <<'REMOTE' | tee "$H3_CUDA_LOG_DIR/cuda-environment.txt"
export PATH=/usr/local/cuda/bin:$PATH
{
    date -u
    uname -a
    cat /etc/os-release
    lscpu
    free -h
    for limit in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current \
        /sys/fs/cgroup/memory/memory.limit_in_bytes /sys/fs/cgroup/memory/memory.usage_in_bytes; do
        if test -r "$limit"; then awk '{print FILENAME ": " $0}' "$limit"; fi
    done
    nvidia-smi
    nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total,memory.free,driver_version --format=csv
    nvcc --version
    gcc --version | head -n 1
    ffmpeg -version | head -n 1
    ffprobe -version | head -n 1
    df -h . "$1"
} 2>&1 | tee outputs/cuda-validation/cuda-environment.txt
REMOTE
