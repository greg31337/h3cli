#!/usr/bin/env bash
set -euo pipefail
: "${H3_CUDA_HOST:?Set H3_CUDA_HOST to an SSH host or user@host}"
H3_CUDA_DIR=${H3_CUDA_DIR:-/workspace/bin/h3cli}
H3_MODEL_DIR=${H3_MODEL_DIR:-/workspace/models/MiniMax-H3}
H3_CUDA_STORAGE_DIR=${H3_CUDA_STORAGE_DIR:-$H3_CUDA_DIR}
H3_CUDA_LOG_DIR=${H3_CUDA_LOG_DIR:-outputs/cuda-validation/remote}
ssh_args=(-o BatchMode=yes -o ConnectTimeout=20)
[[ -z ${H3_CUDA_PORT:-} ]] || ssh_args+=(-p "$H3_CUDA_PORT")
[[ -z ${H3_CUDA_SSH_KEY:-} ]] || ssh_args+=(-i "$H3_CUDA_SSH_KEY")
printf -v remote_dir '%q' "$H3_CUDA_DIR"
printf -v remote_models '%q' "$H3_MODEL_DIR"
printf -v remote_storage '%q' "$H3_CUDA_STORAGE_DIR"
remote() { ssh "${ssh_args[@]}" "$H3_CUDA_HOST" "$@"; }
printf -v rsync_shell '%q ' ssh "${ssh_args[@]}"
