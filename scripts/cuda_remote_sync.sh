#!/usr/bin/env bash
source "$(dirname "$0")/cuda_remote_common.sh"
root=$(cd "$(dirname "$0")/.." && pwd)
remote "mkdir -p $remote_dir"
rsync -rlpt -e "$rsync_shell" --exclude=.git --exclude=.agents --exclude=.codex \
    --exclude=models --exclude=outputs --exclude=misc --exclude=lora \
    --exclude=bin --exclude='.DS_Store' --exclude=__pycache__ --exclude='*.o' --exclude='*.d' --exclude='*.dSYM' \
    --exclude='*.a' --exclude='.cuda-build-config*' \
    "$root/" "$H3_CUDA_HOST:$H3_CUDA_DIR/"
# Models are deliberately never transferred by this helper.
