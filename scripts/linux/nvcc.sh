#!/bin/bash
# CUDA 13 supports deterministic per-translation-unit symbol generation.
# Keep numerical flags in Makefile; this only replaces random symbol suffixes.
set -euo pipefail
seed=''
output=''
previous=''
for arg in "$@"; do
    case "$arg" in *.cu|*.cpp|*.c) seed=$arg ;; esac
    if [[ "$previous" == '-o' || "$previous" == '--output-file' ]]; then output=$arg; fi
    case "$arg" in --output-file=*) output=${arg#*=} ;; esac
    previous=$arg
done
if [[ -n "$seed" ]]; then
    /usr/local/cuda-13.0/bin/nvcc --frandom-seed="$seed" "$@"
else
    /usr/local/cuda-13.0/bin/nvcc "$@"
fi
# NVCC still leaves its PID-bearing cudafe temporary filename in STT_FILE even
# with --frandom-seed. Remove host debug/file metadata before linking; CUDA
# fatbins (including device line information), code and relocations are intact.
if [[ "$output" == *.o ]]; then objcopy --strip-debug "$output"; fi
