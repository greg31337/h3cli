#!/bin/sh
set -eu
arch=${1:-auto}
nvcc=${2:-nvcc}
if [ "$arch" = auto ]; then
    arch=$(nvidia-smi -i "${H3_CUDA_DEVICE:-0}" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '. \r\n') || true
    if [ -z "$arch" ]; then
        echo "ERROR: cannot detect the selected CUDA GPU; choose CUDA_ARCH=86, 89, 90, 100, 120 or fat for compilation"
        exit 1
    fi
fi
case "$arch" in
    86|89|90|100|120) arches=$arch ;;
    fat) arches='86 89 90 100 120' ;;
    *) echo "ERROR: unknown CUDA_ARCH '$arch'; choose 86, 89, 90, 100, 120 or fat explicitly"; exit 1 ;;
esac
supported=$("$nvcc" --list-gpu-code 2>/dev/null) || { echo "ERROR: CUDA compiler not found: $nvcc"; exit 1; }
for sm in $arches; do
    if ! printf '%s\n' "$supported" | grep -qx "sm_$sm"; then
        echo "ERROR: $nvcc cannot build SM$sm; install a newer toolkit (SM100/SM120 need CUDA 12.8+)"; exit 1
    fi
    printf '%s ' "-gencode=arch=compute_$sm,code=sm_$sm"
done
# Lowest supported virtual architecture permits JIT on newer NVIDIA devices.
ptx=$arch
[ "$arch" != fat ] || ptx=86
printf '%s\n' "-gencode=arch=compute_$ptx,code=compute_$ptx"
