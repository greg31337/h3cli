#!/usr/bin/env bash
source "$(dirname "$0")/cuda_remote_common.sh"
tier=${1:-cuda-test}
case "$tier" in test|test-current-host|cuda-test|test-cuda-reference-regression|cuda-smoke|cuda-host-test|cuda-features) ;;
    cuda-memory-test|cuda-tokenizer-test|cuda-conv-test|cuda-attention-test|cuda-sm90-test|cuda-sm120-test) ;;
    *) echo "Unsupported test tier: $tier" >&2; exit 2;; esac
"$(dirname "$0")/cuda_remote_sync.sh"
"$(dirname "$0")/cuda_remote_probe.sh"
set +e
make_flags=""
if [[ -n ${H3_CUDA_CPPFLAGS:-} ]]; then printf -v flag '%q' "CPPFLAGS=$H3_CUDA_CPPFLAGS"; make_flags+=" $flag"; fi
if [[ -n ${H3_CUDA_LDFLAGS:-} ]]; then printf -v flag '%q' "LDFLAGS=$H3_CUDA_LDFLAGS"; make_flags+=" $flag"; fi
if [[ ${H3_CUDA_CUDNN:-0} == 1 ]]; then
    : "${H3_CUDA_CUDNN_FRONTEND_PATH:?Set the server cudnn-frontend/include path}"
    printf -v flag '%q' "CUDNN_FRONTEND_PATH=$H3_CUDA_CUDNN_FRONTEND_PATH"
    make_flags+=" CUDA_CUDNN=1 $flag"
fi
printf -v runtime_env '%q' "${H3_CUDA_ENV:-$H3_CUDA_DIR/cuda-env.sh}"
validation_id=$(date -u +%Y%m%dT%H%M%SZ)-$$
remote "set -e; cd $remote_dir; if test -f $runtime_env; then . $runtime_env; fi; export H3_REFERENCE_MODEL=$remote_models H3_REFERENCE_REGRESSION_OUT=outputs/cuda-validation/reference-$validation_id; mkdir -p $remote_storage/tmp $remote_storage/.cache/cuda; export TMPDIR=$remote_storage/tmp XDG_CACHE_HOME=$remote_storage/.cache CUDA_CACHE_PATH=$remote_storage/.cache/cuda H3_MODEL_DIR=$remote_models; make -j8 CUDA_ARCH=auto $make_flags all; make CUDA_ARCH=auto $make_flags $tier" 2>&1 | tee "$H3_CUDA_LOG_DIR/$tier.log"
status=${PIPESTATUS[0]}
set -e
rsync -rlpt -e "$rsync_shell" "$H3_CUDA_HOST:$H3_CUDA_DIR/outputs/cuda-validation/" "$H3_CUDA_LOG_DIR/"
exit "$status"
