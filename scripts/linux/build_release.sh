#!/bin/bash
set -euo pipefail
source /opt/h3deps/environment.sh
export NVCC=/source/scripts/linux/nvcc.sh
export TZ=UTC
jobs=${H3_BUILD_JOBS:-8}
commit=$(python3 -c 'import json; print(json.load(open("/evidence/source.json"))["commit"])')
mapfile -t targets < <(python3 - <<'PY'
import sys
sys.path.insert(0,'tests')
from cuda_reference_regression import BUILD
from current_cuda_suite import TARGETS
print('\n'.join(sorted(set(BUILD+TARGETS))))
PY
)
make -j"$jobs" PACKAGE_RUNTIME=1 H3_GIT_COMMIT="$commit" all "${targets[@]}"
mkdir -p bin/linux-release
# Static launcher is independent of the runtime's dynamic loader and libraries.
gcc -std=c11 -O2 -Wall -Wextra -Wno-deprecated-declarations -static \
    src/runtime/launcher.c -o bin/linux-launcher -ljson-c -lz -lcrypto -pthread -ldl
H3_TEST_BUNDLE_LAUNCHER=bin/linux-launcher python3 tests/test_linux_bundle.py
python3 scripts/linux/release.py
