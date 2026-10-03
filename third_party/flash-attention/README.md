# Pinned FlashAttention inference headers

Source: Dao-AILab/flash-attention at
`6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15`, the submodule pinned by
PyTorch `cf30153c4c131c8164ee7798e5022d810682e2cb` (installed 2.13.0 oracle).
Only the forward launch header and its local dependency closure are retained.
See LICENSE for the BSD 3-Clause terms; original per-file notices are preserved.

Local changes remove framework-only RNG state from the inference parameter
structure, substitute an unused zero dropout seed, and adapt host CUDA errors
to C++ exceptions caught by the C ABI. Dropout is disabled at compile time.
No tensor/softmax/accumulation arithmetic is changed. The native wrapper owns
pointers, strides and bounded LSE scratch. CUTLASS headers are a build dependency.
This code is isolated from the existing fast CUDA implementation.

The wrapper enables PyTorch's `UNFUSE_FMA` softmax setting and does **not**
enable `--use_fast_math`. The latter introduces approximate-division differences
that pass a local tolerance but accumulate through Qwen. Matching these settings
and the pinned CUTLASS headers gives bitwise equality on the retained Qwen and
production DiT tensors with CUDA 12.8.

Build with `CUDA_SGLANG=1 SGLANG_CUTLASS_PATH=/path/to/pinned-cutlass`. Download
NVIDIA/CUTLASS revision `da5e086dab31d63815acafdac9a9c5893b1c69e2`, the exact
PyTorch submodule, into a separate dependency directory. The compiler uses its
`include/` subtree; `scripts/verify_sglang_dependencies.py` verifies the retained
source manifest before compiling. Preserve its `LICENSE.txt` with distributed
binaries. Existing SageAttention keeps its independent CUTLASS 4.2.1 dependency.

The manifest excludes the unused distributed-GEMM headers; no headers outside
that subtree include them. All retained dependency hashes remain unchanged.
