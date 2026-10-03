# Native H3 attention kernel sources

Imported from thu-ml/SageAttention commit
`d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5` under the included Apache-2.0
license. `upstream.json` records original file hashes. CUTLASS is a separate
build dependency pinned to v4.2.1,
`f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`; supply its source directory as
`SAGE_CUTLASS_PATH`. Preserve that dependency's LICENSE.txt when distributing
it. No dependency is downloaded by `make` or by inference. The header manifest
excludes unused distributed-GEMM headers; no headers outside that subtree
include them. All retained dependency hashes remain unchanged.

Local adaptations remove unused PyTorch/ATen includes, binding code and
training RNG state. Sage3 launch errors propagate to the native caller.
Build portability fixes qualify dependent template calls, use unsigned shuffle
masks, avoid capturing structured bindings under C++17, and terminate the
utility header with a newline. These preserve the kernel arithmetic.
The noncausal Sage3 tail mask maps each packed K row back to its logical
token index. The pinned upstream mask fails the 17-token constant-input
check (39.45% relative output error); this local correction preserves true
sequence lengths. Original and adapted hashes are recorded separately.
The FP4 quantizer device functions retain the upstream arithmetic and scale
layouts. The SM89-named INT8/FP8 device kernel is instantiated for SM120a with
the 2++ FP16 partial / FP32 buffer accumulation mode.

The parent project's `h3_cuda_sage*.cu` files supply native allocation,
stream/stride bindings, INT8/FP8 preprocessing, bounded head/query workspaces,
BF16 outputs and finite-input checks. Inference has no Python, PyTorch or
Triton dependency. The separate validation oracle uses the unmodified
upstream kernels in an isolated environment. Its Sage3 adapter supplies H3's
explicit BF16-rounded attention scale and masks padded logical keys through
the score-correction tensor, independently of the native packed-row fix.
The Sage2++ oracle's zero-V divide-by-zero is recorded as an upstream
exception; the native path must return exact zero under that input.

Sage3 query smoothing reproduces the pinned Triton 3.4 SM120 BF16 reduction:
eight strided row accumulators, sixteen BF16 additions per accumulator,
then lane-pair and warp reductions. It does not substitute an FP32 mean.
Both Sage translation units use upstream fast-math/FMA flags independently
of H3's other CUDA translation units. Sage2 INT8 scales explicitly reproduce
the pinned assembler's fused `amax * (1/127) + 1e-7`; its value-to-scale
division uses Triton's `div.full.f32`. FP8 V packing computes its reciprocal
from the original channel maximum, separately from the rounded stored scale.
These details are necessary for exact packed-byte parity on real late-block
activations. Oracle tools and runtime code remain separate.

K smoothing also preserves the pinned Torch 2.8 BF16 mean's accumulation
order: four independent sums, input-warp reduction, a shape-dependent CTA
split and final reciprocal multiplication. A sequential sum can put a mean
on the other side of a BF16 rounding boundary and change FP4 bins. The native
implementation uses coalesced sequence-major loads and a fixed arithmetic
plan derived from the qualified 5090's 170 SMs and 1,536 threads/SM, independent
of workspace head groups. It does not link or include Torch. Source revision
and header hashes are in `torch-reduction.json`; the applicable notice is
retained in `LICENSE.PyTorch`.
