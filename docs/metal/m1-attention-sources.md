# M1 attention implementation and source audit

The representative experiment is **640×480, 243 frames, BF16, 50 layers**.
It has 72 video latent frames, 405 audio latent positions, and 22,426 packed
tokens for the fixed 16-token benchmark prompt. References can increase the
packed sequence length; continuation changes the protected prefix ranges.
Tests retain the six-evaluation cap.
The older 33,322-token / 362-frame evidence is historical, not a matched baseline.

## Revisions and implementation differences

Candidate A was already MLX Steel, not an independently handwritten kernel.
The existing vendored subset is pinned to Apple MLX
[`59d600b5e64c238427d0f8d897ab7c682ef4d3d2`](https://github.com/ml-explore/mlx/tree/59d600b5e64c238427d0f8d897ab7c682ef4d3d2/mlx/backend/metal/kernels/steel/attn).
`third_party/mlx-attention/manifest.json` checks every original header and the
MIT license during embedding. Those files remain unchanged. Candidate A uses
the existing 64×32 query/key tile, with 32×32, 32×64 and 16×64 diagnostic variants.
BF16 input/output, FP32 accumulation, arbitrary input/output strides and tail
masks are retained.

The comparison examined VPIPE revision
[`f34e2cc3a3adae759eea254419f436f5b7800057`](https://github.com/tgo-app-dev/vpipe/tree/f34e2cc3a3adae759eea254419f436f5b7800057),
specifically `gpu-kernels/metal/attention/attn_steel.metal`, `sol_attn_mma.metal`,
`sol_proxy_mma.metal`, `generative-models/shared/sol-attention.h`,
`metal-sol-attention.h`, and its vendored `steel_attention.h`.
Its BF16 D=128 dense dispatch includes a 32×16, 4×1 warp arrangement. Its
adapted Steel implementation accepts sparse key spans and exposes partial
softmax statistics for SOL; the host manages summary and routing workspaces.
The original MLX headers alone do not supply this entire execution strategy.

Candidate B (`steel-routed`) is a local adaptation of the pinned Apple template
in `src/metal/routed_attention.metal`. It holds the Q fragments in registers across
the key loop and uses a 32×16 dense dispatch; a 64×32 variant is included in
the bounded standalone comparison. It is a distinct implementation, not a
second copy of identical vendor headers. Its exact and approximate SOL
instantiations additionally support route flags, excluded summary columns,
FP32 unnormalized outputs and per-row maxima/denominators. There is no MLX
runtime dependency and no VPIPE source code is incorporated. The derivative
retains Apple's copyright notice and points to the complete retained MIT
license at `third_party/mlx-attention/LICENSE`.

## Hybrid execution and controls

The path is selected explicitly:

```sh
./bin/h3cli ... --backend metal --metal-attention dense \
  --metal-attention-kernel steel
./bin/h3cli ... --backend metal --metal-attention sol \
  --sol-q-block 32 --sol-kv-block 64 --sol-tau 1 \
  --sol-dense-layers 1 --sol-local-radius 1 --sol-min-exact 0.1
```

QKV, output projection and MLP remain MPSGraph BF16 operations. Existing native
Q/K normalization and RoPE now write directly to head-major attention inputs;
attention writes sequence-major output directly consumed by projection. The
native backend remains unqualified; MPSGraph remains the default. Unsupported
CUDA/quantization, layer omission, reuse and token-reduction combinations fail
preflight. Checkpoints and prepared-object keys include recipe version 2 and
every numerical/routing option.

SOL summarizes Q/K/V in FP32 and stores BF16 means. Routing estimates each
query-block score distribution using the diagonal variance of K centroids:
`threshold = mean_score + tau * sqrt(variance + 1e-6)` in log2 units. A centroid
score above the threshold selects exact attention. Increasing tau generally
reduces exact work, subject to mandatory protection and the minimum fraction.
The minimum exact fraction selects a deterministic evenly spread subset of
key blocks; it is a floor, not top-k ranking or an assumed realized fraction.
Initial dense layers are counted from zero: value 1 makes block 0 dense.
Local radius is measured in **video latent temporal frames**, not packed-token
blocks or decoded frames. Q blocks are 32/64; K/V blocks are 32/64/128.

Protected query or key blocks override approximate routing. Any block touching
text, conditioning, reference image/video/audio, target audio, the complete AV
continuation prefix, or first/last video latent frame is exact. Target audio is
also kept exact as a conservative proof-of-concept policy. Local temporal
neighborhoods and incomplete final K/V blocks are exact. The CPU layout builder
uses actual segment and prefix metadata; it has no specialized 39/90/141/192
frame kernel variants. `--sol-min-exact 1` selects the dense backend in DiT.

The exact pass skips excluded key tiles. The approximate pass attends to the
remaining K/V centroids, with key-block multiplicity added to its log-normalizer.
Both produce FP32 partial output, maximum and denominator. The final merge
rescales both partitions to a shared maximum before normalizing once. Empty
partitions contribute zero. Scratch belongs to the DiT context and is reused in
command order; no routing readback is required between layers. This prototype
uses byte route flags, unlike VPIPE's compact sparse span/CSR traversal.

## Validation and interpretation

`tests/metal_native_limits.json` freezes limits before model tuning. Dense
component, dense block/model, approximate SOL and protected-row tolerances are
separate. `tests/metal_sol.c` checks independent scalar approximate attention,
all-exact and constant-key limits, extreme logits, protected queries, both
layouts, tails, guard values, unchanged inputs and scratch reuse. CPU layout
tests cover every reference modality, anchors and all four continuation sizes.

The standalone dense oracle compares identical tensors with all six balanced
MPSGraph/A/B execution orders when `H3_TEST_ATTENTION_BALANCED=1`; it requires
equivalent sequence-major I/O. The retained input conversion before a
head-major-only microbenchmark is not counted as a speedup. Integrated runs
measure preparation through output projection and the complete block.

`--regions` fences complete regions without fencing inner components;
`--components` fences individual operations for attribution. Captured QKV or
boundary tensors add diagnostic readback and file I/O. These modes are marked
non-comparable for whole-step performance. Ordinary B1/B5 use the same command
grouping, conditioning, BF16 linears, sampler and requested evaluations.
Continuation comparisons require byte-identical protected prefixes and apply
the numerical limits separately to the generated suffix, so a long prefix cannot
dilute suffix error. `h3_execution` records composition and dispatch/copy costs; `h3_sol` records
realized exact/approximate/protected/local block counts. Root-command GPU time
does not claim to include all internal MPSGraph command buffers; wall time is
the acceptance measurement.

Source and executable hashes, commands, raw records and tensors are retained
under `outputs/metal-native-m1-243`. Successful implementation tests do not imply
SOL quality or performance acceptance. A losing candidate remains opt-in;
full-model B5/B6 and the post-SOL profile govern promotion and the M2 decision.
