# M2 FP16 attention

**Final M2 disposition:** the user accepted the measured speed and B6 quality,
then chose to start conservative SOL. M2 closes on the measured subset; remaining
held-out conditioning/continuation runs and range coverage are deferred. The
failed frozen screens below remain measured failures. See
`tests/metal_fp16_quality_acceptance.json` for exact evidence and scope. This
acceptance does not extend to new SOL approximation or untested cases.


M2 uses the existing pinned MLX Steel infrastructure through a local adaptation,
with no VPIPE runtime or copied VPIPE source. The default renderer and its BF16
MPSGraph SDPA remain unchanged. The primitive is callable from
`src/metal/metal_fp16.h` and `bin/metal_fp16`. An explicit hybrid now
connects it to the existing DiT, while QKV/norm/RoPE, projection, MLP and sampler
arithmetic remain unchanged. It is not a qualified production tier.

## Source and arithmetic audit

[The audit manifest](m2-source-audit.json) retains exact revisions, source URLs,
licenses and SHA-256 hashes. The compiled MLX subset remains byte-identical to
`third_party/mlx-attention/manifest.json`, including its MIT license. The build
checks those hashes before embedding. Local derivatives retain Apple's notice.

VPIPE's audited `attn_steel.metal` instantiates half-storage D=128 attention with
32×16 tiles and float accumulation. Merely changing the storage specialization
of our existing template would still load its matrix operands into float
fragments. The M2 adaptation explicitly separates **operand type from accumulator
type**. Q/K and P/V matrix operands are half; score and output matrices, maxima,
exponentials, normalization sums and running accumulators are float. Probabilities
round to half only at the PV matrix multiply. Final output rounds to BF16.

The audit also pins MLX's `steel/attn/nax.h`. It uses Metal Performance Primitives
TensorOps cooperative tensors and matrix descriptors. That is a distinct M5
implementation family, not a capability established by running this simdgroup
candidate on M4. NAX remains deferred pending actual M5 qualification.

## Range policy and layouts

Each head scans the original BF16 Q/K/V values on the GPU. Independent powers
of two put their maxima in `[4096,8192)`. Q/K exponents compensate the attention
scale; the V exponent compensates the normalized output. Normal half operands
then represent these BF16 values exactly. Zero values remain zero.

The range policy explicitly zeros and counts values below the minimum normal
half magnitude after scaling. For admitted heads it bounds the resulting
score perturbation by
`128 * scale * (deltaQ * maxK + deltaK * maxQ + deltaQ * deltaK)`.
The fixed bound is `1e-4`; the V absolute perturbation bound is reported too.
Heads exceeding the score bound use original BF16 values and FP32 matrix
operands instead. This is numerical recovery, not a change of requested routing.
It is included in every adapter-inclusive timing and reported by head count.
Neither saturation nor unreported operand clipping is used.

Recipe 4 packs both admitted half operands and bit-preserving BF16 recovery
operands into head-major workspace during conversion. The two conditional
kernels interpret only their selected heads. Q/K/V source buffers stay unchanged;
output can be sequence-major or head-major. No full score matrix is allocated.

Attention writes a temporary FP32 output. GPU checks cover output finiteness
and BF16 representability before committing any head to the caller's output.
Any invalid head prevents the entire commit. Nonfinite source values, BF16
subnormal inputs whose preservation Metal cannot promise, and unsafe FP32
score ranges are rejected. Standalone callers must inspect
`h3_gpu_mixed_range_report` after submission before accepting the result.
The DiT archives each block's range record before reusing scratch, then checks
all executed blocks at the completed-forward boundary. A later successful block
cannot conceal an earlier invalid one. Failure prevents the forward velocity
from updating sampler state. This failure latch is tested with a deliberately
invalid block followed by a valid block and a fresh successful step.

Range scans, packing, conditional recovery, output checking and commit run on
the same command buffer. There is no CPU readback between those operations.
Scratch is reused by the context and costs approximately ten bytes per QKV
element, plus small per-head records; changing its geometry requires a new
context. The harness reports peak live bytes, including its extra oracle buffers;
that total must not be presented as full-render peak residency.

The integration adds about 1.1 MB for a bounded six-slot, 50-block record ring
(the ring reuses slots for longer production schedules). Its 50 small copies
cost 190,400 bytes per full evaluation. CPU Euler already waits at the forward
boundary; GPU Euler requires an additional explicit validation wait there.
There are no per-layer CPU range readbacks. Profiling emits per-block range,
underflow and recovery counts as `h3_mixed` records. Kernel recovery, all scans,
copies and waits are included in the measured integration.

## Explicit selection and state identity

```sh
./bin/h3cli --backend metal \
  --metal-attention-kernel steel-routed --metal-attention-dtype fp16 \
  --metal-tier diagnostic [other render arguments]
```

`--metal-tier reference` and `preview` label candidates; they do not change
arithmetic, sampling, defaults or qualification. FP16 currently accepts dense
routing only. SOL, ANE and native M5 NAX remain separate work. Native
Metal's existing full-depth, BF16-weight and no-reuse/reduction restrictions
still apply. The selected dense tile is 32×16; the `--sol-*-block` options do
not configure dense FP16 tiles. The slower 64×32 tile is a standalone candidate.

Checkpoint attention version 3 binds FP16 recipe 4, including math, range and
layout policy. Dtype and tier serialize independently and participate in both
prepared-cache identities and resume compatibility. Future incompatible recipe
changes must advance the checkpoint version. `h3_recipe` reports operands,
accumulators, storage, routing, device family and tier. No Metal
candidate is silently accepted as Reference.

## Validation and reproduction

`tests/metal_fp16_limits.json` preserves the original operator contract.
`tests/metal_fp16_limits_v2.json` retains exactly the same numerical and
performance limits while describing the explicit bounded-underflow policy.
The independent CPU double oracle checks distributed short-sequence positions.
MPSGraph compares identical converted inputs; error against original BF16 inputs
is reported separately. GPU underflow counters must equal the independent CPU
conversion counts. Guards, input hashes, layouts, aligned/tail shapes, zero and
outlier inputs, large/small and reciprocal scaling, recovery and rejected inputs
are checked. Existing BF16 and protected SOL instantiations are regression-tested.

```sh
make bin/metal_fp16 bin/metal_attention bin/metal_sol
python3 scripts/metal_fp16_validate.py --output NEW_DIRECTORY
python3 tests/test_metal_fp16_records.py
```

The driver snapshots sources, binaries and limits before testing and runs GPU
jobs serially. Its six balanced permutations compare MPSGraph and two fixed
tiles at the 243-frame shape: S=22,426, H=56, D=128. The required real fixtures
cover blocks 0/24/49, three image references (S=24,250) and a 192-frame continuation
prefix at the same 243-frame target. One fixed candidate must pass every fixture;
averaging across fixtures or selecting a different winner per fixture cannot pass.
All adapter costs are timed. Exit code 3 records a performance-gate failure,
distinct from an operator failure. Raw records and an HTML table are retained.

After a standalone pass and matched one-block proof, the serial model driver
accepts a binary built from a retained source snapshot and checks its provenance:

```sh
python3 scripts/metal_fp16_model_validate.py \
  --binary FROZEN_BUILD/bin/h3cli --conditioning CACHE.h3cond \
  --standalone STANDALONE/index.json --block-proof BLOCK/compare.json \
  --output NEW_MODEL_DIRECTORY
python3 scripts/metal_fp16_diagnose.py \
  --workflows NEW_MODEL_DIRECTORY --output NEW_REPORT/diagnostics.json
```

Model acceptance uses unfenced B1/B5, all 50 blocks, original BF16 weights,
CPU Euler and `H3_DIT_COMMAND_BLOCKS=5`. Diagnostic B1 runs measure matched
attention regions/blocks and capture first-evaluation velocities and latents.
Those fences and file writes are excluded from acceptance timings. The offline
diagnosis verifies AV checksums and capture equivalence before reusing the M1
dense-A velocity captures. It reports later-noise teacher-forcing and additional
internal range coverage as outstanding when they have not been measured.

Metal shader validation passes for the selected tile on the focused failure-latch
case. Instrumentation exceeds the device's threadgroup resource limit for the
unselected 64×32 tile; its ordinary operator tests pass. Do not describe that
tile as passing Metal shader validation.

No standalone result establishes whole-step speed or visual Reference quality.
M2B requires the standalone gate; Reference additionally requires the frozen
perceptual corpus, complete B1/B5/B6 checks and production-VAE output. All render
tests remain at 243 frames and no more than six evaluations. Existing strict
M1 failures and optional M5/BF16 branches remain separate.

## Accepted continuation

The user accepted the measured 1.049× whole-step result for continued M2C/D.
[The qualification protocol](m2-reference-qualification.md) documents the added
CPU Euler teacher-input diagnostic, GPU range/score probes, immutable quality
contract, production-VAE B6 driver and paired HTML. Original failed performance
measurements and strict numerical limits remain retained. Runtime attention
recipe 4 and its kernel/adapter arithmetic are unchanged by these diagnostics.
