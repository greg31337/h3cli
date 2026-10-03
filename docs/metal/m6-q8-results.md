# M6: Metal Q8 weights

M6 implements weight-only Q8 for the repeated DiT projections while preserving
the original BF16 checkpoint and default renderer. The four-bit branch is
removed. The user explicitly skipped the CUDA comparison; these results are
local M4 Max measurements, with no M4/5090 ratio.

Implementation and the bounded local validation are complete. Q8 saves about
40% of observed process footprint with approximately unchanged B5 denoising
time, but **fails production quality qualification**. All 15 B1/B5/B6 runs
pass execution checks; the six-step visual screens fail. The retained
[playback and results page](../../outputs/metal-native-m6-q8/review/review.html)
contains all completed comparisons. Q8 remains opt-in.

## Implementation

Recipe 1 quantizes the 200 repeated QKV, attention output, FC1 and FC2 matrices
to signed int8 groups of 64, with FP32 scales and round-to-even packing. Those
matrices use **20.472 GB instead of 38.535 GB**, including scales: **46.875%
less weight storage**. Norms, modulation, conditioning, patch/final projections
and VAE weights retain their existing formats. Packing uses at most 8 MiB of
CPU staging and does not write a quantized checkpoint.

The default Q8 policy performs native Metal dequantization followed by
MPSGraph BF16 linears. Four context-owned BF16 scratch matrices, totaling
0.771 GB, are reused across all blocks. A direct native simdgroup kernel is
also available for isolated measurement; its M4 projection timings determine
its disposition independently. Both policies retain BF16 activations and
explicit BF16 SwiGLU. The Q8 comparison therefore measures the complete
weight-execution recipe, including the explicit MLP path.

Selection, commands and unsupported combinations are documented in the
[Q8 guide](q8.md). Q8 is independent of dense/SOL attention, requires resident
Metal execution, and rejects ANE and SSD streaming. Checkpoint
section 37 and prepared-context keys bind format, recipe, group size and
kernel policy. The existing BF16 checkpoint format remains compatible.

## Projection correctness and performance

Both policies pass all eight small GPU cases against an independent FP64 CPU
dot-product oracle over BF16 reconstructed weights, with bit-identical BF16
outputs. Coverage includes odd shapes, group/tile tails, bias, output guards,
input immutability and activations well beyond FP16 range. The queued test
also passes: distinct weights reuse one scratch slot correctly within a
command buffer and across bounded command retirement, without warm allocation
growth.

Real block-zero checkpoint weights, synthetic BF16 activations and 22,426
rows give the following medians after warmup and three balanced timed repeats.
Dequantization is included in Q8 time.

| Projection | BF16 MPSGraph | Q8 dequant + MPSGraph | Q8 simdgroup | Q8 relative L2 |
| --- | ---: | ---: | ---: | ---: |
| QKV | 356.06 ms | 362.05 ms | 862.30 ms | 0.007062 |
| Attention output | 119.18 ms | 121.03 ms | 309.61 ms | 0.006604 |
| FC1 | 474.87 ms | 482.51 ms | 1161.87 ms | 0.006842 |
| FC2 | 238.37 ms | 242.48 ms | 615.50 ms | 0.007235 |

The BF16 column is the matched dequant-policy control. The direct simdgroup
policy has its own paired BF16 timings in the logs and is 2.40–2.60× slower.
Dequant + MPSGraph is 1.56–1.72% slower in these isolated projections. Both
policies pass the frozen real-projection L2 limit of 0.03, with equal reported
errors. Only the default dequant policy proceeds to complete B1/B5/B6 renders;
the direct policy retains operator coverage and opt-in status. See the
[operator manifest](../../outputs/metal-native-m6-q8/operator-final/manifest.json)
for commands, log checksums and frozen-final provenance.

## Matched render measurements

All runs use 640×480, **243 frames**, the original FL2VA checkpoint, all 50
blocks, seed 42, the same prompt and conditioning, and CPU Euler sampling.
B1 has one evaluation, B5 five, and B6 six. GPU work is serial. B1/B6 capture
diagnostic tensors; performance claims use ordinary B5 steps 2–5. B6 uses
the production VAE. Native attention is the FP16 `steel-routed` adapter path;
the weight labels below do not describe attention precision.

| Execution | B5 median step | Observed footprint | Core weight format |
| --- | ---: | ---: | --- |
| Reference+: MPSGraph BF16 dense | 126.04 s | 46.240 GB | BF16 |
| Native dense FP16 attention | 126.30 s | 47.175 GB | BF16 |
| Native dense FP16 attention | 127.42 s | 28.038 GB | Q8 |
| Native SOL FP16 attention | 121.92 s | 48.515 GB | BF16 |
| Native SOL FP16 attention | 118.95 s | 29.391 GB | Q8 |

Memory is the **maximum observed physical footprint at instrumented
boundaries**, including compressed process memory, in decimal GB. These are
not continuously sampled peaks. Q8 reduces observed footprint by about 40%
against the corresponding BF16-weight recipe. Allocations remain stable
after initial scratch setup; SOL adds routing scratch on its first routed
evaluation.

Q8 dense is 0.9% slower than the matched BF16-weight dense run. Q8 SOL is
2.5% faster than the matched BF16-weight SOL run. Both pass the frozen 5%
regression tolerance. There is only one B5 trajectory per mode, timing ranges
overlap, and a CPU-only final build overlapped part of the Q8 SOL run. These
small differences do **not** establish a repeatable speedup. The demonstrated
benefit is memory reduction. Startup packing increases core loading time
from 2.76–2.78 s to 14.98–15.36 s; the report retains loading and denoising
separately.

Attention and routing are also measured separately: dense FP16 versus
Reference+ is 0.998×, and BF16-weight SOL versus dense FP16 is 1.036×. This
does not meet the independent M4 SOL production target of 1.10×. M6 does not
relabel that target or earlier failed screens.

The SOL policy is minimum exact fraction 0.75, one dense initial evaluation,
dense sigma threshold 0.99, one dense layer, query/KV blocks 32/64 and local
radius 1. B1 is deliberately dense under this policy; later evaluations must
prove actual routing. Existing protected-range and range-recovery logic
remains active.

Both B5 SOL runs evaluate approximately 83.34% of routed-region pairs
exactly. All four native B5 modes report zero invalid heads after recovery;
the existing BF16 recovery handles 80–87 heads across each five-step run.

## Numerical and perceptual qualification

B1 Q8 versus the same BF16-weight attention recipe has relative velocity L2
of **0.017514 video / 0.009606 audio**, passing the frozen 0.05 screen.
Dense/SOL B1 comparisons coincide because the first evaluation is dense.
Quality controls on the actual Reference+ B6 clip pass: benign controls are
accepted and deliberately degraded controls rejected.

All four B6 comparisons fail the six frozen visual screens (LPIPS mean/p95,
SSIM mean/p05, temporal delta and optical flow):

| Candidate versus baseline | Mean LPIPS | Mean SSIM | Audio / A-V screens |
| --- | ---: | ---: | --- |
| Dense Q8 versus dense BF16 weights | 0.21503 | 0.70114 | Pass |
| SOL Q8 versus SOL BF16 weights | 0.21579 | 0.70860 | Envelope error/correlation fail; A/V timing passes |
| Dense Q8 versus Reference+ | 0.20057 | 0.71299 | Pass |
| SOL Q8 versus Reference+ | 0.26727 | 0.64688 | Envelope error/correlation fail; A/V timing passes |

These are independent six-step trajectories, so B1's small velocity error is
not a guarantee of final-frame similarity. Final latent relative L2 is
0.41221 video / 0.05220 audio for incremental dense Q8, and 0.41196 / 0.16358
for incremental SOL Q8. No human visual verdict is inferred from these
metrics. The report therefore sets both `measured_subset_pass` and
`production_qualified` to false. No default or quality threshold is changed.

The report retains each failed screen and separates incremental Q8 changes
from the total change against Reference+. Prior acceptance of FP16/SOL is
not inherited by Q8. Six evaluations do not establish 50-step trajectory
equivalence, conditioning/identity coverage or continuation qualification.
No human-review approval is required to close this experiment.

## Reproducibility and limits

The retained [run index](../../outputs/metal-native-m6-q8/runs/index.json)
binds commands, source provenance, executable, logs, inputs and output
checksums. The report also checks complete finite diagnostic tensors,
matching initial noise and conditioning, actual Q8 dispatches, protected SOL
policy and zero unrecovered invalid attention heads. It rejects changed
artifacts and post-hoc edits to the frozen quality contract.

Renders use the immutable `frozen-v2` binary, SHA-256
`b29907a2fbba78f2b800f705704cda286720f6ef73b02fb09d161944519df1ef`.
Final operator tests use `frozen-final`. The only intervening runtime changes
reject overflowing group-count shapes and source rows exceeding the 8 MiB
staging limit; supported H3 matrix arithmetic is unchanged. The
[source delta](../../outputs/metal-native-m6-q8/final-source-delta.json)
records this evidence split explicitly.

Host regression tests pass: 1,658 sampler checks, 18 container tests covering
193 adversarial containers, five CLI tests, six comparison tests, five
report-integrity tests and host Q8 packing checks. Their commands, outcomes
and checksums are retained in the
[host manifest](../../outputs/metal-native-m6-q8/host-final/manifest.json).

Optional native BF16 attention remains deferred. The prior ANE prototype's
failed promotion is unchanged; Q8 + ANE is unsupported. M5 INT8 arithmetic,
Turbo composition, 50-step trajectories and the held-out conditioning and
continuation corpus are not qualified by this milestone.
