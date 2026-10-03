# M2 FP16 attention — 243-frame M4 results

**Final M2 disposition:** the user accepted the measured speed and B6 quality,
then chose to start conservative SOL. M2 closes on the measured subset; remaining
held-out conditioning/continuation runs and range coverage are deferred. The
failed frozen screens below remain measured failures. See
`tests/metal_fp16_quality_acceptance.json` for exact evidence and scope. This
acceptance does not extend to new SOL approximation or untested cases.


The FP16 kernel and hybrid integration work. The original
complete-step speed gate **failed**: B5 steady median is **118.933069 s versus
124.738281 s**, or **1.048811×**, below the frozen 1.10× requirement. The user
subsequently accepted this result for continued M2C/D testing. The explicit
exception preserves the measured failure and does not promote the default.

The [continued qualification report](m2-reference-qualification.md) records the
frozen perceptual contract, six-noise-level matched-input diagnosis and expanded
GPU range/score evidence. Independent production-VAE B6 completed all six
evaluations with no invalid heads, but **FAILED** the frozen quality screen
(LPIPS mean 0.176154; SSIM mean 0.725328). The held-out corpus stopped at this
gate. M2 remains **OPEN** and the backend unqualified. The tables below retain the
original unfenced B1/B5 measurements.

## Implementation and environment

The [implementation guide](m2-fp16-implementation.md) describes the scaled
FP16 Q/K/P/V operands, FP32 score/softmax/output accumulation, per-head
BF16/FP32 recovery, immutable source inputs and checked BF16 output commit.
The original QKV/norm/RoPE, linears, MLP, residual state and sampler remain in
place. `--metal-attention-dtype fp16` requires explicit Metal,
`steel-routed` and dense routing. Tier labels do not qualify or change execution.

Local hardware: **Apple M4 Max, 128 GiB**, macOS 26.6.2, Apple clang 21,
Xcode 27.0. All render tests use **640×480, 243 frames, 50 blocks**, original
BF16 weights, fixed conditioning/seed, CPU Euler, `H3_DIT_COMMAND_BLOCKS=5`,
and at most six evaluations. Actual primary attention shape is
**S=22,426, H=56, D=128**; the three-image reference fixture has S=24,250.
Tests run serially. Cold startup, component fences, tensor captures and video
decoding are excluded from unfenced denoising comparisons.

The [source audit](m2-source-audit.json) pins VPIPE and MLX/NAX inputs. No VPIPE
source was copied; the existing MIT MLX headers and license remain unchanged
and hash-checked. M5 NAX is audited, not implemented or measured here.

## M2A: operator checks and standalone gate

The final suite contains **23 records**: 15 shape/range/layout cases, BF16 and
SOL regressions, and six balanced production fixtures. All operator checks
pass. Tests include independent CPU-double samples, MPSGraph on identical
converted inputs, source/guard preservation, arbitrary tails, both layouts,
overflow/underflow, reciprocal scaling, conditional recovery, explicit rejection
and unchanged output on invalid inputs. Original BF16 drift is reported
separately. Version 2 retains v1 numerical limits and specifies bounded/countable
underflow rather than silently changing the operator tolerance.

One fixed **32×16** tile passes all final adapter-inclusive median gates:

| Fixture | MPSGraph (s) | FP16 including adapters (s) | Speedup |
|---|---:|---:|---:|
| Synthetic S=22,426 | 1.294032 | 1.146822 | 1.128363× |
| Real block 0 | 1.296667 | 1.146768 | 1.130715× |
| Real block 24 | 1.294446 | 1.146965 | 1.128584× |
| Real block 49 | 1.295915 | 1.176850 | 1.101173× |
| Three image references | 1.505601 | 1.341419 | 1.122394× |
| 192-frame continuation prefix, 243-frame target | 1.295268 | 1.147600 | 1.128675× |

Each fixture uses six balanced permutations, with all range scans, packing,
conditional recovery, output validation and commit timed. The 64×32 candidate
does not pass performance. No per-fixture candidate switching is permitted.

**The late-block margin is thin.** The preceding recipe-4 run measured
**1.098739×** on block 49 and failed; it remains retained as a failure. The final
1.101173× result admitted integration, not a robust production
claim. Earlier recipes and failed gates are retained under `recipe2/`,
`recipe3/`, `recipe4/` and `validation-v2/` in the artifact root.

## M2B: block and full-model results

The paired first block preserves QKV **byte-for-byte**. Projection output
relative L2 is **0.000858893**; complete-block relative L2 is **0.000849959**.
The first-block residual peak is **125,440**, already above FP16's finite range,
which supports retaining the wider residual path. Cold one-block captures are
correctness evidence, not acceptance timings.

| Unfenced measurement | MPSGraph BF16 | FP16 |
|---|---:|---:|
| B1 evaluation | 130.273905 s | 124.506978 s |
| B5 first evaluation | 124.788430 s | 124.099752 s |
| B5 steady median, evaluations 2–5 | 124.738281 s | 118.933069 s |
| B5 steady minimum–maximum | 124.712373–124.753770 s | 117.255226–124.140823 s |
| B5 total denoising | 623.731134 s | 603.361939 s |
| Tracked Metal peak | 38.462 GiB | 39.960 GiB |
| Metal allocator peak | 64.920 GiB | 66.118 GiB |
| Sampled process RSS peak | 49.690 GiB | 47.752 GiB |

The B5 steady speedup is **1.048811×**. Minimum reference divided by maximum
candidate is **1.004604×**. Memory passes the frozen 96 GiB tracked-Metal limit;
tracked live memory is not total residency and can omit MPSGraph internal
allocations, so allocator and process peaks are retained separately. There is
no justification to round this whole-step result into a 1.10× pass or to open
B6 production promotion.

New Reference+ B1 saved AV state is byte-identical to the retained M1 baseline
(`f8bb454917d8a640458d69947342a722a0ea820bc9bac0de0d9dc2752eb87565`).
The shared frozen integrated binary is SHA-256
`65ecc16cdaa79f30ca9a63d2db5e7e491c4f326c105efa24672f99a3e1cabbdb`.

Matched diagnostic B1 fences also show **no warm-block regression**. Summing
blocks 1–49 (excluding cold block 0):

| Diagnostic region | MPSGraph | FP16 hybrid | Speedup |
|---|---:|---:|---:|
| Hidden input through attention output projection | 86.246533 s | 79.643188 s | 1.082912× |
| Complete block | 122.227797 s | 115.556736 s | 1.057730× |

The remainder outside the attention region is almost unchanged: 35.981264 s
versus 35.913548 s. Attention preparation/projection plus attention still account
for about 69% of the hybrid block profile; this region is not the attention
kernel alone. These fenced measurements explain the modest full-step gain and
cannot replace the unfenced B5 gate. Both diagnostic runs produce byte-identical
AV states to their corresponding unfenced B1 runs.

## Range safety and numerical diagnostics

Every B5 evaluation executes all 50 blocks: **250 block records, 14,000 head
evaluations**, with **zero invalid heads**. **82 heads (0.586%)** use recovery;
counts by evaluation are 16, 16, 16, 17, 17. The adapter explicitly counts 49,804
operand underflows across B5. Maximum post-normalization/RoPE Q/K/V magnitudes
are **233 / 378 / 1,536**; maximum FP32 attention output is **633.413696**.
The maximum admitted score perturbation bound is **9.6432e-5** against 1e-4.
These are input-conversion bounds, not a bound on all attention arithmetic or
on the diffusion trajectory.

Recovery uses preserved BF16 inputs with FP32 operands before output commit.
All block reports survive scratch reuse and are checked before velocity updates
sampler state. The focused failure-latch regression confirms that a valid later
block cannot hide an earlier invalid one and that a fresh valid step succeeds.
The record copies add 190,400 explicit bytes and 50 blits per evaluation; shader
packing/scan traffic is not represented by that blit-byte counter. Scratch adds
approximately 1.50 GiB at the primary shape. No per-layer CPU range readback is
introduced. The GPU Euler path has an explicit completed-forward validation
wait; the matched acceptance runs use CPU Euler's existing boundary.

Original strict BF16 model limits remain unchanged and diagnostic for this new
candidate:

| Trajectory | Video relative L2 / cosine | Audio relative L2 / cosine | Old strict screen |
|---|---|---|---|
| B1 | 0.011647 / 0.999933 | 0.017533 / 0.999847 | PASS |
| B5 | 0.232185 / 0.973038 | 0.084309 / 0.996478 | FAIL |

Finite B5 divergence does not establish a perceptual failure or pass. The initial
preview-VAE playback pages remain comparison evidence. The subsequent
[qualification protocol and measurements](m2-reference-qualification.md) now
include a frozen reference-only calibrated perceptual contract and all six
teacher-forced noise levels, plus raw QKV/output/residual ranges and independent
sampled score/softmax probes. The latter are not fused-kernel internal traces.
Full mixed-model Ref2VA/continuation qualification remains separately gated.

The newly matched first-evaluation reference capture also lets us reuse the
retained M1 dense-A profile. Conditioning, seed, sampler and captured/unfenced
state identity are checked before comparison:

| Same-input first evaluation | M1 dense A relative L2 | M2 FP16 relative L2 |
|---|---:|---:|
| Video velocity | 0.007704 | 0.008111 |
| Audio velocity | 0.005496 | 0.005217 |
| Video latent | 0.011062 | 0.011647 |
| Audio latent | 0.018470 | 0.017533 |

The M2 first-evaluation error is comparable to the retained M1 dense error,
with unchanged QKV and passing operator/block checks. The subsequent six-level
teacher-forced experiment confirms comparable local errors at later noise levels:
native BF16 video velocity L2 is 0.007704–0.020855, versus 0.008111–0.019328 for
mixed FP16. This supports trajectory amplification as a major contributor to
the larger independent B5 differences. No defect was found on these tested
inputs; the result does not prove every future conditioning case or schedule.

## Validation and retained artifacts

Host validation passes **1,597 sampler-state checks**, conditioning/container
checks, CLI rejection, prepared-key/dtype/tier round-trip and numerical-record
validation. ASan/UBSan host checks pass, including **130,434 SOL layout/config
assertions** across Ref2VA modalities and 39/90/141/192-frame prefixes at 243
frames. Those protection checks regress the existing SOL layout; they do not
claim FP16 SOL implementation.

Metal shader validation passes the selected 32×16 failure-latch test. The
unselected 64×32 tile exceeds instrumentation threadgroup resources; its
ordinary operator tests pass. The rejected debug run remains in the artifacts.

Artifact root: `outputs/metal-native-m2-243/` (ignored generated data).

* [Standalone table and records](../../outputs/metal-native-m2-243/final/review.html).
* [B1/B5 paired playback](../../outputs/metal-native-m2-243/integrated/workflows/review.html).
* [Standalone manifest](../../outputs/metal-native-m2-243/final/index.json).
* [Model commands, timings and numerical metrics](../../outputs/metal-native-m2-243/integrated/workflows/index.json).
* [One-block comparison](../../outputs/metal-native-m2-243/integrated/block-compare.json).
* [Audited region/range and matched M1/M2 velocity diagnostics](../../outputs/metal-native-m2-243/integrated/diagnostics.json).
* Frozen sources/binaries under `final/frozen/` and `integrated/frozen/`,
  with exact build/input hashes, raw logs, saved states and playback assets.

The final workspace also includes stricter offline record checks, reporting,
documentation, C++ header guards and a metadata-label correction
(`qualified_reference` → `attention_oracle`). These do not change the measured
kernel or adapter arithmetic; the frozen executable and source snapshot remain
the authority for the timings above.

## Disposition

Keep the scaled dense adapter, operator oracles, explicit recovery and state
identity controls as an optional foundation. T135–T141 are implemented and
tested; T142 retains its measured failure and explicit user acceptance. T143 now
has complete six-noise-level evidence; T154 has full-layer T2VA range coverage
with remaining conditioned cases deferred by the user. T153 is frozen and calibrated;
T152 retains its failed independent B6 screen, followed by explicit user acceptance
of that quality. M2 closes on the measured subset; the held-out Ref2VA/continuation
corpus remains unqualified and conservative SOL starts next. No 50-step claim is added. M2E broad
fusion and cumulative SOL/ANE ablations remain conditional on their separately
recorded dispositions. M5 NAX and strict native BF16 stay deferred.

The profile does not justify automatically starting broad M4 BF16 GEMM work or
whole-model Q8. Record the limited dense gain and revisit the attention strategy
and independent quality results before promoting a Reference composition. No human
review or approval is required to retain these results.
