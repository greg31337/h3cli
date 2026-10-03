# M1 implementation and 243-frame validation

The hybrid, second dense candidate, protected SOL, profiling,
checkpoint recipe and validation tools are implemented. **M1 remains open:**
dense B5 and SOL B1 fail the frozen model quality gates. MPSGraph remains the
production default. See [source audit and controls](m1-attention-sources.md).

## Workload and reproducibility

Local M4 Max, 128 GiB unified memory; 640×480, **243 frames**, 24 fps, 50 DiT
blocks, original BF16 weights, CPU Euler, no layer/step reuse or token reduction.
The fixed park prompt has 16 text tokens: the packed sequence is 22,426,
including 72 video latent frames and 405 audio positions. All model runs use
at most six evaluations. B1/B5 use the preview VAE; decoded preview clips do
not establish production-VAE quality.

The 362-frame M0/M1 results remain historical. They must not be combined with
these timings as a speedup comparison. Representative rendering scripts,
benchmark defaults and long audio fixtures now use 243 frames; short numerical
fixtures and maximum-duration validity/rejection tests retain their purpose.

Artifacts are under [`outputs/metal-native-m1-243`](../../outputs/metal-native-m1-243):
each model run keeps its executable, command, source hashes, conditioning
identity/hash, log, JSON profile, memory samples, complete AV state and clip.
`frozen/build-provenance.json` identifies the retained executable/source snapshot
used for later checks. Kernel records additionally keep the input hashes,
binary hashes, warm-up policy and balanced execution order.

## Numerical and integration checks

The first real block's QKV tensors are **byte-identical** across MPSGraph, both
native dense candidates and SOL. Direct head-major norm/RoPE output therefore
preserves the existing preparation arithmetic at this boundary.

| First-block boundary, relative L2 versus MPSGraph | Dense A | Dense B | SOL, block 0 enabled |
| --- | ---: | ---: | ---: |
| QKV | 0 | 0 | 0 |
| Output projection | 0.000803 | 0.000803 | 0.065515 |
| Complete block | 0.000815 | 0.000817 | 0.018562 |

Dense boundaries pass. SOL's complete-block metric passes the 0.02 block limit,
but its projection drift and full-model drift prevent quality acceptance.
The default SOL policy starts with one dense layer; the block diagnostic
explicitly uses zero initial dense layers to exercise SOL on identical input.
Boundary/QKV capture adds I/O and fences, so those runs do not qualify timing.

The SOL numerical harness checks its approximate operator against an independent
scalar double-softmax oracle, in addition to comparison with dense attention.
It covers all-exact and constant-key limits, partial tails, mixed routes,
positive/negative extreme logits, empty approximate partitions, both input and
output layouts, fully aligned tiles, unchanged
inputs, output guards and scratch reuse in one command buffer. Protected query
rows match dense. Approximation errors on random/outlier fixtures are reported
separately from kernel implementation correctness.

CPU layout tests cover all reference modalities, first/last anchors, Q blocks
32/64, K/V blocks 32/64/128, and continuation contexts 39/90/141/192 at the
243-frame target. Host tests also cover checkpoint recipe/options round trips,
invalid settings, malformed conditioning containers and CLI preflight.
Address/undefined-behavior sanitizer checks cover the new layout/configuration
logic and conditioning serialization.

The retained suites include 19 original kernel checks, nine output-layout
checks and three supplemental aligned/mixed-sign checks. All implementation
correctness checks pass; approximate-versus-dense quality failures remain
separate. `kernels/`, `sol-output-layout/` and `supplemental-kernels/` retain
the exact tested executable and fixture hashes with each result.

## Balanced standalone dense timing

Six balanced execution orders on identical sequence-major input/output, after
warming all three implementations. These are median seconds at S=22,426,
H=56, D=128; the bounded tile sweep and numerical gates also pass.

| Fixture | MPSGraph | Dense A | Dense B |
| --- | ---: | ---: | ---: |
| Synthetic production | 1.388411 | 1.370876 | 1.280151 |
| Real block 0 | 1.388468 | 1.370302 | 1.281096 |
| Real block 24 | 1.385656 | 1.371002 | 1.282268 |
| Real block 49 | 1.386432 | 1.366036 | 1.281603 |

The same balanced procedure also passes numerical checks on conditioning-heavy
and continuation inputs (`conditioning-kernels/index.json`):

| Real block-0 fixture | Packed sequence | MPSGraph | Dense A | Dense B |
| --- | ---: | ---: | ---: | ---: |
| Three image references | 24,250 | 1.582615 | 1.641212 | 1.514827 |
| 192-frame continuation prefix | 22,426 | 1.351398 | 1.402295 | 1.289388 |

Only compare candidates within each matched row; these conditioning fixtures
were measured in a later serial session. Dense A is within 5% on these inputs,
while B wins their standalone timing. Preparation/projection-inclusive regions
and full-model acceptance are measured separately below.

This revised shape improves the standalone dense result, while preserving the
historical T035 failure at 362 frames. Kernel timing alone does not qualify
production; dense B's model-level audio failure illustrates that distinction.

## Whole-step B1

Matched cached conditioning, `H3_DIT_COMMAND_BLOCKS=5`, no component fences or
captures in the following measurements:

| Backend | Denoising seconds | Speedup | Video relative L2 | Audio relative L2 | Numerical gate |
| --- | ---: | ---: | ---: | ---: | --- |
| MPSGraph | 123.606 | 1.000× | — | — | Reference |
| Hybrid dense A | 119.410 | 1.035× | 0.011062 | 0.018470 | PASS |
| Hybrid dense B | 119.534 | 1.034× | 0.011132 | 0.021038 | FAIL audio |
| Hybrid SOL | 94.075 | 1.314× | 0.374398 | 0.181594 | FAIL video and audio |

SOL is also 1.269× faster than the same hybrid with dense A, but still fails
quality against that baseline (video/audio relative L2 0.374041/0.175867).

Limits were frozen before model tuning in `tests/metal_native_limits.json`:
dense model relative L2 ≤0.02 and cosine ≥0.999; SOL model relative L2 ≤0.05
and cosine ≥0.998. Limits have not been loosened to accommodate these results.
The cache-creation B1 took 133.316 seconds of denoising; the matched load-many
reference above is the basis for comparison.

The 49 routed SOL layers select 38.66% of query/key block pairs exactly; block 0
is wholly dense. Protected/local/floor selections are included in that fraction.
Peak tracked live buffers rise from about 38.4 GiB to 39.6 GiB. This is an
observed route fraction for this prompt, not a fixed sparsity guarantee.

Dense A proceeds to B5 because B1 passes and shows a possible whole-step gain.
Dense B remains opt-in after its audio failure. SOL's conditional B5
quality/performance acceptance is not opened by a failed B1 quality gate.

## Dense B5 disposition

| Backend | Step 1 | Steady median (steps 2–5) | Steady range | Total denoising |
| --- | ---: | ---: | ---: | ---: |
| MPSGraph | 132.252 s | 126.813 s | 125.417–132.166 s | 643.460 s |
| Dense A | 131.575 s | 129.203 s | 129.103–131.136 s | 650.220 s |

Dense A is **1.9% slower** by the steady median and fails the accumulated
numerical gate: video relative L2 **0.190533**, cosine **0.981852**; audio
relative L2 **0.069513**, cosine **0.997635**. Its B1 timing advantage did not
carry through to the required B5 acceptance result. T134 therefore fails;
B6 promotion testing is not opened. This is not a claim that a component
oracle pass establishes model-level equivalence.

The frozen executable reproduces the earlier dense A B1 AV state byte-for-byte,
including in the component-profile run. MPSGraph's cache-creation, cached and
component-profile B1 runs also produce identical AV states. All B5 steps execute
50 attention calls and 200 MPSGraph linears, with ten ordinary command-buffer
submissions per step. No requested denoising evaluation was skipped.

The exact kernels remain opt-in experiments and SOL foundations. Both dense
production promotion and SOL production promotion remain unqualified.

## Sequence and protection scaling

Additional full-depth B1 diagnostics use the same settings, with a shorter
90-frame fixture or additional conditioning/protection at 243 frames:

| Case | MPSGraph | SOL | Exact block pairs | Video / audio relative L2 | Numerical gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Short, 90 frames | 33.401 s | 28.847 s | 49.41% | 0.123169 / 0.170324 | FAIL |
| Three image references, 243 frames | 147.473 s | 112.144 s | 47.30% | 0.237445 / 0.118396 | FAIL |
| 39-frame continuation prefix, 243-frame target | 130.885 s | 108.748 s | 58.66% | 0.526047 / 0.127598 | FAIL |
| 192-frame continuation prefix, 243-frame target | 130.631 s | 140.425 s | 99.39% | 0.021550 / 0.018073 | PASS |

Both continuation cases preserve the complete video/audio prefix byte-for-byte.
Their generated suffixes are checked separately: video/audio relative L2 is
0.565469/0.133501 for the 39-frame prefix (FAIL) and 0.047230/0.033264 for the
192-frame prefix (PASS, including cosine limits). An unchanged prefix cannot
dilute the acceptance metric.

Protecting 192 frames removes almost all sparsity and makes SOL **7.5% slower**.
Thus none of these cases passes both quality and speed gates, and conditional
SOL B5 testing is not opened. These are single-step scaling observations, not
production qualification: MPSGraph creates each conditioning cache and SOL loads
it. The explicit `--sol-min-exact 1` policy provides the dense fallback; this
prototype does not automatically switch from observed route density.

`scaling/index.json` retains the complete metrics. The original one-block
continuation QKV diagnostic was correctly rejected by the runtime's full-depth
requirement; `qkv-full/` retains its successful 50-block replacement. The harness
now selects full depth for continuation captures and rejects reduced-depth
continuation requests before model loading.

## Full-DiT profile and M2 decision

Preparation/projection-inclusive diagnostic wall times, summed over 50 blocks:

| Backend | Complete attention region | Complete blocks |
| --- | ---: | ---: |
| MPSGraph | 94.302 s | 132.975 s |
| Dense A | 92.050 s | 130.659 s |
| SOL | 58.318 s | 97.309 s |

These runs use component fences; the MPSGraph run additionally captures three
QKV tensors. They locate costs rather than establish a matched speedup. The
first-block boundary diagnostics also retain both candidates' complete-region
timings (MPSGraph 2.035 s, A 3.032 s, B 2.569 s) and complete-block timings
(2.858/3.869/3.402 s); compilation, captures and fences make these cold timings
unsuitable for promotion. The unfenced B1 and B5 measurements above decide
whole-step performance.

Diagnostic component fences attribute approximately 68.9 seconds to MPSGraph
attention, 19.0 seconds to QKV/norm/RoPE, 6.4 seconds to output projection and
38.3 seconds to MLP. The reference captures QKV at blocks 0/24/49; its attention
total includes that diagnostic I/O and is not a standalone acceptance timer.

The SOL profile attributes 32.7 seconds to the complete attention operator,
including 28.1 seconds exact attention, 1.27 seconds routing, 1.24 seconds
approximate attention, 0.12 seconds summaries, 0.01 seconds key statistics and
0.19 seconds merge. These child spans overlap the parent attention span and
must not be added to it. MLP takes 38.6 seconds; QKV/preparation and projection
take 19.2 and 6.4 seconds. Per-block complete-region spans, CPU encode/wait,
submissions, copies, dispatches and memory are retained in the records.

**Defer broad M2 native BF16 GEMM work on M4.** SOL demonstrates reduced attention
work and a whole-step gain, but fails the required quality proof. First improve
or replace the SOL approximation and re-run frozen gates. The MPSGraph hybrid
boundary and both exact dense primitives remain available. A successful SOL
quality result would make MLP/projections the next measured target; it does not
justify claiming that complete native DiT is already preferable. Re-evaluate
native linear scope on actual M5 hardware or after a qualified M4 profile.

## Reproduction

```sh
make bin/h3cli bin/metal_attention bin/metal_sol bin/sol_layout_tests
make test-metal-native-host test-metal-native-sanitize
python3 scripts/metal_native_bench.py --output NEW/save-b1 \
  --conditioning NEW/conditioning.h3cond --save-conditioning --case B1
python3 scripts/metal_native_bench.py --output NEW/dense-b5 \
  --conditioning NEW/conditioning.h3cond --case B5 --backend metal
python3 scripts/metal_native_bench.py --output NEW/sol-b1 \
  --conditioning NEW/conditioning.h3cond --case B1 --backend metal --metal-attention sol
python3 scripts/metal_native_kernels.py --output NEW/kernels --qkv SAVED_QKV
```

Use `--components`, `--regions`, `--capture-qkv`, `--capture-boundaries` and
`--capture-steps` only for labeled diagnostics. `metal_native_compare.py`
compares complete AV tensors; `metal_native_tensor_compare.py` compares captured
projection/block or per-step velocity/latent tensors. Neither tool runs new
denoising evaluations. B6 production promotion remains a separate quality gate.
