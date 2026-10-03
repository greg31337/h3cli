# Shared CUDA fast mode and attention qualification

Historical record: renderer modes and legacy test launchers described below have been
retired. Retained artifacts preserve the original results. Use the
[single-pipeline design](design-single-pipeline.md) and
[current qualification](single-pipeline-final.md) for supported commands.

Work starts from `1277115` (M0/M1). The unchanged reference suite remains pinned
to `4ac45c94f1cbec276333102676ce1600ead71da025df4ca60b9ae2343098e413`.
Hardware: qualification RTX PRO 5000 72GB, SM120, driver 595.91.07; original model under
`/path/to/models/MiniMax-H3`. No dependency upgrades or H3 checkpoint-weight hashing.

## Construction and measured priorities

The unoptimized fast-v2 C0 run used native conditioning and the shared SGLang Euler sampler,
640×480, 124 frames, six evaluations, and full AV decoding. All **119** retained
preparation, trajectory and decoded-media artifacts matched the frozen native
reference exactly. This is a construction proof, not a public third mode.
The existing arithmetic-4 preparation, sigma/noise, Euler, FP32-sensitive heads
and full decoder remain shared.

Its instrumented profile measured 22.66 s text encoding, 10.53 s AdaLN
precomputation, 46.01 s core loading, 27.61 s denoising and 7.82 s video decoding.
The DiT transfers 241.92 GiB over the complete request, with 216.05 GiB streamed
weights. Instrumentation and captures disable some decoder graph optimizations;
these numbers identify costs and do not qualify latency.

Full device residency is not the default optimization: roughly 36 GiB of DiT
weights would exceed the frozen peak-VRAM ratio relative to the existing streamed
renderer. The existing explicit diagnostic remains available for investigation.

The first exact substitutions are request-scoped, separate from old fast flags:

- Pin private file mappings, avoiding a separate pinned allocation and
  `pread` copy. The mapping is privately writable because this driver rejects
  read-only registration; it cannot write back to the model file. Keep the metadata cache keys, process-wide 40 GiB pinned-weight
  budget, memory admission, streamed slots and release fences. Unsupported
  registration uses the existing bounded copied-weight path.
- Combine three range-checked FP16 Q/K/V casts in the full VAE into one kernel,
  preserving every conversion and nonfinite check. FP32 residuals, the original
  full decoder, preview VAE and noise schedule are unchanged.

The internal `H3_TEST_FAST_BASELINE=1` diagnostic disables these substitutions
for construction comparisons. It is not a supported production preset. Older
blanket GEMM/norm/RoPE/TF32 tuning is not automatically restored; every promoted
substitution must pass the frozen fast-quality and performance contract.

## Validation boundaries

`tests/cuda_two_modes_run.py` owns fresh result directories, exact source/binary
identities, pinned runtime setup, six evaluations, complete AV delivery, sampled
NVML/process memory and bounded timeouts. It supports the retained conditioning
cases and 124/243-frame geometry. It rejects concurrent GPU jobs and stale source.
`tests/cuda_two_modes_quality.py` applies the unchanged fast contract to every
decoded frame, temporal differences and audio samples on CPU. Human review is
reported separately; it is never inferred from automated metrics.

Three interleaved 243-frame reference/fast pairs qualify the default timing and
memory policy. Complete short attention clips and real post-RoPE QKV replays
qualify explicit Sage/SOL selection. Instrumented dispatch captures are separate
from performance runs. Failure records and rejected candidates are retained.

Results below will distinguish implemented behavior, measured passes and any
unmet promotion gates. M4 quantized projections and M5/M6 migration/deletion are
outside this change.

The read-only registration prototype fell back on this driver and is retained as
a rejected optimization attempt. An 8 MiB isolated probe identified the unsupported
flag. The corrected private-mapping version passed cache invalidation/recovery
and preserved all 117 preparation/trajectory captures and the complete MP4 in
its first C0 run (97.50 s, versus 121.11 s for the earlier copied-staging run;
these captured runs are not the formal performance result). Sampled peak VRAM
was 6,675,693,568 bytes in both runs. M2’s three unchanged gates passed
in 153.81, 156.69 and 153.41 seconds after build.

## M2 measurements

The construction proof on the optimized source again matched all 117 preparation
and trajectory captures and the complete decoded MP4. Three uninstrumented,
interleaved 243-frame/six-evaluation pairs measured:

| Pair | Reference wall, s | Fast wall, s | Complete media |
| --- | ---: | ---: | --- |
| 1 | 147.662 | 136.250 | Byte-identical |
| 2 (fast first) | 145.123 | 136.315 | Byte-identical |
| 3 | 143.377 | 131.173 | Byte-identical |

The ratio of median wall times is **0.938859** (6.11% less wall time, 1.065×
speedup); peak VRAM ratio is **1.000**. Median core-loading time fell from
30.573 s to 23.904 s. Median denoising remained 65.361 s versus 65.452 s.
This is primarily a loading improvement, not a denoising-kernel speedup.

Three independent decode-only pairs used identical clean C0 AV latents. Each
produced identical MP4 bytes. The fast/reference wall ratio was **1.004644**;
the fused conversion is an exact launch reduction with no meaningful measured
decoder speedup. It is retained as part of the measured combined fast recipe,
not advertised as a separately qualified latency improvement.

The stage audit also found variation in short initialization substages: median
refinement was 0.400/0.526 s and AdaLN 6.599/7.311 s (reference/fast), while text
encoding was 15.449/15.253 s. Aggregated conditioning was slightly beyond the 5%
nonregression boundary in this sample. The timing code and these arithmetic paths
are shared; nevertheless this warrants a bounded final-source recheck rather
than silently discarding the slower boundaries. Both sets are retained.

M2 performance source identity:
`7851afc61560bace2701de9539d4a3c7f8ff5e27d6390f787d59362d4a00ca89`.
The complete evidence and inspectable clips are in the
[M2/M3 playback report](../../outputs/two-cuda-modes/m2-m3/review.html).

## M3 interface and recovery

The main-DiT boundary accepts post-normalization/RoPE **sequence-major BF16**
Q/K/V, 56 heads of dimension 128 and a positive finite FP32 scale. Its layout
flag selects only the output layout: sequence-major or head-major for projection.
All four modes validate the same shape, disjoint output, absolute block/step and
dtype contract before capture/dispatch. Qwen, refiners, encoders and VAEs do not
use this substitution boundary. No legacy BF16 scale rounding is reapplied to
shared-reference inputs.

Dense routes to the shared reference primitive. Explicit Sage2++ and Sage3 use
the existing pinned implementations, notices and 512 MiB workspace limit;
unsupported device/build/shape requests fail explicitly. Neither selection
quantizes projection weights. SOL retains its conservative defaults including
`min_exact=0.75`, packed conditioning protections, absolute steps and both noise
sigmas. Its early/protected/forced-dense bypass uses the shared reference primitive.
There is no Sage/SOL hybrid or silent Sage-to-dense fallback.

The runtime now locks even an explicitly configured dense context and emits
main-DiT dense/Sage2/Sage3/SOL dispatch counts and workspace usage without needing
profiling. Fast-effective statistics recognize the new recipe. The existing
profiling details remain available separately.

A cancellation bug was found and fixed: Sage's sticky nonfinite-input fault
survived cancellation and poisoned subsequent valid submissions. Cancellation
now synchronizes and resets that fault and pending timing events. It is never
cleared per layer, which would mask earlier failures within the same submission.
Tests inject a NaN, require submission failure, cancel, then successfully reuse
the same context with finite inputs for both Sage modes.

The M3 integration-source unchanged reference gate passed in **152.78 s** after build,
with source identity
`bde97c88ce848b1b7aa4391f055cd3d733ba474f04a0e4a789c72e0c1ef19b12`.
All frozen files remain unchanged. Mutable fast-cache/policy, attention failure,
SOL context/policy/layout and sampler tests passed. Local Metal build, 129-symbol
GPU contract and BF16 primitive checks also passed.

### Operator evidence

The 58-case operator campaign covers both output layouts, sequence tails
1/63/65/129/257, zero/constant inputs and real step-3/block-24 H3 QKV with 11,559
rows. Dense fast and forced-dense SOL matched shared reference **bit for bit**
in both output layouts. Real-QKV diagnostics, identical across layouts:

| Mode | Relative L2 | Cosine | Maximum error / reference peak |
| --- | ---: | ---: | ---: |
| Dense fast | 0 | 1 | 0 |
| Sage2++ | 0.014637 | 0.999893 | 0.047101 |
| Sage3 | 0.089944 | 0.996068 | 0.258832 |
| SOL | 0.047398 | 0.998900 | 0.312047 |

These are operator diagnostics, not a waiver of the frozen whole-video/audio
quality gates. Complete clips, ordered references and continuation are evaluated
separately; successful rendering alone does not qualify an approximate preset.

All ten complete renders passed execution, six-evaluation and AV-delivery checks:
four primary C0 choices, four ordered `max` portrait + video/audio R1 choices,
and dense/SOL continuation. Counters prove 300 main-DiT dispatches for each
dense/Sage case; SOL makes 55 shared-dense early bypasses and 245 native routed
calls. The 512 MiB attention reservation remains bounded. No requested mode was
qualified by fallback-only execution.

Continuation audit passed **48 exact checks**: video/audio prefixes in every
input and output across six updates for both modes. The 12 video-latent and
65 audio-latent prefixes remain unchanged through sampling and match between
dense and SOL. Audio equals the clean parent tail; video preserves the seeded
augmented initial prefix, not an incorrectly assumed clean-parent value.

### Quality decisions — no promotion of approximate presets

The frozen gates require every-frame SSIM ≥0.90, LPIPS ≤0.10, temporal RMS
≤0.03, audio relative L2 ≤0.05 and cosine ≥0.995. Complete-frame/audio duration
and finite-output checks pass, but **all three approximate choices fail the
primary and conditioned quality gates**:

| Case / attention | Min SSIM | Max LPIPS | Max temporal RMS | Audio relative L2 | Audio cosine | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| C0 dense fast | 1.0000 | 0 | 0 | 0 | 1.0000 | Exact pass |
| C0 Sage2++ | 0.5264 | 0.4101 | 0.0343 | 0.2643 | 0.9712 | Reject promotion |
| C0 Sage3 | 0.4744 | 0.6417 | 0.0501 | 0.5584 | 0.8350 | Reject promotion |
| C0 SOL | 0.6300 | 0.2989 | 0.0440 | 0.2074 | 0.9822 | Reject promotion |
| R1 Sage2++ | 0.8348 | 0.0465 | 0.0245 | 0.1448 | 0.9896 | Reject promotion |
| R1 Sage3 | 0.6946 | 0.1601 | 0.0259 | 0.4494 | 0.9039 | Reject promotion |
| R1 SOL | 0.8412 | 0.0428 | 0.0234 | 0.0598 | 0.9982 | Reject promotion |
| Continuation SOL | 0.9061 | 0.0356 | 0.0238 | 0.0201 | 0.9998 | This case passes |

C0 compares with the unchanged native reference; R1 compares with shared fast
dense on the same ordered inputs. A separate reference R1 replay also produced **byte-identical complete media**,
confirming that the fast-dense comparator itself remains exact. All metrics cover every decoded
frame and audio sample, using pinned AlexNet/LPIPS weights whose hashes are
recorded. Identity/action/texture/flicker/audio human review remains **unreviewed**.

Real-QKV error and whole-clip drift show why a successful integration test is
insufficient: the approximate kernels perturb an otherwise identical six-step
trajectory. The layouts, full FP32 attention scale, source inputs, dispatch and
sampler have been checked; there is no evidence here that re-enabling old fast
arithmetic would repair the quality failure. No threshold was loosened, no
fallback-only run substituted and no prior acceptance imported as a waiver.
T022–T026 are complete; **T027 is intentionally left open**. The explicit Sage/SOL
options remain opt-in; dense remains the fast default.

The single complete C0 runs took 94.96/93.51/90.85/103.54 s for
dense/Sage2++/Sage3/SOL. R1 took 216.53/168.94/150.23/281.17 s. These are descriptive
single-run timings, not three-pair speed qualifications. In particular, conservative
SOL is slower on these densely protected reference inputs. Full stage, VRAM/RAM,
dispatch and worst-frame records are linked from the playback report.

### Final recovery audit and reference gate

The final audit found the same sticky-fault problem in SOL. A new API-level test
**fails against the prior runtime** after a nonfinite submission, cancellation and
finite retry. SOL now resets the fault only at cancellation after stream completion;
it retains its immutable layout and bounded event storage. The test passes against
the fix with profiling both disabled and enabled. Sage recovery, fast host-cache
and mixed-policy tests also pass on this source.

The final source SHA256 is
`4c51f122af96a6462d726533655f22c45a0dfe83d31562a620d5996c5f5ee59e`.
Its unchanged reference gate passed all **204 artifacts**, including the six large
patch cases, in **190.61 s** after build. The source represented by the M3 video
quality records is the preceding `bde97c88…` integration revision. The final delta
only fixes SOL cancellation/recovery and adds its regression; it changes no
successful attention arithmetic and does not repair or waive the quality failures.

One additional timing audit was rejected: a reference render completed correctly,
but an NVML memory query blocked for **1.354 s**, creating a **1.375 s** sample gap
above the runner's unchanged 1 s limit. Its clips/logs and failed status are retained
under `m3-dense-recheck`; it is not counted as a passing performance set. A single
bounded final-source repeat is recorded separately, without weakening the monitor
or replacing the original valid M2 pairs.

### Final M2 timing decision — T021 remains open

The final-source repeat completed all three pairs with valid telemetry and
byte-identical complete media:

| Pair | Reference wall, s | Fast wall, s |
| --- | ---: | ---: |
| 1 (fast first) | 158.360 | 134.380 |
| 2 | 159.955 | 136.005 |
| 3 (fast first) | 154.502 | 134.080 |

| Boundary | Reference median | Fast median | Fast/reference |
| --- | ---: | ---: | ---: |
| Complete wall, s | 158.360 | 134.380 | 0.84857 |
| Conditioning, s | 21.002 | 22.400 | **1.06653 — fails 1.05 gate** |
| Denoising, s | 64.764 | 66.036 | 1.01964 |
| Full AV decode, s | 17.751 | 17.534 | 0.98774 |
| Peak VRAM, bytes | 8,737,193,984 | 8,737,193,984 | 1.00000 |

The wall-time target passes (15.14% less wall time in this set, versus 6.11% in
the original set), as do output identity, denoising, decoding and memory. However,
the aggregate conditioning comparison still fails its 5% nonregression criterion.
Qwen logs zero fast mapped-weight or fused-cast substitutions; AdaLN precedes the
new core-loading optimization and retains the shared implementation. Within-pair
conditioning ratios are 0.9473, 1.0741 and 0.9994, showing substantial timing/order
variation rather than a consistent per-pair slowdown. This is useful diagnostic
context, **not permission to change to a passing statistic after the test**.

Accordingly, T015–T020 and T022–T026 are implemented and tested, but **T021 and
T027 remain open**. M2/M3 are not reported as fully qualified. The bounded retry
is complete; all failed and passing measurements remain available. Resolving
repeatable conditioning performance and meeting the fixed approximate-attention
quality thresholds are the remaining acceptance work. No additional numerical
or performance threshold has been relaxed.

## Reproduction and scope

With the pinned build/runtime from the reference guide, reference remains the
ordinary CUDA command. Add `--fast-cuda` for the shared-base fast recipe. Select
one main-DiT attention implementation with
`--cuda-attention default|sage2++|sage3|sol`. BF16 projection weights are the only
M3-qualified precision candidate; do not infer M4 FP8/NVFP4 support from these
attention tests. Sage3's internal NVFP4 attention does not change projection
weights. Full decoding remains independent of `--preview-vae`.

Use `tests/cuda_two_modes_run.py --source BUILD --out FRESH_DIRECTORY --fast
--attention MODE` for the complete six-evaluation C0 case. Add `--case R1` for
the retained ordered portrait/video/audio corpus, `--frames 243` for performance,
or `--continue-from CLEAN_AV_STATE --capture` for continuation protection checks.
The runner accepts `--model` and `--fixtures` paths and inherits the configured
libraries. Operator replay accepts `--qkv PATH` and optional `--layout PATH`;
its exact input and executable checksums are recorded in its result file.

The mandatory unchanged gate is still `make test-cuda-reference-regression`
using the bundled fixtures and recorded golden hashes. Run it after every coherent code
change. These mutable fast tests supplement it; they cannot replace or rebaseline
the reference suite. The full original conditioning/held-out matrix, twelve
attention/weight combinations, legacy removal and final long reference replays
remain in M4–M7. No human visual acceptance is implied by this campaign.
