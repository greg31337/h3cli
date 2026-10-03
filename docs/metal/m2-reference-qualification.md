# M2 accepted continuation and Reference qualification

**Final M2 disposition:** the user accepted the measured speed and B6 quality,
then chose to start conservative SOL. M2 closes on the measured subset; remaining
held-out conditioning/continuation runs and range coverage are deferred. The
failed frozen screens below remain measured failures. See
`tests/metal_fp16_quality_acceptance.json` for exact evidence and scope. This
acceptance does not extend to new SOL approximation or untested cases.


The user accepted the measured **1.048811×** M4 B5 result and instructed
continuation. [The acceptance record](../../tests/metal_fp16_performance_acceptance.json)
binds that executable, its measurements and the unchanged original 1.10× limit.
The exception permits M2C/D testing; it does not turn the old performance result
into a measured pass or qualify visual quality. The renderer remains opt-in.

All renders use **640×480, 243 frames, 50 blocks** and at most six denoising
evaluations. B6 uses exactly six. No 50-step or M5 qualification is inferred.

## Identical-input diagnosis

`--capture-steps` in `scripts/metal_native_bench.py` saves each CPU Euler input,
velocity and completed latent. `--teacher-from REFERENCE_RUN` is diagnostic:
every evaluation after the first consumes the preceding saved Reference+
latent. The source geometry, seed, prompt, conditioning and schedule must match.
Captured input hashes prove that native BF16 and FP16 see identical values at
all six noise levels. These clips are not independent diffusion trajectories
and cannot qualify perceptual quality or performance.

The runtime loader requires a fresh CPU Euler run, no step/core reuse, a
six-evaluation budget and step capture. It checks exact file sizes and finite
values for both modalities before replacing either state. Malformed, truncated,
trailing and nonfinite input tests verify that failure leaves both states intact.

`scripts/metal_fp16_teacher_report.py` verifies the recorded hashes, compares
velocities and completed latents against the saved reference at every noise
level, and retains independent CPU tensor metrics. This distinguishes local
arithmetic differences from accumulation along freely evolving trajectories.
It cannot prove that every implementation error is absent; operator, layout,
block and range tests remain complementary evidence.

## Range and score telemetry

`--capture-ranges` enables diagnostic Metal reductions at twelve points in every
block: residual input, raw Q/K/V, normalized/RoPE Q/K, V, attention output,
projection output, attention residual, MLP output and residual output. Counts
include nonfinite values, magnitudes above FP16's finite range and nonzero values
below its normal range. Scans honor the interleaved raw-QKV stride and run on
the command buffer; one host readback occurs at the completed-forward boundary.

An **independent BF16/FP32 score/softmax probe** evaluates sixteen evenly spaced
query rows per head, including both ends, against every key. It reports score
extrema, shifted exponentials, normalization sums and underflow/nonfinite counts.
Its O(16×H×S) scratch is about 77 MiB at S=22,426; it never allocates S×S scores.
This samples the original attention inputs. It is explicitly **not** a trace of
every query or of the fused kernel's online-softmax internal state.

The probe is restricted to the six-evaluation test budget. Its known-value GPU
test covers interleaved Q/K/V, FP16 overflow counting, a constant score/softmax
oracle and nonfinite rejection under Metal shader validation. The separate
`tests/check_metal_diagnostics.py` checks the logged values, not only the exit
status. Diagnostic ranges never control model arithmetic. The existing recipe-4
range/recovery/checked-commit path continues to protect actual FP16 execution.

## Frozen quality screen

Contract version 1 was declared and
calibrated on Reference+ output before observing a new FP16 B6 candidate. The
[freeze record](../../outputs/metal-native-m2-243/continued/quality-contract-freeze.json)
binds the contract, metric implementation, reference media and passing controls.
The old strict numerical thresholds remain untouched.

The [current contract](../../tests/metal_reference_contract.json) is version 2
after the [AI-generated input replacement](../../inputs/README.md). It preserves
those numerical thresholds and updates the input hashes. The measurements and
freeze described here apply to version 1; version 2 requires fresh controls and
a new freeze before qualification.

Production-VAE output is decoded to exactly 243 RGB frames at 24 fps, resized
with an area filter to 320×240 and sampled every fourth frame plus the last.
Audio is decoded to 32 kHz stereo float. Fixed package versions, LPIPS weights,
YuNet/SFace models, source assets and their hashes are recorded. Metrics run on
the CPU, including LPIPS; they do not compete with GPU renders.

| Screen | Bound |
|---|---:|
| LPIPS SqueezeNet mean / 95th percentile | ≤0.06 / ≤0.12 |
| SSIM mean / 5th percentile | ≥0.90 / ≥0.80 |
| Temporal grayscale-delta RMSE | ≤0.04 |
| Mean sampled optical-flow endpoint difference | ≤1.5 pixels |
| Matched face-crop LPIPS 95th percentile | ≤0.12 |
| Aligned SFace cosine 5th percentile | ≥0.95 |
| Reference face detections / matched fraction | ≥3 / ≥0.80 |
| Mean absolute log10 audio spectrum difference | ≤0.20 |
| Audio envelope relative L2 / correlation | ≤0.15 / ≥0.98 |
| Increased clipped samples / new dropout windows | ≤0.0001 / 0 |
| Stream timing difference | ≤0.05 s |
| Relative mouth-motion/audio-envelope lag difference | ≤0.125 s |

Face screens apply to face-tagged cases. Missing detections cannot silently
pass. Face embeddings screen appearance consistency; the relative mouth-motion
proxy does not establish phoneme-level lip sync. No automated metric proves
human visual indistinguishability, and no human observation is claimed.

Positive controls are repeat decoding and bounded ±1 RGB perturbation with
16-bit PCM rounding. Negative controls are blur, horizontal translation,
reversed frames, silence and a 250 ms audio shift. Face controls include a
repeat, bounded perturbation and a missing face. All declared controls passed
their expected dispositions before the contract was frozen.

The held-out corpus includes detail/faces/hands, motion/water/smoke, ordered
image references, reference video/audio, first/last anchors and continuation
contexts 39/90/141/192. Continuations score the generated suffix separately and
the [-6,+12] frame seam window. Both modalities' inherited latent prefixes must
match the source tail byte-for-byte; a long unchanged prefix cannot dilute the
suffix score.

## Serial qualification driver

`scripts/metal_reference_qualify.py` verifies the frozen contract and accepted
performance evidence, starts with a paired independent B6 and continues through
the held-out corpus only if the frozen screens pass. Every run uses the
production VAE. It checks all 50 blocks at all six evaluations, actual arithmetic
recipes, finite complete AV states, attention range records, conditioning and
source/output checksums. It keeps diagnostic timings separate from performance
acceptance and rejects teacher-forced clips as independent quality evidence.
The accepted attention kernel/adapter sources must remain unchanged.

```sh
# Use the pinned quality environment and its checksummed CPU model files.
python scripts/metal_reference_qualify.py \
  --binary outputs/metal-native-m2-243/continued/frozen/bin/h3cli \
  --reference outputs/metal-native-m2-243/continued/calibration-reference-B6 \
  --conditioning outputs/metal-native-m1-243/conditioning.h3cond \
  --freeze outputs/metal-native-m2-243/continued/quality-contract-freeze.json \
  --output NEW_QUALIFICATION_DIRECTORY
```

Paired playback HTML, JSON metrics, raw logs, states and hashes are retained
whether a screen passes or fails. A failure gives the remaining corpus an
unqualified disposition under M2's stop criteria. It does not adjust thresholds,
promote the default backend or add a human-review approval gate.

## Six-noise-level diagnostic result

All six matched-input evaluations pass, covering 300 full blocks. The
[audited report](../../outputs/metal-native-m2-243/continued/teacher-report/report.json)
retains capture hashes, per-step metrics and complete range summaries.
[First-evaluation equivalence](../../outputs/metal-native-m2-243/continued/first-evaluation-equivalence.json)
also confirms that both native candidates' video/audio velocities remain
byte-identical to their earlier tested implementations after adding diagnostics.

| Evaluation | Native BF16 video velocity L2 | FP16 video velocity L2 | Native BF16 audio velocity L2 | FP16 audio velocity L2 |
|---|---:|---:|---:|---:|
| 1 | 0.007704 | 0.008111 | 0.005496 | 0.005217 |
| 2 | 0.018485 | 0.019071 | 0.006078 | 0.007636 |
| 3 | 0.020855 | 0.019328 | 0.010734 | 0.011849 |
| 4 | 0.011912 | 0.013011 | 0.008618 | 0.007542 |
| 5 | 0.010069 | 0.009620 | 0.005704 | 0.004932 |
| 6 | 0.008647 | 0.009088 | 0.006159 | 0.007511 |

The native implementations have comparable local error across the tested noise
levels. Together with the operator and block checks, this supports accumulated
trajectory divergence as a major contributor to the older independent B5
differences. No kernel/layout defect was found on these inputs. This is a
bounded diagnosis, not proof for every conditioning case or denoising schedule.

| Measured tensor | Largest absolute value over six evaluations |
|---|---:|
| Raw Q / K / V | 1,544 / 2,272 / 1,552 |
| Prepared Q / K | 231 / 378 |
| Attention output | 600 |
| Projection output | 115,200 |
| MLP output | 5,177,344 |
| Residual state | 13,762,560 |

No stage reports nonfinite values. These observations support the existing
attention-only mixed-precision boundary; residuals, MLP and projections cannot
be moved wholesale to FP16 from the attention-output range alone.

Actual mixed attention reports **98 recovered heads out of 16,800**, zero
invalid heads and **59,562 counted operand underflows**. Independent score probes
span **−6,239.36 to 6,462.75**, with finite shifted-softmax sums from **1 to
3,981.61**. Those probes count 17,827,974 zero exponentials and 4,682,795,353
nonzero exponentials below the normal half range; these are sampled original
BF16/FP32 softmax diagnostics, distinct from the runtime operand-underflow count.
No sampled diagnostic value feeds the denoising arithmetic.

This coverage is T2VA at six sigmas. Full-model conditioned cases remain subject
to the independent B6 quality gate; the earlier standalone reference/continuation
QKV fixtures are retained and are not mislabeled as full-model range coverage.

## Independent production-VAE B6: failed quality screen

The independent 243-frame candidate completes all six evaluations and 300
blocks with finite AV state and zero invalid attention heads. It reports 104
recovered heads and 59,676 counted operand underflows. Tracked Metal peak is
39.978 GiB. These captured B6 timings are diagnostic and do not replace the
accepted, unfenced B5 performance result.

The **frozen perceptual screen fails**:

| Screen | Measured | Required | Result |
|---|---:|---:|---|
| LPIPS mean / p95 | 0.176154 / 0.235152 | ≤0.06 / ≤0.12 | FAIL |
| SSIM mean / p05 | 0.725328 / 0.638676 | ≥0.90 / ≥0.80 | FAIL |
| Temporal delta RMSE | 0.095199 | ≤0.04 | FAIL |
| Mean flow endpoint difference | 3.874199 px | ≤1.5 px | FAIL |
| Audio envelope relative L2 | 0.228529 | ≤0.15 | FAIL |
| Audio envelope correlation | 0.986837 | ≥0.98 | PASS |
| Log audio spectrum mean difference | 0.022229 | ≤0.20 | PASS |
| Added clipping / dropout windows | 0 / 0 | ≤0.0001 / 0 | PASS |
| Audio length / A/V timing difference | 0 / 0 s | 0 / ≤0.05 s | PASS |

This initial park case is not face-tagged. Full generated-face identity,
lip-sync-proxy, Ref2VA and continuation qualification therefore remain untested;
static face controls are not substituted for those cases. The serial driver
stopped before the held-out corpus, following M2's stop criteria. No metric
threshold was changed after seeing the candidate. No human review is required.

The [final drift report](../../outputs/metal-native-m2-243/continued/final-diagnosis/report.json)
verifies the independent candidate's initial input is identical, and that each
later input follows its own preceding latent. At evaluation 6, video velocity
relative L2 is **0.296662** on that free trajectory, versus **0.009088** on the
same reference input. Final video/audio latent relative L2 is **0.364433 /
0.069340**. These remain diagnostics, separate from the decoded failure above.

Final deliverables:

* [Paired production-VAE playback](../../outputs/metal-native-m2-243/continued/qualification/review.html).
* [Detailed quality metrics and paired clips](../../outputs/metal-native-m2-243/continued/qualification/calibration/quality/review.html).
* [Reference-only calibration controls](../../outputs/metal-native-m2-243/continued/quality-controls-final/review.html).
* [Qualification index and output hashes](../../outputs/metal-native-m2-243/continued/qualification/index.json).
* [Frozen contract and pre-candidate evidence](../../outputs/metal-native-m2-243/continued/quality-contract-freeze.json).

Host checks, malformed teacher-input regressions, stale/independent-record checks,
all four source-prefix protection regressions, CPU audio-metric regressions and
ASan/UBSan pass. The diagnostic Metal known-value/nonfinite test passes shader
validation. Runtime binary/source snapshots, model signatures, commands, logs,
states, clips and metric dependencies are retained under the artifact root.

**Disposition:** T143 and T153 are complete. T154 has full T2VA layer/noise
telemetry; the user deferred its remaining conditioned coverage. T152 retains
its failed automated screen and explicit user acceptance of the observed B6
quality. M2 closes on that measured subset, and conservative SOL starts next.
The backend remains an explicit option; no untested-case or
50-step quality claim is added. ANE, broad fusion, optional native BF16 and M5
qualification retain their separate scope. The accepted playback index is
[available here](../../outputs/metal-native-m2-243/continued/accepted/review.html).
