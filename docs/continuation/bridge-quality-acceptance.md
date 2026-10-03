# Bridge quality and sampler validation: T043–T065

Bridge remains **opt-in**. The action-comparison tests did not establish
the material continuity improvement required by T065. They do establish a
reproducible negative result: no tested pilot candidate reached the declared
10% improvement threshold, and the selected candidate did not improve on the
held-out seed. No quality-supported replacement defaults are recommended.

The implementation follows [design-bridge.md](design-bridge.md). See the
[validation guide](bridge-quality.md) for commands, the fixed
[quality protocol](bridge-quality-protocol.json) for thresholds, and
[todo.md](todo.md) for task status. Hard continuation remains the default;
bridge retains eight rows, maximum strength 0.5, and the stepped profile.

## Action comparisons

Hardware: Apple M4 Max with 128 GiB unified memory. All quality renders use
256×256 video, 20 Euler steps, all 50 DiT blocks, reuse/core reuse 1, and token
reduction off. A new 90-frame walking source uses `inputs/face1.jpg` and
`inputs/body1.jpg`, seed 71, and a fixed garden/camera prompt. Each paired
batch shares its source, seed, prompt, references and cached conditioning.

Thirty continuation renders cover the declared 17-setting pilot plus hard-39,
four additional action pairs, and two held-out pairs. The harness also provides
the complete 60-setting Cartesian sweep; **that full grid was not run**.
The pilot covers lengths 4/6/8/9/10, strengths 0.25/0.40/0.50/0.65, and stepped,
linear and ease-out profiles. It was declared before measuring target quality.

The lowest-scoring stable pilot candidate was **six rows, strength 0.40,
linear**. That choice was frozen before confirmation testing; it is a test
candidate, not a recommendation. All 17 pilot candidates passed the declared
motion-retention, sharpness, background, camera, color and contrast guards.

| Action | Seed | Bridge/hard continuity score | Result |
|---|---:|---:|---|
| Unchanged walking | 72 | 0.9900 | Non-degradation passed |
| Unchanged walking | 73 | 0.9842 | Non-degradation passed |
| Walking while looking upward | 72 | 0.9888 | Small difference; below improvement threshold |
| Turn and begin running, primary | 72 | 0.9929 | Improvement gate failed |
| Turn and begin running, confirmation | 73 | 1.0229 | Improvement gate failed |
| Gradually raise both arms | 72 | 1.0166 | Slightly worse score |
| Turn and walk left | 72 | 0.9724 | Small difference; below improvement threshold |

Lower ratios are better. The changed-action geometric mean on selection seed
72 is 0.9925, also above the required 0.90. The existing eight-row/0.50/stepped
default scored 1.0044 on the primary pilot case. These measurements provide no
evidence for promoting a new default. T051 and T065 remain open.

Every comparison extracts the final 24 source frames and first 48 delivered
frames, with individual PNGs, a timestamped contact sheet, a three-second
window and diagnostic plots. Sampled visual review found closely matched
visible appearance, scene geometry, lighting and camera in the selected pairs.
Both modes performed the requested action changes. No material primary-case
transition advantage was established, so those visual transition checks are
recorded as uncertain rather than passing.

The protocol combines optical-flow acceleration and jerk with independent
stability guards. These are motion diagnostics, not semantic action or identity
scores. The runner enters the fixed left background region late in the window;
its residual-flow spike includes foreground motion and cannot establish garden
deformation. Both modes also share a late lower-frame crop despite the full-body
prompt. Fine face/hand details are limited by 256-pixel resolution. The review
is a sampled frame/plot inspection, not a human video study. Thresholds and
regions were not retuned after observing these results.
This is one source scene and character with two primary-case seeds. The hard
outputs already show coherent action changes, so this fixture does not prove
that bridge helps cases with stronger semantic restarts. It also cannot support
a claim of general multi-scene quality.

## Implementation and validation evidence

The GPU-state sampler uses the same jointly initialized AV state, compact
per-row bridge classes and timestep modulation as CPU sampling. Euler first
extrapolates raw cached velocity, then applies the row strength once. Exact
rows are not written; a sticky GPU audit detects any bit change after every
update. CPU remains the bridge default, with explicit `H3_GPU_SAMPLER=1` opt-in.

Whole-denoiser reuse 2/3 and core reuse 4/6 have dedicated equivalent CPU/GPU
cases. Core reuse always evaluates current-timestep final heads, with actual
uploaded block/head maps audited at every step. Combined velocity/core reuse,
other core intervals, custom reuse schedules and token reduction are rejected.
T060's conditional token-grouping work is not activated while token reduction
remains disabled. The quality prerequisite in T052 remains unmet even though
its GPU implementation is available for explicit testing.

The small GPU suite covers all profiles, actual patch/stereo packing, nonzero
condition offsets, reused/extrapolated raw caches, exact signed zeros and NaN
payloads, invalid ranges/classes and deliberate exact-row corruption. It also
uses the previously VAE-encoded `face1`, `body1`, `face2`, `body2`, and `2.jpg`
numerical fixtures. No new media is decoded from those fixtures.

| Check | Observed result |
|---|---|
| Bridge host suite | PASS: 16,544,096 checks |
| Continuation host suite | PASS: 103,190 checks |
| Host ASan/UBSan | PASS: bridge and continuation suites |
| GPU packing/Euler/exact-row suite | PASS: 30,208,571 checks, 60 twenty-step trajectories, bit-identical |
| GPU-wrapper ASan/UBSan | PASS: 22,656,421 checks, 45 twenty-step trajectories |
| Quality-analysis tests | PASS: eight tests, including discontinuity detection, freeze/blur guards, extraction and stale-review rejection |
| CLI negative cases | PASS: 41 invalid/unsupported configurations; no partial output |
| Direct GPU CLI render | PASS: 20 exact-row audits, valid saved state and synchronized 51-frame AV output |
| Standard `make test` | PASS: host/bridge/kernel/oracle/AudioVAE/mux checks; unavailable external fixtures skipped |
| Full-model sampler matrix | PASS: 12 renders, seven equivalent CPU/GPU comparisons |
| Full-model captured trajectories | PASS: all 21 AV states bit-identical in each of five callback-enabled pairs |
| Callback-free GPU windows 1/0 | PASS: final AV states match CPU reuse 3 bit-for-bit |
| Paired decoded video/audio | PASS: identical RGB and stereo PCM in all seven pairs |
| Core reuse maps/heads | PASS: current classes on all 20 steps; six core evaluations for reuse 4, five for reuse 6 |
| Unmasked Ref2VA/default hard/explicit hard | PASS: final AV latents, full states and MP4 bytes match pre-bridge `6c09804` |
| Ordinary T2VA/first-frame-conditioned generation | PASS: final AV latents and MP4 bytes match pre-bridge `6c09804` |
| Generated-state bridge preparation | PASS: 2,264,109 checks |
| Isolated regression source audit | PASS: 46 current and 44 original files per build match their expected sources, with only declared replay hooks |
| Unchanged-prompt quality | PASS: both seeds pass non-degradation checks and sampled visual review |
| Three-segment stationary bridge pose chain | PASS: state/AV checks and visible raise → lower → clasp progression; secondary gaze details unverified |
| Four-segment locomotion chain | FAIL: requested intermediate action is not visible after the character leaves the camera |
| Primary action/aggregate quality | FAIL: material improvement not established |

All sampler renders retain complete, finite, checksum-valid 90-frame states
and deliver 51 frames plus 68,000 stereo samples per channel, with both muxed
streams starting at zero and lasting 2.125 seconds. Exact AV prefixes remain
bit-identical after every update. CPU/GPU maximum absolute and relative L2
latent errors are zero at equivalent settings. The test driver checks parity
before advancing to the next reuse configuration.

Sampled frames from approximate reuse settings remain coherent, but core reuse
changes action timing/orientation relative to reuse 1. This is not a quality
equivalence claim. Recorded wall times include different cache/preparation
costs and are not a controlled performance benchmark.
ASan/UBSan instrument the C/Objective-C host code; Metal kernels are checked by
the numerical comparisons, bounds cases and exact-row audits.
The standalone legacy BF16 test could not run because
`misc/fixtures/h3_dit_bf16.safetensors` is absent; no local copy was found.
Its direct invocation failed to open the fixture. The standard `make test`
target subsequently passed and reported its optional fixture skips explicitly.

Full-model sampler, chain and pre-bridge regression results are recorded under
[`outputs/bridge-quality`](../outputs/bridge-quality/). Their final status is
collected by `tests/bridge_completion.py`; numerical success cannot override
the failed motion-quality gate.

The first, four-segment locomotion chain passed its numerical/state/AV checks
but **failed visible-action validation**. The runner left the locked camera's
left edge; the next segment showed an empty garden instead of the requested
run-to-walk action. The final segment introduced the character from the right
and raised her arms, without establishing a continuous character trajectory.
Its artifacts and failed visual review are preserved under `chain/`.

A separate three-segment stationary-pose fixture **passed**. It starts from the
already reviewed raised-arms bridge output, then lowers the arms and brings
the hands together. The character stays visible, with stable scene geometry,
appearance, lighting and camera across both new handoffs. The secondary gaze
instructions are not clearly established in the sampled 256-pixel frames.
All three segments use bridge mode with the same references and settings.
These results are recorded separately under `chain-pose/`; they do not replace
or turn the failed locomotion result into a pass. T061 records this bounded
primary-pose progression, not general locomotion continuity.
The exact pose-chain command is:

```sh
python3 tests/run_bridge_quality.py --phase chain --chain-style pose \
  --only chain-pose-lower,chain-pose-clasp \
  --source outputs/bridge-quality/acceptance/arms-72-linear-6-0.40 \
  --output outputs/bridge-quality/chain-pose \
  --selection outputs/bridge-quality/acceptance/selection.json
```

The assembled [pose-chain movie](../outputs/bridge-quality/chain-pose/review/chain.mp4)
contains 153 delivered frames and 204,000 samples per channel (6.375 seconds).
The [locomotion-chain movie](../outputs/bridge-quality/chain/review/chain.mp4)
contains 243 frames and 324,000 samples per channel (10.125 seconds).
Both are assembled without repeated prefixes and have verified AV start times
and durations. Their sidecar manifests identify every source RGB/PCM hash.

The final [completion report](../outputs/bridge-quality/completion.json), using
`--chain-directory chain-pose`, passes hard/normal regressions, CPU/GPU/reuse/AV
checks and the pose chain. Its **only failing gate is same-scene motion quality**.
`--require-complete` correctly exits with failure. T051, T052's quality
prerequisite and T065 remain open; T060 remains conditional and disabled.

## Reproducibility and artifacts

The [quality report](../outputs/bridge-quality/acceptance/quality.json) contains
all metrics and comparisons; the [review gallery](../outputs/bridge-quality/acceptance/review/index.html)
contains the extracted windows. The [fixed selection](../outputs/bridge-quality/acceptance/selection.json)
and [visual review](../outputs/bridge-quality/acceptance/visual-review.json)
are bound to protocol and RGB hashes. Batch manifests record commands,
configuration, references, source/output hashes and frozen executable/shader
hashes. The source provenance record explicitly notes that it was assembled
after source generation from the recorded command and artifacts.

Source state SHA-256:
`15cc8a509b9c219e91c6e4174a25cbb6c77df10660a4a80c7fc1e9b08fa5c1ca`.
Protocol SHA-256:
`d55e7c2f1bedb024947ef6872cad46b581ad5430d30c29ec3f4851495bd41e69`.

Pre-bridge regressions use commit `6c09804` and replay captured conditioning
in isolated test builds to control existing cold multimodal-encoder variability.
The production executable contains no replay hooks. Large generated artifacts
remain in the ignored `outputs/` directory.
The [source audit](../outputs/bridge-quality/source-audit.json) verifies the
isolated builds against the workspace and baseline commit. Separate cold
multimodal-encoder runs are not guaranteed byte-deterministic; quality pairs
share cached conditioning within one process, while compatibility regressions
replay captured conditioning explicitly.
