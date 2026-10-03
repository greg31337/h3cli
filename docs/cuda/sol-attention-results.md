# CUDA SOL implementation and results

Completed 2026-09-23. All planned implementation tasks and final automated
qualification gates passed, including the recorded checkpoint retry. The
implementation remains opt-in; default CUDA attention is unchanged.
See the [task list](../todo.md),
[design](design-sol-attention.md) and [implementation audit](sol-implementation-audit.md).

Playback: [all cases](../../outputs/cuda-sol/index.html),
[640×480](../../outputs/cuda-sol/640.html),
[1344×768](../../outputs/cuda-sol/1344.html).
The gallery contains 26 complete videos and retains the original failed
resume attempt. [CSV](../../outputs/cuda-sol/results.csv) and
[JSON](../../outputs/cuda-sol/results.json) include every render attempt;
[coverage](../../outputs/cuda-sol/coverage.json) and the
[delivery audit](../../outputs/cuda-sol/delivery-audit.json) record checks and budgets.

## Build and use

```sh
make -j8 CUDA_ARCH=120 CUDA_SOL=1
./bin/h3cli -d models/MiniMax-H3 -p "A sailboat on a calm lake." \
  --width 640 --height 480 --frames 243 --steps 2 \
  --cuda-attention sol --sol-min-exact 0.5 -o outputs/sol.mp4
```

The finite calibration selected `min_exact=0.5`. The runtime default remains
0.75. Other CUDA defaults are Q32/KV64, tau 1, one dense initial evaluation,
one dense layer, local radius 1 and no sigma cutoff. Q64 is supported;
`--sol-min-exact 1` explicitly uses the existing dense implementation.

SOL is independent of SageAttention. It operates on BF16 Q/K/V with BF16
centroids/probabilities and FP32 softmax state/accumulation. Its centroids
approximate distant blocks; it is not exact attention. Text, image/video
references, audio, target anchors and continuation prefixes stay protected.
Workspace reservation is fixed at 512 MiB per DiT context before weight-cache
admission; the active deterministic plan generally uses much less.

For exact resume, supply `--resume-sampler-state` and let the checkpoint restore
its policy. CLI generation controls are rejected on resume. New completed-state
continuation segments may select a different policy. Completed SOL AV state
contains provenance and can be decoded with a build that omits SOL.
The `--still` CLI keeps its existing dense-BF16 restriction. Fully protected
still geometry is covered at the operator level, not as a generated-image claim.

## Measurement conditions

- RTX PRO 6000 Blackwell Server Edition, SM120, 97,887 MiB, driver 595.91.07,
  CUDA 12.8.93. These are not RTX 5090 measurements.
- Original BF16 checkpoint, all 50 blocks, two steps, full VAE, reuse/core reuse
  both 1, fast CUDA off for matched render pairs. The first evaluation is dense;
  49 layers in the second evaluation route through SOL.
- Dense and SOL arms have identical prompts, seeds, references and CPU Euler
  selection. Held-out R1 has byte-identical second-evaluation inputs, allowing
  teacher-forced velocity comparison without another render. CUDA Euler is
  covered separately by the checkpoint-resume test.
- Wall time includes process startup, conditioning/reference preparation,
  denoising, full video/audio decoding and encoding. Step timers exclude those
  other phases. Device and process VRAM are sampled with NVML at 50 ms; retained
  records include actual sampling gaps. Startup is observed at approximately
  one-second resolution; post-denoising time combines decoding and delivery.
  Unknown phase timings, including resumed startup, remain blank.
- Video references use `--ref-silent-video`; generated stereo audio is still
  decoded and validated. Standalone reference-audio protection is covered by
  host metadata and synthetic kernel checks.
- One complete render per arm: speedups are provisional. Two-step video/audio
  inspection cannot qualify a 50-step final renderer.

## Completed matched pairs

| Case | Geometry / frames | References | Dense wall, s | SOL wall, s | Dense step 2, s | SOL step 2, s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| R1 | 640×480 / 243 | None | 213.54 | 193.27 | 30.76 | 11.24 |
| R2 | 640×480 / 243 | Nine images, match | 241.78 | 218.42 | 41.30 | 17.93 |
| R3 | 640×480 / 243 | Nine images, max | 370.90 | 326.32 | 83.04 | 39.01 |
| R4 | 640×480 / 158 | Nine images + three two-second videos, max | 424.68 | 371.02 | 103.91 | 50.51 |
| R5 | 1344×768 / 243 | None | 919.30 | 740.70 | 270.50 | 91.23 |
| R6 | 1344×768 / 362 | Nine images, max | 2178.49 | 1674.30 | 790.32 | 285.82 |

<!-- memory-table -->
### Sampled peak memory

| Run | Device VRAM, GiB | Process VRAM, GiB | Host RSS, GiB | Maximum sampling gap, ms |
| --- | ---: | ---: | ---: | ---: |
| R1-default | 41.576 | 40.947 | 1.935 | 50.3 |
| R1-sol | 42.076 | 41.447 | 1.938 | 50.5 |
| R2-default | 43.161 | 41.986 | 1.950 | 420.5 |
| R2-sol | 43.117 | 42.488 | 1.951 | 50.4 |
| R3-default | 45.019 | 44.391 | 2.029 | 50.8 |
| R3-sol | 46.065 | 44.891 | 2.027 | 372.0 |
| R4-default | 45.968 | 45.340 | 1.922 | 50.8 |
| R4-sol | 47.016 | 45.842 | 1.926 | 368.8 |
| R5-default | 51.308 | 50.680 | 4.752 | 390.2 |
| R5-sol | 51.808 | 51.180 | 4.760 | 73.7 |
| R6-default | 62.114 | 60.939 | 6.574 | 420.0 |
| R6-sol | 62.614 | 61.439 | 6.569 | 385.1 |

Peaks are sampled, not an allocator-wide guarantee of the instantaneous maximum.
<!-- /memory-table -->

R1 passes the fixed performance gate: **2.74× routed-step speedup** and
**9.49% lower complete-process time**. Both generated media streams validated.
The initial dense step, startup and VAE explain why total savings are smaller.
R5 improves its routed step **2.97×** with **19.43% lower wall time**. R6 improves
that step **2.77×** with **23.14% lower wall time**. The largest active SOL plan
uses 25.49 MiB inside its fixed 512 MiB reservation.

## Numerical and kernel evidence

All 35 final synthetic operator cases pass the independent oracle. Coverage
includes both output layouts, Q32/Q64, arbitrary tails, exact/mixed/empty
branches, constant/zero/extreme inputs, multiplicity, protected queries and
keys, invalid inputs, guards, aliases, multiple head groups/slabs and 100
repeated dispatches. Tiny CUDA memcheck and racecheck cases pass.

Two real H3 captures contain 22,429 rows, 56 heads and head dimension 128.
Approximate-operator relative L2 is approximately 0.0017 against the independent
reference. One warmup and five timed replays distinguish the three paths:

| Captured block | Existing dense, s | Forced-exact SOL kernel, s | Routed SOL, s |
| --- | ---: | ---: | ---: |
| 1 | 0.564855 | 0.241987 | 0.165085 |
| 25 | 0.567942 | 0.241958 | 0.163555 |

All retained real routing-window bits match the independent router. Only the
last bounded head/query window is exported per real capture; this is not a
full-sequence routing-mask comparison. Small synthetic cases compare full
masks, and the large multi-group/slab case compares sampled outputs across all
heads. The old oracle's unexported-mask `route_match=true` placeholder is not
counted as mask evidence. See the explicit
[route ledger](../../outputs/cuda-sol/coverage.json) and
[real replay audit](../../outputs/cuda-sol/replay-review.json), including all
five timing samples, standard deviations and independent dense checks.

These show both a faster native kernel and an additional gain from routing.
Only about 1.9% of the mixed-reference R4 pairs are approximate, yet its
second step improves 2.06×; the native kernel accounts for much of that gain.
The public all-exact setting deliberately keeps the original dense backend;
the forced-exact SOL path is a diagnostic control.

Held-out R1 velocity passes the frozen relative-L2 ≤0.05 and cosine ≥0.998
screen independently for both modalities:

| Modality | Relative L2 | Cosine |
| --- | ---: | ---: |
| Video | 0.034456 | 0.999408 |
| Audio | 0.009482 | 0.999956 |

Both local macOS and remote Linux host tests pass 430,919 layout assertions,
covering every supported continuation prefix through 345 frames, mixed
boundaries, reference modalities, anchors and partial key blocks.

## Runtime compatibility and output checks

| Area | Result and scope |
| --- | --- |
| CLI/API and serialization | 114 policy/planner, 1,684 prepared/sampler and 208 container assertions pass; old presentation versions and default/Sage metadata remain readable |
| Allocation ownership | Seven context lifetimes, two injected allocation failures and 100 repeated operator dispatches pass; free device memory is identical after each tested teardown |
| Dense bypass and still geometry | Public all-exact bypass is unchanged; all 315 still-layout rows stay protected, with zero approximate pairs and dense relative L2 0.00445 |
| Continuation | 39-frame prefix, 243-frame internal schedule, 204 delivered frames; both modalities preserve the defined initialized prefix byte for byte |
| Checkpoint resume | Pause after evaluation 1, resume evaluation 2; final video/audio latents match uninterrupted CUDA Euler byte for byte, and absolute-step SOL policy is restored |
| Fast CUDA and streamed weights | Tiny complete two-step SOL renders pass; resident mode is used throughout the main matrix |
| FP8/NVFP4 | Both projection interfaces pass; the complete NVFP4 dense/SOL smoke pair passes at 128×128/22 frames |
| Decode without SOL | A build omitting SOL decodes the completed SOL AV state successfully and rejects new SOL generation |
| Media/lifetime | All 26 complete outputs pass full decoding, frame count, geometry, 24 fps, stereo 32 kHz audio and duration checks; all owned children exit and GPU ownership is released |

The tiny fast/streaming/resume/NVFP4 shapes route through the native SOL kernel
but protection leaves zero approximate pairs. They establish compatibility,
not sparse performance at those settings. The NVFP4 dense arm includes cold
cache packing and the SOL arm reuses that cache; their wall times must not be
compared as a speedup. No Sage/SOL combination is introduced.

Complete-clip SSIM for R1–R6 ranges from 0.895 to 0.954. Audio diagnostics are
retained separately from the teacher-forced velocity gate; decoded audio
relative L2 spans 0.045–0.218 for these six pairs. No LPIPS score or human
quality approval is claimed. R6 contact sheets show substantial blur and
streaking in both the dense and SOL two-step outputs. These are diagnostic
renders, not final-quality samples, and neither SSIM nor the velocity screen
establishes 50-step perceptual parity.

## Reproducibility and budget

The measured executable remains at `/path/to/h3-cuda-sol/bin/h3cli` on the tested
node. Its SHA-256 is
`b44ba3372fe98eac12f22b53b519bde8966c1dba51d6fb1beee30b5c77e038a7`.
The [identity manifest](../../outputs/cuda-sol/identity.json) retains source,
model and asset identities; every run has its exact command, environment,
timeout, timestamps and telemetry. The pre-repair source archive and binary
identities remain available in the audit records. No model, raw QKV, latent
or checkpoint files were downloaded.

The clock started at **16:35:34 UTC**, before remote build verification.
Verified local delivery at **19:44:01 UTC** took **3 hours 8 minutes 27 seconds**,
well below eight hours. Every generation uses a two-step schedule; the paused
case stops after its first evaluation. Reporting/downloads remain inside the
shared 30-minute closing allowance. All unrecorded elapsed time is charged
conservatively to the 35-minute contingency by the delivery audit.

| Bucket | Recorded child wall, minutes | Cap, minutes |
| --- | ---: | ---: |
| A: environment/build/contracts | 0.62 | 25 |
| B: synthetic/sanitizers | 0.20 | 35 |
| C: calibration/replays | 14.45 | 35 |
| D: six render pairs | 131.21 | 255 |
| E: compatibility, including failed pause and retry | 15.11 | 50 |
| F: quantization | 2.19 | 15 |
| G: report subprocesses | 0.41 | 30 |

These are child-process totals, not a claim that orchestration was free.
The coverage ledger measures their interval union, identifies overlapping CPU
work, and keeps the global clock running through media checks and delivery.
The final local audit verifies **917 files** and **439 gallery links**. At that
audit, closing work consumed 5.30 minutes and conservative contingency consumed
24.40 minutes, below their respective 30- and 35-minute caps.

## Retained failures and scope

The initial build command used an unsupported `CUDA_ARCH=sm_120` spelling;
`CUDA_ARCH=120` succeeded. Development testing prompted one bounded kernel
repair: snapshot the device error flag uniformly per CTA before entering
barriers. The original identity/results remain retained; qualification uses
the repaired binary and rerun operator suite.

The existing sampler fixture could rewrite a file within the same filesystem
timestamp tick. Tests now separate those fixture writes by 10 ms; production
fingerprinting is unchanged. A platform-specific CLI expectation was also
corrected without altering runtime behavior.

The initial continuation checker compared the video prefix directly to its
parent. Existing hard continuation deliberately initializes that prefix as
`0.999 × parent + 0.001 × seeded noise`, using separate F32 products. A CPU
audit of the saved states verified both dense and SOL prefixes byte for byte
against that formula; audio prefixes match the parent exactly. The original
failed expectation remains in `prefix-checks.json`, and the corrected contract
audit is in `prefix-checks-final.json`. No GPU rerender was needed.

The first tiny pause test reached its 240-second cap during cold model
fingerprinting, before denoising. Its dependent resume attempt therefore had
no checkpoint. The timeout and dependent failures are retained. Fingerprint
preparation and one tiny retry are recorded separately, with the same binary,
rendering caps and numerical criteria.

Sage/SOL composition, other GPU architectures, long-schedule visual quality
and external Metal/CUDA performance remain outside this qualification. No
existing dense default is promoted or replaced by these results.

The initial missing-checkpoint rejection records do not count as successful
policy tests. The retry uses a valid checkpoint and checks the specific policy
error. Expected nonzero exits for build-capability and policy rejections remain
in the failure ledger, alongside unexpected development failures. The final
coverage gates distinguish them from unresolved failures; none remain in the
planned qualification scope.
