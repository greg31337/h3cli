# Conservative adaptive cache with FP8 and NVFP4

Status: **implemented, qualified, compared and accepted by the user**. The user
stated, "the results are acceptable," for all six outputs. This is a separate
experiment after the accepted [BF16 adaptive/SubBlock comparison](adaptive-subblock-results.md).
Its [acceptance record](../../outputs/adaptive-quant/2026-09-26-sm120/human-review.json)
binds the user's statement to these six videos and their source/build identities.

The later [SubBlock 0.75 experiment](subblock-quant-experiment.md) replaces
adaptive caching with sparse attention and keeps all 50 blocks' projections
quantized. It has its own qualification, manifest and human-review status;
the accepted adaptive comparison below remains fixed.

Open the [synchronized three-way report](../../outputs/adaptive-quant/2026-09-26-sm120/index.html)
for both triplets, all six videos, worst frames and detailed metrics. The
[JSON](../../outputs/adaptive-quant/2026-09-26-sm120/report.json),
[CSV](../../outputs/adaptive-quant/2026-09-26-sm120/report.csv) and
[attempt ledger](../../outputs/adaptive-quant/2026-09-26-sm120/ledger.json)
are preserved alongside it. Generated artifacts remain gitignored.

## Frozen comparison

Run two serial triplets on the same RTX PRO 5000 / SM120, with CUDA 13.0.3,
cuDNN 9.20 and the existing model and media libraries. Connection details stay
outside tracked files. Every video is **640×480, 90 frames, 50 steps, 24 FPS**,
seed 42, full AV decoding and the same piano prompt as the previous comparison.
The [machine-readable manifest](adaptive-quant-manifest.json) fixes the order.

| Triplet | ID | Projection precision | Cache |
| --- | --- | --- | --- |
| FP8 | F8-D0 | All blocks BF16 | Off |
| FP8 | F8-Q | All 50 blocks FP8 | Off |
| FP8 | F8-A1 | Block 0 BF16; blocks 1–49 FP8 | Conservative |
| NVFP4 | F4-D0 | All blocks BF16 | Off |
| NVFP4 | F4-Q | All 50 blocks NVFP4 | Off |
| NVFP4 | F4-A1 | Block 0 BF16; blocks 1–49 NVFP4 | Conservative |

Generate one video per row, six in total. Each triplet gets its own fresh dense
baseline; reuse that baseline for all comparisons within the triplet. Retain
failed attempts and never repeat a successful render. The two dense outputs
also provide a consistency check, not repeated-run timing statistics.

Use **resident weights for all six renders**; fail if they do not fit. The 72 GiB
device has room for the full BF16 core at this workload. This differs from the
earlier streamed-weight experiment, so its timing ratios are not interchangeable.
Reserve adaptive state and the BF16 probe weights before admitting the quantized
core. Log actual residency, native quantized calls, cache hits/refreshes, memory,
transfers and phase times. Include load, encoders, denoising, full decoding,
codec and AV-state writes in process wall time.

Prepare both sets of packed projection weights before the timed campaign,
without a video or model-weight hashing. Record preparation time separately.
Use the same verified cache for quantization-only and adaptive runs; require
zero newly prepared matrices during timed rendering. Model identity uses
file metadata. Before each render apply the same bounded metadata/header
readahead policy as the previous campaign; physical page residency is not
guaranteed. Kernel qualification may use bounded latent/operator fixtures.

Only the six fixed render subprocesses receive `H3_TEST_MAX_EVALUATIONS=50`.
The general Makefile and ordinary test limits remain six evaluations. Preserve
the separate 204-output recorded regression, including its existing video,
deadline, fixtures and goldens.

## Mixed-precision recipe

The conservative decision rule is unchanged: fresh block 0, anchored BF16
probe and suffix delta, FP32 normalized-change score, threshold 0.04, four-step
warmup, at most one consecutive hit and a final refresh. Cached tensors stay
BF16 and both output heads execute freshly on every transition.

With quantization enabled, block 0 projections remain BF16. Only blocks 1–49
use the requested native FP8/NVFP4 projection kernels; attention remains dense
BF16. A refresh must dispatch **196 quantized projections** and a hit **zero**.
The quantization-only controls retain all 200 quantized projections per
evaluation. Consequently the combined comparison changes both caching and
block 0 precision; it does not isolate caching at identical per-block precision.

Quantized adaptive execution uses quantization recipe **3** and adaptive
recipe **2**. Their checkpoint sections cross-check these identities so removing
either policy cannot silently reinterpret the cached residual. Packing stays
recipe 2: the existing packed matrices and cache keys remain reusable. Default
BF16 adaptive execution retains recipe 1, and ordinary quantization retains
execution recipe 2. Completed AV sidecars retain both policies and decode
without denoiser weights or a quantization cache.

At this experiment's frozen revision, aggressive cache plus quantization,
SubBlock plus quantization, reference conditioning, continuation, LoRA/Turbo,
thinning, token reduction and Metal adaptive execution were rejected. The later
SubBlock extension is qualified separately above. Neither default changes.

## Qualification and reporting

Require Metal/shared tests, the unchanged complete CUDA gate, projection-policy
and presentation tests, real latent stop/resume across warmup and a cache hit,
same-context cancellation recovery, and malformed mixed-policy checkpoint
rejection. Confirm actual quantized dispatch and skipped suffix work. Exercise
both resident and compressed-stream paths on bounded fixtures without videos.

Publish synchronized three-way playback for each triplet, JSON/CSV, per-step
traces, commands, source/build/runtime identities and all six existing videos.
Measure all 90 frames using SSIM, PSNR, LPIPS and temporal error; compare decoded
stereo audio using waveform and spectral metrics and AV duration. Compare the
combined output both to its dense baseline and its quantization-only control.
Retain historical similarity diagnostic failures without changing thresholds.
Record human acceptance separately from automated similarity diagnostics. The
user accepted these six results; local media, hashes, links and exact case counts
have been validated.

## Completed comparison

The campaign ran serially on the RTX PRO 5000 / SM120 with CUDA 13.0.3,
cuDNN 9.20.0.48, cuBLASLt version query 130101 and driver 595.91.07.
All six rows completed on their first render attempt. Each contains exactly
90 video frames at 24 FPS with 32 kHz stereo audio and all 50 scheduler transitions.
Both fresh BF16 baselines are byte-identical: MP4, clean AV state and presentation
sidecar. Their wall times differ by 0.8%.

| Variant | Wall seconds | Dense/wall ratio | Denoise seconds | Cache hits | Peak GPU GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| F8-D0: dense BF16 | 184.17 | 1.00× | 142.83 | 0 | 40.07 |
| F8-Q: FP8 only | 148.78 | 1.24× | 94.91 | 0 | 22.76 |
| F8-A1: conservative + FP8 | 115.33 | 1.60× | 62.18 | 18 | 23.38 |
| F4-D0: dense BF16 | 182.73 | 1.00× | 144.39 | 0 | 40.07 |
| F4-Q: NVFP4 only | 119.39 | 1.53× | 72.47 | 0 | 14.90 |
| F4-A1: conservative + NVFP4 | 90.40 | 2.02× | 48.13 | 18 | 15.67 |

Adding conservative cache reduced wall time by **22.5% with FP8** (1.29×)
and **24.3% with NVFP4** (1.32×), relative to quantization alone. Denoising
itself improved by 1.53× and 1.51× respectively. These are measured combinations;
no speedups from independent experiments are multiplied together.

Both cache variants performed 32 refreshes and 18 hits, executing 1,618 blocks
instead of 2,500. They made 6,272 native quantized projection calls versus
10,000 for quantization alone. Each hit ran the BF16 first block and fresh heads;
each refresh used 196 quantized projections. Cache reasons were identical in
count: one empty refresh, three warmup refreshes, nine score refreshes,
18 hits, 18 streak refreshes and one final refresh.

FP8/NVFP4 cache preparation took 50.70/39.15 seconds, separately from timing.
Every timed quantized render loaded its existing 200 or 196 packed matrices
and prepared zero new matrices. All weights stayed resident. Peak GPU memory
is NVML used plus reserved memory, sampled once per second; host process-tree
RSS peaked at 1.40–1.41 GiB. There were no telemetry errors or missed samples;
the largest interval was 1.002 seconds.

The phase breakdown remains relevant: packed FP8 loading took about 23.9 seconds
in both rows, while NVFP4 loading took 17.36 seconds alone and 13.13 seconds with
cache. Those observations include paging and loading effects; a single run
does not isolate their causes. Resident-weight ratios should not be compared
directly with the older streamed BF16 campaign.

## Measured output differences

The following compares each candidate with its triplet's BF16 baseline.
SSIM/LPIPS extrema cover all 90 frames; audio relative L2 covers the complete
decoded stereo waveform. Lower LPIPS and audio error indicate closer agreement.

| Variant | Minimum SSIM | Mean SSIM | Maximum LPIPS | Audio relative L2 |
| --- | ---: | ---: | ---: | ---: |
| FP8 only | 0.7003 | 0.7630 | 0.2231 | 0.2269 |
| Conservative + FP8 | 0.6817 | 0.7197 | 0.2471 | 0.2766 |
| NVFP4 only | 0.5950 | 0.6541 | 0.4016 | 0.6505 |
| Conservative + NVFP4 | 0.5861 | 0.6442 | 0.3948 | 0.6534 |

The incremental comparisons against quantization alone are:

| Combined versus quantization only | Minimum SSIM | Maximum LPIPS | Audio relative L2 |
| --- | ---: | ---: | ---: |
| Conservative + FP8 versus FP8 | 0.7695 | 0.1668 | 0.2145 |
| Conservative + NVFP4 versus NVFP4 | 0.8387 | 0.1194 | 0.2741 |

All four candidates and both incremental comparisons fail each of the unchanged
historical SSIM, LPIPS, temporal and audio similarity checks. This is recorded
in the report; no thresholds or goldens were relaxed. The three dense/self
consistency comparisons have SSIM 1, LPIPS 0, temporal error 0 and audio error 0.
PSNR, per-frame temporal error, audio cosine/spectral/level metrics and AV
durations are available in the JSON and linked pair records.

For this prompt and seed, conservative caching provides additional speed with
either format. FP8 retains closer agreement with BF16 on the reported video
and audio metrics; NVFP4 is faster and uses less memory. The NVFP4 combined row
has a slightly lower worst-frame LPIPS than NVFP4 alone, but worse mean LPIPS,
SSIM and audio error; that isolated extremum is not evidence of improved quality.
These results qualify execution and characterize differences. The user's separate
acceptance covers these six outputs at the recorded workload, prompt and seed.
Both features remain opt-in and defaults are unchanged.

## Validation record

- Metal build, shared tests, sampler tests and adaptive policy tests passed.
  CLI checks passed all four cases; campaign/accounting tests passed 16 cases.
- The complete recorded CUDA regression passed after the implementation patch
  and again after the comparison-tooling patch: all **204 outputs**, unchanged
  fixtures, goldens, deadline and ordinary six-evaluation budget.
- FP8 and NVFP4 bounded fixtures retained the 50-step schedule and executed at
  most six transitions per invocation. Fresh/stop/resume latents matched exactly
  across warmup, a hit and the next refresh. Cancellation recovery and
  resident-versus-streaming comparisons also matched exactly. Ordinary quantized
  and BF16 resident/streaming controls matched exactly.
- Both real mixed-precision checkpoint files passed 31 integrity cases each,
  including missing sections, recipe downgrades, malformed history, NaN/Inf
  cache contents and checksums. Presentation round trips retain both recipes.
- One BF16 qualification command mistakenly supplied a quantization cache while
  quantization was off. It failed before inference; the corrected command passed.
  The failed invocation, resolution and all later results are retained in the
  [qualification record](../../outputs/adaptive-quant/2026-09-26-sm120/validation/status.json).
  This did not create or repeat a comparison video.
- All six transferred videos were fully decoded locally, their clean AV states
  checked, their hashes verified, and all static/dynamic report links validated.
  Nine metric pairs each cover all 90 frames. The user accepted all six results;
  the original similarity measurements and failed checks remain recorded.

The [final regression record](../../outputs/adaptive-quant/2026-09-26-sm120/validation/final-gate-result.json)
matches the exact source and binary in the campaign identity:

- Source SHA-256: `8733eae754c52c01f33550c3435e1da2fdeb7f924cb788b62209d0bf9bcdf50b`
- CUDA binary SHA-256: `cd8c9035794f84026c9043357ae49f13c5649a24b18cd84e954d01c31515bd78`
- Golden manifest SHA-256: `fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3`

The CUDA snapshot excludes the local generated Metal shader include. Other
source files match the checkout. Model and packed-weight identity checks use
metadata, without payload hashing. LPIPS uses CPU AlexNet v0.1 with recorded
package versions and weight hashes; the report also records its metric-code hashes.
