# SubBlock 0.75 with FP8 and NVFP4

Status: **implemented, qualified, compared and accepted by the user**. The user selected
SubBlock sparsity **0.75** for both triplets. This is a separate experiment
after the accepted [adaptive-cache/quantization comparison](adaptive-quant-experiment.md).
The user stated, "the results are acceptable," for these six outputs. The
[acceptance record](../../outputs/subblock-quant/2026-09-27-sm120/human-review.json)
binds that statement to their video hashes and frozen source/build identity.

At this workload, adding SubBlock 0.75 provided **no useful wall-time gain**:
FP8 took 150.28 seconds versus 149.13 seconds alone, and NVFP4 took 116.88
seconds versus 116.93 seconds alone. Denoising was about two seconds slower
in each combined row. These are single observations, without confidence intervals.

Open the [synchronized three-way report](../../outputs/subblock-quant/2026-09-27-sm120/index.html)
for both triplets, six videos and worst-frame comparisons. The
[JSON](../../outputs/subblock-quant/2026-09-27-sm120/report.json),
[CSV](../../outputs/subblock-quant/2026-09-27-sm120/report.csv) and
[attempt ledger](../../outputs/subblock-quant/2026-09-27-sm120/ledger.json)
retain detailed timings, metrics and all attempts. Generated artifacts are gitignored.

## Fixed comparison

Generate six videos, one successful render per row, serially on the same RTX PRO
5000 / SM120 and established CUDA 13.0.3, cuDNN 9.20 environment. Keep connection
details outside tracked files. Every video uses **640×480, 90 frames, 50 steps,
24 FPS**, seed 42, full AV decoding and the same piano prompt as the preceding
comparison. The [manifest](subblock-quant-manifest.json) freezes the order and
all settings. Use resident weights throughout, failing if admission is impossible.

| Triplet | ID | Projections | Main attention |
| --- | --- | --- | --- |
| FP8 | F8-D0 | BF16 | Dense |
| FP8 | F8-Q | FP8, all 50 blocks | Dense |
| FP8 | F8-S75 | FP8, all 50 blocks | SubBlock 0.75 |
| NVFP4 | F4-D0 | BF16 | Dense |
| NVFP4 | F4-Q | NVFP4, all 50 blocks | Dense |
| NVFP4 | F4-S75 | NVFP4, all 50 blocks | SubBlock 0.75 |

Adaptive cache is off in all six rows. Both quantized rows in each triplet use
the same 200 projection matrices, including block 0. The paired comparison
therefore isolates the addition of SubBlock. Its first ten evaluations and
block 0 remain dense **attention**; block 0 projections remain quantized.
Protected query/key ranges retain the established BF16 SubBlock policy.

The combined rows must execute 50 blocks and 200 native quantized projections
on every evaluation. At indices 0–9 all 50 attention calls are dense; at
indices 10–49 each evaluation executes one dense and 49 sparse block calls,
with 98 routing launches and protected-query dense work. The fixed sequence
has 132 Q64 blocks; plan 1 keeps all 56 heads together and uses two query slabs
of at most 128 blocks, so each sparse call launches routing twice. Expected totals are 2,500
blocks, 10,000 quantized projections and 1,960 sparse block calls. Record actual
selected/possible block pairs, routing time, attention time, transfers and memory;
the requested 0.75 sparsity is not the final retained density after protection.

Reuse the existing metadata-verified packed caches. Before timing, qualify both
formats and record warm-cache preparation/loading separately. Require zero new
matrix preparations in every timed render. Never hash H3 weight payloads.
Include model loading, encoding, denoising, full decoding, codecs and AV-state
writes in process wall time; retain the same bounded metadata/header readahead.
Keep failed attempts and never repeat a completed render silently.

Only the six render subprocesses receive the explicit 50-evaluation allowance.
Ordinary tests retain their six-evaluation limit, and all 204 recorded regression
outputs, fixtures, goldens and deadlines remain unchanged.

## Execution and checkpoint identity

Quantization packing and all-block projection arithmetic remain recipe 2.
SubBlock kernels remain recipe 1 / plan 1, with the same 64-token blocks,
16-token cells, ten-evaluation warmup and protection rules. No kernel tuning
or attention-policy change is part of this experiment.

Combined execution records **quantization recipe 4** and **SubBlock execution
recipe 2**. These identify the composition, not new packing or sparse arithmetic.
The quantization, attention and SubBlock-plan checkpoint sections cross-check
the policies so removing either selection cannot silently change the trajectory.
The existing CUDA device, build, model and library identities still apply.
Completed version-5 AV presentation sidecars retain both policies and remain
decodable independently of active denoising or the packed-weight cache.

Conservative adaptive cache plus dense quantization retains its existing recipe.
Triple adaptive+SubBlock+quantization remains rejected, as do aggressive cache
plus quantization, references, continuation, LoRA/Turbo, thinning, token reduction,
existing reuse controls and Metal approximate execution. Defaults stay unchanged.

## Qualification and analysis

Require Metal/shared builds and tests, the complete recorded CUDA regression,
policy/CLI ordering checks, version-5 presentation round trips and adversarial
real-checkpoint tests. Bounded 640×480 / 56-frame latent fixtures retain the
50-step schedule and execute no more than six transitions per invocation.
Check exact warmup agreement with quantization alone, stop/resume across index
10, same-context cancellation recovery, resident/streaming agreement and exact
dense equivalence for zero-sparsity SubBlock after the warmup boundary.

The report provides synchronized three-way playback for each triplet,
JSON/CSV, all six existing videos, commands, phase/step traces and source/build
identities. Compare each candidate with its fresh BF16 baseline and each combined
row with quantization alone. Compare the two dense baselines for consistency.
Use the same all-90-frame SSIM, PSNR, LPIPS, temporal and stereo audio metrics
as the prior experiment; preserve all historical diagnostic failures. These are
single observations without confidence intervals. Human acceptance is recorded
separately from similarity diagnostics. Verify local hashes, full media decoding,
AV-state integrity, exact case counts and every report link.

## Completed comparison

The six renders ran serially on RTX PRO 5000 / SM120 with CUDA 13.0.3,
cuDNN 9.20.0.48, cuBLASLt version query 130101 and driver 595.91.07.
Each video completed on its first render attempt. Every output has 90 frames
at 24 FPS, 32 kHz stereo audio and all 50 scheduler transitions. Both BF16
baselines have identical MP4, clean AV-state and presentation-sidecar bytes;
their wall times differ by 0.7%.

| Variant | Wall seconds | Dense/wall ratio | Denoise seconds | Attention seconds | Peak GPU GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| F8-D0: dense BF16 | 187.93 | 1.00× | 143.48 | 30.75 | 40.07 |
| F8-Q: FP8 only | 149.13 | 1.26× | 95.06 | 30.82 | 22.76 |
| F8-S75: SubBlock 0.75 + FP8 | 150.28 | 1.25× | 97.05 | 32.81 | 23.26 |
| F4-D0: dense BF16 | 189.17 | 1.00× | 142.47 | 30.54 | 40.07 |
| F4-Q: NVFP4 only | 116.93 | 1.62× | 72.23 | 30.74 | 14.90 |
| F4-S75: SubBlock 0.75 + NVFP4 | 116.88 | 1.62× | 74.24 | 32.72 | 15.40 |

FP8 plus SubBlock was **0.8% slower overall** than FP8 alone. NVFP4's 0.05%
wall-time difference is effectively a tie in single observations. Its combined
row loaded the transformer in 11.89 seconds versus 13.79 seconds alone,
offsetting the roughly two-second denoising penalty. Denoising increased by
2.1% for FP8 and 2.8% for NVFP4. Loading and paging variation remains included
in wall time, so the tiny NVFP4 difference does not establish an inference gain.

Attention time increased by approximately two seconds in both combined rows.
Routing consumed 5.18 seconds with FP8 and 5.16 seconds with NVFP4; sparse
kernel time was 15.24/15.20 seconds. The full attention totals also include
dense warmup, the dense first block, protected queries, pooling and other
dispatch work. At this shape, routing and protection costs erase the sparse
kernel's potential saving. This observation does not establish performance at
other resolutions, lengths or GPU architectures.

Each combined row executed 2,500 blocks, 10,000 native quantized projections,
1,960 sparse attention calls, 540 dense attention calls and 3,920 routing
launches, with adaptive cache off. Retained pair density during sparse calls
was **45.604% for FP8 and 45.624% for NVFP4**, after budget rounding and protected
key ranges. Requested sparsity 0.75 therefore does not mean 75% of attention
work was removed. Both combined rows added 0.50 GiB to sampled peak GPU memory.

Both packed caches were already warm; qualification recorded preparation/loading
separately at 21.65 seconds for FP8 and 13.26 seconds for NVFP4. Every timed
quantized render loaded 200 cached matrices and prepared
zero new matrices. All weights stayed resident. Peak GPU usage includes NVML
used plus reserved memory sampled once per second; host process-tree RSS peaked
at 1.39–1.49 GiB. No telemetry or attention-timing samples were missed; the largest
sampling interval was 1.002 seconds. Denoising H2D/D2H counters were each
157,440,000 bytes in every row. Full phase and memory traces are in the report.

## Measured output differences

These measurements compare each candidate with its triplet's BF16 baseline.
SSIM/LPIPS extrema and means cover all 90 frames; audio relative L2 covers the
complete decoded stereo waveform. Lower LPIPS and audio error mean closer
agreement, without establishing perceptual acceptance.

| Variant | Minimum SSIM | Mean SSIM | Maximum LPIPS | Audio relative L2 |
| --- | ---: | ---: | ---: | ---: |
| FP8 only | 0.7003 | 0.7630 | 0.2231 | 0.2269 |
| SubBlock 0.75 + FP8 | 0.7053 | 0.7458 | 0.2367 | 0.3386 |
| NVFP4 only | 0.5950 | 0.6541 | 0.4016 | 0.6505 |
| SubBlock 0.75 + NVFP4 | 0.5727 | 0.6307 | 0.4243 | 0.8575 |

The incremental comparisons against quantization alone are:

| Combined versus quantization only | Minimum SSIM | Maximum LPIPS | Audio relative L2 |
| --- | ---: | ---: | ---: |
| SubBlock + FP8 versus FP8 | 0.7106 | 0.2100 | 0.3474 |
| SubBlock + NVFP4 versus NVFP4 | 0.7414 | 0.2129 | 0.6302 |

All four candidates and both incremental comparisons fail every unchanged
historical SSIM, LPIPS, temporal and audio similarity check. The three dense/self
consistency comparisons have SSIM 1, LPIPS 0, temporal error 0 and audio error 0.
The complete JSON retains PSNR, per-frame temporal errors, stereo waveform and
spectral measurements, levels and AV durations. No threshold or golden changed.

SubBlock introduces further visual and audio differences without a useful
timing improvement here. FP8's slightly higher minimum SSIM is accompanied
by lower mean SSIM and larger LPIPS/audio errors; it does not establish better
quality. NVFP4 plus SubBlock has larger differences on all four summary columns.
The user accepted these six videos at their recorded settings. Both options remain opt-in;
defaults and the earlier accepted adaptive-cache experiment are unchanged.

## Validation record

- Metal/shared builds, ordinary tests, sampler tests and adaptive policy tests
  passed. CLI tests passed all four cases; the three campaign/accounting suites
  passed 20 cases in total.
- Complete CUDA regression runs after the implementation, comparison tooling
  and routing-accounting correction each passed all **204 recorded outputs**.
  Fixtures, goldens, deadlines and the ordinary six-evaluation limit are unchanged.
- Bounded FP8 and NVFP4 latent fixtures passed exact warmup agreement with
  quantization alone, stop/resume across the sparse-attention boundary,
  same-context cancellation recovery and resident/streaming agreement.
  Zero-sparsity SubBlock matched dense quantization exactly, including after
  warmup. The [qualification record](../../outputs/subblock-quant/2026-09-27-sm120/validation/status.json)
  retains all 27 commands and 34 checks; real checkpoints passed 31 integrity
  cases per precision, including missing sections and recipe downgrades.
- The first FP8/SubBlock render completed successfully, then a campaign
  assertion incorrectly expected one routing launch per sparse block. At this
  fixed 90-frame shape, the native planner uses two query slabs, producing 98
  routing launches per active evaluation. The validator and its tests were
  corrected; the full CUDA gate passed again. The existing video's hashes,
  complete decode, AV state and dispatch records were revalidated before
  resuming. No render was repeated. The original failed validation, correction
  and qualified tooling identity are retained in the
  [recovery record](../../outputs/subblock-quant/2026-09-27-sm120/validation/router-accounting-recovery/record.json).
- All six transferred videos passed local full media decoding, clean AV-state
  integrity and dispatch checks. All 102 transferred asset hashes matched;
  nine metric pairs each cover all 90 frames, and every static/dynamic report
  link resolves. The [local verification record](../../outputs/subblock-quant/2026-09-27-sm120/validation/local-verification.json)
  also confirms exactly one render per row and verifies all 142 assets of the
  earlier accepted adaptive/quantization comparison remain unchanged.

The frozen renderer is identified by the
[render-build regression](../../outputs/subblock-quant/2026-09-27-sm120/validation/final-gate-result.json):

- Render source SHA-256: `241ff28e092b91c8283079de6c0da0745312d0be3007a07e38d1992c0dc2ecb1`
- Render binary SHA-256: `a3358704c428dcecbe292c91ca885b7e5e4107099f6260f683d520af9c861c61`
- Qualified tooling source SHA-256: `af88f90371adf63c57de4269e18075719b404cc47c884a30cae98a7d7f370f91`
- Golden manifest SHA-256: `fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3`

The tooling correction changes only `tests/cuda_subblock_quant.py` and
`tests/test_subblock_quant_campaign.py` relative to the frozen render source.
All production sources and the render binary remain identical across six rows.
The local generated Metal shader include is excluded from CUDA source snapshots.
Model and packed-weight checks use metadata, without weight payload hashing.
