# Adaptive cache and SubBlock continuation qualification

Status: implementation, automated qualification and human acceptance complete.
The user accepted the results on 2026-09-28. This record follows the
[design](design-adaptive-continuation.md) and [checklist](../todo.md).

## Implemented behavior

BF16 CUDA hard continuation and bridge support conservative/aggressive adaptive
cache, SubBlock, and their combination, with image, video, audio and mixed
references. Existing controls and defaults are unchanged. These combinations
require all 50 layers, reuse/core reuse 1 and no token reduction or LoRA.
First/last anchors, quantized approximation with continuation, Metal
approximation and upscale remain excluded.

Recipe 4 selects the maximum global, generated-video suffix, generated-stereo
audio suffix and active bridge-class score. Its fixed-order FP32 reduction uses
47,288 bytes of scratch. The compact host class plan is 836 bytes; it is passed
as a kernel argument, without a separate device class tensor. Cache allocation
is `packed_rows * 5376 * 6 + 47288`; two of the three BF16 tensors persist in
checkpoints. Ordinary recipe 1/2/3 arithmetic and scratch remain unchanged.

SubBlock protects the entire prefix, all condition/reference/audio rows and
mixed/boundary blocks. Mutable bridge queries remain dense. Prefix protection
does not make cached hidden states exact: final heads still use the approximate
suffix residual on hits. The sampler checks initialized exact latent bits after
every update and scales bridge velocities once. Hits execute only block 0 and
fresh heads, without suffix routing, attention, MLPs or weight reads.

CUDA keeps its CPU-state F32 Euler sampler. Every new AV segment starts with
empty adaptive history. Same-job sampler resume restores committed tensors,
ready/streak/phase/step and conditioning without original source/reference files.
Readers validate geometry, class metadata, budget, lengths and committed
history before allocating cache payloads, including warmup metadata stored
after those payloads in the current file layout. AV 3, presentation 9, sampler
envelope 2 and adaptive section 40 v3 are retained.

## Environment and identity

Work uses the local M4 and RTX PRO 5000 72GB only, with CUDA 13.0 Update 3,
cuDNN 9.20 and the existing qualified Sage/SOL/SubBlock build. Evidence is under
`outputs/adaptive-continuation/run01/`. The starting commit was
`97375f3f419a2f4db6f75ce28a1447bd30b1ca12`, with a clean worktree.

The 12 fixtures and 204 recorded output hashes remain unchanged. Golden manifest
SHA-256: `fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3`.

| Gate | Result | Source SHA-256 | Test seconds |
| --- | --- | --- | ---: |
| [Baseline](../../outputs/adaptive-continuation/run01/baseline-gate/result.json) | 204/204 | `164bdb48dea1cacd9efa26188be396db5d660880c71b10f49757c267ebb4bc5d` | 131.30 |
| [Initial candidate](../../outputs/adaptive-continuation/run01/candidate-gate02/result.json) | 204/204 | `873d20558669f3437bf3b7208fac89b82508441d8538fbe71e74fbcd9543f942` | 111.75 |
| [Qualified candidate](../../outputs/adaptive-continuation/run01/candidate-gate03/result.json) | 204/204 | `7ca54438d19decaea0e60e6acf8c272c22c1adb029375980b0cfb76b6f3b567a` | 117.70 |
| [Post-supplement gate](../../outputs/adaptive-continuation/run01/final-gate01/result.json) | 204/204 | `7aded4cda10694da7c5febe86c71dd850b1a215cb60fa28b45ee4c245ea8178f` | 147.84 |
| [Final post-qualification gate](../../outputs/adaptive-continuation/run01/final-gate02/result.json) | 204/204 | `7aded4cda10694da7c5febe86c71dd850b1a215cb60fa28b45ee4c245ea8178f` | 121.58 |

All 717 portable source/build/test files match between checkouts. The
[inventory comparison](../../outputs/adaptive-continuation/run01/source-identity-check.json)
records the local generated Metal include and three pre-existing remote launch/
environment scripts separately. The latter remain in the regression inventory.

The comparison executable and isolated regression executables retain their own
ELF SHA-256 values. Their production build identity matches:
`H3_BUILD_ID=ea8b2b901bfe358f7e457ef6c2069cb9403aa8784283666bd21eb1f4695620c9`
and
`H3_PORTABLE_BUILD_ID=207019adfe11efe6b615c8bf5beaa68c5ea63d16f7e527090d0d3494ca875130`.
The production source and compiler-option identities are compared independently
of cross-directory executable byte hashes.

## Automated qualification

The local clean Metal build and complete retained suite pass. Focused
sampler/policy/continuation/bridge ASan/UBSan runs pass, including 4,302 sampler
checks, 247 rechecksummed corruption cases, 103,203 continuation checks and
15,945,992 bridge checks. CUDA class-score/operator checks pass. The
[main integration](../../outputs/adaptive-continuation/run01/integration01/result.json)
passes all 43 model jobs and 845 checks, including exact hidden-media replay,
all reference forms, threshold-zero equivalence, zero-strength hard/bridge
identity, contexts 39/90, all bridge profiles and cancellation/retry.
The complete CUDA `make test` retry passes (`cuda-test21.log`), followed by all
19 [current-feature lifecycle checks](../../outputs/adaptive-continuation/run01/current-cuda01/result.json).
The Metal-only attention case passes locally; the default CUDA scratch-volume
CoW skip is covered by the no-skip `/models` run in that current-feature suite.
Pre-change ordinary conservative/aggressive BF16, reference-conditioned BF16,
ordinary FP8 recipe 2, and dense hard/bridge captures match exactly in final
latents, saved cache/history and decision lines. The FP8 control uses
the preserved baseline binary and the current binary at 256×256/22 frames,
four transitions of a six-step schedule. It does not enable quantized
approximation with continuation. The [supplemental suite](../../outputs/adaptive-continuation/run01/supplement01/result.json)
passes 45 commands and 152 checks, including seven expected early CLI errors.
[Post-analysis](../../outputs/adaptive-continuation/run01/supplement01/analysis.json)
confirms eight exact decision replays and all three placement modes. The
[focused live-reuse check](../../outputs/adaptive-continuation/run01/live01/result.json)
also passes. The final post-qualification gate passes all 204 unchanged outputs.

The first hard/bridge combined-feature jobs each exercise two cache hits and
98 sparse suffix calls in six steps, with exact resumed latents/cache tensors.
These threshold-1 functional settings exercise execution paths; they are not
the frozen quality presets and do not establish visual quality.

## Extended checkpoint replay

The final binary passes hard pauses after steps 1 and 4, and combined bridge
pauses after steps 2, 4 and 5 of the original six-step schedule. This covers
warmup, real hits, sparse-phase refresh and the final pending transition.
Original source AV/reference files are hidden during resume. Final video/audio
latents, anchor/delta tensors, cache controls/history and resumed decision lines
match uninterrupted execution exactly.

A mixed image/silent-video/replacement-soundtrack/separate-audio bridge at
`--ref-image-size high` passes full/pause/resume and matches the earlier
integration capture. Actual CLI hard and bridge resumes also pass and produce
MP4s, using omitted and matching explicit controls respectively. Seven early
CLI checks cover omitted/matching controls, differing threshold/hit ceiling/
warmup, too-small budget, and a sufficient larger budget before attempting a
missing model. These are in addition to the eight hidden-media replays in the
main integration.

## Retained contexts and actual prepared-DiT reuse

The seven-pass public-context matrix passes after switching hard/bridge mode,
39/90-frame context, conservative/aggressive/off cache, SubBlock, bridge profile,
reference image and source latent identity. Cancellation, an insufficient cache
budget, missing media, missing/malformed checkpoint and injected allocation
failure all recover to exact fresh-context results. The borrowed source remains
unchanged after the deliberate identity-mutation case is restored.

Automatic partial residency intentionally replans capacity on each request, so
that matrix's eight-block cap verifies invalidation/recovery rather than hot
DiT reuse. The additional explicit-streaming run keeps two contexts within
memory and tests hard adaptive and combined bridge execution. It asserts the
canonical sampler key and compatible placement, retains the same DiT, observes
**four actual prepared-cache hits**, and matches fresh latent/cache/history
results after compact-checkpoint resume and a following new request. The new
request starts with empty history. The focused run takes 401.24 seconds; each
individual generation evaluates at most five of six scheduled steps.

The helper builds on both M4 and CUDA (`metal-context23.log`,
`cuda-context23.log`). Both test modes are now part of the supplemental runner;
the added live mode was executed separately in this campaign because the
original sequential runner had already loaded its earlier script version.
No production binary or frozen comparison artifact changed.

## Delivery and bridge scaling

For both hard and bridge combined-feature jobs, normal delivery and
`--keep-continuation-prefix` retain byte-identical complete `.h3av` files.
The normal 85-frame RGB output is exactly the debug output after its first
39 frames; stereo PCM is exactly the debug output after 52,000 samples per
channel. These checks compare decoded content independently of presentation
provenance.

[Bridge velocity diagnostics](../../outputs/adaptive-continuation/run01/supplement01/bridge-velocity.json)
cover 12 transitions across the normal/debug pair and 180 class measurements.
The maximum `abs(scaled_RMS / raw_RMS - strength)` error is `1.335508e-8`, below the fixed `2e-6`
check. Exact rows remain unchanged and active velocities receive their bridge
strength once.

## Three-segment handoff and drift review

The [hard → bridge → hard review video](../../outputs/adaptive-continuation/run01/supplement01/chain-review.mp4)
contains three newly generated 384×384 segments, each with 124 raw frames,
39-frame context, six steps, conservative threshold 1/max hits 2 and SubBlock
0.75 with warmups 2/3. Each starts with `reason=empty`; each source is the
previous complete approximate `.h3av`. The common prompt is “A woman plays
piano in warm sunlight with flowing piano music.”

All three saved states and decodes pass. The joined video has 255 frames at
24 fps (10.625 seconds) with two-channel audio lasting 10.656 seconds. Full
media decode and finite nonzero waveform checks pass. The small AV duration
difference is retained in the [media measurements](../../outputs/adaptive-continuation/run01/supplement01/chain-media.json).

Static [segment 1](../../outputs/adaptive-continuation/run01/supplement01/chain-1-contact.jpg),
[segment 2](../../outputs/adaptive-continuation/run01/supplement01/chain-2-contact.jpg)
and [segment 3](../../outputs/adaptive-continuation/run01/supplement01/chain-3-contact.jpg)
samples keep the piano, close camera framing and warm backlight. There is no
obvious cumulative scene or color change in these samples; mean RGB changes
by less than 0.006 per channel on a 0–1 scale. Hands remain soft and show
smearing, so these six-step functional clips do not establish fine hand-motion
quality. The framing does not provide facial-identity evidence.

Boundary RGB differences are 0.01237 and 0.01493 versus within-segment medians
0.01052, 0.01023 and 0.00894. Segment audio RMS rises from 0.0744 to 0.1045 and
0.1175; these measurements alone do not establish soundtrack continuity. There
is no matched dense-chain control here, so drift cannot be attributed specifically
to caching or SubBlock. This high-threshold diagnostic chain remains separate
from the frozen quality comparison; the user accepted the reported results
as recorded below.

## Weight placement

The same 384×384/124-frame hard combined job passes with all weights resident, an eight-block resident cap, and forced streaming. All three final video/audio latents and saved cache/history match the original functional reference exactly. Each executes four full refreshes and two block-0-only hits.

| Placement | Wall s | Steady refresh weight-read layers | Hit weight-read layers | Hit H2D bytes | Hit GEMMs / sparse calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| resident | 48.23 | 0 | 0 | 2,098,944 | 8 / 0 |
| partial | 78.88 | 42 | 0 | 2,098,944 | 8 / 0 |
| stream | 74.66 | 50 | 1 | 772,802,304 | 8 / 0 |

The resident and partial cases transfer only the 2,098,944-byte current latent input on a hit. Forced streaming additionally reads/transfers block 0, as required; it reads no suffix block. Startup prefetch can put one layer outside the first step’s counters, so the table uses the second full step for steady refresh traffic. These diagnostics include sampler callbacks/checkpoint writing and are not comparison-video benchmarks.

## Bounded 362-frame capacity

A 1344×768/362-frame hard continuation with one max-size image and combined
conservative/SubBlock settings completed one real transition of a six-step
schedule. It used context 39, threshold 1/max hits 2 and warmups 2/3. The default
4,294,967,296-byte adaptive ceiling admitted the full 119,962-row layout without
reducing geometry or layers.

| Resource / result | Measured value |
| --- | ---: |
| Adaptive device storage, including reduction scratch | 3,869,541,560 bytes (3.60 GiB) |
| Persistent anchor/delta tensors | 2,579,662,848 bytes |
| Saved current checkpoint | 2,723,780,597 bytes |
| Resident BF16 blocks | 23 / 50 |
| Resident block bytes | 17,726,177,280 |
| Double stream slots | 1,541,406,720 bytes |
| Planned future allocations | 44,068,122,624 bytes |
| Reserved headroom | 7,634,452,480 bytes |
| Peak live tensor bytes | 53,125,927,888 (49.48 GiB) |
| Peak pinned host bytes | 20,842,987,520 |
| First denoising transition | 152.71 seconds |
| Whole diagnostic including checkpoint | 261.46 seconds |

The transition executes all 50 blocks and both fresh heads. Video and both audio
prefixes remain byte-identical to initialization; step-0/step-1 latents are
finite and saved adaptive history is committed/ready. Text/reference rows are
included in cache admission. Recipe-4 class/reduction storage remains bounded;
the 512 MiB SubBlock allowance remains part of attention planning.

This is capacity evidence, **not a completed 362-frame render**. It exercises
the dense warmup transition, not a sparse transition or cache hit at this size.
High-resolution hit/sparse behavior and full MP4 delivery are qualified
separately at 124 frames. Actual capacity depends on free VRAM, the reference
layout and existing bridge modulation; the cache ceiling alone does not promise
that every geometry fits.

## Evidence map

| Checklist work | Implementation / evidence |
| --- | --- |
| Baseline and admission (001–004, 013) | Shared `approximate.c` layout checks; baseline gate and `baseline-latents/`; exact replay in `supplement01/` |
| Recipe, classes and execution (005–012) | `adaptive_cache.c`, `cuda_adaptive_cache.cuh`, `dit.c`; host policy and independent CUDA probe tests; `integration01/` and comparison traces |
| Identity, readers and state (014–018) | Prepared/source keys, small-metadata validation before cache allocation, current-schema host round trips and corruption tests; hidden-media replay logs |
| Diagnostics and functional tests (019–024) | Per-region scores, exact-row checks, dispatch/transfer counters; 43-job integration, CUDA operator tests and sampler sanitizers |
| Extended replay/recovery/capacity (025–028) | `adaptive_continuation_supplement.py` and retained-context helper; exact delivery, chain, capacity and separate live-reuse results above |
| Off-path preservation and suites (029–030) | Pre-change recipe 1/2/3 and dense controls; clean M4/CUDA builds, retained suites, 19 current CUDA lifecycle checks |
| Frozen videos and reporting (031–034) | Manifest, 16 successful source/candidate jobs, playback page, media checks, contact/boundary sheets and the tables below; human acceptance remains separate |
| Documentation and final gate (035–036) | README/help and current feature/state contracts; final gate identity and explicit human acceptance status |

Full local logs: `metal-clean16.log`, `metal-build16.log`, `metal-test16.log`,
`metal-sanitize16.log`, `metal-cli19.log`, and `metal-gpu-contract21.log`.
CUDA logs: `cuda-clean20.log`, `cuda-build20.log`, `cuda-probe20.log`,
`cuda-test21.log`, and `current-cuda01/result.json`.
Large sampler checkpoints, raw latent/RGB/PCM captures, baseline binaries and
quantization caches remain in the remote evidence folder. JSON, logs and
review MP4/JPEG/HTML artifacts are copied locally. These temporary preservation
captures are not new numerical golden fixtures.

## Comparison protocol

The campaign produces S01/S02 dense six-step sources and V01–V14 exactly as
specified in the design. V01–V10 use 640×480, 90 raw frames and 50 steps;
V11–V14 use 1344×768, 124 raw frames and six steps with one max-size image.
Context 39 delivers 51 and 85 new frames respectively. Only the ten named
50-step subprocesses raise the ordinary six-evaluation ceiling.

The [frozen manifest](../../outputs/adaptive-continuation/run01/comparison01/manifest.json)
fixes prompts, seed 42, policy flags, source/build/input identities
and output paths before rendering. Wall time includes process startup through
MP4 completion; shared source generation is recorded separately. The runner
verifies full media decode, geometry, stereo soundtrack and duration, preserves
failures, validates hashes before accepting resumed campaign outputs, and
produces source-tail joins with audio and contact sheets. Manifest SHA-256 is
`2db3641cdca13349f63f49d373bc13e81eb343e91bde06821a01c9192801ec9e`;
the comparison binary is
`fa469fa8675f01688aeeb8d8992e906dd9568239aacc6c48ec3df1033b3a403e`.

## Measured videos

Open the [playback page](../../outputs/adaptive-continuation/run01/comparison01/review.html) for all sources, candidates, commands, contact sheets and source-tail joins with stereo audio. All 16 MP4 hashes were verified after copying to the local review folder. Full FFmpeg decode, geometry, frame counts, finite nonzero stereo audio and AV duration checks pass.

| Source | Geometry / frames / steps | Wall seconds | Denoise seconds | Peak GPU GiB |
| --- | --- | ---: | ---: | ---: |
| [S01](../../outputs/adaptive-continuation/run01/comparison01/S01.mp4) | 640×480 / 90 / 6 | 56.35 | 16.68 | 38.69 |
| [S02](../../outputs/adaptive-continuation/run01/comparison01/S02.mp4) | 1344×768 / 124 / 6 | 299.73 | 198.65 | 48.03 |

V01–V10: 640×480, 90 raw / 51 delivered frames, 50 steps. Adaptive warmup 4; SubBlock warmup 10. Conservative is threshold 0.04/max hits 1; aggressive is 0.08/3. SubBlock sparsity is 0.75.

V11–V14: 1344×768, 124 raw / 85 delivered frames, six steps, image 1 at max. Combined uses conservative with threshold 0.12/max hits 2 and warmups 2/3. All bridge rows use context 39, eight bridge rows, strength 0.5 and stepped profile.

| Video / join | Mode / policy | Wall s | Denoise s | Wall speedup | Hits | Blocks | Sparse calls | Peak GPU GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [V01](../../outputs/adaptive-continuation/run01/comparison01/V01.mp4) / [join](../../outputs/adaptive-continuation/run01/comparison01/V01-join.mp4) | hard / dense | 194.07 | 143.97 | 1.00× | 0 | 2500 | 0 | 39.47 |
| [V02](../../outputs/adaptive-continuation/run01/comparison01/V02.mp4) / [join](../../outputs/adaptive-continuation/run01/comparison01/V02-join.mp4) | hard / conservative | 140.55 | 98.79 | 1.38× | 16 | 1716 | 0 | 39.73 |
| [V03](../../outputs/adaptive-continuation/run01/comparison01/V03.mp4) / [join](../../outputs/adaptive-continuation/run01/comparison01/V03-join.mp4) | hard / aggressive | 110.55 | 67.40 | 1.76× | 27 | 1177 | 0 | 39.73 |
| [V04](../../outputs/adaptive-continuation/run01/comparison01/V04.mp4) / [join](../../outputs/adaptive-continuation/run01/comparison01/V04-join.mp4) | hard / subblock | 196.06 | 154.91 | 0.99× | 0 | 2500 | 1960 | 39.97 |
| [V05](../../outputs/adaptive-continuation/run01/comparison01/V05.mp4) / [join](../../outputs/adaptive-continuation/run01/comparison01/V05-join.mp4) | hard / combined | 150.07 | 108.56 | 1.29× | 15 | 1765 | 1274 | 40.23 |
| [V06](../../outputs/adaptive-continuation/run01/comparison01/V06.mp4) / [join](../../outputs/adaptive-continuation/run01/comparison01/V06-join.mp4) | bridge / dense | 185.43 | 144.65 | 1.00× | 0 | 2500 | 0 | 44.37 |
| [V07](../../outputs/adaptive-continuation/run01/comparison01/V07.mp4) / [join](../../outputs/adaptive-continuation/run01/comparison01/V07-join.mp4) | bridge / conservative | 137.74 | 99.08 | 1.35× | 16 | 1716 | 0 | 44.62 |
| [V08](../../outputs/adaptive-continuation/run01/comparison01/V08.mp4) / [join](../../outputs/adaptive-continuation/run01/comparison01/V08-join.mp4) | bridge / aggressive | 105.55 | 67.67 | 1.76× | 27 | 1177 | 0 | 44.62 |
| [V09](../../outputs/adaptive-continuation/run01/comparison01/V09.mp4) / [join](../../outputs/adaptive-continuation/run01/comparison01/V09-join.mp4) | bridge / subblock | 193.91 | 155.40 | 0.96× | 0 | 2500 | 1960 | 44.87 |
| [V10](../../outputs/adaptive-continuation/run01/comparison01/V10.mp4) / [join](../../outputs/adaptive-continuation/run01/comparison01/V10-join.mp4) | bridge / combined | 144.81 | 106.01 | 1.28× | 16 | 1716 | 1274 | 45.12 |
| [V11](../../outputs/adaptive-continuation/run01/comparison01/V11.mp4) / [join](../../outputs/adaptive-continuation/run01/comparison01/V11-join.mp4) | hard / dense | 305.56 | 198.78 | 1.00× | 0 | 300 | 0 | 48.03 |
| [V12](../../outputs/adaptive-continuation/run01/comparison01/V12.mp4) / [join](../../outputs/adaptive-continuation/run01/comparison01/V12-join.mp4) | hard / combined-custom | 322.71 | 218.96 | 0.95× | 0 | 300 | 147 | 49.99 |
| [V13](../../outputs/adaptive-continuation/run01/comparison01/V13.mp4) / [join](../../outputs/adaptive-continuation/run01/comparison01/V13-join.mp4) | bridge / dense | 307.28 | 199.37 | 1.00× | 0 | 300 | 0 | 48.52 |
| [V14](../../outputs/adaptive-continuation/run01/comparison01/V14.mp4) / [join](../../outputs/adaptive-continuation/run01/comparison01/V14-join.mp4) | bridge / combined-custom | 325.71 | 219.04 | 0.94× | 0 | 300 | 147 | 50.48 |

Peak GPU memory is sampled with `nvidia-smi` every 250 ms across the subprocess and postprocessing, on an otherwise idle GPU. Step-level tensor peaks and allocator/transfer counters are retained separately in JSON/logs. These are single-run timings with shared source costs excluded; there are no confidence intervals or minimum-speedup claims.

At 50 steps, conservative caching reduces wall time by about 26–28%, and aggressive caching by about 43%. SubBlock alone is slightly slower than dense in both modes; combining it with caching is slower than the corresponding conservative-only variant. The short target suffix and dense protection retain about 85% of attention pairs at 640×480 and about 84% at 1344×768. Routing and protected-query overhead outweigh the sparse saving in these cases.

Neither six-step combined reference variant takes a cache hit. For example, V12 step 4 has global score 0.1134 but generated-video score 0.2239, above the frozen 0.12 threshold. The per-region maximum correctly forces a refresh. Both still perform real sparse suffix attention after warmup. No variant was retuned or rerendered to improve performance.

### Dense-relative diagnostics

| Candidate | Matched dense | Video SSIM | Decoded stereo relative L2 |
| --- | --- | ---: | ---: |
| V02 | V01 | 0.988837 | 0.0419 |
| V03 | V01 | 0.987695 | 0.0644 |
| V04 | V01 | 0.982328 | 0.0758 |
| V05 | V01 | 0.981225 | 0.0727 |
| V07 | V06 | 0.990133 | 0.0553 |
| V08 | V06 | 0.989511 | 0.1043 |
| V09 | V06 | 0.979784 | 0.1599 |
| V10 | V06 | 0.979995 | 0.1218 |
| V12 | V11 | 0.957061 | 0.2487 |
| V14 | V13 | 0.956717 | 0.2856 |

SSIM and waveform differences describe divergence from the matched dense clip. They are not perceptual acceptance thresholds. In particular, a larger audio difference does not establish whether a soundtrack is better or worse. Human acceptance is recorded separately below.

### Boundary, identity and soundtrack review

[Boundary sheets and waveform measurements](../../outputs/adaptive-continuation/run01/comparison01/boundary-analysis.json)
are linked from the playback page. Static contact sheets for all candidates,
and sampled source-to-suffix boundaries, preserve the same beach, subject,
dress/hair appearance and horizon. The bridge clips depict the requested turn
and smile; hard clips continue the walk. Dense and approximate samples follow
similar poses, with small hair, face and texture differences. No obvious new
large discontinuity appears in these sampled frames. This is a static-frame
inspection, not a claim of smooth playback, precise facial identity or listening
approval; the six-step sources themselves have softer detail.

At 64×64 decoded resolution, source-last to suffix-first mean RGB differences
range from 0.01085 to 0.01201 on a normalized 0–1 scale. Median within-suffix
frame differences range from 0.00540 to 0.00828. Both dense and approximate clips
have a larger join difference than a typical internal frame transition. These
measurements flag the join for human review; they do not define a pass threshold.

The stereo tracks decode completely and remain finite and nonzero. The first
100 ms of the small hard suffix has RMS 0.0120–0.0196 versus 0.0514 in the source's
last 100 ms; dense also shows this level change. Small bridge suffixes start at
0.0258–0.0295. The high-resolution source tail is 0.0142 and targets start at
0.0165–0.0171. Maximum per-channel boundary sample jumps are 0.0076 in the small
clips and 0.0236 in the high-resolution clips. This does not establish perceptual
soundtrack continuity. Listen to the linked joins for level changes, clicks,
ambience continuity and any artifacts.

Human acceptance: **accepted on 2026-09-28**. The user explicitly stated:
“the results are acceptable”. This closes ACB034 for this report's frozen
14-video comparison and reported three-segment diagnostic. Individual playback
actions, listening actions and criterion scores were not separately supplied.
The [acceptance record](adaptive-continuation-acceptance.json) binds this decision
to the frozen manifest and media hashes. The recorded limitations remain unchanged.

### Memory observations

The 640×480 hard adaptive cache occupies 272,158,904 device bytes; the bridge version occupies 272,449,208 bytes, with 181,601,280 persistent bytes for its two saved BF16 tensors. The high-resolution image-conditioned hard case uses 1,568,785,592 cache bytes. All 50 weight blocks remain resident in these comparison runs.

Recipe 4 adds only 47,288 fixed reduction bytes and an 836-byte compact host map. Existing bridge modulation is a separate cost: the 50-step bridge prepares 644 time rows, 543 extra rows and 5,266,179,072 extra AdaLN bytes, plus a 6,552-byte bridge class plan. The exact-row audit snapshot is 3,148,800 bytes at the small comparison geometry. These schedule-dependent modulation costs must be included when planning bridge residency.


## Failures and corrections

- The eight-block automatic residency cap deliberately invalidates/replans a
  partial DiT on each request. It validates retained public-context recovery,
  but cannot prove live prepared-DiT reuse. A focused explicit-streaming helper
  mode now checks the canonical resume key, compatible placement, retained DiT
  identity and four actual cache-hit log entries across hard and bridge modes.
  Its separate run passes in `live01/result.json`. This changes test coverage only.

- Initial CUDA relinking encountered stale objects because source timestamps
  predated baseline compilation. Content-based synchronization and a clean
  rebuild fixed it; the subsequent full 204-output gate passed.
- The first local Metal test attempt could not access the GPU in the sandbox.
  The authorized rerun with device access passed.
- The presentation suite exposed a synthetic trimmed-cache fixture using
  ordinary reference recipe 3. It now uses continuation recipe 4 and separately
  checks rejection of that mismatch, ordinary delivery and keep-prefix delivery.
- Early history validation initially ran before custom warmup metadata was
  read. Host round trips caught it. The loader now reads the small current
  policy/device records before cache allocation; all round trips and corruption
  checks pass without changing file schemas.
- The first full CUDA suite stopped at the GPU API inventory because the new
  continuation probe was missing from the link-contract test. The inventory
  now includes it; the local link check and full CUDA retry pass. The passing
  `cuda-test21.log` is separate from the preserved failed `cuda-test20.log`.
