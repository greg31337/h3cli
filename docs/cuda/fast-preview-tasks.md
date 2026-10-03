# CUDA fast-preview video comparison

Status: **complete; 20/20 execution tasks complete**. All forty requested slots
are reconciled: **26 validated videos, 13 explicitly blocked slots and one user-skipped SOL slot**.
See the [results and wall-time report](fast-preview-results.md) and its
linked video gallery. h3cli implementation and input fixtures remain unchanged.
The earlier reference-image work remains in its
[task archive](../features/reference-image-size-tasks.md) and
[validation report](../features/reference-image-size-results.md).

The user chose to retain all requested features and mark unsupported reference
runs as blocked. The source audit also identified a high-resolution adaptive
cache budget limit. The original matrix had **27 runnable videos and 13 explicitly
blocked slots**. The user subsequently requested skipping slow SOL runs. Preserve
the three completed SOL videos and mark the interrupted P05-1344-ref slot as
user-skipped; achieved coverage is **26 videos, 13 blocked and one skipped**.
Blocked and skipped slots are not videos, passes or zero-second results.

## Fixed workload and constraints

- Use only the authorized RTX PRO 5000 / SM120 for CUDA generation. Local M4
  work is limited to preparation, media review and reporting. Keep connection
  details and machine-specific model paths outside tracked documents.
- **Do not change h3cli source, kernels, policies, memory limits or defaults.**
  Do not relax guards, change goldens/fixtures, or modify historical experiment
  runners/manifests. Reuse tools where compatible; any campaign orchestration
  and generated manifests belong under ignored `outputs/cuda-fast-preview/`.
  Build products remain under ignored `bin/`.
- Freeze the source revision and input-source identity at execution. Planning
  inspected revision `e12bb5cb80731556b32ed55794aee1f4346ceb67`. Use qualified
  CUDA 13.0.3/cuDNN 9.20. Record compiler, driver, device, binary, optional build
  features and effective runtime settings.
- Every successful video uses **124 frames at 24 FPS**, seed **42**, all **50
  blocks**, `--core-reuse 1`, and `--cuda-weight-mode auto`. Request exactly
  **640×480** or **1344×768**. Do not change resolution, frames or steps to
  rescue a slot. Record actual full/partial/streamed residency for each run.
- Use the same prompt below for every run. Text-only cases have no image,
  video, audio or frame-anchor input. Reference cases add exactly
  `--ref-image inputs/1.jpg --ref-image-size max`; they use Ref2VA, not an anchor.
  The 1365×1821 source must resolve to **2048×2720**, **21,760 raw patches**,
  at both output resolutions. Do not resize or replace it.
- Fixture SHA-256:
  `1e41edd8381e7b2757278750b699e8251ad2f2525177056fcd31d4e95856caf6`.
  Use metadata identities for base weights; never hash/scan their payloads.
- Full VAE means omit `--preview-vae`. Preview VAE means supply it and an
  explicit verified `--preview-vae-model` path to existing TAEH3 weights
  (local asset: `models/preview-vae/taeh3.safetensors`). Audio stays enabled
  and decoded normally in both modes.
- No LoRA/Turbo, continuation, upscaling, layer thinning, token reduction,
  sampler resume, imported conditioning/latents, live previews or unlisted
  approximation overrides. Use fresh CLI processes and `--profile` uniformly.
  SOL keeps its current defaults; do not add SOL tuning to this experiment.
- Scope `H3_TEST_MAX_EVALUATIONS=50`, if needed, only to these render
  subprocesses; requested schedules remain 6/12/20. Do not set that variable
  to 12 or 20, or change ordinary six-evaluation test limits. Enable existing
  lightweight experiment timing consistently and disable tensor capture.

Fixed prompt:

> The camera tracks slowly along a sunny beach as a woman in a white dress
> turns toward the camera, smiles, and brushes windblown hair away from her
> face. Gentle waves roll onto the sand behind her. Natural daylight, realistic
> motion. Soft surf and a light ocean breeze are audible.

## Ten variants

Columns map directly to `--steps`, `--preview-vae`, `--reuse`,
`--adaptive-cache`, `--adaptive-cache-warmup`, `--cuda-attention`,
`--subblock-warmup`, `--subblock-sparsity`, and `--cuda-denoise-quant`.
An em dash means **omit the flag**, not zero or a hidden default override.
Set all non-dash scalar values explicitly; preview VAE is a presence flag.

| ID | Steps | Preview VAE | Reuse | Adaptive cache | Adaptive warmup | CUDA attention | SubBlock warmup | SubBlock sparsity | Denoise quant |
| --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: | --- |
| P01 | 6 | off | 1 | off | — | default | — | — | off |
| P02 | 6 | on | 1 | off | — | default | — | — | off |
| P03 | 6 | on | 3 | off | — | sage3 | — | — | nvfp4 |
| P04 | 12 | on | 2 | off | — | sage2++ | — | — | fp8 |
| P05 | 20 | on | 1 | off | — | sol | — | — | nvfp4 |
| P06 | 6 | on | 1 | conservative | 4 | default | — | — | fp8 |
| P07 | 20 | on | 1 | aggressive | 16 | default | — | — | off |
| P08 | 6 | on | 1 | off | — | subblock | 2 | 0 | off |
| P09 | 12 | on | 1 | off | — | subblock | 10 | 0.75 | fp8 |
| P10 | 20 | on | 1 | conservative | 2 | subblock | 4 | 0.80 | off |

Reasons for these selections:

- **P01/P02:** matched six-step dense BF16 controls isolate final decoder
  selection. These are preview controls, not a high-step quality reference.
- **P03:** deliberate aggressive edge: six steps, reuse 3, Sage3 and NVFP4.
  Retain the expected low-step reuse warning and count actual fresh evaluations;
  low quality is a result, not grounds for a selective rerun.
- **P04:** twelve-step middle setting with reuse 2, Sage2++ and FP8.
- **P05:** twenty scheduler evaluations with SOL and NVFP4, without reuse.
- **P06:** conservative FP8 caching at the largest legal six-step warmup
  (`4 = steps − 2`). Only evaluation 4 can hit; evaluation 5 refreshes.
  Block 0 remains BF16 under the existing adaptive/quantized recipe.
- **P07:** maximum warmup 16 on a twenty-step aggressive BF16 schedule.
  Evaluations 16–18 may hit; evaluation 19 refreshes. Hits are not guaranteed.
- **P08:** earliest legal SubBlock warmup 2 and zero sparsity. This is the
  documented dense bypass, so sparse dispatches must remain zero.
- **P09:** SubBlock warmup 10 explicitly at the twelve-step `steps − 2`
  boundary, with FP8. Only evaluations 10–11 can become sparse; block 0 and
  protected queries stay dense.
- **P10:** combined conservative cache and SubBlock in BF16. Warmups 2 and 4
  exercise the required adaptive refresh at the attention transition, with
  the more aggressive previously exercised 0.80 sparsity.

Explicit warmups satisfy `2 ≤ warmup ≤ 16` and `warmup ≤ steps − 2`.
Aggressive cache plus quantization, adaptive+SubBlock+quantization, reuse >1
with either feature, and adaptive with Sage/SOL are excluded as invalid.
See the [warmup contract](approximate-warmup.md),
[adaptive/quantized recipe](adaptive-quant-experiment.md), and
[SubBlock/quantized recipe](subblock-quant-experiment.md).

## Requested slots and known blockers

`pass` means rendered and validated in the completed campaign. `R` and `M` are
known blockers in the unchanged code. Use stable IDs such as `P01-640-text`,
`P01-640-ref`, `P01-1344-text`, and `P01-1344-ref`.

| Variant | 640×480 text | 640×480 one reference | 1344×768 text | 1344×768 one reference |
| --- | --- | --- | --- | --- |
| P01 | pass | pass | pass | pass |
| P02 | pass | pass | pass | pass |
| P03 | pass | pass | pass | pass |
| P04 | pass | pass | pass | pass |
| P05 | pass | pass | pass | S |
| P06 | pass | R | M | R + M |
| P07 | pass | R | M | R + M |
| P08 | pass | R | pass | R |
| P09 | pass | R | pass | R |
| P10 | pass | R | M | R + M |

**S — user-skipped.** The remaining SOL run was stopped at the user’s request.
Preserve its attempt time and logs, with no successful-video wall time.

**R — reference layouts rejected.**
[Current policy](../../src/denoise/approximate.c) rejects any reference with adaptive
cache or SubBlock, even SubBlock at zero sparsity. Preserve these ten reference
slots as blocked. Do not silently disable a feature, replace a reference with
an anchor, or call a modified reference arm the same variant.

**M — adaptive cache exceeds its fixed budget.**
[The planner](../../src/denoise/adaptive_cache.c) reserves `rows × 5376 × 6 + 6168` bytes,
with a fixed 512 MiB ceiling and a maximum of 16,643 packed rows. The
[temporal/layout rules](../../src/host.c) map 124 frames to 37 latent time slices.
At 1344×768, video alone has `37 × 42 × 24 = 37,296` rows, requiring
**1,203,025,944 bytes (1,147.29 MiB)** before text/audio. Therefore the three
high-resolution text-only adaptive slots are also blocked; their reference
slots already have R and additionally exceed M. Quantization and longer warmup
do not shrink this BF16 cache. Do not raise the cap or change resolution/frames.
At 640×480 the video-only lower bound is 341.46 MiB; verify the complete
text/audio layout during preflight rather than assuming admission.

## Measurement and report contract

- Prepare or reuse metadata-verified FP8/NVFP4 packed weights for both FL2VA
  and Ref2VA before timing. Record preparation wall time separately; require
  zero newly packed matrices in timed runs. Verify the tiny VAE asset without
  changing weights. Do not scan base weights.
- Run serially, in P01–P10 order; within each row use 640-text, 640-ref,
  1344-text, 1344-ref. Record known blocked slots without launching doomed
  renders. One successful video per eligible slot, with no warmup videos or
  repeated timing averages. Preserve failed attempts and retry reasons. Do not
  evict system caches; disclose page-cache/order effects in single observations.
- **Total render wall seconds** use a monotonic clock from immediately before
  CLI launch through child exit and output-file closure. Include startup, model
  loading, conditioning/reference encoding, denoising, video/audio VAE and MP4
  encoding/writes. Do not substitute denoise time for total wall time. Exclude
  build/packing, transfers, later validation and report assembly; record those
  durations separately. Failures retain attempt time, with successful-video
  wall time null.
- Set a **7,200-second deadline per launched render** before the campaign.
  Capture exact argv/environment overrides, logs, exit code, timing and cleanup.
  Sample RSS and per-process VRAM once per second, retaining sampling gaps.
  Record actual residency and existing phase/dispatch telemetry. Do not edit
  source to add instrumentation.
- Fully count and decode each video's frames and audio: exact requested
  dimensions, 124 frames, 24 FPS, about 5.1667 seconds, stereo audio and no
  decoder errors. Confirm max-reference geometry in logs; retain media hashes
  and decoder identity. Visual review describes motion, reference identity,
  artifacts and audio separately from functional success. It does not imply
  user acceptance or numerical parity.
- Publish an HTML gallery at `outputs/cuda-fast-preview/<campaign>/index.html`
  containing **all forty slots**, grouped by resolution and conditioning. Each
  success gets an embedded video, download link, exact flags and total wall
  seconds. Blocked slots get reasons and no fake player/timing. Include an
  attempt ledger and JSON/CSV with statuses, timing, media paths, validation,
  memory, effective dispatches and identities.
- Publish a tracked summary at `docs/cuda/fast-preview-results.md` linking the
  gallery, data and reproducibility records. Compare within each resolution
  and conditioning group. P01/P02 isolate decoder selection; other rows change
  several factors and cannot establish isolated gains for one flag. Text-only
  and reference paths use different transformers, so their difference is not
  solely image-encoder overhead.
- Follow [CONTRIBUTING.md](../../CONTRIBUTING.md): reuse qualification for an
  unchanged qualified build. If build flags or campaign test tooling change,
  run the complete unchanged 204-output golden gate after that coherent change.
  Keep it and its video outside the preview matrix, retaining its fixtures,
  hashes and 12-minute deadline. Documentation-only planning requires document
  and link checks, not a new GPU run.

## Tasks

- [x] PV001 Freeze ten exact variants and forty unique slots, prompt/seed,
  fixture identity, geometry, frame/step counts, decoder assets, timeout and
  order. Capture source/binary/environment identities without hashing weights.
- [x] PV002 Verify PRO 5000 assets and build capabilities for Sage2++/Sage3,
  SOL, SubBlock, FP8/NVFP4 and preview VAE. Reuse a suitable build or compile
  unchanged sources with required optional features in isolation. Record R/M
  evidence and complete adaptive layout admission at 640×480. Preserve all
  memory limits and the no-code-change constraint.
- [x] PV003 Prepare/reuse both transformer families' FP8/NVFP4 caches and the
  verified preview decoder. Record setup costs and ensure exact metadata keys
  needed for the campaign are ready before timing.
- [x] PV004 Prepare ignored orchestration, ledger, timing/resource capture,
  variable-step dispatch checks and report assembly, reusing existing tools
  where compatible. Leave historical tools unchanged. Run any golden gate
  required by build/tool changes before the timed campaign.
- [x] PV005 Execute P01: one full-VAE video in each of its four eligible slots.
- [x] PV006 Execute P02: one preview-VAE video in each of its four eligible slots.
- [x] PV007 Execute P03: all four slots; retain the low-step reuse warning.
- [x] PV008 Execute P04: all four slots; verify reuse 2 and Sage2++/FP8 dispatch.
- [x] PV009 Execute P05 with SOL defaults and NVFP4: preserve the three completed
  videos; skip the interrupted 1344-reference slot at the user’s request.
- [x] PV010 Execute P06 at 640-text; document both reference slots as R and
  1344-text as M, retaining both reasons for 1344-ref. Check warmup/final refresh.
- [x] PV011 Execute P07 at 640-text; document the same three blocked slots as
  P06 and check its sixteen-evaluation warmup and final refresh.
- [x] PV012 Execute P08 at both text resolutions; document its two reference
  slots as R. Confirm zero sparse calls for the dense bypass.
- [x] PV013 Execute P09 at both text resolutions; document its two reference
  slots as R. Check dense warmup 0–9, eligible sparse steps 10–11 and FP8 calls.
- [x] PV014 Execute P10 at 640-text; document its three blocked slots as R/M.
  Check adaptive warmup 2, the SubBlock transition refresh at step 4 and BF16.
- [x] PV015 Validate every MP4's full decode, dimensions, frames/FPS, duration
  and audio. Check max-reference canvases, decoder choice, artifact identities,
  deadlines and reaped processes.
- [x] PV016 Audit actual scheduler transitions, fresh forwards, reuse/cache
  hits and refreshes, warmup boundaries, dense/sparse and quantized calls, and
  zero timed packing. Do not claim benefit from flags with no actual hits/sparse
  work; explain protected/dense fallbacks.
- [x] PV017 Reconcile all forty slots and attempts: expected 26 successes,
  13 known blocked slots and one user-skipped SOL slot, with any additional
  failures explicit. Never count
  blockers as successes, selectively rerun a success, or remove a requested
  cell. Verify h3cli implementation and inputs match the execution snapshot.
- [x] PV018 Create the complete video gallery, wall-time tables, JSON/CSV and
  tracked summary. Include separate preparation costs, blocked reasons, failed
  attempts and limits of single-run timing comparisons.
- [x] PV019 Review videos/audio for preview usefulness; summarize observed
  quality/speed tradeoffs by resolution and conditioning. Give a provisional
  shortlist only where successful outputs support it. Do not transfer older
  experiments' visual acceptance to these settings.
- [x] PV020 Check report links, video playback/download paths and record
  consistency; update task status and publish report locations. State achieved
  success/blocked counts without claiming forty videos exist when current
  compatibility/resource limits prevent them.
