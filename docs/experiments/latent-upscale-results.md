# Native latent upscale comparison

Status: **accepted; visual qualification passed; regular CUDA/Metal feature**.
The user accepted the fourteen-video comparison and requested promotion to
a regular feature. All diagnostics and artifact audits passed. The
[acceptance record](latent-upscale-acceptance.json) binds the decision to the
reviewed report, qualified source and exact video hashes. The [task list](latent-upscale-tasks.md),
[design](../features/design-latent-upscale.md), [frozen manifest](latent-upscale-manifest.json)
and [qualification record](latent-upscale-qualification.md) define the scope.

[Open synchronized playback and crops](../../outputs/latent-upscale/comparison-001/index.html).
The [JSON](../../outputs/latent-upscale/comparison-001/report.json),
[CSV](../../outputs/latent-upscale/comparison-001/report.csv),
[stage ledger](../../outputs/latent-upscale/comparison-001/ledger.json) and
[local audit](../../outputs/latent-upscale/comparison-001/local-audit.json)
retain the exact observations and artifact identities.

## Protocol and interpretation

One video per variant at each of two pairs: 672×384 → 1344×768 and
960×544 → 1920×1088. Every video contains 90 frames at 24 FPS, using the frozen
piano prompt and seed 42. Sources and direct baselines run 50 dense BF16 steps.
The comparison contains L0 (source), D0 (direct high resolution), P0 (pixel
Lanczos), I4 (bilinear latent plus four steps), and U0/U2/U4 (learned latent
transfer plus zero/two/four steps). I4/U2/U4 share one persisted target-noise
tensor and starting video sigma 0.25. U0/U2/U4 share one learned transfer.

The direct render can depict a different scene even with the same seed. Its
similarity scores are diagnostics, not exact target values. Source consistency
can penalize new detail; high detail energy can reward noise or ringing. This
single prompt and seed cannot establish general quality or speed, reference
conditioning quality, or parity with MiniMax's official Regenerate-2K system.
Visual acceptance comes from the user's explicit review of this comparison.
It is separate from the earlier caching/attention experiments.

## Cost accounting

Later-job cost charges each candidate for its full transfer, initialization,
refinement and delivery work. End-to-end cost adds low-resolution generation
with its preview decode removed. The pixel baseline includes the source decode
it needs. Shared-noise creation is charged to each refinement job. Verification
PCM decodes and report computation are excluded.

These are observed stages in separate native processes with normal OS caching;
filesystem caches were not forced cold. A future integrated workflow may have
different loading/I/O overhead. No successful render is repeated for timing.
Cost ratios do not establish equal visual quality.

Both pairs completed without failed campaign attempts or successful repeats.
End-to-end workflow costs, in seconds:

| Target | D0 direct | P0 pixel | I4 bilinear + 4 | U0 learned | U2 learned + 2 | U4 learned + 4 |
|---|---:|---:|---:|---:|---:|---:|
| 1344×768 | 767.097 | 170.400 | 255.320 | 186.598 | 232.579 | 260.807 |
| 1920×1088 | 2143.744 | 331.623 | 543.226 | 364.830 | 466.998 | 549.918 |

Including source generation, U0/U2/U4 cost ratios versus direct rendering are
4.11×/3.30×/2.94× at the smaller target and 5.88×/4.59×/3.90× at the larger target.
These ratios describe this measured workflow, not equal-quality replacements.
The source videos themselves took 169.181 and 328.798 seconds including delivery.
All latent-upscale variants preserve the source audio latent and decoded PCM
exactly. D0 generates its own audio and is not required to match the source.

## Diagnostics and usage guidance

| Target | Variant | Source SSIM | Temporal delta RMSE | Detail energy / pixel | Later-job seconds | Peak device GiB |
|---|---|---:|---:|---:|---:|---:|
| 1344×768 | P0 | 0.9940 | 0.0061 | 1.000 | 1.219 | 0.61 |
| 1344×768 | I4 | 0.5938 | 0.0387 | 2.778 | 96.487 | 43.80 |
| 1344×768 | U0 | 0.9574 | 0.0185 | 1.886 | 27.765 | 6.01 |
| 1344×768 | U2 | 0.9537 | 0.0196 | 2.131 | 73.746 | 43.75 |
| 1344×768 | U4 | 0.9520 | 0.0203 | 2.347 | 101.974 | 43.80 |
| 1920×1088 | P0 | 0.9942 | 0.0063 | 1.000 | 2.824 | 0.61 |
| 1920×1088 | I4 | 0.5434 | 0.0364 | 3.319 | 228.476 | 50.43 |
| 1920×1088 | U0 | 0.9514 | 0.0174 | 2.025 | 50.081 | 6.01 |
| 1920×1088 | U2 | 0.9450 | 0.0186 | 2.208 | 152.249 | 50.38 |
| 1920×1088 | U4 | 0.9431 | 0.0190 | 2.495 | 235.169 | 50.43 |

Source SSIM is measured after area downsampling to L0. Temporal delta RMSE
measures differences in consecutive-frame changes against L0, not absolute
flicker. Detail is mean squared native-resolution luminance Laplacian energy.
Higher detail energy is not automatically better: I4 scores highly while its
selected frame-76 crops show pronounced repeated edges and ripples around
hands, keys and piano strings at both resolutions.

Inspection of the selected U0/U2/U4 crops shows preserved source composition
and cleaner boundaries than I4. Refinement adds edge contrast/detail, but these
crops do not establish a perceptual winner between U2 and U4. D0 depicts a
different composition: its source SSIM is only 0.2432/0.2401, and the same center
crop can show a different part of the performer or piano. Its native detail
energy varies in opposite directions between the two sizes, illustrating why
it is not a pixel-aligned reference.

Start with **U0** when cost and source preservation
matter most; it already provides the learned transfer without loading the H3
transformer for a later job. Compare **U2** if additional refinement is wanted.
U4 costs more and increases measured detail, with slightly lower source
consistency and higher temporal discrepancy. There is no established visual
quality gain sufficient to recommend U4 over U2 from this one example. The
implemented default remains K=4/sigma=0.25; no settings were tuned after seeing
these outputs. I4 is a poor candidate in the inspected crops.

Every latent-upscale variant matches L0's audio latent, native pre-AAC PCM and
post-AAC decoded waveform bytes. P0 copies the AAC stream exactly. D0's audio
is independently generated. All outputs are readable 90-frame, 24-FPS videos
with the exact planned dimensions. These checks do not substitute for human
review of full motion, fine detail or audiovisual quality.

The direct-render peak device usage was 44.70/51.33 GiB. The table reports each
candidate's render/decode peak; transfer peaks are lower. U0's later job peaks
at about 6.01 GiB on this CUDA device because it does not load the transformer;
source generation is a separate, larger-memory job.

## Implementation evidence

The frozen source fingerprint is
`d7d8638cc6b99259ab304b3aae66375cf86bf26f55b45eac2ad24eeb5209ffb6`.
The final unchanged CUDA gate passes all 204 outputs, using CUDA 13.0.3/cuDNN
9.20 on RTX PRO 5000 72GB. Its inference/check phase took 100.021 seconds.
The qualified Metal device is an Apple M4 Max with 128 GiB unified memory.

| Scope | Evidence |
|---|---|
| Pinned representation, network and experiment contracts | `tests/upscale/contract.json`, `outputs/latent-upscale/m0-contract/`, frozen manifest |
| Owned source/state/geometry/refinement contracts | 2,930 shared sampler checks, ten adversarial source groups, 24 upscale CLI checks; final CUDA/Metal build logs |
| Native learned graph and resource cleanup | 210 frozen operator comparisons on each backend, cancellation/admission/interleaved-context checks |
| Source portability and conditioning | First/last/both keyframes, ordered mixed image/video/audio, max-sized image, moved bundles with deleted references; both `m6a` condition records |
| Exact refinement lifetime | 14 actual-model runs per backend: K0, full K4, stop/resume and cancel/recover at 0/1/3; every AV callback and final file match |
| Full target canvases and delivery | Both 90-frame canvases on CUDA/Metal; six-evaluation source→upscale→source isolation, one-evaluation direct probes, full decoder tiles/edges, exact PCM |
| Ordinary behavior | Unchanged 204-output golden gate; host, continuation, bridge, reference layout, posterior, memory, progress, sampler, preview/full-VAE suites |
| Experiment tooling | Four accounting/immutability tests, two diagnostic tests, 14-case dry run, persisted shared-noise smoke checks, previous-reader rejection |

Evidence paths are under ignored `outputs/latent-upscale/`; generated media,
models and binaries are not committed. The qualification record retains earlier
operator failures and fixture/setup failures together with their corrections.
No golden bytes, tolerances, geometry caps or ordinary evaluation deadlines were
relaxed. H3 base weights were identified by metadata only.

## Supported and unqualified scope

Production inference is native and requires no Python/Torch worker. Both
backends support exact spatial 2× transfer, unchanged time, full-context 3D
normalization/convolution, clean-audio preservation and exact same-backend
refinement resume. Refinement remains opt-in with K=0/2/3/4; K=0 consumes no
sampler RNG or transformer evaluations. Source bundles own their conditioning
and do not reopen original media. Model metadata and backend/runtime constraints
still apply; model relocation and cross-backend exact resume are not qualified.

The larger Metal target reached about 93 GB process footprint and 95 GB Metal
allocated memory. Passing geometric bounds does not guarantee memory admission
on smaller devices. Existing reference-encoder capacity limits remain in force.

FP8/NVFP4, adaptive/SubBlock/fixed reuse, token reduction, LoRA/Turbo,
continuation/bridge sources and refinement, temporal interpolation, chunked
upscaling and tiled DiT remain outside this qualification. The comparison's
text-only prompt does not establish perceptual quality for references.

## Completion and identities

All 43 tasks are audited in the [completed list](latent-upscale-tasks.md). The campaign
contains 36 successful recorded stages, including shared preparation and PCM
verification, and exactly 14 successful videos. No failed campaign attempt,
extra seed/prompt sweep or successful timing repeat was needed. Earlier
implementation/fixture failures remain in the qualification record.

The comparison adapter binary SHA-256 is
`0bf17c42869e61e9987d49b84cf646c5d5c97636b55d7f29c485b8ab3b395bac`;
manifest SHA-256 is
`ce21046360cd91d8ff1d18d46f16b27105acc9fc8b72264f9ab85ef9a10b1d5e`.
The original reviewed report SHA-256 is
`d11ba7285571e05d0fd39d4930d2675692acdccff95e138de69f482d6a74f04d`.
After adding the acceptance status, the published report SHA-256 is
`ccd9e9c7249f497c8970240d2214401d44af62042fe8624dd8754d2f97e925ae`.
The original report and audits remain under `review-before-acceptance/` in the
local comparison directory. The frozen protocol, execution ledger, videos,
latents and numerical measurements are unchanged.
The JSON records every video hash, environment, model metadata and metrics-tool
identity. `report-audit.json` verifies 14 videos and 42 crops; `local-audit.json`
rechecks every published stage artifact, full media readability and local links
after transfer, and matches the final source against the 204-output CUDA gate.
No code/tool changes followed that gate; later edits are documentation only.

Human visual acceptance is **recorded** for these fourteen artifacts. The user
requested promotion to a regular feature. The acceptance update rechecks all
36 stage records and their artifact hashes, the 42 crops and report links, and
the unchanged source fingerprint against the final CUDA gate. Current CLI/API
controls already expose upscaling without a separate opt-in gate; defaults remain
K=4/sigma=0.25 and the documented compatibility/resource limits still apply.
Fresh comparison runs continue to begin with visual review pending; approval
of this artifact set is not an automatic approval of future outputs.
