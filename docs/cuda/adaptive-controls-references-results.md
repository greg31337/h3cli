# Adaptive controls and reference qualification

Status: implementation and automated qualification complete; the user accepted
the visual/listening results on 2026-09-28. This record accompanies the
[design](design-adaptive-cache-controls-references.md) and
[task list](../adaptive-controls-references-todo.md).

The implementation adds explicit threshold and consecutive-hit controls, BF16
image/video/audio and mixed-reference adaptive caching, and composition with
SubBlock. Reference recipe 3 selects the maximum global/video/audio score;
text recipes 1/2 and their default decisions are unchanged. Sampler section 40
is now v3 and presentation is now schema 9. Older versions are rejected.

Evidence is under `outputs/adaptive-controls-references/run01/` on the local M4
and the authorized PRO 5000 checkout. Videos, contact sheets, logs and JSON
summaries are mirrored locally; large latent/checkpoint files remain on the
server. The initial commit was
`46239d010f9a345aaf37f9322bc1eb2f17541ed8`, with a clean worktree. CUDA uses the
existing CUDA 13.0 Update 3 / cuDNN 9.20 stack. Model weights were not modified.

The final CUDA candidate source identity is
`48c8f921c908000ff034f91bf62c1000bcc603d9f1f2d0ccc386cac7b742df7b`.
All 713 source/build/test files match the local checkout. The comparison excludes
the M4's ignored, generated `src/metal/native_attention.inc`; its Metal generator and
inputs match. The [inventory comparison](../../outputs/adaptive-controls-references/run01/source-identity-check.json)
records that distinction.

## Recorded numerical contract

The immutable manifest SHA-256 is
`fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3`.
All 12 input fixtures and all 204 expected hashes remain unchanged.

| Gate | Result | Source SHA-256 | Seconds after build |
| --- | --- | --- | ---: |
| [baseline-gate](../../outputs/adaptive-controls-references/run01/baseline-gate/result.json) | 204/204 | `f4dad4f5019029b340038ffb0bd13550cbb58c4ad58dc575d9e0982979722a5f` | 98.55 |
| [candidate-gate01](../../outputs/adaptive-controls-references/run01/candidate-gate01/result.json) | 204/204 | `9a27438de8b9ca40322fbd0d176f276810d74ec0c7625ac993737070b13f42ab` | 97.30 |
| [candidate-gate02](../../outputs/adaptive-controls-references/run01/candidate-gate02/result.json) | 204/204 | `253d7ab0634438ea362d0ff59f783f1a814805ddbd544b1b1ddcf2ed3b2f7f5d` | 97.52 |
| [candidate-gate03](../../outputs/adaptive-controls-references/run01/candidate-gate03/result.json) | 204/204 | `f89d69d2bbb7b1be07d6729260f1f6ec26ce508ffdb43fbc832eebe5144f9556` | 133.69 |
| [candidate-gate04](../../outputs/adaptive-controls-references/run01/candidate-gate04/result.json) | 204/204 | `57bfe0c997409dfdb1bb6d3e99c0f9b62619efbbede1878b454d3774225c9104` | 109.66 |
| [candidate-gate05](../../outputs/adaptive-controls-references/run01/candidate-gate05/result.json) | 204/204 | `48c8f921c908000ff034f91bf62c1000bcc603d9f1f2d0ccc386cac7b742df7b` | 141.18 |
| [final-gate](../../outputs/adaptive-controls-references/run01/final-gate/result.json) | 204/204 | `48c8f921c908000ff034f91bf62c1000bcc603d9f1f2d0ccc386cac7b742df7b` | 140.32 |

`baseline-latents/` and `candidate-latents01/` are byte identical through six
evaluations for conservative and aggressive text-only presets. Conservative
hits at step 4 and refreshes for its streak ceiling at step 5; aggressive hits
at both steps. The three component scores are unchanged.

## Functional qualification

The clean local Metal executable/library build and complete `make test` pass.
Reader and parser tests pass
under AddressSanitizer and UndefinedBehaviorSanitizer, including rechecksummed
container corruption, explicit zero, numerical override matching, and reference
layout reconstruction. CUDA operator tests include an independent BF16/FP32
probe oracle and large unchanged reference prefixes with distinct target ranges.
See the [Metal build](../../outputs/adaptive-controls-references/run01/metal-final-build.log),
[full suite](../../outputs/adaptive-controls-references/run01/metal-final-test.log),
[sanitizers](../../outputs/adaptive-controls-references/run01/metal-final-sanitize.log)
and [60 bounded corruption cases](../../outputs/adaptive-controls-references/run01/bounded-corruption.json).
The complete [CUDA `make test` rerun](../../outputs/adaptive-controls-references/run01/cuda-test04.log)
also passes. Its one filesystem-dependent CoW skip on the default scratch volume
is covered by the mandatory no-skip run on `/models` in the current-feature suite.
All 19 [current CUDA feature/lifecycle checks](../../outputs/adaptive-controls-references/run01/current-cuda03/result.json)
pass, including ordinary media/anchors, four continuation segments, exact resume
with reuse/core reuse/token reduction, attention/precision context changes,
conditioning invalidation, residency/cancellation recovery, GPU VAEs, memory and
decoder failure recovery. These supplement the unchanged numerical gate.

The initial full local test attempt was denied GPU access by the filesystem
sandbox; the GPU-enabled clean Metal build and test run proceeded. A retained
media test still expected the removed five-frame alignment behavior; it now
checks that bounded FFmpeg decoding preserves all eight input frames. This
changes no rendering arithmetic or golden data.

`integration01/` records bounded real-model requests, actual hits, fixed encoded
condition rows, exact AV/history replay and source-media-free resume. All eight
core cases below pass, including original-schedule warmup, hit-streak, SubBlock
phase and final-refresh boundaries. The comparison is byte-for-byte across
generated video/audio and saved anchor/delta/history. Original reference media,
including external soundtracks, are hidden during resume. Threshold-zero output
matches the uncached control exactly. Cancellation/retry also matches a fresh run.
New latent diagnostics execute at most six evaluations per subprocess. Existing
lifecycle programs may issue several bounded requests in one process; each
request retains the ordinary six-evaluation ceiling. Only V01–V04 explicitly
raise that ceiling to 50.

| Bounded layout | Attention | Hits / 6 | Cache MiB | Exact resume |
| --- | --- | ---: | ---: | --- |
| One max-size image | dense | 2 | 351.37 | pass |
| One max-size image | SubBlock | 2 | 351.37 | pass |
| Video with embedded audio | dense | 2 | 208.29 | pass |
| Video with embedded audio | SubBlock | 2 | 208.29 | pass |
| Image + separate stereo audio | dense | 2 | 356.44 | pass |
| Image + separate stereo audio | SubBlock | 2 | 356.44 | pass |
| Ordered image + silent video + replacement video + audio | dense | 2 | 592.05 | pass |
| Same mixed set | SubBlock | 2 | 592.05 | pass |

These diagnostic jobs use 256×256 / 22 frames, threshold 1, maximum two hits and
warmup two. Combined jobs use SubBlock 0.75 with warmup three. This deliberately
exercises hits separately from the frozen quality comparison; its thresholds
are not a quality recommendation. The integration runner uses the second
candidate build. The final build adds stricter validation and transfer diagnostics;
the supplementary checks validate its reads and exact resident/streamed output.

The first integration runner completed 34 jobs and 190 assertions, then stopped
because its streaming argument was misspelled `streamed` instead of `stream`.
That attempt performed no denoiser evaluations. The log and failure ledger are
retained; final-build supplemental jobs replace the affected streaming and
remaining capacity/corruption tail. The original incomplete result is not
reported as an overall passing run. The final-build
[supplement](../../outputs/adaptive-controls-references/run01/supplement03/result.json)
passes all 66 commands and 71 explicit assertions, including seven actual mixed
checkpoint forgeries, 60 bounded reference corruptions, 31 corruptions for each
text quantization mode, and loading all 34 initial current-schema checkpoints.

The final-build soundtrack checks compare saved visual/audio conditioning for
the same video track with and without its original audio stream. Silent mode
produces byte-identical visual conditioning and no audio prefix. Replacement
mode produces byte-identical visual and audio conditioning in both cases.
Missing video and a file with no video stream are rejected with zero evaluations.
Additional real encodes cover nine images, two high-size images, video plus
separate audio, and multiple standalone audio clips accompanying an image.

Invalid-audio failure/recovery produces the same next-job latents as a fresh
context. Forced streaming matches resident output exactly for image-only and
mixed adaptive+SubBlock jobs. Each hit reads one layer (block 0), dispatches
eight GEMMs including fresh heads, and dispatches no suffix attention/router.
Measured hit H2D traffic is 770,884,864 bytes, compared with 38,535,349,504 bytes
on full streamed refreshes in these bounded jobs. This includes ordinary latent
transfers. CLI resume with all adaptive controls omitted restores threshold 1
and maximum two hits, with exact output while the original media are hidden.

Explicit conservative/aggressive text defaults match the pre-change captures
through six original-schedule evaluations. Text-only FP8 and NVFP4 custom
controls exercise real hits, exact pause/resume and checkpoint corruption checks.
Each refresh dispatches 196 quantized suffix projections; a hit dispatches zero,
with block 0 and cache data remaining BF16. No quantized comparison videos are
generated, and reference quantization remains unsupported.

The final [retained-context checks](../../outputs/adaptive-controls-references/run01/context-policy-final/result.json)
pass for dense and combined SubBlock, using six-step prepared schedules and zero
denoising evaluations. They explicitly retain a live DiT before changing the
threshold/hit ceiling, then test failure/recovery. An audit found that the earlier
version left caching disabled at this point; the test was strengthened, rebuilt
on both platforms, and followed by the complete `candidate-gate05` regression.
The production CLI binary was unchanged by this test correction.

The first final CUDA `make test` attempt exposed a CLI test that inherited the
20-step default while asserting valid custom controls. It reached the enforced
six-evaluation ceiling before its intended model check. The test now explicitly
defaults to six steps and asserts that no budget error masks control validation.
The corrected nine-case CLI suite passes on both platforms; the original failure
log is retained. Production parsing and the test ceiling were unchanged.

## Memory and streaming

The measured large case uses 1344×768 / 124 frames, one max-size image and three
evaluations of the original 50-step schedule. Its 48,610 packed rows require
1,567,970,328 cache bytes, including 1,045,309,440 persistent bytes. It keeps all
50 BF16 layers resident, executes a real hit, and completes under the default
4096 MiB ceiling.

| Diagnostic | Host peak RSS GiB | Device tensor peak GiB | Pinned host peak GiB | Compact checkpoint bytes |
| --- | ---: | ---: | ---: | ---: |
| Streamed image, 256×256 / 22 | 38.08 | 4.26 | 35.92 | 304,352,167 |
| Streamed mixed + SubBlock, 256×256 / 22 | 38.55 | 6.66 | 35.92 | 467,893,648 |
| Resident image, 1344×768 / 124 | 3.93 | 49.62 | 0.03 | 1,133,322,131 |

Host RSS is measured with `/usr/bin/time -v`; tensor and pinned peaks are engine
allocation counters, not total device usage. Compact diagnostic checkpoints omit
rebuildable prepared tensors and retain required conditioning/history. The large
case reports a 1,133,320,163-byte writer copy in addition to the live state;
streamed execution trades device residency for substantial pinned host storage.

The capacity-only 1344×768 / 362-frame mixed-layout fixture has 137,472 rows and
requires 4,434,303,000 cache bytes (minimum whole-MiB ceiling 4229). It admits an
8192 MiB ceiling, rejects one byte below its exact requirement, and exercises
overflow checks without rendering another long video. Actual media/vision token
counts still determine admission for a real request.

## Frozen comparison

All **18/18 videos pass** mechanical validation, one successful render per frozen
row, with no retries or retuning. The [manifest](../../outputs/adaptive-controls-references/run01/videos01/manifest.json)
contains exact commands, prompts, ordered input/source SHA-256 values and seed 42.
The two-second video/audio inputs were derived with FFmpeg from existing qualified
cleanup outputs. Silent mode deliberately receives a clip containing audio.
Only V01–V04 lift the ordinary six-evaluation ceiling to 50.

Open the [synchronized video report](../../outputs/adaptive-controls-references/run01/videos01/review.html)
for playback, contact sheets, commands and logs. The
[raw results](../../outputs/adaptive-controls-references/run01/videos01/result.json),
[compact metrics](../../outputs/adaptive-controls-references/run01/videos01/comparison-summary.json)
and [attempt ledger](../../outputs/adaptive-controls-references/run01/videos01/failure-ledger.json)
retain the complete evidence. The production CUDA CLI SHA-256 is
`948b42aab7814387783b18206053805c2131279d702f019d3540d2c3fc291579`; the frozen manifest SHA-256 is
`21ddbc0316b8f155d1997bd5baa612ce4fce3e1b7b403526aaba37ba5d18c1b0`.

| Rows | References and output | Policies, in order |
| --- | --- | --- |
| V01–V04 | One `max` image; 640×480, 90 frames, 50 steps | Dense; conservative defaults; aggressive defaults; conservative T=0.06/N=2/warmup=4 |
| V05–V07 | One `max` image; 1344×768, 124 frames, 6 steps | Dense; custom; custom + SubBlock warmup 2 |
| V08–V09 | Two `high` images; 640×480, 90 frames, 6 steps | Dense; custom + SubBlock warmup 3 |
| V10–V12 | Video with embedded audio; 640×480, 90 frames, 6 steps | Dense; custom; custom + SubBlock warmup 3 |
| V13–V15 | One `max` image + separate audio; 640×480, 90 frames, 6 steps | Dense; custom; custom + SubBlock warmup 3 |
| V16–V18 | Ordered `high` image + silent video + replacement-audio video + separate audio; 640×480, 90 frames, 6 steps | Dense; custom; custom + SubBlock warmup 3 |

Here **custom** means conservative with threshold 0.12, maximum two hits and
adaptive warmup two. Every combined row uses SubBlock sparsity 0.75. All rows
use BF16, full VAEs, all 50 blocks, reuse/core-reuse 1 and automatic placement.
All 50 layers remained resident in every comparison video; forced streaming is
qualified separately above.

| Video | Total s | Denoise s | Hits | GPU peak GiB | Video SSIM / audio relative L2 |
| --- | ---: | ---: | ---: | ---: | --- |
| [V01](../../outputs/adaptive-controls-references/run01/videos01/V01.mp4) | 505.54 | 426.36 | 0 | 41.74 | control |
| [V02](../../outputs/adaptive-controls-references/run01/videos01/V02.mp4) | 390.70 | 310.30 | 14 | 42.32 | 0.9430 / 0.1075 |
| [V03](../../outputs/adaptive-controls-references/run01/videos01/V03.mp4) | 289.04 | 209.14 | 26 | 42.32 | 0.9072 / 0.5135 |
| [V04](../../outputs/adaptive-controls-references/run01/videos01/V04.mp4) | 321.34 | 242.86 | 22 | 42.32 | 0.9022 / 0.2277 |
| [V05](../../outputs/adaptive-controls-references/run01/videos01/V05.mp4) | 273.14 | 199.32 | 0 | 48.03 | control |
| [V06](../../outputs/adaptive-controls-references/run01/videos01/V06.mp4) | 295.46 | 199.96 | 0 | 49.49 | 1.0000 / 0.0000 |
| [V07](../../outputs/adaptive-controls-references/run01/videos01/V07.mp4) | 308.01 | 211.44 | 0 | 49.99 | 0.9239 / 0.2045 |
| [V08](../../outputs/adaptive-controls-references/run01/videos01/V08.mp4) | 108.06 | 55.45 | 0 | 41.23 | control |
| [V09](../../outputs/adaptive-controls-references/run01/videos01/V09.mp4) | 121.60 | 60.04 | 0 | 42.36 | 0.8916 / 0.0499 |
| [V10](../../outputs/adaptive-controls-references/run01/videos01/V10.mp4) | 111.92 | 50.57 | 0 | 40.99 | control |
| [V11](../../outputs/adaptive-controls-references/run01/videos01/V11.mp4) | 120.90 | 50.72 | 0 | 41.57 | 1.0000 / 0.0000 |
| [V12](../../outputs/adaptive-controls-references/run01/videos01/V12.mp4) | 124.93 | 54.38 | 0 | 42.07 | 0.9701 / 0.0763 |
| [V13](../../outputs/adaptive-controls-references/run01/videos01/V13.mp4) | 102.79 | 51.06 | 0 | 40.97 | control |
| [V14](../../outputs/adaptive-controls-references/run01/videos01/V14.mp4) | 111.57 | 51.10 | 0 | 41.56 | 1.0000 / 0.0000 |
| [V15](../../outputs/adaptive-controls-references/run01/videos01/V15.mp4) | 115.43 | 54.94 | 0 | 42.06 | 0.9258 / 0.1238 |
| [V16](../../outputs/adaptive-controls-references/run01/videos01/V16.mp4) | 215.02 | 127.07 | 0 | 44.54 | control |
| [V17](../../outputs/adaptive-controls-references/run01/videos01/V17.mp4) | 232.17 | 127.14 | 0 | 45.63 | 1.0000 / 0.0000 |
| [V18](../../outputs/adaptive-controls-references/run01/videos01/V18.mp4) | 244.71 | 139.30 | 0 | 46.13 | 0.9732 / 0.0462 |

Total wall time includes conditioning, weight loads, denoising, VAE decoding,
muxing and the requested AV/sampler saves. It excludes subsequent media checks.
These are single measurements, not repeated benchmarks. GPU peaks are sampled
with `nvidia-smi` every 250 ms; engine tensor/pinned counters and checkpoint
sizes/times are retained in the compact metrics and full logs. The sum of the
18 CLI wall times is 3992.33 seconds.

At 50 steps, conservative, aggressive and custom reduce total time by approximately
23%, 43% and 36%, with 14, 26 and 22 hits respectively. Every hit dispatches one
dense block and eight GEMMs including fresh heads, with no suffix attention,
router calls or weight reads. Final evaluations refresh normally.

None of the frozen six-step adaptive variants produce a hit at threshold 0.12. Their
adaptive-only MP4s (V06/V11/V14/V17) are byte-identical to the corresponding
dense controls. These cases show overhead rather than a cache speedup; the
separate bounded threshold-1 diagnostics prove actual hits and exact replay for
all reference layouts at small geometry. The large-canvas diagnostic independently
confirms a real hit. The fixed matrix was not
retuned to manufacture hits.

Checkpoint serialization explains most of the large no-hit adaptive overhead:
V05 saves 263,666,220 bytes in 5.156 seconds, while V06 saves 1,309,556,668 bytes
in 26.117 seconds. Their denoising times differ by only 0.64 seconds. Combined
SubBlock also costs more denoising time than dense in these six-step fixtures.
These timings do not establish a general speedup for short reference renders.

The reference score guard has observable effects. V02 step 4 has global score
0.01862, but generated-video score 0.05768, so conservative correctly refreshes.
V11 step 4 has global score 0.08706, below its 0.12 threshold, but target-video
score 0.21246 requires a refresh. Large fixed reference spans therefore cannot
hide target changes behind a small global average.

## Media observations and acceptance boundary

All outputs fully decode, have the requested dimensions/frame counts at 24 fps,
and contain finite, nonempty stereo audio. The maximum audio-duration discrepancy
from the requested video duration is 0.008333 seconds; the maximum
reported audio/video start-time difference is 0.000000 seconds. These
container checks do not assess semantic lip sync. SSIM and waveform relative L2
are similarity diagnostics, not perceptual quality scores or acceptance limits.

The [inspection notes](../../outputs/adaptive-controls-references/run01/videos01/visual-observations.json)
record six sampled frames per distinct video and byte-identity checks for the
unchanged outputs. The beach cases retain the white-dress subject, walking/turning
poses and setting. Approximate variants differ in hair, arm and shoreline detail.
The two-image dense control already blends wardrobe attributes: both generated
women wear white although the second reference wears blue. That behavior persists
with SubBlock. Faces and hands in these six-step samples can be soft or indistinct.

Video-reference cases retain the small moving subject and sunlit garden. In the
mixed dense control, the video dominates; the separate white-dress subject is not
visibly reproduced. The combined result should be assessed against that same
control, without assuming every input's appearance will be preserved. Small,
backlit subjects limit detailed facial assessment.

Soundtrack selection, silence and replacement are verified through exact saved
conditioning and source-exclusion tests above. Output audio timing, finite samples
and waveform differences are measured. The assistant's inspection used contact
sheets, without a listening judgment or full-motion assessment. Contact sheets
cannot establish smooth motion, semantic soundtrack influence or the absence
of brief artifacts between sampled frames.

**The user accepted all 18 comparison results on 2026-09-28:** “the results are
acceptable”. This completes visual/listening acceptance for V01–V18, separately
from the automated checks. The [acceptance record](../../outputs/adaptive-controls-references/run01/videos01/acceptance.json)
preserves the approval and campaign identities.

## Final verification

The complete [final recorded gate](../../outputs/adaptive-controls-references/run01/final-gate/result.json)
passes all 204 outputs after the entire video campaign. The 12 fixtures and
golden manifest are unchanged, with no new SGLang numerical oracle or approximate
cases added to that contract. The final local/server source inventories match;
[local build identities](../../outputs/adaptive-controls-references/run01/local-build-identities.json)
identify the M4 executable/library separately from the CUDA binary above.
Current-only state requirements remain sampler section 40 v3 and presentation 9;
old adaptive sections and presentation 8 are rejected.

The verified source, documentation, CUDA executable and static library are also
installed in the server’s normal `~/h3cli` checkout. The
[installation record](../../outputs/adaptive-controls-references/run01/installed-result.json)
records exact hashes and preserved site files; replaced files are backed up under
`run01/installed-backup/`. The previous server checklist was byte-identical to
`docs/legacy-cleanup-todo.md` and remains archived. Installed `--help` exposes both
new controls.
