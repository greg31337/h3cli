# Continuation acceptance record

**PASS — all 50 task items are complete within the initial scope described below.**
The final `--require-complete` gate passes 25 matrix cases plus the paired debug
render, with a separate successful CLI resume: **27 acceptance renders** total.
Every render uses 20 Euler transitions and all 50 DiT blocks. State, per-step
prefix, trim, duration, timestamp, seam, reference, reuse and regression checks
pass. Completed task IDs are in [todo.md](todo.md); usage is in
[continuation.md](continuation.md).

Local artifacts: [review gallery](../outputs/continuation-validation/acceptance/review/index.html),
[color comparison](../outputs/continuation-validation/acceptance/review/color-drift.png),
[audio seam](../outputs/continuation-validation/acceptance/review/audio-seam.png), and
[full measurements](../outputs/continuation-validation/acceptance/metrics.json).

## Environment and reproducibility

- Baseline: `948c748e1ce4c619c85f716224b436ac0a139bee`, the initial clean checked-in
  C implementation. No Git remote is configured, so external upstream provenance
  is not asserted. The preexisting binary had unrelated continuation options;
  baseline binaries were rebuilt from source with `make -B`.
- Hardware: Apple M4 Max, 128 GiB unified memory; macOS Metal.
- Checkpoints: local `models/MiniMax-H3/FL2VA` and `models/MiniMax-H3/Ref2VA`.
- Acceptance renders: 256×256, 20 Euler transitions, all 50 transformer blocks,
  core reuse 1 and token reduction off. Baselines use 56 raw frames; continuation
  chains use 90 raw frames and 39 protected frames.
- Inputs: `inputs/face1.jpg`, `body1.jpg`, `face2.jpg`, `body2.jpg`. Their hashes,
  baseline commands and output hashes are in [continuation-baseline.json](continuation-baseline.json).
- The three-second video fixture uses `body1.jpg` and a 220 Hz tone. Replacement
  audio uses 440 Hz. Three seconds provides a legal 56-frame reference chunk.
  Standalone-audio cases retain the visual reference required by the engine.

Run `tests/continuation_regression.py`, `tests/run_continuation.py`,
`tests/continuation_cli.py --positive`, and `tests/continuation_metrics.py` as
shown in the guide. The matrix is resumable and records commands, output hashes, timings and
available binary hashes. Raw RGB, planar F32 PCM, MP4s, states and logs are retained
under `outputs/continuation-validation/`. These large artifacts are ignored by Git.

## Completed automated checks

- Original host suite: 1,768 checks; new continuation host suite: 103,190 checks.
- AddressSanitizer and UndefinedBehaviorSanitizer: continuation suite passes.
- Python IEEE F32 and PyTorch: all 1,270 native target rows agree with the
  independently evaluated native binary mask equation, including sigma endpoints.
- Existing AudioVAE Metal primitives and FFmpeg AV mux test pass. The original
  suite explicitly skips missing MLX, legacy tokenizer-path and real-weight
  comparison fixtures; these skips do not substitute for the full-checkpoint
  acceptance matrix.
- State tests cover special F32 bit patterns, complete header/payload checksums,
  corrupted/truncated/extended files, geometry/length/overflow rejection,
  incompatible signatures, allocation/clone ownership, exact source-tail copies,
  stereo packing, source immutability and complete noise ordering.
- Contexts 39/90/141/192 agree exactly at 24 fps and 40 Hz. Protected audio stays
  bit-identical; protected video stays bit-identical after its one-time 0.999
  augmentation. Real-model audits check initialization and every Euler transition.
- Ref2VA limits remain enforced, continuation consumes no reference slot, and
  reference/text row maps are unchanged. Cache-key tests distinguish geometry
  while ignoring continuation contents and seed.

## Ordinary-generation regression investigation

Cold T2VA output is byte-identical to the baseline. Cold multimodal outputs vary
in the original implementation itself: repeated unchanged FL2VA binaries produce
different Qwen text embeddings and MP4 hashes. Diagnostic comparison found
identical encoded video conditioning but different multimodal text embeddings,
before any continuation or DiT code runs.

The controlled regression harness therefore captures the original text/reference
conditioning and replays those exact arrays through an isolated current build.
T2VA, FL2VA, image Ref2VA, video Ref2VA and audio Ref2VA all produce **bit-identical final
video latents, audio latents and MP4s**. This investigates the cold differences
without weakening the final-latent check. Replay instrumentation exists only in
the isolated test builds. Hashes are recorded in the tracked baseline JSON and
`outputs/continuation-validation/frozen-regression/results.json`.

The CLI passes 15 rejection cases and a real fresh-process resume. With
`H3_GPU_SAMPLER=1`, it explicitly selects CPU Euler, preserves the exact protected
latents, writes a complete state, and produces a 51-frame/2.125-second AV output.
Whole-denoiser reuse 2 and 3 pass all 20 transitions, using 11 and 8 DiT
evaluations respectively; both options are enabled after those dedicated tests.

## End-to-end results

The two-segment T2VA test and five-segment image Ref2VA chain pass every latent
invariant and save/load check. Each resumed 90-frame target delivers exactly
51 frames and 68,000 PCM samples per channel; the saved state retains all 90
frames. Each call starts a fresh process and consumes its predecessor's `.h3av`.

The native five-segment contact sheet shows the same hiker, backpack, skin tone
and green setting with continued turns and waves. The face/body fixtures depict
different people; the chain prompt favors the face image's hiking appearance.
These close views cover shoulder, head and arm motion, not full-body gait.

All mixed reference cases pass: image + silent video, image + standalone audio,
video with embedded audio, and video with replacement audio. Each uses a separate
continuation state, preserves all protected values through 20 transitions,
round-trips the full state and delivers 51 frames/68,000 samples per channel.

| Acceptance group | Renders |
|---|---:|
| Ordinary T2VA/FL2VA/image/video/audio regressions | 5 |
| T2VA chain | 2 |
| Native image Ref2VA chain | 5 |
| Recursive comparison, reusing the first native segment | 4 |
| Cached trim/debug pair | 2 |
| Changed references and changed prompt | 2 |
| Mixed reference types | 4 |
| Whole-denoiser reuse 2 and 3 | 2 |
| Separate CLI process resume with GPU-request fallback | 1 |

The reuse cases first passed in an isolated build with only the reuse restriction
removed. The production `src/engine.c` was verified byte-for-byte equal to that tested
source when the restriction was removed; the gate and binary/output hashes are
saved in `reuse-staging/`. Their successful records are included in the final
matrix manifest. Missing legacy parity fixtures remain explicit skips, and
GPU-state/core-reuse/token-reduction parity is outside this initial release.

## Cached debug output and AV seam

The paired `trim-audit`/`debug-prefix` runs reuse the same conditioning and
prepared DiT inside one context. Both perform all 20 transitions. Their complete
final AV states are bit-identical. Normal RGB is exactly debug RGB after frame
39, and normal planar PCM is exactly debug PCM after sample 52,000 in each
channel. Output PTS starts at zero for both audio and video.

The copied history aligns with source frames 51–89 and samples 68,000–119,999.
Measured video prefix RMSE is **1.179/255**. The first five context frames have
RMSE 2.681; the remaining context has RMSE 0.734. Interior frames are around
0.25–0.29, with larger differences at the VAE context edges. Video's one-time
0.999 augmentation also introduces a small, intentional difference.

Audio prefix RMSE is **0.000794 F32**, against source RMS 0.007405. Excluding
8,000 samples at each edge gives interior RMSE **0.0000306**. The delivered
boundary's single-sample changes are 0.000178 and 0.000161 for left/right.
These are quantitative waveform checks; no subjective listening claim is made.
The full acceptance gate requires video prefix RMSE below 5/255, audio prefix
RMSE below 25% of source RMS, and interior audio RMSE below 2% of source RMS.
It also requires exact RGB/PCM trimming, saved-state parity, frame/sample counts
and zero-based AV timestamps. Error traces and a boundary waveform plot accompany
the frame comparison in the review gallery.

## Changed references and prompt

`changed-reference` supplies `face2.jpg` and `body2.jpg`; `changed-prompt` keeps
the original images but requests raised arms and a left turn. Both start from
the same source state and seed as the control. All protected F32 values remain
identical, while the generated video/audio suffixes differ. The prompt change
visibly raises both arms and turns left. Image changes alter the new prediction
and details, while this short continuation still retains the hiker's identity:
reference replacement does not guarantee an immediate identity/outfit switch
against strong inherited history. The comparison contact sheet records this
limitation rather than treating a changed output hash as proof of a new identity.

## Five-segment color and visual comparison

Both methods share `native-1`, seeds 42–46, the same prompt and the same clean
face/body references. Native continuation uses predecessor states; recursive
conditioning supplies each predecessor MP4, including its audio, as another
Ref2VA reference. Statistics use raw RGB, before video compression.

| Measurement | Native | Recursive |
|---|---:|---:|
| Mean RGB drift, segments 2–5 versus segment 1 | 14.954 | 18.831 |
| Segment-5 RGB drift | 24.214 | 38.955 |
| Mean skin-color-proxy RGB drift | 4.416 | 11.490 |
| Mean boundary luma-histogram distance | 0.0491 | 0.3775 |
| Segment-5 luma change | −24.628 | −41.032 |
| Segment-5 HSV saturation | 0.604 | 0.655 |

RGB values use a 0–255 scale; RGB drift is RMS distance between segment means.
Boundary distance is half the L1 distance of normalized 32-bin luma histograms.
The fixed skin-color threshold is a diagnostic proxy, with coverage recorded in
JSON, rather than face segmentation or an identity score.

This fixture shows 20.6% less mean RGB drift, 37.8% less final-segment RGB drift,
61.6% less mean skin-proxy drift and 87.0% smaller boundary histogram changes.
Native frames retain the hiker's face, backpack, skin tone and green setting as
she turns and waves. Recursive references repeatedly restart the turn/wave and
show larger facial/skin changes and a darker final segment. The native close
tracking view preserves shoulder/head/arm motion across boundaries; full-body
gait is outside these close-view fixtures.

Composition and scenery evolve, and some chroma/contrast measures favor the
recursive chain. This is a per-generation comparison: native delivery totals
294 frames (12.25 s), while recursive delivery totals 450 frames (18.75 s).
It establishes an improvement in this tested chain, not a general drift bound.
The full per-frame RGB, Cb/Cr, luma/contrast, saturation, histograms and skin-proxy
coverage are in `acceptance/metrics.json`; the standalone plot and first/middle/
last contact sheets are in `acceptance/review/` under the validation directory.

## Profiling and supported scope

A complete 256×256/90-frame state is 702,112 bytes including the 160-byte header.
Observed prefix copies take roughly 45–56 microseconds; save takes around 1 ms
and load rounds below 1 ms. Initial full VAE/config hashing takes about 3.7 s and
is cached per model context. Capturing final state transfers buffer ownership,
without an additional latent copy.

T2VA needs two additional shared timestep rows, about 19.4 MB of BF16 AdaLN
storage; image Ref2VA already has the visual row and needs one, about 9.7 MB.
No full spatial masks are allocated. A measured reuse-2 run spends 0.000595 s
across all 20 CPU mask/suffix Euler updates. Some acceptance jobs shared the GPU or were paused by the test scheduler,
so their elapsed times are not controlled speed benchmarks.

Continuation explicitly forces CPU F32 Euler, including when GPU-state sampling
is requested; DiT and VAE operations remain on Metal. GPU-state parity is not
claimed. Core reuse above one, token reduction, fewer than 50 blocks and first/
last-frame anchors are deliberately rejected, as permitted by the initial design.
The normal unmasked sampler and Metal paths retain their existing behavior.
