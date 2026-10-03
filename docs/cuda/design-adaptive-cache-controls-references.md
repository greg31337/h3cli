# Adaptive-cache controls and image, video, and audio references

Status: **implemented, qualified and user accepted on 2026-09-28**. Originally written against `e3065d9`
on 2026-09-28. The completed implementation checklist is
[archived here](../adaptive-controls-references-todo.md).
The previous cleanup is preserved in [its checklist](../legacy-cleanup-todo.md)
and [results](legacy-cleanup-results.md). Current evidence is in the
[qualification record](adaptive-controls-references-results.md).

## Objective and scope

Add two independent overrides to the existing adaptive presets:

- `--adaptive-cache-threshold T`: the maximum normalized probe change eligible
  for a hit; the actual comparison remains strictly `score < T`.
- `--adaptive-cache-max-hits N`: the maximum consecutive hits before a mandatory
  full refresh.

Support **image, video, and audio references**, including ordered mixed sets,
with BF16 adaptive caching and adaptive cache combined with BF16 SubBlock.
Preserve ordinary text-only defaults, all existing reference preprocessing,
and the native first-block/suffix algorithm.
Custom settings are user-selected approximation controls, not a new quality
guarantee or a claim of Cache-DiT numerical equivalence.

This extension covers all current Ref2VA input forms: `--ref-image`,
`--ref-silent-video`, `--ref-video` with embedded audio,
`--ref-video-audio VIDEO AUDIO` with a replacement soundtrack, and `--ref-audio`
accompanying an image or video. Preserve the existing limits: 12 reference records
total, at most 9 images, 3 videos and 3 audio inputs. Embedded/replacement audio
counts toward the audio limit; a video with audio is one reference record.
Separate audio must accompany an image or video; audio-only requests remain
unsupported under the current H3 contract.
All existing image-size modes and video/audio duration rules remain in force.

First/last-frame anchors, continuation, bridge, upscale refinement, LoRA/Turbo,
layer thinning, token reduction and Metal remain outside scope. Preserve ordinary
reference and SubBlock-only behavior. All quantized adaptive reference sets
remain rejected, regardless of media kind. Existing
text-only conservative FP8/NVFP4 execution accepts the new controls but retains
its BF16 probe, BF16 cache and current quantization/attention guards; no new
quantized video comparison campaign is required.

This is a native C/CUDA feature. No Python worker, external cache runtime,
dependency upgrade, server implementation or new SGLang oracle is needed.

## Pre-change behavior and relationship to Cache-DiT

The native policy in [adaptive_cache.c](../../src/denoise/adaptive_cache.c) uses four
warmup evaluations, threshold 0.04 and one consecutive hit for `conservative`,
or threshold 0.08 and three hits for `aggressive`. Warmup and memory are already
configurable. Each evaluation computes block 0 with dense BF16 attention; the
remaining 49 blocks form one cacheable suffix. The saved anchor is the first
block's BF16 residual at the most recent successful full refresh. Hits leave
both the anchor and saved suffix residual unchanged. Scores are not accumulated.

The [CUDA probe](../../src/cuda/cuda_adaptive_cache.cuh) already returns global,
generated-video and generated-audio scores using deterministic FP32 reductions.
Before this change, only the global score controlled reuse. [dit.c](../../src/denoise/dit.c) owns
the buffers and avoids streaming skipped suffix weights; final video/audio
heads and the Euler transition run on hits as well as refreshes.

The pre-change [reference guard](../../src/denoise/approximate.c) rejected all adaptive
reference layouts. Sampler loading also prohibited them before adaptive tensor
allocation; the implementation replaces both guards with shared validation. Current saved-file
requirements are defined in [the state contract](../features/current-state-contract.md).
Historical warmup/budget designs describe old readers that were removed during
cleanup; those compatibility paths must not be restored.

Cache-DiT's standard DBCache also compares a fresh front-block residual with its
last-refresh anchor and adds a saved middle-block residual on hits. It additionally
allows configurable fresh front/end blocks. Its standard score implementation
uses tensor arithmetic without our explicit denominator floor; accumulated-score
limits and forecasting are optional extensions. These are useful context, not
requirements to reproduce. See the [DBCache design](https://cache-dit.readthedocs.io/en/latest/user_guide/DBCACHE_DESIGN/),
[block wrapper](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/caching/cache_blocks/pattern_3_4_5.py)
and [decision implementation](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/caching/cache_contexts/cache_manager.py).
SGLang documents manually configured H3 Ref2VA caching separately from its
text-only audited quality preset in [the H3 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx#choose-the-quality-level).

## CLI and public API

| Option | Accepted values | Omitted on a new request |
| --- | --- | --- |
| `--adaptive-cache-threshold T` | Finite decimal FP32 value in `[0, 1]` | 0.04 conservative; 0.08 aggressive |
| `--adaptive-cache-max-hits N` | Decimal integer in `[1, 16]` | 1 conservative; 3 aggressive |
| Existing `--adaptive-cache-warmup N` | Explicit integer in `[2, 16]`, at most `steps-2` | 4 |
| Existing `--adaptive-cache-max-mib N` | Positive whole MiB with checked conversion | 4096 MiB |

The threshold controls the measured change, not an output error bound or a
percentage of blocks to skip. Zero deliberately disables hits while still
running cache probes and refresh bookkeeping. One remains an ordinary strict
threshold, **not** an unconditional-hit mode. The hit ceiling need not be less
than the step count: warmup and the final refresh naturally limit the attainable
streak. Never clamp settings, silently change the preset or implicitly enable
adaptive caching. No additional environment override or `custom` preset is added.

Accept decimal fractions and scientific notation for T, consuming the entire
argument in a locale-independent way. Reject empty strings, surrounding space,
hexadecimal floats, signed mantissas, NaN/infinity, trailing text, overflow,
underflow and values outside the range. Exponent signs are allowed. Check the
parsed value before and after conversion to FP32 so out-of-range values cannot
round into the accepted range. Canonicalize zero. N accepts decimal digits only;
reject zero, signs, fractions, overflow and suffixes.

Add `adaptive_cache_threshold` (float), `adaptive_cache_max_hits` (int) and
corresponding explicit-selection fields to `h3_params`. Follow existing API
conventions: a nonzero value or its explicit-selection bit supplies an override;
zero with no bit means unspecified. Threshold zero with its bit set is therefore
a valid explicit policy. Resolve mode, threshold and hit ceiling centrally after
parsing all options, independent of argument order. Validate library calls as
strictly as CLI calls. Keep an explicit-selection marker for resolved zero when
copying/restoring requests so it cannot revert to a preset default.

Both flags require enabled adaptive caching. Resume defers that requirement to
the saved request. Reject them for decode-only, still and other unsupported
operations before model loading. Preserve quantization guards based on the
selected preset; overriding aggressive to 0.04/1 must not accidentally enable
an otherwise unsupported aggressive+quantization request.

Implemented CLI examples:

```sh
./bin/h3cli -p 'The subject moves slowly through a sunlit scene.' \
  --width 640 --height 480 --frames 90 --steps 50 \
  --ref-image inputs/1.jpg --ref-image-size max \
  --adaptive-cache conservative \
  --adaptive-cache-threshold 0.06 --adaptive-cache-max-hits 2 \
  -o outputs/adaptive-reference.mp4

./bin/h3cli -p 'The subject moves slowly through a sunlit scene.' \
  --width 1344 --height 768 --frames 124 --steps 6 \
  --ref-image inputs/1.jpg --ref-image-size max \
  --adaptive-cache conservative --adaptive-cache-warmup 2 \
  --adaptive-cache-threshold 0.12 --adaptive-cache-max-hits 2 \
  --cuda-attention subblock --subblock-sparsity 0.75 --subblock-warmup 2 \
  -o outputs/adaptive-reference-subblock.mp4

# Video with its embedded soundtrack; media paths are user-supplied.
./bin/h3cli -p 'Continue the motion and sound of the reference scene.' \
  --width 640 --height 480 --frames 90 --steps 50 \
  --ref-video /path/to/reference-with-audio.mp4 \
  --adaptive-cache conservative \
  --adaptive-cache-threshold 0.06 --adaptive-cache-max-hits 2 \
  -o outputs/adaptive-video-reference.mp4

# An image and a separate audio reference, preserving their input order.
./bin/h3cli -p 'Use the subject and sound from the supplied references.' \
  --width 640 --height 480 --frames 90 --steps 50 \
  --ref-image inputs/1.jpg --ref-audio /path/to/reference.wav \
  --adaptive-cache conservative \
  -o outputs/adaptive-image-audio-reference.mp4
```

## Decision and refresh contract

Keep the BF16 probe and reconstruction arithmetic unchanged:

```text
probe = BF16(block0_output - block0_input)
score(region) = mean(abs(FP32(probe) - FP32(anchor)))
                / max(mean(abs(FP32(anchor))), 1e-6)

refresh: anchor = probe
         delta = BF16(block49_output - block0_output)
hit:     output = BF16(block0_output + delta)
```

For existing text-only layouts, continue using the current global score, with
the same packed range, reduction tree, arithmetic recipes and default decisions.
Do not change that score to a per-modality maximum as part of this extension.

For all new reference-media layouts, use
`decision_score = max(global_score, generated_video_score, generated_audio_score)`.
The global score covers the current valid packed layout, including text/vision
conditioning and reference hidden rows. The two target scores cover **generated
rows only**. Reconstruct ranges from validated layout metadata; never infer them
from reference count, tensor offsets guessed from T2VA, or rounded sequence lengths.
Require nonempty video and audio target ranges in these supported joint-AV jobs.
In particular, exclude reference-audio prefix rows from the generated-audio
score, and exclude image/video conditioning rows from the generated-video score.
Include both stereo target channels. Silent video references do not create
audio-condition rows; embedded, replacement and separate audio do. Derive this
from the resolved layout and provenance, not the presence of a filename suffix.

This policy prevents large static reference regions from diluting target changes,
and prevents numerous video rows from hiding audio changes. It still monitors
reference/text hidden changes through the global component. It is deliberately
more restrictive than global-only scoring at the same threshold; do not promise
a hit rate or compensate by silently increasing T. Log all three component
scores and the selected maximum. Assign new **adaptive arithmetic recipe 3** to
BF16 reference-media execution; recipes 1 (text BF16) and 2 (text quantized)
retain their existing meanings. This recipe number is independent of file schema
and section versions.

A hit requires a ready cache, consecutive original-schedule indices, completed
warmup, a nonfinal step, the same attention phase, `streak < effective_max_hits`,
and a finite nonnegative decision score strictly below the effective threshold.
Threshold equality refreshes. Empty/discontinuous history, warmup, final step,
the hit ceiling and the SubBlock transition force a refresh. Preserve the existing
reason priority in diagnostics. Nonfinite probes/reductions are errors even on
mandatory-refresh steps; never conceal them by falling back to full computation.

Model, layout, ordered reference content, execution recipe or effective numerical
policy changes must not reuse old history. Stage history during the forward;
commit it only after both heads and the sampler transition succeed. Cancellation
or failure must require reset/recovery and must not publish partial history.

## Reference conditioning and SubBlock

Admit supported Ref2VA media through the shared CLI/library validator, before
expensive loading wherever possible. Validate pointers/counts, kind, order,
soundtrack selection and existing per-kind/aggregate limits. An image/video
reference is required when separate audio is supplied; adding audio-only H3
generation is outside this change. Reject first/last anchors explicitly rather
than treating them as reference images.

| Input form | Required preserved behavior |
| --- | --- |
| Image | `match`, `high`, `max`; 65,536-raw-patch per-image behavior; image VAE and ordered Qwen image spans |
| Silent video | Released video pipeline; no reference-audio rows even if the source has a soundtrack |
| Video with embedded audio | Current soundtrack selection/validation, video and audio condition rows, synchronized provenance |
| Video plus external audio | External soundtrack replaces embedded audio; do not encode/count both soundtracks |
| Separate audio | Existing AudioVAE route, with at least one visual reference; preserve input order |
| Mixed set | Same media semantics and ordering as ordinary Ref2VA; count all video/audio contributions correctly |

Keep the sole released video preprocessing route: bounded 24 FPS decoding,
existing full-video Qwen/timestamp presentation, separate VAE prefix selection,
released 17-frame chunks, posterior sampling and reference RoPE. Preserve each
clip's 2–15-second bounds and the aggregate 15-second video limit. A video-only
visual request is supported; an image is not additionally required. Do not apply
`--ref-image-size` to reference video preprocessing.

Keep reference audio at the existing 32 kHz stereo F32 decode, posterior-mean
AudioVAE encoding, seeded conditioning noise, and pinned audio conditioning
timestep. Preserve each audio input's 2–15-second bounds, three-input ceiling and
aggregate 15-second audio duration limit, including embedded/replacement audio.
Retain current soundtrack duration/alignment and missing/invalid-stream errors;
never substitute generated audio for a failed reference encode. Reference audio
conditions generation; it does not promise byte-identical copying to the output.

Cache anchor and delta tensors span the entire packed hidden layout. Reference
hidden states inside the transformer may be approximated on a hit; **encoded
reference latents and their fixed conditioning timesteps remain unchanged**.
Only generated video/audio rows receive final-head output and Euler updates.
Test those invariants directly; static conditioning does not imply its internal
hidden activations are constant across timesteps.

For adaptive+SubBlock references, retain existing protected query/key ranges for
text, image/video vision conditioning and every image/video/audio reference row,
including both audio channels, plus the current target-frame
protections. Block 0 stays dense at every step. Sparse attention runs only in
executed suffix blocks after its warmup; a hit dispatches no suffix attention.
Force a refresh when the absolute schedule crosses into the sparse phase,
even when that transition follows a hit or a checkpoint. Do not change masks,
selection arithmetic or reference limits to make the new combination fit.

Keep current device qualification (SM120, CUDA 13+) and all other approximation
combination restrictions. Preserve ordinary reference generation, reuse 2/3,
and SubBlock-only reference audio/video behavior.

## Memory, streaming and context identity

The exact reservation remains `packed_rows * 5376 * 6 + 6168` bytes: two
persistent BF16 tensors, one BF16 scratch tensor and reduction storage. References
increase packed rows; image size and reference video/audio lengths are independent
of output resolution and generated duration. Include video condition rows,
reference-audio rows and Qwen vision spans. Use actual
encoded/layout counts for final admission, with checked arithmetic and the
existing configurable ceiling. An early generated-only lower bound may reject
clearly impossible budgets but cannot substitute for exact reference admission.

Reserve adaptive storage before weight admission and count it once. Preserve
SubBlock's separate workspace, existing device safety margins, bounded residency
retries and two-slot streaming behavior. A hit must avoid all suffix GEMMs,
attention/MLP work and suffix weight reads/uploads. Report actual residency and
memory; do not silently disable caching, downsize references or force residency.

Normalize omitted and explicitly matching controls to the same effective values
for policy comparisons and current prepared/live keys. Include the preset,
effective threshold's FP32 identity, effective hit ceiling, score recipe, warmups
and validated layout/reference identity wherever an execution context or adaptive
history can be reused. Preserve ordered media identity, image size-mode keys,
video preprocessing/timestamps, embedded-audio selection and both video/external
soundtrack identities. Changing only a soundtrack or the reference order must
invalidate incompatible history and prepared conditioning.
Immutable conditioning does not mathematically depend on these cache controls;
existing combined keys may conservatively invalidate it, but no broader cache-key
refactor is required. A resource-ceiling override remains separate from numerical
policy and still requires admission on a reused context.

## Persistence and current-only formats

Extend required **sampler section 40 from version 2 to version 3** with the
resolved FP32 threshold and resolved integer hit ceiling. Store the score recipe
through the adaptive recipe identity. Keep the existing sampler envelope schema 2
and BF16 payload sections 41/42; required-section versioning already distinguishes
this contract. The new reader accepts section-40 v3 only for adaptive checkpoints.
Do not restore v1/v2 fallback defaults or migrations. Nonadaptive sampler files
remain structurally unchanged.

Bump presentation schema **8 to 9**, serializing the effective threshold and hit
ceiling alongside the preset/recipe. Adaptive-off fields are zero; an enabled
zero threshold is distinguished by the enabled mode. Use canonical round-trippable
FP32 text, reject nonfinite/out-of-range values, and retain AV metadata binding.
Update all current writers, readers, test constructors and inspectors together;
there is no presentation-v8 backward reader. AV schema 3 is unchanged. Do not bump
conditioning/upscale envelopes merely for these flags, but update their keys and
shared sampler/presentation consumers where required, including cache-off paths.

On resume, omitted settings restore saved effective values. Explicit threshold
or hit-ceiling values must match exactly after canonical FP32/integer parsing;
changing either is rejected before model loading. Changing an admitted memory
ceiling remains allowed. Use original total steps for warmup checks and preserve
absolute step indices, streak, refresh index and attention phase. Validate streak
against the **saved effective ceiling**, not a hardcoded preset value.

Replace sampler loading's T2VA-only preallocation check with a bounded validator
for current T2VA and qualified multimodal Ref2VA layouts. Reconstruct/check ordered
references, condition tensor sizes, target ranges, sequence totals and recipe
before allocating sections 41/42. Check payload lengths, overflow, checksums,
finite values and all existing file-size limits. Serialized counts or a large
budget must not authorize arbitrary allocations. Negative/oversized histories,
forged reference kinds and recipe/layout mismatches must fail cleanly. Extend
the existing SubBlock reference provenance/layout checks to adaptive references:
validate media-kind/audio-presence agreement, stereo reference-audio sizes,
released-video recipe identity and exact reconstructed segment/RoPE layout.
Silent, embedded-audio and replacement-audio records must remain distinguishable.

Freshly written reference checkpoints must resume without original image, video
or audio files, including external soundtracks, or re-encoding conditioning.
Require current saved identities and packed conditioning; reject explicit attempts
to replace references, soundtrack selection, size mode or policy. Preserve both
visual and audio condition anchors exactly across hits, pause/resume and failure
recovery; do not interpret audio-reference rows as generated audio history.
Completed AV decode needs its current sidecar but no adaptive history/model.

## Validation and comparison plan

Use the existing RTX PRO 5000 server for CUDA, model integration and videos;
use local M4 for shared/host checks and a Metal build. Connection details belong
in the [task list](../adaptive-controls-references-todo.md), not portable
source/default paths. Preserve the qualified CUDA/cuDNN/media stack.
Documentation-only planning runs no GPU tests.

Keep the 12 recorded SGLang fixtures and all 204 expected hashes unchanged. Run
the complete gate after coherent implementation/test-tooling changes and once
more after final qualification, following [CONTRIBUTING.md](../../CONTRIBUTING.md).
Do not add approximate reference cases to that manifest or verify against a live
SGLang installation. New tests concern our policy, layout invariants, dispatch,
state integrity, same-build replay and observed media quality.

- CPU tests: strict parsing, API explicit zero, mode/argument order, all bounds,
  short schedules, default resolution, exact threshold equality, ceiling 1/2/3/16,
  maximum streak refresh, final/phase/discontinuity behavior and nonfinite errors.
- GPU tests: retain the existing probe oracle; add packed image/video/vision/audio
  reference and target layouts with ragged boundaries, tiny/zero anchor magnitude,
  an isolated generated-audio change, and large unchanged reference regions masking
  a target change globally. Prove the reference maximum prevents those false hits,
  distinguishes generated from reference stereo rows, and keeps condition latents
  fixed. Cover protected mixed query/key blocks with combined SubBlock.
- State tests: current section/presentation round trips, omitted/matching/conflicting
  overrides, explicit threshold zero, malformed counts/ranges/recipes, oversized
  declarations, stale-section rejection, atomic-write failure and decoder delivery.
  Include reordered media, silent/embedded/external soundtrack mislabeling, wrong
  audio lengths/presence and released-video provenance/segment mismatches.
- Bounded real-model tests: text defaults versus explicit defaults remain exact;
  image sets of 1, 2 and 9 with all size modes; silent video, video with embedded
  audio, video with a replacement soundtrack, image+audio, video+separate audio,
  and mixed image/video/audio sets. Exercise dense and combined SubBlock, both
  presets/custom controls, real hits/refreshes, resident/streamed suffix dispatch,
  cancellation/retry, prepared-context changes and exact uninterrupted-versus-resumed
  decisions and AV latents. Cover per-kind/total count and duration boundaries
  with host/layout tests, and representative real bounded jobs, not an exhaustive
  Cartesian product. Test that silent mode excludes a source's existing soundtrack
  and that external replacement cannot leak or double-count the embedded track.
  Preserve the six-evaluation ceiling per ordinary test invocation. Split original
  50-step schedules into bounded checkpoint segments when later indices are needed.
  Exercise memory admission at 1344×768/124 frames with a max-size reference, and
  checked large-layout sizing at 1344×768/362 frames without requiring another
  full-length video. Run the retained current host/CUDA suites and clean Metal/CUDA
  executable/library builds; no new FP8/NVFP4 render sweep.

Freeze prompts, seed 42, input file hashes, commands and the following 18-video
manifest before rendering. Use `inputs/1.jpg` and `inputs/2.jpg` for image rows.
Select short existing qualified audiovisual media and derive any missing silent,
embedded/replacement-audio or standalone WAV fixtures through reproducible FFmpeg
commands. Keep them under the ignored campaign directory; no extra H3 render is
needed to create inputs. Use two-second video/audio excerpts with distinguishable
embedded, replacement and standalone sound, valid stereo decode and recorded
source/derived hashes. Keep all grouped video/audio durations within current limits.
Choose one fixed prompt per comparison group after inspecting/listening to its
inputs, keeping prompt, ordered inputs and all non-cache settings identical within
that group. Preserve output FPS 24, full video/audio VAE, BF16, all 50 transformer
blocks, reuse/core-reuse 1 and automatic
weight placement. Each row produces **one video**; reuse its control for every
applicable comparison, and retain any failed attempts. Cache-off rows omit all
adaptive controls.

| ID | Output / frames / steps | References | Cache | Attention |
| --- | --- | --- | --- | --- |
| V01 | 640×480 / 90 / 50 | 1 / max | off | dense |
| V02 | 640×480 / 90 / 50 | 1 / max | conservative defaults | dense |
| V03 | 640×480 / 90 / 50 | 1 / max | aggressive defaults | dense |
| V04 | 640×480 / 90 / 50 | 1 / max | conservative, T=0.06, N=2, warmup 4 | dense |
| V05 | 1344×768 / 124 / 6 | 1 / max | off | dense |
| V06 | 1344×768 / 124 / 6 | 1 / max | conservative, T=0.12, N=2, warmup 2 | dense |
| V07 | 1344×768 / 124 / 6 | 1 / max | same as V06 | SubBlock 0.75, warmup 2 |
| V08 | 640×480 / 90 / 6 | 1+2 / high | off | dense |
| V09 | 640×480 / 90 / 6 | 1+2 / high | conservative, T=0.12, N=2, warmup 2 | SubBlock 0.75, warmup 3 |
| V10 | 640×480 / 90 / 6 | one video with embedded audio | off | dense |
| V11 | 640×480 / 90 / 6 | same as V10 | conservative, T=0.12, N=2, warmup 2 | dense |
| V12 | 640×480 / 90 / 6 | same as V10 | same as V11 | SubBlock 0.75, warmup 3 |
| V13 | 640×480 / 90 / 6 | image 1 / max + separate audio | off | dense |
| V14 | 640×480 / 90 / 6 | same as V13 | conservative, T=0.12, N=2, warmup 2 | dense |
| V15 | 640×480 / 90 / 6 | same as V13 | same as V14 | SubBlock 0.75, warmup 3 |
| V16 | 640×480 / 90 / 6 | image 1 / high + silent video + video with replacement soundtrack + separate audio | off | dense |
| V17 | 640×480 / 90 / 6 | same ordered mixed set as V16 | conservative, T=0.12, N=2, warmup 2 | dense |
| V18 | 640×480 / 90 / 6 | same ordered mixed set as V16 | same as V17 | SubBlock 0.75, warmup 3 |

Rows V01–V09 retain their image-only inputs; 1 and 2 denote the numbered input
images. The mixed group has four reference records, two videos and two audio
inputs: one replacement soundtrack and one standalone clip. Its silent-video
source intentionally contains an embedded soundtrack to verify that silent mode
ignores it. Reuse exactly the same frozen media and prompt within each triplet.

Only V01–V04's explicit comparison subprocesses may raise the ordinary test limit
to 50. Do not change the global Makefile/test ceiling. Record total wall time,
denoise time, cache scores/reasons/hits, executed blocks, sparse calls, residency,
transfers and peak memory. Validate dimensions, frame count, duration, full decode
and finite/nonempty stereo audio. Publish synchronized playback/contact sheets
and same-seed video/audio similarity diagnostics against each group's control:
V01, V05, V08, V10, V13 or V16. Those diagnostics describe differences, not new
SGLang parity or absolute quality.

Deterministic operators and real jobs must exercise actual hits, refreshes and
exact resume for image-only, video-bearing and audio-bearing reference layouts,
including a mixed set. Cover both dense and combined SubBlock paths through
representative jobs. If a fixed video group has zero hits, report that honestly
and use separately labeled bounded latent diagnostics to obtain its hit coverage;
do not tune/re-render the fixed comparison or claim a speedup. Six-step videos
validate execution and capacity; V01–V04 provide a longer-schedule image-reference
visual comparison, not equivalent long-schedule validation of every media kind.
Report reference identity, motion, artifacts, soundtrack influence and AV timing
observations separately from mechanical pass/fail. Prior acceptance of text-only
adaptive videos does not qualify these new reference outputs.

## Completion criteria

Both flags are documented and enforced across CLI/API, context reuse, saved
state and resume. Text defaults remain unchanged. BF16 adaptive and adaptive+SubBlock
support all listed image, video and audio reference forms and their valid mixed
sets, with explicit resource admission and preserved visual/audio conditioning.
The retained tests, clean builds, 18-video report and unchanged 204/204 final gate
are complete. Publish measured results and
remaining limitations, distinguish human review status from test results, and
update the current state contract and feature documentation without introducing
backward readers or unrelated cleanup.
