# Technical Design: Released-Compatible Ref2VA Video Conditioning Pipeline

## 1. Objective

Replace the current default Ref2VA **video-reference** preprocessing path with a pipeline that reproduces the released MiniMax H3 conditioning recipe, while preserving the existing implementation unchanged behind:

```text
--legacy-ref2va-video-pipeline
```

The new pipeline becomes the default.

The legacy flag applies only to Ref2VA video references:

```text
--ref-video
--ref-silent-video
--ref-video-audio
```

It must not change:

* Ref2VA image-reference encoding;
* FL2VA first/last-frame behavior;
* T2VA;
* generated-video VAE decoding;
* latent continuation;
* audio-VAE behavior.

The purpose is to correct the Ref2VA video-reference preprocessing contract without destabilizing unrelated working paths.

---

## 2. Current attached-source behavior

The attached source currently implements Ref2VA video conditioning approximately as:

```text
video file
   |
   v
FFmpeg
   |
   +-- resample to 24 fps
   +-- resize
   +-- cap to target frame count
   +-- trim frame count DOWN to 5 + 17*n
   |
   v
condition_pixels
   |
   +-----------------------------> Qwen vision sampling
   |
   v
h3_video_vae_encode()
   |
   v
one continuous causal CNN pass over entire video
   |
   v
48 output moment channels
   |
   v
discard channels 24..47
   |
   v
use channels 0..23 directly
   |
   v
latent mean/std normalization
   |
   v
ceil(T / 4) latent frames
   |
   v
patchify
   |
   v
Ref2VA condition rows
```

`h3_video_encoder_latent_t()` therefore currently returns:

```c
(frame_count + 3) / 4
```

The same temporal count is used for:

* condition-row allocation;
* `h3_layout_ref.latent_t`;
* patchification;
* reference RoPE span;
* reference cursor advancement;
* condition augmentation spans.

The existing raw CNN implementation is useful and should be retained. The problem is that its raw `ceil(T/4)` output is currently treated as the completed Ref2VA conditioning representation.

---

## 3. Released MiniMax H3 behavior

The released Ref2VA VAE configuration specifies:

```text
clip_length = 17
token_drop = 3
latent_channels = 24
tile_size = 256
tile_overlap_min = 64
```

MiniMax's temporal wrapper pads the input to 17-frame chunks, runs each chunk independently through the spatial/causal encoder, concatenates the encoded chunks, and removes three trailing latent tokens.

For legal video-reference lengths, this produces:

```text
17*n + 5 RGB frames
        |
        v
5*n + 2 latent frames
```

Examples:

```text
39 frames  -> 12 latent frames
56 frames  -> 17 latent frames
73 frames  -> 22 latent frames
107 frames -> 32 latent frames
124 frames -> 37 latent frames
```

The released conditioning recipe also does **not** simply take the first 24 VAE output channels. It constructs the VAE posterior, samples it using a fresh generator with conditioning seed 42, converts the sample through FP16, converts it back to FP32, and only then applies latent mean/std normalization. Diffusers explicitly documents these operations as required to reproduce released conditioning.

For video references, current Diffusers also keeps the normalized 24-fps reference as the general reference object but independently snaps the VAE view down to `17*n+5` immediately before video-VAE encoding.

The new implementation should follow these semantics.

---

## 4. Pipeline selection

Add:

```text
--legacy-ref2va-video-pipeline
```

with default:

```text
false
```

Resulting behavior:

```text
Ref2VA image reference
    -> existing image path

Ref2VA video + legacy flag
    -> exact current h3cli video path

Ref2VA video without legacy flag
    -> new released-compatible video path
```

The legacy path must remain available for:

* regression comparison;
* existing user workflows;
* diagnosing behavioral differences;
* reproducing earlier h3cli renders.

No attempt should be made to silently reinterpret old cached conditioning or sampler checkpoints as new-pipeline conditioning.

---

## 5. Separate normalized-video length from VAE-video length

The current source has effectively one video length:

```text
condition_frames[]
```

That is insufficient for the released pipeline.

The new path should maintain at least:

```text
condition_frames[]
    complete normalized 24-fps reference used by Qwen/audio

condition_vae_frames[]
    prefix actually presented to the video VAE

condition_latent_t[]
    resulting temporal latent count
```

For example, if decoding produces 60 normalized frames:

```text
normalized reference:
60 frames
    |
    +----------------------> Qwen sees normalized video
    |
    +----------------------> synchronized soundtrack duration
    |
    v
VAE snap-down
56 frames
    |
    v
17 Ref2VA latent frames
```

This distinction follows the current Diffusers organization: references are first normalized onto H3's 24-fps rate, while the reference VAE step separately snaps a video down to legal H3 VAE geometry.

The legacy path must retain the current behavior in which FFmpeg trims the video before all downstream users see it.

---

## 6. Video decoding

Preserve the existing:

```c
h3_ffmpeg_read_video_f32()
```

semantics for legacy mode.

Add a new normalized-video decoder, or parameterize the underlying decoder internally, so the new path performs:

```text
source video
    |
    v
24-fps normalization
    |
    v
reference-video spatial resize
    |
    v
target-duration cap
    |
    v
NO 17n+5 cadence trimming
```

The resulting F32 channel-major RGB array becomes the normalized reference.

Do not mutate this buffer when selecting the VAE prefix.

Instead pass:

```text
condition_pixels
+
condition_vae_frames
```

to the new VAE encoder.

This also allows Qwen to use the complete normalized reference rather than the VAE-truncated copy.

---

## 7. Reference-video duration validation

The released H3 Ref2VA contract supports up to three video clips, each 2–15 seconds long, with total video-reference duration no greater than 15 seconds.

The new pipeline should enforce these limits against the normalized 24-fps reference duration.

Legacy mode should retain existing validation behavior for backwards compatibility.

Duration validation must occur before VAE snap-down; trimming 48 normalized frames to 39 VAE frames must not cause a valid approximately 2-second input to be rejected.

---

## 8. VAE frame selection

Introduce a single released-pipeline helper such as:

```c
int h3_ref2va_video_vae_frames(int normalized_frames);
```

For supported video references it returns the greatest legal:

```text
17*n + 5
```

frame count not exceeding the normalized reference.

Example:

```text
48 normalized -> 39 VAE frames
56 normalized -> 56 VAE frames
60 normalized -> 56 VAE frames
73 normalized -> 73 VAE frames
```

Then:

```c
h3_ref2va_video_latent_t(vae_frames)
```

must produce:

```c
h3_video_latent_t(vae_frames)
```

rather than:

```c
h3_video_encoder_latent_t(vae_frames)
```

`h3_video_encoder_latent_t()` should remain untouched and explicitly documented as the **legacy/raw causal-encoder geometry helper**.

---

## 9. Refactor the native encoder into two abstraction levels

The current native video encoder conflates:

```text
raw CNN moment generation
+
posterior selection
+
latent normalization
```

The new architecture should expose these separately.

Conceptually:

```text
LOW LEVEL

RGB
 |
 v
ImageNet normalization
 |
 v
native causal CNN
 |
 v
quant_conv
 |
 v
48 posterior-parameter channels


HIGH LEVEL — legacy

48 channels
 |
 v
first 24 channels
 |
 v
latent normalization
 |
 v
legacy latent


HIGH LEVEL — released Ref2VA video

48 channels
 |
 v
temporal wrapper
 |
 v
posterior sampling
 |
 v
FP16 round trip
 |
 v
latent normalization
 |
 v
released Ref2VA latent
```

The legacy public behavior must remain unchanged.

---

## 10. Raw moment representation

Introduce an internal representation similar to:

```c
typedef struct {
    int time;
    int height;
    int width;

    /* Channel-major F32 [48,T,H,W].
     * Channels 0..23: posterior mean.
     * Channels 24..47: posterior log-variance.
     */
    float *values;

    h3_gpu_stats gpu_stats;
} h3_video_moments;
```

The raw encoder must return all 48 quant-conv output channels before:

* selecting the mean;
* sampling the posterior;
* FP16 rounding;
* latent normalization.

The current `encode_tile()` already reads all 48 channels from the GPU, so this should mainly require retaining and reorganizing data that is currently discarded.

---

## 11. Spatial tiling

The existing native implementation preserves the released 256-pixel tile size and minimum 64-pixel overlap.

The new path must preserve that geometry, but spatial stitching should occur on the **48-channel encoder output**, before posterior sampling and normalization.

This matches MiniMax's encoder organization: `tiled_encode()` applies the underlying encoder independently to spatial tiles and blends the encoded outputs before returning the encoded tensor.

Therefore the new conceptual order should be:

```text
17-frame temporal chunk
        |
        v
spatial tiles
        |
        v
raw CNN -> 48 moments per tile
        |
        v
blend/stitch 48-channel moments
        |
        v
one full-spatial moment tensor
```

Do not spatially blend separately sampled posterior latents.

The existing legacy `stitch_latents()` behavior should remain available unchanged.

---

## 12. Released temporal wrapper

The new high-level video-reference encoder processes temporal data as follows.

Suppose the VAE-selected reference contains:

```text
F = 17*n + 5
```

frames.

Pad it by repeating its final frame until the temporary working length is divisible by 17.

For a 56-frame example:

```text
56 original VAE frames
        |
        + 12 copies of final frame
        |
        v
68 working frames
        |
        v
4 independent 17-frame chunks
```

Each 17-frame raw causal encoding produces:

```text
5 moment slices
```

so:

```text
4 * 5 = 20
```

moment slices are concatenated.

Then:

```text
token_drop = 3
```

removes the final three:

```text
20 - 3 = 17
```

giving the released 56-frame reference geometry.

The temporal encoder must restart its causal boundary for every 17-frame chunk; do not run the raw CNN once continuously across all working frames.

That independent chunking is a material part of the released pipeline.

---

## 13. Temporal processing examples

Expected native results:

```text
39 frames
pad to 51
3 x 17-frame chunks
15 raw moment slices
drop 3
= 12 final moment slices

56 frames
pad to 68
4 chunks
20 raw moment slices
drop 3
= 17

73 frames
pad to 85
5 chunks
25 raw moment slices
drop 3
= 22

107 frames
pad to 119
7 chunks
35 raw moment slices
drop 3
= 32
```

Every result must equal:

```c
h3_video_latent_t(vae_frames)
```

Failure of this invariant is an encoder error.

---

## 14. Posterior sampling

After temporal chunking and token dropping, the tensor contains:

```text
[48,T,H,W]
```

posterior parameters.

Split:

```text
mean   = channels 0..23
logvar = channels 24..47
```

Use the MiniMax diagonal-Gaussian semantics:

```text
logvar = clamp(logvar, -30, 20)

std = exp(0.5 * logvar)

latent = mean + std * epsilon
```

The production Ref2VA conditioning seed is:

```text
42
```

and must be independent of:

```text
params->seed
```

A fresh reference-posterior generator should be initialized to seed 42 for each visual-condition encode, matching the released conditioning behavior.

Do not simply reuse the request's video-noise RNG.

---

## 15. Posterior RNG parity

Exact PyTorch posterior sampling parity should be treated separately from correctness of the Gaussian formula.

Provide an internal sampling operation that can accept an explicit:

```text
epsilon[]
```

tensor.

This allows deterministic oracle testing:

```text
official moments
+
oracle epsilon
        |
        v
official sample

native moments
+
same oracle epsilon
        |
        v
native sample
```

Once the formula is proven, compare the native seed-42 RNG stream with PyTorch's released seed-42 stream.

If the current `h3_rng` does not reproduce the official normal samples, introduce a dedicated Ref2VA-posterior RNG implementation rather than changing the project's general sampling RNG.

This keeps model-generation determinism independent from conditioning-encoder determinism.

---

## 16. FP16 conditioning quantization

After posterior sampling and **before** latent normalization, perform the released FP16 round trip:

```text
F32 posterior sample
        |
        v
IEEE FP16
        |
        v
F32
        |
        v
latent mean/std normalization
```

Diffusers explicitly identifies this rounding as part of reproducing H3's conditioning representation.

Use deterministic IEEE-754 round-to-nearest-even behavior.

Do not replace this with BF16.

---

## 17. Latent normalization

After FP16 round-trip:

```text
normalized[c,t,y,x] =
    (sample[c,t,y,x] - latents_mean[c])
    / latents_std[c]
```

using the existing values loaded from:

```text
Ref2VA/video_vae/config.json
```

The attached native encoder already loads these values, so the new pipeline should reuse that code.

Normalization remains F32.

---

## 18. New high-level API

Keep:

```c
h3_video_vae_encode()
```

with its current behavior for legacy callers and image/keyframe paths.

Add a dedicated API conceptually similar to:

```c
h3_ref2va_video_vae_encode(...)
```

whose contract is:

```text
input:
    normalized RGB reference
    VAE-selected legal frame count

processing:
    released 17-frame temporal chunking
    48-channel moment handling
    posterior sample with conditioning seed
    FP16 round-trip
    normalization

output:
    h3_video_latent [24,5*n+2,H,W]
```

This separation prevents accidental changes to working FL2VA/image-reference behavior.

---

## 19. `src/engine.c` visual-reference bookkeeping

The current code recomputes:

```c
h3_video_encoder_latent_t(condition_frames[image])
```

in several places.

Replace this with one authoritative per-visual value established during media preparation:

```text
condition_latent_t[visual]
```

Similarly maintain:

```text
condition_frames[visual]
condition_vae_frames[visual]
```

For an image:

```text
condition_frames    = 1
condition_vae_frames = 1
condition_latent_t   = 1
```

For a released-pipeline video:

```text
condition_frames     = complete normalized 24-fps count
condition_vae_frames = snapped 17*n+5 prefix
condition_latent_t   = h3_video_latent_t(condition_vae_frames)
```

For a legacy video:

```text
condition_frames
    = existing early-trimmed frame count

condition_vae_frames
    = condition_frames

condition_latent_t
    = h3_video_encoder_latent_t(condition_frames)
```

No downstream code should independently infer video condition geometry.

---

## 20. Qwen video conditioning

In the new pipeline, Qwen should sample the complete normalized 24-fps video:

```text
condition_frames
```

not:

```text
condition_vae_frames
```

The existing 2-fps sampling logic can initially remain unchanged.

This matters for videos whose normalized frame count is not already `17*n+5`.

For example:

```text
60 normalized frames:

Qwen:
    sees samples over all 60 frames

video VAE:
    sees frames 0..55
```

This matches the separation in the current released Diffusers implementation, where Ref2VA text conditioning reads normalized references while the VAE encoder independently snaps video references down immediately before VAE encoding.

The Qwen sampling/timestamp implementation should nevertheless receive an explicit oracle regression test.

---

## 21. Reference soundtracks

The new pipeline should base video-soundtrack synchronization on the complete normalized video duration rather than accidentally inheriting VAE cadence truncation.

The VAE-specific snap-down is an implementation detail of visual latent conditioning and should not implicitly shorten:

* Qwen reference presentation;
* the logical reference-video duration;
* the associated audio before its own documented limits are applied.

The legacy path must preserve the current behavior exactly.

This behavior should be validated against the released Ref2VA oracle before rollout.

---

## 22. Layout and RoPE geometry

For new video references:

```text
h3_layout_ref.latent_t
```

must equal the final released-compatible video latent count.

Therefore a 56-frame VAE reference must have:

```text
latent_t = 17
```

not:

```text
latent_t = 14
```

This corrected geometry automatically changes:

* condition row count;
* reference block size;
* video RoPE coordinates;
* reference temporal span;
* shared reference cursor advancement;
* target start position after ordered references.

This is intentional.

The official H3 Ref2VA layout derives each reference block's geometry from its encoded condition latent shape, and reference order advances the shared rotary clock.

---

## 23. Condition augmentation

Do not change the existing:

```text
0.999 * condition + 0.001 * noise
```

condition-row augmentation as part of this work.

The corrected Ref2VA video latent should enter the existing augmentation stage after patchification.

Because augmentation span is currently derived from:

```text
h3_layout_ref.latent_t
```

correcting the layout temporal count should automatically correct the span.

Any attempt to reproduce additional released conditioning-RNG ordering should be a separate follow-up unless oracle testing demonstrates it is required for Ref2VA parity.

---

## 24. Conditioning cache

The new and legacy video pipelines can produce different:

* video latent values;
* latent counts;
* packed row counts;
* layouts;
* Qwen presentation inputs.

Therefore the conditioning cache key must explicitly contain:

```text
ref2va-video-pipeline=released-v1
```

or:

```text
ref2va-video-pipeline=legacy
```

A cached legacy condition must never satisfy a new-pipeline request.

Because the prepared-model key incorporates the conditioning key in the attached source, this also separates prepared DiT state.

---

## 25. Sampler checkpoint/state compatibility

`.h3sample` stores:

* `h3_params`;
* packed condition rows;
* layout references;
* exact prepared state.

The pipeline selector must therefore be persisted as part of sampler parameters/provenance.

Bump the sampler-state schema or required parameter serialization version as needed so an older state cannot silently reinterpret the missing field.

A resumed sampler should continue using its serialized condition rows; it should not re-encode references during resume.

However, state validation must detect attempts to combine a checkpoint created with one Ref2VA video-pipeline version with parameters requesting another.

---

## 26. Legacy compatibility contract

With:

```text
--legacy-ref2va-video-pipeline
```

the following must remain unchanged relative to the attached source:

```text
FFmpeg cadence trim
ceil(T/4) temporal geometry
continuous whole-video causal encoding
first 24 moment channels used directly
no posterior sampling
no FP16 latent round-trip
existing Qwen frame selection
existing soundtrack truncation
existing reference RoPE span
```

The strongest regression requirement is that existing deterministic legacy tests continue to pass unchanged.

Where practical, legacy condition rows should be bit-identical to the current implementation.

---

## 27. Oracle strategy

Do not validate the new pipeline only through completed rendered videos.

Create an official Python oracle using the current released MiniMax/Diffusers implementation and save intermediate fixtures.

For each fixture preserve:

```text
normalized RGB frames
VAE-selected frame count
per-chunk raw 48-channel moments
post-concatenation/post-token-drop moments
posterior epsilon
posterior sample
FP16-rounded sample
normalized latent
patchified condition rows
reference latent geometry
reference layout/RoPE positions
```

The native implementation should match the pipeline progressively.

Visual generation A/B testing happens only after numerical encoder/layout parity has been established.

---

## 28. Test inputs

Use deterministic synthetic videos in addition to real clips.

At minimum test legal VAE lengths:

```text
39
56
73
107
124
```

and normalized non-cadence lengths such as:

```text
48
60
80
110
```

to verify that:

```text
Qwen duration != necessarily VAE duration
```

Also test:

```text
64x64
```

to isolate temporal behavior, and at least one dimension greater than 256 pixels to exercise spatial tiling.

Use simple deterministic patterns whose frame number and pixel coordinates are encoded into RGB values so temporal or spatial misalignment is obvious.

---

## 29. Required parity layers

Validate in this order:

```text
1. raw 17-frame native CNN moments
2. spatially stitched 48-channel moments
3. temporal chunk concatenation
4. token_drop=3
5. mean/logvar split
6. posterior sampling with supplied epsilon
7. seed-42 posterior RNG
8. FP16 round-trip
9. latent normalization
10. patchification
11. condition-row allocation
12. reference layout
13. RoPE/reference cursor
14. one-step DiT input/output
15. complete generation
```

Do not skip directly from latent shape parity to visual output parity.

---

## 30. Success criteria

The new pipeline is complete when:

```text
56-frame VAE reference -> 17 temporal latents
73-frame VAE reference -> 22 temporal latents
107-frame VAE reference -> 32 temporal latents
```

and the native results match the released oracle within explicitly defined numerical tolerances through:

* moments;
* posterior sampling;
* FP16 quantization;
* normalization;
* patchification;
* packed layout.

Legacy mode must continue producing its original:

```text
56 -> 14
73 -> 19
107 -> 27
```

geometry.

End-to-end Ref2VA A/B tests should then establish whether the corrected pipeline improves:

* video motion-reference fidelity;
* scene-reference fidelity;
* identity retention;
* audio/video reference alignment;
* multi-reference behavior;
* stability of the target layout.

---

# Remaining tasks

* [OPEN] T090: Declare the new pipeline production-ready only after raw moments, temporal wrapper, posterior sample, FP16 normalization, patchified rows, reference layout, and at least one complete Ref2VA generation have all passed their corresponding oracle tests.
