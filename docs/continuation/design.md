# Technical Design: Native Masked AV Latent Continuation

## 1. Objective

Add native long-video continuation to a clean upstream `h3cli` implementation by carrying the final H3 video/audio latent state from one generated segment into the next segment.

The continuation path shall:

1. retain the final denoised joint video/audio latent produced by segment N;
2. extract an aligned tail from that latent;
3. construct the target AV latent for segment N+1;
4. initialize the target normally with noise;
5. overwrite the beginning of the new target with the previous segment's latent tail;
6. mark the copied video and audio portions as preserved;
7. run the DiT with different effective timesteps for preserved and generated rows;
8. force zero denoising velocity for preserved rows;
9. generate only the suffix;
10. decode the complete target internally;
11. remove the duplicated protected prefix from the delivered continuation clip;
12. retain the completed segment-N+1 AV latent for the next continuation.

The design must support:

* T2VA continuation;
* Ref2VA continuation;
* all existing Ref2VA image/video/audio reference types;
* the same references on every segment;
* different references on different segments;
* different prompts on different segments;
* multi-segment chains without decoding/re-encoding the continuation context.

The previous rendered MP4 must **not** automatically become a Ref2VA video reference. Continuation state and Ref2VA references are separate mechanisms.

---

# 2. Existing h3cli model assumptions

Clean `h3cli` already exposes the relevant native representations:

* video target latent: F32 `[24,T,H,W]`;
* audio target latent: F32 `[32,2,T]`;
* video rate: 24 fps;
* audio latent rate: 40 Hz;
* separate shifted video and audio sigma schedules;
* a packed DiT sequence containing text/reference/condition/audio/video rows;
* separate FL2VA and Ref2VA conditioning paths.

The current Ref2VA path already supports ordered image, video and audio references and selects the Ref2VA checkpoint whenever `reference_count != 0`. Existing reference ordering and limits should remain unchanged.

Continuation therefore does **not** require a new model checkpoint or another conditioning modality.

It modifies the target AV latent and target-row denoising behavior.

---

# 3. High-level generation pipeline

For the initial segment:

```text
prompt + optional Ref2VA references
                |
                v
        normal H3 generation
                |
                v
   final clean video/audio latent
                |
         +------+------+
         |             |
         v             v
      decode        save AV state
         |
         v
     segment-1.mp4
```

For continuation segment N:

```text
previous AV state
       |
       | extract final aligned context
       v
+--------------------+
| previous latent    |
| video tail         |
| audio tail         |
+--------------------+
       |
       v

new random target latent

VIDEO:
[preserved previous tail][................ fresh noise ................]
|<---- mask 0 ------->|<--------------- mask 1 ---------------------->|

AUDIO:
[preserved previous tail][................ fresh noise ................]
|<---- mask 0 ------->|<--------------- mask 1 ---------------------->|
       |
       +--------------------+
                            |
prompt + optional Ref2VA --> H3 DiT
references                   |
                             v
                    final AV target latent
                             |
                +------------+-------------+
                |                          |
                v                          v
       save complete latent          decode target
                                           |
                                    discard prefix
                                           |
                                           v
                                      segment-N.mp4
```

The copied prefix stays part of the **target timeline**. It must not be converted into `H3_SEG_COND`, `H3_SEG_REF_IMAGE`, or another reference segment.

That distinction is critical because the prefix represents the actual current state of the scene rather than an advisory reference.

---

# 4. Temporal alignment

H3 target video lengths obey:

```text
F = 5 + 17k
```

and clean `h3cli` computes:

```text
video_latent_t =
    2                         when F <= 5
    5k + 2                    otherwise

audio_latent_t =
    round(F * 40 / 24)
```

For continuation, the safest initial implementation should require a context length that is simultaneously:

1. a legal H3 video run; and
2. an exact integer number of 40 Hz audio latent ticks.

Solving those constraints gives:

```text
context_frames = 39 + 51n
```

Therefore:

| Context frames | Video latent steps | Audio latent ticks | Duration |
| -------------: | -----------------: | -----------------: | -------: |
|             39 |                 12 |                 65 |  1.625 s |
|             90 |                 27 |                150 |  3.750 s |
|            141 |                 42 |                235 |  5.875 s |
|            192 |                 57 |                320 |  8.000 s |

Current H3 masked-continuation implementations have independently converged on the same `39 + 51k` shared AV boundaries, with 39 frames recommended as the normal starting point.

### Initial policy

Default:

```text
continuation_context_frames = 39
```

Initially accept only:

```text
39, 90, 141, 192, ...
```

Reject an invalid context rather than silently creating an AV timing mismatch.

A later feature may introduce independent video/audio context lengths, but it should not complicate the first implementation.

---

# 5. Net-new versus raw target duration

There are two lengths:

```text
raw_target_frames
```

and:

```text
net_new_frames =
    raw_target_frames - continuation_context_frames
```

Example:

```text
raw target:       362 frames
protected prefix:  39 frames
new output:       323 frames
```

At 24 fps this produces approximately:

```text
13.46 seconds of new video
```

even though H3 internally generated a 362-frame target.

The existing `--frames` option should continue to mean **raw H3 target length**. This preserves backwards compatibility.

An optional convenience parameter may later be added:

```text
--continue-new-frames N
--continue-new-seconds S
```

which chooses an appropriate legal raw target length and reports the actual resulting duration.

Because H3's new-content increment will necessarily be a multiple of 17 frames, exact arbitrary net durations cannot always be represented.

---

# 6. Continuation-state object

Introduce an in-memory object:

```c
typedef struct h3_av_state h3_av_state;
```

Conceptually it contains:

```text
format/version
model/VAE compatibility signature

render_width
render_height

frame_count
video_t
latent_h
latent_w
audio_t

video_channels = 24
audio_channels = 32
audio_streams = 2

video F32 latent [24,T,H,W]
audio F32 latent [32,2,T]

generation metadata:
    seed
    source target frame count
    optional segment number
```

The state should store the **complete final latent**, not only the current 39-frame tail.

Advantages:

* context size can be changed later without regenerating the previous segment;
* debugging is easier;
* saved clips can be resumed;
* 39/90/141-frame continuation can be compared from the same source;
* the state can be decoded independently for diagnostics.

The latents should initially be serialized as lossless F32. Do not quantize them.

A versioned binary file should include:

```text
magic
format version
header size
geometry
temporal dimensions
payload sizes
model/VAE signature
video payload
audio payload
checksum
```

Suggested extension:

```text
.h3av
```

Example:

```text
segment01.h3av
segment02.h3av
segment03.h3av
```

---

# 7. Public API

Add continuation support to `h3_params` without changing normal generation behavior.

Conceptually:

```c
const h3_av_state *continuation;
int continuation_context_frames;
int trim_continuation_prefix;
```

Default:

```text
continuation = NULL
continuation_context_frames = 39
trim_continuation_prefix = 1
```

`NULL` must execute the existing generation path unchanged.

`h3_result` should own the AV state produced by the generation.

Add APIs conceptually equivalent to:

```c
const h3_av_state *h3_result_av_state(const h3_result *result);

int h3_av_state_save(
    const h3_av_state *state,
    const char *path,
    char *error,
    size_t error_size);

h3_av_state *h3_av_state_load(
    const char *path,
    char *error,
    size_t error_size);

void h3_av_state_free(h3_av_state *state);
```

The result-owned state remains valid until `h3_result_free()`.

---

# 8. CLI

Add:

```text
--continue-from PATH
--continue-context N
--save-av-state PATH
--keep-continuation-prefix
```

Semantics:

### First clip

```bash
./bin/h3cli \
  -d ./models/MiniMax-H3 \
  -p "A woman walks through a forest." \
  --frames 362 \
  --ref-image face1.jpg \
  --ref-image body1.jpg \
  --save-av-state segment01.h3av \
  -o segment01.mp4
```

### Continue using the same references

```bash
./bin/h3cli \
  -d ./models/MiniMax-H3 \
  -p "The woman continues walking through the forest and looks upward." \
  --frames 362 \
  --continue-from segment01.h3av \
  --continue-context 39 \
  --ref-image face1.jpg \
  --ref-image body1.jpg \
  --save-av-state segment02.h3av \
  -o segment02.mp4
```

### Continue while changing references

```bash
./bin/h3cli \
  -d ./models/MiniMax-H3 \
  -p "The woman keeps walking while another character approaches." \
  --frames 362 \
  --continue-from segment02.h3av \
  --continue-context 39 \
  --ref-image face2.jpg \
  --ref-image body2.jpg \
  --save-av-state segment03.h3av \
  -o segment03.mp4
```

The continuation latent is **not** assigned a `<Video N>` reference label.

`<Picture 1>`, `<Picture 2>`, `<Video 1>`, etc. continue to refer only to the explicit Ref2VA references supplied for the current segment.

---

# 9. Geometry validation

Continuation requires compatible latent geometry.

Before generation verify:

```text
previous latent_h == new latent_h
previous latent_w == new latent_w
same video latent channel count
same audio latent representation
compatible H3 VAE/model signature
previous clip contains requested context
target contains requested context
target contains at least some generative suffix
```

For robust multi-segment chaining, additionally require:

```text
net_new_frames >= continuation_context_frames
```

or issue a strong warning.

Otherwise the tail used for the following continuation could contain a substantial amount of already inherited context rather than newly generated content.

Changing the output display resolution while keeping the internal H3 render geometry identical may eventually be supported, but the first implementation should require exact latent geometry.

---

# 10. Tail extraction

For context frame count `C`:

```text
video_context_t = h3_video_latent_t(C)
audio_context_t = C * 40 / 24
```

For 39:

```text
video_context_t = 12
audio_context_t = 65
```

Video source layout is:

```text
[24, source_T, H, W]
```

For every channel, copy:

```text
source:
    T - video_context_t ... T-1

destination:
    0 ... video_context_t-1
```

Audio source layout is:

```text
[32, 2, source_audio_T]
```

For every audio channel and stereo stream copy:

```text
source:
    source_audio_T - audio_context_t ... source_audio_T-1

destination:
    0 ... audio_context_t-1
```

The operation must be direct F32 memory copying.

There is no:

```text
latent -> VAE decode -> pixels/audio -> VAE encode -> latent
```

round trip. This is one of the main advantages of current generated-latent continuation designs.

---

# 11. Initial target construction

Generate the complete normal noise tensors exactly as a standard run would:

```text
video_noise
audio_noise
```

Do not alter normal RNG ordering.

Then construct the starting target.

## Generated suffix

The suffix remains the normal initial noise.

## Preserved video prefix

Follow H3's existing near-clean visual conditioning convention:

```text
video_prefix =
    0.999 * previous_clean_video_tail
  + 0.001 * corresponding_new_run_video_noise
```

Current H3 conditioning uses the visual condition timestep `0.999`, and current native masked-inpaint implementations inject preserved video at this same strength.

## Preserved audio prefix

Use the previous clean audio latent directly.

The H3 audio condition timestep is:

```text
1.0
```

so no corresponding 0.001 augmentation is necessary.

The complete target therefore becomes:

```text
video:
[near-clean inherited prefix][random suffix]

audio:
[clean inherited prefix][random suffix]
```

---

# 12. Mask representation

Do not initially implement arbitrary 5-D floating-point masks.

The continuation use case needs only two binary temporal prefix masks.

Represent them compactly as:

```c
typedef struct {
    int video_prefix_t;
    int audio_prefix_t;
} h3_denoise_prefix;
```

Semantics:

```text
0 = preserve
1 = generate
```

Conceptually:

```text
video mask:
[000000000000][111111111111111111...]

audio mask:
[000000....000][111111111111111111...]
```

The video and audio prefix lengths remain separate internally even when they represent the same 1.625 seconds.

This deliberately leaves room for future independent audio-tail handling.

---

# 13. Per-row H3 timestep semantics

This is the most important DiT modification.

It is **not sufficient** to leave every target row at the ordinary generation timestep and merely zero its Euler update.

Preserved rows must tell the transformer that they contain nearly clean/clean information.

For each denoising step:

```text
sigma_v = current video sigma
sigma_a = current audio sigma

t_v = 1 - sigma_v
t_a = 1 - sigma_a
```

For generated rows:

```text
video generated row timestep = t_v
audio generated row timestep = t_a
```

For preserved rows:

```text
video preserved timestep =
    max(t_v, 0.999)

audio preserved timestep =
    1.0
```

This corresponds to the current native H3 masking rule where a mask value `m` gives a row an effective sigma:

```text
sigma_row = m * sigma_stream
```

and mask `0` therefore makes it a condition-strength row.

---

# 14. Efficient binary timestep implementation

Do not initially port ComfyUI's completely general arbitrary-mask mechanism.

For continuation only four target modulation classes are needed:

```text
VIDEO_GENERATED
VIDEO_PRESERVED
AUDIO_GENERATED
AUDIO_PRESERVED
```

Existing text/reference/condition timestep classes remain unchanged.

At each DiT evaluation calculate timestep embeddings/AdaLN vectors for the distinct classes and assign each packed target row to the appropriate modulation class.

For target video rows:

```text
temporal step < video_prefix_t
    -> VIDEO_PRESERVED

otherwise
    -> VIDEO_GENERATED
```

For target audio packed rows remember that audio is represented as two stereo timelines.

For each stereo plane:

```text
time < audio_prefix_t
    -> AUDIO_PRESERVED

otherwise
    -> AUDIO_GENERATED
```

Do not assume that all preserved audio rows form one contiguous range unless the existing pack order confirms it.

The current `[32,2,T] -> packed rows` conversion should be treated as the authoritative row order.

---

# 15. AdaLN/modulation changes

Current optimized `h3cli` aggressively precomputes and fuses timestep-dependent AdaLN work.

The continuation feature must preserve those optimizations for the normal path.

Recommended implementation:

```text
if no continuation mask:
    execute existing exact path

else:
    use per-target-row modulation-class map
```

The map contains small integer indices, not complete modulation vectors.

Example:

```text
row modulation index:

0 = ordinary video
1 = ordinary audio
2 = preserved video
3 = preserved audio
4... = existing reference/condition classes as required
```

GPU kernels performing AdaLN/gating/final AdaLN must accept either:

```text
single modulation row for a segment
```

or:

```text
per-token modulation-row index
```

Only target video/audio rows require the new mixed behavior.

Reference rows remain unchanged.

---

# 16. Velocity masking

After the DiT predicts target velocity:

```text
video_velocity
audio_velocity
```

force the protected portion to zero.

Conceptually:

```text
video_velocity[:, 0:video_prefix_t, :, :] = 0

audio_velocity[:, :, 0:audio_prefix_t] = 0
```

Current native H3 mask behavior similarly multiplies predicted velocity by the denoise mask.

Consequently, every Euler transition satisfies:

```text
preserved_prefix_next =
    preserved_prefix_current
```

while:

```text
generated_suffix_next =
    normal H3 Euler update
```

The prefix therefore remains constant throughout sampling after its initial video 0.999 augmentation.

---

# 17. CPU sampler integration

Implement and validate continuation first against the simplest exact Euler path:

```text
--reuse 1
--core-reuse 1
token reduction disabled
full 50 layers
CPU sampler
```

This becomes the correctness oracle.

The existing:

```c
h3_dit_denoise_euler(...)
```

API should either accept an optional prefix mask or delegate to a new internal function such as:

```c
h3_dit_denoise_euler_masked(...)
```

The unmasked function must retain identical behavior.

---

# 18. GPU sampler integration

Clean current `h3cli` can keep sampler state in Metal buffers on supported M5 systems, while older systems can use the CPU sampler.

The GPU-state sampler therefore needs the same semantics:

* preserved-prefix initialization;
* per-row timestep classes;
* zero video prefix velocity;
* zero audio prefix velocity;
* no accidental prefix update during reused/skipped velocity transitions.

CPU and GPU continuation runs using equivalent exact settings should produce matching or existing-tolerance-equivalent final latents.

Until this is implemented, continuation should explicitly force the CPU sampler rather than silently running an incorrect GPU path.

---

# 19. Whole-denoiser reuse and core reuse

After the exact masked path works, validate:

```text
--reuse 2
--reuse 3

--core-reuse 4
--core-reuse 6
```

For whole-velocity reuse, protected rows must remain zero even when velocities are extrapolated.

For core reuse, timestep-dependent heads must still distinguish generated and preserved rows on every step.

Do not assume compatibility simply because an unmasked render works.

Unsupported optimization combinations should produce a clear error until validated.

---

# 20. Token reduction

Current token reduction operates on target video tokens and changes the internal execution topology.

Initially disable it when continuation is active.

Later validate that:

* horizontal token pairs never combine incompatible mask classes;
* preserved and generated temporal rows retain correct timestep classes;
* expansion/restoration does not modify protected rows;
* the final protected prefix remains invariant.

Because the continuation mask is temporal and uniform spatially, all spatial tokens within one temporal step should have the same mask class, making eventual compatibility straightforward.

---

# 21. Ref2VA coexistence

Continuation must be independent of checkpoint selection.

Existing logic remains conceptually:

```text
if reference_count > 0:
    use Ref2VA checkpoint
else:
    use FL2VA/T2VA checkpoint
```

Continuation does not itself cause `ref2va = true`.

Therefore these are valid:

```text
T2VA + continuation
Ref2VA images + continuation
Ref2VA video + continuation
Ref2VA image + audio + continuation
Ref2VA video/audio + continuation
```

The continuation prefix is inserted into the target AV latent **after** the target shape is known.

Existing reference conditioning remains in the packed reference/condition regions.

Current H3 continuation implementations similarly allow normal Ref2VA references to coexist with latent-masked continuation because the two mechanisms perform different roles.

---

# 22. Ref2VA reference policy across segments

References are not stored in `.h3av`.

Every generation receives its own reference list.

Example chain:

```text
segment 1:
    face1
    body1

segment 2:
    face1
    body1

segment 3:
    face2
    body2

segment 4:
    face2
    body2
    environment reference
```

The latent context preserves immediate physical continuity.

The current segment's Ref2VA references influence the future generated suffix.

This separation intentionally allows identity/object/reference changes without sacrificing the continuation boundary.

Strongly document that users generally should **not** also supply the previous rendered segment as a Ref2VA video reference merely to obtain continuity. Doing so recreates the recursive reference-feedback path that native latent continuation is intended to avoid.

---

# 23. First/last-frame conditioning

For the first implementation:

```text
continuation + Ref2VA references: supported
continuation + no references: supported
continuation + --first-frame/--last-frame: unsupported
```

Reject the latter combination explicitly.

This avoids mixing three different temporal constraints before masked continuation itself is validated.

A later phase can investigate adding a future `--last-frame` quality-reset anchor, which may be useful for very long chains.

---

# 24. Conditioning and prepared-model cache

The continuation state contents must **not** become part of the prompt/reference conditioning cache key.

The actual copied latent changes every segment but does not alter the Ref2VA text/reference encoding.

However the prepared DiT key must include the **continuation mask geometry**, such as:

```text
video_prefix_t
audio_prefix_t
```

because it changes target-row modulation assignments.

Thus:

```text
same prompt
same refs
same target geometry
same context length
```

may reuse prepared structures even when the actual prefix latent is different.

Changing from 39 to 90 frames must invalidate any prepared mask-layout state.

---

# 25. Decoding and delivery

Internally decode the entire final AV latent:

```text
[preserved duplicate context][new generated suffix]
```

For a normal continuation output, remove the protected prefix.

For video:

```text
delivered_video = decoded_video[continuation_context_frames:]
delivered_frame_count = decoded_frame_count - continuation_context_frames
```

with output frame numbers rebased to zero.

For audio:

```text
trim_samples =
    audio_context_ticks * 800
```

because the H3 AudioVAE uses 32 kHz PCM and an 800-sample hop.

For 39 frames:

```text
65 audio ticks * 800
    = 52,000 samples
    = 1.625 seconds
```

which exactly matches:

```text
39 / 24 = 1.625 seconds
```

The exact shared AV boundary is therefore valuable not only for sampling but also for lossless timing during assembly.

---

# 26. Debug output mode

`--keep-continuation-prefix` should disable trimming.

This produces:

```text
segment-N-debug.mp4 =
    duplicated protected prefix
    +
    generated suffix
```

This mode is essential for development because it lets tests inspect whether the decoded copied prefix actually corresponds to the source segment tail.

It should not be the normal long-video output mode.

---

# 27. State used for the next segment

The `.h3av` state saved after segment N must contain the **complete final raw target latent before prefix trimming**.

The saved target therefore contains:

```text
old inherited prefix
+
newly generated suffix
```

When segment N+1 starts, it extracts its continuation context from the **tail**.

Provided the current segment produced enough new content, this tail is completely new content and no old inherited prefix propagates indefinitely.

---

# 28. Failure handling

Reject continuation when:

```text
state file is corrupt;
state version is unsupported;
geometry differs;
VAE/model signature is incompatible;
requested context is not 39 + 51k;
context is longer than source state;
context consumes the entire target;
audio dimensions are inconsistent;
video dimensions are inconsistent;
unsupported sampling optimization is enabled;
continuation is combined with first/last frame anchors;
```

Error messages should include expected and actual values.

Do not silently resize latent state.

Do not decode/re-encode an incompatible state as a fallback.

---

# 29. Required observability

When continuation is enabled, log:

```text
Continuation source: segment01.h3av
Target raw frames: 362
Protected context: 39 frames / 12 video latent steps
Protected audio: 65 ticks / 1.625 s
Net output: 323 frames / 13.458 s
Mode: Ref2VA
References: 2 images, 0 videos, 0 audio
Sampler mask: video hard prefix / audio hard prefix
Output prefix trimming: enabled
Saved AV state: segment02.h3av
```

`--profile` should additionally report:

```text
state load time
state save time
prefix copy time
continuation state bytes
```

---

# 30. Acceptance criteria

The implementation is complete when all of the following are true:

1. Ordinary T2VA/FL2VA/Ref2VA runs without continuation retain their existing behavior and preferably byte-identical final latents.
2. A generated `.h3av` file loads losslessly.
3. The tail copied into the next segment matches the source latent exactly before video augmentation.
4. Video protected rows receive the 0.999 condition-strength timestep.
5. Audio protected rows receive the 1.0 timestep.
6. Generated rows receive the ordinary current video/audio timestep.
7. Video protected rows do not change during denoising after initialization.
8. Audio protected rows remain bit-identical through denoising.
9. The decoded protected prefix closely reproduces the previous source tail.
10. Prefix trimming produces exactly synchronized 24 fps video and 32 kHz audio.
11. Three or more continuation segments can be chained without using rendered MP4s as continuation input.
12. Ref2VA references work on every continuation segment.
13. Reference sets can be changed between segments.
14. A multi-segment latent-continuation chain shows materially less cumulative color/contrast drift than recursive Ref2VA-video continuation.
