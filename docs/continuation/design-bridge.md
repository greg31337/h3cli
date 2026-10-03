# Technical Design: Bridge Continuation for Same-Scene Action Changes

## 1. Objective

Add an optional **bridge continuation** mode on top of the existing native masked AV latent continuation implementation.

The feature is specifically intended for:

```text
same scene + same character + same general camera/environment
+
meaningful change in action, pose, gaze, body movement, or camera motion
```

Examples:

```text
segment 1:
    woman walks through the forest

segment 2:
    woman finishes the current step, turns toward the camera,
    raises her arms, and begins running
```

or:

```text
segment 1:
    man is sitting and talking

segment 2:
    man stands up and walks toward the window
```

The existing continuation implementation provides an exactly preserved latent prefix followed immediately by fully generated target rows:

```text
[preserved prefix][fully generated suffix]

mask:
0 0 0 0 0 0 0 0 0 0 0 0 | 1 1 1 1 1 ...
```

This produces excellent physical continuity at the boundary but gives the model no intermediate temporal region in which to reconcile the previous scene state with a meaningfully changed prompt.

Bridge mode introduces a small **partially denoised adaptation region** inside the inherited continuation context:

```text
[adaptable inherited context][exact inherited endpoint][generated suffix]

mask:
0.50 0.50 0.40 0.30 0.20 0.10 0.00 0.00 | 1.00 1.00 1.00 ...
```

The bridge region remains initialized from the previous segment's latent state, but is allowed to move progressively under the new prompt.

The final inherited rows nearest the seam remain exactly preserved.

The current hard-prefix continuation remains the default and must retain identical behavior.

---

# 2. Modes

Introduce:

```text
continuation_mode = hard
continuation_mode = bridge
```

Default:

```text
continuation_mode = hard
```

`hard` executes the existing implementation without modification.

`bridge` enables fractional temporal denoising masks for part of the inherited prefix.

Conceptually:

```text
hard:

previous segment
      |
      v
[000000000000][111111111111111111...]

bridge:

previous segment
      |
      v
[555443221000][111111111111111111...]
```

where the digits represent approximate mask strengths.

The first implementation should support only bridge masks that are:

* temporal;
* spatially uniform;
* deterministic;
* monotonic toward the seam;
* shared conceptually between video and audio by time;
* represented internally using a small finite set of mask classes.

Arbitrary per-pixel or per-token masks are outside scope.

---

# 3. Bridge timeline semantics

The existing continuation prefix is retained.

For a 39-frame continuation:

```text
39 source frames
=
12 video latent temporal steps
=
65 audio latent ticks
```

Bridge mode divides this continuation context into:

```text
bridge region
+
exact preservation region
```

Example:

```text
video latent rows:

0  1  2  3  4  5  6  7 | 8  9 10 11 | generated...
<---- bridge region ----> < exact --->

audio:

corresponding time-aligned bridge ticks
+
corresponding exact ticks
```

The exact region nearest the new output boundary is important.

It ensures that the state immediately preceding the first delivered generated frame remains identical to the previous segment.

The bridge region exists earlier in time and is normally trimmed from the delivered continuation output together with the rest of the duplicated context.

---

# 4. Initial recommended policy

For the first implementation, continue using:

```text
continuation_context_frames = 39
```

Recommended default bridge configuration:

```text
bridge_video_steps = 8
exact_video_steps = 4
```

which divides the 12 video latent steps approximately as:

```text
8 adaptable steps
4 exactly preserved steps
```

The precise defaults should remain configurable because testing may show that 6+6, 8+4, 9+3, or another split works better.

A reasonable first mask profile is:

```text
video latent row:

0     1     2     3     4     5     6     7     8  9 10 11

mask:

0.50  0.50  0.40  0.40  0.30  0.20  0.10  0.05  0  0  0  0
```

The generated suffix remains:

```text
1.0
```

These are initial values rather than model constants.

The design must allow profiles to be changed without changing the sampler implementation.

---

# 5. Mask convention

Continue using the existing H3 continuation convention:

```text
mask = 0
    preserve

mask = 1
    normal generation

0 < mask < 1
    partial bridge denoising
```

For a bridge row:

```text
effective_sigma =
    mask * current_stream_sigma
```

and:

```text
effective_timestep =
    1 - effective_sigma
```

subject to H3's existing visual conditioning floor.

For video:

```text
effective_timestep =
    max(
        1 - mask * sigma_video,
        0.999 when required by the visual condition floor
    )
```

The actual implementation should preserve the exact conventions already used by the current masked H3 path rather than introducing an independent sigma interpretation.

For audio:

```text
effective_sigma =
    mask * sigma_audio

effective_timestep =
    1 - effective_sigma
```

with:

```text
mask = 0
    -> timestep 1.0
```

---

# 6. Velocity scaling

The current hard mask performs:

```text
mask = 0
    velocity = 0

mask = 1
    velocity = predicted velocity
```

Bridge mode generalizes this to:

```text
velocity =
    predicted_velocity * mask
```

for bridge rows.

Thus:

```text
mask 0.00:
    row cannot move

mask 0.10:
    row moves only slightly

mask 0.50:
    row adapts substantially

mask 1.00:
    normal generation
```

This operation must occur for every denoising step, including reused or extrapolated velocity paths.

---

# 7. Bridge latent initialization

Bridge rows must not simply contain a perfectly clean previous latent while claiming to have a nonzero effective sigma.

For each bridge row, the target latent should contain a scheduler-consistent mixture of:

```text
previous clean latent
+
the corresponding noise from the new target initialization
```

The implementation should use the same flow/noise interpolation convention as normal H3 target initialization.

Conceptually:

```text
sigma_bridge =
    mask * sigma_initial
```

and:

```text
bridge_latent =
    flow_mix(
        previous_clean_latent,
        corresponding_new_target_noise,
        sigma_bridge
    )
```

For exact video rows:

```text
existing hard-continuation initialization remains unchanged
```

including the existing approximately:

```text
0.999 * clean_latent
+
0.001 * corresponding_noise
```

behavior.

For exact audio rows:

```text
previous clean audio latent
```

continues to be copied directly.

The bridge implementation must reuse the noise tensor already generated for the target.

It must not generate additional random values or alter normal RNG ordering.

---

# 8. Critical distinction: initialization sigma versus sampling sigma

The bridge mask is constant in semantic strength, but the actual effective sigma changes during the denoising schedule.

For bridge row mask `m` at denoising step `s`:

```text
effective_sigma(s) =
    m * sigma_stream(s)
```

The row therefore approaches the clean state along with the generated suffix, but at a reduced noise/velocity level.

The initial bridge latent should correspond to the first effective sigma used by the sampler.

Do not repeatedly inject fresh noise during later denoising steps.

Noise is introduced once during target construction.

Subsequent steps evolve the row using the bridge timestep and scaled velocity.

---

# 9. Quantized modulation classes

The current implementation already optimizes timestep-dependent AdaLN/modulation work using a small set of modulation classes.

Bridge mode should preserve this approach.

Do not initially support arbitrary floating-point mask values per target row.

Define a small configurable set such as:

```text
BRIDGE_000 = 0.00
BRIDGE_005 = 0.05
BRIDGE_010 = 0.10
BRIDGE_020 = 0.20
BRIDGE_030 = 0.30
BRIDGE_040 = 0.40
BRIDGE_050 = 0.50
GENERATED  = 1.00
```

Alternatively a smaller first implementation may use:

```text
0.00
0.125
0.25
0.375
0.50
1.00
```

Each temporal target row stores only:

```text
uint8_t modulation_class
```

The sampler computes timestep/AdaLN values once per distinct class per denoising step.

This retains most of the current optimization architecture.

---

# 10. Bridge profile representation

Introduce a bridge profile independent of target dimensions.

Conceptually:

```c
typedef struct {
    int video_bridge_t;
    int video_exact_t;

    int audio_bridge_t;
    int audio_exact_t;

    int class_count;
    float class_mask[H3_MAX_BRIDGE_CLASSES];

    uint8_t *video_classes;
    uint8_t *audio_classes;
} h3_bridge_profile;
```

A simpler first implementation may generate the profile algorithmically and avoid dynamic arrays.

The profile should be generated after:

```text
continuation context
video latent length
audio latent length
```

are known.

The bridge strength should be specified as a function of normalized time rather than independently hand-authoring unrelated video and audio arrays.

For example:

```text
bridge strength f(t)
```

where:

```text
t = 0
    oldest bridge point

t = 1
    end of bridge / start of exact region
```

Video and audio classes are then derived from the same time-domain function.

This prevents AV bridge behavior from diverging.

---

# 11. Audio/video temporal alignment

Video and audio bridge regions must represent the same durations.

Do not simply assign:

```text
8 video latent rows
8 audio latent rows
```

because their temporal rates differ.

Instead define the bridge in time.

For example:

```text
bridge_duration_frames = 26
exact_duration_frames = 13
```

or equivalent internal rational durations.

Map that interval onto:

```text
video latent temporal rows
audio latent ticks
```

using the same continuation temporal mapping already used by h3cli.

The resulting video and audio class boundaries may not map to exactly identical discrete timestamps.

Use deterministic nearest-boundary rules.

The exact end of the complete continuation prefix must continue to align at the existing shared AV boundary.

---

# 12. CLI

Add:

```text
--continue-mode hard|bridge
```

Default:

```text
--continue-mode hard
```

Add optional bridge tuning parameters:

```text
--continue-bridge-steps N
--continue-bridge-max-strength X
--continue-bridge-profile NAME
```

Recommended initial semantics:

```text
--continue-bridge-steps N
```

specifies the number of video latent temporal rows assigned to the bridge.

The remaining protected prefix rows are exact.

Example:

```bash
./bin/h3cli \
  -d ./models/MiniMax-H3 \
  -p "The woman completes her walking step, turns toward the camera, and begins running." \
  --frames 362 \
  --continue-from segment01.h3av \
  --continue-context 39 \
  --continue-mode bridge \
  --continue-bridge-steps 8 \
  --save-av-state segment02.h3av \
  -o segment02.mp4
```

The first implementation should expose only a small number of tuning parameters.

Low-level class values may remain internal or available only through development/debug options until stable defaults are determined.

---

# 13. API

Extend `h3_params` conceptually with:

```c
typedef enum {
    H3_CONTINUE_HARD = 0,
    H3_CONTINUE_BRIDGE = 1
} h3_continuation_mode;
```

and:

```c
h3_continuation_mode continuation_mode;

int bridge_video_steps;

float bridge_max_strength;

h3_bridge_profile_type bridge_profile;
```

Default:

```text
continuation_mode = H3_CONTINUE_HARD
```

The default values of the remaining bridge fields must therefore have no effect on existing generation.

---

# 14. Initial bridge profiles

Support a very small profile set initially.

### `linear`

Example:

```text
0.50 0.43 0.36 0.29 0.21 0.14 0.07 0.00
```

### `ease-out`

Keeps more adaptability in the older portion and rapidly converges toward the exact endpoint.

Example:

```text
0.50 0.49 0.46 0.40 0.31 0.20 0.08 0.00
```

### `stepped`

Useful for the optimized modulation-class implementation.

Example:

```text
0.50 0.50 0.40 0.40 0.30 0.20 0.10 0.00
```

Recommended initial default:

```text
stepped
```

because it directly maps to a small number of timestep modulation classes.

---

# 15. Preserve an exact seam region

Bridge mode must always retain at least one exact video latent temporal row near the seam.

Recommended validation:

```text
bridge_video_steps
<
video_context_t
```

Prefer:

```text
exact_video_steps >= 2
```

and warn when:

```text
exact_video_steps < 2
```

The default for a 39-frame context should retain approximately the final four video latent rows exactly.

The purpose of the bridge is not to regenerate the seam itself.

It is to allow the incoming trajectory to adapt *before* the immutable endpoint.

---

# 16. Target row classification

Bridge mode introduces these logical classes:

```text
VIDEO_EXACT
VIDEO_BRIDGE_<strength>
VIDEO_GENERATED

AUDIO_EXACT
AUDIO_BRIDGE_<strength>
AUDIO_GENERATED
```

Reference, text and other condition classes remain unchanged.

For each video target temporal position:

```text
inside bridge interval:
    corresponding bridge class

inside exact continuation interval:
    exact class

otherwise:
    generated class
```

For audio, perform the same classification according to time rather than assuming the same row counts.

---

# 17. Packed-row integration

The existing packed target row mapping remains authoritative.

Do not assume video or audio rows are globally contiguous beyond what the current packer guarantees.

When creating modulation indices:

```text
video target row
    -> determine video temporal index
    -> resolve bridge class

audio target row
    -> determine stereo plane and audio time index
    -> resolve bridge class
```

Both stereo audio rows for the same time tick must receive the same bridge mask strength.

---

# 18. CPU sampler integration

Implement bridge support first in the existing exact CPU correctness path:

```text
--reuse 1
--core-reuse 1
token reduction disabled
```

The existing hard mode must execute the current sampler path.

Conceptually:

```c
if (continuation_mode == H3_CONTINUE_HARD) {
    current_exact_existing_code();
} else {
    bridge_sampler_path();
}
```

Refactoring common code is acceptable, but regression testing must demonstrate that hard mode is unchanged.

---

# 19. GPU sampler integration

Once CPU behavior is validated, apply identical bridge semantics to the GPU-state sampler:

* bridge initialization;
* bridge timestep classes;
* bridge velocity scaling;
* exact-prefix zero velocity;
* generated suffix normal velocity.

Equivalent exact CPU/GPU runs should produce results within the same tolerances currently accepted by h3cli.

Until validated, bridge mode should force the known-correct CPU sampler rather than execute an incomplete GPU implementation.

---

# 20. Whole-denoiser reuse

Whole-velocity reuse must preserve the mask operation.

Whenever a predicted or reused velocity is applied:

```text
velocity[row] *= mask[row]
```

must still occur.

A reused bridge velocity must therefore never accidentally become a full-strength generated velocity.

The exact region must continue to have:

```text
velocity = 0
```

even when the velocity tensor originates from reuse/extrapolation.

---

# 21. Core reuse

Core reuse may reuse transformer blocks, but timestep-dependent modulation/head processing must continue to distinguish all bridge classes.

Bridge mode should initially reject unvalidated core-reuse configurations.

After correctness testing, enable combinations incrementally.

---

# 22. Token reduction

Continue the existing policy of disabling token reduction for continuation until explicitly validated.

When bridge mode is eventually supported with token reduction:

* all tokens belonging to one video temporal row must have the same bridge class;
* reduced token groups must never combine rows with different bridge classes;
* expansion must preserve the original class assignments;
* exact rows must remain invariant.

The temporal and spatially uniform nature of bridge masks should make this tractable.

---

# 23. Ref2VA compatibility

Bridge continuation remains orthogonal to Ref2VA.

Supported combinations:

```text
T2VA + bridge continuation

Ref2VA image + bridge continuation

Ref2VA image/image + bridge continuation

Ref2VA video + bridge continuation

Ref2VA image/audio + bridge continuation
```

The previous latent tail remains target continuation state.

It must not become a Ref2VA reference.

Changing Ref2VA references between segments remains allowed.

For the intended use case, testing should initially focus on keeping references constant while changing only the action portion of the prompt.

---

# 24. Prompt behavior

Bridge mode should not attempt to parse or modify prompts internally in its first implementation.

However documentation should recommend transition-oriented prompt phrasing.

Prefer:

```text
The woman completes the current walking step, gradually turns toward the
camera, raises her arms, and begins running.
```

over:

```text
The woman is running toward the camera with her arms raised.
```

The first wording describes a transition from the inherited physical state.

The second describes a replacement state and therefore creates stronger conditioning conflict.

Bridge continuation is intended to make such prompt changes more robust, but it cannot guarantee continuity under mutually incompatible scene instructions.

---

# 25. State format

No `.h3av` format change is required for the basic feature.

The saved state should continue to contain the complete clean final AV latent.

Bridge state is a property of the *next generation request*, not of the previous generated latent.

Optional metadata may record:

```text
continuation mode used
bridge profile
bridge steps
bridge maximum strength
```

for debugging and reproducibility, but these fields must not be required to load or continue from the state.

---

# 26. Prefix trimming

Bridge mode retains the existing output behavior.

The complete decoded target remains:

```text
[duplicate/adapted continuation context][new suffix]
```

The complete continuation prefix is trimmed from normal delivered output.

Therefore the bridge region is normally invisible.

`--keep-continuation-prefix` should remain supported and becomes particularly useful because it allows visual inspection of:

```text
original source tail
vs.
bridge-adapted decoded tail
vs.
exact endpoint
```

The exact endpoint should still closely reproduce the source tail immediately before the generated suffix begins.

---

# 27. Debug output

When bridge mode is enabled, log:

```text
Continuation mode: bridge

Context:
    39 frames
    12 video latent steps
    65 audio ticks

Bridge:
    8 video latent steps

Exact:
    4 video latent steps

Bridge profile:
    stepped

Bridge max strength:
    0.50

Video mask:
    0.50 0.50 0.40 0.40 0.30 0.20 0.10 0.00 0 0 0 0

Audio bridge:
    <corresponding tick interval>

Output prefix trimming:
    enabled
```

Debug/profile logging should also report the number of target rows assigned to each modulation class.

---

# 28. Validation metrics

Bridge quality should be evaluated primarily over the first 1–2 seconds of newly delivered video, rather than only at the single splice frame.

Test cases should include:

### A. No prompt change

```text
woman continues walking through the forest
->
woman continues walking through the forest
```

Expected:

bridge should not materially degrade hard continuation.

### B. Small action change

```text
walking
->
walking while looking upward
```

### C. Meaningful action change

```text
walking
->
turning toward camera and beginning to run
```

### D. Pose transition

```text
arms down
->
raising both arms
```

### E. Motion-direction change

```text
walking forward
->
slowing, turning left, then walking left
```

Primary comparison:

```text
hard-39
vs.
bridge-39
```

Secondary comparison:

```text
hard-90
vs.
bridge-39
```

Useful objective measurements include:

* first-new-frame latent difference;
* optical-flow direction and magnitude;
* temporal acceleration discontinuity;
* background feature displacement;
* luminance and contrast changes;
* character bounding-box motion;
* pose/keypoint velocity when diagnostic tooling is available.

Subjective inspection remains important because semantic "restart" behavior may not correlate strongly with simple pixel metrics.

---

# 29. Recommended initial parameter sweep

Use a fixed source `.h3av`, prompt pair, references, and generation seed.

Test:

```text
bridge steps:

4
6
8
9
10
```

with:

```text
max strength:

0.25
0.40
0.50
0.65
```

and profiles:

```text
linear
ease-out
stepped
```

The most important parameter is likely the combination:

```text
bridge duration
+
maximum denoising strength
```

Too little bridge strength will behave like hard continuation.

Too much bridge strength may allow the incoming scene or character to drift before the seam.

The desired operating point should preserve:

```text
scene identity
background geometry
character identity
camera continuity
```

while allowing:

```text
pose trajectory
action trajectory
gaze
body movement
camera motion
```

to change smoothly.

---

# 30. Acceptance criteria

Bridge continuation is complete when:

1. `hard` remains the default continuation mode.

2. Existing hard-continuation runs retain the same behavior and preferably byte-identical final latents.

3. Bridge mode supports the existing 39-frame continuation context.

4. Bridge temporal rows are initialized from the previous clean latent plus the existing target noise at the intended effective noise strength.

5. No additional RNG draws are introduced.

6. Exact continuation rows retain the existing hard-continuation initialization.

7. Bridge rows receive the correct fractional effective timestep on every denoising step.

8. Bridge-row velocity is multiplied by the configured bridge mask.

9. Exact rows retain zero velocity.

10. Generated rows retain normal full-strength denoising.

11. Video and audio bridge intervals are time-aligned.

12. Both audio stereo timelines receive identical temporal bridge classification.

13. At least one video latent temporal row remains fully exact before the generated suffix.

14. Prefix trimming remains unchanged.

15. `.h3av` files produced by existing hard continuation remain usable with bridge continuation.

16. Ref2VA references remain independent from continuation state.

17. Three or more bridge continuation segments can be chained.

18. For unchanged prompts, bridge continuation does not introduce materially worse scene/background/identity stability than hard continuation.

19. For meaningful same-scene action changes, bridge continuation shows materially fewer pose, camera, or background discontinuities during the first 1–2 seconds of new output than hard continuation.

20. CPU and supported GPU bridge implementations produce equivalent results within existing h3cli tolerances.
