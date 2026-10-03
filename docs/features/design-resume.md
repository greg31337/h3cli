# Technical Design: Stop-After and Resumable H3 Sampling

## 1. Objective

Extend the existing `h3cli` implementation, including the already implemented native masked AV continuation functionality, with the ability to pause a generation after an arbitrary completed denoising step, serialize the complete generation state, terminate the process, and later resume from exactly that point.

The primary workflow is:

```text
full final geometry
full BF16 model
full final N-step schedule

        |
        v

Mac / Metal
steps 0 ... K-1
        |
        v
save .h3sample
        |
        +---- decode preview
        |
        v
process exits

later:

load .h3sample
        |
        v
reconstruct prepared Metal DiT state
        |
        v
resume at step K
        |
        v
steps K ... N-1
        |
        v
final AV latent
        |
        +---- existing .h3av continuation state
        |
        +---- video/audio VAE
                  |
                  v
                 MP4
```

The feature must preserve the existing behavior when neither stopping nor resuming is requested.

The initial implementation targets the existing Metal backend only. No CUDA serialization or resume implementation is required yet.

The serialized format should nevertheless avoid unnecessary Metal-specific assumptions so it can later become the interchange format between Metal and CUDA.

---

# 2. Core semantic requirement

`--stop-after-step K` means:

> Construct the complete requested N-step sampling schedule, run exactly the first K Euler transitions from that schedule, save the resulting state corresponding to sigma index K, and stop.

It must **not** mean:

```text
run a separate K-step H3 schedule
```

For example:

```text
--steps 20 --stop-after-step 4
```

means:

```text
20-step schedule:

sigma[0]
   |
 step 0
   v
sigma[1]
   |
 step 1
   v
sigma[2]
   |
 step 2
   v
sigma[3]
   |
 step 3
   v
sigma[4]  <-- checkpoint saved here

remaining:
steps 4 ... 19
```

A resumed process starts with:

```text
next_step = 4
```

and uses the exact originally serialized 20-step video and audio sigma grids.

This is fundamentally different from:

```text
--steps 4
```

because current H3 uses a separately constructed low-step schedule when a four-step generation is requested. Upstream explicitly defines `--steps N` as N actual passes and uses different low-budget scheduling behavior for very small N.

---

# 3. CLI

Add:

```text
--stop-after-step N
--save-sampler-state PATH
--resume-sampler-state PATH
--preview-on-stop
```

Recommended examples:

```bash
./bin/h3cli \
  -d ./models/MiniMax-H3 \
  -p "$PROMPT" \
  --width 768 \
  --height 1024 \
  --frames 141 \
  --steps 20 \
  --layers 50 \
  --reuse 1 \
  --ref-image person.jpg \
  --stop-after-step 4 \
  --save-sampler-state shot17.h3sample \
  --preview-on-stop \
  -o shot17-preview.mp4
```

Resume:

```bash
./bin/h3cli \
  -d ./models/MiniMax-H3 \
  --resume-sampler-state shot17.h3sample \
  -o shot17-final.mp4
```

Resume and pause again:

```bash
./bin/h3cli \
  -d ./models/MiniMax-H3 \
  --resume-sampler-state shot17.h3sample \
  --stop-after-step 8 \
  --save-sampler-state shot17-step8.h3sample
```

`--stop-after-step` is always an **absolute completed-step number in the original schedule**, not a number relative to the resume point.

Therefore a state saved after step 4 and resumed with:

```text
--stop-after-step 8
```

runs four additional transitions.

---

# 4. Step numbering

Internally maintain:

```c
int total_steps;
int next_step;
```

with:

```text
new generation:
    next_step = 0

after completing Euler transition 0:
    next_step = 1

after completing transition 3:
    next_step = 4

completed generation:
    next_step = total_steps
```

This avoids the ambiguity of storing "current step".

The checkpoint is always taken at a stable boundary:

```text
AFTER the previous Euler update has completed
BEFORE the next DiT evaluation begins
```

Never serialize halfway through:

```text
attention
transformer block
DiT forward
Euler update
Metal command buffer
```

This gives every checkpoint a clean deterministic restart point.

---

# 5. New sampler-state abstraction

Refactor the Euler sampler so loop-local persistent data becomes an explicit structure.

Conceptually:

```c
typedef struct h3_sampler_state h3_sampler_state;
```

The structure should own all mutable information required to continue sampling.

Conceptually:

```text
total_steps
next_step

current video latent
current audio latent

video sigma schedule
audio sigma schedule

reuse schedule

last evaluated step
previous evaluated step

last video velocity
previous video velocity
last audio velocity
previous audio velocity

core-reuse state, when applicable

continuation mask state

generation configuration
```

The existing sampler becomes a wrapper around:

```c
h3_dit_denoise_euler_range(...)
```

rather than containing the complete sampler state in local variables.

This refactor is important even before serialization because current CPU Euler reuse keeps `last_evaluated`, `previous_evaluated` and the velocity history as local variables, while the GPU-state sampler holds BF16 velocity buffers inside `h3_dit`.

---

# 6. Proposed sampler API

Conceptually:

```c
h3_sampler_state *h3_sampler_state_create(...);

int h3_dit_denoise_euler_range(
    h3_dit *dit,
    h3_sampler_state *state,
    int stop_after_step,
    h3_dit_progress progress,
    void *progress_opaque,
    h3_dit_preview preview,
    void *preview_opaque,
    char *error,
    size_t error_size);

void h3_sampler_state_free(h3_sampler_state *state);
```

The range sampler starts at:

```text
state->next_step
```

and stops when either:

```text
next_step == stop_after_step
```

or:

```text
next_step == total_steps
```

The ordinary existing call becomes equivalent to:

```text
start = 0
stop = total_steps
```

and therefore preserves existing behavior.

---

# 7. Serialized file format

Use a separate versioned format:

```text
*.h3sample
```

rather than overloading `.h3av`.

`.h3av` remains:

```text
completed clean AV latent suitable for continuation
```

while `.h3sample` means:

```text
in-progress generation capable of resuming denoising
```

Use a sectioned binary container.

Header:

```text
magic                  "H3SAMPLE"
format_version
header_size
flags
section_count
file_size
global_checksum
```

Each section contains:

```text
section_type
section_version
dtype
element_count / byte_count
checksum
payload
```

Unknown optional sections must be skippable so later versions can add CUDA-specific or optimization-specific data.

Required sections must cause a clear compatibility error if unsupported.

---

# 8. Generation identity section

Save all generation-critical configuration rather than relying on CLI reconstruction.

Store:

```text
H3 mode:
    FL2VA / Ref2VA

render width
render height
output width
output height
requested frame count
aligned frame count

video latent T/H/W
audio latent T

steps
seed

active DiT layers
reuse interval
core-reuse interval
token-reduction configuration
SSD-streaming configuration
spatial RoPE scale
reference-RoPE setting

all numerical/precision compatibility flags
all slower/oracle implementation flags
all TensorOps/int8-related flags

sampler backend:
    CPU-state Metal
    GPU-state Metal

continuation enabled
continuation context frames
video prefix T
audio prefix T
continuation trim behavior
```

Do not depend only on the existing prepared-key string.

The prepared key may also be saved for diagnostics, but every important value should be represented explicitly.

---

# 9. Engine identity

Store:

```text
h3cli checkpoint-format version
h3cli build/version identifier
git commit hash when available

Metal backend state version

device architecture:
    Apple GPU family
    Metal version / Metal4 capability
    unified-memory mode
```

Also save a list of environment variables that can materially affect numerical execution, including any supported:

```text
H3_* sampler
H3_* DiT
H3_* token reduction
H3_* fusion
H3_* command-buffer
```

overrides.

The initial exact-resume guarantee is:

> same checkpoint, same H3 build, same model files, same Metal execution path and compatible Apple GPU.

A resume on a different compatible Metal machine may be allowed, but should initially be classified as a **compatible resume**, not promised byte-identical.

---

# 10. Model fingerprint

Save a content-derived model fingerprint.

It should distinguish at least:

```text
Ref2VA vs FL2VA

transformer
text encoder
vision encoder
video encoder
video VAE
audio VAE
tokenizer/configuration
```

A resume must not silently use a different model tree.

The initial implementation can build a manifest from:

```text
relative file path
file size
safetensors tensor schema
content fingerprint
```

and cache the expensive fingerprint calculation.

The state should also save the model path used originally as informational metadata, but compatibility must be based on model identity rather than pathname.

---

# 11. Prompt and source-reference metadata

Store the original prompt text verbatim.

For every reference save:

```text
kind
original path
audio path, when applicable
embedded-audio flag
reference order
source file size
source content hash
```

This information is for reproducibility and auditing.

The source files are **not required for resume** once the computed conditioning has been serialized.

Do not embed full source images/videos/audio in `.h3sample`.

---

# 12. Exact text conditioning

Serialize the final `h3_text_embedding`.

Current `h3cli` text embeddings contain:

```text
tokens
width
BF16 values
modality tags
```

and the current conditioning cache already keeps these exact values in memory.

Save:

```text
text.tokens
text.width
text.values as raw uint16 BF16 words
text.tags
```

Preserve the BF16 bit patterns exactly.

Do not convert them to F32 for serialization.

This lets resume skip:

```text
tokenization
Qwen token embedding
vision-language presentation construction
Qwen 50-layer encoding
```

and avoids introducing a second Metal numerical path into the resumed conditioning.

---

# 13. Optional tokenizer/presentation diagnostics

For maximum reproducibility, also save an optional diagnostic section containing:

```text
token IDs
multimodal presentation positions
vision-span boundaries
presentation modality tags
```

These are not needed for resume because the final text embedding is already saved.

They are useful for:

```text
debugging
future checkpoint migration
verifying prompt presentation
cross-backend validation
```

---

# 14. Exact Ref2VA/FL2VA condition rows

Serialize the already computed condition inputs consumed by the DiT.

Save:

```text
condition_video_rows F32
condition_video_element_count

condition_audio_rows F32
condition_audio_element_count

conditioned flag

h3_layout_ref array
reference_count
```

Current `h3cli` already caches these host-side values independently of the source media.

This means a resumed Ref2VA run does not need to rerun:

```text
image/video decoding
video encoder
audio encoder
vision encoder
reference patchification
condition augmentation
```

The resumed process consumes the exact original conditioning arrays.

---

# 15. Packed H3 layout

Serialize the complete `h3_layout`:

```text
seq_len

segments[]
    start
    stop
    kind

positions[]
    t
    h
    w

img_cond_rows
img_target_rows
audio_cond_rows
audio_target_rows

signature[5]
```

Current `h3_dit` copies this layout and derives target offsets and temporal dimensions from it.

On resume, do not reconstruct the layout merely from dimensions and references.

Use the exact saved layout and validate its internal consistency.

---

# 16. Exact sigma schedules

Save:

```text
schedule.steps
video_sigma[0 ... steps]
audio_sigma[0 ... steps]
```

as raw F32.

Even though the schedules are currently deterministic and inexpensive to reconstruct, they are part of the mathematical generation state and may change in future engine versions.

Resume should use the serialized arrays directly.

Current H3 uses distinct shifted video and audio schedules.

---

# 17. Exact current AV latent

This is the most important payload.

Save:

```text
video latent:
    F32 [24,T,H,W]

audio latent:
    F32 [32,2,T]
```

representing the state **after `next_step` completed Euler transitions**.

These must be serialized losslessly.

No:

```text
FP16 conversion
BF16 conversion
compression with loss
VAE decode/re-encode
```

is permitted.

For a stopped continuation generation, the video/audio latent already contains:

```text
protected inherited prefix
+
partially denoised suffix
```

so the original preceding `.h3av` is not needed for resume.

---

# 18. Initial noise

Also save the original complete:

```text
video noise tensor
audio noise tensor
```

when practical.

It is not required for ordinary resume because the current latent is authoritative.

However it is valuable for:

```text
debugging
cross-backend reproduction
restarting the shot from step 0
verifying continuation prefix augmentation
future schedule experiments
```

For continuation runs this must be the original new-run noise before the inherited prefix was inserted.

---

# 19. RNG state

Save:

```text
initial seed
exact RNG algorithm/version
exact current RNG state
number of normal values consumed
```

Once denoising has started, the present Euler path should not normally require additional random numbers.

Still save the state so future features do not make checkpoint behavior depend on an implicit assumption that RNG is finished forever.

---

# 20. Continuation metadata

When the existing masked AV continuation path is active, save:

```text
continuation source-state fingerprint
continuation context frames

video prefix T
audio prefix T

raw copied video-tail hash
raw copied audio-tail hash

video prefix initialization parameters:
    clean coefficient = 0.999
    noise coefficient = 0.001

audio-prefix preservation mode

video denoise mask geometry
audio denoise mask geometry

trim-prefix setting
```

Do not duplicate the entire source `.h3av`.

The current partially denoised target already contains the inherited state.

The source-state fingerprint exists so a diagnostic tool can establish provenance.

---

# 21. Preserve the exact continuation mask

Serialize the effective internal mask representation rather than merely storing:

```text
context_frames = 39
```

For the current implementation this will normally be:

```text
video_prefix_t = 12
audio_prefix_t = 65
```

but saving the resolved values protects against future changes to temporal conversion logic.

If per-row modulation maps have become part of the implementation, save their logical class assignment or enough data to recreate it exactly.

---

# 22. Whole-denoiser reuse state

For:

```text
--reuse > 1
```

an exact resume requires more than the current latent.

The CPU sampler currently tracks:

```text
last_evaluated
previous_evaluated

last_video_velocity
previous_video_velocity

last_audio_velocity
previous_audio_velocity
```

to extrapolate skipped velocities.

Move this state into `h3_sampler_state`.

Serialize all four velocity tensors losslessly as F32 for the CPU sampler.

Also serialize:

```text
selected[] reuse schedule
last_evaluated
previous_evaluated
```

so resuming immediately before a skipped evaluation produces the same extrapolated velocity as an uninterrupted process.

With:

```text
--reuse 1
```

these sections may be absent.

---

# 23. GPU-state Euler reuse

Current newer Metal execution can keep sampler latents and cached BF16 velocities in GPU buffers rather than round-tripping them to the host.

For checkpoints:

1. synchronize the Metal command stream at the completed-step boundary;
2. export the current packed latent to canonical F32 host layout;
3. export cached velocity tensors using their **native internal BF16 representation**;
4. record the last/previous evaluation indices;
5. serialize them.

On resume:

1. rebuild the DiT;
2. recreate GPU sampler buffers;
3. load current latent;
4. restore cached velocity BF16 words;
5. restore evaluation indices;
6. continue.

Do not recreate BF16 cached velocities by converting an F32 representation if exact same-path resume is desired.

---

# 24. Core-reuse state

When:

```text
--core-reuse > 1
```

the DiT contains mutable state beyond the Euler latent, including:

```text
core_forward_count
core_residual_ready
core_residual
```

as shown by the current `h3_dit` state.

To support exact stop/resume with core reuse, serialize:

```text
core_forward_count
core_residual_ready

core_residual dtype
core_residual shape
core_residual raw data

current token-reduction mode affecting its shape
```

Import the buffer before the first resumed DiT evaluation.

Until this is implemented and validated, reject checkpoint creation/resume with:

```text
core_reuse > 1
```

rather than silently restarting the residual history.

---

# 25. Token-reduction state

Most token-reduction topology is deterministic from:

```text
layout
step
configuration
```

and therefore does not need to be serialized as large mutable data.

However save all resolved token-reduction parameters:

```text
enabled
begin block
end block
early steps
early end
scale
```

plus the current effective reduced/full mode at the checkpoint.

Any mutable residual whose dimensions depend on the reduced sequence must be serialized under the core-reuse section.

---

# 26. Prepared DiT acceleration cache

The basic checkpoint must be sufficient to recreate the DiT from:

```text
model weights
serialized conditioning
serialized layout
serialized sigmas
serialized execution configuration
```

However, for "save as much state as practical", add an optional prepared-state section.

This may include portable copies of expensive derived data such as:

```text
refined text representation
precomputed timestep/AdaLN schedule
RoPE cos/sin tensors
resolved row/modulation maps
final audio/video modulation maps
```

These are all derived data and may be omitted without changing output.

The checkpoint reader therefore distinguishes:

```text
REQUIRED RESUME STATE
```

from:

```text
OPTIONAL REBUILDABLE CACHE
```

If a cache section is incompatible with the current H3 build, discard that section and recompute it instead of rejecting the entire checkpoint.

---

# 27. Refined text

Saving the post-DiT-text-refinement representation is especially worthwhile.

The DiT loader currently prepares/refines text before the persistent transformer core is used.

Add an export/import mechanism for the refined text tensor.

Serialize the native representation losslessly.

On resume:

```text
if compatible refined-text cache exists:
    restore it

else:
    rebuild from serialized h3_text_embedding
```

This reduces resume startup work while keeping the checkpoint resilient across implementation changes.

---

# 28. Precomputed timestep/AdaLN data

Current H3 separately precomputes per-step DiT schedule/AdaLN data.

Implement an optional export/import representation for these precomputed tensors.

Because this cache can be sizeable, expose checkpoint logging such as:

```text
sampler state:             68 MiB
conditioning:              23 MiB
prepared DiT cache:       210 MiB
total checkpoint:         301 MiB
```

Exact size depends on geometry and schedule.

The initial version may save this cache uncompressed.

---

# 29. What must not be serialized

Do not attempt to serialize:

```text
MTLDevice
MTLBuffer object identities
MTLCommandBuffer
MTLCommandQueue
MPSGraph objects
MPS tensor-data wrappers
compiled Metal pipelines
temporary attention buffers
temporary QKV/MLP activations
mapped model-weight objects
video-VAE decoder object
FFmpeg encoder state
```

These are process/device runtime objects rather than mathematical generation state.

Recreate them on resume.

Likewise, do not copy the immutable 33B transformer weights into `.h3sample`; they remain in the model directory.

---

# 30. Checkpoint save sequence

At a requested boundary:

```text
finish Euler transition K-1
        |
        v
next_step = K
        |
        v
wait for relevant Metal work to complete
        |
        v
materialize canonical host state
        |
        +-- current video/audio latent
        +-- reuse velocity history
        +-- core-reuse residual if required
        +-- optional prepared state
        |
        v
write temporary checkpoint
        |
        v
flush + checksum
        |
        v
atomic rename
        |
        v
optional preview decode
        |
        v
return PAUSED result
```

Do not expose a checkpoint file until every section has been written successfully.

---

# 31. Atomic persistence

Save to:

```text
shot17.h3sample.tmp
```

then:

```text
fsync
validate header/section table
rename → shot17.h3sample
```

Each large section should have its own checksum.

On load:

```text
validate header
validate all byte counts before allocation
validate integer multiplication overflow
validate checksums
validate section uniqueness
```

A partially written state must never be interpreted as a valid generation.

---

# 32. Resume sequence

Resume performs:

```text
read checkpoint header
        |
validate model/build compatibility
        |
restore generation configuration
        |
restore exact text conditioning
        |
restore condition video/audio rows
        |
restore layout
        |
restore sigma schedules
        |
load/rebuild prepared DiT
        |
restore optional prepared caches
        |
restore current AV latent
        |
restore continuation masks
        |
restore Euler reuse history
        |
restore core-reuse state
        |
next_step = saved next_step
        |
continue sampling
```

The tokenizer, text encoder, vision encoder and reference-media encoders should not run during a normal resume.

---

# 33. Resume CLI authority

When `--resume-sampler-state` is supplied, the checkpoint is authoritative for every generation-sensitive parameter.

The user may still specify:

```text
model directory
output path
frames directory
show/preview behavior
profile/logging options
new stop-after boundary
new save-state path
```

Generation-changing parameters such as:

```text
prompt
seed
width
height
frames
steps
references
layers
reuse
core-reuse
token reduction
continuation source
continuation context
```

should initially be rejected if supplied on a resume command.

This prevents accidental "resume" operations that are actually modified generations.

Later a separate fork/edit-state feature could be designed explicitly.

---

# 34. In-process resume optimization

If the same `h3_ctx` that created a pause state is still alive and its prepared DiT key matches, resuming may reuse:

```text
ctx->dit
ctx conditioning
ctx video decoder
```

directly.

The serialized checkpoint remains the source of truth.

This optimization must not change output relative to:

```text
exit process
restart
load checkpoint
resume
```

---

# 35. Partial preview

`--preview-on-stop` should decode the current video latent without modifying it.

The preview is only an inspection artifact.

Conceptually:

```text
checkpoint latent
      |
      +---- copy
              |
              v
          Video VAE
              |
              v
         preview MP4
```

The checkpoint always contains the pre-decode latent.

Never round-trip the preview into the sampler.

Audio preview may initially be omitted because an early partially denoised audio latent is unlikely to be useful for seed/composition selection.

---

# 36. Stopped result status

A pause is not an error.

Extend the result API conceptually with:

```c
typedef enum {
    H3_RESULT_COMPLETE,
    H3_RESULT_PAUSED
} h3_result_status;
```

and:

```text
status
completed_steps
total_steps
```

The CLI should return success after a valid checkpoint is written.

Do not use the existing preview callback's nonzero return mechanism, which currently represents termination/error semantics.

---

# 37. Final completion

If:

```text
next_step == total_steps
```

then execute the existing final path unchanged:

```text
final AV latent
        |
        +---- create/save .h3av
        |
        +---- VAE decode
        |
        +---- continuation prefix trim
        |
        +---- MP4
```

The stop/resume feature must not change the semantics of `.h3av`.

A resumed continuation segment therefore produces exactly the same kind of final continuation state as an uninterrupted continuation segment.

---

# 38. Interaction with continuation

The following workflow must be supported:

```text
segment01.h3av
       |
       v
construct segment 2:
    noise
    +
    39-frame / 65-tick inherited prefix
       |
       v
run first 4 of 20 steps
       |
       v
segment02-step4.h3sample
       |
       v
resume
       |
       v
finish steps 4...19
       |
       v
segment02.h3av
```

After `.h3sample` exists, `segment01.h3av` is no longer necessary to finish segment 2.

Everything derived from it that matters has already become part of:

```text
current AV latent
continuation mask
conditioning
checkpoint metadata
```

---

# 39. State provenance

A resumed final `.h3av` should record optional provenance:

```text
generation was resumed
checkpoint format version
checkpoint source hash
checkpoint initial completed step
number of resume operations
```

This has no effect on H3.

It will be valuable when investigating differences between uninterrupted and resumed renders.

---

# 40. Exactness modes

Define three levels.

**Exact same-path resume**

```text
same h3cli build
same checkpoint
same model
same Mac/GPU family
same Metal sampler mode
same numerical options
```

Acceptance criterion:

```text
final AV latent is byte-identical
```

to uninterrupted generation.

**Compatible Metal resume**

```text
same model and mathematical configuration
different compatible Apple GPU or implementation revision
```

Acceptance criterion:

```text
structurally valid and visually equivalent
```

but byte identity is not guaranteed.

**Future cross-backend resume**

```text
Metal → CUDA
```

Not implemented yet.

The serialization format should preserve enough canonical mathematical state to make this possible later.

---

# 41. Quality-first initial scope

The first fully validated configuration should be:

```text
original BF16 model
50 DiT blocks
reuse = 1
core-reuse = 1
token reduction = off
normal full render geometry
existing continuation masks supported
CPU sampler on M4
```

This is the simplest state machine and the path most relevant to quality-first Mac scouting.

After exact resume is proven there, add:

```text
whole-denoiser reuse
GPU-state sampler
core reuse
token reduction
```

one at a time.

---

# 42. Acceptance test: uninterrupted versus resumed

For a deterministic test:

```text
RUN A:
steps 0...19 continuously

RUN B:
steps 0...3
save
destroy process
reload
steps 4...19
```

Compare:

```text
final video latent bytes
final audio latent bytes
final .h3av bytes
decoded PCM
decoded video frames
```

For same-build/same-device/same-sampler execution they should be byte-identical where the existing implementation itself is deterministic.

Repeat boundaries at:

```text
1
2
4
10
19
```

to expose off-by-one schedule bugs.

---

# 43. Acceptance test: multiple resumes

Also test:

```text
steps 0...3
save

resume
steps 4...7
save

resume
steps 8...11
save

resume
steps 12...19
```

against one uninterrupted run.

This proves the checkpoint is a complete state representation rather than relying on hidden process state.

---

# 44. Acceptance test: continuation

Use the existing validated continuation test:

```text
segment 1 complete
        |
segment 2:
39-frame AV overlap
20-step schedule
pause after step 4
resume
        |
segment 2 complete
        |
segment 3 continuation
```

Compare the final segment-2 latent against an uninterrupted segment-2 render.

Then verify segment 3 also matches, proving that resumed generation produces a valid `.h3av` continuation source.

---

# 45. Logging

On stop:

```text
H3 sampler paused
Schedule:                 20 steps
Completed:                 4
Next step:                 4
Video sigma:               ...
Audio sigma:               ...
Mode:                      Ref2VA
Continuation:              yes
Protected video context:   12 latent steps
Protected audio context:   65 ticks
Checkpoint:                shot17.h3sample
Checkpoint size:           ...
```

On resume:

```text
H3 sampler resume
Checkpoint:                shot17.h3sample
Original prompt:           ...
Mode:                      Ref2VA
Geometry:                  768x1024 / 141 frames
Schedule:                  20 steps
Starting at step:          4
Remaining transitions:     16
Conditioning restored:     yes
Reference encoding skipped: yes
Text encoder skipped:      yes
Prepared cache restored:   yes/no
```
