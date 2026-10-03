# Native AV latent continuation

Continuation carries the complete final normalized video/audio latent into the
next generation. `.h3av` contains lossless F32 `[24,T,H,W]` video and `[32,2,T]`
audio, geometry, seed, a versioned model/VAE signature and a SHA-256 checksum.
No rendered media is encoded again to obtain continuation context.

This guide describes the default `hard` mode. Optional
[bridge continuation](bridge-continuation.md) adapts the older part of the
inherited context while preserving its endpoint. Both modes use the same state
format, reference handling, and output trimming. Both accept ordinary denoiser
reuse 2/3; bridge also has separate core-reuse 4/6 support. Approximate CUDA
adaptive/SubBlock continuation requires both reuse controls to be 1.

Start with **39 frames** of context. Legal contexts are `39 + 51*k`: 39, 90,
141, 192, etc., subject to source/target lengths. These map to 12/27/42/57 video
latent steps and 65/150/235/320 audio ticks. A 39-frame context is exactly
1.625 seconds at 24 fps and removes 52,000 samples per stereo channel at 32 kHz.

`--frames` denotes the **complete internal target**, rounded using the existing
H3 rule. Normal continuation output excludes the protected prefix. For example,
`--frames 90 --continue-context 39` delivers 51 frames (2.125 seconds), while
`--frames 362` delivers 323 frames (about 13.458 seconds). Saved state always
retains the complete raw target. Existing audio rounding at target endpoints is
unchanged; targets 39/90/141/192/etc. also have exact AV durations. To ensure the
next context contains entirely new content, choose a raw target at least twice
the context (90 frames for context 39).

Start with 20 steps, `--reuse 1 --core-reuse 1 --layers 50`, and token reduction
off. Continuation explicitly uses the CPU F32 Euler **sampler**; the transformer
and VAEs execute on the selected Metal or CUDA backend. See [acceptance results](continuation-acceptance.md)
for tested optimizations and hardware. Core reuse greater than one, fewer than
50 blocks, and token reduction are rejected pending dedicated parity coverage.
`H3_GPU_SAMPLER=1` does not bypass the hard-continuation fallback. These limits
describe hard mode; [bridge continuation](bridge-continuation.md) has separate
Metal GPU-state and core-reuse support. Current SGLang CUDA arithmetic uses the
CPU-state F32 sampler for both modes, regardless of GPU-sampler environment flags.
Whole-denoiser `--reuse 2` and `--reuse 3` have passed dedicated 20-transition
prefix-invariance tests and are supported. They approximate generated-suffix
velocities, so begin with reuse 1 when judging output quality.

On qualified SM120 CUDA, BF16 `--adaptive-cache conservative|aggressive`,
`--cuda-attention subblock`, or both also support continuation with ordinary
image/video/audio/mixed references. They require all 50 layers, reuse/core-reuse
1, no LoRA or token reduction, and ordinary video geometry. Quantized
approximation, first/last anchors and upscaling remain excluded. Recipe 4 scores
generated suffixes and active bridge classes independently from frozen history.
SubBlock protects the whole prefix. See the [native contract](../cuda/adaptive-subblock-contract.md#continuation-recipe-4).
The [qualification report](../cuda/adaptive-continuation-results.md) includes
exact replay checks, performance measurements and comparison videos.

Video history is initialized once as `0.999 * clean + 0.001 * new_noise`.
Audio history remains clean. Protected video rows receive `max(1-sigma_v,0.999)`;
protected audio rows receive `1.0`. Protected velocities are zero and the CPU
Euler loop skips these positions, preserving even signed-zero bits.

## References and CLI examples

Keep clean image/body references on every Ref2VA segment to maintain identity.
Use `.h3av` for temporal continuity. References retain their ordinary ordering,
labels, limits and checkpoint selection; continuation consumes no reference slot.
Each call supplies its own references and prompt and may change either.
Recursively supplying the previous generated MP4 as a Ref2VA reference is normally
unnecessary and may reintroduce color/contrast drift.

These examples use `inputs/`. Face/body images can depict different people:
explain in the prompt which identity and outfit to use.

```sh
# First segment: 90 delivered frames plus resumable state.
./bin/h3cli -d models/MiniMax-H3 \
  -p 'A woman with the face in <Picture 1> wears the outfit in <Picture 2> and walks through a sunlit garden.' \
  --width 256 --height 256 --frames 90 --steps 20 \
  --reuse 1 --core-reuse 1 --layers 50 --seed 42 \
  --ref-image inputs/1.jpg --ref-image inputs/2.jpg \
  --save-av-state outputs/segment01.h3av -o outputs/segment01.mp4

# Same references: deliver 51 new frames, save the complete 90-frame state.
./bin/h3cli -d models/MiniMax-H3 \
  -p 'The woman continues walking and waves toward the camera.' \
  --width 256 --height 256 --frames 90 --steps 20 --seed 43 \
  --continue-from outputs/segment01.h3av --continue-context 39 \
  --ref-image inputs/1.jpg --ref-image inputs/2.jpg \
  --save-av-state outputs/segment02.h3av -o outputs/segment02.mp4

# Third segment: changed references and action, preserving the AV prefix.
./bin/h3cli -d models/MiniMax-H3 \
  -p 'The woman walks further into the garden. The person in <Picture 1> wearing the outfit in <Picture 2> approaches.' \
  --width 256 --height 256 --frames 90 --steps 20 --seed 44 \
  --continue-from outputs/segment02.h3av \
  --ref-image inputs/3.jpg --ref-image inputs/4.jpg \
  --save-av-state outputs/segment03.h3av -o outputs/segment03.mp4

# Resume after restarting the process.
./bin/h3cli -d models/MiniMax-H3 -p 'The woman looks upward as the camera follows.' \
  --width 256 --height 256 --frames 90 --steps 20 --seed 45 \
  --continue-from outputs/segment03.h3av \
  --ref-image inputs/3.jpg --ref-image inputs/4.jpg \
  --save-av-state outputs/segment04.h3av -o outputs/segment04.mp4

# Debug: retain duplicated history for comparison with the source tail.
./bin/h3cli -d models/MiniMax-H3 -p 'The woman continues walking and waves toward the camera.' \
  --width 256 --height 256 --frames 90 --steps 20 --seed 43 \
  --continue-from outputs/segment01.h3av --keep-continuation-prefix \
  --ref-image inputs/1.jpg --ref-image inputs/2.jpg \
  --save-av-state outputs/segment02-debug.h3av -o outputs/segment02-debug.mp4
```

For T2VA, omit the reference options. `--ref-video`, `--ref-silent-video`,
`--ref-video-audio VIDEO AUDIO`, and `--ref-audio` keep their existing meanings.
Standalone audio still requires an accompanying visual reference; video
soundtracks need at least a legal 56-frame source chunk and two seconds of audio.
Continuation does not relax these restrictions.

The first implementation rejects continuation with `--first-frame` or
`--last-frame`. A future last-frame quality-reset design must keep inherited
target history independent of future condition rows, define conflicting
constraints, and test last-frame anchoring alongside the protected AV boundary.

File-related CLI options require one-shot `-p` generation. Library callers can
chain states inside one process, including with conditioning/model caches.

## Library ownership

```c
h3_params p = H3_PARAMS_DEFAULT;
p.frames = 90;
h3_result *first = h3_generate(ctx, "The camera follows a walker.", &p);
/* Check first and h3_last_error(ctx) before using it. */
p.continuation = h3_result_av_state(first); /* borrowed */
p.continuation_context_frames = 39;
p.seed++;
h3_result *next = h3_generate(ctx, "The walker turns toward the camera.", &p);
h3_result_free(first); /* next owns its independent complete final state */
h3_result_free(next);
```

`h3_result_av_state()` and metadata/video/audio accessors return borrowed,
read-only pointers. `h3_av_state_clone()` creates independent ownership.
`h3_av_state_load()` returns an owned state, freed with `h3_av_state_free()`.
Input states must live until generation returns. Load/save use caller-provided
error buffers. `result->frames` and `result->audio_samples` describe delivered
media; state metadata describes the complete target. Optional `on_latent_step`
diagnostics observe initialization and every CPU Euler transition and can cancel
by returning nonzero. They select the CPU sampler, including for ordinary runs.

States require identical internal render geometry and matching normalization,
transformer configuration and VAE weights. The signature hashes a versioned
runtime contract, model/VAE configs and complete VAE weights. FL2VA/Ref2VA can
share states when they share this latent space. Contexts cache their signatures;
do not replace model files while a context is alive. State contents never enter
the conditioning cache key; the prepared DiT key includes context geometry and,
for adaptive/SubBlock continuation, source state identity. Each new AV segment
resets adaptive history. Same-job sampler resume restores history and conditioning
without reopening the source AV or reference files.

`--profile` reports signature/load/save time, state bytes, prefix copy time,
modulation storage and sampler timings. At most two extra timestep embeddings
are shared across protected rows. There are no spatial mask tensor allocations.

## File format, version 3

Integers and F32 payloads are little endian; other host endianness is rejected.
The 160-byte header precedes channel-major video and audio payloads.

| Byte offset | Value |
|---:|---|
| 0 | Eight-byte magic `H3AV\r\n\x1a\n` |
| 8, 12 | U32 version 3 and header length 160 |
| 16, 20 | U32 endian marker `0x01020304`, F32 format tag 1 |
| 24–60 | Ten U32 values: render width, height, frames, video T, latent H/W, audio T, 24 video channels, 32 audio channels, 2 streams |
| 64 | U64 seed |
| 72, 80 | U64 video/audio payload byte lengths |
| 88 | 32-byte model/VAE compatibility SHA-256 |
| 120, 124 | U32 geometry profile (0 ordinary, 1 upscale) and reserved zero |
| 128 | SHA-256 of header bytes 0–127 followed by both payloads |
| 160 | Video F32 payload, then audio F32 payload |

Loading validates exact file size and geometry before allocation. Saving writes
an adjacent temporary file, flushes it and renames it atomically. References and
prompts are supplied per call and are not serialized. Complete CLI/API saves
also write current presentation 9, bound to the AV fingerprint. Use
`h3_result_save_av_state()` for that pair. Old schemas are rejected; see the
[current state contract](../features/current-state-contract.md).

## Reproducing validation

```sh
make -j8 all
make test
make test-continuation-sanitize test-bridge-sanitize test-sampler-sanitize
python3 tests/continuation_oracle.py
```

For CUDA, also run the current-feature suite and immutable 204-output gate in
[CONTRIBUTING.md](../../CONTRIBUTING.md). The [adaptive continuation protocol](../cuda/design-adaptive-continuation.md)
uses bounded six-evaluation functional jobs and a separate frozen 14-video
comparison. Older 20-step face/body campaigns remain historical evidence, not
current test prerequisites or compatible saved-state fixtures.
