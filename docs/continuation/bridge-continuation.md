# Bridge continuation

Bridge generation now runs end to end with joint video/audio initialization,
fractional timestep modulation, and F32 Euler sampling. The transformer and
VAEs run on Metal or CUDA. CUDA uses CPU-state Euler; Metal also supports its
separate optional GPU-state bridge sampler. CPU Euler is the bridge default; hard
continuation remains the default continuation mode.

The implementation follows [design-bridge.md](design-bridge.md) and builds on
the [profile and flow mathematics](bridge-foundation.md). Bridge remains opt-in until the [motion-quality and completion gates](bridge-quality.md)
pass. Numerical correctness and matching CPU/GPU results do not establish that
bridge improves a given transition. The current approximation extension is
tracked in [the task list](../todo.md).

## Generate a bridge segment

Bridge is intended primarily for **meaningful action changes within the same
scene**: a walking character turning, changing direction, or gradually raising
their arms while the surroundings, character and references stay consistent.
Use hard mode when the prompt and motion trajectory change only minimally,
including an unchanged action. Exact inheritance is a useful default; bridge
strength is an optional tradeoff, not a universal quality improvement.

Describe the transition in order: “completes the current walking step, slows,
gradually turns left, then begins running.” For a pose change, use “finishes the
current motion and smoothly raises both arms from her sides.” Give the model
time to move between poses. Keep the scene, lighting, camera description and
face/body references consistent across segments.

Bridge does not guarantee continuity when the next request intentionally
changes identity, references, environment, or scene incompatibly. Plan a cut,
generate a new scene, or use a separate transition strategy for those cases.
Partially adapting inherited history cannot make contradictory endpoints
physically continuous.

Use a current schema-3 `.h3av` state with presentation 9. The source and target
must use the same rendered dimensions and VAE
compatibility signature. This example assumes a 256-square source:

```sh
./bin/h3cli -d models/MiniMax-H3 \
  -p 'The woman with the face in <Picture 1> and the outfit in <Picture 2> completes her walking step, gradually turns toward the camera, and raises both arms. Sunlit garden, steady camera, quiet outdoor ambience.' \
  --width 256 --height 256 --frames 90 --steps 20 --seed 44 \
  --reuse 1 --core-reuse 1 --layers 50 \
  --ref-image inputs/1.jpg --ref-image inputs/2.jpg \
  --continue-from outputs/segment01.h3av --continue-context 39 \
  --continue-mode bridge --continue-bridge-steps 8 \
  --continue-bridge-max-strength 0.5 --continue-bridge-profile stepped \
  --save-av-state outputs/segment02.h3av -o outputs/segment02.mp4
```

The defaults are eight bridge video rows, maximum strength 0.5, and `stepped`.
`linear` and `ease-out` also work. Strength must be finite and between zero and
one. Bridge length must leave at least one exact video row; fewer than two
emits a warning. A zero-strength bridge is supported for comparison with hard
mode. Hard mode ignores all bridge tuning fields.

Start with `--reuse 1 --core-reuse 1 --layers 50` for quality comparisons. Bridge
also accepts whole-denoiser reuse 2/3, or core reuse 4/6 with denoiser reuse 1.
The two reuse mechanisms cannot be combined. Other core intervals and custom
`H3_REUSE_STEPS` schedules are rejected. Full layers, token reduction off, and
no first/last-frame anchors remain required.

On Metal, `H3_GPU_SAMPLER=1` explicitly selects the bridge GPU-state sampler.
`H3_CPU_SAMPLER=1` overrides it and keeps the CPU oracle. GPU-state sampling
packs the jointly initialized AV latents once, applies the same class strengths
after raw velocity extrapolation, and audits exact bits after every update.
It supports latent callbacks and previews. CUDA's current SGLang arithmetic
keeps CPU-state Euler even when this environment flag is set. Hard continuation keeps its CPU
fallback; ordinary generation retains its existing sampler selection.

BF16 CUDA bridge also accepts adaptive cache, SubBlock, or both, with ordinary
image/video/audio/mixed references. These combinations require reuse/core reuse
1, all 50 layers and no LoRA/token reduction. Adaptive recipe 4 takes the maximum
of global, generated suffix and each active bridge-class score. SubBlock keeps
all inherited rows protected, including mutable bridge queries. Both heads
remain fresh and raw bridge velocities are scaled once. Quantized approximation,
anchors and upscaling remain excluded. See the [native contract](../cuda/adaptive-subblock-contract.md#continuation-recipe-4).
The [qualification report](../cuda/adaptive-continuation-results.md) includes
exact replay checks, performance measurements and comparison videos.

## Context, references, and saved state

With context 39, the default bridge covers eight video rows (26 frames) and
audio ticks `[0,43)`. The exact endpoint covers four video rows (13 frames) and
audio ticks `[43,65)`. Both stereo timelines have the same classes. The video
mask is:

```text
0.50 0.50 0.50 0.40 0.30 0.20 0.20 0.10 | 0 0 0 0 | 1 1 ...
```

All 39 context frames are normally removed after decoding the full target.
A 90-frame target therefore delivers 51 new frames and 68,000 audio samples
per channel: 2.125 seconds in both streams. `--keep-continuation-prefix` exposes
the adapted context, exact endpoint, and suffix in one 90-frame debug clip.
The saved state always contains the complete 90-frame target in either case.

The current `.h3av` v3 format contains final F32 video/audio latents, geometry,
seed, compatibility signature, and checksum. It stores no required bridge
configuration. A hard state can feed bridge generation; a bridge state can feed
either mode. The source state is borrowed and remains unchanged. A new segment
starts with empty adaptive history; same-job sampler resume restores it without
the source/reference files. The existing
[C API ownership rules](continuation.md) still apply; set `continuation_mode`
to `H3_CONTINUE_BRIDGE` on an initialized `h3_params` and use the same bridge
fields as the CLI.

References retain their normal packing, order, labels, and checkpoint selection.
Supply the desired face/body references on each call. The continuation state
uses no reference slot and never becomes an implicit image, video, or audio
reference. The integration tests retain the same face/body pair while changing
the prompt from walking to turning and raising both arms, then exercise both
hard and bridge continuation from the resulting state.

## Numerical behavior and diagnostics

Bridge initialization reuses the ordinary target noise, without additional RNG
draws, and mixes it with the source tail at `mask * initial_stream_sigma`.
Exact video keeps the original `.999f * clean + .001f * noise` arithmetic;
exact audio is copied byte for byte. Generated suffix noise is untouched.

Packed target rows resolve temporal classes through the existing video mapping
and stereo audio packing. Each active class receives its own effective
timestep and materialized AdaLN vectors for all 50 blocks and final heads.
Reference and condition classes retain their original row indices. Equal
timesteps within a step share one materialized row.

The bridge forward pass returns raw velocity. Both samplers cache raw predictions
for reuse, extrapolate when needed, scale once by the current row's strength,
and apply the F32 Euler update. Exact rows are skipped and compared bit for bit
against their initialized values after **every** step. GPU audit kernels retain
any failure in a flag checked whenever an encoded command window completes.
Core reuse still runs current-timestep final heads on every step; prepared-cache
reuse resets its residual state before each generation.

Normal logging includes mode, total context, bridge/exact durations, profile,
maximum strength, video masks, audio interval, and packed rows per class.
`H3_BRIDGE_DIAGNOSTICS=1` adds exact-row audit results and, with core reuse,
checks the actual uploaded block/head class maps at each current timestep.
On the CPU sampler it also reports class velocity RMS before/after scaling,
update RMS and changed-element counts. Exact classes report zero raw RMS
because their unused predictions are not accumulated.
`H3_PROFILE=1` enables those diagnostics plus construction, initialization,
modulation preparation, and CPU update timing and memory accounting.

The packed modulation maps have the same allocation size as hard mode. Extra
memory comes from the compact class plan, additional modulation vectors, and
the initial AV snapshot used to audit exact rows. At 256 square, 90 frames,
20 steps, and the default profile, the smoke run used 255 distinct time rows
(214 extra), 2,075,437,056 extra AdaLN bytes, a 3,672-byte class plan, and a
701,952-byte snapshot. These measurements are configuration dependent.
Prepared DiT cache keys include bridge length, maximum strength, and profile;
conditioning can still be reused independently.

## Reproduce validation

Use current builds and state files. Full-model GPU jobs run sequentially;
ordinary test requests retain the six-evaluation ceiling.

```sh
make -j8 all
make test
make test-bridge-sanitize test-continuation-sanitize test-sampler-sanitize
```

On CUDA, follow [CONTRIBUTING.md](../../CONTRIBUTING.md) for the current-feature
suite and unchanged 204-output SGLang gate. The new
[continuation protocol](../cuda/design-adaptive-continuation.md) covers adaptive
cache, SubBlock, references, exact resume and a frozen 14-video comparison.
Its functional runner needs NumPy, the qualified CUDA environment and installed
models; it derives its media fixtures from an existing current video.

The original [bridge integration acceptance](bridge-integration-acceptance.md)
and [quality report](bridge-quality-acceptance.md) are historical records.
Their old schemas, fixture paths and 20-step campaigns are not current release
prerequisites. Their failed motion-quality gate remains recorded; numerical
correctness does not establish a quality improvement for bridge.
