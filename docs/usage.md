# Advanced usage and rendering controls

For your first video, start with the [README quick start](../README.md#make-your-first-video).
This guide retains the detailed rendering controls, feature restrictions and
measured tradeoffs. Commands assume a source checkout and run from its root;
with a release package, replace `./bin/h3cli` with the downloaded executable.

- [First render](#2-make-a-first-fast-video)
- [Quality presets](#4-choose-a-speedquality-preset)
- [Resolution and duration](#5-pick-resolution-and-duration)
- [Image, video and audio references](#8-add-image-video-and-audio-references)

For optional approximate Metal/CUDA video reconstruction, see the
[fast VAE preview guide](preview/preview-vae.md): `--preview-vae` accelerates MP4
and `--show` decoding, while `--save-av-state` plus `--decode-av-state` lets
you finalize the same latents with the original VAE later. Normal decoding
remains the default. The native tiny decoder is tested on M4 Max and RTX 5090.

The **original video VAE** has one execution path per backend: the qualified
SGLang decoder on CUDA and the original FP32 decoder on Metal. CUDA retains
bounded streaming, graph replay and GPU stitching. No execution selector is
needed; `--full-vae-execution` and its alternate implementations have been
removed. The [original-model VAE guide](preview/fast-vae.md) covers saved
latent decoding and validation. `--preview-vae` stays separate.

For adapters, use repeatable `--lora PATH[:SCALE]` flags with the original model:
H3 folds a missing BF16 variant at startup and verifies cached variants on later
requests. `--lora-cache DIR` selects storage and `--lora-memory-mib N` bounds
folding allocations (default 512 MiB). This native path works with Metal and
CUDA, requires no Python, and composes with FP8/NVFP4 and preview VAE. See the
[LoRA guide](../lora/README.md#runtime-folding-metal-and-cuda) for first-run costs,
multiple adapters, checkpoints, and the optional offline workflow.

For optional RTX 5090 denoising previews, see the
[FP8/NVFP4 guide](cuda/denoiser-quantization.md). Select
`--cuda-denoise-quant fp8` or `nvfp4`; the default remains `off`. These options
compose with explicit Sage/SOL attention and `--preview-vae`, change video/audio denoising,
and have substantial one-time preparation and fresh-process loading costs.

Qwen attention now defaults to FP32 post-QK scaling. The
[scaling guide](bugfixes/scalingfix.md) documents compatibility modes, changed
historical-seed output and [M4 validation](bugfixes/scalingfix-validation.md).

The [memory and long-reference guide](stability/memory.md) explains the 10-GiB safety
floor, prompt cancellation, device-specific GQA preflight, and streaming video
conversion. Validation results are in [the memory report](stability/memory-validation.md).

Dialogue, cutoff, lyrics, and caption model markers are documented in the
[H3 tokenizer guide](minimax/tokenizer.md), including exact token IDs and parity tests.

Metal sampler pause/resume is documented in the [sampler checkpoint guide](features/sampler-state.md).
Advanced sampler validation is recorded in the [T033–T070 acceptance report](features/resume-advanced-acceptance.md).
`--steps 20 --stop-after-step 4 --save-sampler-state shot.h3sample` saves the
first four transitions of the full schedule. Resume with
`--resume-sampler-state shot.h3sample`; optional pause previews are silent.

For repeated Metal benchmarks, `--save-conditioning shot.h3cond` persists
conditioning and `--load-conditioning shot.h3cond` bypasses the encoders after
strict prompt/reference/model/geometry checks. `--conditioning-schedule` also
saves or requires exact-schedule AdaLN. See the
[conditioning format and benchmark guide](metal/conditioning-format.md) and
[M0/M1 measurements](metal/m0-m1-results.md). `--backend mpsgraph` remains the
reference/default. Native attention requires `--backend metal`.
The [FP16 hybrid](metal/m2-fp16-implementation.md) additionally
uses `--metal-attention-kernel steel-routed --metal-attention-dtype fp16`.
It keeps BF16 state and existing linears, uses FP32 attention accumulation, and
reports range recovery. A `--metal-tier` label does not qualify a production
Reference renderer or change defaults.
[Metal Q8 weights](metal/q8.md) are selected independently with
`--metal-weight-format q8`. The default Q8 policy uses bounded Metal
dequantization followed by MPSGraph linears; BF16 remains the default format.
`--metal-attention-layout fused` selects the optional M2E norm/RoPE range fusion
and dense in-place packing; `adapter` remains the default. Use the BF16 QKV path
(`--use-slower-bf16-qkv`). Both layouts preserve the accepted mixed arithmetic;
routed SOL retains separate BF16 summary inputs. See the
[layout design](metal/design-native.md#m2e--accepted-path-layout-fusion) and
[243-frame fusion measurements](metal/m2e-layout-results.md): dense fusion
saves 0.87 GiB but provides no measured speed gain; retain the adapter for SOL.
The [243-frame qualification report](metal/m2-reference-qualification.md)
retains the accepted performance result, matched-input/range diagnostics and
paired production-VAE B6 playback. B6 failed the frozen quality screen, so the
FP16 backend remains opt-in.

The optional [M3C GPU/ANE QKV prototype](metal/m3c-ane-results.md) adds
`--metal-ane serial|static|dynamic` to that FP16 hybrid. `off` is the default.
It splits independent QKV rows between MPSGraph and scaled FP16 CoreML/ANE,
retaining original BF16 weights/state and GPU recovery. The default row budget
is 4096 (`--metal-ane-rows`), with 512-row compiled chunks
(`--metal-ane-chunk 256|512|1024`). Dynamic decisions are saved in sampler
checkpoints; `--profile` enables detailed ANE timing and provenance records.
`H3_ANE_CACHE_DIR` selects a private application graph cache;
graphs contain no model weights. The isolated Apache-2.0 emitter retains
[VPIPE attribution](../third_party/vpipe-ane/NOTICE) and uses an undocumented
compiled CoreML format. The 243-frame M4 comparison found no meaningful B5
speedup, and ANE failed the incremental B6 quality screen. Keep it disabled for
normal rendering; the results page includes paired playback and measurements.

Native video/audio latent continuation is described in the
[continuation guide](continuation/continuation.md), including `.h3av` state ownership,
CLI chaining examples, supported settings, and reproducible acceptance tests.

[Bridge continuation](continuation/bridge-continuation.md) supports
fractional adaptation of inherited video/audio context with an exact endpoint.
CUDA uses CPU-state F32 Euler; Metal retains its separate optional GPU-state
bridge sampler. Optional denoiser/core reuse and BF16 CUDA adaptive/SubBlock
policies retain their respective admission rules.
Hard mode remains the default continuation mode. The
[quality protocol](continuation/bridge-quality.md) separates numerical correctness from
evidence of smoother action transitions. The [measured results](continuation/bridge-quality-acceptance.md)
have not yet established the required motion-quality improvement.

### 1. Build and inspect the model

Complete the platform setup in [Build](build/source-build.md) first.
Generation downloads missing required files into `models/MiniMaxH3` by default.
Use `--models-path ROOT` to relocate all default main/auxiliary paths, or
`-d/--model-dir` to override only the main checkpoint. An existing
`ROOT/MiniMax-H3` is reused when `ROOT/MiniMaxH3` is absent. Source the setup environment to select FFmpeg/FFprobe 9.0.2;
the standalone Linux executable includes both tools.

```sh
make -j8
mkdir -p outputs
./bin/h3cli --download-models base
./bin/h3cli --info
```

`--info` checks the model layout and prints the selected GPU without
mapping all weights or generating media. Run `./bin/h3cli --help` for the complete CLI
reference.

Generation requires `-p/--prompt`. Each invocation performs one request.
Use `--info` to inspect a model, `--resume-sampler-state` to resume a checkpoint,
or the decode options to render saved latents.

### 2. Make a first fast video

Start with the validated balanced preset. It generates 22 frames at 24 fps
(about 0.92 seconds), displays the evolving middle-video frame after every
denoising transition in a supported graphical terminal, and prints phase
timings:

```sh
./bin/h3cli --profile \
  -d ./models/MiniMaxH3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 \
  --frames 22 --steps 20 \
  --layers 45 --reuse 2 \
  --show \
  -o outputs/fox-fast.mp4
```

This is deliberately not the most aggressive configuration:

- `--steps 20` performs the default 20 denoising passes.
- `--reuse 2` computes 11 fresh denoiser velocities instead of all 20 and
  extrapolates the skipped transitions.
- `--layers 45` runs 45 of the 50 transformer blocks, reducing both time and
  unified-memory use.
- `--show` is optional. It supports Kitty/Ghostty and
  iTerm2/WezTerm/Konsole graphical protocols. It loads a resident preview VAE,
  displays one representative middle-video frame after every Euler transition,
  and then displays all final frames. Display dimensions default to 2x so the
  image has its intended logical size on macOS Retina screens; use `--zoom 1`
  on a non-HiDPI display. This adds preview decode time and roughly 10 GiB of
  temporary model residency; runs without `--show` are unchanged.
- `--profile` is optional and does not select a different generation path.

The first process invocation also pays model loading and filesystem-cache
costs. Compare performance using repeated runs, and alternate variants when
the machines are warming up because this workload is sensitive to thermal
throttling.

For a very short iteration, request four denoising passes directly:

```sh
./bin/h3cli --profile \
  -d ./models/MiniMaxH3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur." \
  --width 512 --height 512 --frames 22 \
  --steps 4 --layers 50 --reuse 1 \
  --show \
  -o outputs/fox-four-step.mp4
```

`--steps N` always means exactly N denoising passes. Four through seven passes
use the same schedule that won the low-budget comparison; increasing from 4
to 7 progressively improves detail and motion. Keep `--reuse 1` at such small
budgets so every requested pass runs the model. `--show` displays one preview
after each pass.

Several tail-heavy schedules were evaluated because most visible cleanup
happens late in a long run. They preserved too few early composition updates
and produced woven texture, weak motion, or clipped colors. The retained mode
uses the released linear base grid with one terminal point. On the 512-square,
22-frame fox test, the selected four-pass result had 0.556 full-video SSIM
against a 29-pass reference; an independent surfer test measured 0.547. The
four-pass denoise took about 3.5 seconds on M5 Max, versus 26.4 seconds for the
reference.

For a low-memory run, add `--ssd-streaming`:

```sh
./bin/h3cli --profile \
  -d ./models/MiniMaxH3 \
  -p "A red fox walks through fresh snow in a pine forest." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 50 --reuse 1 --ssd-streaming \
  -o outputs/fox-ssd.mp4
```

This uses the original BF16 checkpoint without conversion or quantization. It
keeps two DiT blocks in memory and reads the next block from SSD while the GPU
runs the current one. On M5 Max, tracked DiT storage fell from about 36.5 GiB to
2.0 GiB at 512 square and 2.1 GiB at 864x480. A warm 50-block forward measured
1.35 versus 2.49 seconds at 512 square (84% slower), and 2.14 versus 2.68
seconds at 864x480 (26% slower). These are comparisons against the same
full-residency BF16 path, and the results were byte-identical in both checks.

The 2.0--2.1 GiB figure is the DiT's tracked tensor storage, not total system
RAM. Prompt encoding and the two VAEs run in separate phases rather than adding
their full peaks to it; the OS, media buffers, and output resolution still need
headroom. `--show` keeps a preview VAE resident and adds roughly 10 GiB, so omit
it for the lowest-memory run.

SSD streaming is an explicit memory/speed tradeoff and is not the default. It
cannot be combined with `--use-int8-row-fc2`.

### 3. Move toward reference quality

Change one control at a time when evaluating quality. First restore all layers,
then all denoiser evaluations, and finally raise the default 20-pass schedule
to the slower 50-pass reference:

```sh
./bin/h3cli --profile \
  -d ./models/MiniMaxH3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 \
  --frames 22 --steps 50 \
  --layers 50 --reuse 1 \
  -o outputs/fox-close.mp4
```

The defaults are `--steps 20 --layers 50 --reuse 1`; select `--steps 50`
or `--quality lossless` for this close path. It performs 50 complete 50-block denoiser
forwards and is much more expensive than the default, but is the right oracle
when a fast mode changes the subject, anatomy, motion, or composition.
Numerical pixel identity with MLX is not expected because the random-number and
execution engines differ; the depicted content and motion should agree.

### 4. Choose a speed/quality preset

Use `--quality LEVEL` to fill in generation defaults:

| Level | Steps | Reuse | Adaptive cache | Video decoder |
|---|---:|---:|---|---|
| `lossless` | 50 | 1 | off | full VAE |
| `extra-high` | 50 | 1 | off | full VAE |
| `high` | 50 | 1 on CUDA; 2 on Metal | conservative on CUDA; off on Metal | full VAE |
| `preview` | 12 | 2 | off | tiny preview VAE |
| `fast-preview` | 6 | 3 | off | tiny preview VAE |

All five keep 50 layers, core reuse 1, default attention and denoiser
quantization off unless explicitly overridden. Resolution, frame count, seed,
references and weight placement keep their existing defaults or supplied values.
Omitting `--quality` preserves the existing 20-step defaults.

The first three roughly follow [SGLang's H3 quality levels](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx#choose-the-quality-level):
`lossless` and `extra-high` currently share the same H3 denoise path; `high`
adds caching. Our CUDA `high` uses the native conservative policy (threshold
0.04, at most one consecutive hit, four warmup steps), not SGLang's Cache-DiT
implementation or its hardware-specific quality guarantee. Metal uses scheduled
reuse because native adaptive caching requires qualified SM120 CUDA hardware.
The names describe sampling presets, not MP4 compression or a cross-backend
bit-exactness guarantee. `--steps` still counts actual denoiser passes in h3cli.

Video compression defaults to CRF 18 independently of this preset. Select
`--output-quality`, override it with `--ffmpeg-crf 0..51`, or use
`--lossless-video` to preserve the delivered RGB frames exactly. See
[output encoding](features/output-encoding.md) for the SGLang mapping, color
conversion, audio settings, and lossless playback limits.

Explicit flags override preset defaults **regardless of argument order**:

```sh
./bin/h3cli -p "A cat walking along a sunlit beach, gentle waves." \
  --quality preview --steps 20 --reuse 1 --no-preview-vae \
  --width 640 --height 480 --frames 90 -o outputs/cat.mp4
```

If repeated, the last `--quality` selects the preset. The CLI prints the resolved
settings. `--no-preview-vae` selects the full decoder; `--preview-vae` enables the
tiny decoder. When both are supplied, the last decoder flag wins. The tiny model
must be installed as described in the [preview VAE guide](preview/preview-vae.md).

Presets apply to fresh generation, including continuation and bridge. Sampler
resume, decode-only, upscale and inspection modes reject `--quality`; use their
existing controls. Normal feature restrictions still apply: CUDA `high` inherits
adaptive-cache restrictions, including no first/last anchors or LoRA; use
`--adaptive-cache off` to disable it. Preview presets and Metal `high` use reuse,
which must be set to 1 when enabling adaptive cache or SubBlock. Still generation
requires reuse 1, full VAE and adaptive cache off. Presets do not automatically
weaken these checks or enable quantization, sparse attention or layer thinning.

These controls are independent unless noted otherwise:

| Control | Slow reference | Faster option | Aggressive | Main impact |
|---|---:|---:|---:|---|
| Denoising passes | `--steps 50` | `--steps 20` | `--steps 4..7` | The number always names actual denoising passes. |
| Whole denoiser reuse | `--reuse 1` | `--reuse 2` | `--reuse 3` | At 20 steps: 20, 11, or 8 fresh DiT evaluations. |
| Active DiT blocks | `--layers 50` | `--layers 45` | `--layers 40` | Fewer blocks reduce compute and resident transformer weights. |
| Core residual reuse | `--core-reuse 1` | `--core-reuse 4` | `--core-reuse 6` | Refreshes patch/head work every step but runs the expensive core less often. |
| Token reduction | off | optional | `--token-reduction` | Pairs horizontal video tokens inside middle blocks; faster but may change composition. |
| Internal canvas | output size | `384x384` for 512 square output | `320x320` | Runs DiT/VAE smaller, then upscales with vImage. |

On M5, `--use-int8-row-fc2` uses one activation scale per FC2 row and a single
full-width TensorOps product. It is optional because it is less numerically
conservative than grouped int8. It reduced complete denoiser forwards by about
2.6% in reciprocal tests. Matched four-step fox and surfer videos kept the same
subjects, setting, and motion (full-video SSIM 0.919 and 0.828).

`--reuse` and `--core-reuse` are mutually exclusive. Layer thinning can be
combined with either one.

To make the first command faster while keeping its output resolution, add
token reduction:

```sh
./bin/h3cli --profile \
  -d ./models/MiniMaxH3 \
  -p "A surfer riding inside a sharp blue ocean wave, one rider and one white board, realistic spray." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --reuse 2 --token-reduction \
  -o outputs/surfer-fast.mp4
```

At the validated 512 square shape, token reduction cut the `45 layers + reuse
2` denoise profile from 16.69 to 12.60 seconds on the IT M5 Max. Independent
fox and surfer renders stayed coherent, but composition can diverge more from
the close path.

For an aggressive preview, render internally at 320 square and upscale to the
requested 512 square output:

```sh
./bin/h3cli --profile \
  -d ./models/MiniMaxH3 \
  -p "A red fox walking through snow, realistic, tracking shot." \
  --width 512 --height 512 \
  --render-width 320 --render-height 320 \
  --frames 22 --steps 20 --layers 40 --reuse 3 \
  -o outputs/fox-aggressive.mp4
```

This combination produced a clean, recognizable 22-frame fox in validation,
but loses fine detail and can change framing. Do **not** add `--token-reduction`
to both `--layers 40` and `--reuse 3`: that tested combination produced color
ringing, outlines, and ghosted limbs.

As an alternative to whole-velocity reuse, this keeps the timestep-dependent
patch and output heads fresh at every transition:

```sh
./bin/h3cli --profile \
  -d ./models/MiniMaxH3 \
  -p "A surfer riding a blue ocean wave." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --core-reuse 4 \
  -o outputs/surfer-core-reuse.mp4
```

Use `--core-reuse 6` only as an aggressive preview. Values above 6 are not
exposed because validation lost subject fidelity.

### 5. Pick resolution and duration

Width and height must each be multiples of 32, at least 32, and their product
must not exceed `768 * 1344` pixels. Those are mechanical limits, not a promise
that every tiny canvas has good model quality. H3-Base is a 768p model.

| Canvas | Current guidance |
|---|---|
| `512x512` | Safest development size; repeatedly validated with multiple prompts. |
| `768x768` | Validated close-quality square output; substantially more expensive. |
| `1344x768`, `768x1344` | Released 768p-class landscape/portrait limit. |
| `1024x768`, `768x1024` | Valid 4:3 and 3:4 768p-class canvases. |
| `384x384` internal to `512x512` | Validated fast-quality scaling point. |
| `320x320` internal to `512x512` | Validated aggressive scaling point. |
| `256x256` | Native fast-preview canvas with automatic low-resolution RoPE adaptation. |

For a fast native 256-square preview:

```sh
./bin/h3cli -d ./models/MiniMaxH3 \
  -p "A red fox walks through fresh snow in a pine forest." \
  --width 256 --height 256 \
  --frames 22 --steps 20 \
  --layers 50 --reuse 1 \
  -o outputs/fox-256.mp4
```

At 256 square, H3 has only an `8x8` effective spatial-token grid, so it has less
room for fine detail and complex composition. H3 automatically halves spatial
RoPE coordinates at exactly 256 square. This removed repeating lattice
artifacts in long fox renders and stayed coherent on an independent portrait,
without adding tokens or runtime. Use `--use-reference-rope` to restore the
released/MLX coordinates for parity checks. Keep token reduction off at this
size. Native 128 square remains unsupported: its `4x4` token grid did not
recover a recognizable subject even with adjusted RoPE.

`--render-width` and `--render-height` must be set together, must have the same
aspect ratio as the output, and cannot exceed the output dimensions. The model
and VAE use the internal size; terminal frames and the encoded video retain the
requested output size.

H3 emits 24 fps and aligns frame requests upward to `5 + 17*n`:

Use `--seconds N` for a duration-oriented request, or `--frames N` for direct
frame control; the two options are mutually exclusive. Fractional seconds are
accepted. Seconds are converted at 24 fps and then rounded upward to the next
legal H3 temporal shape, so `--seconds 10` produces 243 frames (10.125 seconds).

| Frames | Approximate video duration |
|---:|---:|
| 22 | 0.917 seconds |
| 39 | 1.625 seconds |
| 56 | 2.333 seconds |
| 107 | 4.458 seconds |
| 243 | 10.125 seconds |
| 362 | 15.083 seconds |

Short clips are useful for development. The released workflow is intended for
roughly 4–15 second videos. A request such as `--frames 23` is rounded up to 39
frames rather than producing an arbitrary temporal shape.

### 6. Improve the prompt

A short prompt works, but the released system expects a Context-IR-like
description. State the subject, action, setting, camera, lighting/style, and
desired sound. For example:

```text
Scene: a single red fox in a snow-covered pine forest at dawn.
Action: the fox walks steadily left to right and looks toward the camera once.
Camera: medium-height lateral tracking shot, 50 mm lens, stable framing.
Look: photorealistic fur, cold blue ambient light, warm sunrise rim light.
Audio: soft footsteps in snow, light wind through pine branches, no music.
```

Keep identity and object counts explicit when they matter. `--seed N` controls
the native random stream; the default is 42. Compare options with the same
prompt, seed, resolution, frame count, and step count.

### 7. Preview frames and diagnose performance

- `--show` displays a representative frame after every denoising transition,
  followed by all frames from the completed video. Like Iris, it advertises 2x
  display dimensions by default for Retina terminals; `--zoom N` changes that
  factor without resizing the generated video or the encoded terminal image.
- Live previews show the current clean estimate by default on CPU and GPU.
  Early predictions may change substantially; final output is unaffected.
  `H3_PREVIEW_MODE=noisy` restores historical noisy previews. See
  [denoised previews](preview/denoise.md) for state semantics and validation.
- `--frames-dir DIR` writes final callback frames as PPM files. Intermediate
  `--show` previews are not written there.
- `-o ''` disables MP4 encoding; combine it with `--frames-dir` when FFmpeg is
  unavailable.
- `--profile` reports phase wall time, Metal encoding/wait time, peak live
  tensor storage, cumulative allocation, and dispatch counts.

For example:

```sh
./bin/h3cli --profile -d ./models/MiniMaxH3 -p "A hummingbird hovering over red flowers." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --reuse 2 --frames-dir outputs/hummingbird-frames \
  -o ''
```

### 8. Add image, video, and audio references

`--ref-image-size match|high|max` applies to every image reference on Metal and
CUDA, for both videos and stills. The Qwen encoder and visual VAE receive the
same canvas, preserving aspect ratio subject to 32-pixel alignment:

| Mode | Sizing |
| --- | --- |
| `match` (default) | Downscale to the internal render canvas's pixel area when needed; retain the existing alignment/minimum behavior. |
| `high` | Scale the long edge to 2048, including upscaling small sources. |
| `max` | Scale the short edge to 2048, including upscaling; the long edge can reach 8192. |

`high` and `max` accept source aspect ratios from 1:4 through 4:1 and round
scaled dimensions to the nearest multiple of 32, with ties to even. For example,
a 640×480 image becomes 2048×1536 with `high` or 2720×2048 with `max`. They do
not depend on the output resolution. Metal and CUDA stills now use the same
`max` geometry as CUDA video; their former down-only behavior has changed.

Each image may contain up to 65,536 raw vision patches (16,384 merged tokens).
There is no fixed aggregate patch cap: valid multiple images can exceed 32,768
or 65,536 patches in total, subject to checked tensor/index and available-memory
limits. Existing limits of nine images and twelve total references remain.
CUDA projection packing uses at most 512 MiB of input scratch, plus its 18 MiB
filter; encoder activations and other stages require additional memory. Metal
uses bounded tiled causal attention for Qwen sequences beyond its direct
threadgroup-memory capacity, and bounded dense BF16 vision attention above
32,768 patches per image. References are never silently reduced to fit.

Fresh conditioning caches distinguish `high` and the new Metal/still `max`
geometry. Prepared sampler checkpoints keep their stored reference geometry
without reopening media. Current schema and sampler
build/model compatibility checks still apply. Upscaling preserves
stored `high`/`max` canvases; `match` keeps its target-dependent retargeting.
First/last-frame anchors, reference videos/audio and output dimensions retain
their existing policies. See the [design](features/design-reference-image-size.md)
and [validation record](features/reference-image-size-results.md).

First/last-frame anchors select the FL2VA path:

```sh
./bin/h3cli -d ./models/MiniMaxH3 -p "The fox keeps walking through the snow." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --reuse 2 \
  --first-frame fox.png --last-frame fox-later.png \
  -o outputs/fox-anchored.mp4
```

Ordered references select the distinct Ref2VA checkpoint. Use the flag matching
the media semantics:

Video references use the sole `released-v1` path: independent 17-frame VAE
chunks, posterior sampling with seed 42, FP16 rounding, and released condition
rows. A 56-frame VAE prefix produces 17 latents. See
[Ref2VA video preprocessing](minimax/refvideo.md). Numerical acceptance
comes from the unchanged 204-output [recorded SGLang gate](../CONTRIBUTING.md).

```sh
# One image reference.
./bin/h3cli -d ./models/MiniMaxH3 -p "Use the animal and setting in the reference." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --ref-image fox.png -o outputs/fox-reference.mp4

# Continue a clip but ignore its soundtrack.
./bin/h3cli -d ./models/MiniMaxH3 -p "Continue the motion in this clip." \
  --width 512 --height 512 --frames 124 --steps 20 \
  --ref-silent-video fox.mp4 \
  -o outputs/fox-video-reference.mp4

# Preserve the clip's embedded audio.
./bin/h3cli -d ./models/MiniMaxH3 -p "Continue this audiovisual scene." \
  --width 512 --height 512 --frames 56 --steps 20 \
  --ref-video fox-with-audio.mp4 \
  -o outputs/fox-video-audio.mp4

# Replace a video's soundtrack explicitly.
./bin/h3cli -d ./models/MiniMaxH3 -p "Continue the scene with the supplied music." \
  --width 512 --height 512 --frames 56 --steps 20 \
  --ref-video-audio silent-fox.mp4 replacement.wav \
  -o outputs/fox-replaced-audio.mp4

# An ordered image plus standalone audio reference.
./bin/h3cli -d ./models/MiniMaxH3 -p "Use the animal and music from the references." \
  --width 512 --height 512 --frames 56 --steps 20 \
  --ref-image fox.png --ref-audio music.wav \
  -o outputs/fox-image-audio.mp4
```

Reference flags may be repeated and their command-line order is preserved.
Standalone audio must accompany an image or video reference. Audio references
must be 2–15 seconds; at most three audio inputs are accepted and their total
decoded duration is capped at 15 seconds.
