# Latent upscaling

Run the examples from the repository root. See the [documentation index](../README.md)
for related saved-state and continuation workflows.

Latent upscaling is a regular feature on CUDA and Metal. Save a completed dense
BF16 segment before decoding, then upscale its video
latent while preserving the clean audio. The native learned network doubles
both spatial dimensions, preserves time, and uses the pinned
[LBH BF16 safetensors artifact](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/blob/3f941d5d182014dd5c0a5e16330420ee2d4aa0c6/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors).
Its SHA-256 must be
`4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6`.
Native inference does not require Python, Torch or ComfyUI.

Temporal support follows the source backend: CUDA up to 736 frames, Metal up
to 362, including K=0. Sequence and memory checks still apply; 736 frames at a
3840×2176 target exceed the sequence ceiling. Long upscale execution is not
runtime-qualified by the bounded duration campaign; see the
[duration contract](../cuda/long-video-contract.md).

With `--models-path ROOT`, the upscaler defaults to
`ROOT/latent-upscale/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`.
`ROOT` defaults to cwd `models`. Missing weights download automatically;
an explicit `--upscale-model` path takes precedence.

```sh
mkdir -p outputs/upscale
./bin/h3cli -d models/MiniMaxH3 -p "A pianist plays in a sunlit concert hall." \
  --width 672 --height 384 --frames 90 --steps 50 --seed 42 \
  --save-upscale-state outputs/upscale/source.h3up --state-only

./bin/h3cli --inspect-upscale-state outputs/upscale/source.h3up

./bin/h3cli -d models/MiniMaxH3 \
  --upscale-state outputs/upscale/source.h3up \
  --upscale-refine-steps 4 --upscale-noise 0.25 \
  --save-av-state outputs/upscale/final.h3av -o outputs/upscale/final.mp4
```

Omit `--state-only` during source generation to also decode a source preview.
The source bundle is written before decoding and survives a later delivery
failure. A later upscale does not reopen reference images, videos or audio;
it uses the saved raw conditions and frozen source text/vision semantic view.
FL2VA first/last anchors and Ref2VA sizing/order/audio pairing are retained.
Moving bundles is supported, but the H3 model installation must still match
its saved local metadata identity. Relocating the model changes that identity.

`--upscale-refine-steps` accepts **0, 2, 3 or 4**, default 4. Starting video
sigma is independent of K and the original source schedule; its default is
0.25 and its supported range is `(0,0.5]`. K=0 decodes the learned transfer
directly, performs no noise draws or transformer evaluations, and rejects an
explicit `--upscale-noise`. `--upscale-seed` defaults to the source seed. This
is h3cli's native latent-upscaling recipe; official MiniMax Regenerate-2K parity
is not claimed.

Fresh `.h3up` upscaling restores prompt, geometry and conditions from the source.
Quantization, reuse/adaptive cache, SubBlock, LoRA/Turbo, continuation/bridge,
token reduction and preview VAEs are excluded from the supported dense BF16
scope. Targets double both aligned source axes. Metal retains its ceiling of
1,920 pixels per axis and 2,088,960 total pixels; 960×544 becomes **1920×1088**.
CUDA accepts sources through **1920×1088** and outputs through **3840×2176**,
including transpose (8,355,840 pixels, at most 3840 per axis).
CUDA ordinary generation also accepts the new source canvas. Resource admission can still
reject a valid canvas that does not fit available memory.

For a completed native `.h3av` with its current `.presentation` sidecar, use
the separate zero-refinement path:

```sh
./bin/h3cli --models-path /path/to/models \
  --upscale-av-state outputs/source.h3av --upscale-refine-steps 0 \
  --save-av-state outputs/upscaled.h3av -o outputs/upscaled.mp4
```

This path also accepts dense hard/bridge continuation results. It upscales the
complete video latent with its temporal context, copies audio exactly, and
retains the saved prefix trim for final delivery. It loads only the upscaler
and VAEs; it needs no prompt, references, text encoder or DiT. It requires the
original model installation and backend, a full-VAE native source with an exact
render canvas, and no previous upscaling. Refinement, noise/seed overrides and
sampler import are unsupported for this input type. `--state-only` can save the
larger AV state without decoding. Keep the original AV state for continuations.

In a [render script](sequence-rendering.md#upscaling-before-merging), file-wide
`#upscale 0` runs this transfer for every segment before merging. Independent
segments can instead use `#upscale 2`, `3` or `4` via `.h3up` source capture;
any continuation combined with nonzero refinement is rejected before rendering.

For a short maximum-canvas job on the 72 GB test GPU, use `--frames 22` during
source generation and `H3_CUDA_WEIGHT_MODE=stream` during refinement and resume.
`--upscale-refine-steps 0` skips the transformer and its noise initialization;
it still runs the learned spatial network and full VAE. Source generation is a
separate cost. See the [canvas contract](../cuda/large-canvas-contract.md) and
[phase memory estimates](../cuda/large-canvas-memory-estimates.md). Long-video
estimates do not establish measured long-video support on a 72 GB card.

To pause refinement, add `--stop-after-step 1 --save-sampler-state
outputs/upscale/refine.h3sample --state-only` to a fresh upscale and omit
`--save-av-state`. Boundary zero can be saved without loading the transformer.
Resume with only the checkpoint and H3 model:

```sh
./bin/h3cli -d models/MiniMaxH3 \
  --resume-sampler-state outputs/upscale/refine.h3sample \
  --save-av-state outputs/upscale/final.h3av -o outputs/upscale/final.mp4
```

Resume restores the complete recipe, transformed conditions and noise. It
requires the original backend/runtime and does not need the upscaler weights
or source bundle. Cancellation saves the latest committed boundary when a
sampler output path is requested. For upscale/resume without decoding, use
`--state-only --save-av-state PATH`. Completed AV files can be decoded later
with the ordinary `--decode-av-state` command and their `.presentation` sidecar.

`--upscale-import-sampler` explicitly imports a completed current dense
text-only CUDA `.h3sample` using the metadata identity contract. Incomplete or
conditioned samplers, unsupported schema versions, bare AV states and MP4
files are rejected by this import route. Capture a `.h3up` source for refinement,
or use `--upscale-av-state` with a matching presentation for zero refinement. See the
[source/refinement format](upscale-source-format.md) and
[qualification evidence](../experiments/latent-upscale-qualification.md).

Both 90-frame target canvases passed native CUDA and Metal probes, including
full tiled decoding and exact audio preservation. The qualified Metal machine
is an M4 Max with 128 GiB unified memory; the 1920×1088 probe peaked at about
93 GB process footprint. Geometry admission alone does not guarantee sufficient
memory. The [fixed comparison and playback](../experiments/latent-upscale-results.md)
report U0/U2/U4 end-to-end costs of 187/233/261 seconds at 1344×768 (direct:
767 seconds), and 365/467/550 seconds at 1920×1088 (direct: 2144 seconds).
These single-observation costs include source generation; they do not establish
equal quality. U0 is a useful low-cost starting point, with U2 for evaluating
refinement. The experiment does not establish a visual advantage for U4 over
U2. The user accepted the fourteen-video comparison; the
[acceptance record](../experiments/latent-upscale-acceptance.json) binds that
review to the exact artifacts. Approximation/quantization and other excluded
compositions remain unsupported.
