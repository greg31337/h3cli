# Single-still rendering

`--still` generates one PNG directly at the requested canvas using the ordinary
FL2VA or Ref2VA weights and the stock video VAE. It samples one video latent
frame and two auxiliary audio positions; audio is discarded. After sampling,
the decoder repeats the latent five times and returns raw frame 3. See the
[sampling contract](single-still-generation-contract.md) and
[qualification record](empty-still-results.md).

```sh
./bin/h3cli -d models/MiniMax-H3 --still \
  -p 'A polished red ceramic teapot on a pale wooden table, soft studio lighting.' \
  --width 3872 --height 2176 --steps 50 --seed 42 \
  --save-still-latent outputs/teapot.safetensors -o outputs/teapot.png

./bin/h3cli -d models/MiniMax-H3 --still \
  --ref-image inputs/2.jpg --ref-image-size high \
  -p 'A portrait photograph of the person in image 1 in a sunlit garden.' \
  --width 3872 --height 2176 --steps 50 --seed 42 -o outputs/portrait.png

./bin/h3cli -d models/MiniMax-H3 --offline \
  --decode-still-latent outputs/teapot.safetensors -o outputs/teapot-decoded.png
```

Use `--still-count N` with `--still` to generate several images from the same
prompt and references, using seeds `SEED` through `SEED + N - 1`:

```sh
./bin/h3cli --still --still-count 3 --seed 42 \
  --width 640 --height 480 --steps 50 \
  -p 'A polished red ceramic teapot on a pale wooden table.' \
  --save-still-latent outputs/teapot.safetensors -o outputs/teapot.png
```

This writes `teapot-seed-42.png`, `teapot-seed-43.png`, and
`teapot-seed-44.png` under `outputs/`, with matching `-seed-N.safetensors`
files when `--save-still-latent` is supplied. The omitted count and
`--still-count 1` preserve the original filenames. Counts must be positive
integers, and the final seed must fit an unsigned 64-bit integer.

The batch prepares the prompt, image references, transformer, and sampling
schedule once. It samples each seed into an anonymous temporary file, releases
the transformer, and loads the stock VAE once to decode all images. PNG delivery
starts after every latent has been sampled. Transformer and decoder allocations
remain separate, and only one image's latent and decoded pixels are needed in
RAM at a time. Temporary disk use grows by approximately
`N * 24 * (width / 16) * (height / 16) * 4` bytes and is cleaned up when the batch
finishes or fails. Requested latent files and completed PNGs remain available
if a later image fails.

CUDA batches also retain partial/streamed transformer placement between seeds,
provided the weight options and memory headroom remain compatible.

`--frames-dir DIR` writes `frame-0000.ppm`, `frame-0001.ppm`, and so on in seed
order. `--still-count` is a native CLI option; video render scripts and HTTP
jobs do not accept it.

Canvas dimensions must be positive multiples of 32, at most 16,777,216 pixels
and 8192 per side. Both 3872×2176 and 2176×3872 are admitted. These are genuine
T=1 target canvases; no smaller render or final upscaler is used. Device memory
still determines feasibility; consult the measured limits before a large run.
`--frames` is unnecessary; explicit `--frames 1` is accepted.

Ordered image references retain the shared [reference-size rules](design-reference-image-size.md).
`match` follows the target, `high` fixes the long edge at 2048, and `max` fixes
the short edge, including upscaling. Reference dimensions and target dimensions
are separate. Larger references can substantially increase conditioning cost.

`--show` displays the final image, `--frames-dir DIR` writes one final PPM, and
`--profile` reports phase timing and memory. PNGs and saved latents use atomic
publication. A cancelled final callback or failed PNG encoder preserves an
existing PNG; a saved latent remains usable if later delivery fails.

The sampler remains h3cli Euler with its existing shifted flow schedules and
seed semantics. The ComfyUI example uses `er_sde`/`simple`, so copying its seed
and step count does not imply identical images. Quality and identity observations
are recorded separately from numerical and lifecycle checks.

Still mode rejects duration, anchors, audio/video references, continuation,
sampler/AV checkpoints, conditioning files, a separate render canvas, video
previews, acceleration/quantization/reuse and reduced-layer execution. `--render`
continues to execute video scripts. LoRA behavior is unchanged and not qualified
by the still examples.

## Models and migration

Generation resolves the selected model's stock `video_vae` and never downloads
an image VAE or audio VAE. Codec-only decode reads the saved model variant and
resolves only its three video-VAE/configuration files; it needs no tokenizer,
DiT or audio weights. Explicit `-d`, default model roots and `--offline` follow
the [model-download guide](model-downloads.md).

`--image-vae` and `--download-models image-vae` are retired and return migration
guidance. Existing user checkpoint files are left in place. The schema-1 latent
format is rejected; regenerate with `--still` to produce schema 2. The new stock
VAE is a different decoder and does not promise the old image's pixels.

Schema 2 stores finite normalized F32 `[1,24,1,H/16,W/16]`, model variant,
normalization compatibility and an explicit five-copy/frame-3 decoder recipe.
The decoder validates these before weight allocation. There is no public raw
frame or tile-size override.

## Library use

Initialize `h3_params` with `H3_PARAMS_DEFAULT`, set `still=1`, `frames=1`, the
canvas and normal model directory, then call `h3_generate`. Leave the deprecated
`image_vae` field NULL; a non-NULL value returns migration guidance.

```c
h3_ctx *ctx = h3_load_dir("models/MiniMax-H3");
h3_params p = H3_PARAMS_DEFAULT;
p.still = 1; p.frames = 1; p.width = 640; p.height = 480;
p.steps = 6; p.output_path = "output.png";
p.save_still_latent = "still.safetensors";
h3_result *result = h3_generate(ctx, "A red ceramic teapot.", &p);
h3_result_free(result);
h3_free(ctx);

h3_decode_options options = {.output_path = "decoded.png"};
char error[512];
result = h3_decode_still_latent("still.safetensors", "models/MiniMax-H3",
                              &options, error, sizeof(error));
h3_result_free(result);
```

The decode API's second argument is now the main model directory, replacing the
retired image-checkpoint filename. Rebuild clients and update that argument.
Successful results have `kind=H3_RESULT_STILL`, `frames=1`, `fps=0`, no AV/sampler
state and zero audio fields. `h3_result_free` releases the owned still latent.
Final RGB24 callbacks borrow pixels for the call and receive index 0/count 1/
step -1 exactly once; they can cancel before PNG publication. A library caller
may omit the output path and consume the callback. Low-level still decoding is
in `video_vae.h`; `image_vae.h` retains geometry/artifact helpers.

The [original results](single-still-results.md),
[CUDA results](single-still-cuda-results.md), and
[dedicated-VAE provenance](single-still-sources.md) remain dated historical evidence.
