# Single-still rendering

`--still` renders one image using an actual T=1 H3 target. Prompt-only FL2VA
and ordered image-reference Ref2VA requests are implemented. The initial path
uses original BF16 weights, dense attention, all 50 blocks, reference RoPE and
the ordinary Euler sampler. See the
[measured results](single-still-results.md) and [sampling contract](single-still-generation-contract.md). Default CUDA execution
is also [qualified on RTX PRO 6000 Blackwell](single-still-cuda-results.md).

## Image checkpoint

`--image-vae` defaults to
`ROOT/image-vae/minimax_h3_t1_image_vae_step1597.safetensors`, where
`--models-path ROOT` defaults to cwd `models`. Missing weights download
automatically for `--still` and `--decode-still-latent`. Explicit file overrides
win. Prefetch with `--download-models image-vae`; use `--offline` to require local
files. See the [download guide](model-downloads.md) and
[pinned provenance](single-still-sources.md). The qualified full-file checksum is
`6c3d0bfa055986a803a566a862fcde283a1e63db62829e5ef4a2a5aebf50bb86`.

```sh
shasum -a 256 models/image-vae/minimax_h3_t1_image_vae_step1597.safetensors
# 6c3d0bfa055986a803a566a862fcde283a1e63db62829e5ef4a2a5aebf50bb86
```

The decoder checks capability, format, architecture, all 562 tensor shapes,
whitening and the declared output slice. It expands F16 checkpoint values to
F32 with bounded conversion buffers and uses the released 256/64 spatial tiles.
Expanded decoder weights and scratch use about 9.75 GB on Metal. The existing
110 GB default memory guard remains active.

## Render an image

Build normally with `make -j8`. These commands use the original model and
FFmpeg's PNG encoder:

```sh
./bin/h3cli -d models/MiniMax-H3 --still \
  -p "A polished red ceramic teapot on a pale wooden table, a single still photograph, soft studio lighting." \
  --width 640 --height 480 --steps 6 --seed 42 \
  --save-still-latent outputs/teapot.safetensors -o outputs/teapot.png

./bin/h3cli -d models/MiniMax-H3 --still \
  -p "A portrait photograph of the woman in <Picture 1>, smiling softly, soft studio lighting, neutral background." \
  --ref-image inputs/2.jpg --ref-image-size match \
  --width 640 --height 480 --steps 6 --seed 42 -o outputs/portrait.png
```

`--frames` is unnecessary; an explicit `--frames 1` is accepted. Canvas dimensions
must be multiples of 32 and within the existing 1,032,192-pixel limit. Reference
image order and `match`/`high`/`max` sizing use the shared
[reference-image rules](design-reference-image-size.md): `high` fixes the long
edge to 2048 and `max` fixes the short edge, including upscaling. Six evaluations
were used for qualification; no 50-step generation quality claim is made.

Add `--show` for the final terminal image, `--frames-dir DIR` for one final PPM,
and `--profile` for loading, denoising and image delivery measurements. PNGs and
saved latents are staged and renamed after successful writing. A failed PNG
encoder or cancelled final callback preserves an existing destination.

The target has one video latent frame and two auxiliary audio ticks (four
packed rows). Audio is sampled with the existing audio schedule and discarded;
no audio VAE, audio output or MP4 is created. The DiT is freed before image VAE
loading. This is not a hidden short-video render.

Initial still mode rejects duration, first/last anchors, audio/video references,
continuation/bridge settings, sampler/AV checkpoint options, conditioning files,
a separate internal canvas, preview VAE and live video preview. SOL, native
Metal, ANE, CUDA Sage/quantization, Q8, reuse and reduced-layer execution are
also rejected for this baseline. Explicit LoRA selection retains its existing
provenance but is not qualified by these examples.

## Decode a saved still

This command needs neither `-d` nor tokenizer, DiT or audio weights:

```sh
./bin/h3cli --decode-still-latent outputs/teapot.safetensors \
  -o outputs/teapot-decoded.png --profile
```

The artifact contains exactly one finite F32 tensor `latent` with shape
`[1,24,1,H/16,W/16]`, schema `1`, latent space `h3-normalized-v1`, and an
encoder/whitening compatibility digest. Untyped tensors and incompatible
checkpoints fail before decoder weight allocation. The decoder unwhitens once
and uses the metadata slice (3 for this checkpoint); there is no public slice
or larger-tile override.

## Library use

In C, initialize `h3_params` with `H3_PARAMS_DEFAULT`, set `still=1`, `frames=1`
and `image_vae`, then call `h3_generate`. `h3_decode_still_latent` is the
codec-only entry point. Use `src/vae/image_vae.h` for the low-level image encoder,
decoder and still-latent interchange functions.

A successful still result has `kind=H3_RESULT_STILL`, `frames=1`, `fps=0`, no
AV/sampler state and zero audio fields. It owns `still_latent`; `h3_result_free`
releases it. Final RGB24 callbacks borrow their pixels for the call, receive
index 0/count 1/step -1 exactly once, and may cancel before PNG publication.
A library caller may omit an output path and consume only the callback/result.
Video result kind remains zero. Rebuild C clients for the extended structs;
existing `.h3av`, `.h3sample` and video presentation formats are unchanged.
