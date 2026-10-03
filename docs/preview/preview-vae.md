# Fast video previews and saved-latent finalization

`--preview-vae` selects Ollin's approximate TAEH3 video decoder for the final
MP4 and live `--show` frames. The original VAE remains the default. Native
**Metal and CUDA** backends are implemented, tested on an M4 Max with macOS
26.6.2 and an RTX 5090 with CUDA 12.8 in the original qualification. Current CUDA
video derives its single arithmetic recipe automatically; preview selects only
the decoder.

Sampling, reference encoding, audio decoding, seeds, adapters, and continuation
latents are unchanged. Tiny reconstruction is useful for assessing a scene,
composition and motion, with less reliable fine detail. It does not make a
low-step sample into a higher-step sample. The user separately accepted the
reviewed Metal and RTX 5090 tiny output on 2026-09-18. See the
[Metal report](preview-vae-validation.md) and
[RTX 5090 report](preview-vae-cuda-validation.md) for the accepted corpora.

## Obtain the optional model

The normal build needs neither tiny weights nor a Python inference runtime.
`--preview-vae` downloads the pinned decoder automatically when missing. To
provision it separately:

```sh
./bin/h3cli --download-models preview
```

The default is `ROOT/preview-vae/taeh3.safetensors`, where `--models-path ROOT`
defaults to cwd `models`. Override the exact file with `--preview-vae-model`.
Native downloads verify size/SHA-256 and resume interrupted transfers. Use
`--offline` or `H3_OFFLINE=1` to require existing local files; malformed or
incompatible decoders still fail without a silent fallback. See the
[download guide](../features/model-downloads.md).

- Upstream: `madebyollin/taehv`, revision
  `62f7591f59dfbb4c3c02b7a621d180a9eeaba26c`.
- Complete file SHA-256:
  `4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13`.
- Architecture: 64 decoder tensors, 9,868,236 parameters, 19,736,472 bytes of
  F16 decoder weights. The checkpoint's encoder is unused.
- License: [Ollin's MIT license](../../third_party/taeh3/LICENSE).

The native loader accepts that exact F16 architecture and records the actual
file hash, permitting an explicitly supplied compatible checkpoint. Similar
names on unrelated preview models do not establish compatibility.

## Generate a preview, then finalize it

```sh
./bin/h3cli -d models/MiniMax-H3 --preview-vae \
  --ref-image inputs/2.jpg -p 'The person in <Picture 1> smiles and waves.' \
  --width 288 --height 384 --frames 56 --steps 6 \
  --save-av-state outputs/draft.h3av -o outputs/draft.mp4

# Reconstruct the same final latents with the original VAE, without denoising.
./bin/h3cli -d models/MiniMax-H3 --decode-av-state outputs/draft.h3av \
  -o outputs/final.mp4

# Or replay them through the tiny decoder.
./bin/h3cli -d models/MiniMax-H3 --decode-av-state outputs/draft.h3av --preview-vae \
  -o outputs/draft-again.mp4
```

Tiny decoding is enabled by `--preview-vae`, or by the `--quality preview` and
`--quality fast-preview` generation presets. `--no-preview-vae` overrides either
preset to use the full decoder.
Full finalization reconstructs the same sample; it does not improve a low-step
denoising result.

Keep both `draft.h3av` and `draft.h3av.presentation`. Saving remains explicit;
ordinary generation does not automatically retain files. Live previews use
the same `H3_PREVIEW_MODE=denoised|noisy` choice as before and never become
sampler state. Select the decoder with `--preview-vae` and its checkpoint with
`--preview-vae-model`.

Pause/resume may change the reconstruction policy: pass `--preview-vae` on
resume to request it. The checkpoint restores sampling settings, while the
current invocation supplies decoder/display settings. `--preview-on-stop`
remains silent and does not create a completed `.h3av`.

Decode-only rejects prompts, sampling controls, continuation inputs, and
sampler checkpoints. It does not initialize tokenizer, text/vision encoders,
or DiT. Compatibility verification reads the existing decoder weights and
configuration metadata to verify their identity; this startup cost is included in
CLI total wall time, even though unrelated model weights are never loaded.

## Current states and presentation metadata

Decode requires AV schema 3 and its matching presentation schema 8. Missing,
malformed, stale or old metadata is rejected; there is no sidecarless delivery
or old-format migration option. See the
[current state contract](../features/current-state-contract.md).

The sidecar explicitly records the AV fingerprint, model variant, internal and
output dimensions, video/audio prefix trims, codec recipe, precision, attention,
approximation warmups, geometry profile and upscale provenance. Codec recipes
1 (Metal) and 2 (CUDA) remain active backend delivery contracts. Current
finalization preserves saved output size and trims continuation history.

Each file is staged and replaced atomically. A two-file rename is not a single
filesystem transaction: an interruption between replacements can leave a
mismatched pair. The fingerprint prevents accepting that pair; retry saving
both artifacts from the result. Old saved schemas are intentionally unsupported.
Output MP4s are also staged and published only after FFmpeg succeeds; cancellation
and encoding failures remove partial media and preserve an existing target.

## API and execution limits

Set `h3_params.preview_vae=1` and optionally `preview_vae_model` for generation.
Zero/default selection keeps the original decoder. Use
`h3_result_save_av_state()` to save a result with presentation metadata;
`h3_av_state_save()` remains the original state-only API.
`h3_decode_av_state(model_dir, state_path, &options, error, size)` returns an
ordinary owned `h3_result`; release it with `h3_result_free()`. Decode options
provide output path, decoder selection, and frame/progress callbacks. Nonzero callbacks cancel at safe boundaries.

The decoder takes normalized `[24,T,H,W]` diffusion latents directly. Its
trained `3*tanh(x/3)` transform and `[0,1]` RGB convention replace the full
VAE's own normalization only at reconstruction. Temporal memory is retained
at every residual memory block within a decode and reset between clips or
denoising previews. Frame selection uses global expanded indices with
`index % 20 >= 3`: 7/17/27 latent frames deliver 22/56/90 frames. Production
supports the existing 22..362-frame generation range and canvas limit.

Metal executes the network in FP16 through MPSGraph. Weights and at most two
batch/geometry graphs are cached; input storage is reused. Production batches
at most five latent frames and transfers only their completed RGB frames.
Live previews evaluate all preceding temporal context before selecting the
representative frame. No isolated latent slice is treated as a whole clip.

CUDA uses FP16 activations/weights and FP32 convolution accumulation, with a
separate stream, reusable activation/history buffers and pinned batch readback.
The existing optional `CUDA_CUDNN=1` build enables cuDNN convolutions; an ordinary
CUDA build uses tensor-core cuBLAS convolutions with bounded patch storage.
Python and cuDNN are not required by the fallback. Both routes preserve the
same temporal and color contract and stream one RGB batch at a time.

Advanced CUDA controls, read when the decoder is loaded:

- `H3_PREVIEW_CUDA_CONV=auto|cublas|cudnn`: `auto` uses available supported
  cuDNN plans and otherwise cuBLAS. `cublas` forces the fallback; `cudnn` requires
  cuDNN and a supported plan, reporting an error otherwise.
- `H3_PREVIEW_CUDA_WORKSPACE_MB=1..128`: convolution scratch limit, default
  32 MiB. This is not the total decoder-memory limit. At most 64 convolution
  plans are cached. Device/model/policy changes invalidate the decoder cache.

Streaming delivery converts/resizes/trims each bounded batch before callbacks
and FFmpeg. The full decoder's decode-only path retains its released spatial
tiles and blends temporal overlaps before emitting committed frames. Audio
is decoded normally and streamed concurrently through a bounded pipe, avoiding
video/audio pipe deadlock. Original generation retains its existing buffered
delivery path.

Memory preflight uses a conservative geometry-dependent activation allowance,
not the original decoder's 12-GiB loading reserve. `--profile` reports decoder
load, graph builds, compute/transfer time, observed Metal allocated bytes,
conversion, pipe writes, mux completion and first delivered frame latency.
Metal allocated bytes are a device-wide observation and can include other
cached components; they are not a precise isolated process footprint. See
[the validation report](preview-vae-validation.md) for measured boundaries,
results and outstanding coverage. The [CUDA report](preview-vae-cuda-validation.md)
separately records decoder-owned allocations, device-wide observations and
pinned host storage. Fast CUDA does not promise pixel-identical results across
separate processes; source latents, presentation and audio retain their contracts.

## Reproduce the bounded checks

```sh
make test-preview-vae                 # host-only, no model required
make bin/preview_vae_generate
python3 tests/preview_vae_run.py --name local-invariance --timeout 300 -- \
  ./bin/preview_vae_generate models/MiniMax-H3 outputs/preview-vae/check invariance
```

The persistent ledger is `outputs/preview-vae/metal/budget.json`. It reserves
the timeout before launching work, includes failed runs, and refuses to
exceed the local 1,200-second allowance. Do not reset it to bypass the budget.
Component cases use at most 120 seconds; complete workflows at most 300.
Python/MLX reference replay and the gallery scripts are development tools;
the native binary does not depend on them. No production-length `test2.sh`
run is required for this feature.

The RTX 5090 round uses its own persistent ledger at
`outputs/preview-vae/cuda-5090/budget.json`, with a 1,800-second limit, including
failures and instrumentation. Invoke the same runner with that `--ledger` and
`--budget 1800`; never reset the ledger to extend testing. Both CUDA convolution
routes and a build without cuDNN are covered. Other CUDA devices remain
unqualified for this new decoder. Full-VAE quantization, TensorRT, decoder
training, reduced-frame-rate output and decoding reduced-resolution tiny latents
remain separate proposals.
