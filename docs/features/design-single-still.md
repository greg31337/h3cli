# Single-still H3 support

Status: implemented and locally qualified on Metal (2026-09-22), with
T=1 generation. Implementation baseline: `31d8ee834197b19b4f2ac1d4dc9f8d3174ce3a7c`.
See [usage](single-still.md), [results and limits](single-still-results.md), and
[archived task status](single-still-tasks.md). CUDA compilation and default execution are qualified on the supplied SM120
node; see the [CUDA record](single-still-cuda-results.md).

The requested feature includes text/reference-to-still generation, with
decoder compatibility proven first. The first deliverable decodes one H3
temporal latent into one image using an image-trained VAE. Generation is a
separate, required stage with its own gate: accepting one latent in a decoder
does not establish the DiT's single-image sampling contract. Existing video
rendering remains available with its original VAE and temporal rules.

## Basis and evidence

The reference is VPIPE commit
[`351b20c10761eca8ba2f07f923323ca78bd7da15`](https://github.com/tgo-app-dev/vpipe/commit/351b20c10761eca8ba2f07f923323ca78bd7da15),
dated 2026-09-19. Its `decode_image` path requires checkpoint metadata declaring
`h3_t1_direct`, decodes `T=1`, and selects the declared temporal output slice.
It reuses spatial tiling and keeps image delivery distinct from video delivery.
The commit adds codec/round-trip support, not a text-to-image sampler.

The checkpoint is
[Mamad8/MiniMax-H3-Image-VAE](https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE/tree/c7b9252c73707dba494cf4d99ca45d3f33f561b3).
Its author describes a frozen encoder with a fine-tuned decoder and warns that
the image decoder degrades video reconstruction. Image reconstruction also has
known softness and fine-detail limitations. These are upstream observations,
not h3cli qualification results. See the
[pinned model card](https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE/blob/c7b9252c73707dba494cf4d99ca45d3f33f561b3/README.md).

The full artifact was downloaded and its checksum verified before qualification.
The original 66,616-byte safetensors header and source notices are retained;
a metadata-only fixture is included in `tests/fixtures/still-vae-header.json`.

| Property | Pinned value |
| --- | --- |
| Model revision | `c7b9252c73707dba494cf4d99ca45d3f33f561b3` |
| File | `minimax_h3_t1_image_vae_step1597.safetensors` |
| Published file size | 5,207,808,784 bytes |
| Published SHA-256 | `6c3d0bfa055986a803a566a862fcde283a1e63db62829e5ef4a2a5aebf50bb86` |
| Header inventory | 562 tensors, all `F16` |
| Capability / format | `h3_t1_direct="true"`, `h3_t1_format="full_decoder_v1"` |
| Output slice | `h3_t1_output_slice="3"` |
| Architecture | 24 latent channels, spatial ratio 16, temporal output patch 4 |
| Config | JSON string under `minimax_h3_video_vae`, containing whitening vectors and `source_config` |

VPIPE reports matching encoder/quant-convolution tensors and whitening against
its video checkpoint. Our dtype-aware audit confirms all encoder/quant tensors
match after F16 rounding and whitening matches; the image round-trip path loads
its own encoder. Byte identity is meaningful only at matching dtypes/packing;
our F32 checkpoint must not be declared byte-identical to this F16 payload.

The pinned header contains no tile-size/overlap fields. Use the explicitly
versioned released-profile values **256 pixels / 64 pixels overlap**, as in
the linked VPIPE implementation. Do not pretend these values were read from
the file. Tiling affects spatial RoPE and therefore the decoder function.

## Scope and user contract

Implement the portable codec using the existing shared C VAE operations and
Metal/CUDA GPU interfaces. Qualify local Metal first. CUDA execution requires
its own fixtures on available hardware; no previously used remote node is
assumed available and no cross-hardware speed comparison is required.

The codec milestone includes:

- Explicit image-VAE selection, model inspection, capability validation and
  bounded F16-to-F32 weight loading.
- A `T=1` decoder, deterministic image encode/decode validation, normalized
  still-latent import/export, and lossless PNG output.
- One final frame callback, cancellation, memory admission and isolated cache
  lifetime. No audio decode or muxing for a still.
- Independent numerical fixtures and a final HTML image-comparison report.

The later generation extension includes prompt-only FL2VA and ordered image
references through Ref2VA only after its sampling/layout gate passes. It does
not initially include continuation, video/audio references, first/last-frame
anchors, live denoising preview, AV/sampler checkpoint resume, or automatic
selection of SOL, Q8, Sage, ANE or Turbo. Existing options remain separate
experiments; this feature grants none of them a new quality qualification.

No video-to-still fallback is implicit. Rendering 22 frames and taking a frame,
repeating a latent seven times, or relaxing the stock decoder's length check
does not implement direct single-still decoding.

Implemented interfaces (the generation gate below has passed):

```sh
# Codec-only: does not load a tokenizer, DiT or audio VAE.
./bin/h3cli --decode-still-latent outputs/example.h3still.safetensors \
  --image-vae models/image-vae/minimax_h3_t1_image_vae_step1597.safetensors \
  -o outputs/example.png

# Generation, qualified with six-evaluation examples.
./bin/h3cli -d models/MiniMax-H3 --still --image-vae models/image-vae/minimax_h3_t1_image_vae_step1597.safetensors \
  -p "A woman playing a grand piano, cinematic photograph." \
  --width 640 --height 480 --steps 6 --seed 42 -o outputs/piano.png

./bin/h3cli -d models/MiniMax-H3 --still --image-vae models/image-vae/minimax_h3_t1_image_vae_step1597.safetensors \
  -p "A portrait of the woman in <Picture 1>." --ref-image inputs/2.jpg \
  --width 640 --height 480 --steps 6 --seed 42 -o outputs/portrait.png
```

`--still` selects a real image result, independent of the filename. Its effective
frame count is one; accept an explicit `--frames 1`, reject other explicit
frame counts and `--seconds`. Without `--still`, video frame validation and
rounding are unchanged, including rejection of ordinary `--frames 1`.
Default `--image-vae` to
`models/image-vae/minimax_h3_t1_image_vae_step1597.safetensors` for still generation
and decode; allow an explicit path override. Never download weights during
inference or infer decoder capability from a filename. Image flags on a video
request fail before loading large models. `--preview-vae` and live video-preview options
fail for this first still implementation with an actionable explanation.

PNG is the initial file format: exactly one image, RGB24, no audio stream or
video container. CLI outputs must have a `.png` extension; the library may
deliver solely by callback. A decode-only request derives its canvas from the
latent; conflicting canvas, temporal or generation options fail preflight.

## Implemented integration points

| Area | Implementation |
| --- | --- |
| `src/image_vae.[ch]` | Strict image metadata, inventory, identity, normalized artifact and preflight contracts. |
| `src/weights.[ch]` | Explicit single-file store and bounded opt-in F16 expansion; directory stores reject image-only/duplicate components. |
| `src/vae/video_encoder.c` | Separate image-checkpoint encoder using existing deterministic posterior-mean operations. |
| `src/vae/video_vae.c` | Shared tile operations with actual tile time; distinct image role, slice and spatial-only assembly. |
| `src/host.c`, `src/engine.c` | Dedicated T=1 layout, auxiliary audio T=2, original sampler; ordinary video geometry is unchanged. |
| `src/media/decode.c`, `src/media/ffmpeg.c` | Codec-only delivery, exactly-one callback and staged PNG publication. |
| `src/h3cli.c`, `src/h3.h` | Explicit still API/result and CLI modes with PNG output. |
| Caches and state | Still operation/audio key, separate decoder ownership; existing AV/sampler formats unchanged. |

Both input packing and RoPE now derive tile time from the allocated patch
geometry. Video keeps its existing seven-frame tile allocation and clamping;
the image decoder allocates one frame and retains unclamped output through
spatial assembly. Instrumented tiny-shape tests and bit-identical 243-frame
video checks cover this shared-core change.

## Checkpoint and precision policy

Add an image-VAE descriptor with a component role, canonical artifact digest,
source dtype, validated architecture, whitening vectors, capability version,
output slice, tile profile and execution recipe version. Image and video
decoders must have different cache identities even when tensor names/shapes
match. The image decoder is never installed into the normal video VAE slot.
If a video loader encounters image-only weights or ambiguous duplicate
components, reject them rather than silently using the wrong decoder.

Read `h3_t1_direct` as a strict recognized string (`true`/`1`); absent/false
means no still capability. Reject malformed values, unknown image format,
invalid or nonfinite whitening values, nonpositive standard deviations,
unsupported architecture, duplicate required entries and missing tensors.
Parse the slice as a complete integer string and require `0 <= slice < 4`.
Require an explicit slice in our initial format: unlike VPIPE's fallback to
the video pad, the pinned checkpoint already supplies it. Recheck bounds at
the public decode API, including for descriptors constructed programmatically.

Validate the nested config against our supported H3 architecture and tensor
inventory; a similarly named ComfyUI file is insufficient. Unknown unrelated
metadata may be retained as provenance, but must not enable a capability.
Verify the pinned full-file SHA before real-model qualification. Metadata
digests such as `h3_t1_base_sha256` are recorded separately and do not replace
verification of the actual artifact.

Initially execute the existing F32 VAE operations. Convert finite IEEE F16
weights to F32 explicitly in bounded chunks (at most 8 MiB CPU staging), with
checked sizes and normal memory admission. Do not reinterpret F16 bits as
BF16 or silently change all VAE loaders' dtype behavior. Account for the
expanded decoder weights, rather than budgeting only the 5.2 GB download.
Native F16 VAE acceleration is a later, independently measured optimization.

For codec round trips, load the checkpoint's own frozen encoder through the
existing encoder operations, release it, then load its decoder. Compare it
with our original encoder using dtype-aware tensor/value and latent fixtures
before sharing an encoder across artifacts. Decoder-only requests load no
encoder weights. Do not retain two complete VAE copies for A/B tests.

## Latents, normalization and decoding

The public codec input is normalized, channel-first F32
`[24,1,H/16,W/16]`. Persist it as the single F32 tensor `latent` with shape
`[1,24,1,H/16,W/16]` in a bounded safetensors file. Require metadata for schema
version 1, `h3-normalized-v1` latent space and encoder/whitening compatibility
digest. Record producer provenance separately from decoder selection. Reject
untyped raw tensors, other batch/time/channel counts, nonfinite values and
overflowing geometry. Initially keep the existing canvas limit and multiples
of 32; do not broaden image resolution limits as part of this feature.

Use the deterministic posterior mean for image round trips. Normalize once
with `(z_raw - mean) / std`; the decoder restores `z_raw = z * std + mean`
once before `post_quant_conv`. Reference fixtures must declare whether their
latents are raw or whitened; matching tensor shapes alone cannot establish
this convention. Encoder pixels use the existing ImageNet normalization,
and decoder pixels receive its inverse exactly once.

The direct decode path is:

```text
normalized T=1 latent
  -> validate capability/geometry; unwhiten
  -> spatial tiles at the released 256/64 profile
  -> post-quant projection and decoder with actual T=1 RoPE
  -> four pixel-time slices per spatial output patch
  -> select checkpoint slice (3 for the pinned artifact)
  -> spatial overlap blend, inverse pixel normalization, finite check
  -> RGB24 conversion and one PNG/frame callback
```

Parameterize the common tile core with actual latent time, keeping stock video
calls at seven. For `T=1`, the temporal RoPE coordinate is zero under our
centered formula; spatial coordinates are normalized to each tile's extent.
Registers/suffix rows retain their model-defined positions. No temporal
window priming, dropped latent tokens, chunk overlap blending or video frame
offset arithmetic is applied to the image result.

An image-specific unpacker selects the metadata slice; it does not call the
video chunk's frame-count formula. Retain unclamped image floats through
assembly for reference comparisons and clamp only for final image delivery.
Preserve the video's existing clamp/blend order during the shared-core
refactor. At 640×480 the codec has 1,200 spatial
latent cells; later DiT patching would have 300 image target rows before text
and references. These are different counts and neither is a seven-frame clip.

Validate tile and overlap alignment to the 16-pixel VAE patch grid, tile bounds,
coverage and blend denominators. The initial image path fixes the released
tile profile; reject conflicting video tile overrides instead
of inheriting them silently. Larger tiles are a different numerical recipe,
not a transparent memory optimization. Bounded row dispatches must check both
input and output strides and integer/byte-offset overflow; do not copy VPIPE's
backend-specific band size without checking our kernels.

## Delivery, lifecycle and state compatibility

Add an explicit result kind, with video retaining its default value. A complete
still reports `frames=1`, `fps=0`, `audio_samples=0`, `sample_rate=0` and has
no `h3_av_state` or video presentation sidecar. Its final callback has index
zero, count one and the existing completed-frame marker `denoise_step=-1`.
Latent ownership, result destruction and cancellation must be documented.

Use FFmpeg's PNG encoder through the existing subprocess infrastructure, with
one RGB frame and explicit image output settings. Stage the destination and
publish it only after successful encoding. Abort/cancellation removes partial
output, reports failure and releases buffers. No MP4 muxer or audio VAE opens.
`--frames-dir`, if selected, receives one final PPM; terminal display receives
one image. `--still` selects PNG generation for the current CLI invocation.

Keep `.h3av`, `.h3sample`, continuation and video presentation formats unchanged.
The new still-latent artifact is the supported codec interchange. Initially
reject still use of AV save/decode, sampler pause/resume, continuation and
bridge flags before any model work; do not fabricate zero-length audio inside
old containers. Still generation cache keys include operation kind, layout
recipe, audio policy and model identity. Decoder keys additionally bind the
image artifact, dtype conversion, whitening, slice and tile policy. A decoder
change alone must not invalidate unrelated text embeddings.

Retain the current 110 GB default memory guard and cancellation checks during
loading, conversion, each decoder block, tile assembly and output. For generated
stills, release the DiT before final image decoding unless an explicitly
admitted cache can safely retain it. Progress separates image VAE loading,
image decoding and PNG writing. Measure cold load, warm decode, write time and
observed footprint separately; single-image output does not eliminate weight
loading costs.

## Gate before text/reference-to-still generation

Gate result: **go for integration**. Two pinned independent H3
still implementations corroborate T=1 with two auxiliary audio ticks sampled
and discarded. The frozen [generation contract](single-still-generation-contract.md)
keeps h3cli noise/schedules and original dense BF16 execution. Bounded prompt,
seed-repeat and image-reference tests passed. Full-model cross-implementation
velocity parity and broad/50-step quality are not claimed. The requirements
below describe the gate and remain relevant to future recipe changes.


The VPIPE commit and image-VAE card do not define a one-image DiT target layout,
audio policy or sampler. Establish these using a pinned working H3 reference
or independent model-equation evidence and retained fixtures before exposing
`--still` generation. Prefer a new still geometry/layout builder over changing
`h3_temporal()` globally.

The experiment must resolve generated-video time `T=1`, target/reference
position semantics, task embeddings, video sigma schedule, seed/noise order and
audio rows. An audio-free target is not assumed supported just because the
layout struct permits zero audio: DiT allocation, projections, timestep
classes and sampler loops need a verified zero-row path. If the verified H3
recipe needs auxiliary audio, freeze its exact length, noise and schedule,
keep it internal and record its cost. Do not invent a duration or dummy audio
row merely to satisfy an assertion.

Start with original BF16 weights, dense attention and all 50 blocks; establish
prompt-only FL2VA first. Then validate ordered image references, image RoPE
slots, `--ref-image-size` behavior and reference protection in Ref2VA. A still
reference remains an image, not a one-frame reference video. No frame-count
speedup, semantic quality or reference adherence is claimed from decoder
round trips. If no supported sampling recipe is established, ship the codec
milestones and keep generation tasks explicitly gated with the evidence.

## Validation and acceptance

Freeze fixtures, thresholds, checkpoint/source hashes and metric scripts before
evaluating the implementation. Record reference tensor dtype, normalization,
tile profile and output slice; retain unclamped decoder output as well as PNGs.
Use an independent reference decoder or model-equation implementation, not
only a comparison between two wrappers around our tile core.

| Layer | Required evidence |
| --- | --- |
| Host | Strict metadata/dtype parsing; every invalid slice; malformed tensor/config; wrong checkpoint role; normalization vectors; offset/size overflow; tile coverage; one-image CLI/result contract. |
| Small GPU | Actual T=1 allocations/RoPE/unpack under bounds instrumentation; synthetic slice markers; normalization exactly once; finite output; cancellation and repeated reuse without allocation growth. |
| Real codec | Fixed 256×256 and 512×512 center crops plus 640×480; posterior-mean round trips; original-vs-image encoder compatibility; independent decoder output; wrong slice/stock checkpoint/untiled diagnostic controls. |
| Video regression | Existing short video VAE fixtures and retained **243-frame** decode states preserve stock decoder output/metadata and callbacks; still/video cache switching cannot select the wrong weights. |
| Generation, gated | T=1 layout/noise/velocity fixtures and at most six evaluations at 640×480, followed by a small prompt/image-reference corpus. Prove no hidden 22-frame target or final audio decode. |

For the frozen real-codec fixtures, require candidate-to-reference RGB MAE <= 0.01
and PSNR >= 35 dB on the same unclamped [0,1]-scale output, plus round-trip
PSNR > 25 dB on pinned ordinary-photo crops whose independent baseline passes
that floor. Freeze the actual fixture contract in the evidence milestone;
threshold changes after candidate results are new experiments, not passes.
Do not apply this floor to arbitrary artwork/text or claim the publisher's
dataset mean as our result. The four-slice sweep checks the implemented slice
against reference outputs; a wrong slice need not be visually obvious.

Compare released tiling with a diagnostic whole-image decode on a fixed
above-tile fixture; keep the latter out of production selection. At or below
one tile, the two paths should agree. Test guards with tiny shapes and mocked
allocations rather than allocating oversized tensors. Larger 1344×768 checks
are optional codec-only checks; any generation check at that size remains
capped at two evaluations. No long denoising tests are needed for codec work.

GPU runs remain serial. The six-evaluation ceiling is retained for future
denoising tests; still tests use one output frame, while existing video
regression geometry remains 243 frames. Produce source/reconstruction/reference
HTML pages, metrics, timing/memory records and exact commands with checksums.
There is no mandatory human-review approval gate; failed automated checks and
unqualified platforms/compositions remain explicit.

## Completion

Codec completion means an explicitly selected compatible artifact can decode
one normalized H3 latent into one PNG, independently verified and bounded in
memory, without altering ordinary video behavior. Generation completion is a
separate gate requiring the verified DiT recipe and its own semantic/reference
evidence. The requested feature is not complete until that generation stage
is implemented and evaluated; a codec-only delivery is an intermediate result.
Documentation must distinguish reconstruction from generation and
list precisely which backends and precision paths were exercised.

Adapt the existing h3cli implementation rather than importing VPIPE's model
runtime. Retain source attribution and applicable notices for any upstream
code actually copied during implementation; the runtime imports no VPIPE implementation. Independent test equations and
metadata provenance are documented in [sources](single-still-sources.md). Model provenance remains explicit and separate from code provenance.
