# Replace `--still` with empty-latent generation and stock video VAE decoding

Status: **planned — 0/20 tasks complete**.
Updated 2026-11-21.

Replace the current dedicated-image-VAE implementation of `--still` with the
single-image technique supported by ComfyUI's standard `Empty Latent Image`
and `VAE Decode` nodes. Generate directly at the requested resolution,
including **3872×2176 (8,425,472 pixels)**, using the existing H3 FL2VA/Ref2VA
weights and stock video VAE. Keep one PNG as the public result.

The completed **8/8 integrated text-script rendering milestone** is
[archived intact](render-integration-todo.md), with its
[acceptance evidence](features/render-integration-results.md). The original
[still implementation tasks](features/single-still-tasks.md),
[sampling contract](features/single-still-generation-contract.md) and
[results](features/single-still-results.md) remain historical evidence for the
implementation being replaced. This checklist records future work.

## Existing implementation and intended change

The current code already samples a single video latent frame and two auxiliary
audio positions in [engine.c](../src/engine.c), using
`h3_still_layout_build` in [host.c](../src/host.c). Reuse this foundation.
The principal changes are decoder selection, high-resolution admission,
resource handling and the associated CLI/model/artifact contracts.

| Area | Current behavior | Replacement requirement |
| --- | --- | --- |
| Denoising target | Video `[1,24,1,H/16,W/16]`, audio `[1,32,2,2]` | Retain these shapes and generate at the final canvas |
| Final decoder | Dedicated `minimax_h3_t1_image_vae_step1597.safetensors`, direct T=1 decode | Stock video VAE, five copies of the generated latent, raw decoded frame index 3 |
| Output dimensions | Shared 768×1344 / 1,032,192-pixel limit | Separate checked still limits admitting at least 3872×2176 and its transpose |
| Model dependency | Automatic image-VAE download and `--image-vae` selection | Use the normal model directory and stock video VAE; retire the dedicated checkpoint requirement |
| Saved still | Schema 1 with dedicated image-encoder/whitening digest | Explicit stock-VAE compatibility and decoder recipe, with a documented old-artifact policy |

The existing video decoder is also not a drop-in T=1 API: its public temporal
paths expect `T=5*k+2`, normally at least seven tokens. The dedicated image path
uses different weights and T=1 positions. Neither assumption establishes the
required five-token stock-VAE decode; implement and validate that case explicitly
in [video_vae.c](../src/vae/video_vae.c).

## Source evidence and execution contract

[ComfyUI PR #15677](https://github.com/Comfy-Org/ComfyUI/pull/15677), merged
2026-08-17, adds model-specific adaptation of an empty image latent into H3
video/audio streams. Its
[change at `d937b223d58ebe1c30156d725b6b15761208607c`](https://github.com/Comfy-Org/ComfyUI/pull/15677/commits/d937b223d58ebe1c30156d725b6b15761208607c)
and [sampling preparation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/sample.py)
explain why the generic four-channel image node can be used. Native h3cli can
construct the final H3 shapes directly; it needs no ComfyUI node runtime or
intermediate four-channel allocation.

The [stock H3 VAE source](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/vae.py),
inspected on 2026-11-21, unwhitens the latent, repeats a T=1 input over its
five-token chunk, selects raw pixel frame 3, and finalizes pixels. This is the
stock-video branch; dedicated image VAEs have separate handling. Pin the exact
source revision and reference fixtures in ELS001 before implementation.

The replacement contract is:

- Batch size one, target video T=1 and auxiliary audio T=2. An empty latent
  specifies target geometry; generation still starts from seeded noise and
  samples both streams. Discard generated audio without loading an audio VAE.
- Denoise only the single target latent. Temporal replication happens during
  final decoding, after sampling. Save the original normalized T=1 latent.
- Decode with the ordinary H3 video weights and their normalization. Unwhiten
  exactly once, preserve all five temporal positions in VAE attention, and
  return zero-based raw frame 3 as the sole output image. Preserve spatial
  tiling, blending, pixel normalization and clipping from the pinned reference.
- Keep prompt-only FL2VA and ordered image-reference Ref2VA, including the
  existing `--ref-image-size match|high|max` behavior. Target canvas dimensions
  and reference preprocessing remain separate settings.
- Preserve h3cli's existing Euler/shifted-flow sampler and seeded-noise contract
  for this replacement. The supplied
  [Fizgig example workflow](https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still/blob/main/example_workflows/h3_still_text_to_image-8MP-NoTurboVersion.json)
  uses `er_sde` and `simple`; adopting the latent/decode technique does not
  reproduce that trajectory. Record this difference in qualification results.
- Keep PNG delivery, final display/callbacks, progress, cancellation and
  save/re-decode support. Ordinary video, continuation and upscaling retain
  their existing geometry, sampling, decoder and saved-state contracts.

This milestone uses the existing video-model checkpoints. A separate H3 Image
model, a new upscaler, new samplers and broader acceleration support are outside
this replacement. Retain unsupported-option checks unless a task explicitly
qualifies a combination. Higher resolution alone does not establish prompt
quality or identity fidelity.

Expected target geometry (reference and text tokens are additional):

| Output canvas | Video latent `[B,C,T,h,w]` | DiT target visual rows after 2×2 patching |
| --- | --- | --- |
| 640×480 | `[1,24,1,30,40]` | 300 |
| 1344×768 | `[1,24,1,48,84]` | 1,008 |
| 2144×1216 | `[1,24,1,76,134]` | 2,546 |
| 3872×2176 | `[1,24,1,136,242]` | 8,228 |

All cases retain audio `[1,32,2,2]`: 128 scalar values / four packed audio rows.
The DiT row count is not a VAE tile-token count or a memory estimate.

## M0 — Freeze the replacement contract

- [ ] ELS001 Pin the upstream empty-latent conversion and stock-video still
  decode sources, model configuration, tensor layouts, whitening, RoPE and
  tile profile. Record revision/source identities and the independent reference
  command. Capture deterministic normalized-latent fixtures and decoded pixels;
  choose numerical tolerances before comparing native results. Treat the supplied
  matching-images report as motivation, not an acceptance oracle.
- [ ] ELS002 Specify the CLI/API and artifact migration before changing them:
  `--still` selects the stock video VAE, `--decode-still-latent` resolves that VAE
  without a denoiser, and retired `--image-vae` usage receives migration guidance.
  Define a still-specific maximum area and dimension bounds that admit the full
  8,425,472-pixel target and its transpose. Define the schema-1 compatibility
  policy and a distinct decoder recipe identity; do not silently reinterpret an
  old checkpoint path or claim identical images from the two decoders.

Exit: reproducible decode reference, explicit public behavior and migration
rules. M1 and M2 implement this contract before M3 switches the default.

## M1 — Native single-frame targets at high resolution

- [ ] ELS003 Reuse and make explicit the existing T=1/audio-T=2 target construction
  in [engine.c](../src/engine.c) and [host.c](../src/host.c). Verify seeded video
  and audio noise, packed segments, task IDs, reference slots and positional
  coordinates. No video frame alignment or decoder repetition may expand the
  denoising target. Assert the table's shapes and target-row counts.
- [ ] ELS004 Introduce one checked still-geometry policy and use it for CLI/API
  admission, generation, latent save/load and codec-only decode. Remove the
  still path's accidental dependence on `H3_MAX_PIXELS` and
  `h3_geometry_valid` in [engine.c](../src/engine.c) and
  [image_vae.c](../src/vae/image_vae.c). Audit downstream helpers for residual
  limits or automatic shrinking. Keep multiples-of-32 alignment and positive
  dimensions; use checked multiplication. Preserve video, reference-image and
  upscale limits rather than increasing a shared global cap.
- [ ] ELS005 Audit high-resolution allocation/index arithmetic through text and
  image conditioning, DiT packing/RoPE/attention, sampler buffers, VAE tiles and
  final RGB/PNG delivery on Metal and CUDA. Account for text/reference tokens,
  both audio/video streams and live weights/scratch. Fail with an actionable
  memory error before oversized allocation; preserve the process memory guard.
  No silent resize, reduced layer count, reference dropping or preview fallback.

Exit: host validation admits the target canvases with exact shapes, rejects
invalid/overflowing dimensions, and leaves ordinary video admission unchanged.

## M2 — Stock video VAE still decoding

- [ ] ELS006 Add an explicit still entry point in
  [video_vae.h](../src/vae/video_vae.h) and
  [video_vae.c](../src/vae/video_vae.c). Consume normalized channel-major T=1,
  construct five temporal copies, and execute the stock decoder with the correct
  five-token positions, register/suffix layout and output projection. Audit
  fixed-seven assumptions and backend kernels. Extract raw frame index 3 before
  ordinary video trimming; padding to seven tokens or decoding T=1 directly is
  not a substitute for the independently checked five-token computation.
- [ ] ELS007 Implement spatially tiled still decoding with bounded scratch and
  exactly one assembled image. Keep the reference tile-overlap/blending and
  pixel-finalization order; avoid retaining a full-resolution multi-frame RGB
  movie. Replication may be materialized per tile if mathematically equivalent.
  Preserve all temporal attention inputs while reducing delivery storage.
  Validate partial edge tiles, seam intersections and portrait geometry.
- [ ] ELS008 Route final generation and codec-only delivery through this decoder
  in [engine.c](../src/engine.c) and [media/decode.c](../src/media/decode.c).
  Release completed denoiser resources before decode as needed; isolate decoder
  cache identities by weights, recipe, temporal geometry, precision and tiles.
  Keep exactly one final frame callback (index 0/count 1), PNG publication,
  result ownership, progress accounting and cancellation cleanup.

Exit: identical input latents pass the independent stock-VAE decode comparison
on Metal and CUDA, without altering normal video chunk decoding.

## M3 — Model selection, saved latents and retirement

- [ ] ELS009 Update [model resolution](../src/models/resolve.c), catalog/group
  handling and CLI defaults so still generation no longer requires or downloads
  the dedicated image VAE. Resolve the stock VAE for the selected FL2VA/Ref2VA
  model. Codec-only decode must resolve only its VAE/configuration dependencies,
  with no tokenizer, DiT or audio decoder. Cover explicit paths, default model
  roots, offline operation, missing files and packaged model discovery.
- [ ] ELS010 Implement the saved-latent contract from ELS002 in
  [image_vae.c](../src/vae/image_vae.c) or a dedicated still-artifact module.
  Save finite normalized F32 `[1,24,1,h,w]`, normalization compatibility and
  decoder recipe metadata. The current digest includes dedicated F16 encoder
  tensors; matching whitening alone cannot validate that digest against stock
  weights. Version metadata where semantics change. Test the documented
  schema-1 migration or clear rejection, including codec-only re-decode, rather
  than retaining an implicit dependency on the retired checkpoint.
- [ ] ELS011 Update [CLI parsing/help](../src/h3cli.c),
  [option descriptors](../src/cli/options.def), [public API](../src/h3.h),
  shared request validation and server job/model planning for the new selection
  rules. Preserve `--frames 1`, PNG defaults and existing unsupported-mode
  diagnostics. Define the public API transition for `image_vae` and the decode
  function's model argument. Keep `--render` a video-script operation. Verify
  HTTP still jobs report one image with no AV/audio state.
- [ ] ELS012 Remove the replaced dedicated-image decoder path, mandatory
  capability checks, automatic image-VAE selection and dead implementation/test
  branches after the stock path passes. Audit low-level codec consumers before
  deleting shared utilities. Preserve historical task/results/provenance records
  and user model files. Default still execution must have one documented decoder
  path and no silent fallback to the previous implementation.

Exit: prompt-only and reference stills, saved latents, CLI/API/server operations
and codec-only decode all work without the dedicated image checkpoint installed.

## M4 — Verification and qualification

- [ ] ELS013 Extend the existing [still host tests](../tests/test_still.c),
  [CLI tests](../tests/test_still_cli.py),
  [layout tests](../tests/test_still_layout.py) and model-resolution tests.
  Cover all geometry rows and their transposes, accepted/rejected policy
  boundaries, malformed/nonfinite/truncated artifacts, schema migration,
  explicit flag conflicts and reference sizing. Confirm oversized ordinary
  video requests still fail and invalid input fails before downloads/GPU work.
- [ ] ELS014 Add an independent stock-VAE fixture runner using ELS001's pinned
  source and compatible weights. Compare unwhitening, five-token RoPE/attention,
  selected raw frame, tile blending and final pixels at small sizes and across
  tile boundaries. Exercise Metal and CUDA with guard/sanitizer coverage. Require
  the predeclared numerical thresholds; old dedicated-VAE output and comparisons
  of the native implementation against itself are not reference evidence.
- [ ] ELS015 Run end-to-end prompt-only and ordered image-reference renders on
  both backends at 640×480 and 1344×768. Verify finite trajectories, exact seed
  repeatability within the same execution policy, saved T=1 latent re-decode,
  reference ordering and all three reference-size modes. Check final PNG size,
  one callback/result frame and zero audio decode/output. Record semantic quality
  separately from numerical and lifecycle correctness.
- [ ] ELS016 Qualify direct 2144×1216 and 3872×2176 generation on capable hardware,
  including prompt-only, reference-conditioned and portrait cases. Begin with
  bounded 2–6-evaluation functional probes, then record representative 50-step
  prompt/reference quality runs at the 8 MP target. Retain commands, seed,
  shapes, PNG dimensions, device/build identity, timings and peak memory by
  phase. Demonstrate 8,228 denoised target visual rows, with no lower-resolution
  render or post-generation upscale. Record each backend's tested limits;
  unavailable hardware or OOM is not a passing 8 MP result.
- [ ] ELS017 Exercise resource lifetime and failure recovery: repeated stills,
  video→still→video in a reused context, cancellation during loading/sampling/
  tiles/delivery, allocation failures, encoder failures and existing output
  destinations. Confirm no leaks/stale decoder state, no partial PNG publication,
  no unintended audio/model loads and successful reuse after supported failures.
- [ ] ELS018 Run applicable `make test-still-host`, `make test-current-host`,
  backend tests and current-feature integration. Follow
  [CONTRIBUTING.md](../CONTRIBUTING.md): after every coherent implementation
  change, run the complete immutable **204-output CUDA reference regression**.
  Do not weaken it or regenerate golden hashes. Include ordinary video decoding,
  continuation, reference conditioning and upscale coverage; record missing
  prerequisites as unverified, never as passes.

Exit: codec and generation evidence are complete, high-resolution generation is
measured, and the unchanged video regression contract passes.

## M5 — Documentation and packaged acceptance

- [ ] ELS019 Update the project README, [still guide](features/single-still.md),
  [sampling contract](features/single-still-generation-contract.md), usage,
  model-download guide, API examples and hardware requirements. Describe the
  stock-VAE dependency, high-resolution limits, `--image-vae`/artifact migration,
  decode-only command and measured memory. Include native 8 MP examples and
  the sampler difference from ComfyUI. Retain earlier results as dated evidence
  and publish a distinct replacement acceptance record with source provenance.
- [ ] ELS020 Validate source builds and standalone macOS/Linux packages, including
  offline still generation, saved-latent decode, model selection and bundled PNG
  encoding with no image-VAE checkpoint. Reconcile every task against retained
  evidence, document backend/resource limits, validate documentation links and
  update completion status only after all required work passes.

Completion requires the stock-video-VAE path to replace the current `--still`
implementation, genuine 8 MP denoising/delivery to be demonstrated, and saved
latent/model selection behavior to be explicit and tested. Codec-only success,
a high-resolution dry run or resizing a small generated image does not complete
this milestone.
