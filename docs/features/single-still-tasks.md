# Single-still feature tasks (archived)

Archived from the completed plan on 2026-09-22. Current work is tracked in
[todo.md](todo.md).

Design: [design-single-still.md](design-single-still.md).

Implementation status, 2026-09-22; previous Metal/CUDA milestone results remain
in their existing documents and Git history. This plan adds no four-bit branch.
The reference VPIPE commit implements a single-frame **decoder**, not a
single-frame DiT sampler. Deliver the codec first; generation is separately
gated and included in the requested scope. Codec completion alone does not
complete the feature. The codec and generation are implemented and
locally qualified. See [usage](single-still.md), [results](single-still-results.md)
and [provenance](single-still-sources.md). All 31 tasks are complete.
[CUDA qualification](single-still-cuda-results.md) passed on the supplied
RTX PRO 6000 Blackwell node with CUDA 12.8 and default execution.

| Milestone | Result |
| --- | --- |
| M0–M3 | Implemented; strict host, instrumentation, lifecycle and artifact checks pass. |
| M4 | Metal and CUDA codec/lifecycle tests and exact 22/243-frame video regressions pass. |
| M5 | Integration gate passed with pinned independent T=1/audio-T=2 recipes. |
| M6 | Prompt, exact seed-repeat and image-reference renders pass on Metal and CUDA at 640×480, six evaluations each. |

Generation velocity fixtures are retained Euler transitions, not an independent
full-model parity claim. Cache coverage includes role/key isolation, repeated
codec calls; no new same-context video denoising
sequence was run. The limited image corpus does not qualify 50-step quality or
accelerated compositions. Final HTML pages require no human-review approval.

Testing: local Metal first, then the user-provided CUDA node
`cuda-pro6000`. No full-render cross-hardware speed claim is made. Still fixtures use **T=1** at
256×256, 512×512 and 640×480. Video regressions retain **243 frames** and may
reuse captured latents. Denoising tests remain serial and capped at six
evaluations per render; optional 1344×768 generation is capped at two.

## M0 — Pin the artifact and independent reference

- [x] SS001: Pin VPIPE commit `351b20c10761eca8ba2f07f923323ca78bd7da15`
  and image-VAE revision/artifact from the design. Retain metadata, full-file
  checksum, tensor inventory, source notices and exact reference commands.
- [x] SS002: Audit the image checkpoint against our original VAE: encoder,
  quant-convolution, whitening, decoder and post-quant tensors. Distinguish
  byte equality from dtype-aware equality; document F16-to-F32 conversion.
- [x] SS003: Prepare independent raw/normalized latent, T=1 RoPE, projection,
  four output-slice and reconstructed-RGB fixtures. Freeze metric thresholds,
  image crops, arithmetic and tile profile before testing candidate results.

Exit: reproducible artifact and independent codec fixtures; no claim that
the DiT supports single-image generation.

## M1 — Explicit image-VAE loading and identity

- [x] SS004: Add strict image-VAE metadata/config validation: capability,
  `full_decoder_v1`, explicit slice 0–3, supported architecture and 24 finite
  whitening means/positive standard deviations. Reject malformed values,
  duplicates and stock video weights for direct image decoding.
- [x] SS005: Add explicit single-file weight-store access and bounded F16-to-F32
  conversion for the image component. Validate tensor shapes/ranges and memory
  admission; preserve existing F32/BF16 loader behavior for other components.
- [x] SS006: Separate image/video decoder roles and cache identities. Bind
  artifact, precision recipe, normalization, slice and 256/64 tile profile;
  reject ambiguous or image-only weights in the ordinary video path.
- [x] SS007: Add host parser, conversion and selection tests, including
  missing capability, invalid slice strings, nonfinite data, truncated files,
  overflowing dimensions and coexisting image/video artifacts.

Exit: image weights are explicit and validated; stock video selection is
unchanged and no image decoder is silently substituted.

## M2 — Direct T=1 decoding

- [x] SS008: Extract the reusable VAE tile computation with an explicit latent
  time dimension. Audit all fixed-seven loops, allocation counts, register and
  suffix offsets, RoPE buffers and output patch indexing. Preserve video calls.
- [x] SS009: Implement the image decoder with normalized F32 T=1 input,
  unwhitening exactly once, actual T=1 RoPE and metadata-selected output slice.
  Reuse decoder operators on Metal and CUDA without latent repetition or
  video chunk priming/drop/temporal stitching.
- [x] SS010: Enforce the released 256/64 spatial profile and tile coverage,
  patch alignment, blend and byte-offset bounds. Reject conflicting video
  tile overrides. Preserve ImageNet normalization and finite-output checks.
- [x] SS011: Reuse the encoder operations with an explicit image-checkpoint
  source for deterministic posterior-mean round trips. Release encoder before
  decoder loading; establish compatibility before sharing original weights.
- [x] SS012: Add small GPU oracle tests for RoPE, all four slices, unwhitening,
  odd tile boundaries, guards and repeated calls. Exercise T=1 bounds under
  instrumentation so a surviving fixed-seven loop cannot pass unnoticed.

Exit: one true latent frame produces one correctly selected image, matching
independent fixtures. Video temporal semantics and numerics are preserved.

## M3 — Still artifacts, API and PNG delivery

- [x] SS013: Add the versioned normalized still-latent safetensors contract:
  one F32 `[1,24,1,h,w]` tensor, latent-space and compatibility metadata,
  finite/shape/size checks, bounded loading and staged writes.
- [x] SS014: Expose codec-only `--decode-still-latent` and `--image-vae`
  without requiring tokenizer/DiT/audio initialization. Add explicit result
  kind, ownership and exactly-one final callback; retain existing video defaults.
- [x] SS015: Add atomic one-image PNG delivery, terminal display and optional
  one-frame PPM export. Handle callback cancellation and encoder failure;
  emit no audio or MP4, and remove incomplete output.
- [x] SS016: Add preflight rejection for incompatible dimensions, extensions,
  frame/duration options, preview VAE, AV-state and sampler-resume/continuation
  flags. Keep existing `.h3av`, `.h3sample` and presentation formats unchanged.
- [x] SS017: Integrate memory reservations, bounded conversion/scratch and
  deterministic teardown. Test repeated decode, failure and cancellation;
  log cold load, warm decode, PNG time and observed footprint separately.

Exit: a codec-only command writes exactly one PNG or delivers one callback,
with no video/audio container fiction and no regression to existing APIs.

## M4 — Codec qualification and reporting

- [x] SS018: Run the pinned 256/512/640×480 round trips and independent output
  comparisons. Retain raw RGB, normalized latents, SHA-256 values and complete
  commands; report precision differences rather than hiding them in PNG rounding.
- [x] SS019: Run wrong-checkpoint/slice/normalization controls and a diagnostic
  above-tile whole-image comparison. Validate the released tiling recipe;
  do not assume any larger tile is an equivalent optimization.
- [x] SS020: Run existing short video fixtures plus retained 243-frame decode
  regression and still/video cache switching. Verify unchanged stock decoder
  selection, frame count, callbacks and saved-state compatibility.
- [x] SS021: Verify the CUDA build path and run the same fixtures when a CUDA
  device is available. Record unavailable execution as unqualified; do not
  replace it with a claimed cross-platform pass or hardware performance ratio.
  **Passed:** CUDA 12.8 SM120 build, host/Compute Sanitizer checks, the three
  codec fixtures, controls, exact 22/243-frame video regression, six-step
  prompt/repeat/reference generation, and saved-latent re-decode.
- [x] SS022: Publish an HTML source/reference/reconstruction gallery, quality
  gates, memory/timing breakdown, platform coverage and usage docs. Close the
  codec milestone independently of generation; no human-review approval gate.

Exit: codec evidence passes its frozen gates, or failures are explicitly
recorded without promotion. Existing video tests still use 243-frame geometry.

## M5 — Establish the still-generation contract

- [x] SS023: Obtain a pinned working H3 still-generation reference or independent
  model-equation evidence. Resolve T=1 target layout, task/position semantics,
  sigma schedule, deterministic noise and audio policy for FL2VA.
- [x] SS024: Prototype the smallest shared DiT/sampler path against retained
  layout/noise/velocity fixtures. Verify zero-audio-row behavior if supported;
  otherwise document the source-backed auxiliary-audio recipe and its cost.
  Do not invent dummy audio or a short-video fallback.
- [x] SS025: Record the go/no-go result and freeze a generation-specific test
  contract before public integration. Decoder PSNR does not qualify generated
  image quality or reference adherence. If the recipe is unsupported, retain
  the completed codec and leave M6 gated with an explicit reason.

Exit: a supported, reproducible image-target sampling recipe exists. The
linked VPIPE decoder commit alone does not satisfy this gate.

## M6 — Integrate and qualify generation (gated on M5)

- [x] SS026: Add an explicit still geometry/layout and `--still` request kind
  without changing video rounding. Use original BF16/dense/50-block execution
  first; enforce one image target and the verified audio policy.
- [x] SS027: Add prompt-only FL2VA, then ordered image-reference Ref2VA support,
  preserving image RoPE slots and reference-size rules. Reject unsupported
  video/audio references, anchors, continuation and preview combinations.
- [x] SS028: Bind operation/layout/audio policy to generation and conditioning
  caches. Connect still-latent save, final image decode, result cleanup and
  DiT-before-VAE memory transition without altering old state formats.
- [x] SS029: Expose explicit still/video generation and validate conflicting
  flags in the CLI. Use `--still` and `-o` to select PNG output.
- [x] SS030: Run bounded 640×480 T=1 generation and image-reference fixtures
  (maximum six evaluations), deterministic-seed checks and semantic/reference
  screens. Prove no hidden 22-frame denoising or final audio decode. Keep
  accelerated precision/routing/LoRA compositions separately labeled.
- [x] SS031: Publish final still-generation examples, image-comparison HTML,
  startup/denoise/decode/memory measurements and known limits. Keep decoder
  reconstruction, generation quality and backend qualification distinct.

Exit: a real single-image generation request completes through the verified
sampler and image VAE, with its own evidence. No 243-frame performance run or
inherited FP16/SOL/Q8 acceptance substitutes for this test.
