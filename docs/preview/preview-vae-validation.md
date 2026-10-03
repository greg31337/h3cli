# TAEH3 preview validation: local Metal

Validated on 2026-09-18, using only the local Apple M4 Max. Native
`--preview-vae`, live previews, streaming MP4 delivery and saved-latent
finalization are implemented. The local checks passed after the fixes below.
**The user accepted the reviewed Metal preview quality on 2026-09-18, completing
T044 for this corpus.** CUDA was pending at the end of this Metal round; see the
subsequent [RTX 5090 implementation and validation](preview-vae-cuda-validation.md).
The Metal results below do not qualify other Apple GPUs or CUDA devices.

See the [usage guide](preview-vae.md), [task status](todo.md), and
[structured evidence](preview-vae-validation.json). Large artifacts remain
under `outputs/preview-vae/`, outside source control.

## Environment and provenance

- Baseline source: `84ce9d1fcf0c755956e271b6f21bb2c3fee6476c`; the evidence JSON
  fingerprints the implementation source and final binary. The experiment
  ledger retains executable/script hashes for individual runs made during
  development; earlier unchanged render evidence was reused.
- Apple M4 Max, Apple GPU family 9, 128 GiB unified memory, 107.5 GiB recommended
  GPU working set, macOS 26.6.2 (25G83). Apple Clang 21.0.0 and FFmpeg 9.0.1.
- Optional Ollin TAEH3 checkpoint revision
  `62f7591f59dfbb4c3c02b7a621d180a9eeaba26c`, complete SHA-256
  `4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13`.
  The native path uses 64 decoder tensors, FP16 MPSGraph convolutions and
  activations, temporal batches of at most five, and at most two cached plans.
- Development reference: FastVideo's `minimax_h3_taeh3.py` at revision
  `430e52154e76b902c3cc17a16b3edc1fad790012`, file SHA-256
  `81770cfa4d9aaef0d471733211847e6ba29135ea54f694b65e917299c5605a5d`.
  Its local MLX adapter used Python 3.12.14, MLX/MLX Metal 0.32.2,
  NumPy 2.5.3 and safetensors 0.8.0. These are development dependencies only.
- Normal generation uses the existing `models/MiniMax-H3` model tree. Folded Turbo
  and realism fixtures use the existing `lora/output/` trees. State compatibility
  signatures, original source/input identities and commands are recorded with
  the corpus, gallery and generation artifacts.

The persistent [ledger](../outputs/preview-vae/metal/budget.json) records
**64 attempts, 969.60 seconds of the 1,200-second local allowance**. It includes
failed attempts, decoder warmups, generation, reference replay, mux and media
validation. CPU build/inspection and additional host-only unit checks are
separate. Component cases reserved at most 120 seconds; complete workflows at
most 300. No remote node or production-length generation was used.

## Implemented boundaries

The shared tiny API validates weights before conditioning, loads immutable
weights separately from temporal history, and emits bounded completed-frame
batches. The Metal implementation handles the trained input transform,
convolution/memory blocks, spatial and temporal expansion, pixel shuffle,
global frame selection, finite checks and teardown. The non-Metal tiny backend
returned an explicit unsupported error in this initial Metal round; the later
CUDA implementation is covered by its own report.

Generation selects tiny reconstruction only at video delivery and live display.
Original reference encoders, AudioVAE, denoising and authoritative AV/sampler
state remain in use. API caches distinguish actual checkpoint hashes; request
and preview boundaries reset temporal history. Repeated-request cache behavior
is covered by API tests.

Decode-only dispatch precedes heavyweight generation initialization. A versioned
presentation sidecar retains resize and continuation trimming, with a fingerprint
binding it to the unchanged `.h3av` format. Legacy files require explicit
full-state delivery. The full and tiny decode-only paths share a bounded FFmpeg
writer with concurrent audio delivery and atomic MP4 publication. Original
generation keeps its existing buffered full-decoder delivery.

## Passed coverage

| Area | Evidence and limits |
| --- | --- |
| Host/CLI and files | `make test-preview-vae`: defaults, early option/model rejection, malformed and truncated weights/states, presentation association, legacy overrides, legal frame/trim arithmetic, output protection and decode-only dispatch. Ordinary builds need no tiny weights. |
| Native decoder | Known-pattern and real-weight tests cover latent lengths 7/17/27, batches 1–5, non-five-aligned boundaries, tails, global trim, RGB/pixel-shuffle order, finite output, selected-frame context, repeated requests and reset after failure/cancellation. |
| Memory/lifetimes | Repeated geometry changes, two-plan bound, reusable input storage, memory preflight rejection and callback cancellation pass. Real and pattern tests pass with `MTL_DEBUG_LAYER=1`. This is not forced device-OOM testing. |
| Sanitizers | AddressSanitizer/UndefinedBehaviorSanitizer host, streaming and malformed-model/CLI tests pass. CUDA memcheck/racecheck and cuDNN alternatives were unavailable. |
| Sampling invariance | Controlled two-step CPU-sampler runs have exact raw Euler boundaries and final AV latents across full, tiny, noisy/denoised live display, paused tiny output and resumed full output. Paused output remains silent. |
| Continuation | Hard and bridge full/tiny trajectories match exactly. A 90-frame state retains its history, then delivers 51 frames after trimming 39 video frames and 52,000 audio samples. |
| Cache policy | Full/tiny/full requests, geometry/reference changes, real/pattern/missing/real model selection and cache clear pass. Missing weights fail early; hashes distinguish models without changing sampling. |
| Functional inputs | T2VA; first, last and both FL2VA anchors; face, multiple face/body images, video, audio with image, video with separate soundtrack; reuse and reduction. Both available Turbo and realism folded models pass FL2VA and Ref2VA smoke cases. |
| Finalization | Tiny generation to saved state to full VAE passes for resized T2VA, hard continuation and bridge continuation. Source hashes stay unchanged, denoiser calls are zero, and decoded RGB/audio match original full generation exactly. |
| Media | All 30 completed generated MP4s and all eight three-way replay comparisons decode completely with the expected dimensions/counts, 24 fps, 32-kHz stereo audio and delivery trims. Duration checks allow the existing audio/frame rounding (26 ms). Full/tiny replay audio decodes identically. Joined continuation comparisons contain 141 frames. |
| Failure handling | Tiny and full decode cancellation at frame delivery and before mux completion, forced encoder failure, incomplete frame counts, and large-audio abort stress preserve existing output and reap the encoder. Default, ignored and preblocked SIGPIPE configurations pass. |
| Default regressions | Final `make test` passes, including affected progress, preview, sampler, bridge, refvideo, memory, tokenizer and audio primitives. Ten optional external oracle fixture groups remain skipped, as detailed below. |

Functional generation uses 64×64 internal / 128×128 output canvases, generally
22 frames and two steps. Video-reference tests need 56 target frames to satisfy
the existing two-second reference requirement. These smoke renders prove
integration, not perceptual quality. Quality comparisons use saved higher-step
states instead.

The standard suite reports missing optional fixtures for MLX toy-block, MLX Qwen,
released AudioVAE, audio encoder, visual encoder, reference-video encoder,
Qwen vision, Qwen video-pair, multimodal Qwen and Ref2VA text presentation oracle
groups. These are skips, not passes. Separate real-model feature renders above
exercise the components but do not replace those missing oracle comparisons.

The streamed original decoder matches the buffered decoder for every F32 RGB
value at 288×384/56 and 480×640/22, including spatial seams, temporal blends and
tails. The 288×384 stream also matches the decoder saved before this change:
74,317,824 bytes, SHA-256
`fd952b740049309ceb55e4101166986e541034b7962c1fb58187d37c72cf31e9`.
Exact comparisons protect the original path and authoritative states; they are
not a numerical quality requirement for tiny versus full reconstruction.

## Performance

These are bounded single-machine measurements, with one warmed repeat per
decoder preset. Cold means a new decoder object/graph, with filesystem and OS
shader caches potentially warm. There are no confidence intervals or claims
about production-length clips.

| Preset | Full cold decode | Full warm decode | Tiny cold decode | Tiny warm decode | Warm gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| 288×384, 56 frames | 10.631 s | 10.581 s | 0.363 s | 0.119 s | 89.0× |
| 480×640, 22 frames | 7.979 s | 7.940 s | 0.362 s | 0.123 s | 64.5× |

These decoder-only intervals include compute, RGB transfer and sink checks;
they exclude audio, model-compatibility hashing and FFmpeg. At 288×384 the tiny
warm compute/transfer components are 0.058/0.048 seconds; at 480×640 they are
0.055/0.054 seconds. Decoder loading is 1.557/0.060 seconds (full/tiny) for the
first preset and 0.754/0.036 seconds for the second.

| Preset | Full maximum host RSS | Tiny maximum host RSS | Tiny observed Metal allocation peak |
| --- | ---: | ---: | ---: |
| 288×384, 56 frames | 9.28 GiB | 169.5 MiB | 294.8 MiB |
| 480×640, 22 frames | 9.43 GiB | 304.4 MiB | 666.9 MiB |

The full decoder's logical GPU peak is 9.365 GiB; its observed device allocation
at delivery for the larger preset is 9.385 GiB. Logical allocations, host RSS
and Metal's device-wide allocated-byte observation are distinct metrics. The
last may include other components and shared aliases in a cached generation
context. The standalone tiny measurement includes both cached plans; real
generation additionally holds the sampler, encoders and audio components.

| Saved-state replay | Full video delivery | Tiny video delivery | Full total CLI | Tiny total CLI |
| --- | ---: | ---: | ---: | ---: |
| image05, 288×384/56 | 10.682 s | 0.411 s | 15.68 s | 4.34 s |
| image12, 288×384/56 | 10.710 s | 0.421 s | 15.41 s | 4.35 s |
| multiple images, 288×384/56 | 10.693 s | 0.414 s | 15.38 s | 4.36 s |
| detail, 480×640/22 | 8.074 s | 0.429 s | 12.76 s | 4.35 s |
| chain1, 288×384/90 | 17.740 s | 0.477 s | 22.47 s | 4.46 s |
| chain2, 288×384/51 delivered | 17.720 s | 0.466 s | 22.44 s | 4.43 s |
| face20, 480×640/22 | 8.070 s | 0.430 s | 13.07 s | 4.35 s |

Video delivery includes decode, conversion/resize, pipe writes and mux completion.
Total CLI time also includes compatibility hashing, audio and loading. The
compatibility identity currently takes approximately 3.6–4 seconds to read/hash
the original decoder model files, explaining much of tiny replay's startup cost.
First delivered frame latency for image05 improves from 3.547 to 0.147 seconds;
for trimmed chain2, from 10.594 to 0.210 seconds.

The matched fresh-process, two-step T2VA smoke workflow takes **28.27 seconds
full versus 24.49 seconds tiny (1.15×, 13.3% less wall time)** through MP4 and
saved state. Both final states have SHA-256
`708c6c443c4d32d124c38167124a7e18f6855cb73b12ebe099664847c74a75ad`.
This includes conditioning/model startup and uses a much smaller canvas than
the decoder presets; it must not be presented as a universal end-to-end gain.
Similarly, a cached tiny request must not be compared with a cold full request.

Warmed live previews at 64×64/22 frames take about 3.3 ms per displayed step,
including context and readback. Two continuation previews at 64×64/90 frames
take about 0.123–0.125 seconds together, including initial graph compilation.
The early continuation helper's `source.json` contains an invalid preview timing
due to a missing timestamp; it is excluded. Later helper records are corrected.

The measured decoder speed target of 3× is exceeded locally. Readback/conversion
and fixed startup now matter more, so no speculative extra convolution/fusion
tuning or quality-reducing temporal shortcuts were added.

## Quality evidence and acceptance

Open the [review gallery](../outputs/preview-vae/metal/quality/review.html).
It contains eight matched original/native-tiny/MLX-reference comparisons and
three joined-continuation videos. The
[manifest](../outputs/preview-vae/metal/quality/manifest.json) records exact
commands, state/input/source identities, MP4 hashes and media metadata. Run
logs and the structured evidence retain timings and executable provenance.

The corpus covers `1.jpg`, `2.jpg`, face/body and multiple references, motion,
detail and continuation. The original six 20-step states came from previously
qualified B200 runs; all new reconstruction and testing in this round was local
Metal. A 20-step ordinary Metal face state was reused for `face20`. The older
six-step, 128×128 `face` fixture is visibly broken with the original decoder as
well; it is retained as a smoke comparison, not quality evidence.

Static first/middle/last-frame inspection shows corresponding composition,
poses and broad color in native and reference tiny output. Fine hair, foliage,
facial and clothing texture is softer than the original VAE. The added face20
comparison also preserves the visible pose/expression across sampled frames.
Static inspection alone does not establish motion, listening quality or human
identity acceptance. Following gallery review, the user confirmed on 2026-09-18:
"confirmed in review the quality is acceptable". **T044 is complete.** This
acceptance applies to the reviewed local Metal corpus and continuation joins;
it does not qualify the later CUDA tiny backend or other hardware.

## Failures retained and resolved

- The first gallery baseline attempt rejected an incorrectly calculated test
  sidecar fingerprint. The fixture was corrected to hash the whole canonical
  state; production rejection was correct.
- Three feature-matrix attempts encountered existing reference constraints:
  minimum two-second video, target-frame capping, and audio needing visual
  conditioning. Tests resumed from the unfinished cases with valid fixtures;
  previous successful generation was reused.
- Decoder cancellation exposed a real Darwin SIGPIPE race while a large audio
  write was interrupted. The streaming writer now suppresses SIGPIPE per pipe
  descriptor with `F_SETNOSIGPIPE` on Darwin, retaining a thread-local portable
  fallback. Real full/tiny cancellation and forced-encoder tests, abort stress,
  sanitizer tests and default regressions pass after this fix.

All failed attempts remain charged in the ledger. No timeout, numerical tiny/full
similarity threshold or previous quality approval was converted into a pass.

CUDA kernels, CUDA memory/race instrumentation and no-cuDNN/cuDNN comparisons
were deferred to the [subsequent RTX 5090 round](preview-vae-cuda-validation.md).
The Metal implementation does not qualify CUDA behavior. Production limits above the short tested presets
retain geometry/memory validation, but large-resolution and 362-frame stress
runs were not performed. Reduced-resolution tiny decoding, full-VAE mixed
precision/quantization, TensorRT and new decoder training remain separate work.
