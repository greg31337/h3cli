# CUDA rendering through 1,450 frames with NVFP4

Status: **Planned — 0/32 tasks complete.**
Created **2026-11-24**. This checklist describes future implementation; the
current CUDA ceiling remains 736 frames and extended-duration NVFP4 is still
rejected.

The completed 736-frame checklist is [archived byte-for-byte](long-video-736-todo.md),
SHA-256 `61d69175f8390d6df7409a72e8a4ca50d37605bc1c51e52124b0a91f01921bd3`.
Its [contract](cuda/long-video-contract.md),
[acceptance report](cuda/long-video-results.md),
[memory estimates](cuda/long-video-memory-estimates.md), and
[audio workspace correction](cuda/long-video-audio-workspace.md) remain historical
evidence. Preserve the [large-canvas](large-canvas-todo.md) and
[empty-latent still](empty-still-todo.md) archives unchanged.

## Objective and scope

Extend native **CUDA** generation from 736 to **1,450 aligned frames**, and
qualify opt-in `--cuda-denoise-quant nvfp4` for long native videos. Keep Metal at
362 frames. Cover CLI, public API, HTTP, `--render`, conditioning, hard/bridge
continuation, checkpoints and delivery. BF16 remains the default.

- Preserve upward alignment to `F = 5 + 17*n` and 24 FPS. A request for
  `--seconds 60` or `--frames 1440` becomes **1,450 frames / 60.416667 s**.
  Use `--frames 1450` to request the maximum aligned count directly.
  **1,433 frames / 59.708333 s** is the largest aligned result strictly below
  60 seconds. Reject requests whose aligned result exceeds 1,450; do not clamp,
  truncate, repeat frames or change the frame rate.
- Derive backend temporal limits centrally. CUDA's new maximum is **427 video
  latent steps and 2,417 audio latent steps**. Bridge class storage needs
  **427 video bytes and 2,418 audio bytes**. Contexts remain `39 + 51*k`,
  shorter than the target and no longer than the source; the maximum legal
  CUDA context becomes **1,416 frames**.
- Native CUDA canvases remain bounded by 1920×1088, including transpose and
  existing pixel/axis checks. New BF16/NVFP4 combinations must pass actual
  sequence, indexing, device-memory and host-memory admission. Supporting a
  frame count does not promise that every resolution fits a 72 GB card.
- Support native BF16 and NVFP4 with all 50 blocks, default dense attention,
  reuse/core-reuse 1, and the existing reference/anchor mutual exclusions.
  Include prompt-only FL2VA, first/last anchors, Ref2VA image/audio/video
  conditioning, and hard/bridge continuation. NVFP4 remains restricted to its
  existing qualified SM120 hardware/library requirements.
- Preserve existing supported short-video FP8, attention and approximation
  behavior. Do not enable FP8, Sage/SOL/SubBlock, adaptive cache, layer
  thinning, token reduction or runtime LoRA for the new extended combinations
  as a side effect of removing the NVFP4 rejection.
- Keep reference-input duration rules separate from target duration. The
  existing 2–15-second media rules are unchanged.
- Keep full temporal context, attention and normalization. Row-wise projection
  chunking is allowed; independently denoised temporal windows, temporal
  interpolation, replacement models and multi-GPU execution are outside scope.
  A general QKV-arena redesign is not required by this milestone.
- Preserve existing BF16 source capture and learned upscale temporal support
  where sequence/memory limits permit it. **NVFP4 source capture, NVFP4 AV
  upscaling and NVFP4 combined with `#upscale` remain unsupported** and must
  fail before expensive work. This plan does not add quantized refinement.
- Motion quality and scene consistency remain outside acceptance. Quantized
  results must be finite and internally consistent; they are not required to
  equal BF16 outputs. Preserve numerical identity for unchanged BF16 behavior
  and the existing short-video NVFP4 recipe.

## Development and regression envelope

Continue using local macOS for host/build work and the already authorized
72 GB CUDA server for GPU tests. Public examples use `cuda-test`; keep real
SSH details and raw records in ignored outputs, following
[publication privacy](publishing.md).

1. Runtime video canvases remain **256×256**, including generated clips,
   references, continuation states, replay, preview/full decode, API/HTTP/script
   workers and final merges. All video artifacts stay **at most 20 seconds**.
   The representative long case remains **464 frames / 19.333 s**.
   `--seconds 20` aligns to 481 frames and is a planning-only case.
2. Use all 50 blocks and **two denoising evaluations** for feature tests, with
   the existing six-evaluation ceiling. Run GPU jobs serially. Keep the
   60 GiB development capacity ceiling, actual free-memory checks and
   `max(1 GiB, physical VRAM/10)` device reserve. Account for actual available
   host RAM and compressed/BF16 caches independently.
3. **All cases above 464 aligned frames, including 736 and 1,450, and all
   larger production canvases are planning/metadata/descriptor-only.** Do not
   allocate their complete movie, activation, PCM or decoder-workspace tensors
   on CPU or GPU, render/decode them, or render a longer video and trim it.
   Descriptor queries may cover their true shapes without executing kernels
   on full-size data. Exercise large-path chunk logic with bounded fixtures.
4. For continuation, use a **447-frame source → 464-frame target with
   447-frame context**. Deliver 17 new frames; source plus suffix merges to
   464 frames. Do not merge two complete 464-frame clips.
5. The earlier approved upscale exception remains limited to **256×256 →
   512×512**, at most **56 frames per clip or final merge**, and two denoising
   evaluations. Use it only for existing BF16 upscale compatibility checks.
   It does not authorize long or larger-canvas upscale renders.
6. The user's latest instruction, **“run 204 regression test,”** supersedes the
   earlier skip. Run the unchanged complete gate now as a baseline and again
   after implementation changes as required by [CONTRIBUTING.md](../CONTRIBUTING.md).
   Its fixed **640×480 / 124-frame / six-evaluation** video and component probes
   are an explicit exception to the development canvas rule. Keep all 204
   golden hashes and the recorded fixtures unchanged; do not add skips,
   tolerances, resized fixtures or replacement goldens.
7. These restrictions mean the completed feature can have runtime evidence
   above the old NVFP4 limit at 464 frames, but **no end-to-end 1,450-frame
   validation**. Reports must distinguish this gap from supported host
   planning, descriptor coverage and bounded execution. Do not seek larger
   runtime cases as an automatic prerequisite for finishing this plan.

Documentation work runs link/archive checks only, apart from the separately
requested baseline regression. A passing baseline does not complete any of the
implementation tasks below.

### Verified baseline before implementation

On **2026-11-24**, the complete unchanged regression passed **204/204 exact
output hashes** against the existing standalone Linux package. Suite execution
took **189.66 seconds** (211.62 seconds including artifact verification).
The tested server source matched the local inference source fingerprint:

- Source: `f1ba25d7451626d36853b43ccdb5ff7414e7e2d72a33b39a705076340bfc1974`.
- Standalone executable: `ca5f2b03ffda79f23826b7e8bc7436e3b5e45d7b054e78a49ac9db300f07d363`.
- Golden manifest: `fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3`.

This validates the existing 736-frame implementation. It does not implement or
qualify the proposed 1,450-frame/NVFP4 extension; LNV031 still requires the final
implementation's own gate. Raw evidence remains in ignored outputs.

## Capacity baseline for planning

These are extrapolations of the current
[capacity formula](cuda/large-canvas-memory-estimates.md), not measured
minimums or final NVFP4 admission budgets. Assume 1,450 frames, all 50 blocks,
512 text rows, no extra reference rows, 2 GiB for other allocations and 10%
capacity headroom. Values round upward to whole GiB.

| Native canvas | Packed sequence rows | BF16 resident / streamed | NVFP4 resident, weight substitution only |
| --- | ---: | ---: | ---: |
| 256×256 | 32,674 | 56 / 18 GiB | 27 GiB |
| 640×480 | 133,446 | 93 / 55 GiB | 65 GiB |
| 1024×768 | 333,282 | 167 / 129 GiB | 139 GiB |
| 1344×768 | 435,762 | 205 / 167 GiB | 177 GiB |
| 1920×1088 | 876,426 | 369 / 330 GiB | 340 GiB |

The last column substitutes approximately **10.09 GiB** of packed NVFP4
projection weights for **35.89 GiB** of BF16 weights. It excludes additional
NVFP4 packing/padded-output buffers and is not a claim that the job fits.
Do not divide total VRAM by four. BF16 streaming already retains only about
1.44 GiB of core weight slots; other live buffers dominate long videos.

The implementation must replace these provisional substitutions with explicit
per-phase budgets for actual BF16/NVFP4 buffers, cold preparation, warm packed
loading, resident/streamed placement, conditioning, checkpoints and decoding.
Keep conservative reserves until bounded measurements and lifetime analysis
justify a tighter formula. No task requires 1,450-frame high-resolution
rendering to fit the current server.

The one-million-row CUDA sequence ceiling remains unchanged. The native
1920×1088 maximum fits it under this table's assumptions, but references can
increase the total. A 1,450-frame 3840×2176 upscale has **3,484,320 video rows
alone** and must still be rejected, including K=0.

## M0 — Contract, shared limits and guarded tests

- [ ] LNV001 Freeze the backend/precision/conditioning matrix above and record
  the pre-change source, binary, fixture and golden-manifest identities.
  Inventory literals and indirect limits at 362/736 frames, 107/217 video T,
  603/1,227 audio T, class arrays, 15-second media checks, packed rows and
  integer/serialized sizes. Keep the archived milestone byte-identical.
- [ ] LNV002 Extend the shared temporal helpers in [geometry](../src/geometry.h)
  and [host](../src/host.c) to CUDA 1,450/427/2,417 and Metal 362/107/603.
  Preserve upward alignment, minimum native/script shapes, still behavior,
  and checked seconds/integer conversion. Derive 1,416-frame contexts without
  changing the existing context lattice.
- [ ] LNV003 Extend host-only resource/sequence/index planning through the new
  maximum, using actual text, references, audio, anchors and geometry.
  Reproduce the baseline table without large allocations; check padded NVFP4
  row/scaling extents separately from logical sequence length. Preserve the
  one-million-row limit and actionable capacity diagnostics.
- [ ] LNV004 Extend the guarded campaign entry point to capture precision,
  placement, source/target canvas, aligned duration, evaluations, child jobs,
  decoder inputs and every merged output. Enforce the development envelope,
  including the narrow upscale and immutable-regression exceptions. Reject
  oversized hidden suite defaults before any model or movie allocation.

Exit: production limits and feature policy are explicit and independent of
the much smaller authorized runtime test envelope.

## M1 — Duration, delivery and state compatibility

- [ ] LNV005 Apply shared limits consistently in CLI/API validation, HTTP queue
  and SGLang target adaptation, and script planning. Resolve frame/seconds
  precedence before model work. Audit indirect saved-state execution and
  server preflight; retain reference-media input caps and early Metal
  rejection rather than relying on the first CUDA kernel to fail.
- [ ] LNV006 Extend target-shape handling in FL2VA/Ref2VA layout, first/last
  anchor positions, reference-video soundtrack alignment and condition
  packing. Check row counts, temporal coordinates, byte offsets and generated
  versus reference audio lengths through 1,450; retain image-size and
  first/last/reference/continuation semantics.
- [ ] LNV007 Extend full and preview VAE temporal admission and delivery.
  Preserve 22-frame full-VAE chunks, five-frame overlap, ordered bounded
  stitching and frame/sample timestamps. Audit last chunks, preview indexing,
  cancellation and decode-only paths without retaining a complete RGB movie.
  Keep Metal's limit and upscale/preview exclusions intact.
- [ ] LNV008 Fix generated AudioVAE planning for all new shapes before admitting
  expensive denoising. Descriptor-query every decoder convolution over all
  **86 aligned durations from 5 through 1,450**. Audit both the current
  **3 GiB workspace** and **64 MiB layout-adapter** caps: the latter grows to
  **123,750,400 bytes (118.02 MiB)** at the maximum and is already insufficient.
  Derive checked shared bounds from queried shapes, allocate only required
  buffers, update phase admission and shape-specific diagnostics, and preserve
  the qualified algorithms and FP32 arithmetic. Do not guess that doubling
  the existing cuDNN workspace is sufficient.
- [ ] LNV009 Expand hard/bridge temporal plans, prefix masks and class-index
  checks to 427/2,418 bytes and 1,416-frame contexts. Preserve exact protected
  prefix bytes, the single bridge-velocity scaling, audio alignment, source
  compatibility and suffix trimming. Keep the target strictly longer than
  its valid context on the executing backend.
- [ ] LNV010 Extend AV-state and presentation admission, metadata inspection,
  frame/PCM counts, file-size/offset bounds and backend preflight. Keep
  existing schemas when their fields already express the shape; reject
  malformed or unsupported states before large allocation. Older CUDA states
  remain readable and Metal may inspect metadata without executing long jobs.
- [ ] LNV011 Version the sampler/clean-source temporal payload deliberately.
  Keep section 15 version 1 at **107/604** bytes for legacy durations and
  version 2 at **217/1,228** for 363–736 frames; add a required version 3 with
  **427/2,418** for durations above 736. Do not change version-2 wire lengths
  merely because C arrays grew. Cross-check versions, effective frames,
  inactive tails, integer/container bounds and policy before allocation.
  Preserve existing same-build resume checks and safe old-reader rejection.

Exit: BF16 supports the new temporal shapes across public and saved-state
paths, while old wire layouts and backend rejection behavior remain stable.

## M2 — Long native NVFP4 without unbounded scratch

- [ ] LNV012 Separate native generation precision policy from the stricter
  upscale policy in [upscale_state.c](../src/upscale/upscale_state.c) and its
  callers. Permit opt-in NVFP4 for supported CUDA native canvases through
  1,450 frames, including the extended canvas, with default attention and
  full depth. Keep BF16 defaults, hardware/library preflight and all other
  extended-combination rejections; never silently fall back to BF16.
- [ ] LNV013 Audit NVFP4 QKV, attention-output, FC1 and FC2 projection dimensions,
  cuBLASLt descriptors, 128-row padding, VEC16 scale swizzling, strides,
  element counts and offset arithmetic. Use checked size_t/64-bit products
  where required. Validate true maximum logical/padded layouts without
  allocating them, and fail unsupported GEMM shapes before denoising.
- [ ] LNV014 Bound activation packing and padded-output scratch in
  [cuda_quant.cuh](../src/cuda/cuda_quant.cuh). Support row chunks/offsets for
  quantized projections and reuse scratch across calls; avoid an additional
  full-movie padded output for a short tail. Preserve tensor-wide amax/global
  scales and VEC16 scale indexing across chunks. If arithmetic must change,
  assign an explicit execution recipe while retaining the old short-video
  recipe; do not silently use per-chunk scales.
- [ ] LNV015 Implement bounded NVFP4 MLP execution instead of forcing full
  sequence FC1/activation tensors when quantization disables the fused path.
  Account for both activation and output workspaces and the largest cold
  conversion operand. Preserve the activation-scale contract with bounded
  reduction/recomputation passes if needed. Keep short-video behavior
  unchanged unless an explicitly versioned recipe is necessary; do not
  alter temporal attention or model depth to save memory.
- [ ] LNV016 Qualify resident and compressed-streamed weight placement for
  long NVFP4, including auto admission and the existing allocation retry.
  Preserve packed host storage, direct packed uploads and metadata-cache
  identity; do not retain a complete BF16 device copy alongside packed
  weights. Check cold/warm host RAM, disk preparation, cancellation cleanup,
  immutable-source checks, cache corruption and bounded context reuse.
- [ ] LNV017 Add explicit per-phase NVFP4 memory estimates from actual buffer
  lifetimes and chunk sizes. Include packed weight scales, conversion staging,
  padded activation/output scratch, GEMM workspace, model-independent live
  tensors and phase-specific host copies. Use actual free memory and the
  existing reserve once; remove unsupported fixed scratch assumptions only
  after replacing them with checked bounds, never by disabling admission.
- [ ] LNV018 Extend sampler/presentation precision provenance for new-duration
  NVFP4. Persist and validate precision, arithmetic recipe and chunk/scale
  policy whenever they affect results. Preserve packed-weight cache identity
  when weight packing is unchanged. Resume must restore the same policy and
  reject explicit incompatible overrides; completed NVFP4 AV states remain
  decodable without loading quantized transformer weights.

Exit: NVFP4 is an explicit, memory-accounted native execution mode at the new
duration ceiling; existing BF16 and short NVFP4 behavior is preserved.

## M3 — Connected workflows and explicit exclusions

- [ ] LNV019 Audit engine/API context lifetime, prepared resources, mode changes,
  cancellation and allocation failure recovery for BF16 and NVFP4 long jobs.
  Repeated requests must not reuse an incompatible precision/layout/recipe,
  leak GPU/host allocations or overwrite a valid output on failure. Preserve
  a completed sampler checkpoint when subsequent decoding fails.
- [ ] LNV020 Propagate effective precision, frames, weight placement and cache
  controls through CLI, HTTP and every script worker. Verify script settings
  inheritance, global references, segment-local additions, comma lists,
  `#ref-image-size`, seed sets, dry-run plans and segment/merge metadata.
  Native NVFP4 script jobs must not turn into BF16 in child processes.
- [ ] LNV021 Qualify native NVFP4 hard and bridge continuation, including
  supported references, without adaptive cache or alternate attention.
  Preserve inherited exact rows, source identity and audio/video trimming;
  test each precision's own replay/resume contract without demanding NVFP4
  and BF16 generation equality. Preserve existing cross-precision source
  compatibility rules and never relabel source precision.
- [ ] LNV022 Carry the new temporal ceiling through existing BF16 `.h3up`,
  AV-state K=0 transfer, `#upscale` and refinement planners where sequence and
  memory limits admit them. Keep NVFP4 source capture/AV transfer/`#upscale`
  explicitly rejected before workers or model loading, including states whose
  quantization is only visible in presentation metadata. Preserve continuation
  plus nonzero-refinement rejection, native continuation sources, exactly-once
  prefix trimming, audio identity and maximum-target sequence rejection.

Exit: long NVFP4 works through native entry points and continuation, and
unsupported upscale/approximation combinations fail clearly and early.

## M4 — Bounded verification

- [ ] LNV023 Add host boundary and adversarial tests for 362/363, 464/480/481,
  736/737/753, 1,433/1,434/1,440/1,450/1,451, `--seconds 60`, invalid floating
  values and huge integers. Cover 1,416-frame context planning, maximum native
  and doubled canvases, added reference rows, padded kernel bounds, state
  sections 15 v1/v2/v3, truncation and malformed metadata without full long
  allocations. Verify early unsupported-device/mode combinations.
- [ ] LNV024 Extend descriptor-only AudioVAE and NVFP4 layout/GEMM planning
  probes across the true maximum temporal/sequence extents. Record requested
  workspace, layout peaks and any algorithm discontinuities. Cover execution
  paths exceeding the old 64 MiB audio-layout and quantized-offset assumptions
  through descriptors plus bounded operator fixtures, not long PCM/movie data.
- [ ] LNV025 Add bounded native NVFP4 operator checks for all four projection
  families, tail rows, multiple chunks, tensor-wide scales, zero/nonfinite
  inputs and packed cache reuse. Force a small chunk size to exercise
  production chunk loops on small tensors. Compare chunked execution against
  the retained NVFP4 reference at the declared numerical contract, and check
  resident/streamed agreement, cancellation and allocation plateaus.
- [ ] LNV026 Run guarded **256×256 / 464-frame / two-evaluation** BF16 and
  NVFP4 generation on the 72 GB CUDA machine. Cover prompt-only, first/last
  anchors and supported mixed references using bounded 256×256 fixtures.
  Verify real NVFP4 dispatch, finite latents/PCM, dimensions, frame count,
  timestamps and complete audio delivery. Keep the existing reference-media
  duration rules and record the full decoder's independent peak.
- [ ] LNV027 Verify same-build pause/resume, full/preview AV replay, exact
  continuation history, bridge masking and failure recovery for each
  precision. Use 447→464/447-context continuation and a 464-frame final merge.
  Compare same-recipe seeded trajectories/replay under existing contracts;
  no BF16-versus-NVFP4 quality/parity threshold or visual-review gate is added.
- [ ] LNV028 Exercise guarded CLI/API/HTTP and script BF16/NVFP4 jobs, reused
  contexts and correctly inherited precision. Use shorter clips where needed
  to keep seed variants and final merges below 20 seconds. Verify all
  rejection routes before output creation/model loading. Run only the
  previously approved short BF16 upscale compatibility cases when needed.
- [ ] LNV029 Build on local macOS and run bounded host/state/parser checks,
  short 256×256 Metal compatibility and early rejection above 362 frames.
  Verify CUDA-only NVFP4 rejection and indirect long-state execution through
  resume, decode, HTTP, script and upscale paths. Do not invoke suites with
  larger hidden media defaults or execute long CUDA states on Metal.

Exit: the supported new NVFP4 duration range has bounded end-to-end evidence
at 464 frames, and 1,450-frame behavior has explicit host/descriptor coverage.

## M5 — Measurement, artifacts and completion

- [ ] LNV030 Publish reproducible host-only BF16/NVFP4 capacity reports for
  256×256, 640×480, 1024×768, 1344×768 and 1920×1088 through 1,450 frames.
  Separate actual allocation plans from conservative capacity budgets and
  measured bounded peaks. Reuse authorized test outputs to record preparation,
  denoising, audio/video decode, tracked/device peaks and host RSS for resident
  and streamed modes; do not extrapolate a short-run speedup as a 60-second
  guarantee or claim that larger canvases fit the current card.
- [ ] LNV031 Build final macOS and Linux artifacts and verify source/binary
  provenance against the tested checkout. Run bounded offline package smoke
  tests with models/cache supplied explicitly. Run the complete unchanged
  **204-output CUDA reference regression** after coherent implementation
  changes and as the final code gate; fix failures and rerun the whole suite.
  Use fresh evidence directories, bind the result to exact artifacts, and
  reject stale/missing hardware/model results as passes.
- [ ] LNV032 Update help, README/usage, script/HTTP/API, state, continuation,
  precision and upscale documentation. Publish a new 1,450-frame/NVFP4
  contract, capacity report and acceptance record without rewriting historical
  measurements. State CUDA/Metal limits, alignment, supported combinations,
  unchanged upscale exclusions and the missing full-duration runtime evidence.
  Validate links, archive/golden hashes and task counts; mark completion only
  after the bounded feature checks, artifact checks and final 204 gate pass.

## Completion criteria

All 32 tasks have recorded evidence. Native CUDA accepts supported BF16 and
NVFP4 requests through 1,450 aligned frames, subject to honest sequence and
resource admission; Metal retains 362. Long NVFP4 generation, continuation,
replay/resume and public entry points pass the bounded runtime matrix, while
maximum-duration states, descriptors and arithmetic pass without large
allocations. Old short/736-frame wire layouts and applicable numerical recipes
remain intact. Unsupported precision, reference, backend and upscale
combinations fail before expensive work. The final unchanged regression matches
all 204 recorded outputs.

Completion does **not** claim 60-second/high-resolution runtime qualification,
a 72 GB fit for every supported shape, or motion/scene consistency.
