# Native latent upscaling — implementation tasks

Status: **complete; 43/43 tasks complete; visual qualification accepted; regular feature**. Design:
[Native latent upscaling from saved generation state](../features/design-latent-upscale.md).
The completed 38-task adaptive/SubBlock campaign is preserved in the
[archive](../cuda/adaptive-subblock-tasks.md), including its evidence and acceptance.

Goal: capture a portable clean segment, apply the published 3D H3 latent
upscaler natively, preserve audio, reconstruct target conditioning/layout, run
2–4 optional video refinement evaluations, and deliver/save the larger result.
Implement and qualify CUDA and Metal. The native source, transfer and refinement paths are implemented. CUDA and Metal implementation qualification, all 14 comparison videos, diagnostics and local artifact audits passed.
Evidence: [qualification record](latent-upscale-qualification.md) and
[results/playback](latent-upscale-results.md). The user accepted the
comparison and promoted latent upscaling to a regular feature; the
[acceptance record](latent-upscale-acceptance.json) binds that decision
to the reviewed report and fourteen video hashes.

## Execution contract

- The [design](../features/design-latent-upscale.md) defines source/container,
  normalization, geometry, noise, audio and resume semantics. Resolve its named
  mathematical gates with fixtures before enabling the feature.
- Dense BF16 only for the initial source/refinement qualification. No FP8/NVFP4,
  adaptive/SubBlock/fixed reuse, token reduction, LoRA/Turbo or continuation/bridge
  combinations. Preserve current behavior outside the opt-in feature.
- Target pairs: **672×384 → 1344×768** and **960×544 → 1920×1088**, exactly 2×
  spatially, unchanged time. The larger target needs an explicit new geometry
  profile; do not globally raise ordinary generation limits.
- Completed comparison: **90 frames, 24 FPS, 50 source/direct steps**, seed 42,
  fixed piano prompt; L0/D0/P0/I4/U0/U2/U4 once for each pair. That is 14 videos
  including two low-resolution sources, with 12 target-resolution outputs.
  Refinement uses the same target noise and starting video sigma 0.25 for K=2/4.
  No extra successful repeats, seed/prompt sweeps or timing-only full videos.
- Run GPU jobs serially on the configured CUDA host; retain its connection
  details outside tracked files. Use CUDA 13.0.3/cuDNN 9.20 and the qualified
  math/media environment. Do not scan/hash H3 base weights; record metadata.
- Follow [CONTRIBUTING.md](../../CONTRIBUTING.md): after every coherent source,
  build or test-tool change run the complete unchanged **204-output CUDA golden
  regression**. No changed goldens, tolerance/fixture changes, skipped cases or
  relaxed deadlines. Preserve the ordinary six-evaluation budget; only dedicated
  fixed comparison jobs may perform the 50-step source/direct renders.
- Generated binaries/static libraries stay under ignored `bin/`; evidence,
  state files, reference downloads and media stay under ignored `outputs/` or
  configured model storage. Native source stays in `src/`.
- Every completed task needs evidence. Missing hardware/model, OOM, incomplete
  variants and unqualified conditions remain explicit. Human visual approval is
  distinct from operator correctness and does not follow from earlier experiments.

## M0 — Freeze reference, representation and experiment

- [x] LU001 Pin the documented LBH code/model revisions, verify the BF16
  safetensors artifact and tensor inventory, record separate code/weight notices
  and normalization metadata. Record current source/build/environment and a
  complete baseline CUDA golden pass; do not load `.pth` in production.
- [x] LU002 Trace native and pinned ComfyUI latent domains end to end. Produce
  paired raw/normalized video and stereo-audio fixtures and prove the adapter to
  the upstream wrapper. Freeze wrapper-versus-network normalization and casts;
  reject a guessed extra normalization or an unrecorded removal of it.
- [x] LU003 Define layer/whole-network fixtures, dtype-aware numeric tolerances,
  whole-volume GroupNorm axes/epsilon, convolution padding, interpolation
  coordinates and scale embedding. Cover random, constant, impulse, edge and real
  clean inputs; freeze thresholds before implementing candidate kernels.
- [x] LU004 Audit high-canvas limits, token/index bounds, DiT/VAE workspaces and
  both backends' memory admission. Specify the larger geometry profile and AV/
  sampler compatibility extensions. Establish bounded probes before full renders.
- [x] LU005 Freeze raw-conditioning capture/retarget schemas, reference sizing
  policies, keyframe indices, temporal phases and RoPE reconstruction. Identify
  which text/vision features are truly independent of target geometry.
- [x] LU006 Create the exact 14-output comparison manifest, timing boundaries,
  prompt/seed, shared states/noise, failure ledger, output paths and timeouts.
  Freeze K=0/2/4 and sigma=0.25 for the comparison before visual review.

Exit: reproducible reference and native contracts, feasible resource plan and a
fixed comparison manifest. M1–M4 consume these decisions.

## M1 — Capture portable clean source state

- [x] LU007 Add owned C source/options/plan types and zero/off-compatible CLI/API
  options for capture, upscaling and state-only delivery. Validate mode conflicts,
  paths, unsupported compositions and capability errors before model loading.
- [x] LU008 Implement `.h3up` capture after completed AV denoising and before
  VAE delivery. Retain clean AV, raw pre-augmentation conditions, semantic layout,
  presentation, original schedules/seed and model/execution provenance; omit
  weights and disposable GPU/prepared/history data.
- [x] LU009 Implement bounded versioned container I/O using existing checkpoint
  integrity/atomicity conventions. Validate stage, required sections, counts,
  hashes, finiteness and clean sigma endpoints before large allocations.
- [x] LU010 Support explicitly validated import of completed current dense
  text-only `.h3sample` sources. Reject incomplete or conditioned legacy samplers
  missing raw conditions, bare `.h3av` and MP4 input without fabricating data.
- [x] LU011 Wire state-only source generation and normal save-plus-preview.
  Ensure a saved bundle survives decoder failure, owns all its arrays and remains
  usable after moving it and removing the original reference files.
- [x] LU012 Test bundle roundtrips, malformed/unknown/truncated sections,
  size overflow, atomic replacement, source/output aliases, missing metadata and
  cancellation. Prove feature-off capture/allocation/serialization is unchanged;
  run shared Metal/CUDA tests and the complete golden gate.

Exit: a self-contained clean source; ordinary exact resume retains its contract.

## M2 — Native 3D upscaler

- [x] LU013 Add strict single-artifact safetensors loading for the pinned schema,
  tensor roles/shapes, BF16 dtype and normalization identity. Keep artifact
  acquisition checks separate from ordinary startup and H3 base-model identity.
- [x] LU014 Implement CUDA operators: noncausal padded 3D convolution, temporal
  depthwise/pointwise convolution, full-volume GroupNorm, SiLU, scale modulation,
  residual addition and spatial interpolation. Bound workspace and record plans.
- [x] LU015 Implement matching native Metal operators and reduction/cast rules.
  Do not substitute per-frame encoder normalization or claim numerical equality
  merely because the existing convolution interfaces compile.
- [x] LU016 Assemble the pinned full-context graph with eval-mode behavior,
  exact scale embedding, tensor lifetimes and input/output domain adapters.
  Preserve time and caller-owned AV buffers; no implicit chunking or sharpening.
- [x] LU017 Pass the frozen numeric fixtures on both backends, including T=1
  keyframes, 90-frame latent T=27, spatial/temporal edges and real source latents.
  Report error distributions; distinguish exact identity checks from tolerances.
- [x] LU018 Implement resource estimates, bounded buffers, model release and
  failure cleanup. Test repeated loads, interleaved contexts, OOM/cancellation and
  plan-key isolation; confirm native inference has no Python/Torch dependency.
  Build/test both backends and pass the unchanged complete CUDA gate.

Exit: measured native learned transfer, with neither hidden fallback nor a
chunked/full-context equivalence claim.

## M3 — Rebuild target geometry and conditioning

- [x] LU019 Implement exact 2× planning and the explicit larger-canvas profile.
  Audit host/CLI limits, packed offsets, RoPE, workspace sizes, indices and target
  admission; retain ordinary canvas adaptation/caps and exact temporal metadata.
- [x] LU020 Add versioned larger-geometry AV/sampler validation and final
  presentation/provenance records. Keep old formats readable; old readers must
  reject unsupported new states. Retain exact 1920×1088 render dimensions.
- [x] LU021 Retarget raw FL2VA keyframes and Ref2VA image/video latents according
  to stored native sizing policy; preserve unchanged intrinsic reference grids,
  original-size caps, reference order and audio pairing. Augment exactly once.
- [x] LU022 Rebuild target segments/positions, masks, RoPE and schedule maps;
  reuse only proven geometry-independent conditioning. Invalidate all old
  prepared DiT/AdaLN/attention/reuse state. Never interpolate packed row arrays.
- [x] LU023 Verify no-reference, first/last/both keyframes and ordered mixed
  image/video/audio references with bounded fixtures on CUDA and Metal. Test
  missing/unsupported retarget data, endpoint/identity geometry and unchanged
  audio conditions; pass ordinary tests and the complete CUDA gate.

Exit: valid target conditions and larger layouts; no dropped references or
accidental reuse of source geometry.

## M4 — Low-noise video refinement with frozen audio

- [x] LU024 Implement the recipe's independent K/start-sigma controls and
  short flow-coordinate schedule. Serialize canonical F32 sigmas and fresh
  video RNG identity/noise. Test K=0/2/3/4, invalid/nonfinite sigma, endpoint
  rounding and independence from the original source's step count.
- [x] LU025 Implement clean audio timestep/row handling and zero audio noise/
  Euler writes while preserving joint attention visibility. Avoid sigma-zero
  division. Assert audio byte identity after each transition, cancellation and
  save; ordinary audio sampling remains unchanged.
- [x] LU026 Integrate the new refinement stage into native CUDA/Metal DiT
  execution with the existing backend's velocity convention and update order.
  Rebuild schedule-specific preparation; K=0 executes no DiT and consumes no RNG.
- [x] LU027 Enforce immutable stage policies and clear all old execution caches
  at resolution changes. Verify dense→upscale→dense request isolation, executed
  block/evaluation counters and unsupported-option rejection.
- [x] LU028 Add the required refinement checkpoint section and resume dispatch.
  Save initialized target state, transformed conditions, audio policy, parent/
  recipe identities and complete schedule. Resume without parent/upscaler access,
  new draws or rerunning transfer; reject conflicting controls and old readers.
- [x] LU029 Prove exact same-backend resume at 0,1 and K−1 and cancellation/
  recovery with identical AV bytes and decisions. Cover malformed stage/schedule,
  audio tampering, allocation failure and stale prepared keys; run the complete
  unchanged CUDA gate and relevant Metal tests.

Exit: the larger DiT runs only the requested refinement evaluations; audio and
resume invariants hold through the full job lifetime.

## M5 — Delivery and implementation qualification

- [x] LU030 Deliver/save high-resolution clean AV via the existing full VAEs,
  streaming tiles and mux. Decode saved results without upscaler/model-condition
  dependencies. Verify geometry, frame count, FPS, PCM equality and no AV drift.
- [x] LU031 Finish CLI help, actionable errors, source inspection/provenance and
  state-only behavior. Test ordinary/decode/upscale/resume modes and API lifetime
  ownership; update README with implemented examples and actual restrictions.
- [x] LU032 Run bounded native probes for both target canvases on each backend,
  including memory admission, VAE edges/tiles and reference retargeting. Keep
  each test at most six evaluations; report missing resources as incomplete.
- [x] LU033 Run operator/state/CLI/lifecycle suites, Metal/CUDA builds and
  ordinary tests. Exercise larger-profile disabled/unsupported paths and prove
  source capture plus feature-off generation retains prior results.
- [x] LU034 Pass the full unchanged 204-output CUDA regression on the frozen
  implementation and record source, binary, golden-manifest and environment
  identities. All required backend/canvas/reference gates must pass before M6.

Exit: implementation qualified for the stated scope, ready for visual comparison.

## M6 — Execute the fixed comparison

- [x] LU035 Implement a serial manifest runner for both pairs and all seven
  artifacts per pair. Scope 50-step permission to source/direct subprocesses;
  retain source bundles, shared learned transfers/noise, full stage telemetry,
  commands, attempt status and immutable identities.
- [x] LU036 Test accounting/dry-run paths: exact case count, K transitions,
  shared-cost attribution, no duplicate successful render, stale-output rejection,
  immutable manifest and failure retention. Pass the full CUDA gate after tooling
  changes; freeze source/build before any successful comparison output.
- [x] LU037 Run L0/D0/P0/I4/U0/U2/U4 once at 672×384 → 1344×768 using the frozen
  90-frame/50-step protocol. Confirm actual operator/refinement dispatch, exact
  preserved audio, full decoding and all seven playable outputs.
- [x] LU038 Run the same seven artifacts once at 960×544 → 1920×1088 after the
  larger-profile gate. If D0 or another case fails, retain the attempt and leave
  this task incomplete; no substitute resolution, frame count or precision.
- [x] LU039 Compute paired source-consistency, temporal/detail diagnostics and
  PCM/AV checks from existing outputs; compare D0 perceptually without treating
  same-seed cross-resolution renders as exact ground truth. Report both later-job
  and end-to-end costs, memory and slower/failed cases with no performance reruns.

Exit: 14 inspectable artifacts from one frozen build, with honest cost accounting.

## M7 — Publish and close

- [x] LU040 Publish local synchronized HTML playback, crops/worst frames, JSON
  and CSV using the existing videos. Include recipe/source/model identities,
  raw logs, stage timing, memory, numerical results, limitations and failures.
- [x] LU041 Verify all local assets, hashes, media readability, counts and links.
  Record the user's explicit visual acceptance against the reviewed report and
  fourteen artifact hashes, separately from correctness passes.
- [x] LU042 Update design/README with implemented support, measured quality/cost,
  settings guidance or lack of benefit, and remaining unqualified
  scope. Do not claim official Regenerate-2K parity or generalize one seed.
- [x] LU043 Audit every task's evidence and source identity against the last
  full CUDA gate; rerun after any subsequent code/tool change. Publish the final
  record, retain opt-in defaults and preserve all prior experiment archives.

Exit: completed native feature and reproducible evidence. Visual acceptance is
reported independently and never inferred from implementation completion.
