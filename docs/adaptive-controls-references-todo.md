# Adaptive-cache controls and image, video, and audio references

Status: **complete — 32/32 tasks complete**. Implementation, clean CUDA/Metal
builds, retained suites, 18/18 comparison videos and the final 204/204 recorded
SGLang gate pass. The user accepted the visual/listening results on 2026-09-28. See
[the results report](cuda/adaptive-controls-references-results.md) and
[the design](cuda/design-adaptive-cache-controls-references.md). Evidence is under
`outputs/adaptive-controls-references/run01/` locally and on PRO 5000. The completed cleanup checklist is archived in
[legacy-cleanup-todo.md](legacy-cleanup-todo.md), with
[its results](cuda/legacy-cleanup-results.md). The independent server project
remains in [the server checklist](features/server-tasks.md).

## Required behavior

- Add `--adaptive-cache-threshold T` (finite decimal FP32, 0–1 inclusive) and
  `--adaptive-cache-max-hits N` (integer 1–16). Zero threshold disables hits;
  threshold 1 is an ordinary strict comparison, not unconditional reuse.
- Preserve conservative defaults 0.04/1 and aggressive defaults 0.08/3,
  warmup 4, the current warmup override and 4096 MiB default ceiling.
  Explicit flags override their selected preset without enabling caching.
- Support image, video and audio Ref2VA inputs, including ordered mixed sets,
  with BF16 adaptive alone and adaptive+SubBlock. Cover silent video, embedded
  audio, replacement soundtracks and separate audio accompanying a visual input.
  Preserve limits of 12 reference records, 9 images, 3 videos and 3 audio inputs,
  existing duration rules and image-size modes. Separate audio must accompany
  an image or video; audio-only requests remain unsupported.
- New reference decisions use the maximum of the existing global, generated-video
  and generated-audio scores, excluding reference rows from the target scores.
  Existing text-only score arithmetic and decisions remain unchanged.
- Keep frame anchors, continuation, bridge, upscale refinement, LoRA/Turbo, Metal
  and all quantized adaptive reference sets rejected. Preserve ordinary/SubBlock-only
  media support and text-only quantized conservative caching. Custom controls
  apply to supported text quantization; no additional FP8/NVFP4 video comparison
  campaign is required.
- Save effective controls, validate current-only files, and require exact matching
  numerical overrides on resume. Do not restore the old compatibility readers.
- Keep the **12 recorded SGLang fixtures and all 204 expected hashes unchanged**.
  New approximate cases belong to functional/quality qualification, not that
  manifest or an independent/live SGLang numerical oracle.

## Work and test environment

Use the **RTX PRO 5000 at `cuda-test`** for CUDA builds, integration tests,
SGLang regression and the comparison videos:

```sh
ssh cuda-test
```

Use `~/h3cli`, the existing `cuda-env.sh`, and models installed under `/models`;
resolve actual component paths during preflight. Preserve the qualified CUDA
13.0.3/cuDNN 9.20 stack unless an existing environment check proves otherwise.
Use the local M4 for shared/host tests and a clean Metal build. Do not use the
4090/5090 nodes. Keep machine paths out of production defaults and portable tests.

Follow [CONTRIBUTING.md](../CONTRIBUTING.md): complete unchanged 204-output gates
after coherent source/generator/build/test-tooling changes and after final
qualification, with fresh output directories. Missing hardware/models, skipped
cases, timeouts and stale artifacts are not passes. Ordinary test invocations
retain the six-evaluation ceiling. Only the design's four explicit 50-step
comparison subprocesses may lift it to 50. Documentation-only planning requires
content/link checks and does not run remote tests.

Evidence belongs under ignored `outputs/adaptive-controls-references/<run>/`;
executables/library remain in ignored `bin/`. Track the final report under
`docs/cuda/` and link its exact artifacts and source/build identities.

## Baseline and contract

- [x] ACR001 Record initial commit/dirty state, inventory adaptive policy callers,
  CLI/API fields, CUDA policy transport, generated code if any, prepared keys,
  reference layouts, sampler/presentation readers and tests. Capture pre-change
  text-only default decisions/latents on bounded fixtures for later preservation
  checks. Identify every reference rejection, including allocation-time guards.
- [x] ACR002 Preflight the PRO 5000, qualified libraries, models/Ref2VA partition,
  image/video/audio VAEs, FFmpeg, disk/RAM and current source synchronization.
  Inspect the image inputs and short audiovisual source clips; record reproducible
  fixture derivation, distinct soundtrack identities and duration/count admission.
  Run the complete unchanged 204-output gate
  as a baseline; keep fixture/manifest identities and raw logs.
- [x] ACR003 Confirm implementation mappings for the design: central effective
  settings, global text scoring versus max-of-three reference scoring, arithmetic
  recipe 3 for all BF16 reference media, sampler section 40 v3 and presentation 9.
  Record exact changed key/schema fields before editing; keep current-only
  readers and distinguish arithmetic identities from file versions.

Baseline evidence: initial commit `46239d010f9a345aaf37f9322bc1eb2f17541ed8`,
clean worktree; baseline `baseline-gate/result.json` passed all 204 outputs in
98.55 seconds after build (192.39 seconds including build). Manifest SHA-256
`fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3` is unchanged.
`baseline-latents/` captures both presets through six evaluations, including
hits and conservative streak refresh; `candidate01.complete` confirms exact
candidate latent equality. Inventory and mappings are recorded in the design
and the current implementation: recipe 3, required sampler section 40 v3,
presentation 9, canonical effective control keys. Frozen media derivation and
input/source hashes are in `videos01/manifest.json`.

## Controls and native execution

- [x] ACR004 Add strict locale-independent parsers, public request values and
  explicit-selection bits, long options, help and shared validation for both
  flags. Handle explicit zero threshold correctly, consume complete arguments,
  reject malformed/nonfinite/out-of-range values and unsupported operations,
  and resolve options independent of CLI order. Do not implicitly enable cache.
- [x] ACR005 Centralize effective threshold/hit-limit resolution and carry it
  through CUDA policy, DiT contexts, generation, resume and diagnostics. Replace
  hardcoded decision thresholds/streak ceilings with resolved values while
  preserving strict comparison, default text arithmetic, refresh reason priority,
  warmup/final/phase/discontinuity guards and existing quantization restrictions.
- [x] ACR006 Implement reference-media decision recipe 3 using the maximum of
  global, generated-video and generated-audio probe scores. Derive target ranges
  from validated packed layout metadata, including image/video vision spans and
  both stereo reference-audio channels; exclude condition rows from target scores.
  Require valid nonempty target ranges. Reuse the deterministic reduction and
  BF16 residual arithmetic without changing the existing text recipe.
- [x] ACR007 Admit qualified BF16 image/video/audio and mixed reference sets across
  CLI/library entry points. Preserve count/order/size/patch/duration limits,
  visual requirement for separate audio, embedded/replacement soundtrack accounting,
  released video preprocessing, Ref2VA weights and fixed visual/audio condition rows.
  Keep target-only heads/Euler updates and explicit errors for invalid media or
  excluded combinations. Preserve ordinary and SubBlock-only reference modes.
- [x] ACR008 Integrate adaptive+SubBlock references with protected image/video/audio,
  text/vision, mixed-boundary and existing target-frame query/key ranges. Keep block 0 dense; refresh at the original schedule's sparse-phase
  transition, including after resume. Hits must skip all suffix attention/router
  dispatch; executed suffix blocks retain the current sparse rules and masks.
- [x] ACR009 Audit exact reference-inclusive adaptive sizing, budget errors,
  allocation order and residency. Reserve the current three BF16 tensors and
  reduction storage once before weights; preserve attention workspace and device
  reserves. Retain bounded retries, cleanup and deferred suffix prefetch so hits
  perform no suffix weight reads/uploads, GEMMs or MLP execution.
- [x] ACR010 Update prepared/live execution keys and context reset logic with
  canonical effective controls, score recipe and ordered reference/layout identity,
  including video timestamps and embedded/replacement soundtrack selection/content.
  Explicit defaults and omission must resolve identically. Changed policy or
  references cannot reuse history; allowed budget overrides still recheck admission.
  Preserve transactional commit, cancellation and same-context recovery.

Implementation evidence: shared controls and reference admission are in
`src/denoise/approximate.c`, deterministic parsing/decision logic in `src/denoise/adaptive_cache.c`,
and resolved CUDA transport/execution in `src/cuda/cuda_policy.c` and `src/denoise/dit.c`.
Canonical live/prepared keys include controls and ordered media identity. Reader
validation reconstructs the packed layout only after reference row totals match;
section 40 v3 and presentation 9 carry resolved controls. Local `metal-final-test.log`
and `metal-final-sanitize.log` passed; `bounded-corruption.json` records 60
checksummed reference-state cases. Final-build supplementation passed 66 commands and 71 assertions. The
strengthened retained-context cases pass with dense and SubBlock attention; see
`context-policy-final/result.json`. All 18 frozen videos and the final 204-output gate pass; the linked results
record separates mechanical qualification from the user's visual/listening acceptance.

## Saved state and provenance

- [x] ACR011 Extend required sampler section 40 to v3 with resolved threshold and
  hit ceiling; update current readers/writers/inspectors and recipe identities.
  Keep envelope schema 2 and BF16 payload sections 41/42, rejecting older adaptive
  section versions without migration. Nonadaptive sampler structure stays intact.
- [x] ACR012 Replace the T2VA-only loader preallocation guard with checked current
  T2VA/multimodal Ref2VA layout validation. Validate media kinds/order/counts,
  audio-presence/stereo sizes, released-video provenance, exact reconstructed
  segments/RoPE, condition/target sizes, recipe, payload lengths and budget before
  adaptive tensor allocation. Retain checksums, finite checks, file limits and
  atomic-write failure cleanup; reject forged geometry or oversized declarations.
- [x] ACR013 Update history validation, import/export and both CLI/library resume
  paths for saved effective hit ceilings and thresholds. Omission restores saved
  values; matching explicit overrides pass, differing numerical controls fail
  before model loading. Preserve explicit zero, original-schedule warmup/phase,
  media-free resume from saved conditioning and allowed sufficient budget overrides.
- [x] ACR014 Introduce current presentation schema 9 with canonical effective
  controls and adaptive recipe. Update all producers, decoders, state constructors,
  source/upscale consumers and inspectors; disabled fields are zero. Keep AV
  schema 3, current integrity bindings and no old presentation reader. Update
  conditioning keys/consumers as needed without unnecessary envelope changes.
- [x] ACR015 Add initialization/per-step records for preset, effective threshold,
  maximum hits, score recipe, all component scores, selected decision score,
  reason, streak, executed blocks and cache bytes. Preserve machine-readable
  timing/dispatch/residency evidence and accurate reported provenance after resume.

## Functional and operator tests

- [x] ACR016 Add host/API/CLI tests for strict parsing, all endpoints, explicit
  zero, omitted and explicit defaults, argument order, disabled features, short
  schedules, max-hits 1/2/3/16, threshold equality, forced refreshes and nonfinite
  rejection. Verify unsupported reference/precision/device combinations remain
  rejected and supported SubBlock-only media combinations remain accepted.
- [x] ACR017 Extend independent CPU/GPU probe-policy tests for reference layouts,
  ragged image/video/audio ranges, zero/tiny denominators, large unchanged reference
  spans and isolated generated-audio/video changes. Prove reference-audio prefixes
  cannot dilute or enter target scores, both stereo targets count, and max-of-three
  prevents dilution. Verify deterministic reduction, correct hit reconstruction
  and that nonfinite data
  fail even during forced refreshes. Do not generate expected results with the
  production decision helper.
- [x] ACR018 Add current state round-trip and corruption cases for every new
  field, recipe/layout mismatches, removed section versions, threshold zero,
  invalid streak/refresh indices, too-small budget overrides and payload bounds.
  Cover canonical presentation serialization, CLI/API resume mismatches and
  decode without adaptive model/history. Use bounded synthetic corruption files.
- [x] ACR019 Run bounded real-model reference integration with 1, 2 and 9 ordered
  images and `match`, `high`, `max`; cover each dimension with representative
  cases rather than an exhaustive Cartesian product. Exercise dense and combined
  SubBlock, default and custom controls, threshold-zero full refresh, real hits,
  preserved encoded references and fresh generated-video/audio heads.
- [x] ACR020 Verify exact same-build uninterrupted-versus-resumed decisions and
  AV latents across warmup, actual reference-conditioned hits, maximum-hit refresh,
  the SubBlock transition and the final step. Cover image-only, video-bearing,
  audio-bearing and mixed sets; remove/hide all original media and external
  soundtracks for resume cases. Include cancellation/retry, reused-context
  reference/policy changes and malformed resume recovery; each invocation executes
  at most six evaluations.
- [x] ACR021 Confirm unchanged text defaults and explicit-default equivalence,
  using ACR001's pre-change bounded captures. Exercise custom controls in existing
  text-only precision paths with bounded operator/state coverage, and retain their
  BF16-probe/quantized-suffix rules. No FP8/NVFP4 comparison videos or newly enabled
  aggressive/triple/reference quantization combinations.
- [x] ACR022 Qualify reference-cache memory and resident/forced-streamed execution,
  including 1344×768/124 frames with one max-size image and small mixed-media jobs.
  Include video/vision/reference-audio row costs, small-budget early rejection,
  allocation/cancellation recovery and no suffix transfers on hits. Check
  1344×768/362-frame mixed-reference sizing/overflow without another full video.
  Record actual device/host/checkpoint costs and preserve reserves.
- [x] ACR023 Run bounded real-video integration for silent clips, embedded audio
  and external replacement, including a video-only visual request. Cover dense
  adaptive and combined SubBlock, both presets/custom controls, real hit/refresh
  dispatch and pinned visual/audio conditioning. Preserve released preprocessing,
  chunking/timestamps/RoPE and prove silent/replacement modes neither leak nor
  double-count the original soundtrack. Keep six evaluations per invocation.
- [x] ACR024 Run bounded real-audio/mixed integration: image+audio, video+separate
  audio, multiple audio clips, and ordered image/video/audio sets. Cover dense
  adaptive and combined SubBlock, real hits, stereo target/reference separation,
  constant encoded condition latents and generated-only Euler updates. Test
  per-kind/total count, duration and visual-input requirements at boundaries using
  host/layout fixtures plus representative real encodes, with no exhaustive sweep.
- [x] ACR025 Add media-specific state/conditioning recovery tests: changed order,
  embedded-to-silent switches, replacement soundtrack changes, missing/invalid
  streams, forged audio presence/lengths, video recipe/timestamp mismatches and
  context reuse after failure. Require exact resume without source media across
  both dense and SubBlock paths; verify audio/video provenance is checked before
  adaptive history allocation and invalid states cannot pollute the next job.

## Builds, videos and final acceptance

- [x] ACR026 Clean-build CUDA executable/library and affected test binaries on
  PRO 5000; clean-build Metal locally. Run the retained `make test`, applicable
  local host/Metal coverage and `make test-current-cuda` with fresh current state
  files. Run focused sanitizer/failure tests for changed parsers/readers/layout
  ownership. Record all outcomes; fix failures without weakening checks.
- [x] ACR027 Freeze the design's V01–V18 machine-readable manifest before runs:
  inspected images and reproducibly derived two-second video/audio fixtures,
  fixed prompts per comparison group, ordered media hashes, seed 42, commands,
  limits, output paths, one successful video per row and failure ledger. Verify
  distinguishable embedded/replacement/separate audio and the silent-mode fixture;
  derive inputs with FFmpeg from existing qualified media, not extra H3 renders.
  Implement/update the runner/report using existing helpers; scope the 50-step
  override to V01–V04 only and retain full audio/video decoding.
- [x] ACR028 Run V01–V04 once each: 640×480, 90 frames, 50 steps, one max-size
  image; dense control, conservative defaults, aggressive defaults and custom
  threshold 0.06/max-hits 2. Save total wall/denoise time, all cache decisions,
  dispatch/residency/transfer/memory observations and media/state artifacts.
- [x] ACR029 Run V05–V09 once each: the 1344×768/124-frame/six-step dense,
  custom-adaptive and adaptive+SubBlock triplet with one max-size image; then
  the 640×480/90-frame/six-step dense and combined pair with two high-size images.
  Use the design's exact warmups/thresholds/hit ceilings. Require actual hit
  coverage in a real reference job or a separately labeled bounded diagnostic;
  never retune/re-render the fixed comparison to manufacture a speedup.
- [x] ACR030 Run V10–V18 once each: three 640×480/90-frame/six-step triplets
  for embedded-video audio, image+separate audio, and the design's ordered mixed
  set containing silent video and an external soundtrack. Each triplet compares
  dense control, custom adaptive and custom adaptive+SubBlock. Use frozen commands
  and existing controls; capture all timing/media/cache/dispatch evidence. Require
  actual hits and exact replay for video/audio/mixed layouts in real jobs or
  separately labeled bounded diagnostics; never retune the comparison videos.
- [x] ACR031 Publish one synchronized 18-video report with commands, identities,
  total wall times, mechanical media checks, cache/dispatch evidence and video/audio
  similarity diagnostics against the reused controls. Inspect reference fidelity,
  motion, artifacts, soundtrack influence and AV timing; distinguish observations
  and human visual approval from automated test results. Update README/help,
  adaptive contracts,
  current state documentation and the planned server feature inventory. Correct
  touched documentation's obsolete compatibility claims without unrelated cleanup.
- [x] ACR032 Run the complete final 204/204 recorded SGLang gate after all source,
  test-tooling and build changes and qualification runs. Verify the manifest and
  12 fixtures are unchanged and local/server candidate identities agree. Publish
  the tracked results record linking builds, suites, videos and final gate; audit
  flags/links/examples and close every task with evidence or an explicit unresolved
  result. Do not mark incomplete work or visual approval as passed.

Final evidence: `videos01/result.json` passes 18/18 with no retries;
`final-gate/result.json` passes 204/204 in 140.32 seconds after build. Final source
SHA-256 is `48c8f921c908000ff034f91bf62c1000bcc603d9f1f2d0ccc386cac7b742df7b`.
The report records all wall/denoise times, actual hits, byte-identical no-hit
controls, streaming/checkpoint costs, sampled-frame limitations and media links.
The user accepted all 18 comparison results on 2026-09-28: “the results are acceptable”.
