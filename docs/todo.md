# CUDA rendering to 1920×1088 and learned 2× upscaling to 3840×2176

Status: **complete — 24/24 tasks complete**.

[Acceptance evidence and validation limits](cuda/large-canvas-results.md).
Updated **2026-11-22**.

Extend ordinary CUDA video generation, including `--render`, to
**1920×1088 (2,088,960 pixels)**. Extend the current learned spatial 2×
upscaler to accept that source and deliver **3840×2176 (8,355,840 pixels)**,
including its existing K=0/2/3/4 refinement choices. New larger-canvas
capabilities are **CUDA-only**. Development video tests must be **at most
three seconds**, including sources, references, decode-only runs and final
concatenated script output. Required CUDA acceptance must fit the existing
**RTX PRO 5000 72 GB**: use **22 frames by default**, and **39 frames only
after memory preflight**. Reserve 56 frames for the minimum valid native
continuation/bridge case described below; do not use it for upscale acceptance.

**Continuation/bridge runtime tests are capped at 1344×768.** This covers
their source, sampled/context and delivered canvases, including saved-state
replay and decode-only tests. Allow the portrait transpose with the same
bounds: long axis ≤1344, short axis ≤768, area ≤1,032,192 pixels. Larger
continuation paths receive host-only shape/index/state/allocation checks;
their end-to-end behavior remains **unverified**.

**Upscaling must be tested to the maximum 3840×2176 output**, or its transpose,
from native 1920×1088 sources: use **22 frames**, or **39 after preflight**.
This includes K=0/2/3/4, saved-state replay and full decoding. The continuation
resolution cap does not apply to these upscale jobs. Direct native 1920×1088
rendering also remains in the runtime matrix.

The completed **20/20 empty-latent still milestone** is
[archived byte-for-byte](empty-still-todo.md), with its
[acceptance report](features/empty-still-results.md). Existing
[upscale tasks](experiments/latent-upscale-tasks.md),
[results](experiments/latent-upscale-results.md) and
[format contract](features/upscale-source-format.md) remain the baseline.
Implementation and bounded CUDA acceptance are complete. The linked acceptance
record identifies the unrun long-video cases, measurement limits and the earlier
synthetic-fixture duration violation. The subsequently requested full immutable
regression passes **204/204** on 2026-11-22.

## Scope and acceptance contract

- Native generation uses the requested canvas directly. CUDA accepts the new
  area ceiling with a maximum 1920-pixel axis for the extended profile,
  including portrait transpose. Preserve already valid legacy canvases.
  Dimensions stay positive multiples of 32; existing 24 FPS and frame-grid
  rules remain. Native generation above this ceiling is outside this milestone.
- The existing pinned learned network doubles both source axes, with output
  area at most 8,355,840 pixels and axes at most 3840, including portrait.
  Time and source audio are unchanged. Support K=0 and existing K=2/3/4
  refinement, seeded noise, save/resume and full VAE delivery. This does not
  introduce temporal upscaling, a replacement network or pixel-only enlargement.
- New source and target profiles must be explicit and checked before expensive
  allocation. Keep legacy profile/artifact meanings stable; do not globally
  raise `H3_MAX_PIXELS` or reuse the existing upscale-only policy as an ordinary
  generation shortcut. Extended ordinary generation needs its own policy.
- Metal retains its current native/still/upscale capabilities. Reject only the
  new CUDA-only capabilities on Metal, before model loading or large allocation,
  including when reached through saved states, decode-only or server requests.
  Do not disable the already supported Metal 2× upscaler.
- Cover prompt-only and existing supported conditioning, native continuation
  and bridge/script behavior. Extend previews for native 1920×1088. Upscaling
  retains its current full-VAE requirement and existing unsupported-option
  checks; this milestone does not promise new LoRA, quantization, cache or
  attention-mode combinations merely because a larger canvas is accepted.
- Preserve the current learned arithmetic, full temporal/spatial normalization,
  global attention, source conditioning ownership and audio preservation.
  Safe row batching may bound independent work; splitting the model into
  independently normalized/attended temporal or spatial tiles is not equivalent.
- Support long-duration geometry and safe admission through the existing
  362-frame ceiling. A valid shape still requires sufficient memory. Do not
  silently resize, drop references/refinement, lower depth or shorten duration.

## Memory planning: 5, 10 and 15 seconds

The [memory estimate and derivation](cuda/large-canvas-memory-estimates.md)
contains tensor counts, measured baselines, empirical extrapolations, host RAM,
decoder storage and current indexing barriers. Values below are conservative
**capacity budgets in GiB**, including planner reserve; they are **not measured
new peaks**. Assume dense BF16, all 50 blocks, 512 text rows, no references,
streamed full-VAE delivery and no approximation caches.

| Requested duration | Aligned frames / actual duration | Native 1920×1088: resident / streamed | 2× upscale K=0, later job only | 2× upscale K=2/3/4: resident / streamed |
| --- | --- | ---: | ---: | ---: |
| 5 s | 124 / 5.167 s | 72 / 34 GiB | 8 GiB | 156 / 118 GiB |
| 10 s | 243 / 10.125 s | 99 / 60 GiB | 13 GiB | 262 / 224 GiB |
| 15 s | 362 / 15.083 s | 125 / 87 GiB | 18 GiB | 368 / 330 GiB |

K=0 omits transformer refinement. An end-to-end source→K=0 workflow requires
the larger of the source-generation and later-job peaks. Partial residency lies
between the two weight modes. The 5–15 second refined upscale budgets exceed
the current 72 GB test GPU even with fully streamed weights; source-duration
and memory admission must remain separate from geometric support.

Plan approximately 24–32 GiB host RAM without the streamed-weight cache,
64 GiB for cached native generation and 80–96 GiB for cached refined upscaling,
under the documented working-memory/headroom assumptions. These are planning
envelopes for all three durations, contingent on bounded frame delivery.
Actual conditioning and retained caches require additional accounting.

## Acceptance on the existing 72 GB CUDA card

A read-only capacity check on **2026-11-22** reported **73,415 MiB total
(71.69 GiB)** and **72,787 MiB free (71.08 GiB)** on the RTX PRO 5000 72 GB.
Host RAM was approximately 187.5 GiB total / 182.2 GiB available. These are
planning snapshots, not reservations; recheck actual free GPU/host memory
before every job. Measured short native/upscale results are recorded in the
[acceptance report](cuda/large-canvas-results.md); the table below remains a
conservative planning budget.

Use this bounded matrix; budgets use the assumptions/formula above and round
up to whole GiB, including reserve. Portrait equivalents use the same budgets.

| Workload | Required default | Optional extended case | Placement | Estimated VRAM capacity |
| --- | --- | --- | --- | --- |
| Native 1920×1088 | 22 frames / 0.917 s | 39 frames / 1.625 s | Auto; explicitly stream if needed for the test budget | Resident 49 / 53 GiB; streamed 11 / 15 GiB |
| Learned 2×, 1920×1088 → 3840×2176, K=0 | 22 frames | 39 frames | No refinement transformer | About 8 GiB for either later job; source generation is separate |
| Learned 2×, 1920×1088 → 3840×2176, K=2/3/4 | 22 frames | 39 frames | Explicit fully streamed BF16 weights for the baseline | 27 / 42 GiB; resident comparison 65 / 80 GiB |
| Continuation/bridge at at most 1344×768 | 39-frame source; 56-frame target with 39 preserved | None | Auto or explicit streaming after preflight | Target 50 GiB resident / 12 GiB streamed, before added references |

Set **`H3_CUDA_WEIGHT_MODE=stream`** for the required maximum-canvas refinement
baseline, including resume subprocesses. Resident refinement at this canvas
is outside the 60 GiB development budget even at 22 frames; do not force it.
An optional auto-placement comparison must independently fit the test budget.
K=0 and every positive refinement mode remain separate required cases;
selecting K=0 is not a substitute for testing K=2/3/4.

Development preflight must project **at most 60 GiB required device capacity**
for every phase, including its reserve, and independently pass actual-free-
memory and host-memory guards. Include actual reference/text rows, retained
contexts, VAE workspaces and encoder buffers. The 60 GiB ceiling is a test
budget, not a new production geometry or memory limit. Keep the existing
device reserve of `max(1 GiB, total/10)`; do not subtract it twice when comparing
the capacity table. Run GPU jobs **serially**, release completed phase/context
resources, and measure whole-device usage alongside tracked tensor allocations.

If a proposed 39-frame case misses preflight, select a **22-frame source and
target before generation** and record that choice. Plan reference-heavy and
quality cases at 22 frames from the start, with bounded fixture sizes/counts
and separate small tests for larger reference combinations. Do not silently
shorten an executing request, trim a longer source afterward, remove requested
references, skip refinement or reduce model depth. If a required 22-frame case
still cannot fit, fix allocation lifetimes/placement on this card; an OOM or a
request for a larger GPU does not complete the task. The 5/10/15-second cases
and continuation above 1344×768 remain host-only estimates or metadata checks
and never become runtime acceptance prerequisites. Maximum-resolution upscale
acceptance is required at 22 frames, with 39-frame extensions after admission.

## Development test limit and regression exception

**No development test may generate, upscale, decode or assemble a video longer
than three seconds.** This applies to both new tests and invoked existing
suites, including hidden defaults and intermediate/source/reference clips.
Use **22 frames by default and 39 only after preflight**. The sole 56-frame
real-model exception is native continuation/bridge: the existing contract needs
at least 39 context frames and a target strictly longer than its context.
A 39-frame source plus a 56-frame continuation retaining 39 frames delivers
56 total frames / 2.333 seconds. A 22- or 39-frame continuation target cannot
exercise that contract. Keep this exception at **at most 1344×768** and within
the same 60 GiB development budget; do not upscale its output in this campaign.

A 72-frame or `--seconds 3` request rounds to 73/24 = 3.042 seconds and is
forbidden. Check aligned frames and actual output duration before launching
work. Limit final concatenated script output to 72 delivered frames; two
independent 22-frame segments are a suitable short merging case. No ordinary
generation/upscale test should default to 56 frames or a longer clip.

The 5/10/15-second cases are **arithmetic, metadata-only and allocation-plan
checks**, without generation, full-duration GPU tensors or video decoding.
They remain unverified for end-to-end quality, speed and measured peak memory.
Use small tensors and logical offsets to exercise large-index boundaries.

[CONTRIBUTING.md](../CONTRIBUTING.md) requires the immutable 204-output CUDA
regression after each coherent code change. Its **124-frame / 5.167-second**
render was initially deferred under the user's three-second cap. The user
subsequently explicitly requested the full gate, authorizing that unchanged
case. It passes **204/204** on 2026-11-22 against the final standalone binary.
See the [follow-up evidence](cuda/large-canvas-results.md#full-immutable-regression-follow-up).
No inputs, durations or golden hashes were altered. This authorization applies
to the immutable gate; the bounded large-canvas matrix retains its duration and
continuation limits. Audit other suite defaults before invoking them.

Execution deviation: early broad host-suite runs inadvertently created long
synthetic FFmpeg clips and merges, violating this cap. Those runs are excluded
from bounded acceptance. The dedicated `test-large-canvas-host` target excludes
the two long-media runners; subsequent real-model jobs use the short guard.
See the [acceptance record](cuda/large-canvas-results.md) for retained failures,
corrections and measurement limits.

CUDA runtime tests use the already authorized remote CUDA test machine through
a locally configured SSH alias such as `cuda-test`; keep private connection
details out of committed files. Local macOS work is for document/host checks,
build checks and short legacy/rejection checks, not new high-resolution Metal
qualification. No long render is required to complete this planned milestone.

## M0 — Freeze policy, resource accounting and bounded tests

- [x] CLC001 Freeze the backend/geometry/feature matrix above against the
  current source and model identities. Define distinct extended-generation
  and extended-upscale profile identities, old-reader behavior and backend
  rejection rules. Specify transpose/area/axis boundaries, native and learned
  output provenance, public entry points and which existing optional modes
  remain supported or explicitly rejected. Record immutable baseline artifacts.
- [x] CLC002 Implement a shared checked resource estimator for actual video,
  audio, text/reference rows, weights/residency, DiT intermediates, learned
  network buffers, sampler/state copies and decoder delivery. Return separate
  host/device estimates by phase. Reproduce the 5/10/15-second calculations
  without allocations or model work; expose actionable preflight diagnostics.
  Reproduce the 22/39-frame matrix for the actual 72 GB device, including
  streamed refinement at 3840×2176 and the 56-frame continuation exception at
  no more than 1344×768. Reconcile with short phase measurements; enforce
  the 60 GiB development capacity budget.
- [x] CLC003 Add a development-runner duration guard before launches. Validate
  effective aligned frames, generated context, reference/source media and final
  script duration; default to 22, admit 39 only after memory checks and allow
  56 only for the named continuation/bridge test at no more than 1344×768.
  Enforce the continuation resolution cap on source, latent/context and final
  canvases, including subprocesses, resume and decode-only. Admit upscale
  sources to 1920×1088 and outputs to 3840×2176, including transpose, only at
  22 or 39 frames and after capacity/index checks; distinguish the job kinds.
  Recheck free GPU/host memory, carry streaming policy into resume workers
  and serialize GPU tests.
  Reject defaults that exceed the cap, including `--seconds 3`.
  Inventory applicable test commands and classify long suites as not runnable
  under this request. Preserve the immutable regression and explicitly record
  its duration-limit exception; do not edit global policy to hide the gap.

Exit: a precise CUDA-only contract, reproducible estimates and a test runner
that cannot accidentally launch long video jobs.

## M1 — Shared geometry and artifact contracts

- [x] CLC004 Add checked, backend-aware geometry policies and centralize their
  selection for generation, source capture, upscaling and decode. Audit
  [host geometry](../src/host.c), [engine validation](../src/engine.c),
  [AV state shapes](../src/sampling/av_state.c) and every hardcoded/shared cap.
  Keep still geometry, legacy reference preprocessing and legacy Metal limits
  intact. Reject unsupported backends before large allocations or downloads.
- [x] CLC005 Route CLI, public API, HTTP/server planning and
  [script planning](../src/cli/render_plan.c) through the policy. Replace the
  renderer's hardcoded `768*1344` check; carry backend/profile, output canvas
  and internal render canvas consistently through dry runs and segment commands.
  Keep `--ref-image-size match|high|max` distinct from output sizing and verify
  inheritance/overrides reach the pipeline. Match help and validation errors.
- [x] CLC006 Extend AV, sampler, upscale-source and presentation serializers
  with explicit profile/backend semantics. Audit all readers, continuation
  compatibility, codec-only delivery and resume before geometry-dependent
  allocation. Preserve old supported artifacts and fail clearly for unknown
  profiles or new CUDA artifacts on Metal. Round-trip exact shape, provenance,
  decoder recipe, source audio and refinement state; reject forged profiles.

Exit: all front doors agree on allowed dimensions and backend, and saved state
cannot bypass the policy or silently change a legacy artifact's meaning.

## M2 — CUDA index safety and memory admission

- [x] CLC007 Audit DiT/GPU allocation products, element counts, byte offsets,
  launch grids and signed/unsigned narrowing end to end. Use the actual
  `INNER=7168` QKV width, correcting the upscale planner's `5376*3` proxy.
  Native 15-second and all requested 4K refinement QKV volumes exceed U32
  element indexing. Implement checked wide indexing or bounded equivalent
  operations, including FC1/SwiGLU, projection/scatter, normalization, RoPE,
  attention layouts, sampling and serialization. Fail before unsupported calls.
- [x] CLC008 Bound independent linear/MLP/projection workspace where needed
  for large counts, without changing reductions, precision or global attention.
  Audit cuBLAS/attention interface limits and every intermediate, including
  large byte offsets for tensors whose element count still fits U32. Test
  overflow and boundary dispatch using small windows/logical indices; do not
  allocate a long-video GPU tensor to test arithmetic.
- [x] CLC009 Extend sequence/conditioning preparation and CUDA attention to
  accept the new profiles. Count all text, anchor, reference and audio rows in
  admission; preserve temporal positions and normalization across the full
  sequence. Validate actual reference-encoder limits independently. Establish
  explicit unsupported-mode errors where the existing mode is unqualified;
  never use truncation or implicit token/reference reduction to fit.
- [x] CLC010 Integrate CLC002 with automatic/resident/streamed weight planning,
  host cache admission and the process guard. Account for overlapping resources
  in reused contexts, checkpoint export and preview/decode transitions. Bound
  transfer/decoder lifetimes and retries. Report required versus available
  memory and phase on failure. Retain headroom; reduce conservative reserves
  only after demonstrated allocation-lifetime evidence, with no OOM-driven
  precision or geometry fallback. Required 22-frame native 1920×1088 and
  K=0/2/3/4 upscale-to-3840×2176 acceptance must succeed on the existing 72 GB
  device; use explicit streaming as the refined-upscale baseline and keep
  39-frame extensions conditional. Keep larger continuation and 5/10/15-second
  admission checks host-only. Larger hardware is not a completion requirement.

Exit: all long-duration shapes can be evaluated safely by host planning; a
resource rejection is distinct from integer overflow or unsupported geometry.

## M3 — Native rendering and the existing learned upscaler

- [x] CLC011 Enable full native 1920×1088 and transposed CUDA generation through
  direct CLI/API/server and `--render`. Preserve seeded sampling, audio,
  first/last anchors and ordered supported references. Extend ordinary hard
  continuation and bridge geometry/buffer checks, including the fixed
  `H3_MAX_PIXELS` scratch assumptions in [bridge.c](../src/sampling/bridge.c).
  Verify independent-segment rendering/merging at the new canvas. Execute
  continuation/bridge/context merging only at or below 1344×768; verify its
  larger profiles through host-only checks and disclose the runtime gap.
- [x] CLC012 Extend native preview/full-VAE handling to the new generation
  limit, including [tiny VAE admission](../src/vae/tiny_vae.c), memory reserve,
  callbacks and decode-only. Keep the selected decoder explicit. Cover spatial
  tile edges, portrait output, preview cancellation and exact dimensions;
  unsupported combinations must fail before sampling rather than at delivery.
- [x] CLC013 Extend clean upscale-source capture to native 1920×1088. Update
  source/profile validation and ownership so source bundles retain complete
  raw conditions, audio and metadata at the new size. Preserve current capture
  restrictions, atomic saving, input/output collision checks and portability
  without reopening deleted original reference media.
- [x] CLC014 Extend [upscale planning](../src/upscale/upscale_plan.c), learned
  network and explicit bilinear comparison bounds from 128-latent axes to
  at most 240 per latent axis and 32,640 latent pixels, admitting the 240×136
  target, its transpose and other valid source aspect ratios. Audit CUDA products,
  normalization, resize and convolution packing; preserve the pinned network,
  full-context statistics and exact spatial 2× behavior. Keep Metal's existing
  execution bounds. Do not introduce independent normalization of chunks.
- [x] CLC015 Retarget bundled image/keyframe/video conditions consistently to
  the enlarged canvas, including the separate latent-axis checks in the
  retarget path. Preserve `match|high|max` semantics, original source sizes,
  ordering, temporal phase and once-only augmentation; high/max references
  must not acquire an unintended extra scale. Include their rows and network
  lifetimes in the resource estimator. Preserve source audio bytes exactly.
- [x] CLC016 Extend K=0 and dense K=2/3/4 execution, sigma/noise initialization,
  state-only delivery, stop/cancel/recover and exact same-backend resume to the
  enlarged target. Keep transfer separate from refinement admission; K=0 must
  not load the transformer or sample noise. Preserve parent/artifact hashes,
  audio identity and output provenance. A low-memory device must never silently
  replace requested refinement with K=0.
- [x] CLC017 Implement and qualify full video-VAE/FFmpeg delivery at 3840×2176
  and transpose using 22 frames, or 39 after admission.
  Fix the CUDA stitcher's more-than-16-tiles rejection for the 20-tile axis
  using a valid bounded fallback or qualified larger device plan. Preserve
  blend order and temporal overlaps; stream chunks instead of retaining the
  whole movie. Check codec dimensions, audio sync, exact frame count, output
  publication, progress and cancellation at the maximum upscale canvas.
  Include full saved-upscale decode-only replay under the same frame/memory
  guards. Upscale continues to require full VAE. Keep longer duration
  tile/shape/index/allocation checks host-only.

Exit: short native renders and source→learned-2×→full-video delivery work on
CUDA at both maximum generation/upscale canvases, with old Metal behavior
preserved. Larger continuation has host coverage and a disclosed runtime gap.

## M4 — Verification within the three-second ceiling

- [x] CLC018 Extend host/CLI/state tests for both profiles, transposes, area/axis
  boundaries, malformed/truncated/nonfinite artifacts, profile forgery and
  backend rejection. Test all 5/10/15-second row/byte estimates, U32/signed/size
  boundaries and impossible admissions using arithmetic/metadata only. Verify
  the three-second runner rejects 73/90/124/243/362-frame real-model jobs and
  overlong merged output, rejects 56-frame upscale/default jobs, and allows
  the named 56-frame continuation at ≤1344×768 only after capacity admission.
  Test rejection of continuation above 1344×768, including resuming a larger
  state and codec-only attempts. Ensure the guard permits maximum-resolution
  22/39-frame upscaling and rejects longer or over-budget upscale jobs, including
  output-only geometry overflow. Test the 60 GiB budget and 39→22 selection
  without launching oversized jobs. No high-duration GPU work is hidden in
  fixture setup.
- [x] CLC019 Run 22-frame CUDA native cases at small and maximum canvases;
  add 39-frame coverage only after memory admission. Cover
  text, ordered references and all image-size modes, anchors, same-seed repeat,
  save/re-decode, sampler resume, preview/full output, API/server and scripts.
  Use a 39-frame source and 56-frame continuation for a bounded continuation/
  bridge/merge case at no more than 1344×768, admitted under the named exception.
  Larger script cases must use independent segments, not continuation.
  Plan heavy-reference and full-depth quality runs at 22 frames; select
  streaming before launch where needed. Retain dimensions, token counts,
  audio/frame metadata, phase memory and timings. Record visual observations
  separately from numerical/lifecycle checks.
- [x] CLC020 Extend independent pinned-network operator/transfer comparisons
  for the new spatial bounds, rectangular/portrait edges and full-context
  normalization, using T=7 by default, at most T=12 after admission, and small
  adversarial fixtures. Include actual maximum-canvas transfer evidence;
  keep higher temporal/index-boundary checks host-only. Run reference and
  native GPU computations serially with the same memory ceiling; avoid oracle
  allocations that defeat the short test plan. Freeze tolerances before
  comparison. Verify the implementation against the existing
  independent reference, including normalized input/output, interpolation and
  retargeted conditions. Native self-comparison alone is not acceptance evidence.
- [x] CLC021 Run bounded 1920×1088→3840×2176 cases and transpose with
  K=0/2/3/4, full VAE delivery, reference retargeting, saved-source relocation,
  and exact pause/resume at each refinement boundary. Verify finite pixels,
  source audio/PCM identity, target dimensions and actual duration. Require
  22-frame success for every K mode at the maximum landscape/portrait target
  on the 72 GB card, using explicit streamed BF16 for K>0, including resume.
  Add a 39-frame cross-chunk case only if the measured/estimated phase budget
  fits; otherwise run that case at 22 frames and retain separate short decoder
  boundary checks. Apply the same 22/39-frame and memory guards to K=0,
  API/operator forwards, resumed refinement and decode-only work. No 56-frame
  upscale case is required or allowed here. Keep all production layers and
  requested refinement; distinguish resource
  rejection from an accepted render. Never run the old 90-frame campaign.
- [x] CLC022 Exercise memory denial, allocation/encoder failure, cancellation
  and repeated/reused contexts across native→upscale→native jobs. Use bounded
  CUDA sanitizer probes for new index/launch paths and small-window overflow
  tests. Check cleanup, allocation plateaus, original output preservation,
  valid checkpoint recovery and guard headroom. Calibrate CLC002's separate
  phase estimates against short-run device/RSS measurements and record gaps.
  Use 22-frame lifecycle jobs: native up to 1920×1088, upscale output up to
  3840×2176. Admit maximum-canvas sanitizer work separately; use smaller
  operator fixtures if instrumentation exceeds the budget, while retaining
  ordinary full-canvas evidence. Do not run GPU jobs concurrently or overlap
  unnecessary decoder/DiT/reference processes. Record
  actual total/free VRAM, placement and peak usage for this physical device.

Exit: actual CUDA evidence covers native 1920×1088 and every K mode at
3840×2176 on the existing 72 GB card within the frame and memory budgets.
Long-video and continuation above 1344×768 shape/index/resource checks pass
as host checks; their quality, speed and peak memory remain unmeasured.

## M5 — Short regression, packages and documentation

- [x] CLC023 Run applicable host/operator checks and a separately named CUDA
  short regression suite covering ordinary video, still, continuation,
  references, decoding and legacy upscaling. Audit and bound every invoked
  runner first. Build/test the Linux standalone package with offline short
  native/upscale/decode/resume paths on the same 72 GB card, using 22-frame
  sources and explicit streamed refinement at 3840×2176. Keep continuation
  at or below 1344×768. Check local macOS build/host compatibility
  and rejection of new profiles without qualifying new Metal canvases. Preserve
  all 204 golden hashes. Record the initial duration-limit deferral separately
  from the subsequent explicitly authorized **204/204** immutable regression
  pass; never label a shortened substitute as that gate.
- [x] CLC024 Update README/usage, CLI/API/server/script guides, hardware and
  state/upscale contracts with CUDA-only ceilings, exact 2× behavior and current
  option restrictions. Recompute the linked memory tables from final code and
  record measured 22/39-frame peaks for the 72 GB card beside long estimated
  values, explicitly identifying any unrun optional 39-frame case and the
  56-frame continuation at ≤1344×768. Disclose that larger continuation and
  5/10/15-second videos at the new maximum canvases were not executed. Require separate measured evidence
  for maximum-canvas K=0 and K>0 upscale/full delivery. Include K=0 versus
  refinement and source-generation cost. Publish source/build/device identities,
  bounded commands, visual observations and a task-by-task acceptance record.
  Validate links, archive integrity and private-data hygiene. Mark completion
  only with passing short evidence and all duration/resource/regression gaps
  explicitly disclosed; do not require a forbidden long run for closeout.

Completion means CUDA-native generation to 1920×1088 and the current learned
2× upscaler to 3840×2176 are implemented with safe indexing, honest memory
admission and compatible state/delivery behavior. All development video evidence
must fit the existing 72 GB card and respect the frame/memory budgets and
three-second ceiling. Continuation execution is limited to 1344×768; larger
continuation is host-validated only and must not be advertised as fully
qualified. Upscaling must pass at 3840×2176 with 22-frame inputs and admitted
39-frame extensions. The 5/10/15-second estimates remain labeled as estimates;
the separately authorized immutable regression is recorded as **204/204 passed**.
