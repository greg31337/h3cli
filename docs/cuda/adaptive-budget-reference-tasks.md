# Adaptive-cache budget and SubBlock reference media

Status: **complete — 34/34 tasks**: 22 adaptive-budget tasks and 12
SubBlock reference-media tasks. See the [qualification report and videos](../cuda/adaptive-budget-reference-results.md).
The implementation follows the
[design and compatibility contract](../cuda/adaptive-cache-budget-design.md): a
4096 MiB default for new adaptive requests and `--adaptive-cache-max-mib N`,
including safe GPU admission and large checkpoint/resume support. The required
target is **1344×768 with 362 frames** on RTX PRO 5000.
The appended SubBlock work adds image, video and audio references with adaptive caching off,
including an actual sparse six-step render at the same large target.

The completed fast-preview campaign is preserved in its
[20-task archive](../cuda/fast-preview-tasks.md) and
[results/video report](../cuda/fast-preview-results.md). Its 26 videos, 13 blocked
slots and user-skipped SOL slot remain historical evidence. Do not rewrite
those results as though the larger budget had existed during that campaign.

## Scope and completion criteria

- Change adaptive memory admission and resource metadata while preserving
  numerical recipes, thresholds, reduction order, warmups, final refresh,
  hit streaks and feature compatibility.
- Use PRO 5000 for CUDA and local M4 for CPU tests, Metal builds and review.
  Preserve CUDA 13.0.3/cuDNN 9.20, device reserves, attention workspace and
  existing weight residency/streaming policy. No SOL renders or new quantized
  comparison campaign are required.
- Keep the 4096 MiB new-request default distinct from legacy checkpoints'
  512 MiB. A budget is a ceiling on exact required cache bytes, not an eager
  allocation. Budget changes alone must not change admitted numerical results.
- The adaptive-budget work keeps references/continuation and other unsupported
  adaptive combinations rejected. The appended SubBlock tasks separately allow
  image/video/audio references with adaptive caching off; they do not enable Metal support.
- Run the unchanged 204-output CUDA golden gate after every coherent code,
  build or test-tool change, per [CONTRIBUTING.md](../../CONTRIBUTING.md). Do not
  regenerate goldens, add an independent parity oracle or skip cases.
- Keep bounded tests at six evaluations per invocation. The complete target
  video also uses six steps, full VAE and audio; preserve resolution and 362
  frames. This is capacity qualification, not a quality/speedup claim.
- Store artifacts under ignored `outputs/adaptive-cache-budget/` and binaries
  under `bin/`. Record failures/cleanup and identify base weights by metadata.
  Required large-target failures cannot be counted as passes or optional skips.

## Implementation

- [x] ACB001 Freeze source/runtime identities and the validation manifest.
  Inventory fixed-cap uses, request fields, policy capture, prepared/live
  contexts, loader/writer, resume entry points and diagnostics. Record the
  512 MiB baseline and exact target sizing from the design.
- [x] ACB002 Separate overflow-safe shape sizing from resource admission.
  Return per-tensor, reduction, total device and persistent checkpoint bytes.
  Preserve zero allocation for cache off and exact boundary acceptance.
- [x] ACB003 Add public request fields and the 4096 MiB new-request default.
  Carry the effective ceiling through CUDA policy and DiT ownership with
  checked 64-bit/platform-size conversions; avoid process-global overrides.
- [x] ACB004 Add strict `--adaptive-cache-max-mib N` parsing and help. Reject
  zero, signs, suffixes, fractions and overflow. Enforce feature/backend
  requirements independent of argument order; defer resume checks until saved
  policy is known, and reject explicit decode-only use.
- [x] ACB005 Integrate early safe lower-bound rejection and exact packed-layout
  admission, including text and stereo audio. Do not enforce a conservative
  text upper bound as actual allocation. Distinguish shape, budget and hardware
  errors; report the minimum sufficient whole-MiB ceiling.
- [x] ACB006 Reserve exact cache storage before BF16/packed weight admission.
  Verify live/free/future accounting counts it once, including capacity-test
  overrides, partial residency and streaming slots. Preserve reserves and
  workspace limits; audit large byte/index conversions in cache GPU copy,
  add/subtract, probe and reduction paths.
- [x] ACB007 Make allocation failure, cancellation and retry paths release
  partial buffers and preserve committed history. Keep retries bounded; never
  silently disable caching or change the requested workload.
- [x] ACB008 Revalidate admission when reusing prepared/live contexts, including
  a lower ceiling. Keep immutable conditioning/AdaLN keys independent of this
  resource limit; prevent a key hit from bypassing admission.
- [x] ACB009 Implement required sampler section 40 v2 with resolved budget;
  retain BF16 sections 41/42 and independent section-44 warmup versions. Read
  v1 under its historical 512 MiB policy, reject unsupported required versions,
  and preserve cache-off layout and numerical recipe versions.
- [x] ACB010 Validate serialized shape, counts, budgets and section lengths
  before adaptive payload allocation. Support early rejection of a too-small
  resume override. Keep the 16 GiB container limit, finite-value, integrity and
  committed-history checks; do not trust a stored large ceiling.
- [x] ACB011 Implement CLI/API resume semantics: omission restores saved ceiling;
  any sufficient replacement preserves numerical history; insufficient values
  fail before GPU loading and large adaptive host allocations. Save the effective
  replacement and retain other device/recipe/warmup checks.
- [x] ACB012 Preflight complete checkpoint size and account for host export and
  writer copies. Preserve atomic replacement and cleanup on disk/allocation
  failure. Keep clean AV decoding/version behavior unchanged for this option.
- [x] ACB013 Add initialization/resource diagnostics: budget source, effective
  ceiling, exact device/persistent bytes and rows. Retain step decisions and
  residency/free/live/future/reserve evidence in profiles and experiment output.

## Tests and qualification

- [x] ACB014 Add CPU planner/parser/policy tests for all sizing-table rows,
  exact boundary, just-under budget, large valid ceilings, zero/overflow,
  disabled-feature combinations, argument order and unchanged compatibility.
  Exercise large shape arithmetic without multi-GiB allocations.
- [x] ACB015 Add checkpoint/CLI/API/context tests for v1/v2, missing/optional or
  unsupported required sections, malformed counts/budgets/lengths, oversized
  files, sufficient/insufficient overrides, lower-ceiling reuse and atomic save
  failure. Keep corruption fixtures small and bounded.
- [x] ACB016 Build local Metal and PRO 5000 CUDA; run applicable ordinary,
  adaptive policy/CLI, sampler and weight-residency checks. Complete each
  required unchanged 204-output golden gate and retain exact result identities.
- [x] ACB017 Run bounded BF16 tests on an old supported shape: admitted
  512/4096/8192 MiB ceilings produce identical step latents/decisions. Exercise
  conservative, aggressive and adaptive+SubBlock, forced streaming, low-capacity
  fault injection and cancellation/retry. Preserve arithmetic and test limits;
  cover existing packed-projection policy/accounting in applicable tests.
- [x] ACB018 Qualify **1344×768/124 frames** above the former cap: exact budget
  admission, explicit 512 rejection, allocation accounting and cancellation
  cleanup. Use a bounded original 50-step schedule, at most six evaluations per
  invocation, and report actual hits rather than assuming them.
- [x] ACB019 Qualify **1344×768/362 frames** with default and sufficient explicit
  ceilings on PRO 5000. Compare a bounded continuous prefix with stop/resume;
  test sufficient/insufficient resume replacements and large checkpoint save/
  load. Record partial/streamed weights, cache bytes, peak VRAM/RSS, file size
  and retries. Keep the original 50-step schedule, at most six evaluations per
  invocation, and a 3,600-second deadline each.
- [x] ACB020 Generate one complete **1344×768/362-frame/six-step** BF16 video:
  conservative cache, omitted budget flag (4096 MiB), warmup 4, all 50 blocks,
  dense attention, full VAE, audio and auto weight residency. Reuse the previous
  campaign's prompt and seed 42. Within 3,600 seconds, confirm 362 frames,
  24 FPS, dimensions, stereo audio, finite outputs and full media decode.
  Preserve one successful artifact and all failed attempts.
- [x] ACB021 Update README and current adaptive/warmup documentation with the
  flag, defaults, legacy/resume behavior, sizing, diagnostics and hardware
  limits. Preserve historical accepted recipes, matrices and results. Publish
  a linked report with the target video, wall/stage times, memory/residency,
  actual cache work, checkpoint costs and all test outcomes.
- [x] ACB022 Reconcile tasks and document links. Confirm required large-target
  checks, final coherent-change golden gate, unchanged fixtures, and idle/reaped
  processes. State measured support and remaining limits; do not mark planning
  calculations or failed/OOM target runs as qualification.

## SubBlock reference-media support

This extension supports CUDA Ref2VA with **image, video and audio references,
including mixed reference sets**, all 50 blocks, BF16 projections,
`--adaptive-cache off`, `--reuse 1` and `--core-reuse 1`. Per the user's scope
correction, include embedded video audio and external video soundtracks.
Preserve the existing limits: 12 total references, 9 images, 3 videos, 3 audio
inputs, valid media geometry/duration/patch budgets, and standalone audio must
accompany an image or video. Frame anchors, continuation, adaptive caching with
references and reuse greater than 1 with SubBlock remain outside qualification. Ordinary reference generation with reuse 2/3 and
existing text-only SubBlock/quantization combinations must continue to work.
Keep the new reference+SubBlock+quantization combination rejected until separately
qualified; this work does not request an FP8/NVFP4 campaign.

The required full-video test uses **1344×768, 362 frames, six steps**, the
previous preview campaign's prompt, seed 42, and exactly one
`--ref-image inputs/1.jpg --ref-image-size max`. Use full VAE, audio, auto weight
residency and:

```text
--cuda-attention subblock --subblock-sparsity 0.75 --subblock-warmup 2
--adaptive-cache off --reuse 1 --core-reuse 1 --cuda-denoise-quant off
```

362 frames retains the preceding large-layout target. Warmup 2 is essential:
omitting it would apply the default ten-step dense warmup and exercise no sparse
attention in a six-step run. Steps 0–1 must be dense; steps 2–5 must execute
sparse attention in blocks 1–49 while block 0 remains dense. Expect **196 main
sparse calls and 104 main dense calls**, with additional dense protected-query
calls recorded separately. All reference queries remain dense and reference
keys remain available to generated queries. Do not count an all-dense fallback
as qualification of this feature.

Use PRO 5000 and local M4 under the same environment restrictions as above.
Store artifacts under ignored `outputs/subblock-reference/`. Preserve historical
blocked preview slots and their accepted scope. No additional SGLang reference
fixtures, live oracle or golden changes are needed; the unchanged full parity
regression is mandatory after coherent code changes and again at the end.

- [x] SRI001 Freeze the feature matrix, fixture identities, exact commands and
  validation workload. Audit shared approximation guards, reference kinds,
  Ref2VA selection, layout protection, prepared-context keys, resume and
  completed-state metadata. Keep this scope distinct from adaptive+reference
  support and from ordinary reference reuse 2/3, which already works.
- [x] SRI002 Split the blanket reference rejection into explicit feature/kind
  checks in the shared CLI/API/checkpoint paths. Admit supported image/video/audio
  BF16 SubBlock requests and mixed sets; keep adaptive+reference,
  anchors/continuation, unsupported precision and other excluded combinations
  rejected. Validate every reference kind, including after checkpoint load;
  preserve argument-order-independent errors and backend/build preflight.
- [x] SRI003 Audit and integrate Ref2VA protection end to end: image/video/audio reference
  rows, image/video-derived text/vision-conditioning rows, ordinary text and audio,
  first/last target frames, mixed 64-token blocks and incomplete tails. Verify
  dense protected-query execution and unconditional retention of protected
  keys without changing visibility, RoPE, reference preprocessing or routing
  arithmetic. Retain the independent 512 MiB attention-workspace limit.
- [x] SRI004 Preserve schedule semantics for references: configured absolute
  warmup, dense block 0, zero-sparsity/full-budget bypass and protected ranges.
  Ensure nonzero sparsity with warmup 2 actually reaches the sparse kernel on
  the required six-step layout. Add reference-aware dispatch/protection evidence
  using existing diagnostics where possible; count protected dense calls
  separately from the main dense/sparse calls.
- [x] SRI005 Validate request ownership, memory admission and context reuse for
  changing reference identity, count, order, size mode and canvas. Rebuild
  protection metadata when required; prevent stale layouts across requests.
  Preserve weight residency accounting, capacity failures, cancellation cleanup
  and unsupported-device behavior without relaxing memory limits.
- [x] SRI006 Support checkpoint stop/resume and completed AV decoding for the
  admitted reference+SubBlock path. Preserve ordered reference provenance,
  layout, protection reconstruction, sparsity, warmup and original schedule.
  Reject conflicting overrides and corrupted/mismatched layout metadata.
  Audit older-reader rejection of newly admitted layouts; introduce format or
  recipe changes only if new serialized semantics actually require them.
- [x] SRI007 Add CPU/API/CLI policy and layout tests for one/multiple images, videos, embedded/external audio and mixed sets,
  valid reference-count boundaries, match/high/max sizing, mixed blocks/tails,
  all protected key coverage and precise unsupported combinations. Verify
  ordinary reference reuse 2/3 and text-only SubBlock still pass. Use legal
  image/patch layouts; do not disable existing preprocessing limits for tests.
- [x] SRI008 Add bounded native attention tests with mixed reference layouts:
  verify dense warmup, dense block 0/protected queries, mandatory reference-key
  retention, sparse dispatch and zero-sparsity equivalence to the shared dense
  path. Reuse existing operator tests; do not add an independent SGLang oracle
  or enlarge the recorded parity suite. Include context-switch/capacity cases.
- [x] SRI009 Run bounded Ref2VA generation at a smaller supported canvas with
  one and multiple images, including match/high/max cases where legal. Add
  video with embedded audio, silent video, image plus standalone audio,
  external video soundtrack and mixed image/video/audio cases. Compare
  continuous versus stop/resume across the SubBlock warmup boundary for image
  and mixed media layouts, and test cancellation/retry. Require finite outputs, reference-aware counters and
  unchanged original schedule, at most six evaluations per invocation.
- [x] SRI010 Build Metal and CUDA with SubBlock enabled on CUDA; run applicable
  ordinary, policy/CLI, native attention, reference-layout and sampler tests.
  Confirm the default ten-step warmup still yields dense execution for a short
  schedule, while explicit warmup 2 enables sparse work. Retain all required
  intermediate full 204-output regression results after coherent changes.
- [x] SRI011 Run the required **1344×768/362-frame/six-step, one-reference/max**
  video on PRO 5000 with the exact contract above and a 3,600-second deadline.
  Verify the frozen 1365×1821 input becomes 2048×2720 with 21,760 raw patches,
  actual Ref2VA execution, 196 sparse/104 main dense calls, protected-reference
  handling and recorded residency/VRAM/RSS. Fully decode the output: requested
  canvas, 362 frames, 24 FPS, approximately 15.0833 seconds and stereo audio.
  Record total/stage wall time; review identity, motion, artifacts and audio
  with review limits explicit. Publish the video and qualification report,
  update README/current feature docs, and preserve historical reports. Failed,
  timed-out, all-dense or reduced-geometry runs cannot satisfy this task.
- [x] SRI012 **Rerun the complete unchanged SGLang/CUDA parity regression at
  the end**, after all adaptive-budget and SubBlock source/build/test-tool
  changes are final, using `make test-cuda-reference-regression` with the
  qualified environment and a fresh output directory. Require **204/204 exact
  recorded golden matches**, unchanged 12 input fixtures and the 720-second
  test deadline; run the CPU gate checks too. Fix any failure and rerun the
  complete gate. Link final evidence, reconcile all 34 task statuses, verify
  document/media links and process cleanup, and report actual supported scope.
