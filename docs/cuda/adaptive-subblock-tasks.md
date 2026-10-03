# Native adaptive cache and SubBlock — implementation tasks (archived)

Archived 2026-09-27. Current work is tracked in [todo.md](../todo.md).

Status: **complete; 38/38 tasks complete**. The design is
[Native adaptive caching and SubBlock attention](design-adaptive-cache-subblock.md).
The previous completed campaign is preserved in the
[single-pipeline archive](single-pipeline-tasks.md); its historical task IDs
and render limits do not govern this new comparison.

Goal: implement an opt-in native adaptive residual cache and native BF16
SubBlock attention, qualify them independently and together, and compare them
with the current dense and fixed-reuse paths. Default rendering retains the
unchanged recorded SGLang arithmetic.

## Fixed execution contract

- The user selected **12 variants**, each with **one complete 640×480,
  90-frame, 50-step video**, at 24 FPS. Use the same piano prompt, seed 42,
  original model, full AV decoder and codecs for every variant. The exact
  [matrix and workload](design-adaptive-cache-subblock.md#fixed-comparison-protocol)
  are authoritative. Reuse the single dense baseline video in every comparison.
- Render all variants on the same final source/build. No extra successful
  repeats, seed/prompt sweep, timing-only videos or quantization combinations.
  Kernel warmups and bounded operator/latent tests need no additional videos.
  Retain failed attempts and their time; never cherry-pick successful outputs.
- Use the configured RTX PRO 5000 / SM120 CUDA host and existing model.
  Keep connection details outside tracked files. GPU work is serial. Reuse
  CUDA 13.0.3, cuDNN 9.20 and the pinned math/media environment; no model hashing.
- Preserve [CONTRIBUTING.md](../../CONTRIBUTING.md): after every coherent code,
  build or test-tool change, run the complete **204-output recorded regression**
  before starting unrelated implementation work. No changed goldens, tolerances,
  fixtures, cases or evaluation counts. Documentation-only work needs link and
  document validation. Record source/build/manifest identities for each pass.
- The original regression still includes its 124-frame/six-step render and
  retains its deadline. It is separate from the twelve comparison videos.
  Permit 50 steps only in the dedicated fixed-shape comparison subprocesses;
  preserve the general Makefile/test six-step limit.
- Approximate output is not a parity pass. Require finite valid AV output,
  correct dispatch, bounded memory, request isolation and exact resume of the
  selected recipe. Report slower variants and visual/audio changes honestly.
  Single observations provide neither medians nor statistical confidence.
- Close tasks only with evidence. An incomplete or failed variant remains open.
  Publish inspectable local results without waiting for human review. Record
  review only when confirmed by the user. Both features remain off by default.

## M0 — Freeze the experiment and native contracts

- [x] ACSB001 Record the starting source/build, unchanged golden-manifest digest,
  GPU UUID/capability, driver, toolkit, cuDNN/cuBLAS/codecs and model metadata.
  Verify enough memory/storage and establish the existing complete regression pass.
- [x] ACSB002 Create the machine-readable twelve-variant manifest from the design:
  D0, R2, R3, C4, C6, A1, A3, S75, S80, A1S75, A3S75, A3S80. Freeze prompt/seed,
  exact 90-frame/50-transition contract, full decoder, order and output paths.
- [x] ACSB003 Audit and pin upstream SubBlock router/kernel sources, revision,
  licenses and SM120 ABI. Write the exact scoring/scaling/pooling/tail contract,
  retained-budget rounding and intentional native protection differences.
- [x] ACSB004 Freeze native adaptive recipe 1: dense first block, anchored BF16
  probe/suffix delta, FP32 score reduction, preset thresholds, warmup/streak/final
  refresh, phase invalidation and transaction boundaries. No Python runtime hooks.
- [x] ACSB005 Define bounded independent score/router/masked-attention fixtures,
  numeric tolerances and storage limits before kernel implementation. Fix telemetry,
  cache preparation, timeout and measurement boundaries without generating videos.

Exit: reproducible scope, fixed comparison manifest and reviewable math contracts.

M0 evidence: [frozen contract](adaptive-subblock-contract.md),
[twelve-case manifest](adaptive-subblock-manifest.json), and local
`outputs/adaptive-subblock/work/baseline-result.json` (204/204 passing),
`baseline-environment.json` and `starting-source.json`. No weights were hashed.

## M1 — Add explicit policy, interfaces and capability checks

- [x] ACSB006 Add zero/off-compatible adaptive parameters and presets to the
  C API/CLI. Validate values and reject combinations with old reuse/core-reuse,
  layer thinning, token reduction, LoRA/Turbo and unsupported backends.
- [x] ACSB007 Append the SubBlock attention ID, parser/help and sparsity control;
  add independent CUDA_SUBBLOCK build support. Reject missing support/unsupported
  devices before model loading. Preserve existing attention IDs and options.
- [x] ACSB008 Capture immutable policies in request/GPU contexts; carry original
  schedule and block indices plus packed layout/protection metadata. Keep Qwen,
  refiners, encoders, VAEs and sensitive heads on their existing implementations.
- [x] ACSB009 Add checked memory planning for the adaptive 512 MiB allowance and
  SubBlock within the existing 512 MiB attention budget. Reserve both before
  weight admission; validate overflow, exhaustion and feature-off zero allocation.
- [x] ACSB010 Test API/CLI ordering, conflicting flags, build/device errors,
  policy identity and dense→approximate→dense isolation. Build/test Metal for
  shared interface changes and pass the complete unchanged CUDA regression.

Exit: selectable, bounded policies without any default arithmetic change.

M1–M4 evidence: [native qualification record](adaptive-subblock-results.md).
This includes Metal, both CUDA build configurations, operator/oracle checks,
checkpoint and cancellation tests, and the final unchanged 204/204 regression.

## M2 — Implement adaptive residual caching

- [x] ACSB011 Split main-DiT execution at block 0 while preserving normalization,
  residual rounding and cross-block fusion boundaries. Prove cache off and forced
  misses reproduce the existing dense output exactly on bounded real fixtures.
- [x] ACSB012 Implement native GPU probe subtraction and deterministic FP32
  normalized-change reduction over valid packed rows. Test zeros, epsilon,
  threshold ties, tails, separate audio/video diagnostics and nonfinite rejection.
- [x] ACSB013 Implement anchored suffix storage and hit reconstruction with fresh
  projections/heads. Add empty/warmup/streak/final refresh rules and commit history
  only after successful execution; retain one joint decision for packed AV tokens.
- [x] ACSB014 Integrate hit/miss behavior with weight streaming and prefetch.
  Demonstrate skipped suffix blocks perform no uploads/GEMMs/attention/MLPs;
  preserve buffer lifetime, asynchronous ownership and cancellation safety.
- [x] ACSB015 Exercise hit/miss boundaries, phase changes, context resets, memory
  admission, partial failures and repeated create/release. Validate packed
  reference and prefix/bridge semantics with bounded fixtures before enabling them.
- [x] ACSB016 Add low-overhead per-step scores, refresh/hit reasons, executed-block
  counts, memory and transfer counters. Diagnostic tensor capture remains separate
  from timed generation; confirm counters reflect real work.
- [x] ACSB017 Run host/operator and relevant existing CUDA tests, Metal build/tests,
  and the complete unchanged reference regression on this coherent implementation.
  Record exact source/build and leave unsupported request combinations explicit.

Exit: functional native adaptive cache with exact disabled behavior and bounded state.

## M3 — Implement native SubBlock routing and attention

- [x] ACSB018 Implement native 64/16-token Q/K summaries and the pinned score
  calculation. Match independent routing fixtures for valid-row pooling, score
  scale, log-sum-exp stability, deterministic ties and ragged tails.
- [x] ACSB019 Implement bounded top-block selection, eight-block budget rounding,
  sorted/deduplicated indices and protected query/key handling. Preserve per-row
  attention visibility; test mixed masks, empty/invalid routes and actual density.
- [x] ACSB020 Implement the compiled native SM120 selected-tile BF16 attention
  kernel with FP32 softmax/accumulation and both output layouts. Audit imported
  licenses/dependencies; inference must not require Python, Torch or a worker.
- [x] ACSB021 Integrate head/query tiling, stream ownership and the fixed workspace.
  Compare selected-mask and all-selected kernel output with the independent FP32
  oracle; cover extreme/zero inputs, aliases, bounds and sanitizer checks.
- [x] ACSB022 Route dense warmup, first block, short sequences, protected queries
  and documented unsupported shapes through shared dense attention. Verify exact
  full-budget bypass and explicit fallback counters; runtime faults remain errors.
- [x] ACSB023 Replay bounded real H3 QKV through dense and SubBlock with routing
  overhead included. Confirm sparse work actually executes within memory limits;
  report slower cases without replacing the kernel with an unreported fallback.
- [x] ACSB024 Build with SubBlock enabled/disabled, verify existing attention
  choices and Metal compatibility, run relevant tests and the unchanged full
  regression. Pin native recipe/plan identity before combining features.

Exit: native sparse attention with correct masks, bounded execution and measured dispatch.

## M4 — Compose features and preserve state

- [x] ACSB025 Keep block 0 dense for the adaptive signal; schedule sparse suffix
  execution by absolute denoising index. Force a cache refresh on the dense→sparse
  transition and ensure hits neither advance the schedule twice nor invoke suffix kernels.
- [x] ACSB026 Add required versioned checkpoint sections and execution/provenance
  identities for adaptive tensors/counters and SubBlock policy/plan. Preserve
  existing default states and reject unknown, corrupt or incompatible active recipes.
- [x] ACSB027 Verify latent-only stop/resume across warmup, hit, forced refresh,
  index 10 and the final transition; require identical decisions and AV latents.
  Confirm clean AV decoding works without the optional SubBlock build.
- [x] ACSB028 Exercise combined cancellation/recovery, request isolation, fixed
  memory admission and packed reference/continuation masks with bounded fixtures.
  Leave any unverified consumer explicitly rejected rather than claiming support.
- [x] ACSB029 Run ordinary CUDA tests, Metal build/tests and the final unchanged
  204-output gate; verify optional and default dispatch. Freeze this implementation
  before the twelve-video campaign and record any remaining unqualified scope.

Exit: combined execution and resume are deterministic, with the dense default preserved.

## M5 — Execute the single-video comparison

- [x] ACSB030 Implement the dedicated manifest runner with a scoped 50-step
  environment, immutable source/config checks, serial GPU execution, fixed per-run
  timeout and complete status records. Never alter the ordinary six-step test limit.
- [x] ACSB031 Validate runner accounting with synthetic/dry-run records: exactly
  twelve planned outputs, no repeated successful variant, 50 scheduler transitions,
  exact 90-frame media, new artifacts, failure retention and no silent variant omission.
- [x] ACSB032 Run D0 once, then R2/R3/C4/C6 once each at 640×480/90/50 with the
  fixed prompt/seed, preparation and full decoder. Save complete media, AV states,
  stage timing, memory and actual evaluation/block counts.
- [x] ACSB033 Run A1/A3/S75/S80 once each using the same frozen build and workload.
  Verify cache hit/miss and sparse/dense telemetry; record a zero-hit or dense-only
  outcome honestly without retuning or generating another successful comparison.
- [x] ACSB034 Run A1S75/A3S75/A3S80 once each. Confirm phase invalidation, dense
  probe behavior and joint cache decisions. Validate every final video and audio
  stream; retain the complete twelve-case coverage/failure ledger.

ACSB030–031 evidence: the dedicated runner and its nine passing synthetic
accounting tests are frozen in the final qualified source. All twelve renders and their media validation passed, with one attempt per
variant. Quality measurements and local report publication are complete.

Exit: twelve comparable final videos on one build, with single-observation evidence.

## M6 — Publish evidence and close the experiment

- [x] ACSB035 Compute frame-aligned visual, temporal and audio differences versus
  the shared D0 baseline, including worst frames and AV timing. Retain failed
  historical similarity thresholds as differences, never default-parity passes.
- [x] ACSB036 Produce local synchronized HTML playback, JSON and CSV using the
  twelve existing videos. Include wall/stage ratios, memory, dispatch/fallbacks,
  cache traces/density, commands, identities, limitations and human-review status.
- [x] ACSB037 Copy and validate all local report assets, hashes, video/audio
  readability, links and exact case count. Update README/design with actual native
  support and measured outcomes; label untested hardware/tasks and slower variants.
- [x] ACSB038 Audit all task evidence and final source identity against the last
  passing complete regression; rerun the full gate after any subsequent code or
  tooling change. Close only when all twelve final videos and correctness checks
  exist, publish the resulting recommendation, and keep both defaults off.

Exit: inspectable results for the agreed workload, preserved default parity,
working native implementations and no hidden missing/failed variants.

M6 evidence: [qualification, measured outcomes and recommendation](adaptive-subblock-results.md),
[local synchronized comparison](../../outputs/adaptive-subblock/2026-09-26-sm120/index.html),
[JSON](../../outputs/adaptive-subblock/2026-09-26-sm120/report.json) and
[CSV](../../outputs/adaptive-subblock/2026-09-26-sm120/report.csv). The final local
audit verifies all artifact hashes and all twelve media/AV/dispatch records.
The report-only link-checker correction was followed by another unchanged
204/204 regression; the report records both source/build identities and verifies
that production source did not change. No comparison videos were regenerated.
Every approximate variant misses the complete historical similarity checks;
the user accepted the results and passed visual qualification for all twelve
videos. The [review record](../../outputs/adaptive-subblock/2026-09-26-sm120/human-review.json)
binds that approval to the existing video hashes. Both defaults remain off.
