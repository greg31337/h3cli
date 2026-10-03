> Archived completed campaign. Its execution limits apply to the historical work below.
> The active [task list](../todo.md) covers adaptive caching and SubBlock attention.

# Single CUDA pipeline — implementation tasks

Status: **M0/M1, M3A, M3B and M4 complete; M6/M7 complete; M2/M3 implementation evidence retained (62/62 active checklist tasks complete).**
T021/T027 are superseded, not passed. M5 is removed; T034–T040 are retired,
with necessary compatibility/consumer work reassigned to M3B/M4/M6. Existing
IDs remain stable; M3B adds T062–T071.
See [M6/M7 removal audit and final qualification](single-pipeline-final.md) and
[final playback / speed report](../../outputs/cuda-final/review.html).
See [M3B/M4 results and regression proof](single-pipeline-results.md),
[current single-pipeline design](design-single-pipeline.md), and
[paired playback and speed report](../../outputs/cuda-single/review.html).
See [M0/M1 evidence](two-modes-m0-m1.md),
[M2/M3 evidence](two-modes-m2-m3.md),
[M3A evidence](two-modes-m3a-results.md), and
[M3A playback](../../outputs/two-cuda-modes/m3a/review.html). The
[earlier playback report](../../outputs/two-cuda-modes/m2-m3/review.html) is retained.
Portable regression setup is in [CONTRIBUTING.md](../../CONTRIBUTING.md). Background design: [SGLang baseline and original two-mode plan](design-two-modes.md).
The single-pipeline decisions in this checklist supersede its two-mode requirements;
M3B includes updating the design and public documentation.
The previous completed 40-task campaign is preserved in
[its archived checklist](cuda-sglang-parity-tasks.md), with
[results](cuda-sglang-results.md) and [audit](cuda-sglang-audit.md).

Goal: one CUDA implementation based on the current SGLang-parity pipeline.
Remove `--fast-cuda` and its separate runtime policy completely. Default dense
attention with BF16 projection weights retains exact reference arithmetic 4.
Explicit Sage2++, Sage3 or SOL selections use the same preparation, sampler and
DiT pipeline; M4 adds independent FP8/NVFP4 projection options. Approximate
selections may degrade visual quality and must report that difference alongside
the matched default output. Preserve Metal and independent preview/full-VAE
selection. Remove legacy execution after establishing consumer reachability.

M0–M3A below retain historical implementation and test evidence. Their references
to fast/two-mode behavior describe completed work, not the forward architecture.
Old T021/T027 failures remain in the reports; the new scope supersedes their
promotion requirements without relabeling historical failures as passes.

## Mandatory execution rules

- **Preserve the recorded reference fixtures, golden hashes, case coverage and
  evaluation counts. Run the whole suite after every code change before starting
  the next unrelated implementation patch.** Acceptance depends on recorded
  golden output; source/build digests identify each run. A failed gate
  requires a code fix and another full run, never a rebaseline or waiver.
- Proposed command: `make test-cuda-reference-regression`. A pass must identify
  the exact source/build and recorded golden-manifest digest. Missing fixtures,
  skipped cases, stale results, timeout or unavailable GPU mean incomplete/failed,
  not pass. Keep mutable CLI/state/optional-kernel tests separate. Documentation-only
  edits require document validation rather than GPU rendering.
- Calibrate the per-change gate to **5–10 minutes, hard limit 12 minutes after
  build**, on the qualification RTX PRO 5000. Retain a complete native C0 render:
  640×480, 124 frames, six evaluations, full AV decode. Cover large projection
  shapes and late 50-evaluation states with bounded operator fixtures.
- Normal visual tests use at most six evaluations. Any 1344×768 full render uses
  at most two. Use 243 frames for milestone performance tests. The final retained
  C2 reference replay (124 frames / 50 evaluations) is the sole 50-evaluation
  full-render exception; it is not part of the per-change gate.
- Use the configured CUDA host, model `/path/to/models/MiniMax-H3`; re-probe device and
  installed dependencies before testing. No weight hashing, unplanned dependency
  upgrades, enlarged scratch caps or legacy execution fallbacks.
- Every milestone exits with a passing unchanged reference gate on its final
  source. Retain failed attempts, actual dispatch and incomplete coverage in the
  reports. Default dense BF16 reference parity permits no new drift.
- Explicit Sage/SOL (and M4 quantization) may differ numerically and visually.
  Require correct dispatch, finite/stable execution, valid complete AV output,
  bounded memory and honest side-by-side evidence; the old fast perceptual
  thresholds are diagnostic comparisons, not promotion blockers for explicit
  approximate options. Keep their frozen files and historical failures intact.
  This allowance never changes default reference goldens or tolerances and does
  not waive crashes, corrupt media, ignored flags or protected-range violations.
  Human review is optional; publish videos without waiting for it or claiming it
  occurred. Do not infer a speedup or visual equivalence from successful execution.

## M0 — Establish and freeze short reference regressions

Complete before changing production arithmetic or mode semantics. R0–R4 and
budgets are defined in the design; measure feasibility before freezing them.

- [x] T001 Inventory CLI/API/environment modes, fallback chains, generators,
  build variants, caches, state formats and consumers. Distinguish legacy-only
  orchestration from shared primitives; pin starting revision `84d1555`, working
  tree, hardware and dependency identities.
- [x] T002 Preserve the accepted arithmetic-4 reference executable/source,
  dependency manifests, C0/C1/C2 oracle evidence and unchanged SGLang contracts.
  Establish fixture provenance without hashing model weights; record historical
  waivers separately from new acceptance criteria.
- [x] T003 Build stable R0/R1 host/operator fixtures for 6/50-evaluation noise,
  schedules, packing, BF16 norm/RoPE/linears, FP32 patch/head projections and dense
  attention. Include C2 steps 25/49 and all six large patch-batching cases. Verify
  the gate rejects corrupted results, missing fixtures and missing cases.
- [x] T004 Build R2–R4 native image `match`/`max`, video/audio encoder and delivery
  probes, full VAE tile/audio checks and isolation/lifetime checks. Retain complete
  C0 production conditioning, every sampler update and decoded pixels/PCM; no
  imported-oracle-input or substitute CPU-sampler bypass of the end-to-end gate.
- [x] T005 Implement `make test-cuda-reference-regression` with isolated builds,
  source-change detection, exact-source records, required case counts, bounded
  artifacts and hard timeouts. Measure 5–10-minute normal / 12-minute maximum
  execution; optimize fixture/report overhead before freezing without hiding
  missing coverage.
- [x] T006 Record fixture/output golden hashes and reject missing, changed or
  unexpected artifacts. Document execution after every code change and retain
  current source/build identities. Test configuration is portable across hosts.
- [x] T007 Freeze a separate fast quality/performance contract before optimization:
  prompts, seeds, reference-conditioned cases, held-outs, perceptual/temporal and
  AV metrics, timing boundaries, promotion rules and paired HTML evidence.
  Human notes are optional; do not claim visual review occurred when it did not.

Exit: short gate timed, proven sensitive to regressions, frozen and passing;
fast quality criteria recorded independently of reference tolerances.

## M1 — Resolve two policies over shared SGLang infrastructure

Depends on M0. Keep legacy code temporarily only to migrate its consumers.

- [x] T008 Introduce immutable resolved request policy: mode, base/fast recipe,
  attention, projection precision and presentation. Capture it in components and
  restore nested/thread-local scopes on success, error and cancellation.
- [x] T009 Separate shared SGLang preparation/sampler semantics from the exact
  reference predicate. Audit every old boolean/early return; do not implement
  new fast merely by enabling both old `fast` and `sglang_reference` flags.
- [x] T010 Keep reference as default and explicit `sglang` alias. Require explicit
  fast mode for Sage/SOL, quantization and other non-parity compute options.
  Validate conflicts independently of argument order before allocations; log
  resolved recipes and effective kernels.
- [x] T011 Make preview/full-VAE selection orthogonal to CUDA denoising policy.
  Preserve reference latents and label nonstandard decoding outside full-output
  parity. Keep preview behavior unchanged and inventory still/image-VAE consumers
  and their current restrictions.
- [x] T012 Route eligible fast overrides explicitly, then fall back to shared
  reference primitives only. Isolate handles, library/math/workspace policies
  and plans; fail unsupported explicit build/device requests before model loading.
- [x] T013 Add separate mutable CLI/API/build tests for modes, option conflicts,
  unsupported devices, missing libraries and scope restoration, including Metal
  independence. Keep the frozen reference probe interface stable.
- [x] T014 Pass the unchanged gate and audit reference dispatch/arithmetic before
  proceeding. Ensure no new fallback introduces a hidden third CUDA mode.

Exit: reference preserved; policy and feature ownership explicit.

## M2 — Rebuild and optimize fast mode from the reference pipeline

Depends on M1. No approximate optimization before exact construction proof.

- [x] T015 Build fast recipe 2 from shared reference preparation, noise, schedules,
  Euler updates, all DiT blocks, FP32-sensitive heads and full AV decoding.
  Do not retain old fast orchestration as the new baseline.
- [x] T016 Prove fast with all substitutions disabled matches reference C0 exactly
  through conditioning, every update and full decoded media. Keep this an internal
  construction test, not a third public mode.
- [x] T017 Profile complete requests on the actual server: model/cache loading,
  conditioning, DiT, transfers, decoding and delivery. Select optimizations from
  measured costs rather than automatically copying old fast tuning.
- [x] T018 Port useful plan/cache reuse, bounded allocations, conversions and
  synchronization improvements individually. Preserve memory admission reserves,
  stream ownership and release fences; keep reference behavior unchanged.
- [x] T019 Add selected GEMM/norm/RoPE/elementwise optimizations with fast-only
  math/workspace policies. Initially preserve exact conditioning/Qwen and
  FP32-sensitive heads. Run fast quality checks plus the unchanged reference gate
  for each coherent change.
- [x] T020 Optimize fast decoding from the SGLang range-safe FP16/FP32 baseline
  using identical clean latents. Preserve AV timing and sampler independence;
  retain preview unchanged and reject older TF32 choices without measured benefit.
- **T021 — superseded by M3B, not passed.** Separate fast dense BF16 default
  qualification is no longer a deliverable because that mode will be removed.
  Retain its historical 15.14% wall reduction, unchanged VRAM and failed
  conditioning ratio 1.06653 against the original 1.05 limit. Do not alter the
  statistic, threshold or retained timing audit.

Historical exit: fast dense qualification was incomplete. Forward exit is the
single-pipeline consolidation and default parity proof in M3B.

## M3 — Integrate Sage and SOL into new fast dispatch

Depends on M2. Reuse audited kernels, not their old renderer dependencies.

- [x] T022 Define explicit main-DiT attention interfaces for 56 heads × 128,
  reference QKV normalization/RoPE/scaling, layouts, tails and output projection.
  Prevent early reference dispatch from bypassing selection; exclude Qwen,
  refiners, encoders and VAEs from these substitutions.
- [x] T023 Integrate SageAttention 2++ with pinned dependencies/notices, capability
  checks, bounded workspace and actual-kernel counters. Validate real QKV,
  padding and documented shape fallbacks to shared reference dense attention.
- [x] T024 Integrate SageAttention 3 with audited scales/corrections, bounded
  workspace and explicit unsupported-device/build errors. Keep attention precision
  independent of NVFP4 projection-weight selection.
- [x] T025 Integrate SOL with new QKV/output layouts, protected conditioning and
  continuation ranges, local/early dense regions, absolute step/block indices
  and both noise sigmas. Start from existing CUDA defaults including
  `min_exact=0.75`; route dense work through shared reference primitives.
- [x] T026 Test forced-dense SOL equivalence, tails, ordered references,
  continuation, allocation admission, cancellation and fallback reasons.
  Prove reference never invokes Sage/SOL; do not add a Sage/SOL hybrid.
- **T027 — superseded by M3B, not passed.** Sage2++, Sage3 and SOL failed the
  original C0/R1 quality gates. Preserve those measurements and clips. M3B
  permits quality degradation for explicit attention selections and replaces
  this promotion requirement with stable execution, measured performance and
  side-by-side visual reporting; it does not change the frozen default gate.

Historical result: T022–T026 implemented and tested; approximate quality
qualification failed. See [measurements and clips](two-modes-m2-m3.md).
The forward attention integration/acceptance contract is M3B.

## M3A — Port qualified exact improvements to reference

Depends on M1 and the retained M2/M3 implementation evidence, not acceptance of
approximate attention. T021/T027 were open at M3A completion and are now
superseded by M3B. See the
[candidate assessment and qualification design](two-modes-m3a.md).
Task IDs T053–T061 extend the existing sequence without renumbering prior tasks.
Completed: mapped DiT staging and one-lane shared CUDA Qwen prefetch qualified;
reference VAE cast fusion deferred under T057. Final unchanged gate: 204/204.
See [implementation, retained failures and final measurements](two-modes-m3a-results.md).

- [x] T053 Pin the current reference source/build, arithmetic-4 identity,
  dependency versions and recorded golden outputs. Predeclare same-mode
  candidate A/B toggles, cache conditions, interleaved run order, timing/memory
  boundaries and promotion statistics. Preserve historical oracle evidence.
- [x] T054 Separate exact shared substitutions from `fast_v2` and old fast math
  flags. Capture immutable component policy and expose actual mapped/copied and
  fused/separate dispatch counts for reference and fast. Keep default/explicit
  `sglang` semantics, state identities, dense arithmetic, preview VAE and Metal
  unchanged; diagnostic A/B switches must not introduce a third public mode.
- [x] T055 Port private file-mapped pinned weight staging into eligible reference
  loading. Preserve metadata keys, no weight hashing/writes, range validation,
  memory admission, the 40 GiB process budget, streamed slots and release fences.
  Audit page-aligned pinned accounting and model immutability over mapping
  lifetime. Keep copied pinned/bounded bounce fallbacks for unsupported mapping,
  registration or admission, without hiding genuine read errors.
- [x] T056 Add separate mutable staging tests for unaligned/tail ranges, cache
  hits/misses/invalidation, synthetic file replacement/mutation, registration
  failure, low-memory admission, cancellation, partial cleanup and repeated
  create/release. Check file bytes, host budget/RSS plateau, stream ownership and
  mixed reference/fast contexts; prove both mapped and fallback paths execute.
- [x] T057 Evaluate full-VAE Q/K/V cast fusion independently. Compare cast bits,
  FP16 rounding boundaries, signed zeros, tails, nonfinite/overflow faults and
  graph replay, then repeated identical-latent full decoding. Port to reference
  only with measured benefit under the existing promotion contract; otherwise
  record deferral and retain separate casts. Preserve scratch/precision bounds.
- [x] T058 Qualify candidates using at least three interleaved reference-mode
  A/B pairs at 640×480 / 243 frames / six evaluations on the qualification RTX PRO 5000.
  Attribute staging and fusion separately; report complete wall, loading,
  conditioning, denoising, AV decode, VRAM, host RSS and pinned bytes. Resolve
  the existing conditioning regression above 5% before promotion; preserve the
  declared statistic and telemetry limits, retaining failed/invalid attempts.
- [x] T059 Run bounded supplemental reference tests for default/explicit
  `sglang`, mixed modes, image `match`/`max`, ordered image/video/audio references,
  continuation and clean-latent delivery. Use retained fixtures and complete
  124-frame / six-evaluation clips; compare preparation, sampler states and
  decoded RGB/PCM exactly. Publish paired playback HTML with validated links.
- [x] T060 Enable only evidenced exact improvements in reference by default;
  document accepted/rejected candidates, fallbacks and measured limits. Preserve
  arithmetic identity 4 and fast recipe 2. Use the improved reference for future
  fast timing comparisons without changing historical parity goldens. Keep
  T021/T027 status independent and run the unchanged gate after every code change.
  (Historical completion: M3B now supersedes those two qualification tasks.)
- [x] T061 Finish with `make test-cuda-reference-regression` on the final source:
  all 204 existing artifacts must pass the complete reference suite. Keep
  fixtures, goldens, cases and evaluation counts unchanged; no rebaseline or
  waiver. Supplement with dispatch evidence
  proving the promoted paths execute in reference, not only their fallbacks.
  Record exact source/build/suite identities and regression results confirming
  the current SGLang parity is preserved; any later code edit requires a rerun.

Exit: measured exact improvements shared with reference, unchanged SGLang parity
confirmed on final source, inspectable output and bounded host/device memory.
The mandatory whole-suite gate applies after each code change throughout M3A.

## M3B — Remove fast mode and consolidate explicit attention options

Depends on completed M3A and the implemented M3 kernels. This replaces separate
fast-mode qualification; it does not wait for T021/T027. Completed with all
24 matrix clips, eight supplemental clips, 66 timing runs and a final unchanged
204-artifact default gate. Approximate quality differences remain reported.

- [x] T062 Pin the M3A source/build, arithmetic-4 default and unchanged 204-artifact
  golden-output contract. Inventory every `--fast-cuda` CLI/API/environment selector,
  policy field, math override, cache/state identity and consumer. Predeclare
  matched dense/Sage2++/Sage3/SOL cases, measurement boundaries and reporting;
  preserve original failed quality/timing records and historical executables.
- [x] T063 Remove `--fast-cuda` parsing/help, API mode selection, environment
  activation and separate fast recipe dispatch. Keep one captured request policy
  for attention, projection precision and independent decoder selection. Default
  and explicit `sglang` select the same pipeline; `sglang` is an alias, not another
  renderer. Removed options produce actionable errors, never a silent no-op or
  compatibility renderer. Keep only declarations/adapters required by the frozen
  probe ABI, if any, with obsolete nondefault requests rejected.
- [x] T064 Route all CUDA preparation, Qwen, noise/schedules, Euler updates, DiT
  and delivery through the current reference implementation. Retain M3A mapped
  DiT staging, one-lane Qwen prefetch, admission/fallbacks and arithmetic 4 for
  the default. Remove fast-only dense math and decoder dispatch; keep separate
  reference VAE casts and preview behavior. Do not introduce approximate work
  into default conditioning, sensitive heads, norms or decoding.
- [x] T065 Select Sage2++, Sage3 or SOL directly through attention options in the
  single main-DiT path, without a fast-mode prerequisite. Preserve capability
  errors before model loading, pinned kernels/notices, real-dispatch counters,
  SOL step/layer/local/protected-range semantics and documented dense shape
  fallbacks. Keep attention independent of projection precision, and reject
  quantized requests until M4 supports them. Dense default invokes neither
  approximate kernel; forced-dense SOL remains numerically checked. No Sage/SOL
  hybrid and no optional attention substitution in Qwen, refiners, encoders or VAEs.
- [x] T066 Replace mode-based cache/plan/state identity with base arithmetic plus
  explicit attention/SOL/precision recipe and existing model/fold/layout/device/
  library identities. Preserve supported reference conditioning/resume semantics
  and same-build checks. Reject incompatible legacy/fast-v1/fast-v2 sampler
  states clearly; never relabel them as arithmetic 4. Remove legacy resume
  switches. Preserve safe clean AV/image decode and continuation readers where
  compatible, with documented old-executable re-export otherwise; no weight hashing.
- [x] T067 Audit and move all consumers to the single pipeline: first/last frame,
  ordered references, image `match`/`max`, video/audio, LoRA/Turbo, conditioning
  reuse, continuation, reuse/reduction and CUDA still/image-VAE. Preserve existing
  restrictions and Metal behavior; document explicit unsupported requests. Produce
  a consumer reachability map before deleting any still-used legacy primitive.
- [x] T068 Update separate mutable CLI/API/state/build tests for removed flags,
  option ordering/conflicts, missing libraries/devices and recipe isolation.
  Exercise both dense-to-approximate and approximate-to-dense context orders,
  cancellation/recovery, cache invalidation, streaming/residency, memory plateaus
  and preview/full decoding. Add bounded consumer smoke tests. Preserve every
  frozen test/fixture/interface; run the complete unchanged gate after each code
  patch, not only at milestone close.
- [x] T069 Measure dense/Sage2++/Sage3/SOL with BF16 projections on the local RTX
  PRO 5000 using identical prompts/seeds/model/decoder and serialized GPU work.
  Produce complete C0 and ordered-reference R1 clips at 640×480 / 124 frames /
  six evaluations, plus bounded match/max and continuation checks. Use three
  interleaved matched default/candidate pairs per attention option at 640×480 /
  243 frames / six evaluations for performance claims; predeclare placement,
  cache conditions and statistics. Report wall/loading/conditioning/denoise/decode,
  peak VRAM/RAM and actual kernel/fallback counts. Label slower/unsupported
  combinations honestly; speed or perceptual-threshold failures cannot be hidden.
- [x] T070 Publish side-by-side HTML videos for each approximate attention choice
  against its matched default, including worst frames, temporal/audio metrics,
  commands, identities, timing, memory and known artifacts. Quality degradation
  is permitted for these explicit options; report failed historical thresholds
  without treating them as default-parity failures or asserting visual equivalence.
  Verify local playback links/checksums/case counts. Update the design, README,
  CLI examples and mutable tools to describe one CUDA pipeline with explicit
  attention options and no fast-mode requirement; record human-review status.
- [x] T071 Finish with `make test-cuda-reference-regression` on final source:
  all 204 original artifacts must pass the complete reference suite. Verify actual
  default dense BF16 dispatch, arithmetic 4, full decoder and M3A exact loading
  paths, plus explicit `sglang` equivalence. Record exact source/build/suite
  identities and removal audit. No suite edits, rebaseline, skips or waiver;
  any later code change requires another complete run.

Exit: no fast CUDA mode; one pipeline with explicit dense/Sage2++/Sage3/SOL
choices, inspectable quality differences, stable execution and unchanged default
SGLang parity. Necessary state/consumer work formerly in M5 is handled here.

## M4 — Add independent FP8 and NVFP4 projection options

Depends on M3B. Use the same single CUDA pipeline. The matrix is four attention
choices × three weight precisions; dense BF16 is the default reference cell,
not a separate mode or a thirteenth configuration.

- [x] T028 Dispatch eligible QKV/output/MLP projections by explicit weight
  descriptor before unconditional BF16 branches. Preserve BF16 output conventions;
  keep FP32 patch/velocity heads, norms, encoders and VAEs outside this policy.
  No fast flag or recipe is required to select projection precision.
- [x] T029 Integrate FP8 preparation/scales/kernels with per-option build/device
  capability checks and early explicit errors. Prove default BF16 requests
  neither read nor create packed-weight caches, including after a quantized
  request in the same process. Preserve the default math/library policy.
- [x] T030 Integrate NVFP4 packing/scales, tails and range checks with memory
  admission. Document/report eligible BF16 shape fallbacks; reject nonfinite/
  overflow failures. Keep Sage3 attention precision separate from NVFP4 weights.
- [x] T031 Version quantized caches and state/plan compatibility by metadata,
  base arithmetic, explicit attention/precision recipe and kernel identities,
  using atomic publication and corruption checks without hashing weights.
  Test hit/miss/stale and cross-recipe rejection with bounded preparation;
  fold LoRA before packing. Do not reintroduce mode-based identities.
- [x] T032 Test all twelve attention/precision combinations with bounded operators
  and short sampler runs. Verify independent options, actual dispatch, cache/state
  isolation, cancellation and explicit capability errors; fallback-only execution
  cannot qualify a requested kernel. Preserve default reference state semantics.
- [x] T033 Produce complete 640×480 / 124-frame / six-evaluation clips for every
  supported matrix cell, paired with the dense BF16 default. Report quality
  differences, wall/stage timings and peak VRAM/RAM in validated HTML. Approximate
  cells may degrade quality under the explicit-option contract; retain failed
  stability cases and mark unsupported cells. Finish with the unchanged 204-
  artifact default parity gate and confirm no quantized default dispatch.

Exit: independent documented projection options in one pipeline, an accounted-for
12-cell matrix, inspectable quality/speed tradeoffs and unchanged default parity.

## M6 — Remove remaining legacy CUDA and obsolete infrastructure

Depends on M4 and M3B's consumer/state reachability audit. M5 is removed;
T034–T040 are retired rather than completed. Retain shared primitives only with
identified current consumers. Fast-mode removal is M3B's exit requirement;
this milestone removes residual legacy infrastructure, not another renderer mode.

- [x] T041 Delete the old default renderer, old CUDA RNG/schedule/arithmetic
  dispatch and any remaining unreachable fast-v1/fast-v2 helpers. Preserve the
  single reference-based pipeline and explicit kernel options. Retain helpers
  used by Metal/CPU or migrated consumers; do not delete by filename alone.
- [x] T042 Remove leftover legacy CLI/environment execution, obsolete mode fields,
  state writers and fallback chains. Keep only minimal old-version detection,
  actionable errors and any frozen-probe ABI declarations; no legacy execution
  or hidden compatibility renderer. Preserve recipe-specific state validation.
- [x] T043 Remove dead generated wrappers at their generator, unused dependencies,
  build toggles and tuning branches. Prove remaining primitive consumers,
  including still/continuation/preview, and preserve third-party notices.
- [x] T044 Remove obsolete legacy/two-mode execution tests and tools outside the
  frozen suite; replace needed coverage with explicit-option tests. Preserve
  immutable parity tests and historical evidence without shipping/building old
  renderers. Keep reproducible historical records labeled as such.
- [x] T045 Audit source, linked dispatch, CLI/API/environment and unsupported
  routes for exactly one CUDA pipeline, no fast mode and no orphaned mode
  branches. Check required/optional build variants, cross-recipe context recovery,
  state rejection and bounded Metal/still/preview/continuation smoke tests.
- [x] T046 Pass the unchanged default reference gate after deletion. Record the
  recorded golden-manifest digest, exact build and zero-legacy-reachability audit. Confirm optional
  kernels remain selectable without a mode flag; never rebaseline parity tests.

Exit: one reachable CUDA renderer with explicit kernel/precision options; no
orphaned legacy execution, obsolete fast policy or unsupported compatibility path.

## M7 — Final bounded qualification and documentation

Depends on M6. Preflight each focused campaign to finish within four hours,
including validation/reporting; record incomplete coverage rather than hiding it.
The user reduced optional coverage to the minimum necessary: Sage2++/BF16,
Sage3/NVFP4 and SOL/FP8. Do not rerun the full twelve-cell matrix. Keep BF16
reference gates, C1/C2 replays and conditioning/continuation coverage unchanged.

- [x] T047 Estimate and schedule the three selected option pairs with serialized
  GPU work and one matched default/candidate timing per combination on the same
  source. Use 640×480 / 243 frames / six evaluations for performance comparisons;
  label these single observations without repeated-run statistics;
  keep per-change regression timing separate. Record placement/cache conditions,
  statistics and actual kernels; no separate fast-mode qualification remains.
- [x] T048 Run selected held-outs for no reference, image `match`/`max`, mixed
  image/video/audio and continuation. Include a 1344×768 / 362-frame `max` image
  stress case with the selected explicit attention/precision settings and at most
  two evaluations; label it execution/inspection evidence, not converged quality.
  Report which settings each held-out covers; do not imply untested combinations.
- [x] T049 Replay retained default reference C1 (640×480 / 362 frames / six
  evaluations) and C2 (640×480 / 124 frames / 50 evaluations) once on final source
  against unchanged oracle evidence, checking every update, decoded media and
  allocation history. Do not add a 362-frame / 50-evaluation campaign.
- [x] T050 Generate local HTML playback reports for the three selected pairs with
  paired default/candidate videos, worst frames, audio/temporal metrics, commands,
  wall/stage timings, peak memory and actual-dispatch/fallback counters. Explicitly
  show quality degradation and slower/failed/unsupported cells; verify media links,
  checksums and case counts. Record visual-review status without a human-review wait.
- [x] T051 Document single-pipeline CLI/API/build behavior, independent attention/
  precision choices, measured tradeoffs, removed `--fast-cuda` usage and state
  compatibility. Mark superseded two-mode guides historical. Explain independent
  preview/full decoding, hardware limits, permitted optional quality differences,
  and the unchanged default regression required after every code change.
- [x] T052 Run the final unchanged 204-artifact default gate and close only
  evidenced tasks. Publish the immutable-suite pass history, SGLang parity proof,
  the selected option results and untested scope, performance/quality tradeoffs and no-legacy/no-fast
  audit. Approximate quality differences are permitted, not parity passes;
  historical failed measurements must not be relabeled as successful results.

Exit: one documented CUDA implementation, unchanged default SGLang evidence,
explicit measured acceleration choices, side-by-side videos and a complete
removal audit. No remaining two-mode implementation deliverables.
