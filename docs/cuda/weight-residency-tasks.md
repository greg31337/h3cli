> Archived 2026-09-27. All 31 weight-residency tasks are complete.
> Active work is tracked in [todo.md](../todo.md).

# CUDA automatic and partial BF16 weight residency

Status: **complete; 31/31 tasks complete**. Results and qualification limits are
recorded in the [qualification report](weight-residency-results.md).
The completed, accepted [latent-upscale checklist](../experiments/latent-upscale-tasks.md)
and [adaptive/SubBlock checklist](adaptive-subblock-tasks.md) remain archived.

Goal: make `--cuda-weight-mode auto` use full GPU residency when safe, retain
as many BF16 DiT blocks as safely fit otherwise, and stream only the remaining
blocks. Preserve native SGLang arithmetic, saved-state compatibility and exact
outputs. Qualify partial BF16 residency on **RTX PRO 5000 72 GB at 1344×768**
and on a **physical RTX 5090 node**.

## Scope and implementation contract

- Implement capacity-aware `auto` and partial BF16 residency. FP8/NVFP4 inherit
  corrected full-resident versus compressed-stream admission, with bounded
  regression coverage only. New packed-weight prefetch, partial quantized
  residency, direct disk I/O, new precision/attention kernels, multi-GPU support
  and text/VAE placement changes are outside this work.
- Preserve the public modes: `resident` means the entire eligible core must fit
  or the request fails; `stream` means zero persistently resident core blocks;
  `auto` may choose full, partial BF16 or fully streamed execution. Existing
  `--ssd-streaming` keeps its explicit streaming behavior; conflicting explicit
  controls must fail preflight. A new public tuning flag is not required.
- A residency plan belongs to its context and actual packed geometry. Preserve
  execution of every active block; residency never means layer skipping.
  Start with a deterministic leading-active-block resident set, retaining block
  0 when present. Prefetch the next nonresident block across resident gaps.
- Retain the two BF16 streaming slots and their copy/compute fences whenever
  streaming is needed. Resident blocks use their own tensors directly. Admit
  pinned host copies only for nonresident blocks, retaining the current 40 GiB
  cap, memory guard and bounded fallback. Do not retain duplicate host weights
  just because a block is resident on the GPU.
- Keep AdaLN precomputation, arithmetic order, dtype boundaries, attention,
  sampler/RNG and decoder behavior unchanged. Placement is execution state,
  not a new numerical recipe or a hardware-dependent sampler-file requirement.
  Replan on load/resume and incompatible retained-context reuse.
- Preserve Metal's existing resident/SSD policies. Keep the full video decoder
  resident across tiles; release or evict DiT resources at the existing phase
  boundaries when decoding needs headroom.
- Follow [CONTRIBUTING.md](../../CONTRIBUTING.md): after every coherent source,
  build or test-tool patch run the complete unchanged **204-output CUDA golden
  regression**, within its existing deadline. Keep all fixtures, hashes and
  tolerances unchanged; placement tests supplement this gate.
- Use the qualified CUDA 13.0.3/cuDNN 9.20 environment from the
  [reference guide](cuda-sglang-reference.md). Keep connection details and
  machine-specific paths outside tracked files. Do not hash/scan H3 base weights.
  Native code stays in `src/`, binaries/libraries in ignored `bin/`, generated
  evidence/media in ignored `outputs/`.

## Validation matrix

All primary rows use dense BF16, all 50 blocks, reuse/core-reuse 1, no adaptive
cache or sparse attention, a fixed prompt/seed, full AV decoding, **90 frames
at 24 FPS and six denoising evaluations**. Freeze exact inputs before changes.
This is placement/correctness/performance qualification, not a new visual-quality
campaign. Keep ordinary tests within their six-evaluation limit.

| Hardware | Canvas | Required comparison |
| --- | --- | --- |
| RTX PRO 5000 72 GB | 640×480 | Explicit resident, forced stream and unmodified `auto`; prove that `auto` selects full residency when admitted. |
| RTX PRO 5000 72 GB | 1344×768 | Natural `auto`, forced stream, feasible explicit resident control, and deliberately exercised partial BF16 plans. Use deterministic test caps if natural `auto` fits the full core. |
| RTX 5090 32 GB | 640×480 | Natural partial `auto`, forced stream, and at least two feasible partial residency counts; explicit full BF16 residency must reject insufficient capacity. |

- Use bounded geometry-matched fixtures for edge resident counts and allocation
  failures. Test-only budget/count caps must only reduce available capacity and
  be visible in reports. A forced partial run proves that path; it does not prove
  the production planner would select it naturally.
- The 1344×768 RTX PRO 5000 rows are mandatory even if 640×480 passes. Include a
  672×384 → 1344×768 upscale refinement lifecycle probe with at most six total
  denoiser evaluations; rebuild the residency plan for the target geometry.
- Never replace the physical 5090 qualification with a capped RTX PRO run.
  Record the real GPU, usable VRAM, host RAM/cgroup limits, PCIe link under load,
  storage and software. Run GPU jobs serially per node.
- OOM, missing hardware/model, timeout or a streamed fallback where partial
  execution was required is not a pass. Preserve failed attempts. Distinguish an
  expected strict-resident capacity rejection from a successful resident run.

## M0 — Freeze contracts and controls

- [x] WR001 Record the starting source/build/environment and a complete baseline
  golden pass. Retain baseline streaming outputs and phase timings for both
  nodes before modifying production code. Use file metadata for model identity.
- [x] WR002 Audit all planner callers, actual BF16/packed tensor sizes and phase
  lifetimes. Account for existing allocations separately from future allocations:
  packed reference/continuation rows, AdaLN, norms/heads, activations, attention
  and quantization scratch, adaptive buffers, allocator/library workspaces,
  streaming slots and retained contexts. Document estimate-versus-observed peaks.
- [x] WR003 Specify a context-owned plan with requested mode, effective mode,
  resident block set/count, streamed order, byte budgets, reserve and reason.
  Define exact boundary behavior, explicit-option conflicts, monotonic OOM
  fallback, and cache/replan semantics before changing allocation code.
- [x] WR004 Freeze a machine-independent comparison manifest for the matrix,
  exact prompt/seed, caps/counts, attempt IDs, timeouts, repetitions and numerical
  gates. Build deterministic test controls that force full/partial/zero residency
  without disabling memory checks or affecting ordinary runs.

Qualification note: the PRO 5000 passed all 204 recorded goldens after every
coherent patch. The unchanged 5090 baseline differs on 113 cross-device golden
hashes; every candidate, including the final build, matches all 204 artifacts
from that same 5090 baseline exactly. This is not a 5090 recorded-golden pass,
and the frozen golden manifest has not changed. Full-resolution controls,
composition, real-weight fault probes, all 14 high-resolution upscale lifecycle
cases and the final evidence/documentation audit pass. See the
[qualification report](weight-residency-results.md).

Exit: reproducible baselines and a concrete capacity/lifetime contract.

## M1 — Capacity-aware automatic placement

- [x] WR005 Implement checked byte arithmetic and a testable planner. Select full
  residency without unused streaming slots when it fits. For partial BF16,
  reserve the complete streamed working set first, then select the maximal safe
  resident count. Avoid double-counting live allocations or overlapping peaks
  from phases whose resources are released; fail clearly if even streaming fails.
- [x] WR006 Remove the shared SGLang path's unconditional `auto` streaming rule.
  Apply real capacity admission to BF16 and existing packed precision callers;
  packed modes retain only their current full-resident/streamed choices. Use
  actual descriptor sizes and include adaptive block-0 BF16 storage where needed.
- [x] WR007 Preserve strict `resident`, forced `stream`, environment/API and
  `--ssd-streaming` semantics. Validate conflicting controls before loading and
  capture placement settings in the request/context so later environment changes
  cannot mutate a live plan. Keep unrelated CLI modes and Metal behavior intact.
- [x] WR008 Implement bounded allocation-failure recovery for `auto`: drain work,
  release the failed attempt, reduce residency and retry down to bounded streaming.
  Avoid repeating the same failed plan, leaking pinned/device storage or leaving
  stale CUDA errors. Explicit `resident` stays strict; non-allocation failures
  remain errors, not reasons to silently change placement.
- [x] WR009 Expose requested/effective placement and the budget explanation at
  startup. Distinguish full, partial and zero residency; record every fallback
  and rejected explicit request. Pass shared policy/CLI checks and the full gate.

Exit: `auto` can select full residency safely; explicit modes remain predictable.

## M2 — Native partial BF16 residency

- [x] WR010 Allocate persistent matrix tensors only for resident blocks; prepare
  stream descriptors only for the remainder. Keep small shared/norm tensors
  valid in every mode. Handle zero, one, all-but-one and all active blocks, and
  reduced/noncontiguous active sets without confusing residency with execution.
- [x] WR011 Extend the existing prefetch scheduler to find the next streamed
  block across resident runs and wrap at evaluation boundaries. Compute resident
  blocks directly, avoiding copies into the streamed slots. Prime only necessary
  slots and avoid unnecessary wraparound uploads at completion or pause.
- [x] WR012 Preserve explicit upload-ready and last-consumer-release fences for
  every slot generation. Cover transitions between resident and streamed blocks,
  delayed copies, overwritten slots, prefetch completion/error and teardown.
  No device-wide synchronization or changed GEMM/kernel policy on the steady path.
- [x] WR013 Restrict host cache admission/preloading to nonresident matrices.
  Preserve immutable metadata keys, the process-wide pinned cap, admission
  reserve and copied/bounce fallback. Release registrations safely after in-flight
  copies; prove pinned bytes fall with the streamed set and do not grow per step.
- [x] WR014 Unify ownership and rollback for resident matrices, slots, descriptors
  and prefetch jobs. Exercise cancellation and injected allocation/read/registration
  failures during initialization, block execution and teardown, followed by a
  successful request in the same process.
- [x] WR015 Integrate BF16 adaptive-cache hits/misses, reuse/core-reuse and
  SubBlock warmup/phase boundaries. Preserve skip decisions and defer suffix
  uploads until needed. Every executed block must use its correct weights;
  skipped evaluations/blocks must not cause stale slots or unnecessary transfers.
- [x] WR016 Reconcile cached DiT contexts with mode, geometry, active blocks,
  precision, model/LoRA and available-memory changes. Prove same-build resume
  can replan between full/partial/stream modes with identical sampler payloads;
  do not serialize device pointers or require the original hardware placement.
- [x] WR017 Integrate phase handoff and upscale refinement. Retire host/device
  weights when needed for full-VAE admission; rebuild on target geometry or
  incompatible retained state. Verify decode-only, state-only, paused preview
  and subsequent requests without retaining an obsolete source-resolution plan.

Exit: partial residency executes the existing BF16 math with bounded ownership.

## M3 — Instrumentation and correctness qualification

- [x] WR018 Report resident/slot/pinned bytes, actual uploaded bytes, host-cache
  hits, exposed transfer waits and phase timings. Separate initialization from
  repeated denoising. Distinguish logical file reads from physical disk I/O;
  collect GPU memory, process RSS/anonymous memory, swap and disk-read counters.
  Profiling must not add synchronization to ordinary execution.
- [x] WR019 Add meaningful planner tests for fit boundaries, overflow, reserves,
  already-live allocations, insufficient streaming headroom, active block sets,
  packed descriptor sizes and deterministic caps. Test strict modes and fallback
  termination, rather than merely copying implementation formulas into assertions.
- [x] WR020 Update/extend the memory and slot probes to enter the production
  shared SGLang request path. Exercise full/partial/stream on real weights,
  resident gaps and boundary counts; assert both the executed plan and expected
  transfer accounting. Do not rely solely on the historical low-level probe.
- [x] WR021 Compare fixed native preparation, velocities and video/audio latent
  trajectories bitwise across placements, including failure-driven replanning.
  Compare decoded RGB/PCM separately from container/provenance metadata. Use
  captured baseline controls and unchanged goldens; do not regenerate expectations.
- [x] WR022 Run bounded lifecycle/composition probes: FL2VA anchors, ordered
  Ref2VA inputs, continuation, saved-state resume, adaptive hit/miss, SubBlock,
  retained-context changes and upscale refinement. Check decisions and outputs
  against the matching precision/feature control, plus recovery and allocation
  stability across repeated requests.
- [x] WR023 Verify corrected FP8/NVFP4 `auto` admission with bounded controls:
  admitted packed cores stay resident, constrained cores use the existing packed
  stream path, and outputs match explicit placements. Add no quantized video
  campaign, new quantization recipe or packed prefetch implementation.
- [x] WR024 Build CUDA on both target nodes and Metal locally. Run appropriate
  shared, CLI, state, memory and scheduler tests; prove Metal's ordinary and
  explicit SSD-streamed paths retain their previous policy and results.
- [x] WR025 Pass the complete unchanged 204-output CUDA gate on the candidate
  and bind the result to source/binary/environment identities. Run it after each
  coherent implementation or test-tool patch; finish this gate before timing.

Exit: exact placement equivalence, lifecycle safety and both backends qualified.

## M4 — Hardware measurements and closeout

- [x] WR026 Complete RTX PRO 5000 640×480 controls. Verify natural `auto` really
  selects full residency, eliminates repeated core H2D traffic and matches the
  explicit resident result. Preserve the forced-stream control and full timings.
- [x] WR027 Complete RTX PRO 5000 1344×768 controls, including at least two
  feasible nonzero partial counts below 50. If natural `auto` is fully resident,
  keep that result and force partial via labeled test caps; do not call a
  resident-only run partial qualification. Complete the high-resolution upscale
  lifecycle probe and decoder handoff under constrained placement.
- [x] WR028 Complete physical RTX 5090 qualification with natural `auto`, forced
  stream and two feasible partial counts. Demonstrate actual partial execution,
  strict full-resident capacity handling, bounded host memory and complete AV
  delivery. Record real link/storage behavior; do not extrapolate RTX PRO timings.
- [x] WR029 Run three interleaved matched timing pairs per node/canvas for the
  final automatic policy versus forced streaming. Keep source, inputs and math
  fixed; distinguish cold loading from warmed denoising. Report every pair,
  medians/spread, end-to-end and phase times, transfer waits/volume and memory.
  Explain neutral/slower cases; claim speedups only beyond measured variation.
- [x] WR030 Publish the placement contract, qualification report and current
  README/help guidance. Document workload-dependent admission, actual resident
  counts, host-memory tradeoffs, explicit-mode errors and measured limitations.
  Keep historical device-cache results clearly separate from this implementation.
- [x] WR031 Audit all task evidence, hardware coverage and final source identity
  against the last complete gate. Validate document links and artifact manifests;
  rerun the gate after any later code/tool change. Leave missing or failed cases
  unchecked. Close only when both required nodes and the 1344×768 partial path
  pass, with unchanged numeric outputs and no unresolved lifetime failures.

Exit: a qualified automatic policy, usable partial BF16 residency and reproducible
measurements. No numerical-quality waiver substitutes for placement equivalence.

## Reference points

- Current code: [planner and host staging](../../src/cuda/gpu_cuda.cu),
  [DiT allocation/prefetch](../../src/denoise/dit.c), [AdaLN preparation](../../src/denoise/dit_schedule.c),
  [packed admission](../../src/cuda/cuda_quant.cuh), [phase handoff](../../src/engine.c).
- [Shared pipeline contract](design-single-pipeline.md) and
  [accepted resident-weight measurements](adaptive-quant-experiment.md).
- [Historical 5090 cache experiment](cuda-5090-qualification.md): the old
  device cache was removed; its timings are not results for this implementation.
- [SGLang H3 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx)
  and [layerwise offload implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/managers/memory_managers/layerwise_offload.py).
  Record the consulted upstream revision during WR001. Reuse placement ideas,
  not upstream layer counts, arithmetic changes or timing claims.
