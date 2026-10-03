# Configurable adaptive-cache memory budget

Implementation and measured validation are tracked in the
[budget/reference qualification record](adaptive-budget-reference-results.md).

Status: **implemented and capacity-qualified on PRO 5000**. Tasks are
in [the completed checklist](adaptive-budget-reference-tasks.md).
This design was checked against revision `c76222f`.

## Objective and decision

Allow adaptive caching at larger video layouts, specifically **1344×768,
362 frames**, when the device can admit the complete workload. Replace the
fixed 512 MiB ceiling with **`--adaptive-cache-max-mib N`**, defaulting to
**4096 MiB** for new requests. Allocate only the bytes required by the layout.
The flag is a ceiling, not a request to allocate N MiB or a promise that the
whole render fits. Adaptive caching itself remains off by default.

A 4 GiB default covers this target with the current text-token bound. Explicit
512 restores the old resource policy; larger values support future layouts
without recompilation. Use a deterministic numeric ceiling, with existing
GPU capacity checks making the final admission decision. Do not introduce a
second automatic VRAM-percentage heuristic or an unlimited mode.

The [preview campaign](fast-preview-results.md) demonstrated the current cap
blocking 1344×768 even at 124 frames. It also recorded zero adaptive hits in
its short schedules. Raising the budget removes a capacity restriction; it
does not guarantee cache hits, faster rendering or improved visual quality.

## Current implementation and sizing

[The planner](../../src/denoise/adaptive_cache.c) budgets three full-size BF16 tensors:
probe anchor, suffix residual and scratch, plus 6,168 bytes for reduction.
[The host layout](../../src/host.c) packs video, stereo audio and text together.
For supported text-only geometry:

```text
aligned_frames = smallest 5 + 17k >= requested_frames
video_t        = 2 + 5k
audio_t        = round(aligned_frames × 40 / 24)
video_rows     = video_t × (width / 32) × (height / 32)
rows           = video_rows + 2 × audio_t + text_tokens
elements       = rows × 5376
cache_bytes    = elements × 6 + 6168
saved_payload  = elements × 4  # anchor + residual; scratch is rebuilt
```

Use checked integer arithmetic and the actual canvas/layout routines, including
frame alignment. Do not duplicate a different geometry policy in budget code.

| Canvas / frames | Video T | Audio T | Text tokens | Packed rows | Cache bytes | GiB | Minimum whole MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 640×480 / 124 | 37 | 207 | 58 | 11,572 | 373,272,600 | 0.348 | 356 |
| 1344×768 / 124 | 37 | 207 | 58 | 37,768 | 1,218,250,776 | 1.135 | 1,162 |
| 1344×768 / 362 | 107 | 603 | 58 | 109,120 | 3,519,780,888 | 3.278 | 3,357 |
| 1344×768 / 362 | 107 | 603 | 1,024 | 110,086 | 3,550,940,184 | 3.307 | 3,387 |

362 frames are already aligned and represent 15.0833 seconds at 24 FPS. The
58-token rows use the previous campaign's prompt; 1,024 tokens is a planning
bound, not every request's actual sequence. Admission uses the actual packed
layout. Quantized projections do not shrink these BF16 cache tensors.

The pre-change audit identified these coupled resource checks:

- [adaptive_cache.h](../../src/denoise/adaptive_cache.h) and its planner bound
  shape sizing and admission to the former `H3_ADAPTIVE_MAX_BYTES`.
- [dit.c](../../src/denoise/dit.c) plans and allocates the four cache buffers before
  BF16 or packed weight admission. Retain this useful allocation order.
- [sampler_file.c](../../src/sampling/sampler_file.c) used the same constant to bound
  section 40's element count; sections 41/42 store the persistent tensors.
  [sampler_state.c](../../src/sampling/sampler_state.c) also invoked the capped planner.
- [engine.c](../../src/engine.c), [CUDA policy](../../src/cuda/cuda_policy.c), the CLI
  and both resume entry points must carry and revalidate the resource policy.

## Public API and CLI contract

Add `adaptive_cache_max_bytes` and an explicit-selection marker to `h3_params`.
Carry the resolved byte ceiling in `h3_cuda_policy` and the DiT context. Use a
64-bit representation with checked conversion to `size_t` for allocation APIs.
API zero means the new-request default of 4096 MiB; explicit CLI zero is invalid.

`--adaptive-cache-max-mib N` accepts nonempty decimal digits representing a
positive integer whose MiB conversion fits both the byte representation and
platform allocation size. Reject signs, fractions, suffixes, overflow, `auto`
and zero; never clamp. No new environment override or arbitrary numeric upper
ceiling is needed: actual shape, device and file bounds remain authoritative,
and a large N does not itself allocate memory.

Require enabled adaptive caching, independent of argument order. On resume,
defer that check until the saved feature policy is known. Reject explicit use
with cache off, decode-only, still generation, Metal and other unsupported
adaptive layouts. Existing quantization, SubBlock, reference, continuation,
LoRA, reuse and device restrictions remain intact.

Usage example:

```sh
./bin/h3cli -d models/MiniMax-H3 \
  -p 'A woman walks along a sunny beach as gentle waves roll onto the sand.' \
  --width 1344 --height 768 --frames 362 --steps 50 \
  --adaptive-cache conservative --adaptive-cache-max-mib 4096 \
  --cuda-weight-mode auto -o outputs/adaptive-large.mp4
```

This is a usage example, not a completed benchmark. Omitting the budget flag
on a new request has the same 4096 MiB effective ceiling.

## Admission, residency and lifecycle

Separate pure shape sizing from budget validation. Return elements, per-tensor
bytes, reduction bytes, total device bytes and persistent checkpoint bytes,
without consulting available VRAM. Compare the checked total with the resolved
ceiling; equality passes. Cache off retains zero cache allocations, probes and
synchronization.

Resolve syntax and unsupported combinations before expensive model loading.
An early geometry-only lower bound may reject a clearly impossible budget;
never reject merely because a conservative text upper bound exceeds the cap.
Once conditioning/layout is available, perform exact admission before cache
allocation and core weight loading. Distinguish invalid shape/overflow, policy
ceiling exceeded, insufficient device capacity and allocation failure. A budget
error includes required bytes/MiB, configured ceiling, packed rows and the
minimum sufficient whole-MiB value for the flag.

Reserve the three BF16 buffers and reduction storage before weight admission.
Actual allocations already reduce CUDA free memory and enter live tensor
accounting. Do not subtract them again from the planner's future allocation
estimate. Preserve the independent attention workspace, VAE behavior, two-slot
streaming minimum and `max(1 GiB, total VRAM / 10)` safety reserve. Capacity-test
overrides must count the live cache exactly once.

For the 58-token target, the existing BF16 future-allocation estimate alone is
about **37.42 GiB**, before approximately 35.89 GiB of all-resident core weights,
3.28 GiB of cache, other live tensors and the device reserve. The PRO 5000 will
therefore likely select partial residency. Use the existing
[residency planner](../../src/weights/residency.c) and
[qualified streaming behavior](weight-residency-results.md). Do not force full
residency or relax reserves to pass. A 32 GiB GPU cannot be assumed to fit this
layout merely because its cache fits the configured ceiling.

Retain allocation-failure cleanup and bounded residency retries. Retries
preserve budget and cache arithmetic; an impossible cache allocation must
terminate without endlessly retrying weight residency. Release partial
allocations on every failure/cancellation path. Never silently disable caching,
truncate frames or change resolution to admit a workload.

Recheck admission before reusing a prepared/live context, including when a
caller lowers the ceiling. Budget changes do not alter immutable conditioning
or AdaLN contents: keep their content keys stable and perform resource checks
separately. A key hit must not bypass admission.

## Checkpoints and compatibility

The budget is resource policy, not a numerical recipe. Keep adaptive recipe
versions, score reduction order, thresholds, warmups, streak limits, final
refreshes and SubBlock transition behavior unchanged.

Extend required **sampler section 40 to version 2** with the resolved byte
ceiling. Keep BF16 sections 41/42 unchanged and section 44's device/warmup
versions independent. New adaptive checkpoints use section 40 v2; old readers
reject that required unsupported version. Cache-off file layout stays unchanged.

New readers accept v1 under its historical **512 MiB** ceiling, irrespective of
the new default. On resume, omission restores the saved ceiling; an explicit
replacement is allowed if it admits the exact shape. A sufficient higher or
lower ceiling must not invalidate numerical history or prepared content. Save
the effective replacement in subsequent checkpoints. Retain other same-device,
runtime, arithmetic and warmup resume checks.

Treat serialized limits and counts as untrusted. Before allocating adaptive
arrays, validate required flags/versions, checked element and byte counts,
consistency with saved layout, declared/overridden budget and both payload
lengths. A large stored ceiling cannot authorize a shape or allocation that
its file/layout does not support. The CLI/API load path must reject an explicit
too-small resume ceiling before allocating the large adaptive arrays. Preserve
checksums, finite-value checks and committed-history validation.

Keep the existing **16 GiB sampler-file limit** and unrelated section/shape
bounds. The target's two cache arrays alone occupy **2,346,516,480 bytes** at
58 text tokens. Account for latents, prepared tensors, export copies and
serialization buffers too; the current writer duplicates payloads in RAM.
Preflight complete serialized size before a requested save, report host-memory
and disk costs, and test atomic failure/cleanup. GPU admission does not imply
checkpoint admission. Do not silently drop required history or enlarge all file
limits to avoid this check.

Completed clean AV states need no adaptive history for decoding. Keep their
format/version unchanged solely for this option; record budget and allocation
in runtime/experiment metadata instead.

## Diagnostics and qualification

Emit an initialization record with requested/default/restored budget source,
effective ceiling, exact device/persistent bytes, packed rows and tensor count.
Retain per-step cache decisions and weight planner residency/free/live/future/
reserve evidence. Expose the larger allocation in peak-memory reporting.
Report actual hits; a zero-hit result is valid coverage.

Use only the authorized RTX PRO 5000 for CUDA and local M4 for CPU tests, Metal
build compatibility and review. Preserve CUDA 13.0.3/cuDNN 9.20. No new server,
SOL campaign, quantized comparison matrix, SGLang oracle or golden expansion is
part of this feature.

1. CPU tests cover strict parsing, defaults, disabled-feature combinations,
   exact boundaries, small ceilings, overflow, large shapes without allocation,
   and all sizing-table rows. Test actual and planning-bound text counts.
2. State tests cover legacy v1, required v2, malformed counts/budgets/lengths,
   oversized containers, overrides, prepared-context reuse and atomic write
   failure. Keep synthetic corruption fixtures small.
3. On an old supported shape, admitted 512/4096/8192 MiB ceilings must produce
   identical step latents and decisions. This checks resource invariance,
   without adding independent SGLang numerical parity coverage.
4. At 1344×768/124 frames, verify admission above the old cap, low-budget
   rejection and cancellation/retry. At **1344×768/362 frames**, execute a
   bounded prefix of an original 50-step schedule and compare continuous
   execution with stop/resume. Exercise default and sufficient explicit budgets
   and a too-small override. Record actual partial/streamed residency. Each
   bounded invocation executes at most six evaluations.
5. At a small supported shape, exercise low-capacity fault injection and forced
   streaming: reservations, finite retries, history and cleanup must hold.
   Include conservative/aggressive and BF16 adaptive+SubBlock composition.
   Preserve existing packed-projection policy/accounting through applicable
   tests; no additional FP8/NVFP4 video campaign is required.
6. Produce **one complete 1344×768, 362-frame, six-step text-only BF16 video**:
   conservative cache, default warmup 4, omitted budget flag (4096 MiB), all 50
   blocks, default dense attention, full VAE, audio and auto weight residency.
   Use the previous preview campaign's prompt and seed 42. Confirm exact
   dimensions, 362 frames at 24 FPS, approximately 15.0833 seconds, stereo
   audio, finite outputs and full media decode. This validates capacity and
   decoding, not quality or speedup. Use a 3,600-second per-invocation deadline
   for this video and large bounded tests; timeouts/failures remain failures.
7. Build Metal and CUDA and run applicable ordinary, policy, CLI, state and
   residency checks. Follow [CONTRIBUTING.md](../../CONTRIBUTING.md): after each
   coherent code/build/test-tool change, pass the complete unchanged
   **204-output golden regression**, with its own 720-second test deadline.
   Preserve all fixtures, hashes and ordinary six-evaluation test limits.

Keep artifacts under ignored `outputs/adaptive-cache-budget/`, binaries under
`bin/`, and machine-specific paths outside tracked documents. Record commands,
revision, binary/runtime identities, policy/allocation bytes, weight residency,
actual cache work, sampled VRAM/RSS, checkpoint size, stage timings and total
process wall time. Hash generated artifacts; identify base weights by metadata
without scanning their payloads.

Acceptance requires the target video and large checkpoint/resume checks to
pass on the PRO 5000. A larger constant, a successful plan calculation, an OOM
marked skipped, or a smaller substitute render is insufficient. Report hardware
limits and observed cache hits separately from admission.
