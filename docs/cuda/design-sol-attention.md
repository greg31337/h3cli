> Historical renderer/qualification record. Commands that refer to removed
> fast/legacy modes require the archived source. Use the
> [current single-pipeline design](design-single-pipeline.md) and
> [M6/M7 qualification](single-pipeline-final.md) for supported execution.

# Native CUDA SOL attention

Status: implemented and bounded validation completed, 2026-09-23. Final gates
pass with retained development failures and one tiny checkpoint retry; see
the results and coverage ledger. Runtime remains opt-in.
Tasks: [todo.md](../todo.md). This document does not change runtime defaults.
Implementation details and source provenance: [audit](sol-implementation-audit.md).
The final [results](sol-attention-results.md) distinguish measured support from
the intended contract below.

## Objective and scope

Add an opt-in `--cuda-attention sol` implementation for main H3 DiT attention.
Port the existing H3 SOL routing/protection semantics to native CUDA, retaining
BF16 Q/K/V and output with FP32 softmax statistics and accumulation. Measure
complete denoising and process wall time against the unchanged CUDA dense path.
SOL approximates distant, low-importance blocks; it is not exact dense attention
or hardware 2:4 structured sparsity.

Initial hardware qualification targets the current node,
`cuda-pro6000`, previously measured as an **RTX PRO 6000 Blackwell
Server Edition, SM120, 95.59 GiB VRAM**. Re-probe the GPU UUID, capabilities,
driver, toolkit, available memory and disk before testing. Models were located
at `/path/to/models/MiniMax-H3`. Do not identify this node as an RTX 5090 or
extend its results to other GPUs. SM120 is the initial runtime capability gate;
other architectures remain future qualification work.

All planned validation, including build verification, warmups, fixture capture,
failed attempts, media checks and downloads, must finish within **480 minutes**
on this single node. The scheduled work totals **445 minutes**, including
30 minutes for closing, with **35 minutes of contingency**. Implementation
development is separate from this one bounded validation campaign. An overrun
or failed prerequisite is reported honestly; restarting the clock or silently
dropping mandatory tests does not count as completing the plan.

Retain the latest **two-step limit for every generated video**. Keep the main
performance geometry at **640×480, 243 frames**, with high-resolution coverage
and a **1344×768, 362-frame, nine-image `max`** stress pair. Two-step outputs
support functional validation and inspection, not final-render perceptual
qualification. No human-review response is required to close the campaign.

The initial scope excludes new model architectures, low-bit SOL attention,
new Metal kernels, full Sage/SOL combinations and an exhaustive GPU/quality
matrix. Existing encoders, refiner, VAE, weights, sampler, seed semantics and
projection implementations remain unchanged. Default and Sage modes stay
available under their existing contracts.

## Existing implementation and integration points

| Area | Current behavior | Planned change |
| --- | --- | --- |
| [src/denoise/attention.h](../../src/denoise/attention.h), [src/denoise/attention.c](../../src/denoise/attention.c) | Modes 0/1/2 are default/Sage2++/Sage3; non-default preflight assumes Sage | Add a distinct SOL mode and mode-specific build/device checks; preserve existing numeric IDs |
| [src/gpu.h](../../src/gpu.h), [src/cuda/gpu_cuda.cu](../../src/cuda/gpu_cuda.cu) | `h3_gpu_dit_sdpa_bf16()` is the explicit main-DiT entry point; configure allocates a Sage workspace for every non-default mode | Separate SOL context/path/to/dispatch from Sage; extend configuration with immutable policy and layout metadata |
| [src/denoise/dit.c](../../src/denoise/dit.c) | Main blocks pass block and absolute evaluation indices; Metal receives noise and protected layout metadata | Supply equivalent CUDA metadata and both schedule sigmas without relying on Metal-only hooks |
| [src/denoise/sol.c](../../src/denoise/sol.c), [src/denoise/sol.h](../../src/denoise/sol.h) | Host code builds query/key protection metadata from packed segments | Reuse and test on CUDA; keep Metal behavior and serialized recipes compatible |
| [src/metal/sol.metal](../../src/metal/sol.metal) | BF16 centroids, diagonal variance router, protected/local/floor routes and stable merge | Mathematical reference for a separately versioned CUDA recipe |
| Retired `h3_cuda_fast_attention.cuh` (shared primitives now retained in [src/cuda/cuda_sol.cu](../../src/cuda/cuda_sol.cu)) | Existing BF16 Tensor Core tiled attention, FP32 accumulators, sequence-major inputs | Reuse audited primitives and pipeline patterns for a routed kernel; leave the dense implementation intact |
| [src/h3cli.c](../../src/h3cli.c), [src/h3.h](../../src/h3.h), [src/backend.c](../../src/backend.c) | `--sol-*` currently populates Metal options and requires the Metal backend | Parse shared SOL controls independently, then resolve them into the selected backend policy |
| [src/engine.c](../../src/engine.c), [src/sampling/sampler_state.c](../../src/sampling/sampler_state.c), [src/sampling/sampler_file.c](../../src/sampling/sampler_file.c) | Prepared keys, exact-resume validation and extension 35 carry CUDA attention identity | Persist the full CUDA SOL recipe/policy and reject incompatible exact resume |

The supported operator is B=1, H=KV=56, D=128, noncausal BF16 self-attention.
CUDA Q/K/V are `[sequence,head,128]`. The existing `head_major` parameter selects
the **output** layout; it does not describe the input. Support both output
layouts, valid sequence tails and all packed reference/continuation segments.
Never infer the main-DiT domain from head count: the text refiner also has 56
heads. Diagnostic kernels may support fewer heads for bounded oracle tests.

## CLI, defaults and compatibility

Usage:

```sh
./bin/h3cli -d /path/to/models/MiniMax-H3 \
  --cuda-attention sol \
  --sol-q-block 32 --sol-kv-block 64 --sol-tau 1 \
  --sol-dense-layers 1 --sol-dense-steps 1 --sol-dense-sigma -1 \
  --sol-local-radius 1 --sol-min-exact 0.75 \
  --width 640 --height 480 --frames 243 --steps 2 \
  --ref-image inputs/2.jpg --ref-image-size max \
  -p 'A person walks through a sunlit courtyard.' -o outputs/sol-cuda.mp4
```

The command is a functional example, not a quality recommendation. Initial
CUDA SOL defaults are the explicit values above. The 0.75 minimum is a new,
conservative CUDA starting point, informed by the retained Metal experiment;
it is not an upstream SOL default or an already qualified CUDA preset. Preserve
the existing Metal defaults, including its legacy 0.1 floor. Do not silently
reinterpret old commands when extracting common policy helpers.

Version 1 supports query blocks 32/64 and K/V blocks 64. Reject other CUDA tile
values explicitly while retaining the existing Metal choices. The requested
query block defines routing granularity: a physical CTA must not silently
combine four 32-row masks into one 128-row decision. Fix one default tile and
one alternate diagnostic tile; avoid a broad tuning sweep.

`--sol-*` flags may appear before or after backend selection. Resolve explicit
option bits after parsing, with backend-specific defaults. Require SOL mode
for CUDA SOL controls; reject mixed Metal/CUDA selection, NaN/Inf/out-of-range
values, `H3_CUDA_REFERENCE=1`, and conflicting non-`auto`
`H3_FAST_CUDA_ATTENTION` overrides before model loading or LoRA folding.
`--fast-cuda` remains independent and optional. Do not require it to opt into
SOL or change unrelated kernels implicitly.

Build the feature with `CUDA_SOL=1` / `H3_CUDA_USE_SOL`, independently of
`CUDA_SAGE=1`. A SOL request on Metal, an unsupported GPU or a build without SOL
must fail clearly. No Python, PyTorch, Triton or downloaded dependency belongs
in inference. The test oracle may use an isolated Python environment. CUDA SOL
does not require importing Sage or adding a second CUTLASS dependency.

Projection quantization remains a separate setting: SOL still consumes BF16
activations from FP8/NVFP4 projections. Include bounded composition checks, but
make performance/quality claims only for configurations actually measured.
Reject token reduction and step/core reuse greater than one in this first SOL
recipe, since they change packed-layout or evaluation semantics. Preserve
streamed/resident weight support under the existing memory planner.

## Versioned SOL arithmetic

Use the current local SOL algorithm as the definition, not a claim of bitwise
identity with VPIPE. The [Metal source audit](../metal/m1-attention-sources.md)
documents the differences. Existing Metal quality acceptance does not qualify
a new CUDA kernel or waive its tests.

For each head and each query/key block:

1. Reduce original BF16 Q/K/V in FP32 and store BF16 block centroids. Use valid
   row counts for means; padded rows never contribute. Recompute summaries
   for every evaluated layer and step, never reuse previous activations.
2. Compute per-channel mean/diagonal variance over key centroids in FP32.
   For query centroid `qc`, estimate the log2-score distribution:

   ```text
   a = attention_scale * log2(e)
   mu = a * dot(qc, mean(kc))
   variance = a*a * sum(qc[d]*qc[d] * variance(kc[d]))
   threshold = mu + tau * sqrt(max(variance, 0) + 1e-6)
   exact_importance = a * dot(qc, kc) > threshold
   ```

   Record the scale's rounding convention; match the existing CUDA BF16 dense
   score-scale contract rather than accidentally adding another quantization.
3. Force exact treatment for protected query/key blocks, local temporal
   neighbors, the deterministic minimum-exact subset and incomplete key tails.
   Use `ceil(min_exact * key_blocks)` and the existing evenly distributed
   subset rule. This is a minimum, not a top-k budget. Larger tau generally
   reduces exact work. Report realized coverage, not the requested floor.
4. For exact blocks, execute ordinary tiled BF16 attention with FP32 online
   maxima, normalization sums and numerator accumulation. Audit any BF16
   rounding of probability operands used by Tensor Core P×V and include it
   in the recipe/oracle tolerances; FP32 accumulation is not FP32 operands.
5. For each approximate key block, use the **original query row**, the K/V
   centroids and multiplicity `valid_key_rows`. Query centroids select routes;
   they do not replace query rows in the approximate attention computation.
   Its log2 logit is `a*dot(q, kc) + log2(valid_key_rows)`.
6. Combine contributions through one mathematically consistent online
   softmax. With unnormalized numerator `o`, maximum `m` and denominator `l`
   for each branch, use:

   ```text
   m = max(m_exact, m_approx)
   we = exp2(m_exact - m); wa = exp2(m_approx - m)
   result = (we*o_exact + wa*o_approx) / (we*l_exact + wa*l_approx)
   ```

   Handle empty branches without evaluating undefined infinity differences.
   Apply multiplicity exactly once. Do not add separately normalized outputs.
   Cast the final result to BF16 in the requested output layout.

The reference implementation may use separate exact/centroid/merge kernels
over bounded slabs. Optimize or fuse them only after the independent operator
oracle passes. Keep exact K/V traversal in ascending order, deterministic floor
selection and fixed reduction/partition order for a frozen device/recipe.

Nonfinite input is an error, not a reason to silently continue with another
attention backend. For finite input, an invalid routing statistic can force
that query block fully exact and increment an explicit recovery counter;
nonfinite final output still fails. Carry device error flags to existing
synchronization boundaries; avoid per-layer CPU readbacks.

## Dense policy and protected regions

Apply whole-call dense overrides in this order: invalid policy/noise is an
error; `min_exact=1`; absolute evaluation index `< dense_steps`; enabled
`max(video_sigma,audio_sigma) >= dense_sigma`; block index `< dense_layers`.
`dense_sigma=-1` disables that threshold. Use the absolute sampler index after
resume, not a new process-local counter. Dense overrides call the same default
CUDA attention route as the matched baseline, with the same fast-mode setting.
Report the reason and distinguish this bypass from a test-only forced-exact
routed kernel, so an untested SOL implementation cannot pass by bypassing it.

Reuse `h3_sol_layout_build()` and freeze/upload its metadata once per immutable
DiT layout. Every block overlapping any protected row is wholly protected:

- All text, image/video reference, vision and audio rows, including generated
  audio and audio continuation prefix rows.
- The entire target-video continuation prefix and first/last target latent
  frames, including anchor use cases.
- Local exact neighborhoods measured in target **latent video frames**, not
  packed block distance or decoded video-frame numbers.

Protected queries attend exactly to **all** K/V. Protected keys contribute
exactly to **every** query. Mixed-boundary blocks and incomplete key tails
remain exact. Protected keys alone do not make an unprotected query's complete
output equal dense attention: the other contributions and denominator can
still be approximate. Test these two obligations separately.

Validate all current continuation-prefix sizes on the host and at least one
real CUDA continuation workflow. Require byte-identical stored prefix state
before/after sampling. A single-still layout has its only video frame protected
and must behave as fully exact, with an explicit no-sparsity record.
This is an operator-level compatibility check. Preserve the existing
`--still` dense-BF16 CLI restriction; enabling accelerated still generation
and qualifying its image-decoder outputs is outside this video campaign.

## Kernel scheduling and bounded memory

Use the existing native BF16 Tensor Core primitives as the starting point,
adapting the query tile and K/V loading pipeline for actual route skipping.
Do not load and multiply every K/V tile before applying a route mask. The
centroid branch still visits all approximate summaries; account for this cost.
Retain licensing notices on adapted code. Any additional imported source needs
a pinned revision, file hashes and its original license/NOTICE before vendoring.

Reserve at most **512 MiB of additional SOL workspace per DiT context**, including
summaries, statistics, route storage, partial numerators/denominators, counters,
alignment and temporary conversion buffers. Existing Q/K/V/output are separate
activations and must still be counted in total memory admission.

Never allocate a full token-by-token attention matrix, all-head FP32 partial
outputs, or an unbounded `heads × query_blocks × key_blocks` route tensor.
Choose head groups and query slabs from shape and the fixed workspace cap.
Key centroids span the complete sequence; splitting query work must not truncate
the key domain. Reuse the same allocation across layers/evaluations. A sequence
like the 362-frame high-resolution reference case must not hit an old small-N
cutoff: determine feasibility from checked sizes and the bounded plan.

Use checked 64-bit indexing/byte arithmetic and 64-bit aggregate pair counts.
Compute and reserve workspace before optional weight-cache admission. Shape
changes rebuild the plan and protection metadata; a policy change changes the
prepared-context key. On allocation/launch failure, release partial resources
and fail without silently lowering the exact fraction or switching precision.
No free-memory-dependent arithmetic plan, uncontrolled OOM retry or persistent
per-layer allocation is allowed. Preserve existing host/GPU memory guards and
include SOL allocations in their accounting.

Expose summary, routing, exact, centroid, merge/output and total attention
CUDA-event timings. Report requested/effective modes, recipe/plan, sequence
length, block/evaluation, head/slab counts, workspace high-water mark and
effective exact/approximate/protected/local/floor/recovery counts. Explain
overlapping counter categories; exact plus approximate must equal all pairs.
Collect counters at existing step boundaries rather than synchronizing each
layer. Include denoising-step and total process wall times without profiling
fences in the end-to-end measurements.

## Cache, checkpoint and delivery identity

Add a dedicated CUDA SOL policy structure to `h3_params`; do not serialize the
Metal options structure as CUDA state. Factor shared mathematical helpers only
where their semantics truly match. Nested request scopes restore the complete
previous policy on every exit.

Prepared-context keys include all SOL options, arithmetic/plan versions and
layout/protection identity. Keep default/Sage identities and model/LoRA/packed
projection cache keys unchanged. Scratch buffers may persist; activation
summaries and masks must be recomputed, not cached as reusable model data.

Keep existing default/Sage checkpoint decoding compatible. Extension 35 already
stores CUDA mode/recipe/plan; allocate a new required SOL-policy extension after
checking the current registry, without reusing Metal extensions 36/37. Missing,
unknown or inconsistent SOL policy must fail validation. Use per-mode version
selection so adding SOL does not gratuitously invalidate existing Sage recipes.
Exact resume restores SOL and absolute schedule position; explicit conflicting
flags, unavailable capabilities, or `--resume-default-cuda` must fail. A completed
AV state can start a new continuation segment under a newly selected policy.

Extend completed-state presentation provenance with SOL settings using a new
sidecar version while accepting older versions. Decode-only replay must still
work without SOL kernels or transformer weights and reject generation-only
SOL CLI options. Test both public API and CLI validation paths.

## Acceptance contract

Freeze `tests/cuda_sol_acceptance.json` before tuning or held-out execution.
Use separate calibration and held-out seeds/prompts. These are proposed
conservative screens, not claims that current Metal or future CUDA results pass:

| Gate | Requirement |
| --- | --- |
| Operator correctness | Independent scalar/blocked oracle for the **defined approximate operator**, including centroid rounding and multiplicity; relative L2 ≤0.01 and max absolute error/reference peak ≤0.02; use absolute error ≤1e-5 for a zero-norm reference |
| All-exact implementation | Exercise the routed kernel with every route exact, both output layouts and tails, against independent dense SDPA; same numerical bounds. Separately prove the public dense bypass unchanged |
| Protection and safety | No omitted protected pair, no input/guard corruption, finite output, no invalid-input output commit, unchanged stored continuation prefix; tiny CUDA sanitizer cases clean |
| Approximation screen | On held-out real inputs, full teacher-forced video/audio velocity relative L2 ≤0.05 and cosine ≥0.998, reported separately by modality; protected query attention satisfies the dense operator bound |
| Performance | Target ≥1.10× routed-step speedup at 640×480/243 frames, positive approximate work, and no >5% full-process regression on the primary matched pair; publish all other pair results, including regressions |
| Memory/lifetime | Fixed workspace ≤512 MiB; stable live allocation count/bytes after initialization across repeated synthetic dispatches and context churn; all GPU ownership released after each render |

Measure operator replay variance using one warmup and five timed repetitions
on identical QKV; these are attention operations, not extra denoising steps.
Time three paths on these inputs: existing default dense, forced-exact SOL
kernel, and routed SOL. This separates kernel implementation gains from the
incremental gain from approximation.
Two-step whole-render timings have limited statistical strength: label them
provisional, never a steady 50-step throughput or final-quality claim. Keep
algorithm-correctness failures distinct from approximation-quality failures.
Default execution stays dense even if gates pass.

Compare teacher-forced velocity from the same saved dense evaluation boundary;
do not conflate independent-trajectory divergence with local operator error.
The single second evaluation in a two-step schedule must actually route after
its dense prefix. Record positive approximate-pair counts in every SOL timing
case; an entirely dense run is not evidence of SOL acceleration. Final MP4
SSIM/LPIPS, audio differences, contact sheets and artifact notes are diagnostic
at two steps. Preserve failures rather than relaxing limits after seeing them.

## Eight-hour test budget

The timer starts before remote preparation/build verification and never resets
on failure, fix or reconnect. One GPU job at a time; CPU reporting/downloads
can overlap but the budget does not depend on overlap. Use the same frozen
binary/model/inputs/settings in each dense/SOL pair. Probe prerequisites first;
no model download, LoRA folding, large dependency rebuild or open-ended tuning
is hidden outside the budget. The node is expected to retain its downloaded
model and installed build dependencies.

| Bucket | Maximum minutes | Contents |
| --- | ---: | --- |
| A — Environment, build and host contracts | 25 | Probe, sync source/tests, SOL build, baseline build/CLI compatibility, host layout/policy/checkpoint tests |
| B — Synthetic kernels and sanitizer | 35 | Oracle, tails, layouts, routes, invalid values, forced-exact path, overflow/workspace planning, tiny memcheck/racecheck |
| C — Real QKV and bounded calibration | 35 | One calibration 640×480/243 two-step render, selected captures, attention replays and teacher-forced velocity check; freeze candidate before held-out matrix |
| D — Six matched render pairs | 255 | Twelve complete two-step renders listed below; includes startup, reference preparation, full VAE, audio and per-output media validation |
| E — State, continuation and lifetime | 50 | Matched continuation, split resume, context/cache changes, streaming/low-memory failure and release checks |
| F — Projection compatibility | 15 | FP8/NVFP4 BF16-interface operator checks and one tiny two-step NVFP4 dense/SOL pair; packing/startup included |
| G — Final artifacts and report | 30 | Remaining media diagnostics, download/checksums, galleries, bug ledger, coverage/task reconciliation |
| Contingency | 35 | Only a failed-test repair/retest or measured runtime overrun; no added matrix dimensions |
| **Total ceiling** | **480** | **445 scheduled + 35 contingency** |

Buckets are accounting limits, not a requirement to leave an idle gap. The
calibration run is additional to the twelve held-out renders and charged to C.
Capture at most two blocks at each of the two evaluations; keep raw traces
bounded, retain hashes and metrics, and use the existing diagnostic-trace
cleanup policy after successful validation. Do not delete model or adapter data.
The finite calibration set is `min_exact={0.9,0.75,0.5}` with other settings
fixed; select the fastest screen-passing candidate and freeze it. If none passes,
retain the implementation as unqualified, report the failure and do not label
an unrun performance matrix qualified. No extra parameter sweep is authorized
by this plan.

### D: complete-video matrix

Every row means **two fresh-process renders**, `default` and the frozen SOL
candidate, with identical references, prompt, seed, two-step schedule, BF16
weights, full VAE, all 50 layers, `reuse=core_reuse=1`, and fast CUDA off.
Full decoding/audio muxing is required; no early-stop output substitutes.
Use existing frozen reference assets, including the three two-second clips.

| ID | Geometry / frames | References and sizing | Pair cap, minutes |
| --- | --- | --- | ---: |
| R1 | 640×480 / 243 | None; primary whole-step/full-process comparison | 20 |
| R2 | 640×480 / 243 | Nine images, `match` | 25 |
| R3 | 640×480 / 243 | Same nine images, `max` | 35 |
| R4 | 640×480 / 158 | Nine images + three two-second videos, `max`; six seconds supplied, output covers them | 40 |
| R5 | 1344×768 / 243 | None; high-resolution attention scaling | 50 |
| R6 | 1344×768 / 362 | Nine images, `max`; large-sequence/workspace stress | 85 |
| **D total** | | **12 renders, each two steps** | **255** |

Dense controls here contain the **same references** as their SOL candidates;
the experiment isolates attention mode, unlike the previous reference-overhead
campaign. Earlier timing records guide estimates only, never replace current
matched controls. Preserve 243 frames despite exceeding ten seconds slightly;
the preceding multi-reference campaign's ten-second cap is not this gate.

The [previous node measurements](multi-reference-results.md) give conservative
planning anchors: 1344×768/362 with nine `max` images took 23m09s at one step,
and the high-resolution one-step control spent 589.06 seconds denoising
([retained log](../../outputs/multi-reference-cuda/R1344-SAMPLE-B00-F362-S1/attempt-1/stderr.log)).
Budget R6 for **two** two-step runs without assuming any SOL speedup. The
85-minute cap allows roughly 42.5 minutes per run. These are estimates to
validate at execution, not promised future timings.

### B/C/E/F: bounded coverage outside the render matrix

- **B, 35 minutes:** short lengths `1,31,32,33,63,64,65,127,128,129,257,1025`;
  both output layouts; query blocks 32/64; full-exact, mixed, protected and
  constant-key cases. Include extreme finite logits, zero variance, empty
  approximate branches, key tails, aliases, invalid input, partial allocation
  failure and checked-size overflow. Use tiny representative cases for
  sanitizer runs, not the entire production corpus. The independent oracle
  must not call the CUDA router under test to generate its expected masks.
- **C, 35 minutes:** capture actual packed geometry, real QKV and dense velocity
  at the routed evaluation. Compare recipe output with a blocked FP32 oracle;
  compare route masks away from threshold-rounding ambiguity, and inspect
  ambiguous decisions explicitly. Recompute held-out checks from R1 captures
  without another generated render. Never materialize a production N×N matrix.
- **E, 50 minutes:** allocate 20 minutes to one 640×480/243 continuation pair
  with a 39-frame context from the same saved dense parent, preserving prefix
  bytes. Validate delivered frame/audio counts after the configured prefix
  trimming, rather than assuming the full target frame count is delivered.
  Allocate eight minutes to a tiny uninterrupted two-step SOL render
  versus fresh-process stop-after-one/resume-to-two (three processes, four
  total denoising evaluations, never over two in one render). Allocate 15
  minutes to fast/default SOL context isolation, policy/layout changes,
  repeated short operator dispatches, streamed weights and cache teardown.
  Use seven minutes for failure injection, old/new checkpoint fixtures,
  decode-only validation and fully protected still-layout GPU checks. Cover
  all supported continuation context sizes with host metadata tests in A.
- **F, 15 minutes:** check both projection-quantization interfaces, then one
  128×128/22-frame NVFP4 dense/SOL pair with the same quantized weights. Count
  any cache preparation against this bucket and contingency. Report this as
  composition smoke coverage, not production quantized-SOL qualification.

Save completed AV state from R1's dense run for continuation; do not spend a
separate long render creating a parent. Keep main timing runs unfenced; obtain
kernel breakdowns from C's replays. Freeze CUDA GEMM tuning and all relevant
environment variables identically within each pair. Reuse existing attention,
resume and multi-reference helpers where appropriate, but do not run their
unbounded old corpus commands or overwrite their result directories.

### Deadline enforcement and failure handling

Implement a new `outputs/cuda-sol/` manifest/runner with a monotonic global
deadline and per-bucket/per-case caps. **All GPU work must stop by T+450 min**;
reserve the final 30 minutes for G. Each child timeout is bounded by its case
allowance and the remaining time to T+450, not T+480. Terminate only owned process groups,
collect partial logs/telemetry, verify GPU release and mark the exact failure.
Admit a pair only if its remaining conservative allowance fits; do not generate
an unmatched candidate and infer a control later. Normal execution must cover
the complete listed matrix; budget-aborted coverage is explicitly incomplete.

Retain at least 4 GiB GPU safety headroom, the existing host/RSS limits and
cgroup-aware reclaimable-cache accounting. Inventory disk before capture and
quant-cache work. A missing node/dependency, severe slowdown, OOM, numerical
failure or exhausted retest budget stops the affected branch; no silent preset,
resolution, frame-count, decoder or model changes. Use at most one bounded
repair/retest from contingency and record both attempts. If a kernel change
invalidates completed evidence, mark it stale instead of blending binaries.

## Deliverables and completion

Produce per-test JSON/CSV with exact command/environment and source/binary/model/
asset hashes; startup, each denoising step, aggregate denoising, decode and total
wall time; sampled device/process VRAM, host memory, telemetry gaps, workspace
and routing counts. Show dense/SOL speedups only for valid matched pairs.
Record effective dense bypasses and any reference/continuation protections.

Create index and resolution-specific HTML playback pages with synchronized
dense/SOL videos, contact sheets, frame/duration/audio checks and links to logs.
Download playback assets and test records and verify a frozen checksum manifest.
Keep confirmed bugs, quality concerns, environment/timeouts, failed gates and
untested cases separate. Do not reopen the previous multi-reference defect
scope or fix unrelated renderer issues as part of this attention milestone.

Implementation completion requires CLI/API/kernel/state/memory contracts and
their bounded tests. Quality/performance outcomes may be pass or
fail, but must be measured and reported; they do not automatically authorize a
default change. Claim full test-plan completion only when every mandatory test
ran to a recorded disposition, with no stale or unexplained missing evidence.

## Supporting sources

The local [CUDA attention design](design-attention.md),
[existing CUDA policies](attention.md), [Metal SOL source audit](../metal/m1-attention-sources.md)
and [Metal SOL measurements](../metal/m4-sol-243-results.md) define the starting
point and retained limitations. VPIPE is historical design context through that
source audit; this proposal does not assume an upstream CUDA SOL implementation.

The existing kernel uses tiled online-softmax execution in the
[FlashAttention-2](https://arxiv.org/abs/2307.08691) family. NVIDIA's
[PTX matrix-instruction documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)
documents BF16 matrix operands with FP32 accumulators. These sources support
the kernel approach, not a predicted H3 speedup or quality guarantee.
