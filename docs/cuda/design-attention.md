> Historical renderer/qualification record. Commands that refer to removed
> fast/legacy modes require the archived source. Use the
> [current single-pipeline design](design-single-pipeline.md) and
> [M6/M7 qualification](single-pipeline-final.md) for supported execution.

# CUDA SageAttention acceleration

Status: implemented, qualification in progress; checklist in
[../todo.md](../todo.md), evidence in [attention-validation.md](attention-validation.md).
Design date: September 19, 2026. Qualification target: the same RTX 5090
(SM120, 32 GB) node used for CUDA quantization and runtime LoRA validation.

## Objective and scope

Accelerate the two attention matrix products in the 50 main H3 DiT blocks
using native CUDA implementations of **SageAttention 2++** and
**SageAttention 3 NVFP4**. Start with 2++, then add the more aggressive FP4
route behind a separate explicit choice. Preserve the existing default.
This implements the attention experiment proposed in
[the quantization review](denoiser-quantization-review.md#long-sequences-and-expected-benefit).

This work covers CUDA FL2VA and Ref2VA, including their joint video/audio/text
sequence and reference/continuation conditioning. Metal optimization, other
GPU qualification, sparse/windowed attention, training, new samplers and
distillation are outside scope. Qwen causal GQA, text refiners, reference
encoders and VAE attention retain their current implementations.

Attention quantizes **activations**, independently of
`--cuda-denoise-quant off|fp8|nvfp4`, which selects projection arithmetic.
Either new attention backend must work with each projection mode. LoRA still
folds into BF16 weights before optional weight packing; neither attention
choice changes the folded-model identity or its disk cache. Activation packs
are transient and must be recomputed for every actual block evaluation.

## Evidence and performance baseline

The [runtime LoRA validation](../lora/runtime-validation.md) records a
1344×768, 362-frame, eight-step Turbo FL2VA run with fast CUDA, NVFP4
projections and preview VAE:

| Measurement | Recorded value |
| --- | ---: |
| Complete process | 812.45 s |
| Reported cumulative DiT wall time | 717.763 s |
| Profiled attention | 615.814 s |
| Profiled GEMM | 42.035 s |
| Reported peak GPU allocations | 28.790 GiB |
| Main attention shape | B=1, S=109,078, H=KV=56, D=128 |
| Main attention dispatches | 400 cuDNN calls |

The retained log is
`outputs/runtime-lora/cuda/workflows/1344x768-turbo8-nvfp4-preview.log`.
The two additional attention calls are short text-refiner calls. Roughly
85.8% of that reported DiT wall counter was attributed to attention. A
hypothetical 2× reduction of the attention category would reduce that
interval to about 410 s, before changed memory/streaming costs. The older
counter includes model loading; implementation qualification adds explicit
CLI phase boundaries to measure denoising separately. This is an estimate,
not a SageAttention result.

That development run predates final model-fingerprint hardening and uses
FL2VA; `testf1.sh` uses Ref2VA and three continuation segments. Remeasure on
the implementation build and qualify both modes. Keep startup, model/LoRA
hashing, cache preparation, denoising and decoding as separate measurements.
The primary comparison is against the current **cuDNN** path with identical
settings, not just the slower native fallback.

## Upstream implementation choice

Review upstream source at implementation time and pin exact SageAttention
and CUTLASS revisions before importing code. The links below identify the
reviewed entry points, not a promise to follow future `main` changes.

| Choice | Selected algorithm | Initial support |
| --- | --- | --- |
| `default` | Existing dispatch, including cuDNN under fast CUDA | Unchanged |
| `sage2++` | INT8 QK, FP8 PV, FP16 MMA partials with FP32 buffering | RTX 5090 / SM120 |
| `sage3` | NVFP4 QK and PV with smoothing and compensated scaling | RTX 5090 / SM120 |

Upstream identifies `sageattn_qk_int8_pv_fp8_cuda` with
`pv_accum_dtype="fp32+fp16"` as 2++. Use that explicit entry point for the
oracle instead of the device-dependent `sageattn` selector. Published kernel
throughput excludes preprocessing, which must be included here.
[Official API and benchmark notes](https://github.com/thu-ml/SageAttention/blob/main/README.md).

The inspected 2++ implementation defaults to per-thread Q/K quantization,
K smoothing, no V smoothing and a V scaling limit of 2.25 for this
accumulation path. Its per-thread quantizer is Triton; port its exact scale
grouping and rounding to CUDA. Do not silently substitute the per-warp route
or a different accumulator and continue calling it the same recipe.
[Upstream dispatch](https://github.com/thu-ml/SageAttention/blob/main/sageattention/core.py).
V preprocessing also transposes, pads and permutes values; avoid carrying
its extra full BF16 temporary into the production path.
[Upstream quantization](https://github.com/thu-ml/SageAttention/blob/main/sageattention/quant.py).

SageAttention 3 uses E2M1 values, E4M3 scales for 16-value groups, and
two-level scaling of the softmax probabilities. It smooths K and blockwise Q;
the Q-mean contribution must be restored to the scores. Preserve these
operations, their layout permutations and online normalization as one
versioned algorithm.
[SageAttention 3 paper](https://arxiv.org/html/2505.11594v3).

The upstream wrapper contains PyTorch/Triton preprocessing and materializes
`delta_s = Q_mean @ K_transpose` in FP32. It also mutates K and uses its own
default attention scale. The native adapter must preserve h3cli input
ownership, pass the h3cli scale explicitly, and bound correction storage.
[SageAttention 3 wrapper](https://github.com/thu-ml/SageAttention/blob/main/sageattention3_blackwell/sageattn3/api.py).

Upstream does not guarantee acceptable quality on every video model and
suggests selective 2++ use for sensitive layers/timesteps. H3 therefore needs
its own acceptance evidence. A mixed policy is a conditional experiment
described below, not an implicit fallback.
[SageAttention 3 limitations](https://github.com/thu-ml/SageAttention/blob/main/sageattention3_blackwell/README.md).

## Build and native integration

Proposed build switch: `CUDA_SAGE=1`. A CUDA build without it must continue
to build and run normally; an explicit Sage request reports unavailable
support before loading weights or creating caches. Native inference must not
link libtorch, invoke Python/Triton, use a subprocess, or JIT through Python.
An isolated upstream Python environment is permitted only for validation.

Keep adapted kernels under `third_party/sageattention/`, with original
notices, exact source revisions, checksums and a short patch inventory.
Record the [upstream Apache-2.0 license](https://github.com/thu-ml/SageAttention/blob/main/LICENSE)
and notices for every imported dependency. Pin CUTLASS explicitly; no
implicit network downloads during `make` or inference.

Use CUDA 12.8 or later as the initial toolchain floor, then verify the exact
selected revisions on the node. SageAttention 3's build selects `sm_120a`
for compute capability 12.0 and uses CUTLASS plus the CUDA driver API.
Compile its specialized translation unit separately from generic CUDA code;
`sm_100a` and Hopper kernels are not substitutes. Preserve ordinary
`CUDA_ARCH` and fat-build behavior with guarded unavailable-backend stubs.
[Upstream build configuration](https://github.com/thu-ml/SageAttention/blob/main/sageattention3_blackwell/setup.py).

Suggested integration boundaries:

| File or area | Responsibility |
| --- | --- |
| New `src/denoise/attention.h` / `src/denoise/attention.c` | Stable policy IDs, recipe versions, parsing and host validation |
| `src/h3.h`, `src/h3cli.c`, `src/engine.c` | Public request option, CLI, prepared-context identity, resume and provenance |
| `src/execution.*`, `src/denoise/dit.c` | Capture request policy and identify main-block attention explicitly |
| `src/gpu.h`, `src/cuda/gpu_cuda.cu` | Typed DiT dispatch, profiling, resource admission and stream ownership |
| New `src/cuda/cuda_sage.h` / `.cu` and specialized units | Native pointer/stride API, quantizers, kernel launches and workspace planning |
| `src/sampler_state.*`, `src/sampling/sampler_file.c` | Persist arithmetic policy and required recipe identity |
| `src/media/presentation.c` and delivery consumers | Completed-state denoising provenance |
| `Makefile`, new `tests/attention_*` | Optional native build, correctness, oracle and workflow qualification |

Names are proposed, not existing interfaces. Adapt the upstream tensor
bindings to raw device pointers, explicit strides, an explicit CUDA stream
and caller-owned buffers. Preserve upstream launch parameters and scale
layouts; remove framework allocation and default-stream assumptions.
[SageAttention 3 native binding](https://github.com/thu-ml/SageAttention/blob/main/sageattention3_blackwell/sageattn3/blackwell/api.cu).

## Request and dispatch contract

Proposed CLI:

```sh
--cuda-attention default|sage2++|sage3
```

Add a corresponding zero-default API field. The choice is independent of
`--fast-cuda`, projection precision and VAE selection. For example, the
target workload will combine `--fast-cuda --cuda-denoise-quant nvfp4
--cuda-attention sage2++ --preview-vae`. Quoting `sage2++` is optional.

`default` exactly preserves today's attention routing. Non-default requests
are CUDA-only and initially require SM120 and a Sage-enabled build. Validate
invalid names, unsupported devices/builds and contradictory reference-mode
overrides before expensive preparation. Existing `H3_FAST_CUDA_ATTENTION`
overrides other than unset/`auto` conflict with an explicit Sage selection;
reject that combination rather than silently choosing one.

Dispatch only from the main-block call in `run_block()` in `src/denoise/dit.c`, after
QKV projection, Q/K normalization and RoPE. Add a typed operation/domain
descriptor rather than inferring scope from the log label, head count or
datatype: the text refiner also has 56 heads. Carry block index and absolute
evaluation/step position for profiling and any later qualified mixed recipe.
Contexts capture policy at construction; nested request scopes restore their
previous state on every exit. Do not switch an existing context through
mutable environment variables.

The initial supported operation is dense, noncausal BF16 self-attention with
B=1, H=KV=56, D=128, arbitrary valid sequence tails and no dropout. Inputs
are sequence-major `[B,S,H,D]`; the existing `head_major` argument controls
the **output** layout, not Q/K/V layout. Return either sequence-major or
head-major BF16 as requested by the following projection. Ref2VA image and
video reference tokens remain in the original joint sequence. Continuation
prefixes must not accidentally become a causal mask.

Preserve h3cli's existing BF16 rounding of the noncausal attention scale
before dispatch. Do not let an upstream wrapper recompute `1/sqrt(D)` or
apply the scale twice. Preserve row maps, RoPE, token-reduction permutations
and actual sequence lengths; GEMM padding is not an attention token.
Mask padded K positions out of softmax, ignore padded outputs, and implement
the pinned algorithm's Q-tail smoothing convention explicitly.

Unrelated attention domains use their existing paths. An explicit Sage
request must execute Sage on eligible main blocks: unsupported main-block
shapes, unavailable workspace or initialization failure produce a clear
error. There is no silent run-wide substitution with cuDNN or another Sage
version. Query/head chunking within the same recipe is allowed only after
its numerical behavior is qualified and its plan identity is recorded.
After a CUDA launch failure, abort through the normal backend error path;
do not retry another kernel on a potentially failed stream.

## Arithmetic, buffers and memory admission

Implement tiled online attention without materializing an S×S score or
probability matrix. FP32 softmax statistics and the recipe's higher-precision
accumulation/corrections remain intact. Attention scales and packed values
are not interchangeable with `h3_quant` projection formats, even when both
are named NVFP4. Keep separate descriptors and ownership.

The large case leaves little device headroom: one BF16 Q/K/V-shaped tensor
is about 1.456 GiB. An extra full head-major copy of all three is about
4.37 GiB. Admission must account for original live tensors, output,
quantized values, scale arrays, padding, reductions, correction tiles and
kernel scratch alongside the existing resident/streamed weight policy.
Include all allocations in the reported peak and reserve the selected
workspace before assigning the optional weight cache its capacity.

Use a context-owned, reusable workspace with a conservative initial
**512 MiB additional attention workspace cap**, including every adapter
temporary. Plan bounded groups of heads and query tiles under that cap;
choose sizes from the qualified deterministic plan table. Each query group
must still attend to **all** valid keys. Head grouping is valid because
heads are independent; splitting K requires the original global online
softmax reduction, never independently normalized chunks. If the cap cannot
support a shape, fail admission with required/available byte counts.

For 2++, compute reductions over the full valid sequence for the current
head group, then pack directly from the existing strided BF16 inputs.
Fuse transpose/padding/permutation with quantization where that avoids a
second full representation. Do not overwrite Q/K/V to reclaim memory unless
a future explicit ownership contract permits it. Quantization and output
conversion must be included in operation timing.

For SageAttention 3, the upstream full correction array at S padded to
109,184 requires
`56 * ceil(S/128) * padded(S) * sizeof(float)`, approximately **19.43 GiB**.
This cannot coexist with the recorded baseline. Generate `delta_s` for only
the admitted head/query group, consume it, then reuse the storage. Extend
the launch descriptor with query offsets/strides as needed so local buffers
preserve global Q-block identity and K indexing. Initially use a bounded
correction GEMM to preserve the selected upstream arithmetic; a fused
correction is a separately tested optimization. Do not remove Q smoothing
or change to whole-sequence Q means merely to fit memory.

Preserve numerical behavior across correction chunk boundaries, including
nonmultiple-of-128 tails. Carry the actual H3 scale into the FP4 kernel.
Port preprocessing without the wrapper's in-place K mutation. All transient
memory follows the compute stream's lifetime, supports cancellation and is
released before decoder admission. Reuse allocations and launch plans, not
activation contents. No activation cache is written to disk.

## State, cache identity and reproducibility

Give each backend a stable enum value and a versioned arithmetic recipe.
The recipe covers smoothing, quantizer layout/scales/rounding, accumulation,
tail handling and any mixed schedule. Prepared DiT identity includes the
attention choice and recipe; otherwise a cached default DiT could service a
Sage request. Preserve byte-identical legacy default keys where possible.
Do not invalidate unchanged LoRA folds or packed projection weights.

Reserve the next unused required sampler extension (currently 35, following
fast-CUDA 33 and projection-quantization 34) for non-default attention.
Store policy, recipe and the effective arithmetic plan identity needed for
resume. Old checkpoints lacking it mean `default`. Older readers must
reject the unknown required extension, and new readers reject malformed or
unsupported versions. Restore the saved policy on ordinary resume; reject
conflicting explicit selections before allocating the model. No attention
backend conversion during exact resume is added in this work.

Require uninterrupted versus pause/resume AV-latent equality on the same
qualified build/device and execution plan. Do not claim equality between
Sage and cuDNN or between Sage versions. If workspace/plan changes alter
arithmetic, exact resume must require the saved plan or fail clearly rather
than silently recompute a different one.

Extend completed-state presentation provenance with a backward-compatible
new version for attention policy/recipe, including attention-only requests
with projection quantization off. Preserve reading existing v1/v2 sidecars.
The AV latent format remains unchanged. Decode replay needs neither Sage
kernels nor model weights; attention provenance must not block compatible
Metal/CPU decoding. A new continuation segment may explicitly use a different
attention mode, records its own provenance, and is distinct from resuming
an interrupted denoising trajectory.

## Validation on the existing RTX 5090 node

Use the already provisioned node and its existing model, adapter and preview
VAE assets. Start an isolated attention checkout/build/output directory;
reuse compatible native LoRA and projection caches. Do not create another
full folded model for each attention mode. Record free disk/RAM/VRAM first
and retain the existing private-cache filesystem requirements.

Record commit and binary hashes, source/dependency pins, compiler flags,
GPU name/SM, driver, CUDA, cuDNN/frontend, clocks/power/temperature, PCIe
status, memory limits and all relevant environment overrides. The historical
[5090 qualification](cuda-5090-qualification.md) describes the node; recapture
the actual environment for this work. Do not change shared system packages
to make the upstream oracle install: use a separate environment.

### Correctness gates

1. Host tests cover parsing, capability checks, request isolation, legacy
   defaults, prepared identities, checkpoint/provenance compatibility and
   failures before expensive work. A non-Sage CUDA build and non-CUDA build
   retain their existing behavior; non-default options reject clearly.
2. Native tests cover zero/constant tensors, outliers, tiny values,
   quantization ties/saturation, non-finite input policy, scale direction,
   layouts, padding and guard regions. Probe S around 16/64/128 boundaries,
   observed short shapes and production S=109,078. Use the real H/D values.
   Run compute-sanitizer on bounded representative cases.
3. Compare each native port against its pinned upstream recipe on identical
   immutable inputs and explicit h3cli scale. Separately compare to independent
   FP32 SDPA and current cuDNN. Clone oracle inputs before preprocessing;
   for Sage3, use a minimal recorded harness around the low-level binding
   to supply the scale that its public wrapper otherwise recomputes. Keep
   the upstream quantizers and attention kernel unchanged in this oracle.
   For large S, evaluate selected query rows
   against the entire K/V sequence with a stable blocked reference; do not
   allocate a dense production score matrix. Check intermediate scale/pack
   buffers and correction tiles to distinguish port bugs from quantization.
4. Capture bounded real H3 Q/K/V samples across all 50 blocks and early,
   middle and late evaluations, including separate video/audio/text/reference
   query groups. Report cosine similarity, RMSE/relative error, maximum
   error and non-finites by block, timestep and modality. Freeze numeric
   acceptance limits using a calibration subset before the held-out run;
   keep port-parity limits distinct from approximate-model quality limits.
5. Verify full-core outputs, final AV latents, exact same-mode resume,
   cancellation/cleanup, repeated contexts and allocation-failure behavior.
   Exercise token reduction, core/reuse schedules and continuation without
   accidentally reusing activation packs after inputs change.

### Workload matrix and quality

Use fixed, ordinary scene/dialogue prompts and existing reference fixtures.
Hold prompt, reference bytes, seed, frame count, sampler schedule, Turbo
strength, weight precision, cache state and VAE fixed within every pair.
Record a content hash for each model/adapter/reference and the full command.

| Tier | Cases | Purpose |
| --- | --- | --- |
| Component | Tail lengths, 2,281, 18,225 and 109,078 tokens; actual Ref2VA lengths | Layout, precision, preprocessing cost and bounded peak memory |
| Short model | 288×384/56 frames and 480×640/22 frames; FL2VA and Ref2VA; multiple seeds | Baseline/Turbo quality, projection-mode interactions |
| Target | 1344×768/362 frames, Turbo 1.0, 8 steps, fast CUDA, preview VAE | Primary attention and denoising speed/quality |
| State/workflow | Pause/resume; Ref2VA continuation with 39-frame context; multi-segment run | Saved state, references, audio/video seams and provenance |

At short sizes cross all three attention modes with projection
`off/fp8/nvfp4`; include base-model and Turbo cases. At target size require
paired projection-off and NVFP4 runs for each attention mode in FL2VA and
Ref2VA. Match the geometry and continuation structure of `testf1.sh` through
a parameterized qualification driver using the fixed validation prompts.
Do not modify the user's script to conduct benchmarks. Keep preview VAE
fixed for primary comparisons, and perform matched full-VAE decoding of
selected saved states to separate denoiser artifacts from preview artifacts.

Compare video identity, fine detail, motion, flicker and prompt adherence;
listen for speech/singing clarity, voice identity, distortion and timing;
inspect lip sync and continuation seams. H3's audio tokens share attention,
so image/video-only metrics are insufficient. Preserve paired outputs for
human review and record review status separately from numerical results.
Passing a kernel test does not establish perceptual equivalence.

For this implementation, the user has waived human-review checks and accepted
the current implementation for delivery. Provide final HTML playback pages
without requiring verdicts or notes; record human review as `skipped-by-user`
and retain the measured automated results.

The user also skipped remaining target continuation checks. Retain the
completed short continuation matrix and target pause/resume evidence;
record the unfinished target continuation cases as `skipped-by-user`,
without claiming target multi-segment qualification.

At final closeout the user stopped all remaining tests. Full-render profiling
overhead, decoder replay/full-VAE comparisons, extended audio cases and the
mixed-schedule experiment are recorded as skipped. Deliver the completed
preview and short-continuation galleries with their retained evidence;
do not infer passes from the skipped work.

### Timing and release gates

Expose per-backend call counts and separate times for smoothing/reductions,
packing/layout, correction generation, attention kernel and output conversion.
Also report inclusive attention time, synchronized denoising wall time,
streamed weight bytes, workspace and total device peak. Check counters prove
the requested backend ran on every eligible evaluation. Profiling overhead
must be measured or excluded from the final wall-time pair.

Use warmups and at least three measured component repetitions. Alternate
backend order and repeat target pairs at least twice, adding a third if
results are noisy. Distinguish cold startup/preparation from warm cache
operation. Never compare different seeds, shorter sequences or skipped
evaluations as an attention speedup. Keep a job ledger with per-case timeout
and actual cost; target-run timeouts must exceed the recorded 13-minute
baseline rather than inherit short-probe caps.

Success requires safe memory use under the real 32 GB limit, native runtime
independence, state compatibility, numeric/quality acceptance and repeatable
inclusive attention **and denoising** improvement over matched cuDNN.
Record exact ratios and dispersion; no published speed multiplier is a
guarantee. Recommend a mode only for the tested workloads. Both modes remain
opt-in, and SageAttention 3 may remain opt-in if it fails quality or
does not improve on 2++ after preprocessing.

If pure SageAttention 3 fails quality, run a bounded sensitivity study using
2++ at first/last evaluations and, if needed, selected blocks. Freeze any
successful table as a distinct `sage3-mixed` policy with its own recipe and
state identity, then repeat held-out and performance validation. Never change
the meaning of `sage3` or silently select layers from runtime heuristics.
If no mixed policy passes, publish the failure and retain 2++ as the qualified
alternative; do not relax thresholds after inspecting held-out results.

Deliver `docs/cuda/attention.md` usage documentation, an
`attention-validation.md` report and machine-readable acceptance results
alongside reproducible test drivers. Record failed/untested cases explicitly.
The fresh task list tracks implementation and qualification separately.
