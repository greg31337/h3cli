# Faster original-model H3 VAE execution

> Historical record: `--full-vae-execution` and its alternate implementations
> have been removed. See the [current original-VAE guide](fast-vae.md) for
> supported execution and validation. Results below describe the earlier code.

Status: **CUDA M1–M6 complete and qualified on 2026-09-23. Metal M7 implemented
and tested within its separate four-hour budget.** Metal quality and memory
checks pass; performance qualification is partial: 1.130× / 1.126× C1/C2 decode
speed and 2.056% matched C1 wall savings miss the 1.5× / 5% targets. The
original default is retained. See [Metal results](fast-vae-metal-results.md).
The original research/contract below is retained. See the
[measured results](fast-vae-results.md), [implementation guide](fast-vae.md), [CUDA playback report](../outputs/fast-vae/cuda/review.html)
and [implementation checklist](todo.md). Updated after the VPIPE source
audit and the user's clarification: **exact full quality is not required;
the new mode should be meaningfully better than the current preview VAE.**

## Objective and decision

Reduce the time from completed H3 latents to a finished video, especially for
few-step renders where original-model VAE decoding is a large share of wall
time. Implement and qualify **CUDA first, then Metal**. Keep the original VAE
architecture and checkpoint, allow BF16/FP16 execution where stable, and retain
FP32 for sensitive operations. Preserve reconstruction geometry. Optimize
encoding second, under a separate conditioning-compatibility gate.

The existing [preview VAE](preview/preview-vae.md) is separate: `--preview-vae`
continues to select approximate TAEH3, with its current weights, behavior and
live-preview support. This project accelerates the original decoder used when
that option is absent, including original-model finalization of saved `.h3av`
latents. It does not replace or retune TAEH3. The three intended tiers are:

| Tier | Decoder | Intended use |
| --- | --- | --- |
| Reference | Current original VAE, FP32 | Numerical oracle and maximum reconstruction fidelity |
| Balanced (new) | Original VAE with qualified BF16/FP16 compute and FP32 sensitive operations | Faster rendering with clearly better reconstruction than tiny preview |
| Preview (existing) | TAEH3 selected by `--preview-vae` | Fastest rough iteration, unchanged |

The recommended sequence is:

1. Isolate actual video-decoder costs and establish a repeatable FP32 oracle.
2. Implement a VPIPE-inspired BF16 decoder on CUDA: Tensor Core QKV/MLP
   GEMMs, dense attention and reusable buffers, with FP32 accumulations.
3. Compare a bounded mixed-precision TensorRT prototype and range-safe FP16
   candidate early; select by reconstruction quality and complete-tile time.
4. Measure fixed-tile graph replay and small tile batches; retain only wins.
5. Reduce output readback, unpacking and stitching overhead with bounded memory.
6. Qualify full decoding and reference encoding on CUDA, then transfer the
   useful scheduling/layout work to Metal and measure its own best kernels.

This recommendation is an inference from our implementation and the research
below. Published speedups from other models or devices are not performance
predictions for this implementation.

## Implemented CUDA choices

The implementation starts from repository revision
`2abc877cd65e8b2977772b38731480d26c4f8274`. Native balanced execution uses
BF16 operands with FP32 accumulation, norms, residuals and softmax. VAE dense
attention uses bounded cuBLAS BF16 score/value GEMMs, independently of DiT
Sage/SOL. The native FP16 experiment remains diagnostic; the separately typed
FP16 TensorRT candidate passed the initial quality and complete-tile speed gate
and therefore has an optional C++ adapter and offline engine builder.

GPU unpacking and bounded pinned delivery feed the existing spatial/temporal
stitcher. Balanced/TensorRT generation now also uses the incremental FFmpeg
sink; the float-output API remains materialized when requested. Batch 1 stays
selected after measuring independent contexts/streams at batches 1/2/4.
Capture is opt-in: cuBLASLt rejected CUDA Graph capture on the measured node,
so its clean ordinary-execution fallback is tested without claiming a graph
speedup. Reference encoding remains FP32, with reused im2col scratch and
prepared convolution descriptors/algorithms.

The qualification node is an RTX PRO 6000 Blackwell Server Edition (SM120),
CUDA 12.8, driver 595.91.07; optional TensorRT is 10.13.3.9. This is not a 5090
measurement. The 243-frame comparisons pass the frozen quality and six-step wall-time gates
at both resolutions, with 362-frame decode-only stress and resource-lifetime
checks. Native balanced decoding is about 2.3× faster than legacy fast CUDA,
saving 19.6% / 12.8% complete wall time at C1/C2. Ref2VA identity checks use
fixed GEMM selection to isolate existing timing-based autotuning variability.
The original TAEH3 path is unchanged. Metal uses an independent BF16 MPSGraph
linear/fused-MLP recipe with Steel D=64 FP16-operand attention and FP32
softmax/accumulation. Its separate M4 campaign passes all six quality cases and
159 closing correctness/media checks, with 362-frame decode stress and stable
repeated-decode memory. The bounded ANE row split loses after packing
and is not integrated. CUDA qualification does not transfer to Metal; the
Metal 1.5× decode and 5% whole-render speed targets remain unmet.

## Research: what is available and what actually applies

Sources were inspected on 2026-09-23. These are author-reported results or
proposals, not independently reproduced measurements in this repository.

| Method / implementation | Evidence and limitation | Decision for h3cli |
| --- | --- | --- |
| H3 chunked output/input in ComfyUI | Merged implementation reports bitwise-identical output pixels and nearly unchanged speed, with substantially lower transient memory. Its 175-frame example reduces decoder transient allocation from 3,485 to 607 MB. | Strong precedent for bounded delivery and avoiding duplicate full-video buffers. A memory improvement is not automatically a speedup. [Merged PR](https://github.com/Comfy-Org/ComfyUI/pull/15446) |
| H3 TensorRT encoder/decoder | Runnable integration claims up to 1.7×. Its unquantized compiler enables FP16, and the runtime uses FP16 inputs; a W4A16 variant is also offered. | Now a direct candidate for the balanced tier, with original weights, range checks and three-way quality comparison. No low-bit decoder in this plan. [Project](https://github.com/lihaoyun6/ComfyUI-H3VAE_TRT), [compiler](https://github.com/lihaoyun6/ComfyUI-H3VAE_TRT/blob/main/compile.py), [runtime](https://github.com/lihaoyun6/ComfyUI-H3VAE_TRT/blob/main/minimax_trt_node.py) |
| H3 batched spatial tiles | Runnable batched decoder preserves temporal chunking. Its own same-setting comparison was **11.15 s batched versus 10.36 s standard**; fewer invocations did not offset their overhead. | Benchmark batches 1/2/4; do not presume batching is the main win. Batch independent tiles, never their attention contexts. [Implementation and measurements](https://github.com/Mozer/ComfyUI-MiniMax-H3-MotionCache-FastVAE) |
| `fastvae` for Wan on SM120 | Reports 2.41× at 480p/81 frames on RTX PRO 6000 Blackwell. Uses fused normalization/activation, channels-last layouts, deferred residual/bias operations and causal-convolution caches. Supports Wan, not H3; reported maximum output difference is 0.04 in [-1,1]. | Useful kernel/layout ideas, especially for the convolutional encoder. Its causal decoder is not a drop-in replacement for H3. [Code and benchmark](https://github.com/Occipital-Labs/fastvae) |
| CUDA Graphs, compiler fusion and reusable execution plans | Established inference mechanisms; NVIDIA recommends measuring complete execution and accounting for capture, copies and batch size. | Apply to stable H3 tile shapes after allocation and plan creation are separated from execution. [TensorRT performance guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html) |
| cuBLAS BF16x9 FP32 emulation | Current support table lists compute capabilities **10.0 and 10.3**, CUDA 12.9+. The separate FP64 emulation support includes 12.x. | Not an available FP32 acceleration path for our last measured **SM120** node. A toolkit upgrade alone does not make BF16x9 supported there. [cuBLAS support table](https://docs.nvidia.com/cuda/cublas/index.html#floating-point-emulation) |
| Flash-VAED | ICML 2026 work reports approximately 6× decoding speed using pruning, operator changes and distillation, retaining up to 96.9% of reconstruction performance on Wan/LTX. | Interesting quality/speed research, but no compatible H3 checkpoint established here. Training a new decoder remains outside this implementation; use the original H3 model first. [Paper](https://arxiv.org/abs/2602.19161), [implementation](https://github.com/Aoko955/Flash-VAED) |
| H3 production VAE optimization RFC | Proposes separate timings, stable-region compilation, stacked tiles and independent VAE execution resources. | Supports the experiment ordering, but is a proposal, not measured proof. Multi-GPU sharding is unnecessary for this single-node plan. [vLLM-Omni RFC](https://github.com/vllm-project/vllm-omni/issues/5948) |

The quality clarification removes the need for an all-FP32 speedup. The most
promising route is now the original H3 decoder running on fast 16-bit matrix
kernels, with selective FP32 operations. Qualify against both current decoders
on identical latents; neither a published speedup nor a precision label proves
that the new tier is better than TAEH3.

Do not import framework implementations wholesale. Pin source revisions and
record licenses/notices for any borrowed code. The native implementation should
remain small and independent of Python inference frameworks. A Python/ONNX
export tool is acceptable for an optional offline TensorRT experiment.

## What VPIPE actually does

Audited revision: `aad3a10e654af71ae46cb627e5f4f7da22ff0593`.
This is a source audit; VPIPE was not executed during this planning work.

| Area | Observed implementation | Consequence here |
| --- | --- | --- |
| Precision | `weight_()` preserves BF16 weights or converts other input types to BF16; scratch uses two-byte elements | Prototype BF16 original-model execution before demanding an FP32 kernel win |
| Lifecycle | Loads the decoder blocks once, reuses shape-sized scratch and submits a GPU-only tile as one command stream | Reuse this pattern; h3cli already has much of the resident-tile behavior |
| Output | Unpatchifies and stitches on the host, decodes tiles serially, then collects temporal pieces | Do not attribute its speed to GPU stitching, tile batching or constant-memory output streaming |
| Encoder | Causal convolution is expressed as bounded HWC im2col bands plus GEMMs | Consider the layout/scratch strategy after measuring our encoder |

These observations come from the [video VAE implementation](https://github.com/tgo-app-dev/vpipe/blob/aad3a10e654af71ae46cb627e5f4f7da22ff0593/generative-models/minimax-h3/metal-minimax-h3-video-vae.cc).

VPIPE's VAE uses a BF16 D=64 Steel attention specialization with FP32
accumulation: ordinary simdgroup kernels use 32×16 query/key tiles, while the
M5 NAX variant uses 64×32. Those are dense attention kernels, not SOL/Sage.
[Steel specialization](https://github.com/tgo-app-dev/vpipe/blob/aad3a10e654af71ae46cb627e5f4f7da22ff0593/gpu-kernels/metal/attention/attn_steel.metal),
[NAX specialization](https://github.com/tgo-app-dev/vpipe/blob/aad3a10e654af71ae46cb627e5f4f7da22ff0593/gpu-kernels/metal/attention/attn_steel_nax.metal).

Its dense GEMM library likewise supports BF16 operands and FP32 accumulation.
The `_f16` function suffix is a legacy name: selecting the `_bf16` library
changes the element type. Reading the function name alone would misidentify
the actual VAE precision. [GEMM source](https://github.com/tgo-app-dev/vpipe/blob/aad3a10e654af71ae46cb627e5f4f7da22ff0593/gpu-kernels/metal/gemm/dense_gemm.metal).

The VAE configuration describes feed-forward work as roughly **70% of decoder
arithmetic** and offers an optional, explicitly lossy GPU/ANE row split. This
strengthens the case for optimizing the MLP/GEMMs alongside attention. Treat
the share as an arithmetic estimate, not our measured wall-time breakdown.
ANE is a later M4 experiment, after the GPU path; M5 NAX remains capability
gated. [VAE configuration](https://github.com/tgo-app-dev/vpipe/blob/aad3a10e654af71ae46cb627e5f4f7da22ff0593/generative-models/minimax-h3/metal-minimax-h3-video-vae.h).

VPIPE does not assert FP32 equivalence: its decoder golden test permits relative
L2 below 0.02; its end-to-end tiled encoder/decoder test uses 0.02/0.05 gates.
These support feasibility, not automatic acceptance for our corpus.
[VAE tests](https://github.com/tgo-app-dev/vpipe/blob/aad3a10e654af71ae46cb627e5f4f7da22ff0593/tests/unit-tests/minimax-h3-vvae.cc).

The useful transfer is **same-model mixed precision + fast dense GEMMs and
attention + scratch reuse**. Implement those concepts through CUDA libraries
first. For Metal, reuse our vendored Steel infrastructure where possible and
preserve upstream notices for any additional code; copying the VPIPE runtime
is unnecessary.

## Current implementation and measured motivation

Source audit baseline: `5a04870d0aa3fb53056664b30d42db00f6249d62`.

The full [video decoder](../../src/vae/video_vae.c) is a **36-block transformer**:
hidden width 2,048, MLP width 8,192, 32 attention heads of dimension 64, and
24 latent channels. Normal tiles contain `7 × 16 × 16 + 5 = 1,797` rows,
including four learned registers and one suffix row. Weights and working
tensors use FP32. This is a different workload from DiT attention and from
Wan's causal-convolution decoder.

Each block performs RMSNorm, QKV projection, Q/K normalization plus RoPE,
dense noncausal attention, output projection, scaled residual, another
normalization, SwiGLU MLP and a second scaled residual. A final normalization
and projection produce pixel patches.

| Area | Existing behavior | Opportunity / restriction |
| --- | --- | --- |
| Resident tiled decode | Weights stay loaded across tiles; `run_resident_tile()` already submits a complete tile rather than synchronizing after every layer | Do not propose resident weights or one submission per tile as new features. Keep the separate streaming-weight path working. |
| CUDA FP32 SDPA | Default uses `attention_online<float>`; `fast_decoder_attention()` already offers bounded-score FP32 GEMMs and a tiled FP32 kernel | Retain as comparison/fallback; add VAE-specific BF16/FP16 dense attention for H=32/D=64 |
| Existing fast CUDA VAE math | `H3_FAST_CUDA_VAE` accepts `tf32`, `f32`, `tiled` or `default`; default is `tf32` when fast CUDA is effective | Decoder policy must be explicit and independent of DiT math; legacy TF32 remains a benchmark comparator |
| CUDA linears | cuBLASLt FP32 already exists, with algorithm selection/cache; descriptors are constructed during calls | Add 16-bit Tensor Core GEMMs with FP32 accumulation and prepared descriptors/workspaces; prioritize MLP as well as attention |
| Buffers | Most activations and RoPE already persist; latent/suffix reuse is tied to `fast_cuda_effective` | Make safe buffer reuse independent of global fast CUDA; bound all shape caches. |
| Delivery | `unpack_frame_range()` reads the entire FP32 patch projection to the CPU, unpacks and normalizes it; spatial/temporal blending is on the host | Benchmark GPU packing and bounded asynchronous readback; preserve arithmetic and blend order. |
| Encoder | [Video encoder](../../src/vae/video_encoder.c) uses convolutional blocks and repeatedly allocates intermediates | Secondary target: reuse storage and prepare convolution execution, keeping posterior and padding semantics. |
| Metal | [Metal backend](../../src/metal/gpu.m) uses MPSGraph/MPS for large FP32 operations | Keep as reference; add qualified BF16/FP16 VAE GEMMs and Steel H=32/D=64 attention |

Recent [CUDA SOL records](../outputs/cuda-sol/results.csv) already show why this
matters. These historical full-VAE runs used two denoising steps; they are
context only, not results from the new six-step test campaign:

| Case | Total wall time | Denoising | Post-denoising + delivery | Share of total |
| --- | ---: | ---: | ---: | ---: |
| R1 SOL, 640×480, 243 frames | 193.27 s | 42.06 s | 111.02 s | 57.4% |
| R5 SOL, 1344×768, 243 frames | 740.70 s | 361.77 s | 336.71 s | 45.5% |
| R6 SOL, 1344×768, 362 frames, nine `max` images | 1,674.30 s | 1,076.18 s | 503.69 s | 30.1% |

**The post-denoising figure includes audio decoding, video decoding and delivery. It is
not an isolated video-VAE measurement.** M0 must separate those costs before
setting a realistic improvement forecast. For a measured accelerated fraction
`f` and its speedup `s`, total speedup is `1 / (1 - f + f/s)`; do not multiply
unrelated optimization claims.

## Compatibility and quality contract

The revised objective is **better reconstruction than TAEH3, closer to the
original decoder, at substantially lower cost than current FP32 execution**.
BF16/FP16 rounding differences are acceptable. The reference tier remains
unchanged; do not describe the balanced tier as lossless or bitwise equivalent.
No new trained decoder or low-bit weight format is required for this milestone.

Preserve all of the following:

- Released **256-pixel spatial tiles**, minimum 64-pixel overlap, and the actual
  starts/overlaps returned by the existing planner. Larger or automatic tiles
  are not a quality-neutral optimization; see [tile compatibility](bugfixes/design-tilefix.md).
- Seven-position latent windows advancing by five, initial 22-frame output,
  five-frame temporal overlap, existing crop/drop/pad rules, and exact output
  frame counts. Preserve the short diagnostic path and dedicated image-VAE role.
- Tile-local positional coordinates, registers, suffix, normalization epsilon,
  activation functions, attention scale and dense attention to all tile rows.
- Latent/pixel mean and standard deviation, current video tile clamping before
  blending, vertical-then-horizontal spatial blending and temporal blend order.
  Another implementation's clamp placement is not permission to change ours.
- Encoder reference resizing, causal padding, posterior mean/log variance,
  sampling RNG and RNG consumption order. Encoder changes can affect every
  subsequent diffusion step and need a stricter gate than delivery changes.
- DiT, LoRA, SOL/Sage, sampler, seeds, audio decoder, frame rate, continuation
  trimming, muxing and color conversion settings. Initially change execution
  precision only: no sparse VAE attention, skipped tiles, temporal interpolation,
  altered tile geometry or model architecture.

In particular, overlapping H3 tiles do **not** permit reuse of hidden/K/V values:
their noncausal attention contexts and positional coordinates differ. Causal
convolution caching from Wan cannot be transplanted into this decoder.

### Numerical gates

Freeze fixtures, thresholds and metric definitions in M0 before tuning. The
following are proposed balanced-tier acceptance limits, not measured results.
Record reference repeatability and discrepancies against a pinned official H3
oracle. Validate against current FP32 output and compare **reference, balanced
and existing preview on the same latents**. The precision relaxation does not
relax frame-order, padding, seams, finite-value or memory correctness.

| Comparison | Gate |
| --- | --- |
| Copy, reshape, buffer reuse, scheduling, temporal planning | Bitwise-equal tensors/pixels where arithmetic is unchanged; identical shapes, order and counts |
| Mixed-precision raw tile projection | Finite outputs; relative L2 ≤ 0.02 versus FP32; record per-layer ranges, unclamped output, saturation/overflow and worst rows |
| Final RGB in [0,1], before media compression | Initial floor: per-clip PSNR ≥ 35 dB and SSIM ≥ 0.98 versus FP32; worst-frame PSNR ≥ 30 dB; inspect maximum errors and seam/temporal-change maps |
| Improvement over existing preview | On each rich-content clip, ≥3 dB PSNR gain and at least 20% lower mean LPIPS error versus FP32; report SSIM and worst frames as well. Pin the LPIPS implementation/weights. Near-identical synthetic fixtures use absolute gates instead. |
| Rounded 8-bit RGB | Report changed-channel fraction and maximum code difference; do not require bitwise equivalence for mixed precision. Pure unpack/copy/stitch scheduling changes compare exactly against the same precision inputs. |
| Encoder posterior mean/log variance and fixed-RNG sampled latent | Relative L2 ≤ `1e-6`, scaled maximum error ≤ `1e-5`; exact RNG state/order and finite values; investigate tail frames separately |
| Decoder-policy invariance | Identical denoised video/audio latent payloads for matched generation; original preview-VAE outputs and conditioning caches unchanged |

An isolated FP32-preserving candidate should still satisfy relative L2 ≤ `1e-5`
and scaled maximum error ≤ `1e-4` at operator boundaries. Do not mistake that
diagnostic gate for the mixed-precision decoder's requirement. A balanced
candidate that fails its quality floor remains unqualified or falls back to
more FP32 operations; do not weaken the gates after seeing its scores.

Metrics must include every frame, not only contact sheets or global averages.
Compute errors on raw float output, then on rounded RGB. Encoded MP4 bytes are
not a numerical oracle. Retain three-way videos, difference heatmaps, seam crops
and a local HTML gallery with all three tiers for inspection. Explicitly inspect
seam bands and frame-to-frame error changes so good average metrics cannot hide
a grid or flicker. No waiting for a human-review response is required to publish
the measured results; human observations can refine a later quality decision.

Use completed six-step generation latents and encoded natural-video fixtures
with faces, fine texture, text, motion and saturated/high-contrast regions.
Generation quality alone cannot establish reconstruction quality; compare all
three decoders on identical inputs and retain the natural-video fixtures.

## Execution design

### Policy and ownership

Proposed CLI/API selection, **not implemented yet**:

| `--full-vae-execution` | Behavior |
| --- | --- |
| `legacy` (initial default) | Preserve current dispatch, including existing fast-CUDA environment behavior |
| `reference` | Existing full video decoder using FP32 reference math; do not change the requested DiT mode |
| `balanced` | Qualified native original-model decoder using BF16 or range-safe FP16 operands, FP32 sensitive operations and a validated device/shape plan; safe preflight fallback to reference |

An explicit non-legacy choice with `--preview-vae` is a clear configuration
error; ordinary preview commands remain unchanged. `H3_CUDA_REFERENCE=1`
continues to force the effective reference execution policy. Conflicting
legacy `H3_FAST_CUDA_VAE` overrides with an explicit full-VAE policy must be
reported, not silently change its precision. Log requested/effective policy,
weight/activation/accumulation types, attention/GEMM algorithm, tile geometry
and workspace once per decoder. A versioned recipe records the selected
BF16/FP16/FP32 boundaries for reproducibility; no per-render quality guesswork.

This setting selects **full video decoding**, not audio or the still-image
decoder. Initially keep the latter on their existing paths. Encoder execution
gets its own internal policy when M5 is qualified; decoder-only flags must not
change conditioning. Unsupported CPU/device/shape cases retain the reference
implementation and explain the fallback.

Use a decoder-owned execution plan: weights, activation arena, cuBLAS handles
and prepared plans, attention scratch, input/output staging, optional captured
graphs and bounded completion events. No global math-mode changes shared with
the denoiser. Extend the persistent decoder cache identity with model content
identity, backend/device, role, effective execution/precision, geometry and
plan version. Retire old plans before incompatible reuse.

Decoder policy is presentation state: it must not change latent payloads,
conditioning identities, sampler trajectories or exact-resume requirements.
Record it in output provenance without making `.h3av` decoding depend on the
machine or mode that generated the latents. Exercise generation, decode-only,
cached sessions and full live-preview paths explicitly.

### CUDA mixed precision and scheduling

Start with BF16 weights and activations derived once from the original
checkpoint. Use BF16 Tensor Core GEMMs with FP32 accumulation for QKV, output
projection and the MLP; keep normalization reductions, softmax statistics and
attention accumulation in FP32. Accumulate scaled residual additions in FP32
before an explicitly chosen storage conversion. Measure intermediate ranges
and retain selected residual/output buffers in FP32 if quality requires it.
This is original-model decoding, not DiT weight quantization.

Implement the H3 VAE's actual dense attention contract, S=1797/H=32/D=64 with
sequence tails, bias-bearing projections and existing per-head QKV layout.
Benchmark supported cuDNN/native BF16 attention and a range-safe FP16 candidate
against current online, bounded-score FP32 and legacy TF32 paths. Existing
DiT D=128 SOL/Sage dispatch must not capture VAE calls. Keep the encoder/audio
and shared context's math modes independent.

Profile the complete 36-block tile; attention alone is not the priority gate.
Reuse cuBLASLt descriptors, algorithms and workspace, allowing 16-bit operands
with **FP32 accumulation**. For FP16 candidates, detect nonfinite conversions
and intermediate overflow; use per-operation BF16/FP32 fallback before
publishing a tile. Scaling must be explicit, reversible in the recipe, and
qualified on outlier fixtures. Never silently clamp an overflowing activation.

Select BF16 versus FP16 by complete-decode quality/speed evidence. Preserve
original source weights, and do not keep unnecessary FP32 and 16-bit GPU copies
resident together. Math flags and casts belong in the recorded recipe; generic
fast-math compiler switches are not a substitute for validating the operators.

Capture fixed-shape compute only after weights, addresses, descriptors and
scratch are stable. Keep host I/O, progress, cancellation and memory admission
outside capture. Unsupported capture falls back before writing output; a
runtime GPU fault is an error, not a reason to replay into a partially written
canvas. Check memory/cancellation between bounded tile submissions.

Experiment with B=1/2/4 independent tiles. Linears can flatten batch and row
dimensions; attention must retain separate per-tile contexts, including each
tile's own registers/suffix. Preserve output order for stitching. A batch is
selected only when complete decoding wins after packing, workspace and copies.
Avoid compiling a cross-product of all possible shapes; cap plan/cache count.

Fuse elementwise operations where the chosen precision recipe's gate passes.
Adapt the existing fused Q/K norm/RoPE operation to the decoder dtype instead
of assuming VPIPE's separate passes are required. A lossless copy/scheduling
change still has to preserve its input precision's values exactly.

### Bounded output delivery

Implement GPU projection-to-RGB unpacking with the current frame selection,
mean/std and clamp point. Compare unpack alone bitwise to host code, including
the 22-frame special crop. Do not copy register/suffix projections or unused
frames. Initially leave the proven host stitcher intact to isolate the gain.

Then benchmark ordered GPU stitching and/or a bounded host staging ring. Keep
unblended overlap tails as required by current semantics; do not accidentally
reuse already blended tails. Emit a frame only after its spatial and temporal
overlaps are final. Quantize to output RGB bytes only after those blends, using
the same conversion as the existing writer.

Allow the CLI video sink to consume completed frame ranges incrementally;
retain the materialized float-output API for existing callers and the oracle.
Use bounded pinned buffers and backpressure, including FFmpeg failure and
cancellation. Preserve audio/video timestamps and continuation trimming.
Asynchronous copy/compute overlap is useful only when total delivery improves.

The target memory model is weights plus bounded tile/batch scratch, overlap
tails, one bounded output chunk and a small staging ring. An API caller asking
for a materialized video still pays for that final host output allocation;
do not describe that mode as constant-memory overall.

Reuse the existing memory admission/monitoring logic. Account for native and
library workspaces, captured graph resources, pinned host memory and retained
decoder caches. Reserve before enqueueing, limit in-flight work, and obey the
configured host cap (including the existing 110 GB decimal process/Metal guard) and
available VRAM. A failed admission may select B=1 before execution; a hard
limit or device fault must stop cleanly. Repeated requests must reach a bounded
plateau rather than accumulate plans, events or decoder buffers.

### Early TensorRT experiment, with an explicit stop condition

Use the original full-precision checkpoint to export one 256-pixel, seven-latent
tile. Build a full-precision oracle export plus an explicitly mixed-precision
candidate; avoid prequantized low-bit exports. Compare input normalization,
RoPE, registers, output projection and raw tile pixels before host stitching.

Pin a TensorRT/toolkit version compatible with the actual node. Inspect selected
layer types; retain FP32 norms, softmax/reductions and any unstable operations.
Use BF16/FP16 where supported and qualified. Disable TF32 in the FP32 oracle,
and explicitly record its use in any candidate. Version-specific precision
controls must be audited rather than copied from an older compiler script.
[Accuracy guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)

The first experiment is offline and decoder-only, immediately after the basic
native mixed-precision prototype. It must beat TAEH3 on quality and deliver at
least 10% lower complete-tile time than that native candidate, with acceptable
build/load cost and memory, before adding an optional C++ runtime adapter.
Recheck the advantage against the final native path before qualification.
Keep original host tile/temporal planning. Do not import the fixed
17-frame encoder profile: changing encoder padding would alter conditioning.

If qualified, add an explicit `tensorrt` execution choice in an optional build;
otherwise record the negative result and close that branch. Key local engines
by weight hash, export/plan version, GPU capability, runtime/toolkit versions,
shape and precision flags; bound cache size and reject stale engines. Ordinary
CUDA and Metal builds must not acquire a TensorRT or Python runtime dependency.

### Encoder and Metal follow-through

Profile image/reference-video encoding separately. Reuse padded/convolution
buffers, prepare stable convolution plans and remove redundant normalization or
transfer passes only after proving equivalence. Evaluate channels-last or
cuDNN FP32 convolution only if convolution/layout cost dominates. Preserve
causal context, arbitrary supported reference lengths and image T=1 handling;
never pad all calls to a new fixed temporal length to satisfy a compiler.

Keep the encoder FP32 initially; changing conditioning has downstream effects
that decoder-only quality comparisons do not measure. VPIPE's BF16 encoder is
useful evidence, not permission to lower this plan's posterior/RNG gate.

After CUDA qualification, port shared planning, reuse and delivery improvements
to Metal. Benchmark BF16 dense GEMMs and D=64 Steel attention against our FP32
MPSGraph/MPS path; compare FP16 where range and quality permit. Retain the
fastest qualified combination; native-only execution is not a requirement.
Qualify command-buffer scheduling and memory on the local M4 Max independently.
Keep M5 NAX dispatch separate and unqualified until actual M5 testing.

If MLP remains dominant on M4, make a bounded VPIPE-style ANE feed-forward
experiment: dynamically split rows, keep attention on GPU, measure compilation
and synchronization costs, and disable the split when it loses. ANE remains
optional, opt-in and subject to the same balanced-tier quality gates. It is
not a prerequisite for shipping the CUDA or GPU-only Metal path.

## Measurement and testing plan

### Hardware and profiling

The last measured CUDA node was `cuda-pro6000`, RTX PRO 6000 Blackwell
Server Edition, SM120, about 95.59 GiB VRAM, with models under
`/path/to/models/MiniMax-H3`. Re-probe it before testing; do not label its
results as RTX 5090 results. Metal follows on the local M4 Max with 128 GiB.

Record cold process wall time, decoder weight load, plan/build/capture, warm
decoder time, audio time, video tile compute, attention, GEMMs, elementwise,
packing/copies, spatial/temporal stitching, FFmpeg/mux and final delivery.
Use monotonic wall clocks plus asynchronous GPU events at selected boundaries;
profiling must not add per-operator synchronization to normal runs. Overlapped
durations are not additive. Compare both unprofiled wall time and instrumented
breakdowns, with a bounded timeline capture if needed.

Sample device/process VRAM and host RSS/physical footprint; report allocator
peaks, library workspaces, sampling interval and maximum sample gap. Do not
report sampled VRAM as an exact allocator peak. Separate cold engine/model load
from resident warm runs, and retain the startup penalty in CLI wall time.

### Corpus and gates

Use decode-only `.h3av` inputs wherever possible, with identical video and audio
latent hashes for each pair. Generated-latent fixtures for this campaign must
come from **exactly six completed denoising steps (`--steps 6`)**. Verify their
existence, hashes, requested step count and completed step count before reuse;
if none cover a case, capture one complete six-step state and reuse it across
decoder comparisons. Older generation artifacts with other step counts remain
historical/diagnostic evidence, not substitutes for the six-step video corpus.
Natural-video encoding and synthetic operator fixtures do not run a denoiser.

| Case | Purpose |
| --- | --- |
| C0: small/single tile, short video, planner boundary and tail cases | Operator oracle, tile crop/blend correctness, error paths; include sequence tails rather than only S=1797 |
| C1: 640×480, **243 frames** | Primary complete-decode and few-step wall-time gate; 9 spatial tiles × 14 temporal chunks = 126 tile evaluations |
| C2: 1344×768, **243 frames** | High-resolution gate; 28 × 14 = 392 tile evaluations |
| C3: 1344×768, 362 frames | Decode-only duration/memory stress; 28 × 21 = 588 tile evaluations; no need to rerender nine references merely to time identical decode geometry |
| C4: high-quality natural-video/reference-derived fixtures | Faces, fine texture, motion, text/high contrast, spatial seams and temporal joins; compare every decoded frame |
| C5: conditioning/compatibility | Image T=1, image sizes `match`/`max`, short/multiple video references, continuation crop, FL2VA/Ref2VA, single-still and unchanged preview VAE |

The **243-frame gate stays in place**. All generated test videos on CUDA and
Metal use **exactly six denoising steps (`--steps 6`)**, including 1344×768 and
any generated 362-frame fixtures. This replaces the earlier two-step limit.
Decode-only repetitions run no additional denoising. Use the same prompt,
references, seed, sampler and denoiser policy within a comparison and save/check
its latent payloads. Include fast CUDA off/on compatibility, proving that
explicit reference/balanced VAE math remains independent of DiT math.
Deliberately interrupted cancellation/failure tests are labeled diagnostics;
their partial outputs cannot count as complete six-step inspection videos.

For C1/C2 run a cold reference/balanced pair and three interleaved warm pairs;
warm the resident decoder without excluding required per-request work. Also
decode the same states through TAEH3, retaining its quality metrics and cold/warm
times. C3 needs one reference/balanced pair plus repeated bounded chunks for
memory behavior. C4 needs all three tiers. Qualify only the winning candidate
on the full corpus; losers stop after tile tests. Retest legacy TF32 separately
where it is already used so a gain is not manufactured by choosing a slow
comparison mode.

Target at least **1.5× full video-decode speedup** over FP32 on C1 and a measurable
few-step total-wall reduction (at least 5%), with no >3% complete-decode
regression on C2. Include copies, unpacking and stitching in decode time and
audio/mux in end-to-end time. Report variation; a gain smaller than measured
noise is not a pass. These are acceptance targets, not promised results.
If only memory improves, report a memory improvement and leave the speed goal
open. Unsupported or slower shapes should retain reference dispatch.

### Bounded CUDA qualification campaign

Plan one campaign of **at most eight hours**, including failed attempts and
closing artifacts. Implementation work precedes the campaign. Reserve time
before starting each job; stopping at the cap creates an explicit deferred or
failed record, not a passed gate. Metal has its own later bounded campaign and
must not be presented as covered by CUDA results.

| Activity | Maximum minutes |
| --- | ---: |
| Inventory, builds, artifact/weight identity and fixtures | 30 |
| FP32 oracle, encoder and boundary correctness | 65 |
| Candidate profiling, tile/batch/graph trials, TensorRT spike (at most 30 minutes of this slot) | 60 |
| Winning C1/C2 cold/warm pairs, TAEH3 comparisons and C3 stress | 115 |
| Matched six-step end-to-end checks and reusable generated-state capture | 90 |
| Lifecycle, compatibility, preview/still and memory/failure checks | 45 |
| Metrics, videos, gallery and final report | 35 |
| Contingency | 40 |
| **Total** | **480** |

Preflight estimates using a tile run and measured six-step denoising costs;
do not treat the historical two-step wall times as six-step estimates. Reuse
the end-to-end runs' saved states for the decode corpus instead of generating
them again. Reduce optional candidates first if needed; do not lower the step
count or consume closing time with another optimization sweep. TensorRT build
overruns are deferred with their reason; native delivery is not blocked.

### Four-hour local M4 qualification

M7 has a separate **240-minute hard limit**, including failed attempts and
reports. `tests/fast_vae_campaign.py` detects macOS, samples process physical
footprint and RSS instead of inventing discrete VRAM figures, persists its
original deadline, and admits
jobs using measured estimates. Reserve the last **25 minutes** for closing.
Decoder allocator peaks and macOS physical footprint supplement sampled RSS.

Use retained, proven **six-step** 640×480/243 states. Capture a new six-step
1344×768/243 state only if the measured estimate fits; stop competing GPU work
while it runs. Do not repeat expensive denoising merely to compare presentation
decoders. Reuse those states for FP32, balanced and unchanged TAEH3 decoding.
Use small six-step API fixtures to test decoder-policy latent invariance.
Never label a summed decode estimate as measured whole-generation wall time.

Preflight full 36-block tiles, discard losing candidates there, then run one
cold plus **two interleaved warm pairs on C1** and one cold plus **one warm pair
on C2**. Add repeats only if required coverage and the closing reserve fit.
Keep both **243-frame** gates; do not reduce frame count or six-step completion.
Estimate full-size reference-image checks separately: Qwen vision on the
1360×1808 image takes several minutes even when the output is only 96×64.
The measured M4 allowance is 660 seconds per complete six-step `max` image
case, with a 400-second admission estimate. An individual timeout adjustment
never extends the original campaign deadline; retain the earlier timeout and
identify successful retests separately.
C3 uses a repeated-tail 362-frame state for decode/memory stress and small
repeated chunks for lifetime checks. C0/C4/C5 use compact real/synthetic cases.
The ANE row-split trial is bounded and is rejected if packing/synchronization
already outweighs saved GPU work. It cannot delay GPU-only qualification.

| M4 activity | Maximum minutes |
| --- | ---: |
| Setup, tile candidates, operator/range checks, bounded ANE trial | 30 |
| Six-step fixture capture and compact integration checks | 85 |
| C1/C2 three-tier decode and C3 memory stress | 60 |
| Encoder/compatibility, every-frame metrics and videos | 30 |
| Final HTML, media/provenance audit and documentation | 25 |
| Contingency | 10 |
| **Total** | **240** |

Keep the CUDA campaign and its evidence unchanged. Report Metal speed/quality
and memory independently, including failed speed targets, slower alternatives
and any cases deferred by the cap. A qualified quality/memory improvement alone
does not satisfy the 1.5× decode / 5% whole-render performance target.

### Required visual inspection reports

Retain a machine-readable manifest/results CSV and JSON with command, revision,
build flags, model/input hashes, requested/completed denoising steps, effective
policy, device/toolchain, every gate, timings, memory and errors.

Generate `outputs/fast-vae/cuda/review.html` and, after Metal testing,
`outputs/fast-vae/metal/review.html`, plus an `outputs/fast-vae/index.html`
linking the available reports. Each backend report must provide:

- A row for every video case with full-length **reference / balanced / TAEH3**
  MP4 players from identical latents, shared play/pause and seeking, clear tier
  labels, and downloadable videos. Contact sheets supplement playback.
- Prompt, seed, references, dimensions, delivered frame count, source-state
  hash and six requested/completed steps for generated fixtures; distinguish
  natural-video and synthetic sources and mark incomplete diagnostics clearly.
- Total wall time, isolated VAE time, speedup, peak VRAM/host memory and quality
  metrics beside the videos, with links to commands, logs and CSV/JSON records.
- Representative frame/contact sheets, difference heatmaps and spatial/temporal
  seam crops, including worst-scoring frames with timestamps for inspection.
- Visible pass/fail/deferred status for every case. A missing tier or video is
  explicit incomplete coverage, not a successful comparison.

Copy required playback assets and records from the CUDA node into the local
report directory. Use relative asset links and no external playback dependency.
Validate every link, video readability, frame count/duration and the shared
playback controls locally before marking the report task complete. Publish a
usable partial report even if the time cap or a test failure prevents full
coverage; preserve the failed/deferred status. No human-review response is
required, but report generation and usable playback are mandatory exit gates.

Keep lossy viewing assets distinct from raw numerical evidence; retain
checksums and metric summaries even when large temporary tensors are discarded
after validation.

## Milestone exits and rollout

| Milestone | Exit evidence |
| --- | --- |
| M0 — Evidence and oracle | Isolated phase profile, hashed corpus, frozen numerical contract, measured bottleneck and memory budget |
| M1 — CUDA mixed-precision decoder | BF16/FP16 Tensor Core GEMMs and dense attention, FP32 sensitive operations, explicit policy and stable original-model output |
| M2 — Early TensorRT decision | Mixed-precision experiment evidence; optional integration only after quality/speed gates, otherwise explicit rejection/defer reason |
| M3 — Measured tile scheduling | Qualified graph/batch/fusion wins or documented negative results; bounded replay/cancellation/cache behavior |
| M4 — Bounded delivery | Equivalent unpack/blend/frame order for the same precision inputs, bounded staging, measured decode/memory effects |
| M5 — Encoding | Measured reference-encoding improvements passing posterior/RNG tests, or a documented decision to keep existing kernels |
| M6 — CUDA qualification | Six-step C1/C2 wall-time gates, three-way quality evidence, C3 memory stress, compatibility tests and locally verified HTML video report |
| M7 — Metal follow-through | BF16/FP16 GPU path versus FP32 and tiny, six-step C1/C2/memory qualification, conditional ANE decision and locally verified HTML video report |
| M8 — Close and document | User guide, linked playback reports, complete evidence/coverage ledger, dependency notices and honest release recommendation |

Start opt-in with `legacy` behavior unchanged. Promote balanced dispatch
only for qualified device/shape combinations after correctness, complete-decode
performance and memory gates pass. Keep a selectable reference path. A negative
experiment can complete its investigation task, but cannot mark the overall
speedup or a missing CUDA/Metal qualification gate as passed.
