# Denoiser quantization for faster previews

Research date: 2026-09-18. Scope: repeated H3 denoising only, retaining the
current text/reference encoders, AdaLN precomputation, audio/video decoders,
sampler schedule and delivery. Local source inspected at
`7fcc1ae3d3878060259d864683f7a6a025048b33`. No new GPU experiment or inference
implementation was performed for this review. Recommendations below are
engineering judgments, not measured quantized h3cli speedups.

The recommendation is to build an **8-bit foundation**, then add **native
NVFP4 on Blackwell**. Start with INT8 weight-and-activation computation plus
rotation on Ampere; evaluate FP8 computation on Ada/Blackwell alongside that
quality baseline. Defer INT4/SVDQuant until an 8-bit implementation establishes
whether 24-GB VRAM remains the limiting factor. Do not make ordinary weight-only
compression the principal compute acceleration strategy.

## Recommended choices by card

Here W8A8 means both weights and GEMM inputs are eight-bit; W4A4 means both
are four-bit. Outputs, residuals, normalization and accumulation need not use
that narrow type.

| GPU | First useful implementation | More aggressive candidate | Main reason |
| --- | --- | --- | --- |
| RTX 3090, 24 GB, SM86 | INT8 W8A8 with ConvRot-style activation rotation; BF16 outputs and INT32 accumulation | INT4 W4A4 with SVDQuant or a validated rotation method, if 8-bit residency is insufficient | Native integer tensor cores; no native FP8/FP4 GEMM. Weight movement is a major measured cost. |
| RTX 4090, 24 GB, SM89 | Compare FP8 E4M3 W8A8 against INT8 W8A8+ConvRot; use INT8 as the H3-specific quality reference | INT4 W4A4 for greater headroom, if its packing/correction costs pay off | Both eight-bit arithmetic routes are plausible; capacity remains tight. FP8 has a simpler floating-point path but H3 quality must be checked. |
| RTX 5090, 32 GB, SM120 | FP8 W8A8 as a relatively small implementation step and baseline | **NVFP4 W4A4 is the primary aggressive preview target**, with BF16 retained for sensitive matrices | Eight-bit storage can already remove repeated weight streaming at short presets; four-bit adds activation headroom and native compute throughput. |
| RTX PRO 6000 Blackwell, 96 GB, SM120 | **FP8 W8A8 first** | NVFP4 only if measured complete denoising beats FP8 sufficiently | The existing BF16 core is resident, so four-bit has no residency breakthrough to guarantee a win. |

This refers to the previously tested **RTX PRO 6000 Blackwell**, not the older
RTX 6000 Ada. NVIDIA documents FP8 GEMM from compute capability 8.9, while its
SM120 CUTLASS example explicitly implements block-scaled NVFP4 GEMM. INT4 and
NVFP4 are different arithmetic formats; an NVFP4 checkpoint on Ampere/Ada
requires conversion or emulation and does not confer native FP4 throughput.
[NVIDIA narrow-precision API](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#narrow-precision-data-types-usage),
[NVIDIA SM120 example](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu).

If only one initial quantized path can be funded, INT8 W8A8+ConvRot covers the
3090/4090 need with relevant H3 precedent. If engineering is focused on the
rented Blackwell nodes, do a small FP8 baseline first and prioritize NVFP4
qualification on the 5090. Select by measured shape/device performance;
8-bit INT8 is not automatically the fastest route on Blackwell.

## What our measurements establish

Use the accepted fast-CUDA implementation as the baseline. Old comparisons
against our original scalar attention greatly overstate the opportunity left
for quantization. The current GEMM path uses BF16 operands and FP32
accumulation, cached cuBLASLt plans, and device weight caching when full
residency is impossible. The existing CUDA INT8 entry points report unsupported;
there is no already-working CUDA quantized denoiser to switch on.

The retained matched `image05` profiles use 288×384, 56 frames, 20 denoiser
steps, 50 layers, no token reduction and reuse/core-reuse one:

| Node | Denoising wall increment after reported load | GEMM event increment | Attention event increment | Total recorded streamed DiT bytes |
| --- | ---: | ---: | ---: | ---: |
| 3090 | 101.153 s | 23.939 s | 2.634 s | 415.017 GiB |
| 5090 | 29.682 s | 7.760 s | 0.964 s | 286.822 GiB |
| PRO 6000 | 6.274 s | 4.620 s | 0.640 s | 0 |

The event figures are explanatory components, not independent additive wall
intervals. Node differences include storage, PCIe and CPU staging. The 3090
reports 66.192 seconds of unhidden streamed-weight wait; the 5090 reports
17.698 seconds. Compression plus true quantized residency can remove a much
larger fraction of their wall time than faster GEMMs alone. The PRO 6000's
short-preset denoising is chiefly GEMM work after attention was accelerated.

Sources: [3090 qualification](cuda-3090-qualification.md),
[5090 qualification](cuda-5090-qualification.md),
[PRO 6000 fast validation](cuda-fast-validation.md), and retained
[3090 profile](../../outputs/cuda-validation/rtx3090/quality/fast/image05.log),
[5090 profile](../../outputs/cuda-validation/rtx5090/quality/fast/image05.log),
[PRO 6000 profile](../../outputs/fast-cuda/quality/fast/image05.log).
The earlier 4090 qualification predates this fast path; it establishes the
24-GB streaming constraint but is not a matched current quantization baseline.

Our repeated 50-block core has 5376-wide hidden states, 7168-wide attention
projections and a 14336-wide FFN. Its four stored matrices per block are QKV,
attention output, fused SwiGLU input and FFN output. Their BF16 payload is
**35.889 GiB**. Pure payload arithmetic gives:

| Storage | Same core, before scales and retained high-precision matrices |
| --- | ---: |
| BF16 | 35.889 GiB |
| INT8 or FP8 | 17.944 GiB |
| Packed four-bit values | 8.972 GiB |
| NVFP4 including one byte scale per 16 values | about 10.094 GiB |

These are calculated sizes, not measured whole-model VRAM. Add token refiners,
nonquantized matrices, scales/correction branches, activations, scratch and
library/context overhead. Eight-bit residency on 24 GB is plausible for short
previews but must not be promised at every resolution/duration. Mixed-precision
exceptions can also move it back above the residency threshold.

We already precompute AdaLN and keep its large weight branches out of repeated
denoising. Do not count public 33B-to-20B AdaLN pruning as an additional 39%
denoising speedup available to our loop. The official H3 description likewise
separates roughly 13B AdaLN parameters from the repeated core.
[MiniMax architecture](https://github.com/MiniMax-AI/MiniMax-H3#h3-omni-transformer).

## Public results: useful evidence and confounders

**H3 eight-bit computation.** A community H3 author reports matched compiled
H100 denoising at 3.03 s/step BF16, 2.90 INT8+ConvRot and 2.47 FP8+ConvRot.
Eager INT8 took 11.28 s/step, showing how activation conversion overhead can
lose the entire benefit. The author's H3 comparisons favor rotated INT8 over
FP8 for fidelity, with FFN down-projection particularly sensitive. These
results support fused conversion and an INT8 quality reference; they do not
predict our consumer-GPU speed or establish our quality acceptance.
[Author's H3 measurements](https://huggingface.co/multimodalart/MiniMax-H3-Pruned#8-bit-compute).

**Practical 24-GB capacity.** Another H3 conversion author reports 4090
loader peaks of 20.424–21.025 GiB for mixed INT8 profiles, leaving only
1.471–2.072 GiB free. Those are loader/projection checks, expressly not complete
video-generation peaks. The retained BF16 layers are useful candidate
exceptions, not a universal layer-selection recipe for our FL2VA/Ref2VA and
adapters. [Author's loader tests](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants#measured-rtx-4090-loader-results).

**Four-bit engine evidence.** Nunchaku reports FLUX INT4 3.0× faster than an
NF4 W4A16 baseline on 4090 and NVFP4 3.1× faster than BF16/NF4 on 5090. These
are different-model results and the baselines matter. SVDQuant's high-precision
low-rank correction also needs fusion: naïve branches can erase its benefit.
This is evidence for real low-bit GEMM and careful kernel integration, not
for applying a generic four-bit file loader to H3.
[Nunchaku implementation and results](https://github.com/nunchux-ai/nunchaku#performance).

**H3 FP4 is not a guaranteed resident-card win.** A community FL2VA
SVDQuant/PDD benchmark on PRO 6000 reports 106 s for INT8+eight-step PDD and
111 s for FP4+eight-step PDD. Its 3.4× headline compares against 20-step BF16;
it cannot be credited to quantization. The reported 32-GB budget experiment
on the 96-GB card is not a physical 5090 measurement. A separate real-5090
section also changes attention/FFN memory handling and uses a remote text
encoder. Those results reinforce the importance of residency and memory
planning, while remaining unsuitable as isolated h3cli quantization speedups.
[Author's H3 FP4 benchmark](https://huggingface.co/1ronman1993/MiniMax-H3-SVDQuant-fp4-pdd8#benchmarks-rtx-pro-6000-96gb-comfyui-1344768--141-frames--24fps-fixed-seed).

## Implementation choices worth making

1. **Scope the policy to repeated DiT projections.** Attach the packed-weight
   descriptor to the denoiser, not a global GEMM dtype switch. Initially cover
   QKV, attention output, FFN input and FFN output. Retain AdaLN, input/output
   projections, refiners, norms, residual state, Q/K normalization, RoPE and
   the existing BF16 attention backend. BF16/F32 exceptions must be allowed
   per matrix without disabling quantization for the entire block.
2. **Keep compressed weights compressed through upload and residency.**
   Persist an offline packed artifact or prepare/cache it once per model
   identity. Do not quantize every step, transfer BF16 bytes only to compress
   them on the device, or expand the entire core back to BF16 in VRAM. Retain
   the bounded streaming fallback when activations prevent full residency.
3. **Quantize activation tiles cheaply.** For INT8, apply an orthogonal
   ConvRot-style block transform to weights offline and the matching input
   transform online; use per-row weight/token scales and native integer GEMM.
   Fuse or amortize rotation, amax, quantization and dequantizing epilogues.
   For FP8, use calibrated/scaled E4M3 weights and dynamic activation scaling
   supported by the actual kernel. Do not assume Ada supports every Hopper
   or Blackwell scaling mode. Keep sensitive FFN/output matrices higher
   precision if playback shows damage.
4. **Add native NVFP4 as a separate Blackwell option.** Use E2M1 data,
   fine-grained block scales and BF16 outputs with FP32 accumulation where
   supported. Start with calibrated mixed precision and rotation where useful;
   add low-rank correction only if quality requires it and its cost is measured.
   NVFP4 alone is a format, not a substitute for a quantization strategy.
5. **Do not start with AWQ/GPTQ/GGUF/NF4 weight-only execution.** Those may be
   useful import/storage or compatibility options, and optimized weight-only
   kernels can help some shapes. But H3 processes many token rows per step;
   the main target is matrix multiplication, not LLM single-token GEMV.
   A generic dequantize-then-BF16 route primarily buys capacity and can be
   slower on an already-resident PRO 6000.
6. **Defer elaborate INT4 correction until necessary.** SVDQuant W4A4 offers
   a credible maximum-speed/capacity route on 3090/4090, but needs conversion,
   low-rank branches, suitable fused kernels and separate model/adapter
   qualification. Plain calibrated W4A8 is another capacity compromise;
   packed INT4 weights do not by themselves imply native INT4×INT8 execution.
   Confirm whether a proposed backend widens weights or uses mixed arithmetic.

The C/CUDA implementation can use cuBLASLt or selected CUTLASS kernels without
adding Python to inference. Our existing GEMM plan cache is reusable in concept,
but quantized keys must include layout, scale mode and device. NVFP4 should
explicitly target SM120; B200 SM100 kernels are not interchangeable. NVIDIA
added GeForce-class block-scaled FP4/FP8 support in cuBLAS 12.8 Update 1, so
check the linked library patch version rather than only `nvcc --version`.
[cuBLAS release notes](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cublas-release-12-8-update-1).

LoRA folding must happen in the original weight basis before rotation and
quantization, or use a correctly separate high-precision adapter branch. Preserve
our original per-head QKV ordering and fused SwiGLU halves when converting
ComfyUI/Diffusers artifacts. A checkpoint format's name does not prove those
layouts match. [Existing conversion review](../minimax-official-review.md#r06--keep-external-reference-implementations-trustworthy).

## Long sequences and expected benefit

Projection quantization does not accelerate the unchanged attention score/value
products. At long sequences their quadratic cost can dominate again. Treat
quantized attention as a separate denoiser-only experiment after the projection
path: SageAttention's INT8-QK/FP16-PV route fits Ampere, while its FP8-PV route
covers newer hardware. Its published kernel TOPS excludes quantization and
smoothing; compare the complete replacement against our cuDNN attention, not
only FlashAttention or the original native kernel. FP4 SageAttention3 is more
aggressive; its authors still recommend version 2 for precision-sensitive use.
[SageAttention implementation](https://github.com/thu-ml/SageAttention).

No h3cli quantized speedup has been measured. For the PRO 6000 short pilot,
GEMMs occupy about 74% of the denoising wall increment. Ideal 2×/4× acceleration
of those GEMMs alone would produce about **1.58×/2.23× denoising speedup**,
before paying conversion costs. These are Amdahl-law illustrations, not
forecasts. The 3090/5090 can gain more if quantization also removes unhidden
weight staging; larger images may instead move the bottleneck to attention.

Complete-operation speedup is `1 / ((1-f) + f/s)`, where `f` is the current
wall fraction affected and `s` its speedup. If denoising is half the operation,
2× faster denoising yields only 1.33× overall. Unchanged text/reference encoding,
AdaLN startup and VAE decoding still matter, especially with few sampling steps.
Measure fresh-process and cached-conditioning previews separately. Leave the
selected full/tiny VAE unchanged in each comparison.

## Bounded qualification before choosing defaults

Use the existing 288×384/56-frame and 480×640/22-frame presets, actual H3
projection dimensions and short higher-token component probes. Start with
real-weight matrix plus rotation/quantization/epilogue time, then a complete
50-block evaluation and short matched generation. Keep the same number of
actual transformer evaluations, sampler, references, seed, reduction/reuse,
attention backend and VAE for the primary comparison. Test both FL2VA and
Ref2VA, with Turbo separately after its existing artifact/schedule contract
is preserved. Do not bundle PDD or another distillation scheme into the
quantization comparison.

Review identity, limbs, motion, detail, prompt adherence, speech, voice identity,
lip sync and continuation seams using `1.jpg`, `2.jpg` and face/body inputs.
Use human playback/listening rather than numerical tiny/full or BF16/quantized
similarity thresholds. Structural tests still cover scale layouts, finite
output, buffer lifetime, tails, cancellation, memory admission and provenance.
Log effective arithmetic so unsupported hardware cannot silently fall back to
slow dequantization while claiming native acceleration.

Unlike tiny VAE reconstruction, denoiser quantization changes the authoritative
video **and audio** latents: H3 has a shared audio/video transformer. Keeping
AudioVAE unchanged does not guarantee unchanged sound. Re-decoding those saved
latents with the full VAE cannot undo denoising quantization artifacts; an
original-precision final render requires another denoising run. Record the
quantization/model identity in resume, saved-state provenance and cache keys.

Use a fresh explicitly bounded experiment ledger when implementation is
requested; do not spend the previous completed qualification budgets on
unannounced trials. This review adds no implementation tasks or runtime flags.
