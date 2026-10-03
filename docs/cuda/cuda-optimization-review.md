Review date: 2026-09-17. Local source: `6af5d90`, numerical source SHA-256
`6893c2aaf0ade88b1eb0aae1b1e2c7108004cb74b771a9562bfb1eac69bd33b7`.
QuixiAI source inspected at `69172740de9cf1bb12c8718479cefe47eeaf7a19`.

The current CUDA backend has substantial implementation headroom, principally
in attention. The first SM120 specialization removed register spills and reduced
shared-memory use, producing a measured improvement over our generic kernel.
It did not implement the larger, pipelined, tensor-core attention architecture
used by the QuixiAI fork. Another 2× end-to-end improvement is a reasonable
optimization target, not a result established by this review.

The user's comparison is `test2.sh inputs/1.jpg`: 480×640, 362 frames, 50 steps,
reuse one, original BF16 weights. M4 Max wall time is approximately 190 minutes
per segment. PRO 6000 time is projected at approximately 50 minutes from
denoising progress, giving a provisional 3.8× ratio. The completed PRO wall
time, including decoder and mux, is still needed. Segment one and continuations
must be compared separately because their protected-prefix work differs.

The QuixiAI README reports 45 minutes on one RTX 3090 versus 162 minutes on an
M3 Ultra for original BF16, 1344×768, 10 seconds, 50 layers and a 20-step schedule
with reuse two (11 DiT evaluations). This is useful evidence of what another
CUDA implementation achieves; it does not establish a minimum PRO/M4 speed
ratio for the user's different workload. Its eight-GPU and GGUF rows do not
describe our single-GPU BF16 execution. [Published benchmark](https://github.com/QuixiAI/h3.c#rtx-3090s-vs-an-m3-ultra).

Read-only inspection confirmed the live CLI process ran from `/path/to/h3.c` with the
user's dimensions, frame count and steps. No relevant reference-mode, sampler,
weight-mode, memory-budget or tile overrides were present. One GPU snapshot
showed 100% activity, 540.52 W against a 600 W limit, 2,422 MHz SM clock, 67°C,
and 45,187 MiB allocated. This rules out an obviously idle GPU in that snapshot;
it does not measure tensor-core efficiency or establish sustained clock behavior.
No benchmarks or profiler attachments were run against the busy device, and no
inference code, binaries, settings, or running jobs were changed for this review.

The completed 864×480, 141-frame, 20-step, reuse-one qualification provides the
available detailed timing evidence. These figures are **not** a profile of the
user's longer `test2.sh` workload:

| Work in the completed qualification | Seconds | Share of 554.61 s wall time |
| --- | ---: | ---: |
| DiT attention | 357.97 | 64.5% |
| DiT linear GEMMs | 35.12 | 6.3% |
| Remaining DiT phase time | 19.78 | 3.6% |
| VideoVAE attention | 78.52 | 14.2% |
| VideoVAE linear GEMMs | 18.37 | 3.3% |
| Remaining VideoVAE phase time | 4.09 | 0.7% |
| Time outside those two phases | 40.76 | 7.3% |

Kernel categories use CUDA events; the two residual rows within phases are
phase wall time minus those categories, not individually measured kernels.
Outside-phase time includes conditioning, audio, loading, mux and other host
work and should not be attributed to any one operation without a fuller trace.
The decoder's reported host wait overlaps GPU work and must not be added again.
Evidence: [full profile](../outputs/cuda-validation/pro6000/tuned/full.log),
[run metadata](../outputs/cuda-validation/pro6000/tuned/full.json), and
[derived metrics](../outputs/cuda-research/optimization-review-metrics.json).

The following work is ordered by expected effect on the unchanged BF16 workload.

1. Replace the long-sequence DiT attention kernel with a well-pipelined fused
   implementation. [Our current kernel](../../src/cuda/cuda_attention.cuh) processes 16
   queries against 16 keys per tile. Only one of four warps performs the QK
   matrix multiplication. Sixteen threads compute the softmax rows, and the
   weighted-value product uses ordinary F32 `fmaf` instructions. Four block-wide
   barriers occur per key tile, and K/V loads are not asynchronously pipelined.
   The fixed-dimension SM120 specialization preserves this structure.

   Candidate changes are 64/128 query rows per block, 32/64 key rows per tile,
   work distributed across multiple warps, register-resident softmax state,
   asynchronous double-buffered K/V loads, and tensor-core products for both
   QK and PV. The Quixi implementation uses a 128-query/64-key organization
   with asynchronous K/V staging and reports 46.9 effective TFLOP/s at
   S=9,720, H=56, D=128 on RTX 3090. [Optimization notebook](https://github.com/QuixiAI/h3.c/blob/69172740de9cf1bb12c8718479cefe47eeaf7a19/perf/optimization_status.md#sdpa-dit-self-attention).

   Our saved S=18,225, H=56, D=128 benchmark takes 328.47 ms, equivalent to
   approximately 29.0 TFLOP/s under the same `4*S*S*H*D` FLOP convention.
   Different sequence sizes and intermediate arithmetic prevent treating this
   as a controlled GPU speed comparison, but it supports investigating our
   kernel structure. [Our warmed benchmark](../outputs/cuda-validation/pro6000/tuned/sm120-bench.log).

   There is a numerical distinction: Quixi converts softmax probabilities to
   BF16 before tensor-core PV, whereas our current kernel retains F32
   probabilities. F32 accumulation alone does not make these paths identical.
   Larger tiles and reduction trees can also change rounding. A replacement
   must preserve our BF16-rounded attention scale, QKV/output layouts, masks,
   and F32 accumulators, and must undergo full numerical and visual qualification.
   Keep the current exact path available. Do not silently loosen existing gates
   or advertise byte identity for a new arithmetic path. [Quixi PV implementation](https://github.com/QuixiAI/h3.c/blob/69172740de9cf1bb12c8718479cefe47eeaf7a19/h3_cuda_kernels.cuh#L2029).

   A useful control is cuDNN SDPA through its native C++ API: the documented
   consumer-Blackwell forward path supports BF16 and our 128-wide heads. This
   can establish a library baseline without adding Python inference. cuDNN 9
   shared libraries are already discoverable on this server; the installed
   version, frontend headers and compatible execution plans still need checking.
   An optional backend would add a dependency relative to our current design.
   NVIDIA's Transformer Engine also prefers eligible cuDNN attention on SM120,
   but that is a reason to measure it, not a guarantee for these exact shapes.
   [cuDNN API/support](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html),
   [NVIDIA backend selection](https://docs.nvidia.com/deeplearning/transformer-engine/examples/attention/attention.html#backend-selection).

   A native C++ FA2-style kernel is the alternative if keeping the existing
   dependency set is important. Target SM120 explicitly: PRO 6000 is not B200
   SM100. Current upstream FlashAttention has an SM120 implementation that
   reuses the SM80 MMA path with a 99-KiB shared-memory constraint; that particular
   implementation is CuTe DSL/Python and is reference material, not a direct
   dependency-free drop-in. [Upstream SM120 implementation](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm120.py),
   [NVIDIA architecture limits](https://docs.nvidia.com/cuda/blackwell-tuning-guide/).

2. Optimize VideoVAE attention while retaining its F32 numerical contract.
   [The decoder](../../src/vae/video_vae.c) uses 32 heads of dimension 64 and F32 Q/K/V.
   [CUDA dispatch](../../src/cuda/gpu_cuda.cu) sends every F32 attention call to the
   scalar online kernel, so the existing SM120 BF16 optimization never reaches
   it. Attention accounts for 78.52 of the decoder's 100.98 seconds. This decoder
   is a transformer: optimizing Conv3D alone will not solve this bottleneck.

   First compare a query-tiled F32 kernel with bounded, batched F32
   QK-GEMM → softmax → PV-GEMM. A default 256-pixel tile has S=1,797; storing
   scores and probabilities for all 32 heads costs about 0.77 GiB. That is a
   practical experiment on the 96-GB device, with head batching and a memory
   budget for smaller cards. Do not apply this quadratic-storage route to the
   long DiT sequence. Keep F32 inputs, probabilities and accumulation, with
   no silent TF32/BF16 decoder conversion; matrix reduction order still needs
   tolerance testing. Quixi already implements a bounded GEMM attention path
   for F32. [Their dispatch and GEMM route](https://github.com/QuixiAI/h3.c/blob/69172740de9cf1bb12c8718479cefe47eeaf7a19/h3_gpu_cuda.cu#L2808).

   After the kernel improvement, test batching independent spatial tiles or
   temporal chunks using shared resident weights and bounded activation arenas.
   Currently they execute serially, with a submit/readback per tile. Preserve
   the existing tile dimensions, overlap, coordinate normalization and stitching
   order. Bigger tiles change decoder context and are not an automatically
   equivalent speed improvement. Moving unpacking/stitching to the GPU and
   overlapping readback/mux are secondary candidates, not measured wins here.

3. Remove redundant Q/K normalization work. In
   [h3_qkv_rope_bf16](../../src/cuda/cuda_kernels.cuh), every output dimension recomputes
   both complete head RMS sums. For D=128, that repeats each reduction 128
   times per head. Use a warp/cooperative group per row and head, cache Q/K,
   calculate each inverse RMS once, broadcast it, then apply RoPE. An initial
   version can preserve the serial FMA reduction order to aim for exact output.
   This is a concrete code inefficiency even though its individual share is
   not isolated by the current profile. Measure it separately before claiming
   an end-to-end saving. Inspect similar vision/text head-normalization kernels
   only after establishing which ones matter to the target workload.

4. Tune GEMM choices and host overhead after attention. Our cuBLASLt path
   requests eight heuristic results but always selects the first; the selected
   algorithm is cached, while operation/layout descriptors are rebuilt per call.
   Benchmark valid candidates on the actual projection shapes, test workspace
   budgets above the existing 32 MiB, and retain a reproducible per-device
   selection. Cache descriptors with correct bias-pointer updates and lifetime
   management. [cuBLASLt guidance](https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api).
   Halving DiT GEMM time alone would save about 17.6 seconds in the recorded
   render, only about 3.3% overall; it cannot explain or deliver another 2×.

   Review event traffic, allocations and graph capture after an Nsight Systems
   trace establishes launch overhead. The backend records tensor release events
   for every accessed tensor and waits on recorded upload events; a resident-only
   fast path may amortize this, but streamed slot-ready/release ordering must
   remain intact. Do not remove synchronization on the strength of timing alone,
   especially given the previously fixed streaming races. CUDA Graphs require
   stable storage, cached descriptors and explicit callback/checkpoint boundaries.
   Graph reuse can help repeated fixed-shape work; it will not eliminate long
   attention kernels.

The profile does not make weight streaming the first target on PRO 6000: core
weights were resident, streamed block bytes were zero, and DiT source
read/upload wall time was about 2.46 seconds. Larger host buffers, faster disk,
quantized weights, and a different text-encoder residency policy cannot remove
the measured attention time. Qwen GQA and audio convolution can be improved,
but they were small fractions of this completed workload. Reuse, fewer steps,
Turbo LoRA, token reduction, sparse attention and quantization change the
comparison or mathematical execution; they should not be counted as speedups
of the unchanged 50-step/reuse-one BF16 test.

For scale, applying hypothetical improvements to the completed 554.61-second
profile gives the following Amdahl estimates. They are arithmetic scenarios,
not forecasts or measurements of candidate kernels:

| Assumed change | Resulting wall time | Overall speedup |
| --- | ---: | ---: |
| DiT attention 2× faster | 375.62 s | 1.48× |
| DiT attention 3× faster | 315.96 s | 1.76× |
| DiT attention 4× faster | 286.13 s | 1.94× |
| DiT attention 3× and complete VideoVAE phase 2× faster | 265.47 s | 2.09× |
| DiT attention 4× and complete VideoVAE phase 2× faster | 235.64 s | 2.35× |

Thus another 2× is plausible enough to pursue, especially with attention on the
critical path, but the current evidence does not promise 25 minutes for the
user's longer segment. Establish its completed phase profile before projecting
absolute minutes.

The next measurement campaign should use the unchanged `test2.sh` workload and
the completed qualification as controls. First record actual attention shapes,
per-stage times and cold/warm behavior on an idle server. Then benchmark current
attention, cuDNN/native fused candidates, and the F32 decoder candidates on
those exact shapes and real conditioning, including reference and continuation
rows. Use Nsight Compute on isolated kernels for tensor/F32 utilization,
occupancy, register spills and barrier stalls, and Nsight Systems for scheduling.

Before enabling a candidate, run primitive and real-component comparisons,
full 50-step trajectories, all reference modes at realistic settings, changed-
reference continuations, protected-prefix checks, ordinary/masked restart and
cross-backend resume. Inspect complete video and audio outputs as well as latent
errors. Preserve exact restored checkpoints and deterministic same-path restart;
do not confuse those guarantees with byte identity between different attention
algorithms. The two-step `feature-*` smoke videos are insufficient quality
evidence. Promote only measured end-to-end gains that meet the unchanged
acceptance requirements, retaining the current path for comparison and fallback.
