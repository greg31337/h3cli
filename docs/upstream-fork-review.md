Review date: 2026-09-18. Local baseline: `2a4a750` (H200 qualification).
Upstream comparison base: `8974cc055ea9c02fcd14cc27dfda3e1027c05153`.

The most useful immediate work is a small set of media, CLI, cache-lifetime,
and build fixes. The strongest larger candidates are reduced-precision VideoVAE
execution, persistent conditioning caches, runtime adapters, and a low-resolution
generation followed by latent upscaling/refinement workflow. These should be
adapted individually to our CUDA/Metal, continuation, memory, and checkpoint
contracts; none of the larger forks is a suitable wholesale replacement.

Scope: screened all 37 upstream PRs returned by GitHub (32 open, five closed,
none merged), and metadata for 215 publicly listed direct forks. Sixty forks
had pushes after the upstream main timestamp. For 39 active forks outside the
upstream PR submitter set, default-branch comparisons found 22 ahead, 16
identical, and one inaccessible (`lorenzoperrone/h3.c`, HTTP 404). Also inspected
four non-default branches from thomasantony, Alucard24, and KaedeTai and the
four feature PRs in LachlanStuart's fork. This is a broad discovery review with
targeted patch inspection, not an audit of every branch or every fork's tests.
The inventory is retained under `outputs/upstream-review/2026-09-18/`.

Only local host probes were run. No external fork code was executed and no
model render or inference-source change was made. Published performance and
quality claims below remain the authors' evidence until tested here.

**Fixes and small improvements worth considering**

| Priority | Source | Current local finding | Recommendation |
| --- | --- | --- | --- |
| High | [PR #31: FFprobe CSV side data](https://github.com/antirez/h3.c/pull/31) | `h3_ffprobe_visual_size()` still rejects `3840x2160x`. The function is used before Ref2VA image/video preparation. Both a mock output and a real synthetic MOV with a display matrix reproduced the rejection in our compiled helper. | Accept the one optional trailing separator, retaining strict rejection of other junk. Add parser cases and a display-matrix/rotation input case. This handles a phone-video input failure. |
| Medium | [PR #59: older macOS SDKs](https://github.com/antirez/h3.c/pull/59) | We already include `<limits.h>`, but `src/metal/metal.m` still references `MTLGPUFamilyMetal4` under only a runtime availability check. Older SDK headers cannot compile that enum. | Add the SDK preprocessor guard; verify an older SDK build and a current SDK build before expanding documented build support. |
| Medium | [PR #9: non-finite decoded RGB](https://github.com/antirez/h3.c/pull/9) | `unpack_frame_range()` still uses two comparisons that let NaN pass into RGB. | Adopt an explicit non-finite output policy and regression. Prefer an actionable failure/diagnostic for invalid decoder results; silently mapping them to black can conceal the broken rendering this project has previously investigated. Normal finite saturation should remain unchanged. |
| Conditional, important on Ultra | [PR #64: split long MPSGraph queries](https://github.com/antirez/h3.c/pull/64) | The default Metal SDPA graph still submits one full non-causal attention operation. The PR reports M2/M3 Ultra corruption and an M3 Ultra recovery run. | Evaluate query-row splitting on affected hardware. Each query block retains all keys/values, so this is full attention, unlike a temporal window. The proposed 12,288-row gate and 2,048-row blocks need device/OS validation; concatenating graph branches is not a proven peak-memory bound. Do not enable its global heuristic indiscriminately. |
| Low for current workflow | [PR #7: RES audio sigma schedule](https://github.com/antirez/h3.c/pull/7) | The legacy `h3_dit_denoise()` RES implementation still constructs audio updates on the video sigma grid. A host probe reproduces the reported error. Normal generation uses Euler; no production caller of this RES entry point was found. | Fix before exposing or expanding RES. It does not explain current Euler CUDA/Metal output quality. Keep it separate from changes to the accepted sampler. |
| Useful for distribution | [PR #32: embedded Metal shaders](https://github.com/antirez/h3.c/pull/32) | We resolve shader files from the CWD or executable directory, but still require an external file. | Embed the source as a fallback for a standalone binary. Preserve explicit shader override errors and include embedded source identity in compatibility/build hashes. This is additional functionality beyond our existing relative-path fix. |
| Convenience | [PR #33: model-directory environment variable](https://github.com/antirez/h3.c/pull/33) | `src/h3cli.c` requires the existing model selection; no `H3_MODEL_DIR` fallback is present. | Small optional CLI improvement with explicit `-d` taking precedence. |

The FFprobe finding is reproduced locally. A plain synthetic MP4 returns
`64x32` and passes. After adding a 90-degree display matrix with FFmpeg's
`-display_rotation 90` input option and remuxing to MOV, FFprobe returns
`64x32x` and our helper fails with "FFprobe returned an invalid visual size".
The display matrix was independently checked in FFprobe JSON. An initial
`-metadata:s:v:0 rotate=90` attempt did not create side data on this FFmpeg and
therefore did not trigger the bug. Rotation/orientation handling is a separate
concern and is not fixed merely by tolerating CSV syntax.

The host RES probe uses the current schedule and `h3_res_step()` helpers with
constant audio velocity 0.5 and initial sample 1.0. The expected final sample
is 1.5. At 4/7/20 steps the current formulation returns approximately
1.84343/1.62017/1.49493, while using the native audio sigma grid returns 1.5.
This is a solver reproducer, not an audio-quality measurement or a model run.

**Larger features ranked for our workflow**

1. **Reduced-precision VideoVAE execution: the best contained performance
   experiment.** [LachlanStuart decoder PR #1](https://github.com/LachlanStuart/h3.c/pull/1)
   adds FP16 weights/activations and GPU tile/temporal stitching; its
   [encoder PR #4](https://github.com/LachlanStuart/h3.c/pull/4) adds FP16 encoder
   execution, GPU normalization, and batched tile scheduling. Decoder
   model-backed tests were skipped in the published PR environment; the encoder
   is a draft with primitive coverage. Our VideoVAE paths still offer concrete
   opportunities here, including encoder cost for reference video.
   [KaedeTai's `vae-bf16-mlp` branch](https://github.com/KaedeTai/h3.c/tree/404d3bf3e7af995da889f0ef8715723a6f050058)
   offers a narrower BF16 MLP experiment plus optional M5 INT8 FFN work. Its
   reported INT8 decoder gain is specific to that fork/device, not an M4 or
   CUDA result. Start with matched saved-latent decodes and short refvideo
   renders, checking detail, motion, seams, and audio/video continuity; retain
   the current path behind an explicit option.

2. **Persistent text/reference conditioning cache: useful for repeated CLI
   jobs.** [yangweijie/bin/h3cli](https://github.com/yangweijie/h3.c/blob/d52faa123a0be298228b4716b9d1c8586be42c63/h3.c)
   persists raw conditioning across processes and separates it from resident
   model caches. Our ordinary conditioning cache is in-memory; sampler
   checkpoints serve a different purpose. Reusing conditioning can avoid
   repeated encoder loading/execution when changing seeds or sampling settings.
   Adopt the idea with our own versioned, content-based cache identity covering
   model/tokenizer, preprocessing, reference inputs, and execution semantics;
   do not copy the fork's environment/path assumptions. Measure complete
   repeated-job time, not denoising speed alone.

3. **Runtime LoRA switching: substantial usability improvement.**
   [neokree's implementation](https://github.com/neokree/h3.c/blob/d5b6f48c4467421ad9ed73ce3339705a7ab402ca/docs/lora.md)
   supports adapter sets and strengths with an additive low-rank branch,
   including streamed weights and AdaLN targets. We currently fold adapters
   offline. Runtime application would avoid preparing a new model variant for
   each strength/style combination. It adds per-forward computation, so it is
   not inherently faster than an already-folded model. The fork's measured
   machine is M4 Pro; its documentation explicitly excludes the M5 fused QKV
   integration. We would need CUDA, fused-operation, cache, and checkpoint
   integration. Preserve mixed ranks and alpha scaling, and reject unsupported
   adapter targets rather than silently ignoring them.

4. **Low-resolution generation, learned latent upscaling, and refinement:
   promising for high-resolution previews/production.**
   [LachlanStuart's upscaler PR #3](https://github.com/LachlanStuart/h3.c/pull/3)
   supplies latent interchange and an external PyTorch/MPS upscaler sidecar;
   its fork main adds restart refinement. The PR explicitly does not include
   the restart sampler. The
   [yangweijie workflow](https://github.com/yangweijie/h3.c/blob/d52faa123a0be298228b4716b9d1c8586be42c63/README.md)
   also exposes latent export/import and refinement. This could reduce expensive
   full-resolution steps, particularly for the user's larger canvases. It is a
   different quality/speed workflow, not ordinary RGB resizing or exact sampler
   resume. Design an explicit new-stage handoff that carries audio, geometry,
   references, and model identity. No whole-operation gain is established here.

5. **Quantized weights: prioritize capacity-constrained GPUs and Macs.**
   [QuixiAI](https://github.com/QuixiAI/h3.c/tree/69172740de9cf1bb12c8718479cefe47eeaf7a19)
   implements native GGUF weight consumption on Metal/CUDA and multi-GPU
   sequence parallelism. This is the same fork revision inspected during our
   previous CUDA review; its attention work already informed our fast path.
   GGUF and multi-GPU remain separate new capabilities. Quantization is not
   automatically faster: its published Q2_K 10-second render is slower than
   its BF16 counterpart. The largest likely local benefit is fitting more
   weights on a 3090/5090, with quality review required. Multi-GPU is worthwhile
   only if we intend to operate multiple GPUs; it does not improve one H100.

6. **Prequantized INT8 disk cache: M5 memory/startup work.**
   [PR #55](https://github.com/antirez/h3.c/pull/55) streams precomputed attention
   and optionally MLP INT8 weights instead of quantizing them every startup.
   The author reports reduced DiT residency, additional disk use, and cases
   where streaming makes short clips slower. This is not immediately an M4 or
   CUDA speedup. Its model-ID mismatch is only a warning and the identity uses
   paths/sizes/mtimes; our adapter/checkpoint workflow needs stronger cache
   invalidation before adopting the format.

7. **Adaptive reuse and long-context GQA: targeted experiments.**
   [yangweijie's first-block cache](https://github.com/yangweijie/h3.c/blob/d52faa123a0be298228b4716b9d1c8586be42c63/h3_dit.c)
   decides whether to reuse the remaining core from a first-block residual
   change metric. This is an adaptive alternative to our fixed reuse interval,
   with quality drift and extra activation storage; the fork disables it for
   SSD streaming and token-reduced steps. Separately,
   [Alucard24's long GQA kernel](https://github.com/Alucard24/h3.c/blob/043460e8a9aecfb9a5dc37de4073e7ea594a5015/h3_cuda_kernels.cu)
   demonstrates bounded-memory online softmax. That design could help lift
   our Metal reference/prompt preflight limit. Do not import the kernel wholesale:
   it uses legacy BF16 query scaling, and the inspected short-sequence branch
   still lacks our barrier between reading the maximum and reusing reduction
   scratch. Preserve our scaling modes and race fix.

For larger work, use the existing short render budgets and compare video/audio
quality. Kernel correctness, memory safety, and output/state integrity remain
required; numerical agreement between fast and default renders is not the
acceptance gate. Fork speed claims do not replace matched local measurements.

**Already addressed here, or lower priority**

| Upstream work | Local disposition |
| --- | --- |
| [#63 GQA scratch race](https://github.com/antirez/h3.c/pull/63), [#44 scratch alignment](https://github.com/antirez/h3.c/pull/44), [#4 query scaling](https://github.com/antirez/h3.c/pull/4) | Already covered: the barrier applies to reference/scaled-query/legacy variants; score storage is 16-byte aligned; scaling semantics have explicit modes. Do not regress them by copying an older fork kernel. |
| [#56 control tokens](https://github.com/antirez/h3.c/pull/56) | Seven tokens already registered in both Objective-C and portable tokenizers. |
| [#48 SafeTensors overlap/gaps](https://github.com/antirez/h3.c/pull/48) | Global payload-layout validation already rejects overlaps, gaps, and unindexed trailing bytes. |
| [#11 frame-alignment overflow](https://github.com/antirez/h3.c/pull/11) | Overflow is already guarded; our helper saturates at `INT_MAX`, and generation rejects out-of-range requests. The PR's zero sentinel is a different API contract, not a missing overflow fix. |
| [#14 Turbo folding, shader lookup, alignment, tiling](https://github.com/antirez/h3.c/pull/14), [#1 VAE tiles](https://github.com/antirez/h3.c/pull/1) | Covered by our adapter tooling, shader resolver, safe unaligned loading, and VAE tile work. Do not reintroduce blanket rejection of valid unaligned files. |
| [#2 resume](https://github.com/antirez/h3.c/pull/2), [#12 continuation scripts](https://github.com/antirez/h3.c/pull/12) | Our sampler checkpoints and continuation/bridge workflows already cover the central goals. |
| [#34 denoised preview](https://github.com/antirez/h3.c/pull/34), [#35 decoder progress](https://github.com/antirez/h3.c/pull/35) | Already implemented locally, with our broader preview/progress handling. |
| [#60 multi-frame preview](https://github.com/antirez/h3.c/pull/60), [#61 preview v2](https://github.com/antirez/h3.c/pull/61) | Closed, unmerged. Motion previews are a possible UI improvement; additional frame delivery/encoding is not free even when the VAE already decoded the chunk. |
| [#19–#24 CUDA port series](https://github.com/antirez/h3.c/pull/24), [#43 GB10 CUDA/studio](https://github.com/antirez/h3.c/pull/43) | Our qualified CUDA path is substantially beyond the incremental port series. GB10 is separate hardware work; the studio component is the interesting independent feature. #22 is model-download documentation, already covered by our build/model guidance. |
| [#57 window slabs](https://github.com/antirez/h3.c/pull/57) | Closed because it targeted the wrong repository. It fixes an optional windowed-attention path we do not have; not a missing fix for our dense attention. |
| [#8 M4 docs](https://github.com/antirez/h3.c/pull/8), [#10 callback docs](https://github.com/antirez/h3.c/pull/10), [#40 grammar](https://github.com/antirez/h3.c/pull/40), [#53 integrations](https://github.com/antirez/h3.c/pull/53) | Documentation/convenience items, not new inference improvements; assess wording against our extended API if adopted. |

[skaiy/h3cli-studio](https://github.com/skaiy/h3.c-studio) is worth studying if a
GUI is desired: persistent jobs, storyboard shots/takes, reference snapshots,
and an explicit preview-versus-adoption distinction fit our continuation
workflow. [PR #45](https://github.com/antirez/h3.c/pull/45) offers a macOS GUI
and phone-image normalization; [matrixfede's PR #43](https://github.com/antirez/h3.c/pull/43)
offers a web studio. A wrapper around our existing CLI would preserve the
qualified engine and avoid merging another backend.

Other backend forks are research references rather than immediate speed wins.
[maderix's ANE fork](https://github.com/maderix/h3.c-ane) explicitly calls itself
a proof of concept; it uses private ANE APIs, quantized weights, and externally
prepared conditioning. [thomasantony's VDN branch](https://github.com/thomasantony/h3.c/tree/vdn-metal-port)
and [zihaomu's ROCm work](https://github.com/zihaomu/h3-vdn.c/tree/vdn-h3-rocm)
introduce different model/backend requirements. The ROCm fork explicitly
withdraws its OpenVDN stable claim and reports unresolved generated-video
correctness; its separate original-H3 work must not be conflated with that
claim. [Alucard24's Vulkan branch](https://github.com/Alucard24/h3.c/tree/vulkan-backend)
is relevant if non-CUDA Linux hardware becomes a goal. None establishes a new
speedup on our qualified M4/H100/H200 setup.

Suggested order: fix FFprobe/piped completion/build guard, then split cache
invalidation and define invalid-RGB handling; test Ultra query splitting when
affected hardware is available. For performance, start with a bounded VideoVAE
precision experiment and persistent conditioning, then design runtime adapters
and a separate upscale/refinement workflow. Quantization, adaptive reuse, and
multi-GPU deserve their own measured, opt-in projects.
