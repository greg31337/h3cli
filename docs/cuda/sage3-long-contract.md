# Sage3 native CUDA videos through 1,450 frames

`--cuda-attention sage3` selects Sage3 independently of projection precision.
It supports native CUDA generation through 1,450 aligned frames with BF16 or
`--cuda-denoise-quant nvfp4`, subject to the existing canvas, sequence and memory
limits. Default attention and BF16 projection defaults are unchanged.

The qualified target is SM120 with the native Sage build and CUDA libraries.
Inference has no Python, Torch or Triton dependency. Missing kernels, libraries
or device capabilities fail explicitly before model preparation.

| Request | Supported policy |
| --- | --- |
| CLI, API, HTTP, script workers | Default or Sage3 attention; BF16 or NVFP4 projections |
| FL2VA prompt, first/last anchors, supported Ref2VA references | Same combinations, existing conditioning rules |
| Hard/bridge continuation | Requested attention/projection modes reach every worker; source compatibility rules remain |
| Sampler resume | Saved attention/projection recipes are restored; conflicting overrides are rejected |
| Completed AV full/preview replay | Decoder assets only; no Sage execution |
| Source capture, `.h3up`, AV transfer, learned upscale, `#upscale` | Existing BF16/default-attention restrictions remain |
| Extended FP8, Sage2++, SOL, SubBlock, adaptive caching, LoRA, reduced depth/reuse | Rejected |
| Metal | Existing 362-frame limit; CUDA attention options rejected |

Extended requests keep all 50 blocks and reuse/core-reuse 1. Alignment remains
`5 + 17*n`: `--seconds 60` aligns to 1,450 frames (60.416667 seconds); 1,433
frames is the largest aligned result strictly below 60 seconds. The canvas
ceiling remains 1920×1088, including transpose. The one-million-row limit
includes text, audio and references. Reference durations retain their existing
independent limits. Actual host and device admission is always required.

AV continuation uses the requested generation settings. Select
`--cuda-attention sage3` for the continuation as well, or inherit it from the
render script. Sampler checkpoint resume restores the saved selection.

## Arithmetic and workspace contract

Attention recipe 1 / plan 1 remains unchanged. H=56, D=128, BF16 Q/K/V and
BF16 output are retained. Every query sees all logical keys. Head groups and
128-aligned query chunks bound scratch to 512 MiB; no full sequence-squared
correction matrix is allocated. Centering, BF16 rounding, reduction order,
logical padded-key masking and FP4 scales retain the pinned implementation.

The shape-only planner is shared by native CUDA and host tests. Its worst
admitted padded sequence is 1,000,064 rows. It checks the fixed 4 MiB cuBLAS
workspace, fault area, centered/packed K/V, scales, reductions, Q chunks,
FP32 corrections, output and alignment allowance. Correction GEMM dimensions
fit signed 32-bit descriptors; source and destination tensor addresses use
64-bit arithmetic. The V packer's source-row multiplication is widened before
multiplication. Its full-sequence stride can exceed signed 32-bit element
offsets even though packed scratch stays within the workspace budget.

Sage scratch is allocated before weight admission. The existing physical
reserve, `max(1 GiB, physical VRAM / 10)`, is applied once. Sage3 is not assumed
to reduce memory: it adds a 512 MiB workspace while retaining BF16 tensors.
The [capacity report](sage3-long-memory-estimates.md) separates estimates from
the bounded measurements.

Extended NVFP4 + Sage3 selects projection execution recipe 5, with global
activation scales and at most 8,192-row projection/MLP slabs. Packed-weight
cache identity remains recipe 2. Short shapes retain their original recipe.
No unrelated projection streaming or MLP recomputation optimization is included.

Sampler section 35 stores attention mode/recipe/plan; section 34 stores
projection mode/recipe. Early readers validate extended combinations before
payload allocation. Presentation metadata and prepared-context identity carry
the same selection. Resume restores it and rejects conflicting overrides.

## Frozen validation scope

Baseline source commit: `f4187a8317d12070a8cd5e5c1085e86a6dcea301`.
Baseline gate source SHA-256:
`64586f9e9bf138ac6da1da50e7b623575df69302439f8c91f859cce2be20fc64`.
Baseline Linux artifact SHA-256:
`c97421d1521b7c7a5773ca1f5099f57633477706d29cf50e5575c6b7a9fa60e6`.
The fresh baseline passed 204/204. The golden manifest is unchanged:
`fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3`.

Runtime evidence is restricted to 256×256, at most 464 frames, two denoising
evaluations and all 50 blocks, on serial GPU jobs below the 60 GiB development
ceiling. Continuation uses 447 source/context frames and a 464-frame target;
the suffix is 17 frames and the final merge is 464 frames. Larger shapes are
metadata/descriptor-only: no production-sized Q/K/V, correction matrix, video
or PCM allocation is permitted. Existing immutable 204-output and bounded BF16
upscale exceptions remain unchanged.

Operator fixtures contain at most 4,096 rows. Test-only forced query/head
limits exercise multiple chunks and final tails without large allocations.
The existing numerical contract is frozen: finite outputs, native-to-pinned
Sage3 relative L2 ≤0.01; zero/constant versus independent blocked FP32 maximum
absolute error ≤0.02. Other FP32 differences are reported separately as
approximation measurements. Guard regions, immutable inputs, both layouts,
nonfinite rejection, exact short parity and bounded sanitizer checks are
required. Tolerances must not be changed to accommodate failures.

Same-build, same-recipe resume/replay and preserved continuation history retain
their exact contracts. Full-generation BF16/Sage3 equality, perceptual quality,
motion and scene consistency are not acceptance criteria. Neither maximum
duration runtime performance nor a 72 GB fit for every admitted shape is
established by these bounded tests. See the [acceptance results](sage3-long-results.md).
