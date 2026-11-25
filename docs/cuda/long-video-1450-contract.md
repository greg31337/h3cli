# CUDA duration and long NVFP4 contract

Native CUDA video supports up to **1,450 aligned frames at 24 FPS**, with BF16
or opt-in `--cuda-denoise-quant nvfp4`. Metal remains limited to 362 frames.
The native CUDA canvas ceiling remains 1920×1088, including transpose.
Sequence and actual host/device memory admission apply independently.

Alignment remains `5 + 17*n`, rounded upward. `--seconds 60` or `--frames 1440`
produces 1,450 frames (60.416667 seconds). Request 1,433 frames for the largest
aligned duration strictly below 60 seconds (59.708333 seconds). Counts above
1,450 fail; frames are never truncated or repeated to satisfy the limit.
Reference media retain their separate 2–15-second input rules.

## Supported combinations

| Path | BF16 | NVFP4 |
| --- | --- | --- |
| Native CLI/API/HTTP/`--render` | Supported | Supported on qualified SM120/cuBLASLt |
| Prompt-only, first/last anchors, Ref2VA | Supported | Supported |
| Hard/bridge continuation | Supported | Supported |
| Sampler pause/resume, AV full/preview replay | Supported | Supported with precision provenance |
| `.h3up` source capture and learned upscale | Supported within sequence/memory bounds | Rejected |
| AV K=0 transfer and `#upscale` | Supported within sequence/memory bounds | Rejected |

Extended duration/canvas requests use all 50 blocks, default dense attention,
reuse/core-reuse 1 and existing conditioning exclusions. FP8, Sage, SOL,
SubBlock, adaptive cache, token reduction, runtime LoRA and layer thinning
remain unavailable for these extended combinations. Their existing supported
short-video combinations are unchanged. BF16 remains the default.

NVFP4 requires SM120 and cuBLASLt 12.8 Update 1 or newer. An unavailable kernel
fails explicitly; it never changes the requested precision to BF16. Packed
weight placement uses `H3_CUDA_WEIGHT_MODE=auto|resident|stream`; automatic
placement retains its allocation-failure retry. Streaming retains compressed
host weights and uploads packed operands directly. Model files stay immutable.

## Temporal and state layout

At the maximum, video has 427 latent times and audio has 2,417. Bridge class
arrays contain 427 video and 2,418 audio bytes. Continuation contexts remain
`39 + 51*k`, no longer than the source and strictly shorter than the target;
the maximum CUDA context is 1,416 frames. Prefix masking, bridge velocity,
suffix trimming and coupled audio timing retain their existing contracts.

AV schema 3 and sampler/source envelope 2 remain unchanged. Required temporal
section 15 has fixed versioned wire extents:

| Effective frames | Version | Video/audio class bytes |
| --- | ---: | ---: |
| Through 362 | 1 | 107 / 604 |
| 363–736 | 2 | 217 / 1,228 |
| 737–1,450 | 3 | 427 / 2,418 |

Readers check duration/version/length before allocating latent payloads.
Inactive class tails must be zero. Old readers reject unknown required
versions. Metal can inspect portable AV metadata but rejects execution of
long states, including indirect decode/resume/continuation/upscale routes.
Same-build checkpoint provenance still applies.

## NVFP4 execution recipe 5

NVFP4 on an extended duration or canvas selects execution recipe 5. The old
short-video recipe remains 2 (or its existing explicit composition). Packed
weight layout/scaling and metadata cache identity remain recipe 2.

Recipe 5 uses at most 8,192 rows per QKV, attention-output, FC1 or FC2 GEMM.
Each row retains its entire K dimension. One reduction over the complete input
supplies the tensor-wide activation amax to every chunk. VEC16 UE4M3 scales
and 128-row padding are local storage layouts with that shared global factor.
Tail output scratch is bounded by one slab.

The MLP first computes FC1/SwiGLU in slabs and reduces the activated values
over every row. It then recomputes each slab and applies FC2 using that global
maximum. This adds FC1 work while bounding temporary storage. Full temporal
attention, normalization and model depth are preserved. Recipe 5 is recorded
in sampler and presentation metadata and included in prepared-resource keys.
AV replay needs the decoder, not packed transformer weights.

## Admission and verification scope

The logical sequence ceiling remains one million rows, including text, audio
and references. Maximum padded NVFP4 layout is checked separately. All actual
bounded GEMM descriptors are queried before denoising. Generated AudioVAE
convolutions are queried before extended generation: all 8,084 descriptors
across 86 aligned durations fit the 3 GiB workspace cap. Peak layout storage is
123,750,400 bytes per adapter; its checked cap is now 128 MiB.

The [capacity report](long-video-1450-memory-estimates.md) distinguishes host
estimates from measured runs. Admission uses current free memory and reserves
`max(1 GiB, physical VRAM / 10)` once. Supporting the maximum frame count does
not promise every canvas fits a 72 GB card. A maximum-duration 3840×2176
upscale has 3,484,320 video rows alone and fails the sequence limit even at K=0.

Runtime qualification is restricted to 256×256 and at most 464 frames
(19.333 seconds), two evaluations, all 50 blocks. Larger durations/canvases use
metadata and descriptor checks only. The immutable 204-output regression and
short BF16 256→512 upscale checks retain their separately approved exceptions.
**No end-to-end 1,450-frame run is claimed.** Motion and scene consistency are
outside this acceptance scope. See the [results](long-video-1450-results.md).

The completed [736-frame contract](long-video-contract.md) and associated
archives remain historical evidence.
