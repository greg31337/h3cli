# Uniform reference-image sizing and capacity

Status: **implemented and validated**. Completed tasks are in
[the archived checklist](reference-image-size-tasks.md), with the
[validation record](reference-image-size-results.md).
This design covers `--ref-image-size` for Ref2VA images
on Metal and CUDA, for both video and still generation.

## Behavior

Keep `match` as the default. Make `max` use the existing CUDA-video geometry
on every backend/output path, and introduce `high`:

| Setting | Scale before alignment | Upscaling | Dependence on output canvas |
| --- | --- | --- | --- |
| `match` | `min(1, sqrt(render_width * render_height / (source_width * source_height)))` | No, except existing 32-pixel alignment/minimum | Uses the internal render canvas, falling back to requested output dimensions |
| `high` | `2048 / max(source_width, source_height)` | Yes | None |
| `max` | `2048 / min(source_width, source_height)` | Yes | None |

For `high` and `max`, require positive source dimensions and an aspect ratio
within the inclusive range 1:4 to 4:1. Round each scaled dimension to the
nearest multiple of 32, with ties to even and a minimum of 32. Implement the
rounding explicitly so it does not depend on the process floating-point
rounding mode. Preserve the existing `match` sizing and admission behavior;
do not add the fixed-edge aspect-ratio restriction to `match`.

Apply one selected policy to every `--ref-image` in request order. Preserve
aspect ratio subject to grid rounding; do not crop or pad. Both the Qwen vision
encoder and visual VAE consume the same resolved canvas and prepared image.
Geometry must not branch on backend, CUDA arithmetic selection, or still/video
output. Existing backend pixel decoding/resampling and encoder arithmetic stay
in place; identical geometry is not a claim of identical pixel/tensor values.

Examples, with a 640×480 internal render canvas for `match`:

| Source | `match` | `high` | `max` |
| --- | --- | --- | --- |
| 640×480 | 640×480 | 2048×1536 | 2720×2048 |
| 1365×1821 | 480×640 | 1536×2048 | 2048×2720 |
| 32×32 | 32×32 | 2048×2048 | 2048×2048 |
| 4000×1000 | 1120×288 | 2048×512 | 8192×2048 |

`high` bounds the long edge to 2048. `max` fixes the short edge at 2048 and can
produce a long edge of 8192. Both can enlarge small inputs. This intentionally
changes Metal and CUDA-still `max`, which previously avoided upscaling.

The option continues to affect only image references. First/last-frame anchor
fitting, reference-video sampling/sizing, audio, output resolution and the
still/video execution recipes keep their existing meanings. Existing limits of
nine images and twelve total reference entries remain; changing reference-count
policy is separate from removing the artificial aggregate patch limit.

## Baseline implementation and SGLang evidence

Before this change, the split was in [engine.c](../../src/engine.c): CUDA-video `max` calls
`h3_sglang_reference_image_canvas`, while other cases use the down-only helper
in [host.c](../../src/host.c). Still rendering does not select the CUDA-video
recipe, so it took the latter path even on CUDA.

The capacity investigation checked installed SGLang 0.5.20 and upstream commit
[`b252aceffecd1e313cb5a03d3cbf56c99fc8c9ce`](https://github.com/sgl-project/sglang/tree/b252aceffecd1e313cb5a03d3cbf56c99fc8c9ce).
Its H3 resolver uses a 2048-pixel short edge, allows upscaling and restricts
aspect ratio to 1:4–4:1. The released image processor has a maximum area of
16,777,216 pixels, 16-pixel patches and 2×2 spatial merging. These give a
maximum of **65,536 raw vision patches per image**, or **16,384 merged tokens**.
[Resolver source](https://github.com/sgl-project/sglang/blob/b252aceffecd1e313cb5a03d3cbf56c99fc8c9ce/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py#L125),
[processor configuration](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/5d9b308/Ref2VA/processor/preprocessor_config.json).

The inspected reference-image batching path has no 32,768 aggregate patch cap.
Its Ref2VA request profile also leaves the condition-count maximum unset;
h3cli's count limits remain its own contract.
[SGLang task profiles](https://github.com/sgl-project/sglang/blob/b252aceffecd1e313cb5a03d3cbf56c99fc8c9ce/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py#L187).

A CPU-only probe of the installed processor accepted two 2720×2048 images
(43,520 patches total) and one 8192×2048 image (65,536 patches). This establishes
preprocessing behavior, not full-render memory capacity or numerical parity.

The previous 32,768 guard appeared in [vision_encoder.c](../../src/conditioning/vision_encoder.c) and
the CUDA group/patch-projection functions in [gpu_cuda.cu](../../src/cuda/gpu_cuda.cu).
It bounds expanded BF16 input packing to `32768 * 8192 * 2 = 512 MiB`, plus
18 MiB for the padded filter. It is an implementation scratch limit, not an
H3 image-size limit. Unrelated occurrences of `32768` are outside this change.

## Shared geometry and API integration

Introduce one host-side, checked resolver that accepts the sizing mode and
source/render dimensions. Put geometry in `host.c`/`host.h`, independent of
SGLang runtime selection. Route all production reference-image preparation
through it. Keep an existing helper as a thin compatibility wrapper where
needed by frozen regression probes or historical tools; do not maintain two
separate implementations of `max`.

Keep enum values `MATCH=0` and `MAX=1`; append `HIGH=2`. Update the public API,
CLI parser/help, request validation, saved-state validation, and ordinary tools
that enumerate these choices. Leave SGLang parity tooling unchanged. Reject
unsupported enum values before model work. Do not change the default, numeric
identities of old enum members, or
the shared CUDA arithmetic identity solely for this geometry extension.

Return enough checked geometry information to derive pixel, patch and merged
token counts without repeated unchecked multiplication. Log mode, source and
resolved dimensions, per-image patch count and aggregate image patches at
reference preparation. Errors should identify the reference and distinguish
bad dimensions, unsupported aspect ratio, arithmetic overflow and insufficient
memory. Do not silently shrink, truncate or drop references to fit a budget.

## Patch capacity and execution

Admit up to 65,536 raw patches for each resolved reference image. `max` reaches
that boundary at 4:1; `high` has at most 16,384 patches, at 1:1. A raw image
patch count is `(width / 16) * (height / 16)`; duplicating a still image for the
temporal patch does not double that count. Apply image checks at the image
boundary, without imposing an image-area rule on a combined video batch.

Remove the 32,768 aggregate admission guards. Do not replace them with a
65,536 aggregate guard: two or more valid images may exceed either number.
Use checked aggregate counts, tensor byte sizes, kernel index ranges and the
existing host/device memory protections. For example, nine maximum-area images
have 589,824 raw patches; this is a sizing/admission test, not a promise that
every GPU can render that request. Keep existing reference-count, video-duration
and process-memory safeguards with distinct error messages.

Keep the CUDA packing workspace bounded by processing patch-projection rows in
chunks of at most 32,768. Pack the padded filter once per projection, reuse a
bounded input buffer, and write each chunk to its checked destination offset.
Handle full chunks and a short final chunk, preserving row order and bias
application. Retain the current single-launch path and arithmetic for inputs
that already fit. Chunk only the independent projection rows; preserve the
vision transformer and each image's full attention group, including images
whose own attention sequence exceeds 32,768 patches.

Audit all larger-count consumers: patch packing, position/RoPE preparation,
normalization, QKV/MLP tensor sizing, per-image attention boundaries, merging,
deepstack output slicing, text insertion, VAE conditioning and packed DiT
layout. Use checked `size_t` arithmetic for allocations/offsets and explicit
range checks before narrowing to GPU indices. Preserve ordered `<Picture N>`
binding and prevent attention between separate image groups. Metal and CUDA
still may retain their existing sequential image encoding; uniform sizing and
admission do not require identical batching implementations.

Budget the actual live inputs, activations, outputs, weights and temporary
workspaces, rather than treating the 512 MiB projection scratch as total encoder
memory. Release each temporary at its last consumer and respect asynchronous
GPU lifetime requirements. Failed allocation, cancellation or malformed input
must release partial outputs and leave a reusable context. Do not raise global
memory caps or add a new memory-tuning flag as a shortcut.

### Metal large vision attention

The full 65,536-patch test exposed a stall in MPSGraph D=72 vision attention.
For sequences above 32,768 patches, pad each BF16 head from 72 to 128 with
zero lanes, call the existing bounded dense BF16 kernel with the original
72-dimensional scale, then discard the padded output lanes. Preserve all keys
and the complete per-image attention sequence. Smaller vision requests retain
their MPSGraph path. Complete GPU consumers before releasing padded scratch,
since tensor release marks Metal storage purgeable. Include graph intermediates
and padded buffers in unified-memory admission. This path receives functional,
finite-value and guard coverage without a new numerical-parity comparison.

### Metal long-sequence text attention

The consumer audit found a second resource limit: the original Metal Qwen GQA
kernel stores one full attention-score row in threadgroup memory (7,936 entries
on the qualification M4). Larger image presentations therefore need a bounded
path as well as larger vision admission. Keep the original kernel for sequences
that fit. Above that boundary, use 512-key tiles, FP32 online softmax and value
accumulation, preserving causal attention across the complete sequence. Scale
the completed FP32 QK dot before softmax in reference mode. Keep the existing
explicit scaled-Q/legacy modes. Validate completion, finite outputs, guards and
cancellation; do not add numerical comparison with another backend or model.

The public GQA limits expose both the direct scratch boundary and the total
query-index limit. Text MLP/shape checks and ordinary memory protection can be
more restrictive than the attention limit. No device/process cap is raised.

## Cached conditioning, saved states and upscaling

The sizing enum already participates in the in-memory conditioning key,
persistent conditioning identity and sampler parameters. Extend validation and
serialization to admit `HIGH=2`. Add a geometry-policy identity for affected
image conditioning so legacy Metal/still `max` caches cannot be reused as new
uniform `max` results. Existing CUDA-video `max` geometry remains valid; avoid
invalidating unrelated text-only or `match` state.

Prepared sampler checkpoints retain their stored conditions and resolved
geometry on exact resume; do not resize or reopen their references. A legacy
prepared `max` state may resume using its recorded geometry, but must not seed
a fresh request's new-policy conditioning cache. Test legacy/new policy
distinction explicitly. If additional serialized metadata is required, use a
versioned optional record with a documented legacy interpretation, retaining
existing enum values and rejecting unknown values.

For saved upscaling sources, both `high` and `max` are intrinsic image canvases:
preserve their stored reference geometry when the output canvas doubles.
`match` keeps the current target-dependent retargeting behavior. Test high-mode
source save/load and planning, and preservation of legacy stored canvases.
Do not require original media to be reopened by resume or an upscale job.

## Validation contract

This work adds functional, geometry, capacity and lifetime coverage. It does
**not** add SGLang cases, regenerate oracle captures, compare new embeddings or
latents, measure decoded-media parity, or perform Metal-versus-CUDA numerical
comparisons. Historical R0/R1 qualification is background evidence only.

The existing regression remains mandatory and unchanged: all **204 recorded
output hashes** in `tests/cuda_reference/manifest.json`, the same 12 input
fixtures, runner, probes, cases and deadline. Run the complete gate on the
qualified RTX PRO 5000 environment after each coherent source/build/test-tool
change, following [CONTRIBUTING.md](../../CONTRIBUTING.md), and retain a final
204/204 pass. This explicitly required existing numerical gate is the only
numerical-parity qualification in this work. Do not expand it or regenerate
goldens to accommodate a change. Documentation-only changes need link/document
checks and do not claim a GPU test pass.

Additional tests belong in separate ordinary test targets:

| Layer | Required coverage and assertions |
| --- | --- |
| Host geometry/API | All modes; portrait/landscape/square; up/downscaling; unchanged `match`; target versus internal render canvas; 4:1 boundaries; ties-to-even; invalid/overflow dimensions; CLI/API rejection; backend/output-independent geometry |
| Capacity planning | 32,768 no longer an aggregate limit; 65,536 per-image boundary; 43,520-patch two-image batch; 49,152-patch three-square batch; aggregate above 65,536; nine-image maximum counts; overflow and resource-budget rejection |
| GPU projection/vision | Chunk-boundary/tail writes, including rows 32,764/32,768/32,772 and 65,540; per-image groups and output sizes; finite values; output guards; no unwritten rows or out-of-bounds access; full attention sequence for a 65,536-patch image |
| Cache/state | Mode and policy identity separation; high-mode round trips; old enum compatibility; unknown-value rejection; prepared resume preserves geometry; `high`/`max` upscale references remain fixed |
| Lifecycle | Request order retained; small/large/small requests; alternating policies; injected allocation failure; cancellation and recovery; repeated-context cleanup |

Use deterministic generated fixtures under ignored `outputs/`, without changing
the recorded parity fixtures or production `inputs/`. Host expectations come
from the specified geometry rules, not an imported SGLang executable. GPU
checks assess shapes, completion, finite data and memory safety, not numerical
agreement with another implementation. State round trips check serialization,
not rendered tensor equivalence.

Build and run the following functional matrix on local Metal and the authorized
RTX PRO 5000 CUDA server, using the qualified CUDA 13.0.3/cuDNN 9.20 environment:

| Output | Mode and references | Purpose |
| --- | --- | --- |
| Video | `match`, one 640×480 image | Default behavior/control |
| Video | `high`, two distinct 640×480 images | New policy and ordered multiple references |
| Video | `max`, two distinct 640×480 images | 43,520 total patches, beyond the old batch cap |
| Video | `max`, one 4000×1000 image | 8192×2048 canvas and 65,536-patch single-image boundary |
| Still | `high`, one 1365×1821 image | New policy in still generation |
| Still | `max`, one 1365×1821 image | Still upscaling now uses the shared short-edge policy |

Each row runs once per backend: **12 outputs**. Use dense BF16, full model
depth, a fixed prompt/seed, two denoising evaluations, and no approximate reuse,
adaptive cache or sparse attention. Video output is 640×480, 90 frames at 24
FPS; still output is 640×480 with the normal image VAE. Validate successful
completion, reference geometry/counts, finite saved values where available,
complete media decoding, requested output dimensions/frame count and resource
cleanup. Record time and memory descriptively without performance or perceptual
quality gates. Do not add FP8/NVFP4 or a separate quality/parity campaign.

Run GPU jobs serially per device. Missing hardware/model, OOM, timeout or
incomplete output does not pass a required row. Resolve resource handling for
the selected matrix; retain all failed attempts. Keep connection details and
machine-specific paths outside tracked documents. Native code belongs in
`src/`, build products in ignored `bin/`, and generated evidence in ignored
`outputs/`.

## Completion

Update README/API/help with all three modes, explicit upscaling, rounding,
aspect-ratio and patch-capacity behavior, including the Metal/still behavior
change and cache compatibility. Replace the README's unconditional description
of image references as down-only. Record build, host, functional and unchanged
204-output regression results separately, with their exact scope. Completion
requires both backend builds, all required behavior tests and a final 204/204
golden pass; it does not claim newly established SGLang numerical parity.
