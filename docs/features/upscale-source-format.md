# Clean upscale source container

Native latent upscaling is a regular CUDA/Metal feature. See the
[qualification record](../experiments/latent-upscale-qualification.md),
[accepted comparison](../experiments/latent-upscale-results.md) and
[completed tasks](../experiments/latent-upscale-tasks.md).

Fresh upscale jobs automatically acquire the pinned BF16 network under
`--models-path ROOT` at
`ROOT/latent-upscale/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`.
An explicit `--upscale-model` wins. Use `--download-models upscale` for advance
provisioning or `--offline` to require local files; see the
[model download guide](model-downloads.md). Refinement checkpoint resume restores
the already transferred latent and does not require the learned network.

`.h3up` envelope schema 2 has its own eight-byte magic, `H3UPSRC\x01`. It shares the sampler
container's explicit little-endian header/table representation, per-section
SHA-256, whole-file SHA-256 and atomic temporary-file/fsync/rename writer.
Ordinary `.h3sample` readers reject its magic. Source readers reject ordinary
sampler magic; current completed-sampler import is a separate explicit operation.

The 96-byte header and 72-byte table entries are documented in the
[sampler format](sampler-state.md). Source files are bounded to 2 GiB,
semantic text to 65,536 rows, and combined raw visual/audio conditioning to
1 GiB. Geometry and declared payload counts are checked before tensor
allocations. The source must have completed all its original transitions with
both terminal sigmas exactly zero. State, conditioning and semantic tensors
must be finite.

Required section 45, version 1, has the following LE fields:

| Fields | Representation |
| --- | --- |
| Record version, stage, semantic policy, model identity kind | Four U32 values; clean source uses 1/1/1/1. |
| Source render width, height | Two I32 values. |
| Keyframe count, first/last frame indices | One U32, two I32; unused indices zero. |
| Twelve ordered reference descriptors | Original width/height and frozen semantic width/height, four I32 per entry; audio-only/unused entries zero. |
| Exact clean audio identity | 32 SHA-256 bytes over the F32 audio payload. |

The existing condition sections contain **raw pre-augmentation** arrays in this
container. Their interpretation is determined by the required clean-source
stage. Source text, tags, token/position diagnostics and visual spans retain the
frozen source Qwen view. Reference descriptors, layout and original schedule
remain evidence for retargeting, not permission to reuse prepared high-resolution
execution state. Source and reference paths in provenance are informational;
loading never opens them.

Original noise, prepared GPU tensors, cached velocity/residual histories and
approximation sections are excluded. Unknown optional sections can be ignored;
unknown required sections and required-section version changes fail. Initialized
refinement samplers use the required stage extension below.

Capture runs after denoising and before VAEs or media delivery. An error during
preview delivery does not remove the already saved source. `--state-only`
requires a source export and skips preview/VAEs/mux. Runtime sampler memory
and serialized source arrays have independent ownership after loading. A failed
atomic replacement leaves the previous complete file intact.


All AV states use schema 3 of the 160-byte AV header. Offset 120 contains
U32 geometry profile (0 ordinary, 1 upscale); offset 124 remains zero. Profile 1 permits 2,088,960 pixels,
32-pixel alignment and at most 1,920 pixels on either axis. Samplers carry a
required section 46 with I32 profile 1, parsed before geometry-dependent
allocations. Ordinary generation and continuation retain profile 0.

Presentation v8 records `upscale_profile GEOMETRY METADATA_IDENTITY`, followed
by `upscale RECIPE STEPS SIGMA PARENT_SHA256 ARTIFACT_SHA256`. Recipe zero and
`none` identities identify source capture or the direct validation baseline.
Completed recipe 1 identifies learned transfer; recipe 2 is the explicit
bilinear comparison. Metadata identity 1 uses the existing local file-metadata
model contract without scanning H3 weights. Moving a source bundle/reference
media is supported; moving/changing the model installation changes that local
identity and is deliberately rejected. Older schemas are unsupported.

## Refinement checkpoint stage

Refinement uses ordinary `.h3sample` magic with **required section 45 version 2**.
Its prefix is the source record above, with record version 2 and stage 2. The
following fields are appended:

| Fields | Representation |
| --- | --- |
| Recipe and total refinement transitions K | Two U32 values. |
| Starting video sigma | Canonical F32. |
| Refinement seed | U64. |
| Parent source, transfer artifact, target noise, transformed conditioning | Four 32-byte SHA-256 identities. |

K is 2, 3 or 4 and starting sigma is finite in `(0,0.5]`. With
`q0 = sigma/(12−11*sigma)`, the interior flow coordinates are
`qi = q0*(1−i/K)` and video sigmas are `12*qi/(1+11*qi)`, evaluated in F64 and
stored as F32. The first sigma is the exact supplied F32 and the last is zero.
All audio sigmas are zero. Rounded video sigmas must strictly decrease.

Section 21 stores the fresh F32 video noise. Audio noise is absent and its draw
count is zero. The entire clean audio prefix remains visible to joint attention;
there are no audio Euler writes. Its payload hash is checked at initialization,
every completed transition, validation and recovery. The transformed conditions
are augmented once before saving boundary zero. The immutable condition identity
also covers semantic values/tags, positions, segment/reference descriptors and
keyframe indices. Prepared cache keys include the required stage record.

Resume needs the matching H3 model installation and original backend/runtime;
it does not need the source bundle, reference media or upscaler artifact. It
restores noise and sigmas without new draws or transfer. A cancellation can save
the latest committed boundary to the requested sampler path; atomic replacement
preserves the previous checkpoint if recovery cannot be published. Earlier
readers reject required section version 2 instead of interpreting audio sigma
zero as an ordinary generation schedule.

K=0 creates a completed clean AV result directly from transferred video and
unchanged audio. It consumes no random draws and executes no DiT evaluations;
it has no refinement sampler state and rejects explicit noise/stop controls.
