# Development and implementation notes

This page collects the implementation notes and validation links previously
kept in the project README. Measurements describe the hardware and settings
in their linked reports; paths keep their documented restrictions.
Run checkout commands from the repository root.

- [Build and dependencies](build/source-build.md)
- [Source layout](../src/README.md)
- [Rendering controls](usage.md)
- [Contribution requirements](../CONTRIBUTING.md)

## Feature and validation records

Missing supported models download automatically through native libcurl. Use
`--models-path ROOT` (default `models`) for the shared model location,
`--offline` to require local files, `--download-models base,preview` for advance
provisioning, or `--list-models` for local status. See the
[model download guide](features/model-downloads.md).

Run `bin/h3cli --server` for durable queued jobs, polling, cancellation and
artifact downloads. The [server guide](features/server.md) documents
SGLang request fields (including `quality`, mapped to `--quality`) plus one
`h3cli` flag string that overrides them. All native operations share the CLI
execution path. See the [design](features/design-server.md),
[validation report](features/server-results.md), and [completed tasks](features/server-tasks.md).
Server qualification includes small M4 and packaged CUDA renders. The
[PRO 5000 release report](build/linux-distribution-pro5000.md) records current
artifact validation; the [original server report](features/server-results.md)
records the M4 cases.

The [cleanup results](cuda/legacy-cleanup-results.md) document removed
execution options, current-only saved files, builds, sample renders and the
unchanged 204-output SGLang parity gate.

Opt-in CUDA attention acceleration is described in the
[SageAttention guide](cuda/attention.md), including its build and
validation requirements.
Ordinary BF16 CUDA video rendering defaults to the native SGLang-equivalent
pipeline. Select `--cuda-attention sage2++`,
`sage3` or `sol` directly, independently of `--cuda-denoise-quant fp8|nvfp4`.
These explicit options may change visual quality; the default retains frozen
reference parity. Preview VAE remains an independent delivery option.
The [single-pipeline CUDA design](cuda/design-single-pipeline.md) documents
commands, build requirements, state compatibility and side-by-side validation.
`--cuda-weight-mode auto` keeps the full BF16 core when capacity permits, then
uses partial residency with two streaming slots. `resident` requires the full
core; `stream` retains no core blocks. FP8/NVFP4 keep their full-resident or packed
streamed paths. See the [placement contract](cuda/weight-residency-design.md)
and [hardware qualification](cuda/weight-residency-results.md).
See the [M3B/M4 measurements and playback](cuda/single-pipeline-results.md),
[completed implementation checklist](cuda/single-pipeline-tasks.md), and retained
[M3A exact-optimization results](cuda/two-modes-m3a-results.md).
See [the mandatory per-change test rule](../CONTRIBUTING.md).
The native adaptive-cache and SubBlock experiment is specified in its
[design](cuda/design-adaptive-cache-subblock.md) and
[completed task list](cuda/adaptive-subblock-tasks.md), with one 640×480 / 90-frame / 50-step video
for each of twelve comparison variants. Both features are opt-in CUDA/SM120
experiments and preserve the default recorded arithmetic when disabled.
The [latent upscaling feature](features/design-latent-upscale.md)
builds on saved clean AV and conditioning, with native 2× spatial transfer and
optional 2–4 high-resolution refinement steps. CUDA/Metal implementation and
both full-canvas profiles are qualified. The [completed comparison](experiments/latent-upscale-results.md)
contains fourteen videos, measured costs and local playback. The user accepted
the results and promoted latent upscaling to a regular CUDA/Metal feature. See
the [completed task list](experiments/latent-upscale-tasks.md).
The project is being built as a
sequence of working vertical slices: deterministic host/model metadata first,
then portable Metal block parity, prompt encoding, prompt-to-video/audio, and
first/last-frame conditioning and then ordered references.

Prompt-to-video/audio, first/last-frame conditioning, and ordered Ref2VA
image/video/audio references work end to end. The current work is incremental
H3-specific Metal performance and memory optimization on M3 Max and M5 Max.

[single-still generation](features/single-still.md) supports prompts,
ordered image references, saved T=1 latents and direct PNG decoding with an
explicit image VAE. Local Metal codec and bounded generation results are in
the [Metal qualification record](features/single-still-results.md) and
[CUDA qualification record](features/single-still-cuda-results.md).

For Linux builds, device selection, BF16 weight streaming and remote test tiers,
see the [CUDA backend guide](cuda/cuda.md). The [RTX 4090 record](cuda/cuda-qualification.md),
[RTX PRO 6000 record](cuda/cuda-pro6000-qualification.md),
[RTX 5090 record](cuda/cuda-5090-qualification.md),
[RTX 3090 record](cuda/cuda-3090-qualification.md),
[H100 record](cuda/cuda-h100-qualification.md),
[H200 record](cuda/cuda-h200-qualification.md) and
[B200 record](cuda/cuda-b200-qualification.md) list measured results
and hardware coverage. Model, reference, continuation
and sampler logic is shared between the two backends.

## Tests and runtime requirements

```sh
make test
# Host/API and current saved-format checks:
make test-current-host
# Current reference-video preprocessing:
make test-refvideo test-video-posterior
```

Both platforms run current host/API, CLI, serialization and tokenizer tests.
Metal additionally runs native GPU primitives; CUDA runs its host/GPU operator
suite. The archived MLX and pre-SGLang CUDA golden targets are removed. Metal
shaders compile at runtime without Xcode's optional offline Metal toolchain.

On the qualified CUDA build, run `make test-current-cuda` with a fresh output
directory and current AV state to cover real-model integration, explicit
attention/precision, conditioning, placement and decoder recovery. Then run the
complete immutable `make test-cuda-reference-regression` gate: all 204 recorded
SGLang outputs must match exactly. See [the commands and environment](../CONTRIBUTING.md)
and [cleanup qualification](cuda/legacy-cleanup-results.md). Source the
qualified runtime environment before building and testing.

Source builds use FFmpeg/FFprobe 9.0.2 from the setup environment for media
inputs and MP4 output (`H3_FFMPEG`, `H3_SGLANG_INPUT_FFMPEG` and `H3_FFPROBE`
select explicit executables). The standalone Linux package supplies its own
9.0.2 tools. Generated RGB24 and
32 kHz stereo F32 PCM are fed through concurrent pipes; no intermediate
uncompressed media file is created.

## Implementation and performance notes

The remainder documents the implementation behind the tutorial presets and the
environment variables retained for exact A/B diagnosis.

### Sampler and DiT controls

The default sampler uses the released shifted video/audio schedule. `--steps`
always names the number of denoising passes, with terminal zero added after the
last pass. Whole-denoiser reuse evaluates the first and last pass plus every
requested interval, then extrapolates skipped video and audio velocities on
their independent schedules. With very small step counts, keep `--reuse 1`.

For the low-budget path, the released linear base grid won against
actual-video-sigma linear spacing,
quadratic and cubic warps, exact 30-point tail subsets, mild power warps,
zero-order held full-grid velocities, linear velocity extrapolation, and RES.
The more tail-heavy candidates often sharpened the subject but damaged motion
or left a repetitive woven background; sparse RES and long extrapolation
intervals failed much more visibly.

Layer thinning ranks the checkpoint's actual AdaLN gates while protecting
structurally important first and final blocks. Unused weights and schedule
tensors are not retained, so `--layers 45` and `--layers 40` reduce both
transformer time and unified-memory use. Core reuse holds the previous full
transformer residual while refreshing the patch projection and timestep-aware
head; it remains mutually exclusive with whole-velocity reuse.

### Exact DiT fusions

Every active DiT block fuses its attention residual gate with the following MLP
AdaLN. The rounded BF16 residual is still written exactly, but the same row is
kept in threadgroup memory for normalization, eliminating one dispatch and one
global reread. Away from token-reduction boundaries, the MLP residual gate also
produces the next block's attention AdaLN and carries that normalized state
across the loop. `H3_DISABLE_FUSED_GATE_ADALN=1` and
`H3_DISABLE_FUSED_CROSS_BLOCK_ADALN=1` restore the two-kernel oracles.
The final audio/video AdaLN kernels bind directly to offsets in the residual
stream, avoiding two slice blits and 18.8 MiB of scratch at 512x512 (29.4 MiB
at the 864-class benchmark shape).
`H3_DISABLE_FUSED_FINAL_SLICE=1` restores the copy-plus-AdaLN oracle at load.
The BF16 final heads then apply AdaLN while loading their 16x16 projection
tiles, preserving the standalone rounding and accumulation order while
removing another equally sized normalized activation. The two optimizations
together save 37.5/58.9 MiB. `H3_DISABLE_FUSED_FINAL_HEAD=1` restores the
offset-AdaLN-plus-linear oracle at load.

### Token-reduction internals

`--token-reduction` is an independent aggressive DiT mode. After block 3 it
pairs adjacent horizontal target-video tokens while leaving text, audio,
conditions, and reference tokens exact. The complete full-resolution state is
kept as a bypass. During the first ten noisy evaluations it restores before
block 40; subsequent detail-forming evaluations restore before block 30. Each
token returns as its original value plus the update learned by its pair, so
within-pair detail is not discarded.
The pooling kernel writes only true-pair baselines into a dense tail of the
already allocated attention scratch buffer; odd-width singleton tokens need no
baseline. The full bypass uses the oversized QKV tail when it fits, with a
guarded dedicated fallback only for reference-heavy layouts. Common text-only
canvases therefore add no activation arena at any token-grid width. Pooling
also snapshots both source tokens while their BF16 values are already in
registers, avoiding a separate full-hidden blit and redundant source read. The
same entry kernel keeps each pooled row in threadgroup memory and emits the
first reduced block's attention AdaLN, eliminating another global residual read.
At the restore boundary, the first full-resolution attention AdaLN is fused
into expansion: a 10.5 KiB threadgroup row avoids a global residual reread while
still writing the exact bypass needed by the following residual branch.
On a thermal-balanced 512x512x22, 19-forward IT M5 Max A/B this reduced denoise
time from 39.13 to 28.06 seconds (28.3%). Final video/audio latent relative L2
was 5.56%/15.14%. First/middle/last fox frames retained one clean muzzle,
coherent legs, and sharp fur; an independent surfer remained consistent with
one rider and board through the wave spray. It changes composition and is
therefore opt-in rather than the close-reference default.
`H3_TOKEN_REDUCTION_BLOCKS` can override the later `4:30` interval;
`H3_TOKEN_REDUCTION_EARLY=STEPS:END` overrides the early schedule and `0`
disables it. `H3_DISABLE_TOKEN_REDUCTION=1` provides an in-context exact oracle.
`H3_DISABLE_FUSED_TOKEN_POOL_ADALN=1` and
`H3_DISABLE_FUSED_TOKEN_ADALN=1` independently restore the two-kernel entry and
exit boundaries for diagnosis.
Token reduction composes cleanly with the validated `--layers 45 --reuse 2`
settings: on the same 512 benchmark it reduced that profile from 16.69 to
12.60 seconds (24.5% marginal), and independent fox and surfer renders stayed
coherent. Do not combine it with both `--layers 40` and `--reuse 3`; that
6.47-second experiment produced chromatic ringing and ghosted limbs despite
acceptable latent norms.

### Internal canvas and video VAE

`--render-width` and `--render-height` run the model and VAE on a lower
same-aspect internal canvas, then high-quality vImage-scale RGB frames to the
requested output size before callbacks, terminal display, and encoding. This is an
explicit quality/speed tradeoff: a measured 384-to-512 prompt render reduced
M5 DiT time by 33% and video-VAE time by 18% while retaining a clean,
recognizable photorealistic result. Both values must be multiples of 32; the
exact output canvas remains the default.
For square 512 output, 384 is the fast-quality point and 320 is the validated
aggressive point. The latter produced a coherent walking fox and repeated at
8.02 seconds of DiT versus about 15.82 seconds natively. Native 256 uses the
same-cost spatial-RoPE adaptation described above; it remains a fast composition
preview rather than a substitute for a 512- or 768-class final render.
The VideoVAE decoder defaults to fixed 256-pixel spatial tiles with at least
64 pixels of overlap, matching the released MiniMax-H3 configuration.
`H3_VAE_TILE_PIXELS=<N>` (256–320, multiples of 16) is an explicit
override; `H3_VAE_TILE_PIXELS=auto` enables the previous geometry-dependent
256–320 performance heuristic. Larger tiles can decode faster but change
reconstructed pixels and tile-boundary behavior. Values above 320 are unsupported
because larger tiles have demonstrated severe grid/quilt artifacts. Invalid
explicit values now fail; unset the variable to restore 256. These settings affect video
reconstruction only; DiT latents, text/reference conditioning, and audio stay
unchanged. `H3_PROFILE=1` reports the selected policy and geometry.
See [tile policy and upgrade notes](bugfixes/tilefix.md) and
[M4 validation](bugfixes/tilefix-validation.md).

### Weight residency and streamed prompt encoding

On M5-class GPUs, persistent transformer weights request zero-copy loading.
A tensor is mapped only when its actual file offset meets dtype and Metal page
alignment requirements; otherwise it is copied into aligned shared storage.
`H3_ZERO_COPY_WEIGHTS=1` requests this selection for all weights,
`transformer` limits it to transformer weights, and `0` disables it. Other GPUs
retain copied loading by default. Header padding alone does not guarantee mapping
eligibility: the current official transformer shards require copied fallback
under the stricter checks. This can increase M5 load time and resident memory.
Valid unaligned safetensors are supported without rewriting the files;
`H3_PROFILE=1` reports per-tensor fallback reasons. See
[runtime robustness and validation](bugfixes/bugfix1.md).

The streamed Qwen text encoder preallocates a small ring of future layer
buffers and fills them on eight I/O workers while Metal executes the current
layer. The default ring depth is two layers on M3/older hardware and three on
M5, where the target machine has 128 GiB. `H3_QWEN_PREFETCH=0` restores the
single-layer synchronous reference path; values 1-8 select the worker count,
and `H3_QWEN_PREFETCH_DEPTH=1` through `6` overrides the ring depth.

`--ssd-streaming` is a separate, more aggressive residency mode for the DiT.
Only its small per-block normalization weights remain resident. Two complete
BF16 matrix slots alternate while a background reader fills the next slot in
checkpoint-offset order; the current Metal command buffer runs concurrently.
Darwin uncached reads avoid retaining a second copy in the filesystem cache.
The first active block is prefetched again during the final block, so a cached
DiT is ready for its next denoiser evaluation. Measurements reached
about 13--14.6 GiB/s from the internal SSD. `H3_PROFILE=1` reports total bytes,
read throughput, and the part of the read wait that was not hidden by GPU work.

### Metal 4 and TensorOps paths

M5 GPUs automatically use native BF16 Metal 4/TensorOps for the DiT QKV and
attention-output projections at sequence lengths up to 2,048. The compact
Morton schedule routes Q/K/V directly into head-major attention inputs, avoids
three MPSGraph input transposes, and is byte-identical to the portable path. It
improves a complete 512x512 50-block forward by about 2% across repeated IT/US
M5 Max runs. For 2,049-3,072 rows, including 864x480, two row-offset Morton
dispatches preserve the efficient tile geometry and improve the complete
forward by about 2% in balanced runs. Still larger sequences stay on MPSGraph.
`H3_NAX=0` disables TensorOps for exact A/B diagnosis. The selection is guarded
at runtime and falls back to the unchanged portable library if compilation is
unavailable.

`H3_NAX=1` forces the broader native BF16 linear path. It passes the complete
50-block MLX fixture, but remains opt-in: exact-shape microbenchmarks favor its
128-row tile while full DiT runs currently favor MPSGraph scheduling. This
keeps a working NAX integration available for later quantized/fused kernels
without making a benchmark regression the default.
`H3_NAX=mlp` selects a more specialized Metal 4 path: paired FC1 gate/up
TensorOps tiles apply SwiGLU in threadgroup memory and write only the
14,336-wide activated intermediate, then FC2 also stays on TensorOps.
`H3_DISABLE_NAX_MLP=1` keeps the MPSGraph MLP in a context created this way for
same-process A/B testing. The path is deliberately opt-in because scheduling
depends on the OS GPU stack: the primary macOS 26.5.2 M5 Max gained 1.3-2.0%
in isolated real-weight MLP runs but lost about 1-3% in a complete 50-block forward,
while an otherwise identical macOS 26.5 M5 Max gained 1.4% in a same-context
forward A/B. The resulting 50-block velocities were close (1.9% video and 2.4%
audio relative L2), but not byte-identical.

### Specialized projection kernels

The narrow DiT audio/video output heads convert their small released F32
weights to BF16 once and use the Iris-derived 16x16 tiled linear directly on
BF16 activations. At the production 320-render geometry, isolated paired-head
measurements are 2.30x faster on M3 Max and 1.83x faster on M5 Max, with
relative L2 `8.64e-4`; the absolute M5 saving is about 0.6 ms per evaluated
step. Full fox and surfer sequences remained clean and measured 29.9/38.4 dB
against the F32-head renders. `H3_DIT_F32_FINAL=1` restores the close-reference
head and its extra activation buffers.
The F32 `96->5376` video and `32->5376` audio patch projections use a dedicated
16x16 cooperative tile, retaining F32 weights, inputs and accumulation while
rounding the tile result directly to BF16.
Paired production-shape measurements are 1.77x faster on M3 and 1.62-1.78x
on M5; the complete generated RGB stream is byte-identical to the scalar path.
Fusing the final cast improves the 2835-row tile itself from 2.499 to 1.734 ms
on M3 and 1.555 to 1.186 ms on M5, and removes 38.27/59.66 MiB of F32 scratch
at 512/864-class geometry. `H3_DISABLE_FUSED_PATCH_CAST=1` restores the tiled
F32 output plus standalone cast; `H3_SCALAR_PATCH=1` selects the scalar
diagnostic path.
The same tile binds its output directly into the packed hidden stream, removing
the BF16 media staging buffers and their blits. This saves another 19.13/29.83
MiB and improves the 2835-row boundary from 1.847 to 1.730 ms on M3 and 1.282
to 1.184 ms on M5. Contiguous T2VA uses byte offsets; FL2VA/Ref2VA use compact
destination-row maps so each modality remains one large dispatch. A complete
six-segment Ref2VA M5 ABBA remained byte-identical and improved 5.067 to 5.033
seconds per measured forward pair. `H3_DISABLE_FUSED_PATCH_PACK=1` restores the
staging buffers and packing blits.

### Scheduling and activation memory

The DiT core is split into two ordered Metal command buffers so GPU execution
of the first part overlaps CPU encoding of the second. Thermal-balanced ABBA
measurements select a 60%-depth split on M5 (30/50, 27/45, and 24/40), with
roughly 0.5-1.8% wins; M3 automatically splits only the validated 30/50 case,
which measured 1.2% faster, because 24/40 regressed there. The operation order
and generated bytes are unchanged. `H3_DIT_COMMAND_BLOCKS=0` restores one
command buffer; values 1-50 override the split for further tuning.
DiT activation buffers also follow their actual intra-block lifetimes: the QKV
projection arena is reused first for attention heads and then for the normalized
MLP input, while the current attention-output arena becomes the MLP output after
its branch has been consumed. This removes 61.25 MiB at 512-class geometry and
99.63 MiB at 864-class geometry without changing dispatches or arithmetic.
`H3_DISABLE_DIT_ACTIVATION_ALIAS=1` restores separate diagnostic buffers.
MPSGraph tensor-data wrappers for immutable DiT weights and biases are retained
with their resident buffers. This avoids rebuilding the same binding metadata
for every block and denoiser evaluation without copying tensor storage; measured
ABBA gains were 1.6% on M3 Max and 0.4-1.1% on M5 Max. Activation wrappers stay
transient because retaining them regressed the M5. The outputs remain
byte-identical, and `H3_DISABLE_GRAPH_DATA_CACHE=1` restores transient wrappers
for all tensors.
On M3/older hardware, the four MPSGraph segments in each DiT block also reuse
one `MPSCommandBuffer` wrapper for their shared underlying Metal command buffer.
Repeated thermal-balanced runs measured 1.0-1.6% faster on M3 Max; M5 measured
neutral, so it retains fresh wrappers. `H3_REUSE_MPS_COMMAND=0` or `1` overrides
the automatic selection. Results are byte-identical.
On M5, the serving Euler sampler keeps its patch-packed F32 latents and cached
BF16 velocities in Metal buffers. Each selected denoiser refresh is completed
before the next is encoded, avoiding MPSGraph back-pressure while removing all
intermediate latent/velocity readbacks and repacking. Two warm eight-run A/B
sequences measured small 0.1% and 0.3% gains with byte-identical final latents;
the path also saves roughly 16 bytes of transient host state per video-latent
element (about 136 MB at the 768p shape). M3 and older GPUs retain the CPU
sampler by default. `H3_CPU_SAMPLER=1` restores it on M5;
`H3_GPU_SAMPLER=1` selects the GPU-state path explicitly, and
`H3_GPU_SAMPLER_WINDOW=0` enables the slower unbounded encode-ahead diagnostic.

### Checkpoint layout and media pipeline

The released checkpoint stores DiT QKV rows interleaved per attention head.
Native Metal consumes that layout directly in the fused QK-normalization/RoPE
kernel, avoiding a checkpoint transpose and extra RAM. The earlier identity
interpretation was the cause of the noisy diagnostic outputs.

Saved h3cli files use [current-only versioned formats](features/current-state-contract.md).
Older states and sidecarless AV delivery are unsupported; save fresh files with
this build. CUDA video derives its SGLang arithmetic automatically.

The public generation path decodes the joint audio latent with a streamed native
BigVGAN/AudioVAE and writes synchronized H.264 plus 32 kHz stereo AAC. The native
waveform agrees with the corrected MLX oracle to relative L2 `6.94e-5`.
`--first-frame`, `--last-frame`, and their combination use the released visual
VAE encoder, Qwen3-VL vision tower and three-deepstack multimodal presentation,
0.999 condition augmentation, and fixed condition rows in the native DiT. The
first and last images both preserve aspect ratio, scale to cover the internal
render canvas, and center-crop any excess. For example, 682×1024 inputs at a
672×1024 render size lose approximately five pixels from each side in either
role. This keeps a shared anchor's input framing consistent between adjacent
segments. Regenerate saved conditioning caches containing a first-frame anchor
created before this change; their resize-policy identity no longer matches.
`--ref-image`
selects the distinct Ref2VA transformer, preserves ordered `<Picture N>`
presentation, and uses the shared
[reference-image sizing policy](usage.md#8-add-image-video-and-audio-references).
`--ref-silent-video` performs bounded 24 fps decoding, independent released
17-frame VAE chunk encoding, two-frame Qwen sampling, and timestamped
`<Video N>` presentation. `--ref-video` preserves an embedded soundtrack,
`--ref-video-audio VIDEO AUDIO` supplies an explicit replacement, and
`--ref-audio` appends an ordered standalone clip. Reference audio is decoded as
32 kHz stereo F32, encoded by the native AudioVAE posterior-mean path, mixed as
0.999 clean latent plus 0.001 seeded noise, pinned to the audio condition
timestep 1.0, and packed as width-32 rows on the same rotary timeline as visual
references. Audio inputs are 2-15 seconds, at most three are
accepted, their total decoded duration is capped at 15 seconds, and a standalone
audio reference must be combined with an image or video reference.

The default `released-v1` video front end retains the full normalized reference
for Qwen and soundtrack timing, selects a separate VAE prefix, and requires
2–15 seconds per clip with at most 15 seconds total video duration. Its released
VAE encoder is integrated into generation, including corrected packing and reference RoPE.
See the [encoder validation](minimax/refvideo-encoder.md) and [integration results](minimax/refvideo-integration.md).

The native audio encoder matches the corrected MLX oracle at relative L2
`3.59e-6` on a real two-second stereo fixture. The correction is important: the
original MLX reshape interleaved left/right samples, whereas the official
PyTorch/SGLang path folds intact stereo channels into the batch dimension. On
the 128 GB M5 Max, clean end-to-end image+audio and embedded-video+audio renders
completed in 74.58 and 76.99 seconds respectively, each with about a 40.1 GB
peak physical footprint and zero swaps.

### Profiling and diagnostic paths

`--profile` reports each Metal-backed phase separately: wall time, CPU-side
command encoding, complete commit-to-fence wait, root-command GPU timestamps,
peak live tensor storage, cumulative allocation, and dispatch counts. The wait
measurement is the complete command turnaround; the root GPU timestamp alone
can omit child buffers scheduled internally by MPSGraph and is labeled
accordingly.
It also emits a completed `h3_step` JSON record per denoising evaluation with
dispatch counters, physical/resident/compressed memory, swap and Metal allocation.
`H3_PROFILE_COMPONENTS=1` adds diagnostic component fences and JSON timings;
those runs must be kept separate from ordinary performance comparisons.

The DiT fast path evaluates each BF16 `fc1 -> SwiGLU -> fc2` block as one cached
graph, avoiding separate graph boundaries and persistent intermediate tensors.
Set `H3_DISABLE_FUSED_MLP=1` to retain the close-reference operation boundaries
for numerical diagnosis.

On supported M5 Metal 4 TensorOps hardware, the native int8 MLP engine is the
default. It dynamically quantizes activations, uses per-output-channel weight
scales, and gives the sensitive FC2 input one scale per 1,024 channels.
The selected FC2 kernel keeps scaled partial products in private cooperative
fragments instead of repeatedly spilling a 32 KiB threadgroup tile. A fixed
50-layer, 19-transition 512x512 render measured 36.30 seconds with BF16 MPS and
25.80 seconds with int8 on M5 Max. Beginning, middle, and final decoded frames
retained the same subject, composition, and motion; small edge and fur details
can differ. The current diagnostic implementation retains both BF16 and int8
MLP weights only when an A/B diagnostic requests them. Normal int8 loading
releases each block's BF16 FC1/FC2 buffers after their submitted quantization
finishes, reducing measured peak tensor storage to 25.9 GiB from the BF16
path's 36.4 GiB. Runtime weight quantization still adds startup time.

The fastest M5 path also quantizes each DiT QKV projection and writes its
Q/K/V tiles directly in head-major attention layout before the existing Q/K
normalization and RoPE kernel. In a fixed 50-layer, 19-transition 512x512
render this reduced denoising again, from 25.80 to 19.32 seconds. Sampled
beginning, middle, and final frames remained a coherent detailed fox walking
through snow; quantized attention can change framing and fine detail. Use
`--use-slower-bf16-qkv` for the close-reference BF16 projection. Normal int8
loading releases the redundant BF16 QKV weights after quantization.

The following attention-output projection is int8 as well on the default M5
path. Crossed same-model tests improve a complete forward by another 4.5-5.5%
at 512 and 864. A decoded fox render remained clean and closely matched the
int8-QKV-only composition; its thermally hot denoise measured 19.18 seconds.
Use `--use-slower-bf16-attention-output` to retain that projection in BF16.

On that int8 path, SDPA now leaves its result in native
`[head,row,dimension]` order. A specialized 256-thread kernel gathers and
quantizes each H3 row directly into the projection's row-major int8 buffer,
eliminating the intervening full-width BF16 transpose without changing any
output byte. Thermally controlled crossed runs improve complete 512 and 864
forwards by roughly 0.2-1.2%. Use
`--use-slower-row-major-attention-output` to restore the explicit BF16
row-major SDPA output and ordinary quantizer.

The M5 path also folds QKV and MLP activation quantization into the preceding
gated AdaLN kernel. This removes 99 standalone quantizer dispatches per
50-layer forward while preserving the previous output bytes, improving crossed
512/864 measurements by about 0.3-0.6%. Use
`--use-slower-unfused-int8-inputs` to restore the standalone quantizers.

The fused gated-AdaLN path loads its full 5,376-wide H3 rows as BF16x4 vectors
and writes int8x4. It stages the rounded values locally before computing the
original per-thread RMS sequence, so the reduction tree and every output byte
remain unchanged. Crossed measurements save roughly another 0.1-0.5%. The
existing `--use-slower-unfused-int8-inputs` option retains the portable scalar
and standalone-quantizer fallback.

Q/K RMS normalization and RoPE are performed inside the int8 QKV projection
tile as well. The fused epilogue is byte-identical and improves complete
forwards by 2.1-3.2% at 512 and 1.0-1.8% at 864 in crossed M5 measurements.
Use `--use-slower-unfused-qkv-rope` to restore the separate Q/K kernel.

That epilogue processes four adjacent Q/K dimensions per work item with
BF16x4 loads and stores. The per-element arithmetic and BF16 rounding order are
unchanged, while crossed cool-state measurements improve complete forwards by
about 0.4-1.0% at both 512 and 864. The same
`--use-slower-unfused-qkv-rope` option restores the scalar standalone path.

At up to 2,048 rows, the exact RMS loop uses BF16x4 loads followed by four
explicit ordered FMAs. This preserves every output bit and improves 512-class
forwards by another 0.5-0.6%; larger shapes retain scalar loads because the two
forms tie there. Use `--use-slower-scalar-qkv-rms` to force scalar loads.

The int8 attention-output projection caches its 128 row and column scales in
1 KiB of threadgroup memory instead of rereading them for every cooperative
fragment element. Above 2,048 rows the fused QKV kernel uses the same idea and
then recycles that storage for inverse RMS values; smaller QKV shapes retain
direct loads because the two forms tie there. Both are byte-identical and
improve complete forwards by about 0.2-0.7% where selected. Use
`--use-slower-uncached-int8-scales` to restore direct device-scale loads.

For sequences of at most 2,048 rows, the H3 attention-output projection also
compiles its 7,168-by-5,376 shape into the TensorOps kernel. The result remains
byte-identical while saving about 0.2-0.8% in crossed complete 512-forward
measurements. Larger sequences retain the dynamic-shape kernel because the
specialization regresses there. `--use-slower-uncached-int8-scales` restores
the general dynamic, direct-scale-load implementation.

FC1 also uses an H3-specialized, compile-time 5,376-wide TensorOps loop. It is
byte-identical to the generic loop and saves about 0.1-0.4% in crossed complete
forwards. Use `--use-slower-dynamic-fc1-k` to restore the runtime-bound loop.

```sh
./bin/h3cli --profile -d ./models/MiniMaxH3 \
  -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 50 --reuse 1 -o outputs/fox-int8.mp4
```

Use `--use-slower-bf16-mlp` to force the portable close-reference MPS/BF16 MLP
path for numerical comparison. Older Metal hardware selects that path
automatically when the required native TensorOps kernels are unavailable.
For FC2 activation quantization, sequences of at most 2,048 rows use an exact
128-thread reduction. Each thread retains its eight BF16 input values while
computing the group maximum, avoiding a second device-memory read when it emits
the int8 values; crossed M5 measurements improved complete 512 forwards by
about 0.2-0.8% without changing any output byte. Larger sequences retain the
measured 256-thread kernel. `--use-slower-grouped-quantizer` forces the latter
at every size for A/B comparison.

The native baseline targets the original `FL2VA/` and `Ref2VA/` checkpoint
trees. Model phases are loaded and released separately so the 33B transformer,
Qwen encoder, and decoders never have to coexist in unified memory.

### Shader location

Every Metal component uses the same lookup policy: `H3_SHADER_PATH` first,
then a supplied absolute shader path, then a supplied relative path in the
current working directory, then the same relative path under the executable's
directory, then the parent of that directory when the executable is in `bin/`.
The default path is `src/metal/shaders.metal`. Preserve sibling `bin/` and `src/`
directories to launch from any working directory. The executable location is
canonicalized, including symlinks.
An invalid explicit override fails without trying another shader. Missing-file
errors list the attempted locations. See [runtime release notes](bugfixes/bugfix1.md).

The mixed Metal path also supports `--metal-attention sol`. Configure
`--sol-dense-steps N` using absolute schedule indices and
`--sol-dense-sigma F` using the maximum video/audio sigma (`-1` disables it).
The first N evaluations and sigmas at or above F remain dense; conditioning,
continuation prefixes and range-recovery heads stay exact. SOL has its own
quality/performance gate and is not qualified by the accepted dense M2 result.
See [the native design](metal/design-native.md).
