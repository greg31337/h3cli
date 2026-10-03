# CUDA caching and sparse attention

These advanced controls have hardware and feature restrictions. Start with
the quality presets in the [usage guide](../usage.md#4-choose-a-speedquality-preset).
Run the examples from the repository root.

## Adaptive cache and SubBlock attention

Native adaptive caching uses a fresh dense first block to decide whether to
reuse the previous suffix residual. Select `--adaptive-cache conservative` or
`--adaptive-cache aggressive`; the default is `off`. The presets default to change
thresholds 0.04/0.08 and maximum hit streaks 1/3. Both refresh the first
four evaluations by default and always refresh the final evaluation.

Override either preset with `--adaptive-cache-threshold T` (finite decimal FP32,
0–1 inclusive) and `--adaptive-cache-max-hits N` (integer 1–16). A hit requires
`score < T`: zero disables hits, and one still performs an ordinary comparison.
These flags require an enabled cache and do not enable it themselves. Omission
on resume restores the saved controls; explicit values must match numerically
at FP32 precision. The same controls apply to supported text-only FP8/NVFP4
conservative caching.

SubBlock needs an explicit optional build with the qualified CUDA 13.0 Update 3
and shared cuDNN 9.20 environment:

```sh
source outputs/setup/linux-env.sh
make -j8 CUDA_ARCH=120 CUDA_SUBBLOCK=1
```

Select `--cuda-attention subblock --subblock-sparsity 0.75` (or `0.80`). It keeps
the first ten scheduler evaluations by default, block 0, protected query blocks and short
sequences dense. Other eligible blocks use native selected-tile BF16 attention.
The retained budget rounds upward to eight key blocks; protected keys can raise
the actual density. `--subblock-sparsity 0` uses the exact shared dense path.

Set `--adaptive-cache-warmup N` or `--subblock-warmup N` to change the initial
exact/dense evaluation count. Explicit values must satisfy **2 ≤ N ≤ 16** and
**N ≤ steps − 2**, leaving at least two evaluations after warmup. For example,
`--adaptive-cache conservative --adaptive-cache-warmup 6` keeps the first six
evaluations exact; `--cuda-attention subblock --subblock-warmup 4` permits sparse
attention starting at zero-based step 4. Each flag requires its corresponding
feature. Omitting them preserves the fixed 4/10 defaults, including short runs
that finish during warmup. The final adaptive refresh and SubBlock's first-block
and protected-query rules are unchanged.

Warmup uses absolute indices in the full denoising schedule, including after
resume. Checkpoints restore both counts; explicit resume flags must match the
saved settings. When both features are enabled, adaptive caching refreshes at
the configured SubBlock transition. The accepted comparison videos use the
original 4/10 defaults; the [warmup controls](approximate-warmup.md)
describe persistence and validation.

The two features can be combined for CUDA text-to-video on SM120 with BF16
projections and all 50 blocks. Both require reuse/core-reuse 1 and reject
frame anchors, LoRA/Turbo, layer thinning, token reduction, other
approximate attention and Metal. Adaptive cache, SubBlock, and their combination
support **image, video and audio references with BF16 projections**, including
ordered mixed sets, silent video, embedded video audio and external replacement
soundtracks. Reference cache decisions use the maximum of the global,
generated-video and generated-audio change scores. Target scores exclude
reference rows, so a large unchanged reference cannot dilute target changes.
Both generated stereo audio channels contribute. Encoded reference latents
remain fixed; reference hidden states may use the cached suffix approximation. Existing Ref2VA limits apply: 12 references total, at most 9 images,
3 videos and 3 audio inputs; standalone audio requires an image or video.
Quantized references remain unsupported. Ordinary reference generation with
reuse 2/3 is unchanged.

Both features also support **BF16 hard continuation and bridge**, with or without
those references. Continuation uses cache recipe 4: the maximum of global,
generated-video suffix, generated-stereo-audio suffix, and each active nonzero
bridge-class score. Frozen history cannot dilute a changing suffix or bridge
class. The entire inherited prefix stays protected in SubBlock, including
mutable bridge rows. This can reduce the sparse speedup with long context.
The exact initialized history is checked after each Euler update; bridge
velocities retain their existing one-time strength scaling.

Each new `.h3av` segment starts with empty adaptive history. Same-job
`.h3sample` resume restores it and needs neither the original source AV nor
reference files. CUDA uses CPU-state F32 Euler with CUDA transformer/VAEs.
Quantized approximation with continuation, first/last anchors and upscale
remain unsupported. See the [continuation design](design-adaptive-continuation.md)
and [implementation, measurements and comparison videos](adaptive-continuation-results.md).

```sh
./bin/h3cli -d models/MiniMaxH3 -p 'The woman continues walking along the beach.' \
  --width 640 --height 480 --frames 90 --steps 50 --seed 42 \
  --continue-from outputs/segment01.h3av --continue-context 39 \
  --continue-mode hard --adaptive-cache conservative \
  --cuda-attention subblock --subblock-sparsity 0.75 \
  --save-av-state outputs/segment02.h3av -o outputs/segment02.mp4
```

The source must have matching render geometry and latent-space identity.
Change `--continue-mode hard` to `bridge` to adapt the older context; add the
desired reference flags on each segment. This example delivers 51 new frames
and retains all 90 frames in the saved state.

For example, add `--adaptive-cache conservative --adaptive-cache-threshold 0.06
--adaptive-cache-max-hits 2` to a BF16 reference-video request. These controls
change the reuse policy; they do not guarantee hits or a particular quality level.
See the [controls and reference qualification report](adaptive-controls-references-results.md)
for exact replay tests and the 18-video comparison.

Adaptive storage defaults to a **4096 MiB ceiling** for new requests. Set
`--adaptive-cache-max-mib N` to another positive whole-MiB ceiling. This allocates
only the exact cache required by the packed layout: three BF16 tensors plus
6,168 reduction bytes for ordinary recipes, or `rows × 5376 × 6 + 6168` bytes.
Continuation recipe 4 uses 47,288 reduction bytes instead; its compact class
map is included in the host plan, without a full device tensor per class. For example,
1344×768/362 frames with 58 text tokens needs 3,519,780,888 bytes (minimum
3,357 MiB). The ceiling does not guarantee sufficient VRAM; weight residency,
activations, device reserves and SubBlock's separate **512 MiB attention
workspace limit** still apply. Initialization reports the resolved ceiling,
actual device bytes, persistent checkpoint bytes and packed rows.

On sampler resume, omitting the budget flag restores the saved ceiling. A
sufficient higher or lower replacement preserves cache history; a too-small
replacement fails before adaptive payload allocation. Current adaptive
checkpoints require section-40 v3, including resolved threshold and hit ceiling;
older adaptive sections are rejected. Text BF16/quantized arithmetic recipes
remain 1/2; BF16 references use recipe 3; BF16 continuation/bridge uses recipe 4.
Presentation schema 9 records effective
controls for independent decode. The sampler envelope remains schema 2. Two
persistent BF16 cache tensors are saved, and serialization needs an additional
host copy. The sampler container remains limited to 16 GiB.

Active sampler checkpoints require the same device, build and runtime; clean
AV states remain independently decodable. See the
[budget design](adaptive-cache-budget-design.md) and
[budget/reference qualification](adaptive-budget-reference-results.md).
For a six-step image-reference preview that actually uses sparse attention:

```sh
./bin/h3cli -p 'A person walks along a sunny beach.' \
  --width 1344 --height 768 --frames 362 --steps 6 \
  --ref-image inputs/1.jpg --ref-image-size max \
  --cuda-attention subblock --subblock-sparsity 0.75 --subblock-warmup 2 \
  -o outputs/reference-subblock.mp4
```

Leaving SubBlock warmup at its default ten keeps a six-step run entirely dense.
Reference queries remain dense and reference keys stay available to generated
queries, including blocks that mix protected rows with generated video rows.

Conservative caching with dense attention also supports native
FP8 or NVFP4 projections:

```sh
./bin/h3cli -d /path/to/MiniMax-H3 -p 'A pianist plays in a sunlit concert hall.' \
  --width 640 --height 480 --frames 90 --steps 50 \
  --adaptive-cache conservative --cuda-denoise-quant fp8 \
  -o outputs/adaptive-fp8.mp4
```

Use `nvfp4` for the four-bit variant. This combination keeps block 0 in BF16
and quantizes the four projection matrices in blocks 1–49. Cached tensors
remain BF16 and decision scores use FP32. Quantization-only execution still
quantizes all 50 blocks. Aggressive caching remains incompatible with
quantization. The [six-video experiment](adaptive-quant-experiment.md)
records the comparison method, distinct checkpoint recipes and qualification.
At 640×480 / 90 frames / 50 steps on RTX PRO 5000, adding conservative cache
reduced measured wall time by 22.5% beyond FP8 alone and 24.3% beyond NVFP4 alone.
All four quantized candidates failed the historical similarity checks; the user
accepted all six comparison results. The report includes synchronized
three-way playback and incremental comparisons against quantization alone.

SubBlock also supports FP8/NVFP4 with adaptive cache off. Combine
`--cuda-attention subblock --subblock-sparsity 0.75` with
`--cuda-denoise-quant fp8` or `nvfp4`. Both quantized variants retain all 50
blocks' quantized projections, including block 0; SubBlock's dense first-block
protection applies to attention. The separate
[SubBlock/quantization comparison](subblock-quant-experiment.md)
uses the same six-video workload. At 640×480 / 90 frames / 50 steps, adding
SubBlock 0.75 gave no useful wall-time improvement: FP8 took 150.28 versus
149.13 seconds alone; NVFP4 took 116.88 versus 116.93 seconds. Denoising was
about two seconds slower in both combined rows, with further video/audio
differences. The user accepted all six comparison outputs.
Triple adaptive+SubBlock+quantization remains unsupported.

Use the fixed [comparison manifest](adaptive-subblock-manifest.json)
through `tests/cuda_adaptive_subblock.py`; it scopes the 50-step allowance to
the twelve specified render subprocesses. Ordinary tests retain their six-step
limit. The [native contract](adaptive-subblock-contract.md) defines the
math, protection rules and independent validation tolerances.

The completed [qualification and comparison](adaptive-subblock-results.md)
contains all twelve videos at 640×480 / 90 frames / 50 steps, with synchronized
playback and visual/audio metrics. Conservative adaptive caching measured a
1.51× wall-time ratio versus dense, with the smallest differences among the
approximate variants. Combined SubBlock recipes offered little additional
timing benefit. Every approximate variant failed the complete historical
similarity checks. The user accepted the results and passed visual qualification
for all twelve comparison videos. Both features remain off by default.

## CUDA SOL attention

Build the native routed BF16 attention backend on SM120 with the same CUDA 13.0
Update 3 toolchain and runtime environment:

```sh
source outputs/setup/linux-env.sh
make -j8 CUDA_ARCH=120 CUDA_SOL=1
./bin/h3cli -d models/MiniMaxH3 -p "A sailboat on a calm lake." \
  --width 640 --height 480 --frames 243 --steps 2 \
  --cuda-attention sol -o outputs/sol.mp4
```

SOL approximates selected distant key/value blocks with centroids. CUDA defaults
are Q32/KV64, `--sol-tau 1`, `--sol-min-exact 0`, one dense initial step and
layer, and local radius 1. Q64 is supported; CUDA requires KV64. Text, references,
audio, target anchors and continuation prefixes remain protected. Metal also
defaults to `--sol-min-exact 0`. Zero removes the minimum exact-block quota;
content-based routing and protected/local exact regions still apply.
`--sol-min-exact 1` uses the existing dense CUDA path. See the
[zero-default speed comparison](sol-zero-results.md).

The backend is opt-in and independent of SageAttention. Projection precision and streamed weights remain separate controls;
Exact checkpoint resume
restores the saved SOL policy; change it only for a fresh generation or a new
completed-state continuation segment. Completed `.h3av` files can be decoded by
a build without SOL.

See the [design](design-sol-attention.md),
[implementation audit](sol-implementation-audit.md), and
[measured support and limitations](sol-attention-results.md).
Two-step validation measures execution and approximation drift; it does not
qualify final 50-step rendering quality.
