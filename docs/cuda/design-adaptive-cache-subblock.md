# Native adaptive caching and SubBlock attention

Status: **implemented, qualified and compared; all 38 tasks complete**. This document
defines the experiment on the existing single CUDA pipeline. Work is
retained in the [completed task list](adaptive-subblock-tasks.md). The completed pipeline campaign
is retained in its [archived checklist](single-pipeline-tasks.md).

The [completed qualification and twelve-video results](adaptive-subblock-results.md)
record measured behavior, limitations and the recommendation to keep both
features off by default. The user accepted the results and passed visual
qualification for all twelve videos. A1 has the smallest measured differences;
combined recipes offered little additional timing benefit in this workload.

The subsequent [FP8/NVFP4 experiment](adaptive-quant-experiment.md) extends
conservative caching with a BF16 first block and quantized suffix. It has a
separate six-video manifest and review; this twelve-video recipe-1 record stays fixed.

The [warmup controls](approximate-warmup.md) later made the initial 4/10 counts
configurable, with checkpoint persistence and unchanged defaults. The fixed
comparison matrix below retains its original counts.

The [SubBlock 0.75/quantization experiment](subblock-quant-experiment.md) separately
combines sparse attention with all-block FP8/NVFP4 projections and no adaptive cache.

## Objective and fixed scope

Implement two independent, opt-in accelerations: an adaptive residual cache
around the main DiT block stack, and native BF16 SubBlock sparse attention.
Compare each separately and in combination against dense execution and the
existing fixed reuse controls. Keep the default SGLang arithmetic and its
recorded 204-output regression unchanged.

The user-selected comparison workload is **640×480, 90 frames, 50 steps,
one complete video per variant**. All variants share one prompt, seed, model,
schedule, decoder and delivery configuration. These are single observations,
not repeated-run statistics or a broad quality qualification.

Initial implementation and numerical qualification target CUDA on the configured
RTX PRO 5000 / SM120 host, with the existing model installation. Connection
details remain outside the repository. Re-probe hardware and use the current
[CUDA 13.0.3 / shared cuDNN 9.20 setup](cudnn-920-qualification.md).
Native inference must not import Python, Torch, Triton, Cache-DiT or FlashInfer.
Python remains available for build tools, isolated numerical references and
reporting. Preserve existing Metal builds and behavior; reject explicit new
options there until a Metal implementation is separately qualified.

This campaign excludes a Metal port, multi-GPU execution, model/codec upgrades,
new quantization recipes, Sage/SubBlock fusion, SOL/SubBlock fusion, layer
thinning, token reduction, LoRA/Turbo tuning and additional full-video sweeps.

## Existing foundations and new work

| Existing feature | Relevant behavior | Addition |
| --- | --- | --- |
| `--reuse` | Fixed whole-DiT evaluation schedule; separate video/audio velocity extrapolation | Retained comparison control |
| `--core-reuse` | Fixed refresh interval for the full transformer residual; projections and heads stay current | Reuse its ownership/state patterns, with a fresh first block and a measured decision |
| SOL | Protected packed ranges, routing metadata, bounded native attention workspace | Reuse layout/protection helpers; SubBlock has separate scoring and dropped-block semantics |
| Sage2++ / Sage3 | Quantized dense main-DiT attention | Preserve unchanged; do not combine in this campaign |
| Sampler checkpoints | Explicit recipes, exact continuation of mutable state, section checksums | Add adaptive history and SubBlock policy identities |

The extension points are [dit.c](../../src/denoise/dit.c),
[attention.h](../../src/denoise/attention.h), [gpu.h](../../src/gpu.h),
[gpu_cuda.cu](../../src/cuda/gpu_cuda.cu), [sol.c](../../src/denoise/sol.c),
[sampler_state.h](../../src/sampling/sampler_state.h) and
[sampler_file.c](../../src/sampling/sampler_file.c). Keep all new source under `src/`,
without an `h3_` filename prefix; executables and libraries remain in `bin/`.
Implemented modules are `src/denoise/adaptive_cache.c`/`.h` for policy and context state,
`src/cuda/cuda_adaptive_cache.cuh` for GPU score operations, and
`src/denoise/subblock.c`/`.h` plus `src/cuda/cuda_subblock.cu`/`.h` for routing policy,
workspace planning and native kernels. Keep the comparison runner in
`tests/cuda_adaptive_subblock.py` with a separate fixed case manifest;
do not extend the frozen reference manifest with approximate outputs.

The motivation is DBCache's fresh boundary blocks and residual-change decision,
documented in the [Cache-DiT design](https://cache-dit.readthedocs.io/en/latest/user_guide/DBCACHE_DESIGN/).
The native recipe below explicitly defines its own anchor and refresh semantics;
it does not claim output identity with the Python package.

## Adaptive-cache recipe 1

### Block partition and decision

Use all 50 main DiT blocks. Block 0 is a freshly computed, dense BF16 probe on
every scheduler evaluation. Blocks 1–49 form the cacheable suffix. Input
projections, the text refiner and final timestep-dependent video/audio heads
retain their current execution. This is an F1/B0 partition; arbitrary F/B
partition tuning is outside the first implementation.

Let `x_t` be the packed hidden state entering block 0, `y_t` its output, and
`z_t` the output after block 49. Define:

```text
probe_t = BF16(y_t - x_t)
change_t = mean(abs(FP32(probe_t) - FP32(probe_anchor)))
           / max(mean(abs(FP32(probe_anchor))), 1e-6)

refresh: suffix_delta = BF16(z_t - y_t)
         probe_anchor = probe_t
hit:     z_t = BF16(y_t + suffix_delta)
```

The mean covers valid packed rows and channels, including audio; padding is
excluded. Reduction accumulates in FP32 using a fixed, versioned reduction
order. Compare with strict `< threshold`. Store the anchor from the last
successful suffix refresh, not the preceding hit. Do not accumulate per-step
scores, extrapolate the cached suffix, or update it recursively on a hit.

Force a suffix refresh when the cache is empty, during the initial warmup,
at the maximum hit streak, and on the final scheduler evaluation. A layout,
model/fold, execution-recipe or attention-phase change invalidates the cache.
Nonfinite activations or scores are errors, not cache hits or silently repaired
values. Stage refreshed cache state until the complete forward succeeds and
publish it with the successful sampler transition. Cancelled or failed requests
cannot publish partial history or checkpoints containing uncommitted decisions.

| Preset | Warmup evaluations | Change threshold | Maximum consecutive hits |
| --- | ---: | ---: | ---: |
| `conservative` | 4 | 0.04 | 1 |
| `aggressive` | 4 | 0.08 | 3 |

These starting values follow the conservative/stride experiments in the
[LMSYS H3 article](https://www.lmsys.org/blog/2026-08-27-minimax-h3-h200/).
Their effectiveness with the explicitly defined native score is recorded in
the [completed comparison](adaptive-subblock-results.md#twelve-video-comparison).
Keep both presets fixed for the comparison; a zero-hit result is evidence,
not permission to tune against the rendered clip.

One decision applies to the joint packed stack. Log video/audio probe-change
statistics separately to expose cases where the more numerous video rows hide
audio changes. Those diagnostics do not introduce independent modality caches
or silently change the decision formula.

### Ownership, memory and streaming

Use a request-owned context for the anchor, suffix delta, last refresh index,
hit streak and recipe. Keep tensors on the GPU; transfer only bounded scalar
decision data to the host. Reuse existing scratch where lifetime analysis
permits, without aliasing a saved anchor or asynchronous input.

Budget at most **512 MiB additional adaptive-cache storage**, including two
persistent BF16 tensors, one full-size BF16 scratch tensor and reductions.
Compute the exact checked byte requirement from the packed shape before weight
admission. Reject an explicit request that exceeds the budget; do not enlarge
existing limits or silently disable caching. Include this reservation alongside
the independent attention workspace in peak-memory reporting.

On a hit, skip suffix weight uploads, GEMMs, attention and MLP execution.
Audit prefetch initiation and cancellation so skipped blocks are not needlessly
streamed and outstanding reads cannot overwrite live buffers. After a miss,
use the current streaming path. Cache off must retain the original allocation
and dispatch path without new reductions or synchronization.

## Native SubBlock recipe 1

### Operator and routing

Add a distinct main-DiT attention backend for noncausal BF16 Q/K/V, head
dimension 128, the current sequence-major input layout and both supported
output layouts. Do not select it by head count alone: refiners can share that
shape and must remain on their existing implementation.

Start with 64-token query/key blocks and four 16-token sub-blocks on each side.
Pool valid Q/K rows, estimate per-query-block/per-head key-block importance
using the upstream log-sum-exp score, and retain the highest-ranked blocks.
The design follows the article's SubBlock algorithm; the frozen source audit
pinned the exact score equation, scaling, pooling dtype and tail weighting
before implementation. Document any intentional difference as part of the native
recipe rather than claiming upstream-equivalent routing.

`sparsity` denotes the fraction requested for removal. Use 0.75 and 0.80 for
the comparison. For `B` valid key blocks, the ordinary selection budget is
`min(B, 8 * ceil((1 - sparsity) * B / 8))`; deterministic ties prefer the lower
key-block index. Protected blocks are unioned into that selection, so observed
density can exceed the nominal budget. Deduplicate indices and preserve the
original attention visibility mask for every query row.

Reuse packed segment metadata to protect text, conditioning images/videos,
audio, target anchors and continuation prefixes. Protected query blocks use
dense attention; protected key blocks remain available to eligible generated
video queries. A mixed query block is conservatively protected. Incomplete
tails use true row counts and masks, never padded averages or extra visible
keys. Unsupported mask shapes use the shared dense path with an explicit
reason. This conservative protection policy is a native extension to audit
and measure, not a claim about the article's exact retained density.

Selected tiles use BF16 operands with FP32 softmax state and accumulation;
dropped tiles contribute neither scores nor values. There is no centroid
substitute for a dropped block. The router and kernel must avoid materializing
the full token-by-token attention matrix. Bound routing storage by head groups
and query slabs within the existing **512 MiB attention workspace**.

### Scheduling and native integration

Use dense attention for scheduler indices 0–9 and sequences below 4096 tokens,
following the [upstream eligibility rules](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3).
Also keep block 0 dense in every native SubBlock variant so the probe path
remains consistent when composing features. Use sparse attention from index 10
on eligible suffix blocks. Index means the
original denoising schedule index, not a count of cache misses or kernel calls.
The full-budget diagnostic routes through existing dense attention exactly;
a separate test forces the new kernel to compute an all-selected mask and
compares its arithmetic to the independent masked-attention oracle.

The initial runtime capability gate is SM120. Upstream's
[SM120 integration](https://github.com/sgl-project/sglang/pull/37332) supplies
an adapter and numerical cases around FlashInfer's block-64 kernel. Audit and
pin the relevant implementation, ABI and licenses. Implement a compiled native
C++/CUDA path, reusing licensed code where suitable; do not add a Python worker
or a runtime JIT dependency. A Python/CuTe reference may inform tests but is not
the shipped inference path. No toolkit/runtime upgrade is part of this work.

Build with `CUDA_SUBBLOCK=1`, independent of Sage/SOL build options.
Missing build support or unsupported devices fail before loading the model.
Legitimate shape/warmup/protection fallbacks are counted; launch failures,
invalid route indices and nonfinite inputs remain errors. A run with only
dense fallbacks does not qualify the sparse implementation.

## Combining the two features

Always keep the adaptive probe block dense, including after SubBlock warmup.
On cache misses, execute the suffix with the attention policy selected for that
absolute schedule index. On hits, skip the suffix; there is no new attention
dispatch for those blocks. The transition to sparse suffix execution at index
10 invalidates any cached dense suffix and forces a refresh. A final suffix
refresh does not imply dense attention in a SubBlock variant.

Adaptive caching and SubBlock are separate request options, but existing
whole-velocity/core reuse cannot be enabled with adaptive caching. In recipe 1,
SubBlock also requires those old reuse controls to be 1. Reject combinations
with layer thinning, token reduction, Sage/SOL selection, projection quantization
or LoRA/Turbo in this initial experiment. Existing commands remain unchanged.

First/last-frame and ordered references require the packed-layout/protection
tests before their new-option route can be enabled. Continuation requires
separate bounded prefix/bridge correctness tests; retain an explicit error until
those pass. These checks do not authorize additional full comparison videos.
Still/image generation retains its existing restrictions.

## Interface and state

The implemented options are:

| Option | Values | Default |
| --- | --- | --- |
| `--adaptive-cache` | `off`, `conservative`, `aggressive` | `off` |
| `--cuda-attention` | Existing values plus `subblock` | Existing dense default |
| `--subblock-sparsity` | Finite number in `[0, 1)` | `0.75` when SubBlock is selected |

Keep warmup, block partition, hit limits and router geometry in versioned
presets for the first campaign. Reject SubBlock-only controls without SubBlock
selection, regardless of CLI argument order. Extend the public C parameters
with zero/off-compatible fields and append attention IDs without renumbering
existing values. Resolve and capture immutable policy before allocating contexts.

Add required checkpoint sections for active new recipes. Adaptive checkpoints
store the anchor/delta tensors, shapes, last committed evaluation, last refresh,
hit streak and attention phase. SubBlock stores its recipe, geometry, sparsity,
warmup and deterministic plan identity. Recompute routing from current Q/K/V;
do not persist stale routes as reusable attention results. Preserve existing
build/device/library/model checks and reject missing, corrupt or incompatible
required sections. Resume must reproduce subsequent decisions and AV latent
bytes, including boundaries around warmup, a hit, a forced miss and phase change.

Include recipes in prepared execution identity and completed-state provenance.
Do not invalidate unrelated encoder conditioning or duplicate immutable weight
caches. Completed AV decoding uses saved latents and must work without the
optional attention build. Default arithmetic identity and golden files stay
unchanged. Request cleanup must isolate dense→approximate→dense contexts.

## Fixed comparison protocol

### Shared workload

| Property | Required value |
| --- | --- |
| Model/task | Original H3 FL2VA model, text-to-video with generated audio; no reference inputs |
| Dimensions | 640×480 |
| Video | Exactly 90 decoded frames at 24 FPS; 3.75 seconds |
| Sampler | 50 scheduled evaluations/transitions, original video/audio sigma grids and Euler updates |
| Layers / precision | All 50 blocks, BF16 projections, existing sensitive FP32 operations |
| Seed | 42 |
| Decoder | Current full video/audio decoder and pinned codecs; preview off |
| Prompt | `Cinematic medium shot of a woman passionately playing a grand piano in a sunlit concert hall. Her fingers move across the keys as the camera slowly glides sideways. Warm natural lighting, realistic details, flowing piano music.` |

`--steps 50` must retain all 50 scheduler transitions; caching changes how many
blocks execute, not the schedule length. The article's 50-point example executes
49 transitions, so it is not the timing baseline for this campaign. The native
dense variant is the shared baseline for every comparison.

### Twelve variants, twelve comparison videos

| ID | Attention | Cache/reuse setting |
| --- | --- | --- |
| D0 | Dense | All reuse/cache off |
| R2 | Dense | Existing `--reuse 2` |
| R3 | Dense | Existing `--reuse 3` |
| C4 | Dense | Existing `--core-reuse 4` |
| C6 | Dense | Existing `--core-reuse 6` |
| A1 | Dense | Adaptive `conservative` |
| A3 | Dense | Adaptive `aggressive` |
| S75 | SubBlock 0.75 | Adaptive off |
| S80 | SubBlock 0.80 | Adaptive off |
| A1S75 | SubBlock 0.75 | Adaptive `conservative` |
| A3S75 | SubBlock 0.75 | Adaptive `aggressive` |
| A3S80 | SubBlock 0.80 | Adaptive `aggressive` |

Unspecified legacy reuse/core-reuse values are 1. No additional prompt, seed,
resolution, duration, repeated timing video or precision matrix is included.
The one D0 video is reused in every side-by-side view. Fix this manifest before
rendering; do not add or retune variants after viewing results.

Run all twelve on one frozen candidate source/build with all required features
available. Complete code-level qualification first. Run D0 first to establish
the baseline and bound disk/time needs, then use the declared table order.
Serialize GPU work; use the same CPU/NUMA placement, power settings and cache
preparation for every arm. Allow bounded kernel warmups and an untimed
metadata-based input/weight page-cache preparation for each arm, with their
costs recorded. Do not generate warmup videos, hash model weights or clear
machine-wide caches. Cache preparation must be identical across variants.

The dedicated campaign runner must set `H3_TEST_MAX_EVALUATIONS=50` only for
these fixed-shape comparison subprocesses; ordinary tests retain their existing
six-evaluation ceiling. Invoke that runner directly after building, or use a
target-scoped override. Do not change the global Makefile test limit or the
recorded regression's runtime environment. A per-render timeout is fixed before
the first run from a conservative estimate; timeouts remain failures.

Exactly one successful final comparison video is retained per variant. A failed
attempt may be repaired and retried, with its log and elapsed time retained;
never select the fastest or best-looking result from successful repeats. A
change to rendering or workload semantics invalidates comparisons tied to the
preceding build; document that explicitly before replacing an affected final
artifact. Reporting-only corrections retain the original render identities and
require the complete recorded regression, with both source identities recorded.

### Evidence and report

Save each MP4, clean AV state, command/environment record, source/binary/runtime
identities, run status and compact per-step telemetry under an ignored
`outputs/adaptive-subblock/<run-id>/<variant>/` directory. JSON/CSV and
HTML reports link the same twelve videos, with synchronized baseline and
candidate playback rather than rendering extra comparison clips.

Report process wall time, loading/preparation, denoising, full AV decoding and
delivery separately. Include GPU tensor/workspace peaks, sampled process/device
VRAM, host RSS/pinned memory and sampling gaps. Publish the observed ratio
`D0_time / variant_time`; one observation cannot supply a median, error bar or
reliable ordering of small timing differences.

Record 50 schedule indices and both sigmas; whole-DiT forwards; suffix refreshes
and hits/reasons; first-block and total block execution counts; change scores;
router/dense/sparse dispatches and fallback reasons; nominal versus actual
retained density; and router/kernel/transfer timing. Keep expensive QKV or
activation captures out of timed rendering. Bounded operator captures can be
replayed separately without generating another video.

Verify frame count, dimensions, FPS, complete decoding, stereo audio format and
AV duration using the existing delivery contract, including codec padding.
Compute every-frame SSIM/PSNR, LPIPS with pinned weights, temporal differences
and worst-frame views against D0. Measure audio waveform/spectral differences,
RMS/clipping/silence and AV timing; retain playable audio. Similarity metrics
describe deviation, not subject fidelity or a perceptual guarantee. Label human
review as pending unless it actually occurred.

## Correctness and exit criteria

The [mandatory recorded regression](../../CONTRIBUTING.md) remains a separate
test with its original fixtures, 204 hashes and 640×480 / 124-frame / six-step
render. Run it after each coherent code/tooling change and on final source.
The new 90-frame comparison does not replace or modify that gate. Its required
render is separate from the twelve-video comparison budget.

Use bounded operator, host and sampler fixtures for score reductions, threshold
ties, warmup/streak/final refresh, phase invalidation, packed masks, ragged tails,
route bounds, selected attention, memory limits and cancellation. Compare the
sparse kernel to an independent FP32 masked-attention oracle with fixture-specific
BF16 tolerances fixed before implementation; never relax them after failures.
Forced miss/cache off and shared-dense fallback must retain exact default output.
Use latent-only bounded replay for checkpoint tests; do not add full-render
resume videos. Test actual native dispatch, including a SubBlock hit and an
adaptive hit/miss where the fixture intentionally triggers each branch.

Build optional CUDA support both enabled and disabled, run relevant existing
CUDA tests, and build/test Metal after shared-interface changes. Feature-off
requests must allocate no new cache/router state. Architecture, mask or memory
failures must not be mislabeled successful approximate execution.

Completion requires working native implementations, passing correctness/default
regressions, all twelve valid final videos and a locally inspectable report.
Keep an explicit ledger of failed or incomplete work until resolved.
A slow or visibly different variant remains
reported as such; implementation success does not imply speed or quality
qualification. Recommend a preset only on the evidence produced here, and
keep both features off by default. Broader prompt, reference, hardware and
Metal quality qualification remains future work.
