# Native adaptive and SubBlock implementation contract

This supplements [the design](design-adaptive-cache-subblock.md). Both options
remain off by default. The twelve cases in
[`adaptive-subblock-manifest.json`](adaptive-subblock-manifest.json)
are the fixed experiment, not additional reference goldens.

For conservative cache with FP8/NVFP4, use the separate
[mixed-precision recipe](adaptive-quant-experiment.md#mixed-precision-recipe).
The arithmetic below is unchanged; that extension adds explicit checkpoint
identities and keeps the probe block BF16.

For SubBlock with FP8/NVFP4 and adaptive cache off, see the separate
[SubBlock/quantization composition](subblock-quant-experiment.md). Its attention
kernel and plan remain the recipe-1 arithmetic defined here.

## Source audit

The audited distribution is SGLang 0.5.20 (Apache-2.0), including
`multimodal_gen/runtime/layers/attention/backends/subblock_sparse/`:

| File | SHA-256 |
| --- | --- |
| `router.py` | `3092f904de0d95502d16389d9cb4023de2d6fa70f44fcfa0ce546ce429258f4b` |
| `kernels.py` | `7b8b2943ed05f13a380aa1c1843973cf3b7c3c7af6d34d6fce74cb7958bf663f` |

The SM120 reference ABI is FlashInfer 0.6.18's
`cute_dsl/sparse/bsa_attn_sm120.py` (Apache-2.0), SHA-256
`8e058085a7c7f999a0e23f716778ff2c4a20457cb5556aec41256375d22f9ff6`.
It accepts BF16 `[batch, sequence, heads, 128]`, int32
`[batch, heads, query_blocks, selected_keys]` indices, optional per-query counts
and true key-block lengths. It returns BF16 sequence-major output. These package
versions and content digests pin the audited source; no claim of an installed
Git checkout or a different development revision is made.

The native implementation does not import that Python/CuTe runtime. Its tensor
core primitives may derive from the existing MIT-attributed CUDA SOL kernel;
retain the original attribution when doing so.

## SubBlock arithmetic

Blocks contain 64 tokens and four consecutive 16-token cells. Pool each channel
in FP32 over **real rows**, divide by the true count, and round to BF16. Query
pooling multiplies by `scale * log2(e)` **before** BF16 rounding; key pooling
does not scale. Empty cells do not contribute. Compute the 16 cell-pair dot
products using BF16 operands and FP32 accumulation, then stable base-2
log-sum-exp across valid key cells and valid query cells, converting the result
to natural-log units with `ln(2)`. A partially occupied cell has equal cell
weight, not a count multiplier. This matches the audited estimator; it is not
the exact softmax mass of a ragged block.

Scores remain FP32. Retain `min(B, 8 * ceil((1-sparsity)*B/8))` key blocks.
Native selection is exact and deterministic, breaking equal scores by smaller
key index, unlike upstream's approximate threshold search. Sort selected indices
and union protected keys without duplicates. Protected/mixed query blocks use
all keys. These protection rules and always-dense block 0 are intentional
native differences. Actual per-row attention visibility must still be honored;
an unqualified visibility layout must be explicitly rejected or use documented
dense fallback. Invalid values and execution failures are errors.

Selected tiles use BF16 Q/K/V, FP32 softmax and accumulation, and BF16 output.
No centroid contribution represents omitted keys. By default, absolute denoising steps 0–9,
block 0, sequences shorter than 4096 and full retained budgets use existing
dense attention. Forced all-selected kernel testing is a separate operator path.

## Adaptive arithmetic and transactions

Run block 0 densely every evaluation. `probe = BF16(block0_output - input)`.
The anchor is the probe from the most recent successful suffix refresh.
Compute `mean(abs(FP32(probe)-FP32(anchor))) /
max(mean(abs(FP32(anchor))), 1e-6)` over every valid packed channel with a fixed
FP32 reduction tree. Text-only recipes 1 (BF16) and 2 (quantized suffix) use
the global score. BF16 reference recipe 3 selects the maximum of the global,
generated-video and generated-audio scores; target ranges exclude condition
prefixes and include both generated audio channels. A hit requires strictly
less than the effective threshold.
Do not accumulate scores or update the anchor after hits.

Conservative uses threshold 0.04 and at most one consecutive hit; aggressive
defaults to 0.08 and at most three. `--adaptive-cache-threshold` overrides the
threshold with a finite decimal FP32 value in [0,1]; `--adaptive-cache-max-hits`
overrides the ceiling with an integer in [1,16]. Explicit zero threshold disables
hits. Controls require enabled caching, are canonical in execution keys, and
must match saved values on resume. Both refresh steps 0–3 by default, an empty cache,
the last step, and any recipe/layout/attention-phase change. The configured
SubBlock transition (default step 10) forces a refresh when both are active.
The [warmup controls](approximate-warmup.md) allow explicit counts 2–16, at most
`steps-2`; omitted counts retain the fixed defaults even on short schedules.
A miss records
`delta = BF16(suffix_output - block0_output)`; a hit reconstructs
`BF16(block0_output + delta)`. Both final heads remain fresh. Nonfinite probes
are errors. Only a successful forward and Euler transition commits history;
failed work cannot be serialized as a successful transition.

The original experiment used a 512 MiB adaptive ceiling. New requests now
use a configurable 4096 MiB default through `--adaptive-cache-max-mib N`, as
specified by the [budget contract](adaptive-cache-budget-design.md). Exact
storage includes two persistent BF16 tensors, one BF16 scratch tensor and
6,168 bytes of reduction scratch. Disabled cache allocates nothing. Current
adaptive checkpoints require section 40 v3 with explicit controls and budget;
older versions are rejected. Changing an admitted ceiling does not change the
arithmetic above. Presentation 9 records controls separately from cache history.
SubBlock uses at most the existing 512 MiB attention allowance, tiled over
heads/query blocks, never a token-by-token score matrix.

Adaptive caching and SubBlock, independently or together, support ordered BF16 image, video and audio references,
including mixed sets, embedded audio and external soundtracks. Existing limits
remain: 12 total references, 9 images, 3 videos, 3 audio inputs, and 65,536 raw
patches per image. Standalone audio requires a visual reference. Text,
vision-conditioning and all image/video/audio reference rows are protected, as are the
first and last generated frames. Every 64-row block intersecting those ranges
is protected; protected keys are retained regardless of router score and
protected queries use shared dense attention. Reference count, order, identity,
size mode and canvas participate in conditioning/prepared keys. Sampler resume
reconstructs protection from the validated saved layout and checks ordered
media-kind/audio provenance. See the [qualification record](adaptive-budget-reference-results.md).
Reference quantization and anchors remain excluded. Reference
latents and condition timesteps stay fixed. Every executed first block and both
final heads remain fresh; hits skip all suffix attention and streaming. See the
[reference controls design](design-adaptive-cache-controls-references.md) for
reference scoring, transaction and state validation details.

## Continuation recipe 4

BF16 hard/bridge continuation works with either feature and their combination,
including all supported reference kinds. It keeps reuse/core reuse 1, all 50
layers and ordinary geometry; quantized approximation with continuation, first/last anchors,
upscale, LoRA and Metal approximation remain excluded.

The probe takes the maximum score over global rows, generated-video suffix,
generated stereo-audio suffix, and each active nonzero bridge class per modality.
There are 23 bounded slots: global, two suffixes, ten video classes and ten audio
classes. Frozen prefix rows contribute only to the global score. Unit-strength
bridge prefix rows remain distinct from suffix rows. Empty classes score zero;
the denominator floor and full-tensor nonfinite checks apply independently.
Recipe 4 uses a separate fixed-order FP32 reduction with 47,288 scratch bytes;
ordinary recipe arithmetic and 6,168-byte scratch are unchanged.

SubBlock protects the entire prefix, including mutable bridge rows, along with
all existing protected and mixed blocks. Protected attention does not promise
exact hidden states: cached suffix residuals still approximate them. The Euler
transition preserves initialized exact latent bits and scales raw bridge
velocity once. Cache history commits only after successful heads, update and
exact-row audit. Hits dispatch no suffix router, attention, MLP or weights.

New AV segments reset cache history. Same-job sampler resume restores it with
absolute-schedule warmup and sparse-phase refresh. Current sampler and
presentation schemas remain unchanged; recipe 4 identifies the new arithmetic.
Prepared keys bind source identity, mode, context, effective bridge settings,
references and controls; inactive bridge tuning is ignored in hard mode.
The [design and comparison protocol](design-adaptive-continuation.md) defines
the bounded qualification and separate human visual/listening acceptance.

## Original 12-variant experiment qualification

The protocol below records the original adaptive/SubBlock experiment. The
continuation extension uses the separate S01/S02 and V01–V14 protocol in its
[design](design-adaptive-continuation.md#fixed-video-comparison) and
[results](adaptive-continuation-results.md).

Independent CPU oracles use explicit BF16 rounding and FP32 arithmetic; dense
masked attention additionally uses FP64 accumulation as a check. Fixtures cover
zero, alternating signs, deterministic random values, extreme finite values,
NaN/Inf, ties, ragged lengths 1/15/16/17/63/64/65/127/129, protected and mixed
blocks, both output layouts, alias rejection, empty routes and exhaustion.
Keep synthetic fixtures below 16 MiB each and operator captures below 512 MiB.
Score tolerance is absolute 2e-4 plus relative 2e-5 for non-tied ordinary inputs;
require exact selected sets when the oracle margin exceeds twice that bound.
Tie fixtures require exact index ordering. Attention requires relative L2 <=
0.01 and max absolute error <= 0.0625 for unit-scale fixtures; zero and constant
fixtures must satisfy their exact mathematical invariants. Extreme cases must
remain finite or explicitly reject overflow. Do not change these bounds in
response to failures without documenting a recipe revision and restarting
qualification. Disabled and full-budget bypass results must be byte identical.

Campaign subprocess timeout is **7200 seconds per variant**, fixed before D0.
Serial runs use identical environment and metadata/page-cache preparation, with
no extra video warmup. Preparation stats the model metadata and advises the first
1 MiB of each FL2VA weight file with `POSIX_FADV_WILLNEED`; full weight paging
remains timed. It does not hash weights, evict system caches or promise physical
residency. A single process includes load, encode, all 50 scheduler
transitions, full decode, codec and state write in wall time. Report stage
times separately. Sample process RSS and device memory once per second and
record missed samples; GPU allocation counters supplement this sampling.
Per-step records include both sigmas, forward/block counts, cache decisions,
scores and reasons, dense/sparse/router calls, selected/possible block pairs,
transfer counters and timings. Timed runs disable diagnostic tensor capture.
Exactly twelve successful outputs are allowed; failed attempts remain visible.
Visual/audio metrics report differences, not parity or human approval.
