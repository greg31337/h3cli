# Two CUDA execution modes on the SGLang baseline

Status: **historical design, superseded by [the single-pipeline design](design-single-pipeline.md)**.
M3B removes fast mode; the policies and examples below describe the earlier plan.
Tasks: [archived checklist](single-pipeline-tasks.md). The completed SGLang campaign is retained
in [its archived checklist](cuda-sglang-parity-tasks.md) and
[measured results](cuda-sglang-results.md).

## Objective

Ship exactly two CUDA execution policies:

| Mode | Selection | Contract |
| --- | --- | --- |
| Reference | Default; explicit `--cuda-reference sglang` remains an alias | Preserve the accepted native SGLang arithmetic and content on the frozen corpus. Original BF16 weights, dense attention and full original-model AV decoding form the parity preset. |
| Fast | `--fast-cuda` | Start from the same pipeline, then enable measured optimizations with visually acceptable video/audio. Dense, Sage2++, Sage3 or SOL attention; original BF16, FP8 or NVFP4 DiT projection weights. |

Fast is a set of explicit substitutions in the reference pipeline. It must not
be the previous renderer with reference corrections selectively copied into it.
Remove the former default/legacy CUDA renderer and its diagnostic execution
variant after all supported consumers have migrated. Metal remains a separate
backend and keeps its existing behavior.

Preserve original models, requested evaluations, seed interpretation, ordered
conditioning, AV timing, full transformer depth and sampler controls unless the
user explicitly selects a supported approximation. Preview VAE remains an
independent presentation option; do not replace or redesign it. No new model
architecture, Q4 weight option, or combined Sage-plus-SOL kernel is included.

## Baseline and current problems

Starting revision: `84d1555` (`fix VRAM budget for FP32 projection`). Record the
full revision and any working-tree delta in M0. Reference arithmetic is
`H3_SGLANG_VERSION=4`; old fast arithmetic is `H3_FAST_CUDA_VERSION=1`.
The development/qualification device is the qualification RTX PRO 5000 72GB at
the configured CUDA host; model path `/path/to/models/MiniMax-H3`. Re-probe hardware
and dependencies before implementation testing. No new model-weight hashing.

The migration used SGLang 0.5.20, PyTorch 2.13.0+cu130, reference cuBLAS identity
130101, audio-decoder cuDNN 9.10.2 and isolated encoder cuDNN 9.20.
The current [runtime guide](cuda-sglang-reference.md#build-and-runtime) uses
cuDNN 9.20 throughout with encoding in the main process; the oracle is unchanged.

The previous campaign established content parity on eleven complete cases,
including 124 frames / 50 evaluations. Its speed, memory-observation and old
fast-output exceptions were accepted explicitly. Those records remain historical
facts; their waivers do not authorize new reference drift or qualify fast v2.

At the starting revision, separation was architectural, not just a flag
(M1 changes are recorded separately above):

- `h3_sglang_requested()` returns false when fast CUDA is requested.
- `h3_sglang_resolve()` falls back to mode zero for attention, quantization,
  LoRA, preview, still and other settings. Mode zero means legacy arithmetic.
- `src/engine.c`, conditioning, sampler and delivery code branch on `cuda_reference`;
  fast currently inherits different RNG, schedule and preparation behavior.
- `src/cuda/gpu_cuda.cu` maintains `reference`, `sglang_reference`, `fast_requested`
  and `fast`. SGLang branches can bypass later optimized/quantized dispatch.
- `H3_CUDA_REFERENCE=1` is an older pedantic diagnostic, not SGLang parity.
  Missing kernels and older devices can currently fall back to the old renderer.
- State identities, model signatures, caches and CLI options encode these
  distinctions. Changing only the booleans would silently mix incompatible data.

The corrected patch projector is part of the baseline. Keep its 1 GiB bounded
row batching: a projection-only comparison found 11.15 ms at a 1 GiB cap versus
11.17 ms at 4 GiB, with 1.00 versus 2.27 GiB actually retained for the tested
large reference layout. This is not a useful optimization target without new
evidence. [Retained comparison](../../outputs/cuda-sglang/patch-cap-benchmark/result.json).

## Request policy and dispatch

Replace mutually exclusive recipe booleans with one immutable, resolved CUDA
request policy. Illustrative fields (names are implementation choices):

```c
mode                 // REFERENCE or FAST
base_recipe          // SGLang arithmetic identity 4
fast_recipe          // 0 or new fast identity 2
attention            // dense, sage2++, sage3, sol
projection_precision // bf16, fp8, nvfp4
presentation         // original full VAE, explicit full-VAE option, preview
```

Resolve and validate it before model allocations, LoRA folding or cache lookup.
GPU/component constructors capture it; nested scopes restore it on success,
failure and cancellation. Do not introduce a third policy through an environment
variable, unsupported-hardware fallback or a test-only production branch.

Both modes share tokenization, image/video/audio preprocessing, packing and
protection metadata, Torch-compatible initial noise, sigma construction,
FP32-sensitive state, Euler semantics, heads and the initial full-VAE recipe.
Initially keep Qwen and all conditioning encoders exact in fast mode as well.
Changing a fast DiT kernel must not change reference preparation or decoding.

Use explicit stage dispatch: request validation, selected fast override if
eligible, then the shared reference implementation. The fallback is a reference
primitive inside the selected policy, never the removed renderer. An explicit
Sage/SOL/quantization request with an unsupported device/build must fail clearly
before expensive preparation; only documented shape fallbacks within a supported
backend may use dense reference kernels, with reason and counters reported.

Do not simply make both old `g->fast` and `g->sglang_reference` true. Audit the
ordering of linear, QKV, attention, VAE and weight-descriptor dispatch so an
early reference return cannot ignore fast options. Isolate cuBLAS handles,
algorithm/workspace policies and cached plans where fast choices differ.

### CLI and feature behavior

| Request | Planned behavior |
| --- | --- |
| No CUDA mode option | Reference mode. |
| `--cuda-reference auto` or `sglang` | Reference; `auto` may coexist with explicit `--fast-cuda`, while explicit `sglang` conflicts with it. |
| `--fast-cuda` alone | New fast dense BF16 preset; only optimizations qualified in this project enabled. |
| Sage/SOL or FP8/NVFP4 without `--fast-cuda` | Actionable error asking for explicit fast mode; no implicit loss of parity. |
| `--cuda-reference legacy` or `H3_CUDA_REFERENCE` set to a nonzero legacy diagnostic | Reject with migration guidance; never execute or silently reinterpret it. |
| `--preview-vae` / explicit full-VAE execution | Orthogonal decoder selection, with its own capability checks. Keep reference denoising when fast is absent; label nonstandard delivery as outside full-output SGLang parity. |
| LoRA/Turbo or existing reuse/reduction controls outside the parity preset | Support through explicit fast mode; preserve requested settings and log their recipes. Do not retain legacy merely to service these options. |
| CUDA still generation / direct image decode | Migrate to common CUDA infrastructure with their task-specific image-VAE contract. Keep existing restrictions until explicitly qualified; this task is not SGLang still-parity qualification. |
| Saved sampler / conditioning / AV states | Apply the version and compatibility rules below. No implicit arithmetic migration. |

Preview/full-VAE selection must not change RNG, schedules, reference encodings
or sampled latents. The existing `--full-vae-execution reference` names a decoder
policy, not a third CUDA mode. Preserve any shared/Metal decoder implementation
it needs; remove the global legacy CUDA diagnostic that previously forced it.
Log mode, base/fast recipe, attention, quantization, decoder and effective fallback.

Example final interface (commands are the target, not current compatibility):

```sh
./bin/h3cli -d /path/to/models/MiniMax-H3 -p "$PROMPT" --frames 124 --steps 6
./bin/h3cli -d /path/to/models/MiniMax-H3 -p "$PROMPT" --frames 124 --steps 6 --fast-cuda
./bin/h3cli -d /path/to/models/MiniMax-H3 -p "$PROMPT" --frames 124 --steps 6 \
  --fast-cuda --cuda-attention sage2++ --cuda-denoise-quant fp8
./bin/h3cli -d /path/to/models/MiniMax-H3 -p "$PROMPT" --frames 124 --steps 6 \
  --fast-cuda --cuda-attention sol --cuda-denoise-quant nvfp4 --preview-vae
```

## Frozen reference regression gate

**Run the complete reference regression after every code change.** Preserve
recorded fixtures, output hashes, case coverage and evaluation counts. Acceptance
depends on the recorded golden state; source and binary digests identify the run.
No candidate failure permits rebaselining, relaxing comparisons, skipping a case
or reusing a result from an older source tree. The existing full SGLang contract
remains unchanged.

“Every code change” means each coherent implementation patch, including changes
to CUDA/host code, generators, shaders, build flags, libraries and test harnesses;
run the complete frozen gate on that exact tree before starting the next unrelated
patch. Fix a failure and rerun the whole gate. Documentation-only edits require
link/checklist validation rather than a GPU run. A timeout, missing fixture or
unavailable GPU is an incomplete gate, not a pass or permission to skip it.

Use `make test-cuda-reference-regression` with the environment settings in
[CONTRIBUTING.md](../../CONTRIBUTING.md). Record source/build hashes, case results,
durations and the golden-manifest digest in `outputs/cuda-reference-regression/<run>/result.json`.
Build in an isolated directory and detect source changes during execution.

Record the expected fixture and output hashes in a reviewable golden manifest.
Run the full regression on the current source and retain its result before
further implementation. Build/runtime paths are caller configuration, separate
from expected numerical results. Record source, binary, fixture and output
identities without scanning or hashing model weights.

Keep mutable CLI migration/new-fast tests separate. They cannot replace or alter
the frozen reference suite. Establish stable production CLI/API probe entry points
before freezing; fixtures must execute the real production kernels and sampler,
not a mock or a duplicate implementation maintained solely to pass tests.

### Per-change scope and time budget

Target **5–10 minutes**, hard run limit **12 minutes** on the pinned RTX PRO
5000 after build. Measure this in M0 before freezing. Speed up fixture loading
and reporting before freeze if necessary; do not silently remove coverage later.
Do not reinstall SGLang or rerender its corpus each time. Validate against retained
oracle-qualified goldens and the frozen native reference build.

| Gate | Coverage | Target allowance |
| --- | --- | ---: |
| R0: host and identity | CLI reference selection, exact tokens/packing/positions, RNG and all sigma values for 6 and 50 evaluations, Euler, test-lock and state identity checks | 45 s |
| R1: CUDA operations | BF16 rounding/norm/RoPE/linears, FP32 patch/head projections, dense short/medium attention; real early/middle/late-step tensors; the six large patch-batching cases including 1344×768 / 362-frame row geometry | 90 s |
| R2: conditioning and delivery | Fixed image `match`/`max`, video+audio preprocessing, vision/CNN/audio encoder probes, full VAE tile and audio decode; saved-latent delivery/preview independence | 120 s |
| R3: complete reference render | Original C0 prompt/seed, 640×480, 124 frames, six actual evaluations, all 50 blocks, full AV VAE; native inputs, every sampler update and complete decoded media | 240 s |
| R4: isolation and artifacts | Small reference→fast→reference contexts, repeated allocations/cancellation, completion counts and local artifact checks | 45 s |

Allowances total nine minutes plus one minute of normal overhead; the additional
two minutes are failure/host-variance allowance, not extra scheduled cases.
No full 1344×768/362-frame render belongs in this gate. Production row-count
bugs are covered by bounded operator tests instead.

Reference retention is stricter than visual similarity: compare exact integer
and RNG data, every C0 AV update and all repeatable frozen native outputs
bit-for-bit. Compare decoded pixels/PCM rather than MP4 container metadata.
If a selected operation is demonstrably nondeterministic on the pinned baseline,
establish its narrow repeatability envelope in M0 before any refactor; reuse the
existing SGLang numerical limits as an upper bound, never as permission to lose
already-proven exactness. Fail nonfinite values, missing steps and stale artifacts.

R3 must use native conditioning/noise and the production sampler; imported oracle
inputs, a CPU-sampler substitute or a teacher-forced trajectory cannot stand in
for it. Operator replays remain separately labeled. Keep checks from late C2
steps 25 and 49 in R0/R1 so the short routine gate exercises long-schedule
boundaries. These probes do not establish a new complete 50-step trajectory.

## Rebuild fast CUDA from the shared pipeline

First implement fast v2 with **all substitutions disabled** through test-only
policy configuration. It must reproduce the reference fixture outputs exactly,
including preparation, sampler, velocities, full VAE and delivery. This is a
construction test, not a third user-visible mode. Do not start approximate
optimization before it passes.

Then profile representative complete requests and introduce one optimization at
a time. Prefer reductions in transfers, redundant conversions, allocation and
synchronization before arithmetic changes. Bring over old fast GEMM tuning,
cooperative norms/RoPE, resident/streamed weight caching, attention or decoder
work only as individually measured substitutions. Preserve stream ownership and
release fences; do not globally change reference compiler/math/workspace flags.

Keep FP32-sensitive residual/sampler/head behavior and original BF16 checkpoint
storage unless a fast-specific change passes its own quality gate. Initial fast
full decoding starts from the SGLang range-safe FP16/FP32 VAE, not automatically
from the older TF32 decoder. Preview and explicit full-VAE alternatives stay
independent and are tested against identical saved latents.

Qualify fast defaults against the accepted SGLang reference, not against legacy
fast images. Freeze a separate fast-quality contract before optimization, with
per-frame perceptual/temporal limits, reference identity/action checks, audio/AV
checks and supported presets. The reference contract is never weakened to admit
fast results. Produce paired full clips and worst-frame views; optional human
notes supplement the results and are not a required waiting step. Report unreviewed
visual status honestly; automated metrics alone do not establish every perceptual
property.

Promote an optimization only with repeated matched measurements demonstrating a
gain beyond noise in denoising or total wall time, no unexplained regression in
the other boundary, bounded VRAM/RAM and acceptable quality. Target at least a
5% gain for a default substitution; retain smaller effects only with an explicit
measured combined benefit. Record rejected candidates. Historical process-time
advantages caused by SGLang cleanup are not fast-kernel improvements.

## Attention and quantization integration

The [M3A results](two-modes-m3a-results.md) qualify private mapped pinned DiT
weight staging and one Qwen prefetch lane for shared CUDA. Metadata checks, the
40 GiB host-cache cap, copied/bounce fallbacks and explicit prefetch overrides
remain. Reference full-VAE cast fusion is deferred: its isolated gain did not
reach the promotion threshold. The final source passes all 204 frozen artifacts;
reference arithmetic remains unchanged. The frozen gate must still run after
every code change. See the [assessment](two-modes-m3a.md) for the original contract.
Existing M2/M3 conditioning and approximate-attention qualification failures
remain open; they are not waivers for reference changes.

Attention choices are mutually exclusive and independent of projection precision:

| Fast main-DiT attention | BF16 projections | FP8 projections | NVFP4 projections |
| --- | --- | --- | --- |
| Dense | Required | Required | Required |
| SageAttention 2++ | Required | Required | Required |
| SageAttention 3 | Required | Required | Required |
| SOL | Required | Required | Required |

Reuse existing vendored Sage and CUDA SOL implementations after auditing their
interfaces against the shared reference QKV norms, RoPE, scales, BF16 storage,
packed strides and projection-compatible output. H3 main attention is 56 heads
of dimension 128; it is not the eight-head microbenchmark. Explicit main-DiT
entry points prevent accidental substitution in Qwen, refiners, encoders or VAEs.

Sage precision is internal to attention; selecting Sage3 does not automatically
quantize projection weights. SOL remains its existing routing/summary algorithm,
not a Sage/SOL hybrid. Protect text, reference images/video/audio, conditioning
prefixes and continuation ranges; preserve absolute step/block indices and both
noise sigmas. Dense protected/early regions and failed routing use the shared
reference dense implementation. Start from the current conservative CUDA SOL
defaults, including `min_exact=0.75`; tuning is a separate fast-quality decision.

FP8/NVFP4 apply only to eligible repeated DiT QKV, output and MLP projection
families. Keep FP32 velocity heads, patch projections, norms, encoders and VAEs
outside this weight policy. Packed weights/scales must dispatch by descriptor
before any unconditional BF16/reference GEMM branch. Preserve BF16 outputs and
audited accumulation. LoRA folding precedes packing; cache keys include the
effective fold, weight metadata, format, recipe and kernel requirements.

Keep cached quantized variants persistent, metadata/version validated and atomic.
Reference requests must never read a packed-weight cache or trigger its creation.
Share a packed payload between attention choices only when its recipe is actually
identical. Architecture/build checks and fallback rules are explicit; do not
claim support for untested devices merely because a common build succeeds.

## State, memory and legacy removal

New fast behavior requires fast identity **2**. Preserve reference arithmetic
identity **4** only if the unchanged reference gate passes. Changes to outer
policy/state schemas have separate versions; schema migration is not an excuse
to change reference arithmetic. Cache/plan identities include base recipe, mode,
fast recipe, attention/quant recipes, SOL settings, conditioning/model identity,
layout, dtype, device and relevant library/workspace choices.

Reference and fast requests may share truly immutable inputs only with complete
identity checks. Mutable tensors, RNG state, plans and scratch remain isolated.
Test reference→fast→reference and reverse order, cache hits/misses, cancellation,
error recovery, streaming/residency and decode-only. Keep the 110 GB process guard,
bounded host weight staging, 1 GiB patch buffer and existing attention/VAE budgets.
Reserve activation/workspace headroom before optional weight-cache admission.

Old fast-v1 or legacy CUDA mid-sampler states cannot be resumed as fast-v2 or
reference. Reject them with a precise restart/export message. Preserve immutable
reference state payload semantics; existing same-build/device/environment checks
still apply. Rebuild only versioned derived caches that can safely be regenerated.
Clean AV/image latent containers are data, not permission to revive an old execution
mode: retain shared readers and validate an explicit supported decoder identity,
or report that re-export with the archived executable is required. Do not add
model hashing to recover an old content signature.

Remove `--resume-default-cuda` as a CUDA arithmetic-switch mechanism and remove
old CUDA policy fields/extension writers when no current consumer needs them.
Minimal version detection/error reporting is allowed; an executable legacy
compatibility backend is not. Preserve shared Metal/container security code.

Make a reachability inventory before deletion:

1. Delete old default/legacy dispatch, PCG/old-schedule selection in CUDA,
   `H3_CUDA_REFERENCE` execution, old fast pipeline orchestration, old-only cache
   identities and state writers, fallback chains and exclusive build/dependency
   options. Update generated wrappers at their generator, not only emitted code.
2. Migrate generic kernels still used by the SGLang base, new fast substitutions,
   image VAE or explicit decoder policies into named shared primitive modules.
   Retaining a used primitive is different from retaining an old renderer; prove
   its consumers and numerical contract. Do not delete files solely because
   their names contain `reference`.
3. Preserve Metal/CPU helpers needed outside CUDA. Migrate CUDA still, reference
   media, continuation, LoRA, preview and explicit full-VAE consumers before their
   old dependencies disappear. A missing feature must not quietly route to legacy.
4. Remove obsolete legacy execution tests/tools and replace CLI/state migration
   tests separately. Preserve the frozen parity suite, historical artifacts,
   attribution and archived qualification reports. Old binaries may remain as
   external historical evidence, not as a buildable/shipped third mode.

## Bounded milestone validation and delivery

Every implementation patch runs the frozen R0–R4 gate. In addition, use bounded
mode-specific tests at milestone boundaries; do not multiply full renders by
every development iteration.

- Cover all twelve attention/precision combinations with operation/layout,
  descriptor, capability and short sampler smoke tests. Test positive kernels,
  negative build/device cases and mixed references; an unsupported environment
  must not make a required combination silently pass by falling back.
- Qualify complete 640×480 / 124-frame / six-evaluation clips for all twelve
  supported fast combinations, plus the immutable reference. Use a frozen
  reference-conditioned case and paired reports. Add no-reference, image `match`,
  image `max`, mixed image/video/audio and continuation held-outs for the proposed
  defaults and recommended presets; avoid an exhaustive cross-product.
- Use 640×480 / 243 frames / six evaluations for matched fast performance.
  Reuse frozen reference artifacts where valid, but use fresh interleaved pairs
  for performance claims. Report complete wall, denoising, conditioning, full
  decoding, time-to-playable, sampled VRAM and host/pinned/swap measurements.
- Limit any 1344×768 full-render test to **at most two evaluations**. Use one
  362-frame `max` image-reference stress case for the selected default fast
  preset; the per-change reference guard covers its large projection separately.
  Two-step videos qualify execution and inspection only.
- At final cleanup, replay the retained C1 362-frame/six-evaluation and C2
  124-frame/50-evaluation reference cases once to catch accumulated drift after
  legacy removal. C2 is a specific final exception to the normal six-evaluation
  test limit, not a new per-change cost. No 362-frame/50-evaluation campaign.

Preflight and time the milestone matrix before launch; target at most four hours
per focused campaign on the local node, including failures, validation and report
generation. Stop with explicit incomplete coverage if the estimate/budget is
exceeded; do not silently reduce evaluations or waive a failed test. This is
separate from the mandatory, much shorter per-change gate.

Deliver source/build/dependency identities, recorded golden-state manifest and
pass history, fast quality/performance decisions, all twelve combination results,
memory/fallback counters, version-migration instructions and local HTML playback
pages with audio, commands and worst-frame comparisons. A closeout audit must
prove only reference and fast policies remain reachable on CUDA. Document the
scope of measured hardware and visual quality without converting historical
acceptance or successful compilation into new qualification.
