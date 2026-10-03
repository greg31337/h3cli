# Adaptive cache and SubBlock for continuation and bridge

Status: **implemented, automatically qualified and accepted by the user**.
See the [acceptance record](adaptive-continuation-acceptance.json).
Design date: 2026-09-28. The implementation checklist is [../todo.md](../todo.md).
The accepted reference-media extension is preserved in its
[design](design-adaptive-cache-controls-references.md),
[completed checklist](../adaptive-controls-references-todo.md), and
[results](adaptive-controls-references-results.md).

## Objective and scope

Allow the existing adaptive-cache and SubBlock flags with `--continue-from`
in both `--continue-mode hard` and `--continue-mode bridge`. Support each
feature independently and together, including ordinary image, video, audio,
and ordered mixed Ref2VA conditioning. Continue to generate joint video/audio.
Keep the existing prefix initialization, bridge strengths/timesteps, sampler,
reference preprocessing, decoding, and delivery semantics.

This is a BF16 CUDA extension for the existing qualified SM120 stack. Both
features remain opt-in. No new public flags or presets are required.

| Continuation execution | New support |
| --- | --- |
| Adaptive conservative or aggressive, dense attention | Hard and bridge; existing threshold/hit/warmup/budget controls |
| SubBlock alone | Hard and bridge; existing sparsity/warmup controls |
| Adaptive conservative or aggressive plus SubBlock | Hard and bridge; both sets of controls |
| Each row with reference media | Existing image/video/audio/mixed forms and limits |

For these new combinations require `--reuse 1 --core-reuse 1 --layers 50`,
token reduction off, BF16 projections, and no LoRA. Preserve all ordinary
continuation options and supported approximation combinations when this
extension is not selected. In particular, do not remove existing continuation
reuse 2/3 or bridge core reuse 4/6 support when adaptive and SubBlock are off.

First/last-frame anchors, quantized adaptive/SubBlock continuation, new Metal
approximation kernels, still generation, and upscaling are outside scope.
Upscale continuation is a separate project; neither upscale-source capture nor
refinement admission changes here. Existing text-only FP8/NVFP4 approximation
and ordinary quantized continuation behavior must remain unchanged. No new
quantized comparison campaign is required.

Reference support includes `--ref-image`, `--ref-silent-video`, `--ref-video`
with embedded audio, `--ref-video-audio VIDEO AUDIO`, and `--ref-audio` with a
visual reference. Preserve order, visual/audio counting, duration bounds,
image-size modes, RoPE, and the current visual-reference requirement for
standalone audio. Continuation history is distinct from those reference inputs;
it does not replace the visual reference required by an audio-only Ref2VA request.

## Pre-implementation audit

This section records the starting point. The shared validator, recipe-4 probe,
prefix audit and current-schema reader changes are now implemented. The
[native contract](adaptive-subblock-contract.md#continuation-recipe-4) describes
the resulting behavior; the checklist records qualification evidence.

- [approximate.c](../../src/denoise/approximate.c) rejects any continuation pointer
  when adaptive caching or SubBlock is selected. The source AV state is loaded
  after initial CLI preflight, so shared engine validation must remain authoritative.
- [dit.c](../../src/denoise/dit.c) already supports prefix/bridge modulation, fresh
  final heads, prefix-preserving Euler updates, and exact bridge-row audits.
  Adaptive hits reuse the BF16 residual of blocks 1–49 after a fresh dense
  block 0. The existing probe only distinguishes global/video/audio ranges.
- [sol.c](../../src/denoise/sol.c) constructs the protection metadata consumed by
  SubBlock. It already protects the full video prefix, all audio, reference
  and text rows, boundary video frames, and blocks straddling those regions.
- [sampler_file.c](../../src/sampling/sampler_file.c) explicitly rejects continuation,
  bridge, or nonzero prefixes before allocating adaptive tensors.
  [sampler_state.c](../../src/sampling/sampler_state.c) also treats every saved prefix
  as a first-frame anchor for approximate validation. Those are different layouts
  and must be validated separately.
- The current CUDA SGLang arithmetic recipe selects CPU-state F32 Euler;
  DiT and VAEs execute on the GPU. Keep that selection, transaction order, and
  existing sampler-mode restrictions. Historical continuation docs and some
  source comments describe different defaults; use current dispatch and checks
  as authority. This project does not add a GPU-state adaptive sampler.

Removing only the CLI guard would leave unsafe score aggregation and unreadable
checkpoints. Admission, packed-layout semantics, policy identity, and saved-state
validation must be extended together.

## Shared layout and admission contract

Build one validated continuation description usable by fresh generation, DiT
creation, checkpoint loading, and presentation validation. Distinguish ordinary
references, FL2VA anchors, hard prefixes, bridge profiles, and frozen-audio
upscale layouts explicitly; do not use a fabricated `first_frame` string to
stand in for every prefix.

Validate existing geometry/model/VAE compatibility, context `39 + 51*k`, source
history length, and a nonempty generated video/audio suffix. Keep the current
geometry and bridge context/row limits. Validate mode, context, target prefix,
bridge class tables, masks, and effective timesteps against each other. Hard
mode continues to ignore bridge tuning; inactive bridge fields cannot alter its
cache identity or decisions. Reject unsupported combinations through the same
policy at CLI, library, resumed-state, and prepared-context entry points.

Resolve target video and audio segments from authoritative packed metadata.
Condition/reference rows must never enter the target regions. Target stereo
audio consists of a left tick range followed by a right tick range, not
interleaved channels. Validate counts and offsets before reading class tables.

## Adaptive continuation score: recipe 4

Preserve existing recipes 1 (text BF16), 2 (text quantized), and 3 (BF16 reference
media without continuation), including their arithmetic, reduction ordering,
default decisions, and scratch sizing. Add **recipe 4** for all supported BF16
continuation layouts, with or without references and in either continuation mode.
Recipe selection must inspect validated continuation metadata; the pointer in
`h3_params` is insufficient after checkpoint restoration.

Keep the existing residual arithmetic:

```text
probe = BF16(block0_output - block0_input)
score(region) = mean(abs(FP32(probe) - FP32(anchor)))
                / max(mean(abs(FP32(anchor))), 1e-6)

refresh: anchor = probe
         delta = BF16(block49_output - block0_output)
hit:     output = BF16(block0_output + delta)
```

Recipe 4 selects the maximum of:

1. The global score across all valid packed rows, including conditioning and history.
2. The generated-video suffix score, excluding every continuation prefix row.
3. The generated-audio suffix score, excluding prefix ticks in both stereo channels.
4. For bridge mode, one score for each nonempty, nonzero-strength bridge class,
   separately for video and audio. Use the existing ten strength classes per stream.

Frozen rows contribute only to the global component. Zero-strength bridge rows
are treated as preserved rows, even if an internal temporal table assigns them
a bridge class ID. Empty classes are excluded from the maximum and explicitly
marked absent in diagnostics; both generated suffix regions must be nonempty.
Audio class scores include the corresponding ticks from both channels.

Do not multiply class scores by bridge strength or average the class scores.
Otherwise a short changing bridge region could be hidden by long frozen history,
many generated video rows, or a small strength. The global component continues
to monitor hidden-state changes in conditioning and preserved rows: frozen
latents do not imply constant hidden states under joint attention.

Use deterministic FP32 reductions with a fixed order and no atomic floating sums.
There are at most 23 scored components: global, two generated classes, and
20 bridge classes. Reuse compact temporal class metadata where possible; avoid
a token-by-token attention matrix or a separate full-size hidden tensor for
each class. Check nonfinite input/anchor values across the entire packed tensor,
including frozen and condition rows, even on a mandatory-refresh step.

Threshold and hit-ceiling semantics stay unchanged: strict `score < threshold`,
zero threshold means no hits, and threshold 1 does not guarantee hits. Defaults
remain conservative 0.04/1 and aggressive 0.08/3. A more conservative score may
reduce hits; record that outcome rather than silently increasing thresholds.

## Refresh, masks, and transactions

Retain dense block 0, fresh final video/audio heads, original-schedule warmup,
mandatory final refresh, maximum-hit refresh, discontinuity handling, and the
dense-to-SubBlock phase refresh. Never advance the anchor on a hit. Resume uses
the original schedule index, not a new warmup counter.

Every new continuation segment begins with an empty adaptive cache, even when
the source was generated with adaptive caching. `.h3av` carries final latents,
not reusable intermediate hidden history. Only a validated same-job
`.h3sample` resume restores anchor/delta/history. A mode, context, bridge profile,
source content, prompt/reference layout, model, numerical policy, or schedule
change must invalidate incompatible prepared execution and live cache state.

On every hit or refresh, apply the existing sampler logic to current head outputs:

- Hard continuation skips all preserved latent positions during Euler updates.
- Bridge scales raw velocities once by the existing class strengths, updates
  mutable classes, and skips exact classes. Do not scale a cached suffix residual
  as if it were a velocity, scale a velocity twice, or blend the protected history.
- Preserve bit patterns of exact video/audio rows relative to the **initialized
  target prefix**, including signed zero. Existing clean/noise initialization
  remains authoritative; this is not a promise that initialized video history
  equals the clean source tensor byte for byte.
- Encoded reference latents and the borrowed source AV state remain unchanged.

Stage cache changes during forward execution. Publish history only after both
heads, the sampler transition, and exact-row checks succeed. A failed or cancelled
job must not export partial history or contaminate the next request. Preserve
existing recovery behavior; fresh retry must match a fresh context. Extend hard
prefix audit coverage and reuse existing bridge audit machinery.

Zero-strength bridge must reduce to hard behavior under matched effective inputs,
including score regions, cache decisions, source initialization, and final latents.
Any pre-existing limitation in this identity must be isolated and documented
before using it as an approximation acceptance check.

## SubBlock continuation and composition

Use the existing conservative query/key protection policy for both modes:
protect the **entire** continuation video prefix, including mutable bridge rows;
all audio; all text/vision/reference rows; existing first/last video frame blocks;
and any mixed block touching a protected row. Protected queries use dense attention
over the valid keys. Unprotected queries always retain protected key blocks in
addition to router selections, without duplicates.

This preserves attention access to history and conditions. It does not claim that
protected hidden states match a completely dense run after previous approximate
layers. Exact latent preservation is enforced separately by the sampler masks.
Keep the current attention visibility rules and deterministic route selection;
no invalid layout may silently acquire unrestricted visibility.

Do not sparsify mutable bridge queries in this first implementation. Long context
can substantially reduce the sparse speedup; report protected/selected/possible
pair counts. Block 0, warmup steps, short sequences, and full retained budgets
keep their existing dense paths. The final adaptive refresh need not turn off
SubBlock: a refresh runs all 50 blocks with the attention policy for that step.

When combined, force a refresh at the original SubBlock transition and rebuild
the suffix delta for that phase. Hits run no suffix attention, router, MLP/GEMM,
or suffix weight prefetch/transfer. Prefix masks and final heads still run.

## Memory, saved state, and identity

Account for full packed sequence length, including history and references, in
the three existing BF16 cache tensors. Add checked recipe-specific class/reduction
workspace to the adaptive budget before weight admission. Keep the 4096 MiB
default, explicit sufficient overrides, and current SubBlock workspace cap.
Existing recipe sizing must remain unchanged. Preserve resident/partial/streamed
weight policies, reserved attention memory, bounded allocation recovery, and
deferred suffix reads on hits. Record host, device, pinned, and checkpoint costs.

Keep the current containers and field layouts where sufficient: AV schema 3,
presentation 9, sampler envelope 2, adaptive section 40 v3, and BF16 sections
41/42. Recipe 4 is a new mathematical identity, not a historical reader. Derive
score-class plans from the existing saved layout/prefix/bridge records rather
than serializing a redundant untrusted class map. Update recipe validation in
all readers/writers, inspectors, presentation producers, and execution keys.
Do not reintroduce removed schemas or migration paths. If implementation finds
a genuinely necessary new field, document and test one explicit current-schema
change before adding it; do not reinterpret old bytes.

Before allocating sections 41/42, reconstruct and validate geometry, references,
prefix, bridge tables, target segments, score membership, recipe, exact tensor
sizes, budget, and payload lengths. Move/reuse the necessary checks currently
performed only after allocation. Reject forged continuation flags, mismatched
prefixes, bridge mode without a bridge plan, classes beyond bounds, anchors
disguised as prefixes, quantized recipe 4, and unexpected frozen-audio layouts.

Resume omission restores saved controls and continuation metadata. Matching
explicit numerical controls remain accepted; differing ones fail. Budget-only
changes retain current admission semantics. Restore ready/streak/phase/step
history exactly. Original source AV files and reference media must not be needed
for sampler resume after conditioning, initialized latents, and history are saved.
Bind live/prepared keys to canonical mode/context/bridge masks and the derived
score plan, including ordered media identity and source state identity.

Keep `.h3av` and `.presentation` complete and current. Normal delivery trims the
context; `--keep-continuation-prefix` changes delivery only. Saved latents,
decisions, and subsequent continuation must be identical between those choices.

## Validation and evidence

### Functional and operator coverage

Use local host/layout/container tests and independent CPU mathematical references
for small CUDA operators; do not introduce a live SGLang oracle or new numerical
goldens. Cover:

- Hard and bridge admission with each feature and both, both adaptive presets,
  custom threshold/hit/warmup controls, ordinary references and negative cases.
- Context 39 and 90 plus boundary/invalid contexts; ragged query/key blocks;
  both stereo channels; stepped, linear, ease-out and zero-strength bridge.
- Large unchanged history with isolated changes in generated video, generated
  audio, and each populated bridge class. A changing minority region must not
  be diluted. Cover empty classes, tiny denominators, threshold equality,
  nonfinite values, and exact preservation on hits and refreshes.
- Dense protected queries/keys and mixed blocks; actual sparse dispatch after
  warmup; no suffix dispatch/transfers on hits; forced phase and final refresh.
- Exact same-build uninterrupted/resumed latents, anchor, delta, history, and
  decisions before/after warmup, a real hit, streak refresh, sparse transition,
  and final evaluation. Hide source AV and original media during resume.
- Threshold-zero equivalence to the matched uncached attention path; zero-strength
  bridge versus hard; normal/debug delivery equality of full latents and decoded
  RGB/PCM suffixes. Compare content, not whole files whose provenance differs.
- Same-context mode/context/profile/reference/policy changes, cancellation and
  retry, failed allocation/media/checkpoint recovery, and a three-segment chain
  that exercises both hard-to-bridge and bridge-to-hard handoffs.
- Rechecksummed malformed checkpoint tests with bounded sizes; reject before
  expensive allocation, avoid partial files, and preserve current-only readers.
- Full/partial/forced-streamed placement and cache limits, with measured
  1344×768/124-frame reference jobs and bounded 362-frame capacity diagnostics.

Reuse the retained bridge, sampler, adaptive-reference, and SubBlock test helpers
where practical. Do not restore retired MLX/Metal or pre-SGLang golden baselines.
Ordinary functional invocations retain `H3_TEST_MAX_EVALUATIONS=6`; resume longer
schedules in bounded invocations when needed. Explicitly freeze and record any
test-only score fixtures used to exercise hits; they are not quality evidence.
No benchmark may silently retune thresholds to manufacture a speedup.

### Fixed video comparison

Freeze a machine-readable manifest before running. Use BF16, seed 42, full native
VAEs, context 39, all 50 layers, reuse/core reuse 1, and one successful video per
row. Record exact prompts, source/reference hashes, commands, source/build IDs,
effective policies, total wall time, denoise time, memory, cache decisions, router
coverage, and failure/retry history. Source generation is outside each candidate's
timing and reported separately; candidate timing includes loading through MP4.

Produce two shared dense six-step sources: S01 at 640×480/90 frames without
references, and S02 at 1344×768/124 frames with one `inputs/1.jpg` reference at
`--ref-image-size max`. Keep their `.h3av` files and presentation sidecars. The
Ref2VA source must have the compatible model/VAE signature for reference targets.
Reuse a matching already-qualified source only if its complete identity is verified.

| IDs | Mode | Resolution / raw target frames / steps | Variants |
| --- | --- | --- | --- |
| V01–V05 | Hard, S01 | 640×480 / 90 / 50 | Dense; conservative; aggressive; SubBlock 0.75; conservative + SubBlock 0.75 |
| V06–V10 | Bridge, S01 | 640×480 / 90 / 50 | Same five variants |
| V11–V12 | Hard, S02 + same max-size image | 1344×768 / 124 / 6 | Dense; custom adaptive + SubBlock 0.75 |
| V13–V14 | Bridge, S02 + same max-size image | 1344×768 / 124 / 6 | Same two variants |

For V01–V10 use preset thresholds/hit ceilings, adaptive warmup 4, and SubBlock
warmup 10. Only these ten explicitly named subprocesses may raise the test
evaluation ceiling to 50. For V12/V14 use conservative with threshold 0.12,
maximum two hits, adaptive warmup 2, and SubBlock warmup 3. These are unqualified
short-preview settings, not a new recommended preset. Bridge uses eight video
rows, maximum strength 0.5, and the stepped profile in comparison videos; other
profiles belong to bounded functional tests. Prompts match within each comparison
group; hard continues the source action and bridge introduces a modest action
change within the same scene.

This is **14 comparison renders plus at most two shared source renders**. With
context 39 the two target lengths deliver 51 and 85 new frames respectively;
do not confuse raw target frames with delivered suffix length. Derive stitched
source-tail/suffix review clips with FFmpeg, without additional H3 rendering.
Include synchronized audio and context-boundary contact sheets. Cover embedded,
silent, replacement, separate-audio, and mixed references in bounded functional
jobs using reproducibly derived media rather than another full video matrix.

Report no-hit and all-dense outcomes honestly. Require real-model hit and sparse
execution coverage in the functional campaign, using separately labeled bounded
diagnostics if the frozen quality presets do not exercise them. Assess boundary
motion, identity/reference fidelity, audio joins, artifacts, and cumulative
multi-segment drift. Quality differences against dense are measurements, not
failures of the exact SGLang contract. Record human visual/listening acceptance
separately from automated checks; do not assume previous acceptance covers these clips.

### Environment and release gates

Use the existing PRO 5000 at `cuda-test`, checkout `~/h3cli`, and installed
models under `/models` for CUDA; use the local M4 for shared tests and Metal
build/non-regression coverage. Keep host-specific paths in the task/run records,
not production defaults or portable tests. Use the qualified CUDA 13.0.3/cuDNN
9.20 stack and current build configuration. Do not use the rental nodes.

Follow [CONTRIBUTING.md](../../CONTRIBUTING.md): run the complete immutable
204-output SGLang gate after each coherent source/generator/build/test-tooling
change and once after final qualification, using fresh directories. Keep all
12 fixtures and expected hashes unchanged. Also run clean CUDA/Metal builds,
retained `make test`, `make test-current-cuda`, relevant sanitizer tests, and
unchanged default recipe/continuation functional comparisons. No new SGLang
reference cases or independent trust anchor are needed.

Evidence lives under ignored `outputs/adaptive-continuation/<run>/`; build
outputs remain in `bin/`. Publish a tracked report under `docs/cuda/`, with all
14 comparison videos linked, actual wall times, environment/source identities,
gate results, and limitations. Update README/help, current adaptive/continuation
contracts, current state documentation, and the planned server feature inventory.
Archive historical acceptance records without rewriting their results.

Completion requires all supported combinations to work through generation,
save/load, resume, and continuation chains; unchanged exact history; demonstrated
hit/sparse execution; passing builds/suites and 204/204 golden outputs; and an
honest quality/performance report. Human acceptance remains explicit. A feature
may have limited speedup with long protected history; no minimum speedup is promised.
