# RTX 5090 FP8 / NVFP4 preview validation

Both formats are a **go for opt-in base-model previews on the tested RTX 5090**.
They execute native W8A8/W4A4 projections, meet the denoising performance
targets on both short presets, reduce device memory, and have separate human
video/audio acceptance. The user also accepted the separate short hard/bridge
continuation and eight-step Turbo strength-1.0 gallery. Neither format becomes
the default; acceptance covers the exact reviewed recipes and media below.

Evidence date: **2026-09-19**. The user accepted FP8 with “Accept FP8 quality”
and NVFP4 with “Accept NVFP4 quality”. These decisions bind to the twelve
quantized videos and complete triplet manifest, not to arbitrary future recipes.
The [tracked acceptance/index summary](denoiser-quantization-acceptance.json)
records exact hashes. Local artifacts:

- [Six-case matched gallery](../../outputs/quant-5090/quality/review.html),
  [commands and media hashes](../../outputs/quant-5090/quality/manifest.json),
  [quality decisions](../../outputs/quant-5090/quality/acceptance.json).
- [Complete machine-readable index](../../outputs/quant-5090/validation-index.json),
  [experiment ledger](../../outputs/quant-5090/budget.json),
  [environment/source/fixture snapshot](../../outputs/quant-5090/environment.json).
- [All-media validation](../../outputs/quant-5090/media-index.json) and
  [continuation smoke joins](../../outputs/quant-5090/continuation-review.html).
- [New continuation/Turbo gallery](../../outputs/quant-5090/followup/review.html),
  [manifest](../../outputs/quant-5090/followup/review-manifest.json),
  [quality decisions](../../outputs/quant-5090/followup/acceptance.json), and
  [adapter/cache identity audit](../../outputs/quant-5090/followup/adapter-cache-identities.json).

## Configuration and provenance

Starting revision: `f4893c70a1c86ebe2c8f801a64ea72091d6ddd4f`. Physical
GeForce RTX 5090, SM120, 32,607 MiB reported / 31.36 GiB CUDA addressable;
driver 595.91.07, nvcc 12.8.93, cuBLASLt version query **120804**, cuDNN 9.8.0,
frontend 1.11.0, GCC 13.3.0. Work and packed artifacts are under
`/path/to/h3-quant`; original FL2VA/Ref2VA checkpoints are under
`/path/to/models/MiniMax-H3` on the node's local drive. `/workspace` is unused.

Native cuBLASLt was sufficient; no CUTLASS/framework dependency was added.
GeForce NVFP4 requires the support introduced in CUDA 12.8 Update 1's cuBLAS;
the runtime checks the linked library, not merely the compiler's major/minor.
See [NVIDIA's release notes](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html)
and [narrow-precision contracts](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html#narrow-precision-data-types-usage).
The implementation is original code using the existing NVIDIA library license.

The baseline binary is preserved as `h3-baseline` (SHA-256
`ba5e501cb2614a500407134e5623882ef9b5a352e80dbff8f0629fa4e3f61441`).
The accepted gallery candidate is preserved as `h3-quality-candidate`
(`57658eba9dccf6ea91d9fcd859639c84f03279b20e30e6244d5d57c6732d4b39`).
The final `h3cli` is
`e5add3341b510c51cfebd4de8f9cfbfa64ba42ddb5104e39eb8d0c86f4079759`.
Their embedded source/build IDs are in `*.identity.json`. Later changes added
validation/lifecycle fixes, accelerated equivalent hashing, and optional
diagnostics disabled in normal runs; recipe 1's quantization arithmetic did
not change. The final implementation ran the state, cancellation, memory-pressure,
replay, native diagnostics and both-resolution warm comparisons.

Initial pilot records predate the per-run source inventory. Their binary hashes
and preserved build IDs, plus quantized AV-sidecar model fingerprints, provide
retrospective provenance; they are not represented as having a complete original
per-run source inventory. `environment.json` is the immutable pre-acceptance
snapshot; its pending quality fields are superseded by `quality/acceptance.json`.

Recipe 1 quantizes **all 200 repeated projection matrices**, with no BF16/FP8
exceptions, rotations or low-rank corrections. Dynamic scaling on the separate
`1.jpg` pilot was adequate for the held-out review; further quality interventions
were unnecessary. Whitelist/recipe changes must invalidate these approvals.

## Matched quality and first-process measurements

The held-out set has six off/FP8/NVFP4 triplets: `2.jpg`, body/detail at
480×640, face/speech, FL2VA first/last frames, multiple face/body references,
and reference video with embedded audio. It covers both model variants.
The calibration pilot uses `1.jpg`; dynamic amax scaling does not fit
statistics from the held-out set. Each triplet preserves the seed, prompt,
references, existing Euler sampler, fast-CUDA attention, full VAE and codec.
All use **50 blocks, 20 actual evaluations, reuse/core-reuse one, no reduction**.
Quantized profiles report 4,000 native projection calls per render:
`20 × 50 × 4`. Seeds/inputs/commands are in the gallery manifest.

| Preset / mode | DiT load | Denoise wall | Complete MP4 wall | Peak DiT owned allocations |
| --- | ---: | ---: | ---: | ---: |
| 288×384, 56 frames / off pilot | 8.827 s | 35.519 s | 71.99 s | 24.921 GiB |
| same / FP8, cold packed cache | 56.968 s | 5.891 s | 176.73 s | 18.920 GiB |
| same / NVFP4, cold packed cache | 42.898 s | 3.355 s | 73.30 s | 11.056 GiB |
| 480×640, 22 frames / off | 8.660 s | 32.340 s | 66.09 s | 24.807 GiB |
| same / FP8, verified prepared cache | 47.546 s | 7.390 s | 79.86 s | 19.059 GiB |
| same / NVFP4, verified prepared cache | 36.069 s | 4.265 s | 65.42 s | 11.192 GiB |

These are single runs. The first FP8 run additionally populated the full-model
fingerprint cache; NVFP4 reused that fingerprint. **176.73 versus 73.30 seconds
is not a matched cold-start format comparison.** The per-matrix cold preparation
totals were 47.594 s FP8 and 33.836 s NVFP4. Prepared-cache verification/loading
in the detail case cost 38.413 s and 27.162 s respectively, before other DiT
initialization. Original source bytes and artifact payloads are rehashed.

Denoise wall here is the difference between the cumulative `H3 DiT / GPU Euler
denoise` and `H3 DiT / load` wall marks. GPU category timers overlap; conversion
is included in GEMM time. They must not be added to wall time. Peak figures
are allocator-owned device bytes, not total process VRAM including vendor pools.

In the off pilot, 286.822 GiB crossed the DiT streaming path; detail used
290.914 GiB. Both quantized cores remained resident with **zero repeated weight
streaming**. One core's artifacts occupy about **17.94 GiB FP8 / 10.09 GiB
NVFP4**, versus about 35.9 GiB for the original BF16 core. This residency change
is a major part of the full-denoiser speedup. The planner budgets packed
weights/scales plus 1 GiB and estimated activations plus 1 GiB before its normal
reserve; the initial pilot plans were 18.94 / 11.09 GiB for packed weights.

The non-DiT work remains material. In the detail FP8 run, text encoding's
cumulative component wall was 15.195 s, audio VAE 0.375 s, and full video VAE
2.901 s. Delivery, reference preparation, signatures and process setup account
for additional complete-MP4 time. A short fresh-process FP8 command can lose
wall time despite substantially faster denoising.

## Warm-context previews

Each mode ran one first request and one same-context repeat, with `face1.jpg`,
seed 72, 50 blocks, 20 evaluations and **preview VAE enabled in all modes**.
Finite-latent callbacks force the same per-step readback for all three modes.
The first/last callback interval measures denoising; API wall includes MP4
completion and saving its small AV state. Order was off→FP8→NVFP4 for the
first preset and NVFP4→FP8→off for the second. These are bounded repeats,
not statistical confidence intervals or comparisons to the full-VAE table.

| Preset / mode | First request, prepared cache | Warm denoise | Warm MP4 + state | Peak DiT allocations |
| --- | ---: | ---: | ---: | ---: |
| 288×384 / off | 62.590 s | 28.143 s | 29.189 s | 24.917 GiB |
| 288×384 / FP8 | 75.035 s | 5.755 s | 6.811 s | 18.916 GiB |
| 288×384 / NVFP4 | 62.529 s | 3.204 s | 4.252 s | 11.052 GiB |
| 480×640 / off | 63.145 s | 30.585 s | 31.283 s | 24.734 GiB |
| 480×640 / FP8 | 74.989 s | 7.258 s | 7.914 s | 19.057 GiB |
| 480×640 / NVFP4 | 63.517 s | 4.095 s | 4.737 s | 11.190 GiB |

FP8 warm denoising is **4.89× / 4.21×** faster than off; NVFP4 is
**1.80× / 1.77×** faster than FP8. Both exceed 1.25× / 1.15× targets at both
presets, with no unexplained second-preset regression. Off's warm request still
streams 264.141 / 269.883 GiB; quantized requests stream none.

GPU event increments for the warm request:

| Preset | Mode | GEMM, including conversion | Activation conversion | Attention |
| --- | --- | ---: | ---: | ---: |
| 288×384 | off | 7.521 s | — | 0.796 s |
| 288×384 | FP8 | 4.538 s | 0.199 s | 0.692 s |
| 288×384 | NVFP4 | 1.994 s | 0.324 s | 0.694 s |
| 480×640 | off | 9.188 s | — | 1.177 s |
| 480×640 | FP8 | 5.578 s | 0.267 s | 1.056 s |
| 480×640 | NVFP4 | 2.430 s | 0.381 s | 1.049 s |

Attention's algorithm/precision did not change. Modest timing differences
include scheduling effects; no attention speedup is attributed to quantization.
There is no correction branch. Native calls increase from 4,000 to 8,000 across
the two requests, confirming both ran 20 full evaluations.

For prepared caches, the measured first-request overhead is amortized by the
**second preview** for FP8 at both sizes and NVFP4 at 480×640. The 288×384
NVFP4 first-request difference is below useful single-run timing precision;
its second request clearly wins. For a new cache, a planning estimate is
`1 + ceil(max(0, first_quant - first_off) / (warm_off - warm_quant))`.
Using the cold pilot's first-request deficit and the 288×384 warm savings gives
roughly **six previews for FP8 and two for NVFP4**. That last estimate mixes
pilot/full-VAE and warmed/tiny-VAE measurements and different fingerprint-cache
states: it is **not a directly measured cold-cache crossover**. Use the raw
preparation/startup numbers to budget a first run; do not assume every one-shot
preview will be faster.

## Real projection and memory-limit probes

All four actual block-0 weight shapes were timed with deterministic activations,
one cold call and four warm calls. The timer includes activation conversion,
native GEMM, cropped output and synchronization; it excludes weight loading.
These probes establish throughput/finite output, not perceptual similarity.

| Rows | Sum of four warm projections, BF16 | FP8 | NVFP4 |
| --- | ---: | ---: | ---: |
| 2,028 | 6.461 ms | 3.846 ms | 1.781 ms |
| 8,193, including row padding/tail | 26.415 ms | 15.767 ms | 7.281 ms |

Native projection compute improves about 1.68× BF16→FP8 and 2.16× FP8→NVFP4,
less than the full-denoiser gain from avoiding weight streaming. At 8,193 rows,
quantized scratch/padded outputs can exceed a single BF16 projection's memory
even while the full weight core is smaller. No long/high-resolution render
is inferred from these component probes; attention and activation memory still
grow with tokens.

An injected **12,000,000,000-byte allocation budget during DiT loading** caused
the resident FP8 load to fail, release partial resources and retry once using
packed streaming. The subsequent full 50-block, two-step generation completed
with finite latents (3.190 s denoise at 128×128). Both formats' native tests also
exercise forced packed streaming. This does not qualify a production-size
NVFP4 allocation-failure recovery or long-form throughput.

## Initial-round correctness, media and regression evidence

The initial round contained **81 ledger invocations: 79 passed, two retained failed attempts.** Total
charged experiment time is **3,575.087 / 3,600 seconds** (59.585 minutes), with
no allowance extension. All warmups, cold preparations, failed attempts,
sanitizers, render/replay work and media artifact generation are included.
Host-only builds/unit checks are separate. Nothing longer than the authorized
short presets was rendered; no `test2.sh` or 362-frame segment was run.

- Native FP8/NVFP4 known-pattern, sign, multi-scale, zero, 1/17/129/3-row tail,
  plan/operand reuse and finite-output tests passed. Artifacts passed concurrent
  writers, source-content changes, invalid ranges/shapes, truncated/corrupt or
  incompatible headers, and checksum-valid invalid scale/NaN payload tests.
  Optional device diagnostics and default-CUDA projection dispatch passed.
- Compute Sanitizer memcheck passed for both native format tests and all four
  real projection shapes at 129 rows, with zero reported errors. FP8 synccheck
  and NVFP4 racecheck passed. These cover owned kernels; they do not certify
  internal vendor kernels or full repeated-cancel workflows under instrumentation.
- Same-context off→FP8→FP8→NVFP4→NVFP4→off and changed geometry/reference passed
  before the combined job's later timeout. Same-mode repeats and restored off
  latents matched their own policy. Separate cancellation/recovery cases passed
  for both formats. Teardown/cache-clear assertions passed in the split cases.
- Both formats passed pause/resume, explicit incompatible-mode rejection,
  stopped/live tiny previews, default-CUDA composition, token reduction and
  reuse. Same-policy resumed latents matched uninterrupted latents; there is
  no BF16-versus-quantized similarity gate.
- All four CLI full/tiny saved-state replays passed without quantization flags
  or packed-cache arguments. Pre-launch and final input-state SHA-256 values
  match. Sidecars retain denoising provenance; full-VAE replay does not undo it.
- Both formats completed a 90-frame source plus hard and bridge continuations:
  39-frame / 52,000-sample overlap, **51 delivered new frames**. Four 141-frame
  joins are retained. These 64×64/two-step clips are functional coverage;
  their visible/audible seams have not received separate human acceptance.
- Final media checks passed **67 complete MP4 decodes and 57 finite,
  checksummed AV states**, including expected geometry/frames, 24 fps,
  stereo 32 kHz audio, duration/mux integrity and sidecar fingerprints/trims.
  Paused previews are intentionally silent. No black intervals were flagged.
- Final `make cuda-test CUDA_ARCH=120 CUDA_CUDNN=1 ...` passed the 129-symbol
  contract, runtime, BF16, sampler, bridge, audio, Qwen scaling and preview/cache
  suites. The no-cuDNN build and both native format probes passed separately.
- Local M4 build, 1,769 core checks, 103,204 continuation checks, 669 progress
  checks, Metal sampler/GPU preview checks, 129-symbol contract, 36 offline-LoRA
  tests, preview CLI tests and sampler CLI/container corruption tests passed.
  The final focused host run passed four option/budget/gallery tests and
  **1,493 sampler checks** on both platforms. Legacy state/sidecar handling is
  covered; there was no unnecessary full-model Metal rerender.

The two retained failed attempts were: (1) initial NVFP4 cuBLASLt heuristic
status 7, fixed by setting scale pointers before querying/checking a plan;
(2) the combined workflow job hit its 300-second cap after the switch cases,
so cancellation/state/continuation were split into bounded jobs. Both costs
remain in the ledger. No failed candidate's output was counted as accepted.
An intermediate no-cuDNN copy omitted `scripts/cuda_arch.sh`; restoring the
build helper fixed its architecture selection. The final no-cuDNN build passed.

## Remaining-task follow-up

The user authorized an **unlimited cumulative experiment allowance** and supplied
`lora/downloads/minimax_h3_turbo_v4_step600_ema.safetensors`. The same ledger
records the extension after 3,575.087 seconds; all original attempts and failures
remain intact. Component, render and preparation/sanitizer jobs still have
120/300/600-second caps. Work started from revision
`8da675b61091f7fc3212420f7cff695dad93e1c5` on the same physical 5090 and local
drive. Production quantization arithmetic and the accepted `h3cli` binary are
unchanged. This round extends test helpers and evidence reporting.

Both standalone full-core preparation commands passed, verifying all 200 packed
matrices: FP8 **35.557 s**, NVFP4 **24.652 s**. A 6,000,000,000-byte injected
NVFP4 load budget forced resident allocation failure followed by successful
compressed-streaming recovery. The FP8 full-VAE test forced pre-decoder eviction
with 4 GiB of additional device allocations, then completed another generation.
NVFP4 passed the same eviction/recovery test with 12 GiB of additional device
allocations. These cases took 115.197 s and 89.030 s respectively.
With the full decoder cached, FP8's subsequent request selected compressed
streaming and still completed; NVFP4 retained enough headroom for residency.
Successful recovery does not imply that every later request can keep both the
full decoder and quantized core resident on 32 GB.

The new API cases cover Ref2VA→FL2VA→Ref2VA in one context, and base→Turbo→base
through separate contexts. A model directory is immutable for a context; adapter
switching does not invent an unsupported in-place model replacement API. Return
to the original policy checks its own exact latent/state identity, with no
BF16-versus-quantized similarity threshold. Repeated lifecycle cases cancel after
core blocks 1 and 2, then after denoising step 1, verify no aborted MP4 or retained
partial prepared core, and complete a recovery request.

The first base→Turbo→base FP8 job completed all three renders with correct model
identities but failed the helper's byte-exact latent comparison. The initially
queued Ref2VA→FL2VA→Ref2VA job hit the same assertion. Cache clearing
also reruns timed BF16 GEMM selection. Repeating with
`H3_FAST_CUDA_GEMM_TUNE=0` passed the unchanged exact comparison in **202.070 s**
for adapters and **207.339 s** for variants. NVFP4 adapter isolation passed in
**185.860 s**, and NVFP4 variant isolation in **174.534 s**, with the same control.
The helper now fixes algorithm selection for model/adapter identity assertions;
ordinary quality/performance renders retain timed tuning. The original failed
attempts and their 248.145 s / 207.480 s costs are preserved. This test-control correction does
not change production code or introduce a quantized quality threshold.

FP8 follow-up instrumentation passed lifecycle memcheck (330.564 s), pressure
memcheck (146.056 s), lifecycle synccheck (196.263 s) and lifecycle racecheck
(454.713 s). Memcheck reports zero errors and leaked bytes; synchronization/race
checks report zero errors or warnings. Memcheck is unfiltered. Synccheck and
racecheck select the owned `quant_` kernels while running the complete lifecycle;
they do not certify vendor-internal kernels or constitute a general inter-stream
race proof.
NVFP4 lifecycle and allocation-pressure memcheck also passed (325.174 s and
123.944 s), both reporting zero errors and zero leaked bytes. NVFP4 lifecycle
synccheck and racecheck passed in 177.095 s and 432.479 s with zero errors or
warnings. All eight follow-up sanitizer jobs completed within their 600-second
individual caps.

Offline folding uses optional pinned **NumPy 2.2.6 / safetensors 0.6.2**.
The Turbo adapter SHA-256 is
`5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3`.
Strength 1.0 folded 259 targets in 13 shards in **389.534 s**, including the
existing integrity checks. Original checkpoints remain untouched; unchanged
model assets are symlinked into the complete Ref2VA model. Folded BF16 content,
rather than the adapter filename, determines each packed matrix key.
The separate strength-0.5 fold passed in **377.965 s** with the same pinned
adapter and original-basis folding pipeline. Both fold manifests are archived
under the follow-up evidence directory.
All four block-0 projection families passed real-weight native execution at
strengths 1.0 and 0.5 in both formats. The cache audit verifies three distinct
source keys and packed paths per family/format for base/1.0/0.5, and three
distinct whole-model fingerprints in completed NVFP4 AV states. Strength 0.5
also completed an eight-step full-NVFP4/full-VAE smoke render in **109.276 s**.
Only the four real projection families were prepared in FP8 for strength 0.5;
that strength has no full-FP8 render or perceptual qualification in this round.

The [follow-up gallery](../../outputs/quant-5090/followup/review.html) contains
matched full-VAE off/FP8/NVFP4 hard/bridge continuations at 288×384, 90 raw frames,
20 steps, and a separate 56-frame, eight-step Turbo comparison at strength 1.0.
Each continuation delivers 51 new frames after a 39-frame / 52,000-sample trim.
The joined review clips have 141 frames; the seam is at **3.75 seconds**. Source
audio is trimmed to exactly 120,000 samples before joining so codec padding does
not move the boundary. New playback/listening decisions are separate from the
six-case base-model approvals and bind to the new manifest and media hashes.

All follow-up execution checks passed after controlling GEMM selection for the
two identity-test repeats. The shared ledger now has **122 attempts: 118 passed
and four retained failed attempts**, totaling **9,162.284 seconds**. This round
added 41 attempts and 5,587.197 seconds; the two new failures are the resolved
identity assertions described above. The cumulative allowance is unlimited;
no per-job timeout was increased. No new inference-code fix was needed.

Final validation passed **117 complete MP4 decodes and 101 finite, checksummed
AV states**, with no black intervals flagged. The nine continuation renders
all completed 20 steps; each quantized render reported **4,000 native projection
calls**. The Turbo triplet completed eight steps, with **1,600 native calls** in
each quantized case. Six 141-frame joins preserve the intended frame/sample
trims and contain a 3.75-second seam. Contact-sheet inspection found coherent
framing across the sampled boundary frames. On **2026-09-19**, the user confirmed
“the new gallery is acceptable”, accepting playback/listening quality for both
the hard/bridge continuation joins and the Turbo strength-1.0 triplet.
[The decisions](../../outputs/quant-5090/followup/acceptance.json) bind to the
exact gallery manifest and all nine reviewed media hashes. This acceptance
does not extend to strength 0.5 or other adapters.

The updated helper built on both CUDA and local Metal. Six focused host tests
passed locally and on the node, including unlimited-ledger accounting and stale
approval detection when either quantized media or its baseline is replaced.
All 36 offline LoRA tests passed with the pinned fold dependencies. Production
`h3cli` remains SHA-256
`e5add3341b510c51cfebd4de8f9cfbfa64ba42ddb5104e39eb8d0c86f4079759`;
the unchanged production/Metal/default-CUDA regressions from the initial round
are reused with that identity. T014, T045, T047, T050 and T052 are complete;
all **64 tasks** in [todo.md](../todo.md) are now complete.

## Reproduction and remaining scope

The [usage guide](denoiser-quantization.md) documents the public options/API,
cache location, preparation, required libraries and state behavior. Build the
helpers with the normal CUDA flags, then use the ledger for every experiment:

```sh
make -j8 bin/h3cli bin/quant_native bin/quant_projection bin/quant_workflow bin/quant_prepare \
  CUDA_ARCH=120 CUDA_CUDNN=1 \
  CUDNN_FRONTEND_PATH=/path/to/models/h3-preview-deps/cudnn-frontend/include
python3 tests/quant_run.py --name native-new-fp8 --timeout 120 -- \
  ./bin/quant_native fp8 outputs/native-cache
python3 tests/quant_qualify.py --gallery
python3 tests/quant_report.py --publish docs/cuda/denoiser-quantization-acceptance.json
```

The last two commands only regenerate indexes from existing evidence. Do not
reset `budget.json` to run additional tests. This follow-up explicitly selected
`--unlimited` on its first command, preserving spent time and per-case caps.
Exact historical commands and
timeouts are in each `<case>.json`, with raw `<case>.log` output.

The remaining-task round is driven by `tests/quant_followup.py`: `media`,
`sanitizers`, `adapters`, `variants`, and `scales` select bounded jobs; `gallery` refreshes
HTML without rerendering, and `identities` audits existing source/packed keys.
Both folded strengths must be prepared before `scales`. The `run` helper reuses
only passed records with identical commands. A failed attempt is retained and
requires an explicitly named repeat.

Useful follow-ups are reducing verified fresh-process startup cost and testing
prepared-cache retention across repeated library requests. Other GPU families,
additional adapter strengths, quantized attention, Metal quantization,
new distilled schedules and production-length quality qualification remain
outside this series.
