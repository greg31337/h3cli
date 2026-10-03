# Native adaptive cache and SubBlock qualification

Status: **accepted; visual qualification passed for all twelve comparison videos**.
Both features remain opt-in. The [design](design-adaptive-cache-subblock.md),
[frozen contract](adaptive-subblock-contract.md) and
[twelve-case manifest](adaptive-subblock-manifest.json) define this experiment.

Open the local [synchronized video comparison](../../outputs/adaptive-subblock/2026-09-26-sm120/index.html),
[complete JSON](../../outputs/adaptive-subblock/2026-09-26-sm120/report.json) or
[CSV](../../outputs/adaptive-subblock/2026-09-26-sm120/report.csv).
Generated media and evidence remain gitignored under `outputs/`.

The user confirmed: “accept the results, visual qualification passed”. The
[review record](../../outputs/adaptive-subblock/2026-09-26-sm120/human-review.json)
binds this approval to the twelve existing videos and their render identity.
The measured differences and historical diagnostic outcomes below are retained.

## Qualified implementation

Native adaptive recipe 1 and SubBlock recipe/plan 1 run on CUDA SM120 without
Python, Torch or a worker in inference. Adaptive decisions use a fresh dense
first block and anchored BF16 suffix residual. SubBlock uses native BF16 tensor
core attention with deterministic routing, protected packed ranges and bounded
workspace. Their composition follows absolute scheduler indices and forces a
refresh when sparse attention starts at index 10.

The qualified scope is text-to-video with generated audio, all 50 blocks and
BF16 projections on the RTX PRO 5000. Reference conditioning, continuation,
LoRA/Turbo, quantization, thinning, token reduction and combinations with the
existing reuse controls are explicitly rejected. Metal retains its existing
behavior and rejects the new execution options. Other hardware and tasks are
unqualified. Both features remain off by default.

The final complete recorded regression passed **204/204** outputs in 118.029 s,
with 209.861 s including its isolated build. The campaign's earlier frozen
render build also passed 204/204 in 126.695 s. No recorded inputs,
goldens, cases, tolerances or evaluation counts changed.

| Identity | SHA-256 |
| --- | --- |
| Final source and test tooling | `63844ba882eaffee2b82fc2e126112badf16a17cd24f1a7e8fdf72a77e837b53` |
| Final requalification CUDA `bin/h3cli` | `342492540f8523ad7ed70e97c45162d707d0819dffb0134e44ba985cf2356cd1` |
| Frozen render source and test tooling | `d1e936cb93948f5d44bae325a9d7962afff648aa57ebcfb3337709745530bcb0` |
| Frozen render CUDA `bin/h3cli` | `d1208c2e7e47649f62da7d363a508acd3f126906ffea944aa18c78e4d1d4a0c5` |
| Recorded golden manifest | `fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3` |
| Twelve-case manifest | `fbf0034cec63e645b0e0c1397578bcd5304927cc37923ff03b2d517e33d6e8a4` |

Runtime: CUDA toolkit 13.0.3 (nvcc 13.0.88), driver 595.91.07, cuDNN
9.20.0.48, cuBLAS 13.1.1.3 and recorded SGLang recipe 4. Model identity uses
metadata; no H3 weights were hashed. Documentation is outside the regression's
source fingerprint. All twelve videos use the same frozen render binary.

After numerical metrics completed, the report's original regex link checker
misread a JavaScript-generated image URL as a literal file. Its replacement
uses HTML parsing for actual attributes and explicitly validates dynamic image
and video targets. This was the only source-file change after the render freeze;
all production source remained byte-identical. The full recorded regression
was rerun successfully before republishing existing metrics and media. Isolated
CUDA binaries embed their source directories, so both build hashes are retained
in `report-tool-qualification.json`; no comparison video was regenerated.

## Correctness evidence

Evidence is retained locally under
`outputs/adaptive-subblock/work/qualification-evidence/`, including failed
qualification attempts and the successful follow-up checks.

| Check | Result |
| --- | --- |
| Metal build, ordinary tests and sampler tests | Pass; 1,782 sampler checks, 129 GPU sampler checks, 210 adversarial file cases and 52 CLI cases |
| GPU interface/link coverage | Pass on Metal and CUDA; includes adaptive probe entry point |
| CUDA builds with `CUDA_SUBBLOCK=1` and `0` | Pass, including executable and static library |
| Ordinary CUDA tests and sampler tests | Pass (`final-qualification-c/ordinary.log`) |
| Host adaptive policy and CLI ordering/conflicts | Pass |
| Adaptive GPU probe | Pass: deterministic BF16 subtraction/FP32 reduction, ragged tails, separate video/audio scores, epsilon floor, aliases and nonfinite recovery |
| Native sparse operator and independent oracle | 15 fixtures pass, including ties, tails, zero/constant values, protected blocks, both output layouts, forced all-selected execution and rejected aliases/nonfinite inputs |
| Workspace tiling | Pass with 56 heads, a 641-token ragged sequence and a deliberately small workspace; covers partial head groups and query slabs |
| CUDA Compute Sanitizer | `memcheck`, `racecheck`, `initcheck` and `synccheck`: zero errors on a ragged 65-token, two-head sparse fixture |
| Shared dense paths and context isolation | Exact output for disabled/full-budget/warmup/probe/protected-query paths; policy capture, allocation failure/recovery and feature-off zero workspace pass |
| Combined latent stop/resume | Byte-identical decisions and AV latents across warmup, hits, forced refresh, index 10 and final index 49 |
| Cancellation/recovery | Byte-identical resumed boundaries 10–12 after cancelling in the same context |
| Approximate checkpoint corruption | All 37 missing/version/flags/shape/value/checksum cases reject invalid state |
| Clean AV decode without SubBlock build | Pass: completed 56-frame AV state decoded through the default build |
| Campaign accounting | Nine synthetic tests pass, including complete coverage, successful-repeat refusal, artifact/build tampering, retained failures and exact 90-frame media |
| Final unchanged recorded regression | 204/204 pass |

Latent qualification retained the original 50-step schedule, running at most
six transitions per invocation, without producing extra comparison videos.
At sparse index 10 the combined fixture executed 50 blocks, 49 sparse calls,
49 routers and 98 protected-query calls. Its next adaptive hit executed one
dense block, no sparse/router calls and eight GEMMs, versus 204 GEMMs on a
refresh. Measured H2D tensor transfers fell from 38,537,150,208 bytes to
1,982,208 bytes, confirming that the suffix weights were not uploaded on hits.

Active checkpoints require versioned policy, anchor/delta, plan and CUDA device
identity sections. A clean completed AV state omits the active execution
dependency and remains decodable without the optional SubBlock build.
References and continuation remain rejected because their end-to-end consumers
were not qualified; operator protection tests do not imply support for them.

## Real-QKV qualification and its limits

A real 5,298-token, 56-head capture exercised both layouts, routing, protected
dense ranges and workspace admission. The native protected pipeline took
4.931 ms versus 4.166 ms for dense attention in the head-major replay. Routing
took 0.882 ms and the sparse kernel 2.651 ms. These are single operator
observations; this sparse case was slower. The kernel really executed, with
193,156 selected out of 385,784 possible block pairs after protection.

An initial raw-value oracle attempt correctly matched routing and relative
error but failed an incorrectly applied unit-scale absolute bound: captured
values reached 120.5 and the maximum output error was 0.07106. The frozen
contract assigns the 0.0625 absolute bound to unit-scale fixtures. The follow-up
divided V by an exactly representable power of two (128), giving maximum input
magnitude 0.94140625 without changing Q or K. With the original tolerances it
passed: relative L2 0.0018172, maximum absolute error 0.00055516 and maximum
router-score error 0.000003425. Complete native output buffers in both layouts
scaled exactly, and routes and scores remained byte-identical. No kernel,
golden or tolerance was changed to obtain this pass; the raw failure remains
in the evidence.

## Twelve-video comparison

The twelve videos use the frozen 640×480 / 90-frame / 50-step workload and one
shared D0 baseline. Each variant uses the same binary, prompt, seed, model,
streamed-weight policy, decoder and codecs. There are no successful repeats.
The dedicated runner alone grants its renderer subprocesses 50 evaluations;
the ordinary six-evaluation limit remains intact.

Advisory page-cache preparation does not guarantee equal physical residency.
Weight paging, load, encode, denoise, full AV decode, codec and state writes
remain inside end-to-end wall time. Operator/event stage timings can overlap
and must not be added as independent contributions. Tensor transfer counters
exclude small control readbacks, including the adaptive probe's 24-byte scores
and sparse fault counters.

All twelve videos completed on their first attempt. Every output passed finite
AV-state checks, 50-transition/dispatch checks, exact 90-frame media validation
and full video/audio decoding. Metrics reuse the same D0 video throughout.

| Variant | Wall seconds | D0/wall | Adaptive hits | Minimum SSIM ↑ | Maximum LPIPS ↓ | Audio relative L2 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 | 264.10 | 1.00× | 0 | 1.000 | 0.000 | 0.000 |
| R2 | 165.59 | 1.59× | 0 | 0.601 | 0.359 | 0.185 |
| R3 | 123.71 | 2.13× | 0 | 0.436 | 0.512 | 0.263 |
| C4 | 109.02 | 2.42× | 0 | 0.470 | 0.514 | 0.270 |
| C6 | 94.98 | 2.78× | 0 | 0.417 | 0.526 | 0.456 |
| A1 | 174.82 | 1.51× | 18 | 0.864 | 0.094 | 0.129 |
| A3 | 133.39 | 1.98× | 30 | 0.723 | 0.232 | 0.228 |
| S75 | 228.22 | 1.16× | 0 | 0.676 | 0.237 | 0.300 |
| S80 | 229.74 | 1.15× | 0 | 0.658 | 0.279 | 0.594 |
| A1S75 | 173.67 | 1.52× | 17 | 0.699 | 0.225 | 0.311 |
| A3S75 | 134.34 | 1.97× | 29 | 0.665 | 0.271 | 0.362 |
| A3S80 | 138.46 | 1.91× | 29 | 0.674 | 0.274 | 0.516 |

SSIM and LPIPS compare all 90 aligned decoded RGB frames. LPIPS uses AlexNet
v0.1 with recorded package and weight hashes. The report also includes PSNR,
mean scores, the three worst SSIM frames, and temporal RMS of changes in the
candidate-minus-baseline RGB error. Audio comparisons use decoded 32 kHz stereo
samples and include waveform/cosine, spectral and log-spectral differences,
levels, clipping and silence fractions. All decoded waveforms contain 3.776 s
of samples, including codec padding; both stream durations are 3.75 s. Padding is
identical across variants and retained in the audio comparisons.

**Every approximate variant fails the complete historical similarity checks.**
Those unchanged diagnostics require every-frame SSIM ≥0.90, LPIPS ≤0.10,
temporal RMS ≤0.03, audio relative L2 ≤0.05 and audio cosine ≥0.995. A1 passes
the LPIPS threshold but misses SSIM, temporal and audio thresholds. Visual
acceptance is established by the user's review recorded above; the numerical
diagnostics continue to describe differences from D0 and do not establish
default parity.

Adaptive caching reduced denoising wall time from 203.03 s for D0 to 116.93 s
for A1 and 75.49 s for A3. The combined variants recorded their required phase
refresh and never dispatched suffix attention on hits. Standalone SubBlock
executed 1,960 sparse block calls per video; retained block-pair density was
45.59% for S75 and 40.82% for S80 after protection and budget rounding.

The standalone SubBlock wall ratios do **not** demonstrate an attention-kernel
speedup: total measured attention event time was 32.28 s for S75 and 30.21 s
for S80, versus 28.16 s for D0. Streamed weight transfers, overlap and physical
page residency affect wall time, and there are no repeated runs to separate
those effects. S80 was slightly slower than S75. Adding SubBlock to A1 changed
wall time by only 1.15 s; both aggressive combinations were slower than A3
alone and introduced larger measured differences.

Sampled peak device memory was 6.22 GiB, or 6.30 GiB for combined variants;
peak process RSS was approximately 36.68 GiB. The adaptive allocation was
272,408,088 bytes, within its 512 MiB cap. No attention timing samples were
missed; the largest memory sampling gap was 1.008 s. The report retains tensor
and pinned allocation counters as well as sampled peaks.

## Recommendation

Keep both features off by default. For an approximate
alternative with the closest measured match to D0, start with **A1**: it has
the smallest measured visual, temporal and audio differences among the eleven
approximate variants, with an
observed 1.51× wall-time ratio. A3 trades larger differences for an observed
1.98× ratio. The fixed reuse controls are faster but diverge more visually.
There is no measured reason on this workload to prefer the combined recipes
over adaptive caching alone.

Do not extrapolate these streamed-weight, single-prompt observations to resident
weights, other seeds/tasks, H200 or other hardware. The user's visual approval
covers this twelve-video comparison; both defaults remain off.

The original local audit passed all **163 asset hashes**, twelve complete media
decodes, AV checksums/finite values, exact case/transition/frame counts, static
and dynamic report links, metric identities and final source identity.
Its log is `outputs/adaptive-subblock/work/local-report-audit.log`.
The acceptance update verifies **164 asset hashes**, including the review
record, and confirms unchanged videos, AV states, numerical measurements and
qualified source. Its audit is
`outputs/adaptive-subblock/work/visual-acceptance/audit.json`.
