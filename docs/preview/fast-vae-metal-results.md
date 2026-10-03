# Original-model VAE on M4 Max

> Historical record: `--full-vae-execution` and its alternate implementations
> have been removed. See the [current original-VAE guide](fast-vae.md) for
> supported execution and validation. Results below describe the earlier code.

Status: **M7 implemented and tested on 2026-09-23, with partial performance
qualification.** Quality, correctness and bounded-memory checks pass; the
1.5× decode and 5% whole-render speed targets are unmet. The campaign finishes in **175.5 minutes (2 h 56 min)**
within its unchanged four-hour deadline, including setup, failures and
reporting. The original FP32 path remains the default.

Evidence: [Metal playback report](../outputs/fast-vae/metal/review.html),
[campaign ledger](../outputs/fast-vae/metal/ledger.json),
[source and hardware inventory](../outputs/fast-vae/metal/inventory.json),
[source snapshot](../outputs/fast-vae/metal/source.json),
[validation sources](../outputs/fast-vae/metal/validation-source.json), and
[frozen gates](../outputs/fast-vae/metal/gates.json).

## Selected implementation

`--full-vae-execution balanced` now selects an actual Metal decoder instead
of the former FP32 fallback. It uses original VAE matrices in BF16 storage,
shape-cached MPSGraph linears and a fused MLP graph. Bias, SwiGLU, residuals,
norms and final projection output remain FP32. The graph declares BF16 matrix
results before casting to FP32; Apple controls the internal matrix lowering.
This is a measured close-reference recipe, not an FP32 arithmetic guarantee.

Dense attention specializes the existing Steel-derived kernels for H=32/D=64,
with BF16 Q/K/V storage, FP16 matrix operands and FP32 softmax/accumulation.
The decoder preserves spatial/temporal geometry, performs GPU patch-to-RGB
unpacking, and feeds the original ordered host stitcher and bounded frame sink.
It adds bounded scratch and graph caches, allocation admission and finite/range
checks before frames are published. Original image, tiny, audio and encoder
execution remain separate. M5 NAX is neither selected nor qualified.

## Complete decoder measurements

Local hardware: 40-core M4 Max, 128 GiB unified memory, macOS 26.6.2.
Times include tile computation, unpack/copy and stitching. Warm repeats discard
sink payloads; first repeats also write raw RGB. Weight loading is separate.
Both decoders remain resident during alternating pairs. A first decode is not
a purged operating-system file-cache measurement.

| Case | FP32 decode | Balanced decode | Speedup | Tracked FP32 / balanced peak |
| --- | ---: | ---: | ---: | ---: |
| 640×480, 243 frames | 110.833 s | 98.063 s | 1.130× | 9.365 / 4.684 GiB |
| 1344×768, 243 frames | 344.813 s | 306.094 s | 1.126× | 9.365 / 4.684 GiB |
| 1344×768, 362 frames, single first-decode stress pair | 517.750 s | 458.245 s | 1.130× | 9.365 / 4.684 GiB |

C1/C2 table values are warm medians; C3 has one first-decode pair for stress,
not a warm performance gate. C1 has two warm observations per decoder: FP32 110.832–110.833 s and balanced
98.055–98.071 s. Its first decodes were 110.909 / 98.158 s. The 1.5× C1 decode
target is **not met**. C2's no-more-than-3% decode regression gate passes.

The matched complete six-step C1 runs take **674.858 / 660.981 seconds**
(FP32 / balanced), a **2.056% wall-time reduction**. The 5% whole-render target
is **not met**. Application denoising times are 538.327 / 538.226 seconds and
video decode times are 110.901 / 98.219 seconds. Both finish all six steps and
produce identical saved latents. This is one matched pair with roughly
one-second parent timing resolution, not a confidence interval. There is no
matched C2 whole-render pair; C2 has a complete six-step fixture capture and
paired complete-decoder measurements.

Tracked decoder allocations are not total process memory. The report retains
sampled RSS/physical footprint, application memory snapshots, Metal allocation
counts, sampling gaps and compression/swap observations. Unified GPU buffers
overlap process footprint and must not be added to it.
Whole-render sampled footprint is approximately 45.17 GiB for both C1 runs:
the denoiser peak dominates despite the decoder's smaller allocation peak.

## Quality and compatibility

Every comparison decodes the same state with FP32, balanced and unchanged
TAEH3. Metrics use native-resolution float RGB before codec loss, every frame,
Gaussian SSIM and LPIPS AlexNet v0.1. Full playback clips and worst-frame/seam
images are local in the report.

| Case | Balanced PSNR / SSIM | TAEH3 PSNR | Frozen quality gates |
| --- | --- | ---: | --- |
| C0 cropped six-step state | 71.672 dB / 0.999964 | 31.766 dB | Pass |
| C1 640×480 / 243 | 73.084 dB / 0.999974 | 31.447 dB | Pass |
| C2 1344×768 / 243 | 74.031 dB / 0.999974 | 35.159 dB | Pass |
| C3 1344×768 / 362 | 73.853 dB / 0.999974 | 32.527 dB | Pass |
| C4 faces/text | 69.340 dB / 0.999980 | 25.350 dB | Pass |
| C4 texture/motion | 72.001 dB / 0.999979 | 28.132 dB | Pass |

C1's worst frame is 70.969 dB and worst seam band is 69.976 dB. Balanced LPIPS
is 0.00001735 versus 0.13599 for TAEH3. This is reconstruction evidence for the
tested corpus, not a claim that all precision differences are invisible.

The retained C1 park scene has verified complete six-step BF16 provenance.
C2 is a new complete six-step piano scene; capture plus preview presentation
took 2921.48 seconds. C3 repeats the tail of C2 for decode/memory stress and is
not an independent 362-frame generation. C4 uses denoiser-free image pans/text
passed through the unchanged original encoder. The matched C1 whole-render
pair is a separate piano prompt. No inspected generated video reduces the
requested six completed denoising steps.

Compatibility checks pass: unpack/crop/tail/range operators, standalone
and streamed equality, cancellation, failed admission, sink failure and retry,
cached reference/balanced/preview latent identity, shape changes, original still
and preview checks, and all four encoder posterior/RNG pairs. The lifetime
test's sampled footprint returns to a stable approximately 5.23 GiB after the
temporary second decoder is freed. Exact sampler resume also preserves latents
when switching presentation policy. A 90-frame continuation with 39 protected
frames produces 51 delivered frames, including decode-only from its saved
presentation sidecar. The final image-max pair passes with identical saved
latents, taking 414.629 / 408.594 seconds (reference / balanced). All **159
closing checks pass**, including **36 videos** with frame-count/duration/audio
checks, continuation/decode-only pixel identity, source/archive checksums and
memory evidence. Three harness unit tests also pass, covering persistent
deadlines, six-step enforcement and inter-runner lock/dependency behavior.

## Rejected and limited experiments

Final 36-block tile tests retain FP16 MPSGraph, direct FP16 MPS, BF16 MPSGraph,
unfused MLP/QKV and FP16/BF16/MPSGraph attention alternatives. BF16 fused MLP
is approximately 0.778 s versus 0.880 s for the FP32 reference. Unfused BF16 MLP
takes approximately 0.827 s and direct FP16 MPS approximately 0.884 s. Attention
variants are much closer; a sub-percent isolated difference is not a qualified
whole-decoder gain. The selected recipe's raw projected-tile relative L2 is
0.000928 (limit 0.02). The FP16 MPSGraph candidate fails that raw gate at
0.04393 as well as losing on speed; direct FP16 MPS passes it at 0.000317
but does not beat the selected BF16 fused decoder. The forced-reference
override reproduces the original FP32 tile bitwise. The rejected FP16
MPSGraph diagnostic also grows from 9.02 to 12.99 to 16.72 GiB process footprint
across three tiles, despite flat tracked Metal buffers. The selected BF16
recipe settles at 5.09 GiB. No full-render FP16 MPSGraph qualification is claimed.

The bounded ANE experiment uses real VAE weights, representative range-safe
synthetic activations and 1797 rows. It tests 256/512 ANE rows while the GPU
computes the remaining rows, including conversion, packing, synchronization
and merging back to GPU storage. All eight numerical observations pass
relative-L2 <0.02 (observed approximately 0.00065–0.00096), but none passes
the speed gate. GPU-only W1/W2 take approximately 11.1/10.0 ms; splits take
27–43/32–50 ms. Packing dominates. This prototype is rejected and no production
ANE decoder option is enabled. Cold compilation takes approximately
0.47–0.81 seconds per model. A reporting-only retest fixes the buffer-byte field:
reported ANE buffers are 37–82 MiB, excluding process/library overhead. Its
cached model loads take 0.045–0.077 seconds and the speed decision is unchanged.

Setup records retain Objective-C/Metal compile errors and a failed exact-unpack
check. Unpack normalization was corrected to use the ARM64 reference's fused
multiply-add before final operator testing. An unavailable optional offline
Metal compiler did not prevent runtime Metal compilation. Planned continuation
commands with a 22-frame context were superseded before execution by the valid
39-frame-context checks; those original plans remain visible in the ledger.
The first image-max compatibility attempt hit its 300-second timeout during
initialization, before denoising. Its logs and failed status remain retained;
the replacement pair uses a measured M4 vision allowance of 660 seconds each
without extending the campaign deadline.

## Scope and reproduction

The local harness uses its original persistent 240-minute deadline, a 25-minute
closing reserve, measured admission estimates and serial GPU jobs. C1 uses
one first plus two warm pairs; C2 uses one first plus one warm pair. Six-step
state reuse, compact compatibility cases and an early ANE rejection fit the
M4 budget. CUDA records remain independent and unchanged.

The retained manifests, commands, binary/source/model checksums, metric package
versions and test outputs are linked from the report. Regenerate closing
artifacts with:

```sh
outputs/fast-vae/metal/venv/bin/python tests/fast_vae_metal_audit.py
python3 tests/fast_vae_metal_report.py --verify-media
PYTHONPATH=/tmp/h3-report-qa python3 tests/fast_vae_report_browser.py \
  outputs/fast-vae/metal/review.html
```

The browser helper requires Chrome and `websocket-client`. Rendering itself
adds no Python, TensorRT or ANE dependency. See the [usage guide](fast-vae.md).
