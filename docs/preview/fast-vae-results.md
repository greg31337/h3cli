# Faster original-model VAE: CUDA results

> Historical record: `--full-vae-execution` and its alternate implementations
> have been removed. See the [current original-VAE guide](fast-vae.md) for
> supported execution and validation. Results below describe the earlier code.

Qualification date: 2026-09-23. **M1–M6 complete.** The CUDA quality, six-step
wall-time, memory, compatibility and local playback checks pass.
The campaign completed in **187.9 minutes (3 hours 8 minutes)**, including
setup, failed attempts and reporting, within the eight-hour limit.

Use `--full-vae-execution balanced` for the native CUDA implementation.
`--preview-vae` remains the existing TAEH3 path; `legacy` remains the default.
The optional TensorRT adapter requires an offline engine built for the current
checkpoint, GPU and runtime. See [usage and build instructions](fast-vae.md).

The [playback gallery](../outputs/fast-vae/cuda/review.html) contains complete
videos, shared playback/seeking, float-domain metrics, worst frames, seam crops,
commands, logs and memory records. The [report index](../outputs/fast-vae/index.html)
links available backends. Metal acceleration is M7 and is not qualified here.

## Hardware and comparison contract

- RTX PRO 6000 Blackwell Server Edition, SM120, approximately 95.6 GiB VRAM;
  **not RTX 5090**. Driver 595.91.07, CUDA toolkit 12.8.
- Optional TensorRT 10.13.3.9. Source base
  `2abc877cd65e8b2977772b38731480d26c4f8274`; the retained inventory includes
  individual source hashes, build flags, binary hashes and engine metadata.
- C1: 640×480/243 frames. C2: 1344×768/243 frames. Both use seed 42, six
  completed denoising steps, original BF16 DiT weights and `--fast-cuda`.
  Only the presentation decoder changes in matched complete renders.
- Matched complete renders and decoder profiles enable `H3_PROFILE=1`.
  Absolute timings include bounded profiling overhead; comparisons use
  the same instrumentation. No GPU tests run concurrently.
- Complete-decoder measurements reuse the same saved latents. FP32/native
  pairs alternate order, with a first decode and three warm repeats. TensorRT
  also has three warm repeats. The legacy fast-CUDA comparator has one warm
  repeat. “Cold” means a new process/decoder, with filesystem caches intact.
- C3 expands the last latent section of C2 to 362 frames for **decode-only**
  geometry/memory stress. It is not an independent 362-frame generation.
  C4 is a denoiser-free photo-pan/text corpus encoded by the original encoder.

## Complete video decoder

Seconds include tile compute, output copy/unpack and ordered host stitching.
They exclude audio and FFmpeg. First repeats also write raw comparison RGB;
warm repeats discard sink payloads. Load time is reported separately.

| Case / mode | Load | First decode | Warm median | Warm range |
| --- | ---: | ---: | ---: | ---: |
| C1 FP32 | 0.67 | 105.98 | 105.31 | 105.30–105.39 |
| C1 legacy fast CUDA | 0.66 | 27.60 | 27.43 | one repeat |
| C1 native BF16 | 0.57 | 11.97 | 11.70 | 11.69–11.80 |
| C1 TensorRT FP16 | 8.55 | 5.52 | 5.40 | 5.40–5.43 |
| C1 TAEH3 preview | 0.29 | 0.95 | 0.83 | 0.83–0.85 |
| C2 FP32 | 0.66 | 329.06 | 328.37 | 328.32–329.08 |
| C2 legacy fast CUDA | 0.68 | 86.28 | 85.78 | one repeat |
| C2 native BF16 | 0.57 | 37.13 | 36.80 | 36.75–36.92 |
| C2 TensorRT FP16 | 8.35 | 17.82 | 17.29 | 17.26–17.30 |
| C2 TAEH3 preview | 0.25 | 2.99 | 2.84 | 2.83–2.84 |

Native balanced is **2.34× / 2.33× faster than legacy fast CUDA** at C1/C2;
the corresponding FP32 comparisons are 9.00× / 8.92×. TensorRT is
5.08× / 4.96× faster than legacy fast CUDA while warm. Its approximately
8.4-second validation/load cost reduces the gain on short cold requests.
These are decoder speedups, not whole-render speedups.

## Complete six-step renders

All four modes completed six steps at both 243-frame geometries. These are
complete process wall times, including startup, denoising, audio and mux:

| Geometry | FP32 | Legacy fast CUDA | Native balanced | TensorRT | Native wall saved vs legacy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 640×480 | 166.29 s | 88.82 s | 71.45 s | 71.48 s | 19.6% |
| 1344×768 | 661.56 s | 421.41 s | 367.45 s | 357.50 s | 12.8% |

The native candidate passes the 1.5× decoder and 5% whole-wall gates, with no
high-resolution regression. Versus FP32, its wall-time reductions are 57.0%
and 44.5%. TensorRT saves 19.5% and 15.2% versus legacy fast CUDA. Native and
TensorRT cold C1 wall times are effectively tied; that difference is noise.

These are single final complete runs per mode, supported by the repeated
decoder measurements above. Whole-render peak VRAM remains dominated by DiT,
approximately 41.1 GiB at C1 and 50.9 GiB at C2; halving decoder residency does
not halve the full-render peak.

The FP32 C1 profile spends about 82.6 seconds in attention and 19.4 seconds
in GEMMs. Native balanced reduces those to about 4.95 and 4.15 seconds.
Legacy fast CUDA already accelerates attention; BF16 linears account for much
of the additional native gain. No DiT attention kernel is changed here.

## Reconstruction quality

Every frame is compared in float RGB at native resolution before codec loss.
Frozen gates require PSNR ≥35 dB, SSIM ≥0.98, worst-frame PSNR ≥30 dB,
at least 3 dB over TAEH3 and LPIPS at most 80% of TAEH3's error.
All completed rows below pass. LPIPS uses AlexNet v0.1; metric and model
hashes are retained with each case.

| Corpus | Native PSNR | Native SSIM | Native LPIPS | TensorRT PSNR | TAEH3 PSNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| C0 short/tail | 71.65 | 0.999970 | 0.00000898 | — | 29.95 |
| C1 | 73.46 | 0.999969 | 0.00000907 | 68.37 | 33.29 |
| C2 | 74.28 | 0.999974 | 0.00001516 | 68.73 | 36.16 |
| C3 repeated tail | 74.02 | 0.999974 | 0.00001760 | 69.04 | 33.25 |
| C4 face/text motion | 68.83 | 0.999979 | 0.00000474 | 67.23 | 25.35 |
| C4 texture/text motion | 71.73 | 0.999978 | 0.00000285 | 67.53 | 28.13 |

These are close reconstructions, not bitwise equivalence. For example, native
C2's maximum float error is 0.02759 (maximum RGB8 channel difference 7),
despite its high average PSNR. The gallery includes maximum/temporal errors,
every-frame CSVs, tile-boundary bands and magnified seam differences.

The independent PyTorch FP32 tile oracle gives relative L2 3.081e-7 and
maximum absolute error 1.609e-5. The FP32 TensorRT export gives relative L2
7.415e-7. The range audit's largest FP16 matrix operand is 177.71 on its
captured tile. These checks support the export/precision recipe on the tested
data; nonfinite runtime output still causes an error.

## Memory and execution decisions

Tracked decoder peaks are approximately 9.36 GiB for FP32/legacy, 4.97 GiB
for native BF16, and 4.73 GiB for TensorRT. TensorRT includes its aligned
external allocator; summing internal/external high-water marks is conservative.
Library/context overhead and whole-device sampled usage are separate.

Three native C3 decodes take 56.15, 55.61 and 55.63 seconds, with the same
tracked peak each time. Whole-device sampled peak is 6.12 GiB and main-process
sampled RSS is 1.51 GiB. Lifetime checks also assert stable live allocations,
streamed/materialized pixel identity, cancellation, low-memory rejection,
nonfinite-input recovery, failed sinks and clean retry. Host memory for
streaming output scales with a temporal chunk; the materialized float API
intentionally retains its full-video allocation.

Decisions from measured experiments:

- Keep tile batch 1. Independent stream/context batches 1/2/4 take warm
  medians 102.81/189.20/418.70 ms per batch, including readback and ordered
  stitching. Batch 2 offers about 8% per-tile improvement for twice the
  decoder residency; batch 4 loses throughput. Shared-weight multi-stream
  batching is not implemented or claimed.
- Keep fused Q/K normalization, RoPE and BF16 stores for a 42 MiB scratch
  reduction. Final tile latency is effectively equal to unfused execution
  (about 90–91 ms); no independent fusion speedup is claimed.
- Keep bounded pinned, same-stream asynchronous D2H staging and the existing
  host stitcher. Pageable/pinned tile timing is effectively equal. This
  removes an extra synchronization boundary but does **not** establish a
  cross-tile copy/compute overlap speedup.
- CUDA Graph capture is off by default. cuBLASLt capture was rejected on this
  node. The opt-in diagnostic aborts capture and safely retries ordinary
  execution before publishing output; no graph speedup is claimed.
- Integrate optional TensorRT: its complete-tile warm result is about 39 ms
  versus native's approximately 90 ms, comfortably exceeding the 10% gate.
  Engine manifests enforce checkpoint/config/engine hashes, shape/precision,
  GPU UUID/SM and CUDA/TensorRT versions. No runtime Python or engine builder.

## Reference encoding

FP32 causal convolution retains its arithmetic and first cuBLASLt heuristic,
while reusing bounded im2col scratch and descriptors. Single-run measurements:

| Input | Baseline seconds | Reuse seconds | Allocations before → after |
| --- | ---: | ---: | ---: |
| Image `match` | 0.755 | 0.727 | 1207 → 932 |
| Image `max` | 4.214 | 3.857 | 8588 → 6422 |
| Short video | 0.407 | 0.405 | 239 → 212 |
| Temporal-tail video | 0.500 | 0.507 | 239 → 212 |

The max-image result supports retaining reuse; the video differences are too
small to claim a speedup. Persistent scratch adds about 18.5 MiB to the image
case's tracked peak. A broader cuDNN/layout rewrite was not warranted by this
bounded experiment. Posterior mean/log variance, sampled latents and every
RNG epsilon value are **bitwise identical** across all four isolated pairs.
The existing official-posterior fixture retains its prior maximum difference
of 0.00048828125 at an FP16 rounding boundary; the RNG fixture matches exactly.

Both Ref2VA `match`/`max` six-step pairs also produce bitwise-identical AV states
with `H3_FAST_CUDA_GEMM_TUNE=0`. Original cross-process runs with timing-based
GEMM selection differed by relative L2 0.0940 (`match`) and 0.0359 (`max`).
Those runs remain diagnostics rather than evidence of decoder-policy identity.
The fixed-selection runs isolate encoder reuse and decoder choice; no production
autotuning behavior has been changed to obtain that comparison.
An unchanged legacy-reference repeat differs from its first run by relative L2
0.09358, confirming that this variability also occurs without the new decoder.
Use fixed GEMM selection when requiring reproducible cross-process Ref2VA
comparisons. The main C1/C2 performance runs retain normal timed selection,
and all four decoder modes produced identical saved AV states there.

## Scope and retained failures

The implementation uses original thin C/CUDA/C++/ONNX code. No VPIPE or
ComfyUI implementation was copied; pinned reference revisions and optional
SDK versions are in the guide and inventory. Original TAEH3, audio and image
VAE choices remain independent. Ordinary builds have no TensorRT dependency.
Local M4 Max checks cover build, host regressions and documented fallback,
not Metal mixed-precision performance.

The ledger retains failed experiments and harness fixes: CUDA Graph rejection,
TensorRT allocator alignment, a single-tile metric division by zero, host tests
intercepted by the generation-only step cap, explicit `fast_cuda` in a resume
request, too-short reference-video inputs, and a stale session-test object that
omitted its last repetition. Successful named retests are
required by the closing audit. Failed attempts produce no qualified partial
generation; every new completed inspection render uses six denoising steps.

The closing audit passes **191 checks**: 11 latent-identity groups, three
fusion/delivery identity comparisons, 83 media files, 26 complete CLI render
records, 14 API-render provenance records and 54 required test results.
All four encoder posterior/RNG pairs and all 11 required clip-quality cases
also pass. The inventory retains 163 model-file identities and verifies that
132 production source files match the frozen archive used for the final build.
Memory sampling's largest recorded gap is 1.37 seconds; sampled peaks can miss
shorter transients. Peak main-process RSS falls from approximately 4.44 GiB
with legacy delivery to 1.78 GiB with balanced delivery on C2.

The [local media/link audit](../outputs/fast-vae/cuda/report-validation.json)
and [Chrome playback check](../outputs/fast-vae/cuda/browser-validation.json)
verify the delivered pages and their synchronized controls.
