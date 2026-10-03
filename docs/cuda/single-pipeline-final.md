# Final single-CUDA qualification (M6/M7)

> Subsequent decoder cleanup removes `--full-vae-execution`, the balanced and
> TensorRT alternatives, and their consumers listed in this historical report.
> See the [single-decoder validation record](vae-single-results.md).

M6 and M7 are complete on the RTX PRO 5000 72 GB at the qualification host.
There is one CUDA video implementation; the obsolete fast/legacy infrastructure
is removed. The closing unchanged regression passed **204/204 artifacts**, and
both complete C1/C2 oracle replays passed. Optional testing was limited to three
representative pairs as requested, with all quality differences reported.

Final source identity:
`ae6a8449b68f8a3fd64a18d55125415f6645ffddb75500da94ab4f11a6814a79`.

[Playback and results](../../outputs/cuda-final/review.html) ·
[Side-by-side optional videos](../../outputs/cuda-final/review-options.html) ·
[Closing gate](../../outputs/cuda-final/evidence/final-gate/result.json) ·
[Source audit](../../outputs/cuda-final/source-audit.json) ·
[Report/checksum audit](../../outputs/cuda-final/report-audit.json).

## Removal and retained consumers

There is one CUDA video request policy: SGLang arithmetic 4 with independent
attention and projection precision. Default dense BF16 keeps the frozen oracle.
Sage2++, Sage3 and SOL change main DiT attention; FP8/NVFP4 change eligible
QKV/output/MLP projections. They do not introduce another renderer or sampler.

M6 deletes the unreachable old fast Tensor Core attention, GEMM autotuning,
QKV fusion, TF32 decoder route, cuDNN SDPA wrapper, and device-weight cache.
Mapped host weight staging and bounded stream/copy fallback remain. Six unused
Metal-derived CUDA kernels are removed at the generator; all 47 remaining
portable generated kernels have CUDA launches. The authoritative Metal shaders
are unchanged. Third-party notices remain with the current consumers.

| Retained code | Current consumer |
| --- | --- |
| Portable scalar kernels and fixed-head/online attention | CUDA still, component APIs and frozen operator fixtures |
| Generic PCG, schedule and non-SGLang arithmetic in shared C | Metal/CPU and single-still execution |
| cuDNN reference convolution | Video/audio conditioning encoders and reference decoder |
| cuBLAS FP16 full-VAE attention and softmax | Explicit balanced full decoder, independent of preview VAE |
| Shared GEMM plan representation | Convolution, quantized projections and full-VAE GEMMs |
| `h3_execution_exchange` diagnostic bit and zeroed old stats | Immutable low-level probe ABI; no production math selection |
| Retired API fields and state-version checks | Actionable rejection of fast/legacy requests and old sampler states |
| SGLang reference scopes | Text, vision, audio, decoder and main DiT rounding boundaries |

Generation, resume, decode and quant preparation no longer set the old diagnostic
bit. Public APIs reject nonzero retired fields. Legacy CLI/environment selections
fail instead of activating compatibility rendering. Sampler files keep existing
wire-format/version detection; no old trajectory is relabeled. Clean AV states
retain their independent presentation and decoder compatibility checks.

Old mutable fast/two-mode launchers and device-cache tests are removed. Generic
operator and media comparison tools are now `cuda_single_attention.py` and
`cuda_single_quality.py`. The frozen sixteen-file suite, all goldens, fixtures,
thresholds, dependencies and runner are unchanged. Historical documents and
artifacts remain labeled as such.

## Validation scope and limits

The user requested minimal optional testing instead of another full matrix.
Selected combinations are Sage2++/BF16, Sage3/NVFP4 and SOL/FP8. Each gets one
matched default/candidate run at 640×480 / 243 frames / six evaluations, NUMA0,
warm packed weights and original full VAE. These are individual observations,
not medians or a statistical ranking. Their complete clips also supply visual
comparisons, avoiding duplicate C0 renders. Each campaign retains a four-hour
limit; all GPU work is serialized.

BF16 reference coverage remains complete. The twelve-cell M3B/M4 campaign is
historical evidence, not a final-source rerun. Other optional combinations are
not qualified again by this reduced campaign.

BF16 held-outs cover another no-reference prompt, image match/max, mixed
image/video/audio and continuation. Approximate conditioning/continuation
combinations are not rerun; protected-range operator evidence is retained. One 1344×768 / 362-frame / max-image /
two-evaluation case checks execution only. Retained C1 (362/6) and C2 (124/50) passed on final source against unchanged
oracle trajectories/media. All 224 video/audio velocity and state arrays are
bitwise equal. Both decoded-media comparisons pass the unchanged limits, and
live device/pinned allocations remain constant across every denoising evaluation.
Decoded RGB pixels are identical in both cases. C2 MP4/audio is byte-identical;
C1 decoded audio has relative L2 0.002483 and cosine 0.999997, within the unchanged
audio limits. This is not a claim of bitwise C1 audio equality.

Approximate quality threshold failures are reported, not promoted to parity
passes. Playback pages show reference/candidate videos, worst frames, temporal
and audio metrics, timings, peak memory and dispatch counters. Human review is
not assumed or awaited. Historical failures remain in the evidence.

Local Metal storage/cancellation smoke passes. A 256×256 still decode with
cancellation and two repeated decodes matches the retained pixels byte-for-byte;
live GPU bytes remain constant. Host sampler, continuation, bridge, progress,
preview-cache and still metadata tests pass. The initial mutable continuation
cache assertion expected the removed fast-mode key and was migrated to explicit
attention-policy separation. A subsequent CUDA host fixture inherited the
six-evaluation render cap; its failed attempt is retained and the external
launcher now removes that cap only for host-only checks. The high-resolution
stress launcher initially set a two-evaluation budget, which the existing guard
rejects (it accepts 6 or 50). That attempt stopped before loading the model; the
external launcher uses a ceiling of six while the render remains exactly two
evaluations. No renderer, frozen test or threshold changed for these repairs.

## Minimal optional performance observations

Each row is one adjacent default/candidate pair on final source, 640×480,
243 frames, six evaluations and full VAE. Values above 1× are faster.

| Attention / weights | BF16 default wall | Candidate wall | Wall speedup | Denoise speedup | Candidate peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sage2++ / BF16 | 147.22 s | 119.43 s | 1.233× | 1.166× | 8.64 GiB |
| Sage3 / NVFP4 | 129.15 s | 90.44 s | 1.428× | 1.715× | 8.58 GiB |
| SOL / FP8 | 129.21 s | 183.07 s | 0.706× | 0.569× | 8.78 GiB |

SOL/FP8 is slower here: 41.7% more wall time and 75.7% more denoising time.
Sage3/NVFP4 has the largest observed improvement among these selected pairs.
This reduced test does not rank untested combinations or isolate quantization
from attention. All three default peaks were 8.14 GiB; optional GPU memory did
not decrease, although quantized candidates used less host RSS (11.16 GiB for
NVFP4 and 18.92 GiB for FP8, versus approximately 36.85 GiB for BF16).

Default wall times varied from 129.15 to 147.22 seconds. In particular, the
Sage2++ pair includes a loading difference (30.55 versus 22.40 seconds); its
whole-request gain must not all be attributed to attention. Denoising improved
from 64.70 to 55.50 seconds. Packed quantized weights were warm; cold packing is
not measured here. These are observations, not medians or repeatability claims.

All three optional clips exceed the unchanged historical image/audio quality
limits; SOL/FP8 also exceeds the temporal-error limit. They remain explicit
approximate choices, not reference-parity passes or visually equivalent outputs.

| Candidate | Minimum frame SSIM | Maximum frame LPIPS | Audio relative L2 |
| --- | ---: | ---: | ---: |
| Sage2++ / BF16 | 0.47841 | 0.52120 | 0.46174 |
| Sage3 / NVFP4 | 0.40263 | 0.57634 | 0.71072 |
| SOL / FP8 | 0.57255 | 0.46261 | 0.40357 |

Every decoded frame is compared; the HTML includes the worst frame and amplified
difference images. No human review or acceptance is asserted. The local report audit verifies all
551 downloaded records/playback assets by checksum and finds no missing links.

## Conditioning and bounded stress results

All five BF16 held-outs passed. These are execution/coverage observations, not
paired SGLang performance claims. Every clip uses the original full decoder.

| Case | Frames / evaluations | Wall | Peak VRAM |
| --- | ---: | ---: | ---: |
| Different unconditioned prompt, BF16 | 124 / 6 | 82.31 s | 6.22 GiB |
| Image `match`, BF16 | 124 / 6 | 112.92 s | 6.22 GiB |
| Image `max`, BF16 | 124 / 6 | 153.60 s | 7.76 GiB |
| Continuation, BF16 | 124 / 6 | 105.20 s | 6.22 GiB |
| Mixed image/video/audio, BF16 | 124 / 6 | 193.97 s | 9.07 GiB |
| 1344×768 image `max`, Sage3/NVFP4 | 362 / 2 | 285.38 s | 36.25 GiB |

The last clip demonstrates bounded high-resolution execution, not converged
visual quality. All other rows use 640×480. Peaks cover the complete request,
including conditioning and decoding. The initial stress budget-guard failure is
retained alongside the corrected two-evaluation run.

## Per-change frozen gate history

Each coherent cleanup patch passed all 204 artifacts before the next patch.
All final measurements use the same source identity.

| Patch | Scope | Test time after build |
| --- | --- | ---: |
| M6p1 | Production and generated-kernel removal | 130.77 s |
| M6p2 | Shared tool and consumer migration | 128.68 s |
| M6p3 | Mutable render geometry validation | 125.91 s |
| M6p4 | Remaining obsolete launchers and flags | 123.24 s |
| Closing gate | Same final source after complete qualification | 130.07 s |

These times measure the regression runner, not end-user rendering. Exact source
identities and build logs are retained per patch. The closing gate passed again
on the final source after the complete-render measurements (238.62 seconds
including its isolated build). The optional campaign took 15.70 minutes; oracle
replays plus comparisons took 8.00 minutes, and held-outs including the repaired
setup took 15.67 minutes. Each stayed within its four-hour budget.

## Usage and regression requirement

Use the [single-pipeline commands and build options](design-single-pipeline.md).
`--fast-cuda`, legacy CUDA selection and old `H3_FAST_CUDA*` tuning are retired.
`--preview-vae` remains independent. FP8/NVFP4 and Sage qualification is specific
to the installed SM120/pinned libraries; this campaign does not establish Hopper,
Ampere or other GPU performance. Minimal builds reject unavailable video paths;
they never fall back to the removed renderer.

Run `make test-cuda-reference-regression` after every coherent code change and
before another unrelated patch. Compare all outputs with the recorded golden
state. See [CONTRIBUTING.md](../../CONTRIBUTING.md) for portable test setup.
