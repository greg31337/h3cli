# CUDA fast-preview results

**Completed: 26 validated videos, 13 explicitly blocked slots and one user-skipped SOL slot across all 40 requested slots.** All 20 tasks in the [archived checklist](fast-preview-tasks.md) are complete. No h3cli implementation, defaults, kernels, policies, fixtures or goldens were changed. Each completed slot has exactly one renderer invocation and one video; none was selectively rerendered. The interrupted SOL slot retains its attempt record.

Open [the video gallery](../../outputs/cuda-fast-preview/2026-09-27-pro5000/index.html) for all forty cells, grouped by resolution and conditioning, with embedded videos, downloads, exact commands and total wall seconds. The [JSON](../../outputs/cuda-fast-preview/2026-09-27-pro5000/results.json), [CSV](../../outputs/cuda-fast-preview/2026-09-27-pro5000/results.csv), [attempt ledger](../../outputs/cuda-fast-preview/2026-09-27-pro5000/attempts.json) and [final audit](../../outputs/cuda-fast-preview/2026-09-27-pro5000/audit.json) retain measurements and validation evidence. Generated artifacts are gitignored.

## Workload and qualification

All runs used the RTX PRO 5000 72GB / SM120, driver 595.91.07, CUDA 13.0.3 (nvcc 13.0.88) and cuDNN 9.20. Each requested **124 frames at 24 FPS**, seed **42**, all **50 blocks**, `--core-reuse 1`, `--cuda-weight-mode auto` and `--profile`. Local M4 work was preparation, media review and reporting. Optional Sage2++/Sage3, SOL, SubBlock, FP8/NVFP4 and preview-VAE support were compiled into an isolated build.

The [frozen manifest](../../outputs/cuda-fast-preview/2026-09-27-pro5000/manifest.json) records all ten exact flag combinations. P01/P02 are six-step dense BF16 controls with full/preview video VAE. P03 uses six steps, reuse 3, Sage3 and NVFP4; P04 uses twelve steps, reuse 2, Sage2++ and FP8; P05 uses twenty steps, SOL and NVFP4. P06–P10 exercise the specified adaptive and SubBlock warmup boundaries. Audio remains enabled throughout. These six-step controls are preview baselines, not a high-step quality reference.

> The camera tracks slowly along a sunny beach as a woman in a white dress turns toward the camera, smiles, and brushes windblown hair away from her face. Gentle waves roll onto the sand behind her. Natural daylight, realistic motion. Soft surf and a light ocean breeze are audible.

Every completed reference arm used exactly `inputs/1.jpg --ref-image-size max`: **1365×1821 → 2048×2720, 21,760 raw patches**, at both output sizes. The fixture and TAEH3 preview decoder retained their frozen SHA-256 identities. Base model weights were identified by filesystem metadata; their payloads were not hashed. Model loading and necessary quantization reads are ordinary execution.

Source revision: `57cd0e34cb170d0abf8337af8a454c4b63784329`. Input-source SHA-256: `2e828c3f295a20ef214aa50ddb2903b78e82bbc69fa14151b956884f4826fa68`. CLI SHA-256: `d928b8e338b48334981368630a93c0e903420ffd2832273cf2216004b2b83531`. The [environment record](../../outputs/cuda-fast-preview/2026-09-27-pro5000/qualification.json), [runtime/weight identities](../../outputs/cuda-fast-preview/2026-09-27-pro5000/identity.json) and [media tool identities](../../outputs/cuda-fast-preview/2026-09-27-pro5000/media-tools.json) provide reproducibility details.

| Complete golden gate | Outputs | GPU/test seconds |
| --- | ---: | ---: |
| [gate-001](../../outputs/cuda-fast-preview/2026-09-27-pro5000/gate-001/result.json) | 204/204 exact | 124.72 |
| [gate-002](../../outputs/cuda-fast-preview/2026-09-27-pro5000/gate-002/result.json) | 204/204 exact | 111.29 |
| [gate-003](../../outputs/cuda-fast-preview/2026-09-27-pro5000/gate-003/result.json) | 204/204 exact | 126.05 |
| [gate-final](../../outputs/cuda-fast-preview/2026-09-27-pro5000/gate-final/result.json) | 204/204 exact | 100.82 |

All gates used the unchanged recorded hashes and fixtures and stayed inside the 720-second test deadline. The campaign ran the exact CLI binary qualified by the first gate. Later replays reused those qualified binaries through the complete unchanged production collector. The eight CPU runner tests also passed.

## Total process wall time

Seconds below run from immediately before CLI launch through child exit and closed output files. They include model loading, conditioning, reference encoding, denoising, video/audio decoding, codecs and writes. Build, packing, transfers, subsequent validation and report assembly are excluded. Blocked and user-skipped slots have no successful-video render time.

| Variant | 640×480 text | 640×480 reference | 1344×768 text | 1344×768 reference |
| --- | ---: | ---: | ---: | ---: |
| P01 | 102.25 | 123.89 | 220.55 | 283.52 |
| P02 | 79.48 | 114.17 | 191.76 | 255.93 |
| P03 | 61.37 | 77.68 | 88.90 | 101.54 |
| P04 | 71.81 | 108.61 | 160.78 | 196.52 |
| P05 | 105.19 | 385.32 | 522.62 | S |
| P06 | 83.20 | R | M | R+M |
| P07 | 125.42 | R | M | R+M |
| P08 | 61.87 | R | 170.74 | R |
| P09 | 89.91 | R | 268.46 | R |
| P10 | 117.24 | R | M | R+M |

**S:** P05-1344-ref was stopped at the user’s request to skip slow SOL runs. The three already completed SOL videos remain available. The interrupted attempt consumed 80.720 seconds and was terminated with SIGTERM and reaped; it is excluded from successful-video timings. **R:** unchanged policy rejects reference layouts with adaptive cache or SubBlock, including zero-sparsity SubBlock. **M:** the fixed 512 MiB adaptive-cache budget rejects the high-resolution layout. Quantization and a longer warmup do not reduce this BF16 cache allocation. No guards, geometry or frame counts were relaxed.

The [preflight record](../../outputs/cuda-fast-preview/2026-09-27-pro5000/preflight.json) covers all forty slots. A conservative 1,024-token text-only layout requires 404,431,896 bytes at 640×480 and 1,249,410,072 bytes at 1344×768. The actual prompt has 58 text tokens; admitted adaptive runs use the complete actual layout. For rejected reference slots, the memory proof is a text-only lower bound, not admission of a complete reference layout. The policy rejection already prevents those renders.

Observed DiT sequence lengths were 11,572/37,768 for text-only and 22,460/48,656 with the max reference. Text and reference use FL2VA and Ref2VA respectively, so their timing difference is not just encoder overhead.

All successful renders kept the DiT core weights resident, either in BF16 or packed form. None exercised weight streaming. Sampled per-process peak VRAM ranged from 14.99 to 48.04 GiB; per-slot measurements are in the CSV/JSON.

## Decoder comparison

P01/P02 change only the final video decoder among requested flags. Their wall-time differences still contain loading and page-cache variation. These phase measurements make the decoder contribution explicit:

| Group | Full VAE load + decode | Preview VAE load + decode | Observed total reduction |
| --- | ---: | ---: | ---: |
| 640 text | 11.157 s | 1.383 s | 22.3% |
| 640 ref | 10.221 s | 1.347 s | 7.8% |
| 1344 text | 33.561 s | 4.902 s | 13.1% |
| 1344 ref | 32.125 s | 4.627 s | 9.7% |

The paired decoded audio is identical for all four P01/P02 groups. Sampled preview frames preserve the controls’ actions and framing while smoothing fine face, hair and fabric texture.

## Actual approximation work

P03 performed three full forwards and three reuse updates per six-step schedule: 150 Sage3 calls and 600 NVFP4 projections. Its low-step warning is preserved. P04 performed seven full forwards and five reuse updates: 350 Sage2++ calls and 1,400 FP8 projections. Each completed P05 video performed twenty full forwards and 4,000 NVFP4 projections. SOL counters, dense fallbacks and pair counts are retained per slot; its requested defaults were unchanged.

| Slot | Full 50-block forwards | Probe-only cache hits | Reuse skips | Dense calls | SubBlock calls | Retained sparse pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P06-640-text | 6 | 0 | 0 | 300 | 0 | — |
| P07-640-text | 20 | 0 | 0 | 1000 | 0 | — |
| P08-640-text | 6 | 0 | 0 | 300 | 0 | — |
| P08-1344-text | 6 | 0 | 0 | 300 | 0 | — |
| P09-640-text | 12 | 0 | 0 | 502 | 98 | 40.01% |
| P09-1344-text | 12 | 0 | 0 | 502 | 98 | 35.46% |
| P10-640-text | 20 | 0 | 0 | 216 | 784 | 36.27% |

Adaptive probes execute block 0 even on a cache hit; they are separated from full forwards above. The original raw ledger’s `fresh_forwards` field counts all evaluated steps, including those probes; the report’s `full_forwards` and `probe_only_forwards` are the unambiguous counts. Warmup and final refresh assertions passed. P10 refreshed at the SubBlock attention transition at zero-based step 4. P08 dispatched no sparse attention. P09 stayed dense through steps 0–9 and used sparse attention only in steps 10–11. Protected queries, block 0 and rounded budgets make retained density different from the requested sparsity.

P06, P07 and P10 had **zero adaptive cache hits**: every step executed all 50 blocks. These settings therefore demonstrate the warmup and transition policies, without an observed adaptive-cache speedup. P10 did execute sparse attention after its transition.

Every timed quantized run prepared **zero matrices**. Dispatches reconcile with actual executed blocks. Detailed per-step counts, timings, routing, transfers, cache decisions and residency are linked from each gallery card. Do not attribute a gain to adaptive caching when its hit count is zero, or to zero-sparsity SubBlock when no sparse work occurred.

P08’s lower 640-text wall time is not a sparse-attention gain: denoising took 25.754 seconds versus 25.664 seconds for P02, while text encoding, AdaLN preparation and core loading were faster. Both executed 300 dense attention calls. This illustrates the importance of preparation/page-cache variation in these observations.

## Preview usefulness and limits

The [media review](../../outputs/cuda-fast-preview/2026-09-27-pro5000/review.json) binds observations to media hashes. Review used seven distributed frames per video, including first and last, plus complete decode, motion measurements and browser playback checks. It is a sampled visual review, not user acceptance or numerical parity. Audio review checked full PCM decoding, stereo waveforms, amplitude and clipping; semantic listening was not performed.

- **P04 is a useful provisional starting point** for inspecting these fast previews: all four sampled sequences retain clear actions and readable detail at lower observed wall time than P02. Its 640-reference sample includes an extra background figure and an additional turn, so inspect that output before choosing it. The result combines steps, reuse, attention and precision; it does not isolate one flag’s gain.
- **P02 provides the dense preview control.** Its framing and actions closely follow P01, with substantially cheaper video decoding and smoother texture.
- **P03 is unsuitable for judging face, identity or clean motion here.** All four outputs have severe blur, ghosting or washout despite functional passes. They remain in the gallery, with no quality-driven reruns.
- **P05 produces coherent sampled motion but is expensive with these settings.** Its twenty evaluations and SOL execution are poor fits for a low-latency preview in the measured reference and large-resolution cases. It is not an isolated SOL-versus-dense benchmark.
- **P06–P10 are boundary and policy experiments.** Their actual hit/sparse counts and supported resolutions govern interpretation. They do not provide a single uniform recipe for both conditioning modes.

These are single observations in fixed order, without cache eviction, warmup videos, repeats or confidence intervals. Weight-loading variation is material. Other than the requested P01/P02 decoder contrast, rows change multiple factors. No high-step quality reference was generated.

## Preparation, failures and audit

| Packed-cache preparation | Seconds |
| --- | ---: |
| FL2VA-fp8 | 24.404 |
| FL2VA-nvfp4 | 16.424 |
| Ref2VA-fp8 | 53.144 |
| Ref2VA-nvfp4 | 32.577 |

The 800 current packed artifacts cover both transformer families and both precisions. Existing metadata-keyed files were reused through copy-on-write; required missing artifacts were prepared before timing. The [setup ledger](../../outputs/cuda-fast-preview/2026-09-27-pro5000/setup-results.json) records these costs separately. [Artifact transfer timings](../../outputs/cuda-fast-preview/2026-09-27-pro5000/transfer-timings.json) and each run’s validation duration are also outside render time. The [initial build timing record](../../outputs/cuda-fast-preview/2026-09-27-pro5000/initial-build-timing.json) gives launch-to-last-log-write estimates; these are not process-exit timings. Initial interactive staging/editing was not instrumented as a benchmark.

The optional Sage3 build initially emitted unsupported generic SM120 PTX. An invocation-only `SAGE_NVCCFLAGS=-gencode=arch=compute_120a,code=sm_120a` override fixed the isolated build. The existing `quant_prepare` utility rejected its placeholder core-byte count; an ignored campaign driver supplied the actual 50-block size to unchanged production packing functions. Neither workaround edited h3cli sources. Logs retain both setup failures.

P03-640-text completed successfully but initially failed the campaign trace parser: reused-step JSON followed carriage-return progress text instead of beginning a line. The parser was corrected, all 204 goldens replayed, and the same MP4 revalidated without rerendering. The [original failed validation](../../outputs/cuda-fast-preview/2026-09-27-pro5000/runs/P03-640-text/attempt-001/validation-initial-failed.json) and [orchestration revision record](../../outputs/cuda-fast-preview/2026-09-27-pro5000/orchestration-revisions.json) are retained. The qualification between P03 text and reference can affect page-cache timing. One unsuccessful wrapper launch before that replay is also retained in the driver logs.

All 26 videos fully decode to the requested dimensions, 124 frames, 24 FPS and 32 kHz stereo audio. All successful renders exited normally and were reaped inside the 7,200-second deadline. Resource monitoring sampled once per second, with gaps retained; the largest observed gap was 1.001 seconds. The [browser validation](../../outputs/cuda-fast-preview/2026-09-27-pro5000/browser-validation.json) confirms all 26 players, seeking, download paths and four group filters. The final audit verifies source/input identities, all forty statuses, media hashes, CSV/JSON agreement and report links.
