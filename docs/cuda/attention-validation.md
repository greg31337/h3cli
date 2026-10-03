# RTX 5090 attention validation

Implementation and delivery are complete. The user stopped the remaining
tests and requested closing tasks only. Both native Sage modes are opt-in. The corrected build has passed component correctness, intermediate
packing, memory-safety and performance checks. CPU-state resume, twelve short
GPU-state pairs and eight target-resolution resume pairs also pass. Calibration,
held-out numerical evaluation, target benchmarks and short continuation checks
are complete. Remaining target continuation checks were skipped at the user's
request; unfinished profiling, decoder and mixed-schedule checks are also
skipped. No production recommendation
or perceptual-equivalence claim is made from component results.

## Environment and reproducibility

The existing RTX 5090 node reports 32,607 MiB of device memory, SM120,
driver 595.91.07 and CUDA toolkit 12.8.93. The isolated upstream oracle uses
Torch 2.8.0+cu128 and Triton 3.4.0. Native inference has no Python, Torch or
Triton dependency. Optional fat CUDA builds with and without cuDNN pass CLI
capability tests. Local Metal host/CLI compatibility checks also pass.

SageAttention is pinned at `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`;
CUTLASS at `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`. The native K-mean
implementation preserves the pinned Torch reduction order on this 5090.
Imported source hashes, licenses and arithmetic provenance are recorded in
[the dependency inventory](../../third_party/sageattention/README.md).

Current executable SHA256 values:

- Model: `701969987f285bd489726b1e67333a8a9443497085eb87121b0e10e8aa4c5102`.
- Component: `c3d66d6c3535148d31c93fb65f87141605f925a7b6884a36979e19c838b99855`.

The isolated checkout is `/path/to/h3-attention/final-src`. Component
records are under `outputs/kmean-order`; fresh model records use the `v3`
suffix. Local evidence is under `outputs/attention-5090/outputs`. Each job
retains its command, binary hash, timing, memory and dispatch observations.
The asset manifest includes model shards, references and preview decoder.
Turbo adapter SHA256:
`5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3`.

Paired model runs fix `H3_FAST_CUDA_GEMM_TUNE=0`: independent timed selection
of projection algorithms changed conditioning in an earlier resume experiment.
This setting is part of the qualification contract. The model matrix uses
GPU-state Euler; separate tests cover CPU-state resume.

## Inclusive component measurements

Each operation includes smoothing, packing, correction, attention and output
conversion. There is one warmup and three synchronized measured samples,
with H=56, D=128, immutable BF16 Q/K/V and BF16-rounded H3 attention scale.

| Tokens | cuDNN | Native fallback | Sage2++ | Sage3 |
| ---: | ---: | ---: | ---: | ---: |
| 2,281 | 0.673 ms | 1.139 ms | 0.756 ms | 0.518 ms |
| 18,225 | 40.419 ms | 67.612 ms | 19.056 ms | 15.141 ms |
| 109,078 | 1.485910 s | 2.382582 s | 0.609853 s | 0.492179 s |
| 110,036 | 1.517809 s | 2.433793 s | 0.620043 s | 0.500390 s |
| 113,988 | 1.630621 s | 2.613594 s | 0.666291 s | 0.539923 s |

At 109,078 tokens, Sage2++ takes 41.04% of cuDNN time and Sage3 takes
33.12%. These are attention-operation ratios, not render speedups. Sage2++
is slower than cuDNN in the 2,281-token probe. Full sample distributions
are retained in the machine-readable report.

Eight additional measurements without profiling differ from matched
production-length component means by at most 0.60%. Full-render profiling
overhead was stopped at the user's request after one completed unprofiled
baseline and an interrupted second baseline. The full profiling-overhead
comparison is unqualified. Model denoising timing uses CLI phase
boundaries; the cumulative DiT wall counter also includes loading.

At 113,988 tokens, the harness tracks 7,073,400,832 device bytes with either
Sage mode, including four BF16 tensors and the admitted 512 MiB workspace.
NVML observes a process peak of 7,667,187,712 bytes. There are no concurrent
CUDA processes during component timing. Model weight residency and streaming
are recorded in the completed target matrix below. The 113,988-token probe
covers the larger reference-image case confirmed in the model logs.

Sage3 plan 1 uses full query spans where they fit and fixed query chunks at
larger shapes. Every query attends to all valid keys. Packing, reduction,
correction and conversion allocations fit the 512 MiB cap; no full
production-sized correction matrix is allocated.

## Correctness and retained regressions

The current build passes:

- 74 native functional cases, including tails, both layouts, guards,
  immutable inputs, zero/constant inputs and non-finite rejection.
- 360 oracle comparisons: 240 short, 56 long, eight retained-regression,
  48 reduction-boundary and eight larger Ref2VA production-shape cases.
  All finite upstream outputs match exactly. Independent blocked FP32 SDPA
  error is recorded separately from port parity.
- Four additional retained-regression checks confirm the short-case RAM
  scratch path preserves exact output. This validation-only I/O optimization
  leaves native binaries and arithmetic unchanged.
- All 17 intermediate tensors across both modes at S=129, and all 17 on
  the retained S=2996 real input, byte-for-byte against pinned upstream.
- Both representative compute-sanitizer cases with zero errors; allocation,
  alias, cancellation, changed-input and cleanup failure tests.
- Exact video/audio latent equality in twelve short GPU-state resume pairs
  covering FL2VA/Ref2VA Turbo, both Sage modes and projection off/FP8/NVFP4.
  Two CPU-state Ref2VA Turbo/NVFP4 pairs and eight target-resolution pairs
  across both families, both Sage modes and projection off/NVFP4 also pass.
  All 22 pairs have finite latents and byte-identical entire AV-state files.
  The selected backend's dispatch counts match the whole, paused and resumed
  step counts, with zero calls to the other Sage backend.
- Exact default-backend video/audio latents across the previous and corrected
  builds in ten retained matched calibration cases. The Sage fixes preserve
  the tested baseline outputs.
- Ten model-backed context requests covering policy alternation, exact repeat
  equality, token reduction, reuse, core reuse and cleanup.
- 1,552 sampler-state checks, presentation compatibility, 165 adversarial
  container cases and eight CLI tests (platform-specific skips recorded).

The port gate remains relative L2 at most 0.01 globally and in each query
domain. Two earlier all-block calibration failures are preserved as fixtures:

| Mode | Step / block / tokens | Failure and correction |
| --- | --- | --- |
| Sage2++ | 7 / 49 / 2,038 | Relative L2 0.014377. Corrected fused scale arithmetic, full division, FP8 reciprocal rounding and upstream fast-math settings. |
| Sage3 | 0 / 15 / 2,996 | Text-query relative L2 0.010340. Corrected K-mean summation order at two BF16 rounding boundaries. |

Both fixtures now produce exact upstream output in both layouts. The corrected
build also matches upstream on all 143 remaining real Ref2VA BF16 captures
from the interrupted calibration. Fresh calibration and held-out runs use
new output directories and the current executable throughout.

The pinned Sage3 kernel masks some padded keys by packed index. The native
port corrects the logical index mapping. Its independent upstream oracle
keeps the upstream binary unchanged and masks logical padded keys through
the score-correction input. The explicit H3 scale and tail adapter are
recorded per comparison. Pinned Sage2++ produces non-finites for all-zero V;
the native zero case is checked against the independent exact-zero contract.

## Model and workflow gates

The fixed corpus is [attention_corpus.json](../../tests/attention_corpus.json).
Calibration crosses FL2VA/Ref2VA, base/Turbo, projection off/FP8/NVFP4 and
two short shapes. Captures cover all 50 blocks at early, middle and late
steps for the designated seed. Validated raw diagnostic traces are removed
with user approval; measurements, checksums, media and regression fixtures
are retained.

Numerical limits are frozen from completed calibration before held-out evaluation:
1.25 times calibration maxima plus 1e-7 per mode, projection and query
domain. Final latent gates retain hard ceilings of relative L2 0.1 and
maximum absolute error 1.0. A failed gate leaves a mode unqualified.
The frozen file SHA256 is
`ce781b8af6037b8b8f17813d778be4bd5999b885cfe20aa9e0a6137a9ffc8e81`.
Its recorded calibration and probe hashes match the retained evidence.
Neither the procedure nor the hard ceilings were relaxed after calibration.
The limits were written at 06:58:06 UTC on 2026-09-20 and their unchanged
hash was verified at 07:28:08 UTC as held-out evaluation began.

All twelve calibration groups are complete across both model families,
base/Turbo and projection off/FP8/NVFP4: 72 renders, 3,600 full-core
comparisons and 1,800 captured Q/K/V inputs. All 7,200 mode/layout port
comparisons match upstream exactly. All 50,400 expected main-block attention
calls were observed, with no concurrent CUDA processes in the render records.

Both modes exceed a video latent-error ceiling in every calibration case:

| Mode / latent domain | Cases above either hard ceiling | Maximum relative L2 |
| --- | ---: | ---: |
| Sage2++ video | 24 / 24 | 0.622031 |
| Sage2++ audio | 6 / 24 | 0.235268 |
| Sage3 video | 24 / 24 | 0.705839 |
| Sage3 audio | 24 / 24 | 0.277073 |

These are calibration measurements. All twelve held-out groups are also
complete across FL2VA/Ref2VA, base/Turbo and all three projection settings:
108 renders, 3,600 core comparisons and 1,800 Q/K/V captures. All 7,200 port
comparisons are exact globally and in each query domain. All 75,600 expected
main-block attention calls were observed, with no concurrent CUDA processes
or telemetry errors. All 108 local clips match their recorded source hashes.

All 72 held-out candidate pairs fail at least one frozen final-latent gate:

| Mode / latent domain | Failed domain pairs | Maximum relative L2 | Maximum absolute error |
| --- | ---: | ---: | ---: |
| Sage2++ video | 36 / 36 | 0.390997 | 5.639960 |
| Sage2++ audio | 19 / 36 | 0.515987 | 1.587259 |
| Sage3 video | 36 / 36 | 0.588909 | 5.649175 |
| Sage3 audio | 36 / 36 | 0.663015 | 2.083075 |

Of 48,600 held-out attention/core query-domain measurements, 117 exceed their
frozen limits: 40 for Sage2++ and 77 for Sage3. These approximation failures
are separate from exact native-to-upstream parity. Human playback/listening
checks are skipped at the user's request; the implementation is accepted for
delivery and final HTML playback pages will be provided for optional inspection.
Both modes retain their opt-in status.
In the initial FL2VA Turbo/NVFP4 group, the
largest whole-attention relative L2 against independent FP32 SDPA occurs at
step 7, block 49: 0.142260 for
Sage2++ and 0.432776 for Sage3. Worst query-domain errors and their block/step
locations are retained in the machine-readable report. These approximation
errors are distinct from the passing native-port gate.

The target matrix uses 1344×768, 362 frames, Turbo strength 1.0, eight steps,
fast CUDA and preview VAE. Each FL2VA/Ref2VA and BF16/NVFP4 combination has
two alternating-order pairs across default, Sage2++ and Sage3. Separate
cache warmups precede measured renders. Records separate startup, denoising,
decoding, transfers, weight-cache behavior and observed peak memory.
Unmeasured cache warmups use 128×128/22 frames and two denoising steps. An
initial one-step warmup was rejected by CLI validation before GPU use; its
record is retained separately. The two warmup drivers were corrected to the
CLI minimum and the target queue resumed without changing native binaries,
frozen limits or completed calibration/held-out evidence.

All 24 target renders are complete: FL2VA Turbo at 109,078 main-DiT tokens
and Ref2VA Turbo at 113,988 tokens, with two alternating-order passes per
projection/attention setting:

| Model | Projections | Attention | Mean denoising | Mean process wall | Denoising fraction of default | Observed peak GPU memory |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| FL2VA | NVFP4 | Default | 673.531 s | 761.879 s | 1.0000 | 28.377 GiB |
| FL2VA | NVFP4 | Sage2++ | 297.183 s | 392.308 s | 0.4412 | 29.951 GiB |
| FL2VA | NVFP4 | Sage3 | 257.896 s | 345.470 s | 0.3829 | 28.885 GiB |
| FL2VA | BF16 | Default | 789.621 s | 837.795 s | 1.0000 | 23.068 GiB |
| FL2VA | BF16 | Sage2++ | 416.185 s | 464.121 s | 0.5271 | 23.576 GiB |
| FL2VA | BF16 | Sage3 | 367.963 s | 416.388 s | 0.4660 | 23.576 GiB |
| Ref2VA | NVFP4 | Default | 733.007 s | 830.616 s | 1.0000 | 29.641 GiB |
| Ref2VA | NVFP4 | Sage2++ | 320.855 s | 426.427 s | 0.4377 | 30.150 GiB |
| Ref2VA | NVFP4 | Sage3 | 279.270 s | 374.146 s | 0.3810 | 30.137 GiB |
| Ref2VA | BF16 | Default | 853.808 s | 911.762 s | 1.0000 | 24.035 GiB |
| Ref2VA | BF16 | Sage2++ | 445.063 s | 502.610 s | 0.5213 | 24.545 GiB |
| Ref2VA | BF16 | Sage3 | 393.645 s | 451.625 s | 0.4610 | 24.545 GiB |

Each render records exactly 400 main attention calls, 362 output frames and
the expected audio trimming contract, with no concurrent CUDA processes.
Saved AV-state SHA256 values match between repeats in each of the twelve
projection/attention settings. All 24 copied clips match their recorded source
hashes, and all telemetry records are error-free. The maximum observed process
memory is 30.150 GiB, within the node's 32,607 MiB device limit.
The denoising range divided by the mean is at most 0.103% across the two
passes (at most 0.079% for NVFP4 projections). FL2VA Sage2++/NVFP4 startup varies
by 16.35 seconds, accounting for most of its 4.29% process-wall range; its
denoising range is only 0.0017%. These target measurements include profiling;
the user skipped completion of the full-render profiling-overhead comparison.

All twelve NVFP4 runs reuse 200 packed projection artifacts and execute 1,600 native
NVFP4 projection calls, streaming 80.750 GiB of compressed weights during
denoising. Each BF16 run streams 287.109 GiB. Device weight-cache residency
and hits are zero in all 24 runs. This traffic is unchanged by the
attention policy within each projection setting. The legacy
H2D/source-upload timers omit these compute-stream compressed uploads even
though their byte counters include them. Do not infer upload bandwidth from
those incomplete timers; denoising and process-wall measurements include the
uploads. The separate CUDA weight-cache counters describe the BF16 device
cache, while quantization cache hits describe disk artifact reuse.

The eight target resume pairs use 24 jobs and observe all 6,400 expected main
attention calls. All sixteen completed renders satisfy their video/audio
trimming contracts; paused jobs save checkpoints without a delivered clip.
Their maximum observed device memory is 29.137 GiB, with no telemetry errors
or concurrent CUDA processes. The short continuation matrix completed all 21
renders and 18 exact protected-prefix checks across default, Sage2++ and Sage3,
including same-mode three-segment chains. All 8,400 expected main attention
calls were observed, telemetry is error-free, and all 21 local playback clips
match their recorded source hashes. Peak device memory is 13.730 GiB.

The user skipped the remaining target continuation checks after the Sage2++
initial clip completed. Its second segment was stopped and Sage3 target
continuation was not started. These target continuation cases remain
unqualified; the completed short matrix and target exact-resume results are
retained separately. Decoder-only preview/full-VAE replay, extended audio
cases and their additional playback groups were skipped before execution.
Presentation compatibility is covered by completed host checks; the skipped
decoder matrix provides no additional hardware qualification.
Human review is skipped at the user's request.
The final offline galleries provide paired video/audio and continuation
playback without requiring verdicts, notes or review-completion checks.

A separate plan-9001 diagnostic build and driver support two bounded mixed schedules
on calibration data only: Sage2++ at the first/last steps, optionally also
at the first/last four blocks, with Sage3 elsewhere. It cannot silently
change the public `sage3` policy or relax held-out limits. The user skipped
this experiment before execution. No mixed policy is released; pure Sage3
remains opt-in because its measured numerical quality gates failed.

## Delivered playback pages and closure

The final [paired playback gallery](../../outputs/attention-5090/outputs/review.html)
contains 44 groups: 36 held-out and eight target-setting groups, each comparing
default, Sage2++ and Sage3. The
[short continuation gallery](../../outputs/attention-5090/outputs/continuation-short-v3/review.html)
contains 21 clips. All 153 playback assets are retained locally with source
SHA256 checksums. Human-review controls and completion requirements are omitted.

At 18:21:06 UTC on 2026-09-20 the remaining tests were stopped on explicit
user instruction. The active baseline repeat records return code -15 from
cancellation; it is not a numerical or runtime failure. Its controller,
followup controller, driver and render process all exited. No remaining test
queue is running. Completed evidence and cancellation records are retained.
The first unprofiled baseline completed in 777.241 seconds; one sample does
not establish profiling overhead for the three attention modes.

See [the closed task list](../todo.md) and
[the machine-readable report](attention-validation.json) for passed, failed
and user-skipped checks. Skipped work is not counted as a validation pass.
