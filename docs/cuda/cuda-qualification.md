# CUDA qualification record

This records the initial RTX 4090/SM89 integration. Subsequent SM120 results,
full 50-block residency, and Blackwell tuning are recorded in
[RTX PRO 6000 qualification](cuda-pro6000-qualification.md).
Later streaming-card results are in the [RTX 5090](cuda-5090-qualification.md)
and [RTX 3090](cuda-3090-qualification.md) records.

Qualification uses the original BF16 MiniMax-H3 models and the shared model,
conditioning, continuation, sampler and container implementations. The frozen
Metal baseline is commit `cb22cc590bfffabca0ba727e0f182ec0f07f9ada`.
Logs and generated artifacts are under `outputs/cuda-validation`; server
artifacts are under `/path/to/h3.c/outputs/cuda-validation`.

## Environment and scope

| Environment | Recorded configuration |
| --- | --- |
| Metal reference | Apple M4 Max, macOS, original Metal/MPS backend |
| CUDA qualification | NVIDIA RTX 4090, SM89, 24,564 MiB reported VRAM |
| Linux | Ubuntu 24.04 x86-64, CUDA toolkit 12.8.93, driver 580.159.03 |
| Host memory | 251 GiB physical; container limit **61,999,996,928 bytes** |
| Development storage | Source, models, dependencies, caches, temporary files and results under `/workspace` |
| Model execution | BF16 weights; F32 accumulation; no TF32, FP8, INT8, or multi-GPU substitution |

SSH access was verified before source changes. The environment probe records
the container limit separately from physical RAM because checkpoint-page cache
capacity materially affects streaming performance on this node.

The fat object passes all 41 primitive comparisons on the 4090. At this initial
stage, SM86, SM90, and SM120 were compiled but unavailable for hardware testing,
and architecture-specific tuning was deferred. Reduced-block residency was
tested on the 4090; the later PRO 6000 run covers full 50-block residency.

## Completed checks

* GPU API contract: all 126 declarations linked on both backends. The final
  Linux and macOS `make test` targets pass. Managed, device-only, and mapped
  pinned tensors also pass actual GPU arithmetic and device-copy checks; both
  bounce-buffer and registered-file streaming modes pass the slot audit.
* Host suite: sampler/container validation, 122 adversarial container cases,
  41 CLI cases, 1,270 continuation oracle rows, layout/posterior/memory/progress.
* GPU suite: BF16 rounding, 129 sampler checks, 22,656,421 bridge checks,
  AudioVAE primitives, repeated causal GQA scaling tests and 653,390 preview
  checks. Protected prefixes and 45 twenty-step CPU/GPU trajectories match.
* Streaming race regression: 64 distinct payload generations alternate across
  two asynchronous slots before the final CUDA submission; all payloads match.
  Managed, device-only, and pinned storage, range transfers, ready/release
  fences, and optional registered-file uploads are covered.
* Tokenization: 35 Metal corpus cases for each of FL2VA and Ref2VA match exactly.
* Metal primitive fixtures: 41 records, including long BF16 attention in both
  layouts. The original 39 records remain byte-identical after adding the two
  long-sequence cases.
* Memory policy: a real 25-block denoiser has identical F32 output in resident,
  streamed, and automatic allocation-failure fallback modes. Resident tensor
  peak is 19,291,873,560 bytes; streamed/fallback peak is 1,597,895,680 bytes.
  The fallback test injects a 3 GiB allocation budget after resident planning.
* Cache-pressure qualification: every latent and decoded RGB byte matches
  on a cached rerun; a reservation leaving about 4 GiB free triggers conditioning
  and decode cache eviction, and generation completes. After releasing the
  reservation, the original prompt and fresh-process checkpoint restart match
  the original F32 boundaries exactly. Record/pressure/pause takes 1,006.51
  seconds; fresh restart takes 60.60 seconds.
* CUDA denoiser reuse: six-step uninterrupted/cached/restarted runs match
  every F32 boundary exactly with reuse interval two and a stop after step
  three (387.98 seconds to record, 54.32 seconds to restart).
* CUDA core reuse and token reduction: four-step uninterrupted/cached/restarted
  runs match all F32 boundaries exactly after stopping at step three. Core reuse
  interval two takes 380.40/48.35 seconds to record/restart; token reduction
  takes 386.87/48.21 seconds.
* Same-device CUDA checkpoint audit: every F32 latent at every completed
  boundary matches an uninterrupted run, including cached reruns and a new
  process. Two-way Metal/CUDA handoffs restore the first boundary exactly.
* CUDA CLI generation: T2VA, first-frame, last-frame and combined first/last-frame
  modes pass with all 50 transformer blocks, two steps, 128×128 output and
  22 frames. All four MP4s have the expected geometry and 32-kHz stereo audio;
  AV-state serialization also completes. These are functional checks rather
  than full-quality two-step rendering claims.
* Multi-image CUDA generation and image-plus-separate-audio generation pass
  at 128×128/22 frames/two steps (549.36 and 447.46 seconds, respectively).
  The corrected embedded-audio, silent-video, replacement-audio and separate-audio
  fixtures also pass on Metal (36.99, 34.48, 35.65 and 29.54 seconds).
* All nine CUDA reference-mode cases pass. The three video cases use 56-frame
  targets; embedded-audio and silent-video cases take 917.76/917.83 seconds
  with two workers, and replacement audio takes 382.46 seconds alone. Logs
  confirm the released-v1 pipeline, and every MP4 passes frame/audio checks.
  `references-summary.json` additionally verifies every AV checksum, finite
  latents, and the same numerical source hash across all nine cases.
* The first two CUDA continuation segments pass with 20 transitions, 256×256
  output, 141 raw frames, and seeds 72/73. Every callback validates the exact
  initial state and protected prefix; the second delivers 102 new frames and
  136,000 stereo samples after trimming 39 frames. Wall times are 568.55 and
  539.16 seconds. Independent Metal/CUDA chain drift compounds: video-latent
  relative L2 is 0.407467 for segment one and 0.676157 for segment two. These
  are separate trajectories, not identical-prefix checkpoint handoffs.
* The third CUDA segment also passes (536.50 seconds), with seed 74 and the
  same two image references. The first/middle/last-frame review shows a
  continuous scene with changing pose and no exact replay across segments.
  Every requested seed is independently checked against the initial noise;
  `chain-review.json`, `.png`, and `.html` retain the temporal measurements
  and review. This is evidence against the prior exact same-seed replay,
  not a guarantee that the model never repeats a semantic action.
* The fourth CUDA continuation segment changes both image references and uses
  seed 75. It passes all 21 boundary callbacks and frame/audio checks in 563.94
  seconds; its inherited prefix remains protected throughout. The complete
  four-segment chain uses original BF16 weights, 50 blocks, reuse/core-reuse one,
  and no token reduction.
* CUDA→Metal native AV continuation passes from a CUDA-generated 56-frame
  video state to a 141-frame Metal target with changed image references and a
  new seed. The inherited 39-frame prefix remains exact at every checked
  boundary; the delivered result has 102 frames and 136,000 stereo samples.
  The current-source run takes 36.43 seconds and retains its callback audit.
  Correcting the driver's GPU-request metadata leaves the AV state byte-identical.

The initial two-step cross-backend handoff measured relative L2 0.028369
(CUDA→Metal) and 0.036477 (Metal→CUDA) after the next computed transition.
Those earlier checkpoints retain matching archived executables because later
source changes correctly invalidate their build identity. Cross-backend
numerical equivalence does not imply bit-identical new computation.

A current-source CUDA→Metal handoff with six steps and reuse interval two
also passes. The stored boundary at step three and the reused transition at
step four are exact; subsequent computed transitions have relative L2
0.00533546 and 0.04022534, with final maximum absolute error 0.24028362.
This additionally checks portability of the saved velocity history.

The full CUDA tier passes: the 864×480 acceptance render, all four twenty-step
continuation segments, and the twenty-step same-device restart. Recording and
pausing the latter takes 484.89 seconds; restarting takes 169.18 seconds, with
every F32 boundary exact. A CUDA→Metal handoff of that checkpoint also passes
all seventeen remaining transitions: the restored step-three boundary is exact,
final relative L2 is 0.04387003, maximum absolute error 0.28294683, and cosine
0.99903741. Its complete measurements are in `cuda-to-metal-final.log`.
For this 128×128 controlled handoff, separate final video/audio latent relative
L2 values are 0.04485684/0.00926461, decoded RGB mean error is 4.09/255, and
PCM relative L2 is 0.01971738. `cuda-to-metal-final-media-comparison.json`
records the decoded-media comparison. This smaller, identical-conditioning
workload is distinct from the independent full-resolution comparison.

Native Metal→CUDA AV continuation also passes in 498.03 seconds, with a
39-frame inherited prefix, two image references, and a new seed. Every initial
and protected-prefix comparison passes; the delivered result has 102 frames
and 136,000 stereo samples. Its AV state is the explicit-GPU control for the
final default-selection regression.

The reverse twenty-step Metal→CUDA handoff also passes (354.95 seconds).
It restores step three exactly and retains the stored CPU sampler mode; all
seventeen computed boundaries stay within the original gate. Final combined
relative L2 is 0.06335567, maximum absolute error 0.71190931, and cosine
0.99799270. Together the two directions check both saved CPU and GPU sampler
modes, independently of each target backend's default.

The final sampler-default correction keeps CUDA continuation state on the GPU
without requiring an environment override, preserves the existing Metal default,
and reports the actual DiT backend. The Mac host/GPU, 41 primitive, 70 tokenizer,
and storage/fence suites pass afterward. Its frozen/current Metal comparison
again produces eight byte-identical MP4/AV files. A matching final-source
checkpoint pair passes in 27.93/9.55 seconds (record/restart).

Long numerical qualification uses source hash
`37da7e300aa7c3a02fe72490561b4cd3c7f53735db5e0efd62b70c9c982bc09f`;
its source and matching Metal executables are archived under `qualified-37da`,
and the earlier CUDA executables under `qualified-37da-cuda` on the server.
The final sampler-default/reporting correction has source hash
`827e91148b9545dd71bc3659dc96412159fb80e14c63187b7ed4e8287d0e8584`.
No kernel arithmetic changed between these builds.

The final source passes the full Linux regression, primitive, tokenizer and
storage/fence suites. All 41 primitive fixtures also pass with the final fat
binary on SM89. The main server checkout at `/path/to/h3.c` is rebuilt with
that source and automatic architecture detection. Final-stage records are
mirrored locally under `outputs/cuda-validation/final-source`; the isolated
server validation directory is `/path/to/h3.c-final`.

The final ordinary CUDA AV output is byte-identical to the earlier two-step
control (SHA-256
`980c58d005fe4bf3b107e170fb9b77ab666f212ecdc183851f5e9d3611ba2151`).
Record/restart takes 607.33/58.84 seconds. Default CUDA continuation also equals
the earlier explicit-GPU control byte-for-byte; its uninterrupted and restarted
AV files match exactly (421.62/59.32 seconds). The checkpoint stores the
protected 12 video steps and 65 audio ticks.

Final-source two-step checkpoint handoffs retain the original numerical gate:

| Handoff | Relative L2 | Maximum absolute error | Seconds |
| --- | ---: | ---: | ---: |
| CUDA→Metal, ordinary | 0.03468087 | 0.29567309 | 9.76 |
| Metal→CUDA, ordinary | 0.03647747 | 0.23798077 | 46.03 |
| CUDA→Metal, continuation | 0.02389965 | 0.78177571 | 26.36 |
| Metal→CUDA, continuation | 0.01682458 | 0.23076923 | 58.64 |

Every restored boundary is exact. Both continuation directions also preserve
every protected F32 video/audio bit after the remaining transition. The real
bridge case passes in 390.65 seconds with the default GPU sampler, two image
references, seed 77, 141 raw/102 delivered frames, and exact mask invariants at
all three boundaries. These final checks validate the default-selection change
without relabeling the earlier full-render build provenance.

## Released-model numerical comparisons

The component suite compares against independently recorded Metal tensors.
Bounds are declared in the tests; real vision and full DiT use the repository's
existing scale-aware acceptance bounds. Selected measurements are below;
per-component JSONL files preserve all outputs, and the complete component
suite passes with the optimized attention path enabled.

| Component/output | Relative L2 | Maximum absolute error | Cosine |
| --- | ---: | ---: | ---: |
| Qwen text conditioning | 0.0004265 | 1 | 0.999999909 |
| Multimodal text conditioning | 0.0004356 | 1 | 0.999999905 |
| Image vision merged features | 0.0197180 | 0.203125 | 0.999809557 |
| Video vision merged features | 0.0280753 | 0.4375 | 0.999644636 |
| Video encoder | 0.00000782 | 0.00019974 | ≈1 |
| Audio encoder | 0.00000109 | 0.00000180 | ≈1 |
| Audio decoder | 0.00000199 | 0.00000067 | ≈1 |
| Video decoder | 0.00000068 | 0.00000179 | ≈1 |
| One transformer block, final | 0.0024410 | 64 | 0.999997021 |
| Full 50-block DiT, video velocity | 0.1004383 | 0.3203125 | 0.995981345 |
| Full 50-block DiT, audio velocity | 0.0225512 | 0.0859375 | 0.999748568 |

The block's large absolute values make its scale-aware error more informative
than absolute error alone. The worst vision deep-feature relative L2 is
0.0507358. The default DiT fixture uses identical real text conditioning and
deterministic unit-scale latent noise. A separate artificial, low-amplitude
`dit-stress` diagnostic fails its stricter 5% relative-L2 threshold; its original
goldens and gate are retained. It is not counted as a passing production test.

Independent two-step CLI comparisons are finite and have matching geometry.
These measurements include conditioning and generation differences; they are
diagnostic rather than evidence of full-quality equivalence. Detailed
`*-reference-backend-comparison.json` records retain all metrics and signal
statistics. RGB measurements use all frames downscaled to 160×90.

| Reference workload | Video latent L2 | Audio latent L2 | RGB mean error /255 | PCM RMS error | PCM relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Image + separate WAV | 0.163974 | 0.012960 | 15.543 | 0.001131 | 0.015861 |
| Video + embedded audio | 0.147685 | 0.066497 | 11.748 | 0.006492 | 0.145919 |
| Silent video | 0.139567 | 0.088640 | 11.499 | 0.001455 | 0.565611 |
| Video + replacement WAV | 0.117556 | 0.196033 | 9.134 | 0.007482 | 0.255390 |

The silent-reference case generates very quiet audio, making its relative PCM
error large compared with its small absolute RMS difference. No perceptual
quality gate is inferred from these two-step functional renders.

Independent twenty-step continuation chains diverge further because each
backend inherits its own preceding output. All four segments pass exact prefix
invariants against their own source; that does not make their independently
generated scenes equal. The temporal review remains coherent, including the
change of reference identity in segment four.

| Segment | Video latent L2 | Audio latent L2 | RGB mean error /255 |
| --- | ---: | ---: | ---: |
| 1 | 0.407467 | 0.148365 | 17.032 |
| 2 | 0.676157 | 0.413130 | 40.084 |
| 3 | 0.753601 | 0.549382 | 49.768 |
| 4 | 0.750813 | 0.737335 | 37.504 |

## Performance investigation

The initial direct Conv3D implementation took approximately 50 seconds per
reference image at 864×480. Bounded im2col packing (at most 32 MiB) plus F32
cuBLASLt reduced the same encoder stage to 1.4–1.5 seconds. Warmed isolated
large-convolution tests improved 10.8× and 35.4× with relative L2 at most
0.00000110; small convolutions retain their direct path. An additional odd-filter test
forces repeated 605-row chunks with changing output alignment and also passes
(relative L2 0.00000215), exercising the GEMM algorithm cache across those offsets.

The first full-size run also exposed scalar attention as a hotspot. A tiled
BF16 implementation reuses each key/value tile across 16 queries while keeping
softmax and value accumulation in F32. Production-sized comparisons measure
relative L2 around 0.0000514 and preserve the scalar implementation behind
`H3_CUDA_REFERENCE=1`. Warmed isolated speedups are 2.34× at 2,048 tokens and
2.71× at 18,225 tokens (four heads, width 128). At 18,225 tokens, reference and
tiled calls take 0.244723 and 0.090401 seconds, respectively. These isolated
timings are separate from concurrent diagnostic runs. All seven released-model
component groups pass with the optimized path enabled.

Qwen text conditioning read 46.9 GiB in roughly 118 seconds on this storage.
CUDA transfer-event timing was about 2.2 seconds, or 21 GiB/s; GEMM and attention
together took under one second. Storage latency and available host page cache
therefore dominate that phase. These measurements should not be generalized to
a node with local NVMe or a larger memory limit.

The direct-convolution and scalar-attention full-size profiling runs were
intentionally interrupted after identifying their hotspots. Their logs and
executables are preserved with `direct-conv-` and `scalar-attention-` prefixes;
they are not completed qualification results.

The completed CUDA acceptance render takes 3,313.85 seconds (55.23 minutes),
versus 2,169.10 seconds for the Metal reference on its different host/storage.
CUDA denoising takes 2,980.14 seconds and streams 718.491 GiB of core weights;
the shared loader records 1,394.86 seconds of host wait. CUDA events record
1,421.96 seconds in attention, 88.48 seconds in GEMMs, and 35.25 seconds in H2D
transfers at 21.12 GiB/s. Loader/upload wall times overlap GPU work and must not
be summed as exclusive categories. DiT tensor peak is 5.455 GiB. Video decoding
takes 139.71 seconds with a 9.365-GiB tensor peak. A runtime snapshot records
63°C, a 450-W power limit, 2,730-MHz SM clock and PCIe 4.0 ×16.

The 864×480/141-frame/20-step CUDA MP4 and AV state pass geometry, checksums,
finite-latent and stereo-audio checks. The sampled first/middle/last frames are
coherent and broadly resemble the Metal control, without visible tile corruption.
They are not numerically equivalent trajectories: final video-latent relative L2
is 0.403669 (cosine 0.918026), and audio-latent relative L2 is 0.309433 (cosine
0.951511). Downscaled RGB mean error is 21.67/255; decoded PCM relative L2 is
0.567661. These differences include separately computed conditioning and all
20 transitions. `full-backend-comparison.json` records all metrics, and
`full-backend-review.png` shows Metal above CUDA at frames 0, 70 and 140.
This establishes working rendering, not bit-identical output or a general
perceptual-quality equivalence claim.

The fixed single-image `cuda-bench` workload also passes: 864×480, 141 frames,
20 steps, seed 72, `inputs/2.jpg`, 50 blocks, reuse/core-reuse one, and token
reduction off. Total process wall time is 2,078.92 seconds (34.65 minutes),
including decoding and muxing. Its DiT phase takes 1,755.55 seconds with a
5.301-GiB tensor peak, 718.491 GiB of streamed core weights, 1,255.28 seconds
of attention, and 84.93 seconds of GEMMs. The shared loader records 346.67
seconds of host wait; H2D events total 38.02 seconds at 19.58 GiB/s. The
runtime snapshot shows 70°C, 2,730 MHz and PCIe 4.0 ×16. This run has one
reference and follows prior model workloads; its timing is not an isolated
speedup comparison against the earlier two-reference render. `bench.json`,
`bench.ffprobe.json`, `bench.log` and `bench-runtime-snapshot.txt` retain the
workload, environment and profile.

## Final acceptance and limitations

The Metal 864×480, 141-frame, 20-step Ref2VA render completed in 2,169.10
seconds with valid video and 32-kHz stereo audio. Sampled first/middle/last
frames are visually coherent. The final current-source Mac `make test`,
41-record primitive comparison and 70-case tokenizer comparison pass. All four
20-step, 256×256 continuation segments pass; the latter three deliver 102 frames
after the 39-frame overlap, with different seeds and changed image references
on the fourth segment. The 20-step same-device Metal restart also passes every
completed-boundary comparison (44.80 seconds to record, 22.83 seconds to restart).

The full CUDA render, four-segment chain, twenty-step restart, fixed benchmark,
and both full-schedule checkpoint handoff directions pass.
Final-source CUDA default-sampler, masked restart, bidirectional handoff, and
bridge checks all pass. Final-source Metal regression results are complete: the
frozen/current
comparison is byte-identical across all eight MP4/AV files, ordinary checkpoint
restart passes, and masked continuation pause/restart passes at every boundary
(63.81/27.35 seconds with a 39-frame prefix and 102 delivered frames).

The legacy `make parity` target cannot complete because this checkout lacks
the external `misc/fixtures` directory. It fails identically on the frozen
pre-CUDA baseline and on the current Metal build. Checked-in Metal-generated
fixtures provide independent coverage; the absent legacy suite is not reported
as passing.

Initial reference test attempts used a 1.5-second clip, omitted the image required
for a separate audio reference, or capped video decoding at the 22-frame smoke
length. The existing validation correctly rejected these inputs. The harness
now uses 2.5-second sources, 56-frame video targets, and an image alongside the
separate WAV file. Earlier rejected attempts are retained under
`short-reference-attempt` and `short-target-attempt`; they are not backend failures
or successful qualification results.

The initial available-hardware integration completed 114 tasks. Its remaining
hardware qualifications and conditional architecture tuning are tracked in
[todo.md](todo.md), including the subsequent PRO 6000 work.
This does not erase the missing legacy fixtures, the separate failing strict
`dit-stress` diagnostic, or the measured independent-trajectory drift above.
