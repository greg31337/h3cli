# CUDA BF16 / SGLang qualification results

The native reference path passes the frozen content gates for all eleven
planned cases. The user **accepted the results, closed all 40 tasks and selected
SGLang-equivalent rendering as the ordinary BF16 CUDA default**. The original
joint performance/memory measurements do not all pass; their exceptions are
accepted and retained without changing the frozen thresholds.
The final measured tables and fast-regression status are in the
[complete report](../outputs/cuda-sglang/review/review.html) and
[machine-readable gate audit](../outputs/cuda-sglang/review/qualification.json).

Use the [selected full-length pairs](../outputs/cuda-sglang/review/final-review.html)
for visual inspection. Each pair includes every output frame, audio, synchronized
seeking, exact commands, metrics and amplified difference images. Earlier failures
and teacher-forced diagnostics remain in the complete report, with separate labels.

## Content

The primary matrix is 640×480 at 24 fps: 124 frames / 6 evaluations,
362 frames / 6 evaluations, and 124 frames / **50 evaluations**. SGLang's
corresponding sigma grids have 7, 7 and 51 points. The three held-out prompts
and first-frame, last-frame, both-frame, image-only and mixed image/video
conditioning cases use six evaluations.

All eleven complete free-running video/audio trajectories are bit-identical
to the pinned oracle. Every decoded video frame passes PSNR, SSIM, LPIPS,
temporal and worst-region gates; decoded audio passes its frozen bounds.
Primary teacher-forced and interrupted/resumed replays also pass, including
all 50 C2 evaluations. No oracle inputs are needed for ordinary native rendering.

The main corrections match the oracle's CPU RNG and schedules, BF16 rounding
boundaries, cuBLAS algorithms, dense FlashAttention, FP32 velocity heads,
range-safe full VAE, image/video preprocessing, conditioning geometry and
reference-audio encoder. The mixed-reference failure was localized to tiny
FP32 encoder errors before the audio patch projection; matching every encoder
operation restores an exact complete trajectory. Details and rejected candidates
are retained in the [numerical audit](cuda-sglang-audit.md).

## Timing and memory

The tracing-free primary campaign uses three interleaved fresh-process pairs
per case. Process exit and time to a verified playable video are measured
separately from generation and individual stages. Filesystem caches are not
forcibly cleared between runs.

Native generation includes lazy model loading, conditioning, denoising, full
decoding and delivery. SGLang loads CPU-offloaded weights before its reported
generation region. The report preserves both boundaries; native loading is
not subtracted to obtain a pass. Its expanded stage tables also show native
video-model loading and the complete AV decode/delivery subtotal.

All nine primary pairs completed and passed content checks. The following are
medians of three runs, in seconds; ranges and individual measurements remain in
the report. Each cell is **native / SGLang**.

| Case | Process wall | Playable output | Reported generation | Denoising | Full AV decode stages |
| --- | ---: | ---: | ---: | ---: | ---: |
| C0: 124 / 6 | 111.621 / 256.300 | 111.250 / 232.017 | 110.989 / 43.994 | 27.954 / 31.822 | 7.779 / 6.675 |
| C1: 362 / 6 | 228.706 / 372.744 | 228.483 / 341.631 | 228.047 / 133.868 | 114.579 / 112.166 | 22.433 / 16.379 |
| C2: 124 / 50 | 338.029 / 461.550 | 337.630 / 437.698 | 337.000 / 246.949 | 241.052 / 235.797 | 7.979 / 6.456 |

The native process finishes sooner, and all three denoising medians pass the
5% tolerance. Reported generation fails its gate with the different loading
boundaries retained above. The native AV decode-stage totals are 16.5%, 37.0%
and 23.6% slower; these already fail before adding its separately measured
video-model loading and delivery. No loading or delivery work is removed from
complete generation or process wall time.

| Case | Native sampled peak GiB | SGLang sampled peak GiB | Largest gap, native / SGLang |
| --- | ---: | ---: | ---: |
| C0 | 6.217 | 16.434 | 0.066 / 9.162 s |
| C1 | 10.748 | 19.387 | 0.462 / 9.284 s |
| C2 | 6.303 | 16.434 | 0.087 / 9.156 s |

Native sampled VRAM is lower, but SGLang's NVML queries sometimes block for
approximately nine seconds during startup/cleanup; one native C1 recording
also has a 462 ms gap. These exceed the required 100 ms sampling interval.
Missing samples are not interpolated, and a lower sampled peak alone does not
qualify memory. Source data: [all nine paired measurements](../outputs/cuda-sglang/review/matched-campaign-v13/result.json).

C2's instrumented 50-evaluation profile shows stable allocations, approximately
38.57 GB of bounded pinned staging, approximately 39.55 GB process residency and
zero process swap. Lifetime tests cover shape changes, cancellation, overflow,
admission failures, streaming sinks and clean recovery. The existing 110 GB
process cap is unchanged.

## Build and comparison scope

The final arithmetic-4 executable and matching encoder worker are retained at
`/path/to/qualification/final-reference-v4`. See the
[build/runtime command](cuda-sglang-reference.md). Native rendering uses C/CUDA;
Python is used for the oracle and tests. Reference startup uses model-file
metadata identities to avoid weight hashing. Tests do not duplicate model files.

Primary timings retain their frozen arithmetic-2 source/binary. Arithmetic 4
adds the soundtrack corrections used by R1 and rejects identity
1–3 caches/checkpoints. The primary computation is unchanged; both identities
and the source delta remain visible. Fast before/after testing uses the final
arithmetic-4 executable directly against the pre-change baseline, with the
protected fast arithmetic identity still 1.

Hardware qualification is limited to the measured RTX PRO 5000 72GB. This
corpus does not establish 362-frame/50-evaluation parity, other geometries,
other schedules, or equivalence for h3-specific continuation policies.
All tasks are closed in [the archived task list](cuda-sglang-parity-tasks.md) by explicit user acceptance;
the remaining failed measurements are preserved in the report.

## Accepted fast-mode differences

The final build's fast BF16/full-VAE and NVFP4/preview media differ from the
pre-change executable. BF16 repeats are deterministic within each build, and
the difference affects decoded video and audio, not just MP4 metadata. The
source audit and focused fixed-tuning fixture do not establish unchanged full
rendering behavior.

The user explicitly instructed: **“ignore if the behavior of fast mode has
changed.”** This exception is retained separately from the frozen contract in
the [waiver record](../outputs/cuda-sglang/review/qualification-waivers.json).
The measured differences remain visible in the
[fast comparison report](../outputs/cuda-sglang/review/fast-regression.html).
No fast tuning defaults or algorithms were changed in response to the failure.

The first NVFP4 preview pair failed before rendering because the isolated test
directories lacked `models/taeh3.safetensors`. The existing 22,709,752-byte
preview decoder was copied once and shared by both test builds. The transfer
added no weight hashing; fast preview's existing tiny-model digest is preserved.
Those failed attempts are retained; only that missing-file
pair was repeated, preserving every other measured result.


The repaired campaign retains three uninstrumented pairs plus a separate
CPU-sampler diagnostic pair for each fast variant. Medians are **candidate /
pre-change baseline**; denoising is the sum of six CLI step durations (0.01 s
resolution), and memory is sampled total device use.

| Fast variant | Process wall s | Denoising s | Sampled peak GiB |
| --- | ---: | ---: | ---: |
| BF16 / full VAE | 111.546 / 111.645 | 24.17 / 24.19 | 39.465 / 39.465 |
| NVFP4 / preview VAE | 75.253 / 75.304 | 13.82 / 13.82 | 14.541 / 14.541 |

Timing and sampled-memory medians pass the 5% comparison band. The full BF16
sampler diagnostic differs, while NVFP4's CPU-sampler arrays match; final media
is not unchanged. No further investigation is required under the user's waiver.
The final mixed-context fixed-tuning fixture matches its immutable baseline.
The final no-cuDNN build succeeds and rejects reference mode before model load.
Relevant host and local Metal compatibility checks also pass.

The final source/test snapshot is `records/reference-source-v26.tar.gz`, with
its file manifest and archive checksum beside it. All 748 retained files were
verified after synchronization to the server directory; the
arithmetic-4 executable and encoder worker remain byte-for-byte the tested build.
