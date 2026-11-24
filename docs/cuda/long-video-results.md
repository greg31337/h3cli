# CUDA duration acceptance

Completed **2026-11-23: 24/24 tasks**, with **204/204 unchanged CUDA reference
outputs matching exactly**. Native CUDA accepts supported requests through 736
aligned frames; Metal retains its 362-frame ceiling. The implementation and
restrictions are in [the duration contract](long-video-contract.md), with
[capacity estimates](long-video-memory-estimates.md) and
[recorded measurements](long-video-metrics.json).

Follow-up: a reported 719-frame render exposed an additional audio convolution
workspace cap above the runtime-qualified duration. The
[audio workspace correction](long-video-audio-workspace.md) records the fix,
expanded descriptor-only checks and replacement Linux artifact. The identities
below describe the original milestone artifacts.

## Runtime coverage and limits

The bounded CUDA campaign contains 22 passing monitored executions. Every
qualified video, including references, codec probes, script workers and final
merges, is 256×256 at 24 fps and at most 464 frames / 19.333 seconds. Cases use
all 50 blocks, dense BF16, streamed transformer weights and two denoising
evaluations. The RTX PRO 5000 test device reports 73,415 MiB physical VRAM;
the guard retains the 60 GiB admission ceiling, physical-device reserve and
host-memory checks. Jobs run serially. The GPU returned to 22 MiB used after
the final regression.

| Workflow | Evidence |
| --- | --- |
| Native CUDA | 464-frame generation; saved AV/source capture; byte-identical full MP4 replay; preview replay. |
| Conditioning | Both first/last anchors; combined image/audio/silent-video references; separate reference-video soundtrack extraction with a long target. References are two seconds. |
| Checkpoints | Pause after one of two evaluations; resumed video/audio latents exactly match uninterrupted generation. Legacy readers reject required temporal section 15 v2. |
| Continuation | 447-frame source, 464-frame target, 447-frame context in hard and bridge modes. Protected prefixes compare exactly; 17-frame suffix replay is byte-identical; each final merge has 464 frames. |
| Reused API context | Repeated seeded video/audio latents and decoded PCM match; PCM is finite. RSS changes by 139,264 bytes and free VRAM is unchanged across repeats. |
| Recovery | Injected host-memory denial, device-allocation failure, full/preview cancellation, encoder failure and output-sentinel preservation; subsequent decoding succeeds. |
| Render scripts | Real long worker and two seed sets of two 22-frame shots; each short merge has 44 frames. Recursive media inspection includes 256×256 preflight probes. Host contracts preserve global/local references, comma lists, image-size inheritance and seed offsets. |
| HTTP | Real 464-frame native video, 256×256 still, and early rejection of 737 frames. |
| Final Linux standalone | Offline 464-frame generation, byte-identical full replay, preview replay and saved-source host planning. |
| Metal/native and final macOS standalone | Offline 256×256/22-frame rendering, byte-identical full replay and preview replay. Eighteen early-rejection checks cover native requests, actual CUDA long AV/checkpoint/source files and K=0/2/3/4. Metal HTTP rejects a long target with models absent. |

All 14 captured AV states in the qualified CUDA cases contain finite video and
audio values. Media checks verify dimensions, decoded frame counts, 24 fps,
timestamps and audio/video duration agreement. Full 464-frame output lasts
19.333333 seconds; its generated audio lasts 19.325 seconds, within the existing
delivery tolerance. Full VAE delivery retains bounded 22-frame chunks and
five-frame overlap.

**736-frame execution, larger-canvas long execution and learned long-upscale
execution remain unverified at runtime.** Their coverage is host planning,
metadata and logical-index checks without complete long-movie allocations.
No learned 2× runtime job was run, because its output would exceed the required
256×256 canvas. Motion and scene consistency are outside acceptance.

## Resource measurements

These are sampled whole-device VRAM and descendant RSS peaks, separate from
conservative admission budgets. Device sampling can miss short peaks; RSS sums
resident descendant pages and may count shared pages more than once. Tracked
tensor peaks exclude driver/library allocations. Per-phase times are recorded
in the [measurement file](long-video-metrics.json); nested phases must not be
summed as independent wall time.

| Case | Wall seconds | Whole-device MiB | Tracked tensor GiB | Descendant RSS GiB |
| --- | ---: | ---: | ---: | ---: |
| Native 464 frames | 82.74 | 5,568 | 4.998 | 36.780 |
| Full saved-state replay | 7.66 | 5,568 | 4.998 | 0.981 |
| Preview replay | 3.89 | 962 | 0.578 | 0.967 |
| First/last anchors | 78.18 | 5,568 | 4.998 | 36.874 |
| Mixed references | 120.67 | 6,868 | 6.273 | 36.958 |
| Reference-video soundtrack | 119.60 | 6,866 | 6.273 | 36.919 |
| Hard continuation | 74.10 | 5,568 | 4.998 | 36.856 |
| Bridge continuation | 76.79 | 5,568 | 4.998 | 37.339 |
| Long script worker | 67.09 | 5,568 | 4.998 | 36.917 |
| HTTP video/still/rejection case | 114.13 | 9,912 | Not emitted | 36.944 |
| Final Linux standalone native | 107.60 | 5,568 | 4.998 | 36.765 |

The 256×256/736-frame host estimate is 50 GiB resident or 12 GiB streamed
capacity under the documented assumptions. Near-30-second 1344×768 and
1920×1088 budgets exceed the 72 GB card even with streamed transformer weights.
The 736-frame 3840×2176 upscale also exceeds the one-million-row sequence
ceiling and is rejected independently of memory.

## Host checks and immutable regression

Bounded macOS/Linux checks cover backend limits, alignment, nonfinite seconds,
huge integer conversions, 702-frame continuation context, anchors, audio rows,
state metadata and maximum-canvas allocation/index products. Existing checks
include 103,238 continuation assertions, 15,945,992 bridge assertions, 4,328
sampler assertions, 534 HTTP contract checks, 41 script tests, four guard tests,
20 sampler-file tests covering 250 adversarial containers, and 18 regression
runner tests. The long-source host plan accepts a 512×512 2× target and K=2/3/4
schedules without executing the upscaler. The maximum CUDA convolution count
probe covers 3,626,434,560 elements and byte sizes above 32 bits without
allocating those tensors.

The final unchanged reference gate completed in **119.47 seconds**, matching
all **204** recorded outputs. It used verified native binaries from the final
source; standalone artifact execution was validated separately as described
above. The historical 640×480/124-frame/six-evaluation fixture is the sole
explicitly authorized exception to this milestone's runtime canvas restriction.
No fixture, golden hash or required case was changed or skipped.

| Identity | SHA-256 |
| --- | --- |
| Final gate source fingerprint | `d928d3fe748088bab98997c8be65b4c93bb58d27a1731e6fed7d56178fc03f36` |
| Golden manifest | `fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3` |
| Tested native Linux executable | `e80848f32b48b2d33f1e5a0e8b65329053e59d8fbca2033ffe118d7d43d162c4` |
| Final Linux standalone | `9de678dbb6ef280eb383ec85c556525171e3feae208a3ae38b8b2126063ebec4` |
| Final macOS standalone | `b589d443798179d7b43e5e8aa3cd58f7be3bcac7132e454f72d4f0a7ff48c27a` |
| Linux package source snapshot | `f61ab61fe009469452b3f6f859287104318dcc49b552cc65553d35605e517c58` |

The baseline commit is `7300e37e1ec51b19d0a9709faa8f742dab371268`.
All 216 production source files plus the Makefile match the tested workspace
in both final packages. The Linux launcher suite completes 16 tests: 15 pass
and one is skipped for an unavailable noexec mount. The macOS package
is an ad-hoc development build. Private commands, raw logs, build evidence and
artifacts remain in ignored `outputs/long-video/`; the native gate result is in
`outputs/cuda-reference-regression/long-video-final-204-001/` on the test server.

## Retained failed attempts and archive integrity

The initial CUDA baseline failed when its retained FFmpeg executable could not
locate the pinned x264 shared library. Supplying that dependency allowed the
complete generation/replay case to pass. The first macOS package attempt built
the executable but could not create its disk image inside the sandbox; the
complete package build passed with disk-image access.

The first script fixture reused its segment name for its final merge, and
planning rejected the collision before inference. A second attempt rendered
its outputs but failed the complete-artifact audit because the old codec probe
was 96×96. That entire attempt is excluded from qualification, despite its
individual process records reporting success. Preflight now uses the first
active segment's canvas; the corrected script matrix and rebuilt artifacts
pass with 256×256 probes. All failed attempts remain in the private logs.

The archived plans are unchanged: [large canvas](../large-canvas-todo.md)
SHA-256 `bf366b522ba261e92adac1a9b2c7acba57c1f3b2b539498be51fa05613f539a3`,
and [empty-latent still](../empty-still-todo.md)
SHA-256 `5131acf0e6c3a847374487ec77453d8c4850715682750497704c3c1c8a362166`.
