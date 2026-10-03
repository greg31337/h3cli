# Adaptive budget and SubBlock reference-media qualification

All 34 tasks are complete. Both required 1344×768/362-frame videos passed
capacity/execution qualification, and the final unchanged golden regression
passed 204/204 outputs. The reference video has visible ghosting; visual
acceptance is not claimed.

The resource ceiling defaults to 4096 MiB for new adaptive requests and can be
set with `--adaptive-cache-max-mib N`. It does not change the adaptive arithmetic,
thresholds, warmups or accepted precision combinations. Sampler section 40 v2
stores the effective byte ceiling; v1 retains 512 MiB. An explicit sufficient
resume ceiling replaces the saved value without discarding numerical history.

SubBlock now accepts image, video and audio references with BF16 projections, adaptive cache
off, reuse/core-reuse 1 and all 50 blocks. Reference, text/vision and audio rows,
first/last video frames, mixed blocks and tails retain protection. Mixed sets,
embedded video audio and external soundtracks are supported. Existing limits
of 12 total references, 9 images, 3 videos and 3 audio inputs remain; standalone
audio requires a visual reference. Quantized references, adaptive references
and continuation remain unsupported. Ordered checkpoint provenance and its reconstructed layout must
agree. No new numerical recipe is required: existing protection arithmetic and
serialized layout semantics are unchanged. Older builds cannot resume these
files because active sampler compatibility requires the same build identity.

Both features remain opt-in; this is capacity and execution qualification,
not visual acceptance or a speedup claim. Historical accepted experiment videos
and blocked fast-preview slots are unchanged.

## Evidence

- [All invocation timings, memory and logs](../../outputs/adaptive-cache-budget/2026-09-28-pro5000/run-index.md)
- [Final evidence audit](../../outputs/adaptive-cache-budget/2026-09-28-pro5000/evidence-validation.json)
- [Frozen workload manifest](../../outputs/adaptive-cache-budget/2026-09-28-pro5000/manifest.json)
- [Final gate and GPU cleanup](../../outputs/adaptive-cache-budget/2026-09-28-pro5000/final-complete.json)

- [Frozen baseline](../../outputs/adaptive-cache-budget/2026-09-28-pro5000/baseline.json)
- [Final implementation source snapshot](../../outputs/adaptive-cache-budget/2026-09-28-pro5000/source-004.json)
- [Serial campaign runner](../../outputs/adaptive-cache-budget/2026-09-28-pro5000/campaign.py)
- [Local Metal ordinary tests](../../outputs/adaptive-cache-budget/2026-09-28-pro5000/local-metal-final.log)
- [Local sampler checks](../../outputs/adaptive-cache-budget/2026-09-28-pro5000/local-sampler-004.log)

The PRO 5000 uses 72 GB VRAM, CUDA 13.0 Update 3 and cuDNN 9.20. Local CPU/Metal
checks use the M4. Model identification uses file metadata only; no base model
payloads are hashed. Native bounded tests preserve their original 50-step
schedule and execute at most six evaluations per invocation. Large jobs have
3,600-second deadlines. Full qualification videos use six steps, all 50 blocks,
full VAE, audio, automatic weight residency and the original beach prompt/seed 42.

## Memory and compatibility audit

Exact cache bytes are `rows × 5376 × 6 + 6168`; checkpoint cache arrays occupy
`rows × 5376 × 4`. The device buffers are allocated before weight admission, so
existing free/live accounting already counts them. Future activation estimates
and device reserves are unchanged. Large tensor offsets and byte counts use
`size_t`; the target's element count fits the existing row/column dispatch types.
Allocation failures free the entire attempted DiT, including partially allocated
cache tensors, before any bounded residency retry. No failed request silently
changes caching, frame count or resolution.

Admission checks run before conditioning when a geometry-only lower bound can
reject the request, and again against the actual packed layout before a prepared
DiT can be reused. The ceiling is excluded from conditioning/AdaLN cache keys.
Checkpoint loading checks geometry, exact element counts, both payload lengths,
required versions and saved/override ceilings before adaptive tensor allocation.
The writer counts all serialized sections before allocating payload copies and
retains atomic replacement and the 16 GiB container bound. Completed AV files
have no resource-budget extension.

## Tests and review scope

Local M4 builds of the CLI and static library pass. The ordinary Metal test
suite passes its available CPU/GPU checks; optional released-model and MLX
fixture tests explicitly skip because those local fixtures are absent. These
skips are not counted as passes. CUDA model-backed checks run separately on the
PRO 5000.

The CUDA host checks include 1,778 ordinary assertions, 3,204 sampler-state
checks, 430,919 existing layout/config assertions across reference modalities,
48 conditioning-key checks, 342 image-geometry/cache checks, 52 sampler CLI
cases, eight adaptive CLI tests, two reference CLI tests and 231 adversarial sampler
container cases across 20 tests. Planner checks include exact-byte boundaries,
overflow, live/future/reserve accounting, slot costs, packed-weight accounting
and bounded residency retries. SubBlock operator checks include a mixed
image/video/audio layout, incomplete tails, both attention layouts, dense
warmup/block 0, protected queries, zero-sparsity equality, workspace rejection
and context recovery.

The small original 50-step schedule at 256×256/22 frames executes six
evaluations under 512, 4096 and 8192 MiB, plus forced weight streaming. All
admitted variants produce identical saved step latents. Their conservative
trace refreshes steps 0–3, hits at step 4 and refreshes at step 5 because the
maximum hit streak is one. Aggressive and combined adaptive/SubBlock paths
also run. The combined path refreshes at the sparse-attention transition;
its measured decisions, rather than an assumed speedup, are retained.

The 1344×768/124-frame bounded prompt uses 12 text tokens, giving 37,722 rows,
1,216,767,000 device cache bytes, 811,173,888 persistent bytes and a minimum
1,161 MiB ceiling. Default and minimum-whole-MiB runs match exactly. Explicit
512 MiB fails before conditioning. Cancellation followed by retry succeeds.
These bounded tests retain the 50-step schedule; they are not completed videos.

The 362-frame bounded prompt also uses 12 text tokens: 109,074 rows,
3,518,297,112 device bytes and 2,345,527,296 persistent bytes. The continuous
two-evaluation prefix completed in 345.80 seconds. Automatic placement selected
28/50 resident BF16 blocks, 21,579,694,080 resident weight bytes and
1,541,406,720 streaming-slot bytes. Its existing future allocation estimate was
40,165,863,424 bytes and reserve 7,634,452,480 bytes. The exact cache was already
included in 4,543,130,328 live bytes before admission; it was not subtracted
again from future capacity. No residency retry was needed without fault
injection.

That compact checkpoint is 2,431,414,460 bytes. Serialization took 46.304 seconds;
the writer reported 2,431,412,492 bytes of payload copies in addition to the
2,345,527,296-byte adaptive export. Peak sampled process-tree RSS was 19.53 GiB
and device memory 52.14 GiB. These RSS measurements include streaming host
storage; they are not just checkpoint overhead. Rebuildable prepared tensors
were deliberately omitted from the bounded driver's files, preserving every
required mutable sampler/cache tensor.

With the 32 GiB injected allocation limit, the same workload first attempted
28 resident blocks, released that failed DiT and retried with 13. A later fused
MLP allocation still exceeded the cap and the job exited cleanly. The normal
same-shape run then succeeded. This establishes bounded failure/recovery; it
does not qualify a physical 32 GiB GPU for this workload.

The separate 8192 MiB run paused after one evaluation in 214.76 seconds. A
512 MiB resume override rejected the file in 21.43 seconds, before adaptive
payload allocation or transformer-weight loading; integrity validation still reads the
mapped checkpoint. A sufficient lower override of 3584 MiB resumed in 237.53
seconds and matched both the continuous step latent and its adaptive decision
trace exactly. Loading took 33.588 seconds and saving the resumed checkpoint
47.014 seconds. The effective replacement is persisted. The resumed file grows
by 116 bytes of provenance to 2,431,414,576 bytes; its cache tensors are unchanged.

The bounded 124- and 362-frame checks finish during adaptive warmup and therefore
record zero hits. This is successful capacity/resume coverage, not evidence of
a speed improvement. The smaller six-evaluation tests above demonstrate actual
cache hits separately.

## Required full videos

[Adaptive video](../../outputs/adaptive-cache-budget/2026-09-28-pro5000/runs/adaptive-362-video/video.mp4)
([sampled frames](../../outputs/adaptive-cache-budget/2026-09-28-pro5000/runs/adaptive-362-video/contact-sheet.jpg)):
1344×768, 362 frames, six steps, seed 42, conservative cache, omitted budget
flag, warmup 4, dense attention, BF16, all 50 blocks, automatic weights, full
VAE and stereo audio. The exact frozen beach prompt and argv are in the
[run record](../../outputs/adaptive-cache-budget/2026-09-28-pro5000/runs/adaptive-362-video/result.json).

| Adaptive video measurement | Result |
| --- | ---: |
| Total process wall time | 901.37 s |
| Text encoder | 12.881 s |
| AdaLN preparation | 5.909 s |
| Transformer weight loading | 15.094 s |
| Denoising | 780.860 s |
| Audio VAE | 0.647 s |
| Video VAE loading / decoding | 3.418 / 78.092 s |
| Final FFmpeg drain | 0.368 s |
| Peak device memory / process-tree RSS | 52.04 / 16.02 GiB |
| Resident BF16 blocks | 29/50 |
| Exact cache / saved adaptive tensors | 3,519,780,888 / 2,346,516,480 bytes |
| Cache hits / full refreshes | 0 / 6 |

The final FFmpeg drain overlaps a streamed decode/encode pipeline; it is not
all encoding time. Total wall time includes startup, profiling and AV-state
saving. With the shorter six-step AdaLN schedule, one additional core block
fits compared with the bounded 50-step checkpoint tests. No allocation retry
was needed. The actual post-warmup score is 0.216146499, above the conservative
0.04 threshold; the final step refreshes by policy.

FFprobe and a complete decode confirmed 362 frames at 24 FPS, 1344×768 and
stereo 32 kHz audio. Sampled frames at 0, 3, 6, 9, 12 and 15.04 seconds show a
consistent beach scene, a turn toward the camera and a hand moving toward the
hair. No gross corruption is apparent in these samples. Contact sheets cannot
establish temporal smoothness, subtle hand/face fidelity or perceptual audio
quality. These six-step capacity results do not imply visual acceptance or
quality equivalence to a dense 50-step render.

[SubBlock reference video](../../outputs/subblock-reference/2026-09-28-pro5000/runs/reference-362-video/video.mp4)
([sampled frames](../../outputs/subblock-reference/2026-09-28-pro5000/runs/reference-362-video/contact-sheet.jpg)):
the same required output geometry, prompt and seed, with exactly one
`inputs/1.jpg --ref-image-size max`, BF16, adaptive off, reuse/core-reuse 1,
SubBlock 0.75, warmup 2, all 50 blocks, full VAE and audio. The 1365×1821 input
becomes 2048×2720 with 21,760 raw patches.

| Reference video measurement | Result |
| --- | ---: |
| Total process wall time | 1,023.60 s |
| Reference VAE encoder / Qwen vision | 5.356 / 1.302 s |
| Text encoder / AdaLN preparation | 13.083 / 6.056 s |
| Transformer weight loading | 15.095 s |
| Denoising | 892.806 s |
| Audio VAE | 0.634 s |
| Video VAE loading / decoding | 5.183 / 78.228 s |
| Final FFmpeg drain | 0.396 s |
| Peak device memory / process-tree RSS | 50.69 / 16.80 GiB |
| Resident BF16 blocks | 28/50 |
| Main sparse / main dense / protected-query calls | 196 / 104 / 392 |
| Actual retained tile pairs | 42.05% |
| SubBlock workspace | 399,891,744 bytes |

The dense warmup steps took 153.172 and 154.741 seconds. Each remaining step
executed 49 sparse calls, one main dense call and 98 separate dense
protected-query calls, taking 146.156, 146.305, 146.318 and 146.112 seconds.
Mandatory protected keys/queries raise actual density above the nominal 25%
keep rate. The 512 MiB workspace bound and existing device reserve are unchanged;
no allocation retry was needed. Detailed GPU event timings are sampled (the
final trace records 4,096 profiled and 5,116 unprofiled events), so stage/step
wall times above are the complete timing evidence. These unequal-conditioning
videos are not a controlled speed comparison.

Full decoding verifies 362 frames, 24 FPS, 1344×768 and stereo 32 kHz audio.
The sampled frames retain the reference's broad hair/clothing appearance and
show turning and hair movement, but **an extra overlapping figure/ghost image
is visible behind the main subject in several frames**. This artifact is a
visual limitation of the measured six-step output. No matching dense reference
baseline was generated in this capacity campaign, so its cause cannot be
attributed specifically to SubBlock. Visual acceptance is not claimed. The
contact-sheet review does not assess every frame or establish perceptual audio
quality.

## Reference-media checks

At 640×480/56 frames on the original 50-step schedule, the image-reference
continuous prefix and resumed prefix match exactly across warmup 2. Steps 0–1
execute 50 main dense calls; steps 2–3 execute one main dense call, 49 sparse
calls and 98 separate protected-query calls each. The two-image `high` case
uses two 1536×2048 canvases (12,288 raw patches each); the one-image `max` case
uses 2048×2720 and 21,760 patches. The default ten-step warmup keeps the short
three-evaluation run dense. Image-reference cancellation and retry pass.

The same-path/size/mtime image replacement test passes: the live context
invalidates stale conditioning, and reused/fresh contexts produce identical
BF16/F32 conditioning and prepared text. This test uses zero denoising
evaluations. CPU key tests separately cover identity, count, order, sizing mode
and canvas.

| Bounded media case | Wall time | Result |
| --- | ---: | --- |
| Video with embedded audio | 61.62 s | Pass |
| Silent video | 54.25 s | Pass |
| Image plus separate audio | 38.60 s | Pass |
| Video plus external soundtrack | 57.10 s | Pass |
| Mixed image/video/audio, continuous four-evaluation prefix | 64.73 s | Pass |
| Mixed set, pause after two evaluations | 54.45 s | Pass |
| Mixed set, resume for two more evaluations | 28.46 s | Exact prefix match |

All these runs retain the original 50-step schedule. Steps after the configured
warmup execute 49 main sparse calls, with dense block 0 and separate dense
protected-query work. Saved states validate ordered reference kinds, audio
presence and reconstructed positions/segments. The recorded golden video
fixture is copied unchanged; the separate audio WAV is a derived experiment
fixture, not an added or modified golden.

Mixed-media cancellation and retry also pass (107.98 seconds). The original
continuous, paused and resumed files are retained at the paths recorded in each
invocation. The local report bundle contains metadata, logs, step hashes and final
videos; it does not duplicate multi-GiB sampler payloads.

## Preserved failures and harness corrections

The first campaign's CPU gate self-check ran in an isolated build without its
binary input fixtures; copying the unchanged bundle into that directory resolved
the self-check. The full recorded regression itself had passed.
A subsequent watcher started before its build directory existed; imports were
changed to use the frozen source snapshot. The pre-correction image-only campaign
was stopped when the requested scope expanded to video/audio; its interrupted
streaming attempt and all completed records were preserved.

Two final-campaign harness issues were corrected without production changes.
The allocation-cap test initially expected the word `memory`, while the runtime
correctly returned `CUDA tensor allocation exceeds injected test budget`. Its
original failure/retry log is retained. The initial looped standalone audio
fixture contained 61,952 samples (1.936 seconds) and correctly failed the existing
two-second minimum. Padding the derived WAV produced 69,952 samples (2.186
seconds); all separate-audio cases then passed. The short WAV, its identity and
the rejected attempt are preserved. Golden fixtures and runtime limits were
not altered. Failed or interrupted attempts are excluded from qualification.

The final ordinary CUDA host suite also exposed omitted binary fixtures in the
source-only snapshot. Copying its fixture directories initially carried only
manifests, so the 55 existing workspace fixture files (13,586,331 bytes) were
transferred and hash-verified before rerunning the suite. Both failed fixture-
lookup attempts remain in the evidence. This restored existing inputs; no
fixture was generated or golden comparison added.

## Completed-state and final checks

Both complete AV payloads pass finite-value checks. Independently decoding the
reference `final.h3av` took 82.69 seconds. Decoded RGB (1,120,960,512 bytes) and
stereo PCM (3,866,624 bytes) match the original output exactly by SHA-256. The
successful duplicate MP4 was removed, retaining one final reference video and
the comparison records. This checks the ordinary completed-state path without
adding adaptive resource metadata to the clean AV format.

The complete CUDA host suite passes, including continuation, bridge, reference-
video layout/posterior, memory, progress, sampler container and CLI checks.
CUDA GPU contract/runtime, BF16 primitives, sampler, bridge, audio, Qwen scaling,
preview primitives/cache and the standalone protected-key route assertions pass.
The route capture verifies that every generated query retains both protected
endpoint key blocks and protected queries retain every key block, including the
incomplete tail. Final CPU golden-gate self-checks also pass.

The unchanged recorded regression passes on every implementation snapshot and
again after qualification:

| Gate | Exact outputs | Test seconds | Source fingerprint prefix |
| --- | ---: | ---: | --- |
| [001](../../outputs/adaptive-cache-budget/2026-09-28-pro5000/gate-001/result.json) | 204/204 | 111.119 | `044f7cc24304` |
| [002](../../outputs/adaptive-cache-budget/2026-09-28-pro5000/gate-002/result.json) | 204/204 | 98.816 | `ca37727b90e18` |
| [003](../../outputs/adaptive-cache-budget/2026-09-28-pro5000/gate-003/result.json) | 204/204 | 95.991 | `0dff9e015915` |
| [004](../../outputs/adaptive-cache-budget/2026-09-28-pro5000/gate-004/result.json) | 204/204 | 98.105 | `4613320b5642` |
| [Final fresh gate](../../outputs/adaptive-cache-budget/2026-09-28-pro5000/gate-final/result.json) | 204/204 | 124.136 | `4613320b5642` |

The final source fingerprint is
`4613320b56426903b9be4b41b04250c47ecced9461ad2b69992d96a1bd02f897`.
All 12 recorded input fixtures and the golden manifest remain unchanged; no
live SGLang oracle, extra golden cases or regenerated expected outputs were
used. The final test stage stays within its 720-second deadline. Both required
large videos stay within their individual 3,600-second deadlines. Final checks
found no active GPU processes; each test subprocess was reaped.

## Measured scope

The larger adaptive ceiling is qualified on the PRO 5000 at 1344×768/124 and
362 frames, including checkpoint overrides, partial residency and bounded
failure/cancellation recovery. Image/video/audio reference SubBlock is qualified
for BF16 CUDA execution, mixed sets, absolute warmups, checkpoint resume and
completed AV decoding. One full max-size image-reference video qualifies the
required 362-frame execution path. Reference audio still needs a visual
reference, and existing count/duration/geometry limits remain.

Adaptive+reference, quantized SubBlock+reference, anchors/continuation and
SubBlock on Metal remain rejected. No FP8/NVFP4 or SOL render campaign was run.
The resource ceiling does not promise that a workload fits smaller GPUs. The
features remain opt-in, and the observed reference-video ghosting is
retained as a visual limitation rather than hidden behind the execution pass.

The final evidence audit passes all 69 completed invocations and excludes the
archived failed/invalid-fixture attempts. Every bounded generation remains at
six or fewer evaluations of the original 50-step schedule. All captured bounded
F32 step values are finite, and admitted-budget/resume decisions match exactly.
Decoded audio is finite and non-silent: adaptive RMS/peak 0.002352/0.018528;
reference RMS/peak 0.017495/0.158668, with no samples at or beyond ±1. These are
signal checks, not a listening review. The report links the two final videos,
complete run records, checkpoint costs, intermediate gates and failure records.
