> Historical renderer/qualification record. Commands that refer to removed
> fast/legacy modes require the archived source. Use the
> [current single-pipeline design](design-single-pipeline.md) and
> [M6/M7 qualification](single-pipeline-final.md) for supported execution.

The optional `--fast-cuda` workflow is implemented and functionally tested.
Matched short PRO 6000 renders completed **2.31–2.90× faster** than default
CUDA. The user accepted the quality on 2026-09-17, closing the remaining
perceptual acceptance gates. T121–T164 are complete within the documented
bounded validation scope. This report does not certify long-form quality or
a production-length speedup.

Test host: RTX PRO 6000 Blackwell Server Edition, SM120, 96 GB, CUDA 12.8.93,
driver 595.91.07, Ubuntu 24.04, GCC 13. Optional cuDNN 9.8.0/frontend 1.11.0
and OpenSSL were enabled for the main corpus. Artifacts are under
`outputs/fast-cuda/` locally and `/path/to/h3.c/outputs/fast-cuda/` remotely.
M4 Max was used for representative default and checkpoint-handoff regression.
No original `test2.sh`, 362-frame/50-step or historical long render was launched.

## Complete-operation results

Both modes use original weights, all 50 layers, reuse/core-reuse one, token
reduction disabled, matched seeds and references, and `--profile`. The frozen
presets are 288×384/56 frames/20 steps; continuation targets 90 frames with
39-frame overlap and 51 new second-segment frames; detail is 480×640/22/20.
Reference video comes from the coherent default 1.jpg clip. The additional
`video-audio-distinct` pair uses a separately time-adjusted audio track; the
original replacement case exercised the API with audio extracted from the
same source. No smoke clip counts as quality evidence.

| Case | Default | Fast | Ratio |
| --- | ---: | ---: | ---: |
| image05 | 63.75 s | 27.55 s | 2.31× |
| t2va | 62.90 s | 25.41 s | 2.48× |
| first | 66.62 s | 26.40 s | 2.52× |
| last | 66.56 s | 26.35 s | 2.53× |
| first-last | 68.40 s | 25.95 s | 2.64× |
| images | 65.61 s | 24.35 s | 2.69× |
| video | 94.19 s | 39.27 s | 2.40× |
| silent | 92.98 s | 36.46 s | 2.55× |
| video-audio | 93.99 s | 37.61 s | 2.50× |
| video-audio-distinct | 98.64 s | 39.22 s | 2.52× |
| audio | 66.30 s | 24.15 s | 2.75× |
| image12-seed2 | 63.25 s | 23.05 s | 2.74× |
| detail | 65.16 s | 24.05 s | 2.71× |
| chain1 | 83.37 s | 29.70 s | 2.81× |
| chain2 | 83.42 s | 29.61 s | 2.82× |
| bridge | 84.18 s | 29.80 s | 2.82× |
| changed-reference | 83.79 s | 28.85 s | 2.90× |
| handoff | 82.63 s | 28.50 s | 2.90× |

The first twelve JSON wall clocks include about 0.1–0.3 seconds of media
validation after process completion. Their logs also contain the CLI's own
MP4 wall time (1.jpg: 63.45 vs 27.22 seconds). Later records separate operation
and validation time. These small accounting differences do not determine the
speed conclusion. Each MP4 was probed and decoded in full by FFmpeg: expected
geometry/frame count, 32-kHz stereo audio, no packet/decode errors.

Every render JSON retains its exact command, binary/source identity, seed,
settings and input hashes; later records also hash video/audio/continuation
references. The original default executable SHA-256 is
`b1fcc7e43a0c8d4e1af1e5e4c87d13085ef992625f5e9c5fc502c6e64d8de37f`.
1.jpg SHA-256 is
`4e0f2d688e9ed72a8c97a30dc326f12a4928ffb8dd3f5e79c59b998e9a97ddff`.
The complete Ref2VA tree fingerprint captured in the sampler state is
`2eed3a9f8c38f909ca8919cb6b744ceebb1bfe070bc4937244bfece87970f4d1`;
AV compatibility digest is
`0046f2b99169f9e4c41b289f07b8a4bf044b2e29eae46e56c5465560fb814548`.
Models remained untouched; folded LoRA tests use separate trees and manifests.

Total accounted PRO rendering/decoder time, including calibration pilots and
artifact conversion: **43.69 minutes**, below 45 minutes; no individual
render exceeded 300 seconds. M4 rendered the default 1.jpg case in 173.69 s
and the explicit fast-CUDA checkpoint handoff in 144.21 s: 317.89 s total,
below its 20-minute budget. CPU model folding/building and isolated kernels
are recorded separately from rendering. Budgets include failed test attempts.

## Savings and residual costs

The initial default pilot took 63.55 s. Native BF16 attention plus TF32 decoder
attention reduced it to 50.24 s before accelerated SHA and GEMM plan caching.
The cuDNN/cache pilot captured under Nsight took 52.33 s and is not a fair
unprofiled timing comparison. The subsequent complete pilot took 24.50 s.
Compatibility hashing fell from 32.59 to 5.50 s by hashing the same model bytes
with optional CPU acceleration; digests and integrity checks are unchanged.

For the matched second segment, DiT time through denoising fell from 27.70 to
13.07 s. Those are cumulative context times including 3.15/3.31 s loading;
the denoising increments are approximately 24.55/9.76 s. VideoVAE fell from
17.28 to 5.00 s; fast mux took 0.19 s. VideoVAE readback/unpack/stitch totaled
approximately 0.045/0.192/0.049 s. GPU frame stitching/delivery overlap was
therefore not added for this small potential whole-operation saving.

Cold plan construction is visible: the first fast corpus DiT load was 6.83 s,
versus later continuation loading around 3.31 s. cuBLASLt plans are bounded and
cached, and refresh bias pointers. The shared-weight projection probe measured
cold setup of 56–176 ms versus warmed GEMMs of 0.78–7.75 ms, depending on shape.
These component figures do not claim an equally large whole-generation cache
speedup. Three warmed cuDNN short-shape trials were 0.4701/0.4532/0.4534 ms;
TF32 decoder trials were 1.1732/1.1718/1.1734 ms. Complete renders were not
repeated exhaustively; timings across different corpus inputs are not a
statistical repeatability estimate.

Nsight Systems captured a full short operation. Large BF16/F32 projection GEMMs
now dominate GPU time. SwiGLU accounted for 0.151 s and decoder softmax 0.139 s;
additional broad activation/modulation fusion was not selected. Existing bias
epilogues remain fused. CUDA free/event API times include overlapping GPU waits
and are not additive phase costs. Repeated buffers and descriptor reuse were
retained; stream-ordered allocation was deferred because the trace did not
establish a material gain beyond these changes.

## Isolated kernels and evaluated experiments

One warmup and three timed iterations, each case capped at 120 seconds. These
are complete real/production sequence shapes without long video generation.

| Shape (S,H,D), ordinary layout | Default | Native / query-tiled | cuDNN / TF32 |
| --- | ---: | ---: | ---: |
| DiT (2,281,56,128) | 5.529 ms | 1.091 ms | 0.459 ms |
| DiT (18,225,56,128) | 328.722 ms | 55.017 ms | 24.291 ms |
| DiT (18,945,56,128) | 353.639 ms | 59.268 ms | 26.532 ms |
| Decoder (1,797,32,64) | 19.013 ms | 8.238 ms | 1.172 ms |
| Decoder (4,096,32,64) | 98.021 ms | 42.344 ms | 4.099 ms |

The original benchmark logs named the aggregate fast-attention count
`native_calls`; it includes cuDNN. New logs correctly call it
`fast_attention_calls`. Candidate names and startup/dispatch logs establish
which implementation ran. cuDNN wins the measured supported DiT shapes;
native remains the dependency/capability fallback. DiT never uses quadratic
score scratch. Unsupported dimensions, causal attention and GQA retain the
existing implementation. Decoder head batches use at most the configured
score budget (64 MiB default); zero-budget fallback was exercised.

Native experiments covered query/key tiles 64×32, 64×64, 128×32 and 128×64,
four/eight warps, padded shared layouts 128/136/144 and asynchronous staging
on/off, both layouts, batch-two tails and short/long sequences. The 128×64
shape with 136-element padding and double buffering remains the selected
native shape. At S=18,225, padding 128/136/144 took 113.69/53.68/55.08 ms;
disabling pipelining took 62.43 ms. The separate generic vector-load prototype
measured 54.53/81.91/58.76/51.46 ms for the four tile shapes. It stays a
component experiment; a small isolated loading improvement has not been
qualified on the visual corpus. The production kernel reports 205 registers,
zero local bytes, 69,632 shared bytes and one active block/SM.

Cooperative Q/K RMS+RoPE measured 1.326→0.143 ms for S=2,281 and
10.338→1.075 ms for S=18,225 (56 heads, D=128). Layout/copy checks cover both
QKV packings, partial RoPE and D=72/128/256. No tensor-error threshold is used
to qualify this fast arithmetic.

The saved-state decoder comparison completed in 11.08 s default, 3.77 s F32
GEMM, 6.62 s query-tiled F32, and 3.67 s TF32 GEMM with a 32-MiB projection
workspace. TF32 at 64/128 MiB took 3.72/3.77 s, so workspace stays at 32 MiB.
All candidates decode the same 56-frame AV state; sampled default/TF32 frames
look comparable. The selected TF32 route is covered by the user's workflow
quality acceptance; alternatives are not separately certified.

Shared-weight batching was evaluated with actual decoder projection shapes:
1/2/4 tiles reduced time per tile by roughly 1% for the QKV projection and
12% for the MLP expansion, while tensor peaks grew from 109→286 MB and
267→664 MB respectively. This is a projection feasibility test, not a full
batched decoder. It did not establish a worthwhile end-to-end saving; serial
decoding, tile coordinates, overlap and stitching remain unchanged.

A fixed-shape attention CUDA Graph prototype verified fresh input on replay.
At the short shape, ordinary/graph execution measured 1.0749/1.0707 ms with
0.349 ms capture overhead. Long-shape replay was slightly slower. No production
graph was enabled, and no scheduling/checkpoint boundaries were removed.

Nsight Compute was attempted, but the host driver denied hardware counters
(`ERR_NVGPUCTRPERM`). Resource/occupancy queries and Nsight Systems are available;
hardware stall/throughput-counter claims are not. Compute Sanitizer memcheck
passed the native batch-two, 129-row head-major tail with zero errors.

## Functional and default regression coverage

Fast functional tests cover defaults/TLS isolation, mode/candidate cache keys,
required checkpoint extension and version rejection, reference precedence,
causal/GQA/shape and scratch fallbacks, bias-pointer refresh, finite output,
QKV layout, batch/tail indexing and cleanup. SHA-256 tests retain exact file
integrity semantics across 4,097 chunked lengths. Runtime stress passed 64
asynchronous streamed-slot handoffs, first resident overwrite/stream transition,
16 cancellations and in-flight teardown. Streamed ready/release fences remain
active; resident release events are omitted only with an overwrite fence.

Ordinary and continuation resume passed uninterrupted-control comparison of
structure, completed step boundaries and finite state, fresh-process restart,
explicit default-CUDA handoff, and a CUDA-fast→M4-default handoff. Fast checkpoint
resume on Metal without the explicit handoff flag is rejected. Build/model/env
identity checks remain strict. Cache tests switch fast/default/fast in one
context and exercise decoder/prepared eviction under reserved VRAM pressure.
Hard and bridge continuation, changed seeds/references and a representative
cross-mode AV handoff produced the expected 51 new frames. No soft continuation
mode exists in this repository; none was invented or claimed as tested.

All 50-layer resident, streamed, auto and injected-3-GiB-budget evaluations
passed. The small-layout component reported 35.93 GiB peak tensor storage
resident versus 1.49 GiB streamed. Short rendered DiT contexts peaked around
36.7–36.9 GiB and the decoder at 9.365 GiB, plus bounded workspaces. The partial
one-second GPU telemetry window peaked at 84,089 MiB; it includes changing
workloads and is not a per-preset peak. Forced streaming on this 96-GB card is
not hardware qualification of a physical 24/32-GB GPU.

Reuse, core-reuse and token reduction checks passed.
Realism (20 steps) and Turbo (6 steps) folded Ref2VA generations passed at
288×384/56; their cold compatibility operations took 62.32/43.53 s and are
not original-weight performance comparisons. This exposed and fixed the
unconditional FL2VA startup requirement. FL2VA-only, Ref2VA-only, combined,
incomplete and wrong-selected-mode installations are now covered. Existing
LoRA host tests passed (36 tests); folding/provenance remains unchanged.

Final default CUDA `make test` and fast tests passed. The default suite retains
41 primitive numerical comparisons, 35 tokenizer/presentation fixtures per
mode, SM120, convolution/attention, sampler and host checks. All seven available
released-weight component comparisons passed on CUDA and Metal. The optional
build with both cuDNN and OpenSSL disabled builds and passes native fast tests.
Default macOS `make test`, sampler/adversarial/CLI checks and the runtime stress
passed. Legacy `make parity` could not run because its ignored MLX fixtures
(e.g. `misc/fixtures/h3_dit.safetensors`) are absent; related optional default
tests report their skips. These omissions are not reported as passing tests.

## Quality review and completion gate

Open `outputs/fast-cuda/quality/review.html`, `decoder/review.html`, and
`functional/review.html`. Review JSON files start pending and never infer
acceptance from finite latents, mux success or a perceptual scalar. Sampled
frames from image, T2VA, first/last, multiple-image, video and continuation pairs
show comparable subjects/scenes/colors without the prior malformed tiled
rendering. Native 480×640 frame 11 was inspected for detail and seams. Poses
and trajectories differ; those differences are allowed by this contract.

On 2026-09-17, the user supplied the outstanding quality acceptance:

> I confirm that quality is very good and acceptable; proceed accordingly

This workflow-level confirmation closes
T131/T135/T137/T139/T140/T152–T155/T161/T164. Together with the recorded
implementation and automated validation, all T121–T164 are complete. The
[acceptance record](cuda-fast-acceptance.json) records the reviewer, exact
statement and scope. `outputs/fast-cuda/quality-acceptance.json` also binds that
decision to the existing corpus files by SHA-256. Earlier per-clip review files
retain the assistant's still-frame observations and unfilled playback/listening
details; no individual review actions or scores were invented from the user's
overall confirmation.

Acceptance covers the documented tested workflow and bounded corpus. It does
not extend to future rerenders, unshipped kernels, untested GPUs
or long-form output. `--fast-cuda` remains opt-in. The ~2× target is met on the
measured bounded workloads, with CPU hashing, projection GEMMs, loading and
remaining decoding costs explaining the residual wall time.
See [cuda-fast.md](cuda-fast.md) for exact build/run/target commands.

## RTX 5090 follow-up

The [RTX 5090 qualification](cuda-5090-qualification.md) adds actual 32-GB
SM120 testing. It revealed repeated full-core uploads while most VRAM was
unused. Fast mode now retains a bounded subset of streamed weights without
changing their bytes or the model computation. On the matched short pilot,
wall time fell from 86.63 to 58.31 seconds; default CUDA took 122.32 seconds.
The report records the additional image/detail/continuation pairs, memory
limits and resume/reference checks. On 2026-09-18, the user confirmed that
"both 3090 and 5090 videos look good". This visual approval is recorded as a
hardware follow-up in [cuda-fast-acceptance.json](cuda-fast-acceptance.json);
per-clip playback/listening actions and scores were not separately supplied.

## RTX 3090 follow-up

The [RTX 3090 qualification](cuda-3090-qualification.md) adds SM86 and 24-GB
testing. Native and optional cuDNN fast paths work; cuDNN is faster at the
measured attention shapes. The existing partial weight cache adapts to this
card's capacity. A matched 192×256/22-frame/ten-step preview takes 149.06 seconds
by default and 81.05 seconds with fast mode. The standard 20-step fast pilot
takes 158.83 seconds; its default counterpart reaches the fixed render cap,
so no complete-run ratio is assigned to that pair.

The smaller comparison and compact functional helpers preserve the shared
render ledger. Their reduced settings are recorded explicitly and do not
replace the prior twenty-step quality acceptance. The report also records
memory, continuation, resume and safety checks, rejected registered uploads,
and the user's 2026-09-18 visual approval of this new corpus alongside the
5090 outputs. Both hardware follow-ups have local SHA-256 artifact snapshots
in their respective `quality-acceptance.json` files. Approval retains the
documented short/compact settings; two-step outputs remain functional evidence.

## H100 follow-up

The [H100 qualification](cuda-h100-qualification.md) closes the remaining
SM90 port tasks T109/T114. Generic qualification preceded a measured default
attention specialization: 512 threads reduce the 18,225-token, 56-head case
from about 982 to 785 ms while preserving exact BF16 output. A matched full
default MP4 and saved AV state are byte-identical before and after tuning;
component, continuation and twenty-step checkpoint/restart regressions pass.

The existing cuDNN-first fast policy is fastest on this H100. Six matched
twenty-step pairs take roughly 36–41 seconds in fast mode versus 93–120
seconds in tuned default mode, including media validation. The 80-GB card
holds the full core resident. Native fallback, memory pressure, cache/mode
switching, reference smokes and sanitizer checks pass. Registered weight
uploads do not improve the measured short case and remain optional.

The shared render ledger totals 2,512.06 seconds across 41 invocations.
The report records isolated production-scale kernels separately, the
installed cuBLASLt limitation under global forced PTX, the successful
project-only PTX check, and missing optional fixtures/LoRA coverage.
H100 playback/listening acceptance is pending; earlier hardware approvals
do not cover these outputs. The review page is
`outputs/cuda-validation/h100/quality/review.html`.
