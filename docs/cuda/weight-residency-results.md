# CUDA weight residency qualification

Status: **complete**; all 31 items in the [archived task list](weight-residency-tasks.md) are closed.
Placement equivalence, lifecycle checks and repeated measurements pass on both
physical GPUs. The PRO 5000 passes the recorded golden gate; the 5090 preserves
its unchanged baseline exactly, with the pre-existing cross-device golden
differences documented below.

## Contract and source identity

The [design](weight-residency-design.md) describes admission, ownership and
scheduling. The [fixed manifest](weight-residency-matrix.json) specifies dense
BF16, all 50 blocks, 90 frames at 24 FPS, six evaluations, seed 42 and a fixed
piano prompt. Reuse, adaptive caching and SubBlock are disabled in the primary
matrix. Separate bounded probes cover their composition with placement.

| Snapshot | Source SHA-256 | PRO 5000 recorded golden gate |
| --- | --- | --- |
| Baseline | `9f41a364f799f5cd70498b190955d4c8c3b93f0c434ba4f7a71b80003bd4822d` | 204/204, 125.50 s |
| Initial implementation | `773d2856110a0aa6c5c127a2d9726e4cc3b8177ec55fec664d13cb2fd6ef361e` | 204/204, 96.02 s |
| Slot/packed-size checks | `47183a56c73bd6f39d0e9fc0d2529534ddef485e213709c40d530f708298bc6f` | 204/204, 95.23 s |
| Lifecycle/composition checks | `dc704d0995204fd2bffd6710e28f68d1cb55daf9a71e87e809c0943782e4bda6` | 204/204, 97.61 s |
| GPU contract registry coverage | `26f8b557335e7fffb5da250b83c81a183ed93022dc6902053d1be2c7e1294065` | 204/204, 97.94 s |
| Final request-capture test coverage | `0a476b6e6b9bfab284e48282717076020488d1e13665ee4992a2c65f9cb68f9c` | 204/204, 99.90 s |

Gate times exclude isolated compilation and remain below the unchanged
12-minute deadline. No fixture, expected hash, tolerance, arithmetic recipe or
saved-state format changed. Base-model identity uses file metadata, never a
scan or hash of the base weights.

The contract-registry and subsequent legacy-test updates only change tests.
Production sources remain identical to the lifecycle/composition snapshot.
Each fresh CUDA build has its own binary identity; numerical and source checks
establish equivalence across builds rather than assuming byte-identical CUDA
objects from different build directories.

The **unchanged 5090 baseline does not match the cross-device recorded goldens**:
113 of 204 hashes differ. Its first differences appear in text preparation. Every candidate above matches
all 204 artifacts from that same 5090 baseline exactly. These are distinct
results: the recorded golden gate passes on the PRO 5000, and placement preserves
the 5090's original output. This is not a 5090 golden pass. NVIDIA's
[cuBLAS reproducibility contract](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility)
requires matching architecture and SM count; it does not establish the precise
cause of this observed cross-device difference.

## Hardware and software

| Measurement | RTX PRO 5000 | RTX 5090 |
| --- | --- | --- |
| Device | RTX PRO 5000 72GB Blackwell | GeForce RTX 5090, physical device |
| Driver-reported VRAM | 73,415 MiB | 32,607 MiB |
| NVIDIA driver | 595.91.07 | 580.178.04 |
| Physical host RAM | 187.512 GiB | 123.434 GiB |
| Container memory limit | No root cgroup limit exposed | 59,999,997,952 bytes, approximately 55.9 GiB |
| PCIe observed under load | Gen3 ×16 | Gen5 ×8; device maximum reports ×16 |
| Model filesystem | XFS on an NVMe partition | 500 GiB overlay filesystem; underlying device not established |

Both use CUDA 13.0.3 (`nvcc` 13.0.88), cuDNN 9.20.0.48 and cuBLAS
13.1.1.3, targeting SM120 with native SGLang, cuDNN and SubBlock enabled.
GPU workloads run serially on each node. The link and host-memory differences
are part of the measurements, not properties inferred from GPU model names.

The pinned upstream cookbook's 5090 example uses 14 resident DiT layers and a
different 864×480, 124-frame, 20-step workload. Its reported times are not a
baseline for these runs. Native AdaLN precomputation releases preparation
weights before core admission, so copying an upstream resident-layer count
would discard usable capacity here.

## Capacity and lifetime audit

Each BF16 core block contains QKV, attention-output, FC1 and FC2 matrices totaling
770,703,360 bytes: 735 MiB. Fifty blocks consume 35.889 GiB. Partial or streamed
execution reserves two 735 MiB slots; full residency allocates no slots. Norms
and heads remain valid independently of the matrix placement.

BF16 admission happens after refinement, AdaLN preparation/pruning and layout
maps. Their surviving tensors, adaptive buffers and the already-created CUDA
workspace are reflected in current free memory. Large temporary preparation
weights have been released. The future estimate uses the actual packed sequence,
including reference and continuation rows: `sequence × 358400 + 1 GiB` covers
activation arenas, optional unfused buffers, norms/heads and later scratch.
The separate reserve remains `max(1 GiB, total VRAM / 10)`.

| Primary canvas | Future estimate | Observed full-core DiT tensor peak | Observed streamed DiT tensor peak |
| --- | --- | --- | --- |
| 640×480 | 3.819 GiB | 38.027 GiB | 3.574 GiB |
| 1344×768 | 10.200 GiB | 42.646 GiB | 8.193 GiB |

These are initial-matrix observations, not device-wide memory peaks. CUDA driver
and library allocations, other retained components and safety reserve also
affect admission. The planner selects the largest count within its conservative
budget; it does not promise the largest count that happens to survive one run.

Packed admission uses the existing descriptor's aligned payload, scale plane
and global slot, rather than a BF16 compression ratio. Adaptive quantization
adds the BF16 block-0 storage. Packed modes retain their existing resident or
per-GEMM streamed path, with preparation/conversion scratch allowance. They do
not acquire partial residency or new prefetch code.

The production callers are `load_dit` → `h3_gpu_plan_bf16_weights` after
preparation, and `h3_gpu_quant_configure` → `h3_gpu_plan_weights` before packed
preparation. The former counts surviving allocations through CUDA free memory;
the latter budgets preparation and conversion scratch before loading its core.
The earlier `h3_gpu_plan_weights` call in `dit.c` now applies only to Metal.
BF16 preparation's 640×480 live allocation is 109,200,984 bytes; the 1344×768
case is 113,788,824 bytes. Reference/continuation layouts feed their actual
sequence into the future estimate, rather than reusing a target-only canvas.
Allocator and library overhead is covered by the separate reserve, not presented
as exact tensor accounting.

The text encoder remains a one-pass streamed component. The full video decoder
remains resident across tiles. Existing decoder headroom checks release host
staging and, when needed, the retained DiT. Placement changes and target geometry
changes invalidate retained DiT contexts. Constrained automatic plans recheck
capacity on each new request; placement is not serialized into sampler state.

## Correctness matrix

| GPU | Canvas | Successful resident counts | Strict full-resident control |
| --- | --- | --- | --- |
| RTX PRO 5000 72 GB | 640×480 | stream 0; natural auto 50; explicit resident 50 | Passed |
| RTX PRO 5000 72 GB | 1344×768 | stream 0; natural auto 50; test caps 16 and 32 | Passed |
| RTX 5090 32 GB | 640×480 | stream 0; natural auto 30; test caps 8 and 16 | Expected capacity rejection |

Every successful row matches native input/velocity/latent trajectories and
decoded RGB/PCM from its unchanged streamed baseline. Preparation captures also
match across placements. Container and provenance headers are excluded. Forced
partial counts are labeled test caps; the PRO's natural high-resolution choice
is full residency.

The final 5090 correctness rerun confirms the same counts. Its 30-block plan
stores 21.533 GiB of persistent BF16 matrices and streams the remaining 20
blocks. The two slots add 1.436 GiB. Strict full residency rejects capacity
before core allocation; it is recorded as an expected rejection, not a successful
render.

The deliberately capped cases also show the expected resource scaling. Every
row reserves 1.436 GiB for two slots. These are correctness runs, not paired
performance claims.

| GPU / canvas / test cap | Resident matrices, GiB | DiT tensor peak, GiB | Pinned, GiB | Six-evaluation core uploads, GiB |
| --- | --- | --- | --- | --- |
| PRO 5000 / 1344×768 / 16 | 11.484 | 19.678 | 24.436 | 146.426 |
| PRO 5000 / 1344×768 / 32 | 22.969 | 31.162 | 12.951 | 77.520 |
| 5090 / 640×480 / 8 | 5.742 | 9.316 | 30.178 | 180.879 |
| 5090 / 640×480 / 16 | 11.484 | 15.058 | 24.436 | 146.426 |

The real-weight memory probe enters the production shared policy, executes a
noncontiguous 25-block active set, and covers zero, one, all-but-one and all
resident counts. It checks exact uploaded bytes over repeated evaluations,
stable pinned bytes, captured options and allocation-failure recovery. The
slot probe delays both copies and consumers over 32 generations without a host
fence between generations, then checks read-error recovery and teardown with a
pending upload. Registration/unregistration failure probes remain in the suite.

The public retained-context probe passes on both GPUs: mode changes, explicit
option conflict, load and step cancellation followed by recovery, paused preview,
resume, changed geometry and cache clearing. A 12 GiB tensor-allocation fault
in a complete PRO 5000 primary render forces `50 → 24 → 11` resident blocks.
Its preparation, all six native trajectories and decoded RGB/PCM remain exact.

First/last anchors, ordered Ref2VA inputs, continuation, reuse and core reuse
pass across streamed, two-block partial and natural automatic placement on the
5090. Adaptive tests exercise actual hits and refreshes with unchanged decisions.
The small SubBlock fixture exercises warmup and its short-sequence dense bypass;
the separate 640×480 probe is used to qualify actual sparse execution.

That 640×480 probe passes all six placements: SubBlock alone and combined with
adaptive caching, each streamed, capped at two resident blocks, and natural
auto. After two warmup steps, SubBlock executes 49 sparse blocks per full
evaluation while retaining dense block 0. The combined run refreshes at the
attention-phase boundary and later takes an adaptive hit. Routing selections,
block counts, decisions and native captures match exactly. Natural auto admits
29 blocks for SubBlock and 28 for the combined feature, reflecting their live
workspace and cache allocations.

Bounded FP8 and NVFP4 checks cover explicit resident, explicit stream, natural
auto and an 8 GiB test capacity. Natural auto selects packed residency; the
reduced capacity selects the existing compressed stream path. Native preparation
and two evaluation trajectories match within each precision. This is admission
coverage, not a new quantized-video quality campaign. The updated public-API
pressure probe also completes two-evaluation FP8 and NVFP4 requests after
actual allocation failures force compressed streaming (12 GB and 6 GB injected
tensor limits respectively).

On the PRO 5000, a retained source context generates a 672×384, 90-frame bundle
in two evaluations, then refines at 1344×768 in four evaluations. Streamed and
16-block partial runs match final AV payloads and decoded RGB/PCM. Audio remains
identical to the source latent, and the target geometry gets a new plan.

The same saved partial-placement checkpoint also resumes with full residency,
one resident block and forced streaming on the PRO 5000. All native captures,
final AV payloads and decoded RGB/PCM match. A separate decode-only request
reproduces the same RGB/PCM without loading the transformer.

The full upscale lifecycle probe also passes all 14 cases at 1344×768 with
16 resident blocks: preparation, uninterrupted refinement, stop/resume at
boundaries 0, 1 and 3, and cancellation/recovery at the same boundaries.
Every invocation uses at most four denoising evaluations. Trajectories and
final AV payloads remain identical across every interruption, and the frozen
audio checks pass throughout.

Metal's available test suite passes. Real-model two-evaluation resident and
explicit SSD runs also match the unchanged Metal source snapshot bitwise.

The final CUDA build and library, GPU symbol contract, host/GPU operator and
scheduler suite, CLI/policy tests, sampler/upscale state tests and LoRA runtime
tests pass on both nodes. The SM90-specific tuning check is inapplicable to
these SM120 GPUs. The PRO host filesystem skips one optional cloned-file LoRA
check; the 5090 runs it. No golden case is skipped.

Earlier broad-suite attempts exposed omitted binary fixtures in the isolated
golden build and missing GPU contract registry entries. The unchanged bundled
fixtures were restored, registry coverage was added, and legacy fault tests were
updated to set budgets before request capture. Failures and reruns remain in
the artifact collection; none was converted into a passing result.

## Repeated timing and resource measurements

All values below use the final source and its passing PRO golden gate. Each
row has three interleaved matched pairs: auto/stream, stream/auto, auto/stream.
Every timed run repeats the native and decoded-output comparisons. Times include
fresh-process loading, full AV delivery and identical diagnostic captures.
The file cache was not flushed; these are cold contexts, not controlled cold-disk tests.

| GPU / canvas | Auto median (range), s | Stream median (range), s | Median wall reduction |
| --- | --- | --- | --- |
| PRO 5000 / 640×480 | 54.33 (54.19–55.72) | 76.97 (76.67–77.27) | 29.4% |
| PRO 5000 / 1344×768 | 135.19 (134.80–136.41) | 155.10 (154.68–155.30) | 12.8% |
| 5090 / 640×480 | 40.77 (39.29–42.66) | 45.72 (45.06–50.80) | 10.8% |

The complete pairs, shown as **auto / stream seconds**, are:

| GPU / canvas | Pair 1 | Pair 2 (stream first) | Pair 3 |
| --- | --- | --- | --- |
| PRO 5000 / 640×480 | 55.72 / 77.27 | 54.33 / 76.97 | 54.19 / 76.67 |
| PRO 5000 / 1344×768 | 136.41 / 155.10 | 134.80 / 155.30 | 135.19 / 154.68 |
| 5090 / 640×480 | 40.77 / 50.80 | 42.66 / 45.72 | 39.29 / 45.06 |

The phase medians separate initialization from repeated computation. Six-step
denoise time is the sum of native step measurements; warmed step time is the
median of steps 2–6, then the median across runs. Each cell is **auto / stream**.
Independent phase medians need not sum to the median total.

| GPU / canvas | Text, s | DiT load, s | Six-step denoise, s | Warmed step, s | Full video decode, s |
| --- | --- | --- | --- | --- | --- |
| PRO 5000 / 640×480 | 12.101 / 12.191 | 15.371 / 32.788 | 16.778 / 19.337 | 2.802 / 3.229 | 8.714 / 8.755 |
| PRO 5000 / 1344×768 | 12.020 / 12.062 | 15.550 / 32.630 | 83.325 / 83.798 | 13.969 / 14.042 | 22.508 / 22.320 |
| 5090 / 640×480 | 9.750 / 9.580 | 12.929 / 16.086 | 12.425 / 12.684 | 2.071 / 2.106 | 5.276 / 5.632 |

Audio decoding medians stay below 0.34 seconds in every group. The 640×480
PRO case also shows a denoising improvement; the 1344×768 case is essentially
compute-bound, with overlapping warmed-step ranges and less than 1% difference
in six-step medians. Full residency does not provide a meaningful high-resolution
denoising speedup in these measurements. The 5090 shows a small denoising gain
and variable load cost. All nine matched pairs improve end-to-end, but three
pairs do not establish a universal speedup for other workloads or storage.

| GPU / canvas / placement | DiT tensor peak, GiB | Sampled GPU peak, MiB | Peak RSS, GiB | Peak anonymous RAM, GiB |
| --- | --- | --- | --- | --- |
| PRO 5000 / 640×480 / auto | 38.027 | 39616 | 1.351 | 1.046 |
| PRO 5000 / 640×480 / stream | 3.574 | 5760 | 36.679 | 36.486 |
| PRO 5000 / 1344×768 / auto | 42.646 | 44350 | 1.948 | 1.703 |
| PRO 5000 / 1344×768 / stream | 8.193 | 8998 | 36.714 | 36.522 |
| 5090 / 640×480 / auto | 25.107 | 26506 | 15.149 | 14.955 |
| 5090 / 640×480 / stream | 3.574 | 5908 | 36.677 | 36.489 |

GPU samples cover the whole process at approximately one-second intervals;
they can miss short peaks. Tensor peaks are allocator counters for the DiT,
excluding driver/library allocations. Sampled process swap is zero in all runs.

| Placement | Core uploads over six evaluations, GiB | DiT pinned bytes, GiB | Logical host-cache preload, GiB | Matrix cache hits |
| --- | --- | --- | --- | --- |
| PRO full auto | 0.000 | 0.031 | 0.000 | 0 |
| 5090 auto, 30 blocks | 86.133 | 14.387 | 14.355 | 480 |
| Forced stream, both GPUs | 215.332 | 35.921 | 35.889 | 1200 |

Resident loading still transfers its weights once. The upload column measures
only streamed core matrices, including initial priming: `6 × streamed blocks
× 735 MiB`, with no unused final wraparound upload. Input latents, conditioning,
text and decoder transfers are separate. Host-cache preload is logical file
traffic, not physical storage traffic. Partial residency reduces both pinned
copies and repeated core transfers in direct proportion to the streamed set.

| GPU / canvas | DiT H2D event time, s | Exposed upload wait, ms |
| --- | --- | --- |
| PRO 5000 / 640×480 | 4.831 / 20.426 | 0.985 / 219.332 |
| PRO 5000 / 1344×768 | 4.843 / 24.095 | 1.027 / 540.887 |
| 5090 / 640×480 | 5.389 / 9.456 | 5.877 / 173.717 |

These event counters are cumulative through denoising, including initialization;
the cells are **auto / stream**. Copy-engine time overlaps compute and must not
be added to wall time. Existing event profiling is opt-in; normal execution
does not gain a device-wide synchronization.

| GPU / canvas | Physical read range, auto / stream, GiB |
| --- | --- |
| PRO 5000 / 640×480 | 0.000–0.000 / 0.000–0.019 |
| PRO 5000 / 1344×768 | 0.000–1.165 / 0.000–0.757 |
| 5090 / 640×480 | 92.479–106.391 / 107.516–113.632 |

Physical reads come from Linux process I/O counters. They include model and
other file I/O and may include read-ahead; they are not the host-cache logical
preload counter. The PRO mostly serves weights from the filesystem cache.
The 5090 overlay host has substantially higher and more variable physical I/O
under its 55.9 GiB cgroup limit. Its measured link is Gen5 ×8, while the PRO
runs at Gen3 ×16. These results support reduced host memory and loading cost,
not a GPU-only or disk-independent performance claim.

## Reproduction and artifacts

Build with the [pinned CUDA environment](cuda-sglang-reference.md). Run
`make test-weight-residency`, then set `H3_MODEL_DIR` to the model directory and
run `make cuda-memory-test`. The context, delayed-slot and feature probes are
[`weight_residency_context.c`](../../tests/weight_residency_context.c),
[`cuda_reference_exact.cu`](../../tests/cuda_reference_exact.cu), and
[`weight_residency_composition.py`](../../tests/weight_residency_composition.py).

[`cuda_weight_residency.py`](../../tests/cuda_weight_residency.py) records commands,
binary/source/manifest identities, placement diagnostics, native step timings,
transfer counters, GPU samples, process RSS/anonymous memory/swap and physical
read counters. Host-cache logical bytes and process physical reads are different
measurements. Cached stream enqueue/read rates are not disk bandwidth.
`--timing` runs the manifest's three interleaved auto/stream pairs.
Each invocation loads a fresh process; operating-system file-cache state is
observed rather than forcibly cleared.

Initial single-run wall times are retained as correctness evidence, not used
as speedup claims. The repeated measurements above use the final source.

Generated evidence is retained under ignored `outputs/weight-residency/`; the
links below refer to the local collection, not committed model data. Commands,
per-run identities, native hashes, resource samples, logs and MP4s are adjacent
to each matrix result.

| Evidence | Local artifact |
| --- | --- |
| Final recorded golden gate | [PRO 5000, 204/204](../../outputs/weight-residency/evidence/pro/candidate-05-gate/result.json) |
| 5090 baseline limitation and unchanged outputs | [Original baseline](../../outputs/weight-residency/evidence/5090/baseline-gate/result.json), [final equivalence](../../outputs/weight-residency/evidence/5090/candidate-05-equivalence.json) |
| Primary full/partial/stream matrix | [PRO 5000](../../outputs/weight-residency/evidence/pro/candidate-03-final-matrix/result.json), [5090](../../outputs/weight-residency/evidence/5090/candidate-03-final-matrix/result.json) |
| Final interleaved timing runs | [PRO 5000](../../outputs/weight-residency/evidence/pro/candidate-05-final-timing/result.json), [5090](../../outputs/weight-residency/evidence/5090/candidate-05-final-timing/result.json) |
| Full-video allocation-failure recovery | [50 → 24 → 11 blocks](../../outputs/weight-residency/evidence/pro/candidate-03-oom/auto-12GiB-allocation-failure/result.json) |
| Conditioning, reuse and adaptive composition | [24 bounded comparisons](../../outputs/weight-residency/evidence/5090/candidate-03-composition/result.json) |
| Actual sparse SubBlock and combined execution | [Six full-geometry comparisons](../../outputs/weight-residency/evidence/5090/candidate-03-sparse/result.json) |
| Packed admission and allocation recovery | [FP8/NVFP4 placements](../../outputs/weight-residency/evidence/5090/candidate-03-quant/result.json), [pressure recovery](../../outputs/weight-residency/evidence/5090/candidate-05-quant-pressure/result.json) |
| High-resolution upscale placement equivalence | [Streamed](../../outputs/weight-residency/evidence/pro/candidate-03-upscale-stream/numeric.json), [16 resident blocks](../../outputs/weight-residency/evidence/pro/candidate-03-upscale-auto/numeric.json) |
| Final resume placement equivalence | [Resident](../../outputs/weight-residency/evidence/pro/candidate-05-resume-resident/numeric.json), [one resident block](../../outputs/weight-residency/evidence/pro/candidate-05-resume-auto/numeric.json), [streamed](../../outputs/weight-residency/evidence/pro/candidate-05-resume-stream/numeric.json), [decode-only](../../outputs/weight-residency/evidence/pro/candidate-05-decode-only/numeric.json) |
| High-resolution stop/resume and cancellation/recovery | [All 14 lifecycle cases](../../outputs/weight-residency/evidence/pro/candidate-05-upscale-lifecycle/result.json) |
| CUDA host/GPU/CLI/state suites | [PRO 5000](../../outputs/weight-residency/evidence/pro/candidate-05-final-suite-exit.json), [5090](../../outputs/weight-residency/evidence/5090/candidate-05-final-suite-exit.json) |
| Metal unchanged-source comparison | [Resident and SSD](../../outputs/weight-residency/metal-placement-complete.json) |
| Detailed measured counters | [Resource and phase measurements](../../outputs/weight-residency/measurements.json) |
| Final evidence audit | [750-artifact index and qualification checks](../../outputs/weight-residency/artifact-index.json) |

The primary matrix uses the production-complete snapshot. The final two snapshots
only adapt existing tests to the captured-request contract; their production
source hashes are identical. The final source reruns the complete gate, broad
suites and all timing pairs. Its result files bind the actual freshly built
binaries, rather than relying on a previous build's identity.
