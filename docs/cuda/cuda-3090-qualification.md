> Historical renderer/qualification record. Commands that refer to removed
> fast/legacy modes require the archived source. Use the
> [current single-pipeline design](design-single-pipeline.md) and
> [M6/M7 qualification](single-pipeline-final.md) for supported execution.

# RTX 3090 qualification

The supplied server has a GeForce RTX 3090, SM86, 82 multiprocessors,
24,576 MiB reported by `nvidia-smi` and 23.56 GiB available to the CUDA runtime.
It runs Ubuntu 24.04, driver 595.71.05, CUDA toolkit/runtime 12.8,
cuDNN 9.10.2 and cudnn-frontend 1.11.0. The host has AMD EPYC 7H12 CPUs;
the container memory limit is 124,999,999,488 bytes, with no swap.

SSH access and model inventory were verified before building. Development,
dependencies, temporary files, CUDA caches and outputs use the local drive
under `/workspace`. Original FL2VA and Ref2VA checkpoints are in
`/path/to/models/MiniMax-H3`. Source and binaries are in
`/path/to/h3.c-3090`; environment and build wrappers are
`/path/to/h3-3090/env.sh` and `/path/to/h3-3090/make.sh`.
Local evidence is under `outputs/cuda-validation/rtx3090`.

The build targets SM86 directly and includes optional cuDNN SDPA and OpenSSL
compatibility hashing. Python is used by test helpers, not inference. The
original BF16 model weights are neither quantized nor modified.
An independent `CUDA_CUDNN=0` build also passes native fast-dispatch and
stream/cache runtime tests. `CUDA_ARCH=auto` correctly emits SM86 and compute86
targets. The tested main binary's SHA256 is
`ef249f20d5501b6a765cb6ec5ccb05db6190846f2f01cc539aa0d6edba8cab58`.

The GPU negotiates PCIe 4.0 ×16 under load. This two-socket host exposes the
GPU on NUMA node one. Effective DiT H2D event bandwidth is about 13–15 GiB/s;
the CPU/file-to-pinned-buffer path costs more wall time than the DMA events
alone. Results therefore describe this host and GPU combination.

## Attention and streaming

The cuDNN-first policy works on SM86 and outperforms the native fast candidate
at the measured DiT shapes. Each isolated case uses one warmup and three timed
iterations, BF16 inputs, 56 heads, dimension 128 and ordinary output layout:

| Tokens | Default | Native fast | cuDNN fast |
| ---: | ---: | ---: | ---: |
| 2,281 | 52.941 ms | 3.470 ms | 2.316 ms |
| 18,225 | 3,300.134 ms | 207.219 ms | 131.846 ms |
| 32,768 | Not measured | 652.174 ms | 419.460 ms |
| 110,592 | Not measured | 7,594.195 ms | 5,190.263 ms |

The long-shape logs confirm actual cuDNN plans and finite outputs. They test
isolated attention, not full-model capacity or complete long renders.

Head-major layout, tail shapes, unsupported dimensions and causal/GQA fallback
also pass. Decoder F32 attention at 1,797 tokens takes 32.596 ms by default,
2.620 ms with TF32 products, 3.352 ms with F32 products and 22.734 ms with the
tiled candidate. These are isolated kernel results, not full-render speedups.

The standard 288×384, 56-frame, 20-step fast pilot completes in 159 seconds.
It retains 15.94 GiB of unchanged streamed weights, serving 303.48 GiB of
uploads from GPU memory. Actual DiT streaming falls from 718.49 GiB to
415.02 GiB; peak DiT tensor/cache allocations are 18.17 GiB. The default pilot
completes denoising but reaches the 300-second cap during video decoding.
Its partial log is retained; it is neither a passing render nor a measured
complete-run speed comparison.

The remaining matched previews use the smaller `--compact` preset in a
separate directory (192×256, 22 frames and ten steps; continuation targets
90 frames with 39-frame overlap). Their times must not be compared directly with the
standard pilot or with previous GPUs' larger presets. Wall times exclude the
separate full-media validation:

| Matched compact case | Default | Fast | Speedup |
| --- | ---: | ---: | ---: |
| `1.jpg`, 22 frames | 149.06 s | 81.05 s | 1.84× |
| Continuation segment one, 90 frames | 191.01 s | 108.69 s | 1.76× |
| Continuation segment two, 51 new frames | 177.41 s | 97.14 s | 1.83× |

Disabling the optional weight cache takes 90.57 seconds on the same compact
image workload, versus 81.05 seconds with automatic caching. These are single
matched runs, not guarantees for production lengths. Whole-run fast/default
comparisons include accelerated compatibility hashing as well as GPU changes.

`H3_CUDA_REGISTER_WEIGHTS=1` is unsuitable on this host: its compact image
candidate reaches the 300-second cap during denoising. Keep it unset. Normal
bounded bounce-buffer uploads, cuDNN-first attention, TF32 decoder attention
and automatic weight caching are the recommended configuration here.

A separate 256-MiB model-file upload diagnostic uses the production streaming
API, disables the optional cache, checks transferred bytes, and measures one
warmup plus three timed uploads. Ordinary uploads achieve 6.64 GiB/s with no
CPU affinity, 6.51 GiB/s pinned to CPU node zero, and 5.00 GiB/s pinned to the
GPU's node one. Registered uploads achieve only 0.81–0.89 GiB/s. These include
host staging costs and are not directly comparable to DMA-only event rates.
Neither registration nor CPU affinity improves this host's measured path.

## Validation scope

This qualification uses bounded short renders and component tests. Render
invocations share a 2,700-second ledger and each has a 300-second maximum;
isolated kernel cases have a 120-second maximum. Two-step smoke outputs are
functional evidence only. The standard quality candidate uses 20 steps;
compact previews use ten. All comparison renders keep 50 transformer layers,
reuse/core-reuse one and token reduction disabled.
The shared ledger records 22 invocations totaling 2,645.15 seconds, including
both capped performance experiments. The standard default pilot and the
registered-upload compact pilot are the two timeout results; no additional
long rendering round was started to replace them.

The default host/CLI/state/tokenizer suite, all 41 Metal primitive fixture
comparisons, and all 23 released-model comparisons pass. Released components
cover vision, text, video encoding, audio encoding/decoding, video decoding,
one DiT block and a complete small DiT evaluation. The SM120-specific tuning
test correctly skips this SM86 card; it is not counted as SM120 evidence.

Fast-dispatch tests pass with cuDNN, native attention, reference override,
zero decoder scratch, layout tails and cooperative QKV operations. Runtime
regressions pass for storage/range transfers, asynchronous slot generations,
cancellation, in-flight teardown, bounded cache admission, file replacement
and modification, phase release/refill and allocation-pressure cache eviction.

Real-weight memory tests pass with 25 and 50 layers in default and fast modes.
The 25-layer test covers resident, forced streaming, automatic resident and
injected allocation-failure fallback. At 50 layers, the expected resident
capacity rejection is recorded separately from successful forced/automatic
streaming and a 3 GiB injected allocation budget. Supported default modes
produce identical F32 outputs; fast modes check finite outputs and memory
behavior. CUDA memcheck reports zero errors and zero leaked allocations for
the cache/slot regression. Native attention racecheck reports zero hazards.
The registered-upload runtime regression also passes; its rejection above is
for performance, not a discovered data-integrity failure.

Two-step resume checks pass in default and fast modes, including fast
continuation restart and an explicit fast-to-default checkpoint handoff.
The default restart checks exact F32 completed-boundary trajectories; its
final saved video/audio latents also match the uninterrupted control exactly.
Fast restart checks valid finite output, not numerical equality of the future
trajectory. The controls include separate checkpoint capture, so their times
are not standalone render timings.

Default and fast continuation audits use rectangular 192×256 states, changed
face/body references, two steps, a 90-frame target and 39-frame overlap. They
check initial noise and prefix construction, unchanged protected prefixes at
every Euler boundary, source-state immutability, 51 delivered new frames,
audio trimming and AV-state roundtrip. All 19 saved AV states pass independent
checksum/geometry/finite-value inspection.

The prepared-state/decoder-cache pressure test passes after reserving VRAM to
leave 4 GiB free: prepared conditioning is evicted, the decoder is retained,
and generation/checkpoint capture still completes. It uses forced streaming
and disables optional weight caching to bound the fixture; the separate
runtime suite checks optional weight-cache pressure eviction.

Additional two-step fast smokes pass for multiple image references (`2.jpg`
and `body1.jpg`) and a video with separately replaced audio. The latter uses
the successful 20-step fast pilot as its reference. The first/last-frame
render was not started because its timeout reservation would exceed the
shared budget. LoRA/Turbo renders were not repeated: no folded LoRA model
was provided on this server. These scope limits do not substitute smoke
outputs for quality evidence. Every successful reviewed MP4 passes full
FFmpeg decoding and stream/dimension/frame-count checks.

On 2026-09-18, the user accepted the visual quality of this corpus and the
5090 outputs: "both 3090 and 5090 videos look good". The
[acceptance record](cuda-fast-acceptance.json) records this hardware follow-up
separately from the prior PRO 6000 decision. The local
`outputs/cuda-validation/rtx3090/quality-acceptance.json` snapshots existing
artifacts by SHA-256. Per-clip playback/listening fields and scores retain
their original values because those details were not separately supplied.
Two-step outputs remain functional evidence only.

The earlier sampled-frame observations remain recorded: the compact image pair has similar
subject/scene structure in sampled frames; the ten-step continuation pair has
soft or distorted face/limb detail in both modes and different poses. This
preview preset retains its ten-step scope; approval does not establish
production-length quality or separately confirm audio review.
Local galleries are
`outputs/cuda-validation/rtx3090/quality/review.html` (20-step candidate),
`compact/review.html` (matched ten-step pairs), `functional/review.html` and
`smoke/review.html` (two-step functional outputs).

T107 is complete within this recorded qualification scope; 118 of the original
120 CUDA port tasks were complete at this point. The subsequent
[H100 qualification](cuda-h100-qualification.md) records T109 and conditional T114
work. No engine change was needed on the 3090. Test helpers now support
bounded compact comparisons, default resume and rectangular fast continuation
audits, and reject reuse of results generated with different command settings.

## Running on this server

```sh
source /path/to/h3-3090/env.sh
cd /path/to/h3.c-3090
/path/to/h3-3090/make.sh -j8 all
./bin/h3cli -d /path/to/models/MiniMax-H3 --fast-cuda \
  --ref-image inputs/1.jpg --width 288 --height 384 --frames 56 \
  --steps 20 --layers 50 --reuse 1 --core-reuse 1 --seed 1001 \
  -p 'The woman in <Picture 1> walks through a sunlit forest.' \
  --profile -o outputs/preview.mp4
```

Leave weight mode on `auto`: all 50 BF16 transformer blocks cannot reside
in 24 GB. Fast mode can cache a subset of streamed weights in spare VRAM.
`H3_FAST_CUDA_WEIGHT_CACHE_MB=0` disables that optional cache for comparisons;
it is not required for ordinary operation.
