> Historical renderer/qualification record. Commands that refer to removed
> fast/legacy modes require the archived source. Use the
> [current single-pipeline design](design-single-pipeline.md) and
> [M6/M7 qualification](single-pipeline-final.md) for supported execution.

# RTX 5090 qualification

The supplied server has a GeForce RTX 5090, SM120, 170 multiprocessors,
32,607 MiB reported by `nvidia-smi` (31.36 GiB available to the CUDA runtime).
It runs driver 580.126.20, CUDA toolkit/runtime 12.8, cuDNN 9.10.2 and
cudnn-frontend 1.11.0. Tests use the original BF16 FL2VA and Ref2VA models.
The container has a 91,999,997,952-byte memory limit and no swap.

All development, dependencies, temporary files, CUDA caches and outputs are
under `/root`. Checkpoints are local files in `/path/to/models/MiniMax-H3`;
`/workspace` is not used. Source/builds are in `/path/to/h3.c-5090`, with setup
in `/path/to/h3-5090-env.sh` and the reproducible build wrapper
`/path/to/h3-5090-make.sh`. Local evidence is under
`outputs/cuda-validation/rtx5090`.

The link negotiates **PCIe 5.0 ×8 under load**, despite the card's ×16 maximum.
The measured H2D event throughput is about 26 GiB/s. This matters on a card
that streams weights; these results should not be generalized to a host with
a different PCIe link, CPU memory bandwidth or checkpoint storage.

## Streaming optimization

The old planner selected either all-resident weights or two streaming slots.
The complete 50-block BF16 core needs about 35.9 GiB, plus other weights and
activations. Consequently the 5090 streamed every block on every evaluation
while leaving most VRAM unused. Fast attention alone could not remove this
transfer bottleneck.

`--fast-cuda` now caches a bounded subset of unchanged weight bytes on the GPU.
It copies hits into the existing fenced slots and reserves activation/workspace
headroom. The cache is optional, drops under allocation pressure, and releases
before decoding when needed. `H3_FAST_CUDA_WEIGHT_CACHE_MB=0` disables it;
a positive value caps its size in MiB. Default CUDA and fully resident
execution keep their existing policy. See [cuda-fast.md](cuda-fast.md) for
ownership, file-identity and prepared-cache behavior.

The first matched `inputs/1.jpg` run used 288×384, 56 frames, 20 steps, seed
1001, all 50 layers, reuse/core-reuse one and token reduction off. Times include
loading, inference and MP4 generation; full media validation is separate.

| Mode | Wall time | DiT load + denoise | Streamed DiT weights | Peak DiT device allocations |
| --- | ---: | ---: | ---: | ---: |
| Default CUDA | 122.32 s | 74.33 s | 718.5 GiB | 2.24 GiB |
| Previous fast CUDA | 86.63 s | 65.49 s | 718.5 GiB | 2.24 GiB |
| Fast CUDA with cache | 58.31 s | 35.37 s | 286.8 GiB | 24.92 GiB |

The cache retained 22.68 GiB and served 431.67 GiB of slot uploads from device
memory. This single matched run is about 1.49× faster than previous fast mode
and 2.10× faster than default. It is not a prediction for long production
renders. Original binaries, commands, hashes and profiles are retained in the
evidence directory; later qualification records carry their own binary hashes.

Additional matched pairs (default → cached fast, seconds): `2.jpg` second seed
128.02 → 56.01; multiple images (`2.jpg` + `body1.jpg`) 124.77 → 55.71;
480×640/22-frame detail 123.81 → 55.66; 90-frame continuation segment one
140.24 → 61.57 and segment two 141.19 → 61.57. The second segment delivers
51 new frames after its 39-frame overlap. Fast bridge, changed-reference and
default-to-fast AV-state handoff runs completed in 64.17, 61.52 and 61.57 seconds.
All those clips use 20 steps; two-step functional smoke clips are kept in a
separate folder and are not quality or performance comparisons.

The final binary repeated the pilot in **56.61 s**. Enabling the existing
`H3_CUDA_REGISTER_WEIGHTS=1` option took **56.41 s**, an insignificant difference
in a single pair, so normal bounded uploads remain the default. Both runs
retained the same cache size and streamed the same 286.8 GiB. The final binary
SHA-256 is `2e389978bc272fa29c898a83e85f8125caf33d9657827911b9390e8b69ffa421`.

Isolated attention used one warmup and three timed iterations, with a
120-second cap per case. For BF16, 56 heads, head dimension 128, ordinary
output layout:

| Tokens | Default | Native fast | cuDNN fast |
| ---: | ---: | ---: | ---: |
| 2,281 | 5.813 ms | 1.164 ms | 0.679 ms |
| 18,225 | 316.336 ms | 69.302 ms | 40.382 ms |

The cuDNN-first policy remains appropriate on this 5090; native attention
remains the fallback. Tests also cover head-major layout, tail shapes,
unsupported head dimensions and the decoder attention candidates.
Additional isolated production-scale shapes pass: native/cuDNN attention
takes 220.04/130.21 ms at 32,768 tokens and 2,487.58/1,487.45 ms at 110,592
tokens. These are kernel timings, not full-render timings or long-form quality
evidence. The build without cuDNN also passes its fast-dispatch and weight-cache
runtime tests.

## Validation scope

The suite uses bounded short renders and component tests, not `test2.sh` or
362-frame/50-step production generations. The round ledger includes the initial
pilot and the cache rerender; renders are capped at 300 seconds and the total
render round at 2,700 seconds. Qualification results and exact commands are
recorded alongside the output files.
The completed ledger contains 36 invocations totaling **2,422.78 seconds
(40.38 minutes)**, including pilots, functional checkpoint/caching runs and the
upload comparison. Component tests and isolated kernels are separate from the
render ledger. No long production render was launched.

The default released-model suite passed all 23 tensor comparisons across
vision, text, video encoder, audio encoder/decoder, video decoder, one DiT
block and a complete small DiT evaluation. Default primitive parity, tokenizer,
host/state/CLI tests, fast attention/decoder/QKV checks and asynchronous runtime
tests also passed.

Both 25- and 50-layer memory tests passed in default and fast modes. The
25-layer cases exercise resident, forced streaming, automatic resident and
injected allocation-failure fallback. The 50-layer cases explicitly record
the expected resident-capacity rejection and verify forced/automatic streaming
plus an injected 3 GiB allocation budget. Rejecting an impossible full-resident
request is successful capacity handling, not a resident execution pass.

Cache regressions exercise repeated asynchronous slot generations,
cancellation, bounded admission without cyclic eviction, file replacement,
observable metadata changes, explicit phase release/refill and allocation
pressure. Checkpoints must remain immutable during inference; edits that
preserve metadata are outside the cache contract.
CUDA Compute Sanitizer memcheck with full leak checking reports zero errors
and zero leaked allocations for this regression. The registered-upload option
passes the same test. The final Mac build links all 127 GPU API declarations;
Metal runtime/BF16 primitives and shared host, CLI, sampler and prepared-cache
regressions pass with the Metal no-op cache-release implementation.

The 20-step ordinary and continuation checkpoints pass fresh-process restart;
explicit fast-to-default sampler handoff also passes. Fast/default/fast cache
switching, prepared/decoder reuse, reuse/core-reuse/token-reduction options,
device-memory pressure pass. Reference execution
and complete media decoding pass for T2VA, first/last/both frames, multiple
images, ordinary/silent/replaced-audio video and standalone audio. The latter
reference-mode sweep is two-step smoke evidence; its outputs are not used to
judge quality. Inputs include `1.jpg`, `2.jpg`, `face1.jpg`, `face2.jpg` and
`body1.jpg`.

On 2026-09-18, the user accepted the visual quality of this corpus and the
3090 outputs: "both 3090 and 5090 videos look good". The
[acceptance record](cuda-fast-acceptance.json) records this hardware follow-up
separately from the prior PRO 6000 decision. The local
`outputs/cuda-validation/rtx5090/quality-acceptance.json` snapshots existing
artifacts by SHA-256. Per-clip playback/listening fields and scores retain
their original values because those details were not separately supplied.
Two-step outputs remain functional evidence only.

Local review pages are `outputs/cuda-validation/rtx5090/quality/review.html`,
`functional/review.html` and `smoke/review.html`. The smoke page is explicitly
functional-only. The final two-step checkpoint/restart test is in
`final-resume`; the earlier 20-step checkpoints and their matching binaries
remain on the server under `functional` and `checkpoint-build`. Checkpoint
build-identity validation remains strict across source changes.

## Running on this server

```sh
source /path/to/h3-5090-env.sh
cd /path/to/h3.c-5090
/path/to/h3-5090-make.sh -j8 all
./bin/h3cli -d /path/to/models/MiniMax-H3 --fast-cuda \
  --ref-image inputs/1.jpg --width 288 --height 384 --frames 56 \
  --steps 20 --layers 50 --reuse 1 --core-reuse 1 --seed 1001 \
  -p 'The woman in <Picture 1> walks through a sunlit forest.' \
  --profile -o /path/to/h3.c-5090/outputs/preview.mp4
```

Leave weight mode on `auto`. Forcing `resident` cannot fit all 50 BF16 layers
on this card. The build includes optional cuDNN and OpenSSL acceleration; the
model files have not been quantized or modified.

RTX 3090/SM86 and H100/SM90 were unavailable during this round. The subsequent
[3090 qualification](cuda-3090-qualification.md) closes T107; the later
[H100 qualification](cuda-h100-qualification.md) records T109/T114 work.
The previously qualified PRO 6000 and this 5090 share
SM120 model semantics; memory capacity changes residency and speed.
