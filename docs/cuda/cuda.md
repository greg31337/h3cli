> Historical renderer/qualification record. Commands that refer to removed
> fast/legacy modes require the archived source. Use the
> [current single-pipeline design](design-single-pipeline.md) and
> [M6/M7 qualification](single-pipeline-final.md) for supported execution.

# CUDA backend

The initial contract is Linux x86-64, one SM86-or-newer NVIDIA GPU, the original
BF16 checkpoint and the shared H3 model/sampler/continuation implementation.
FP8, INT8, NVFP4, GGUF and multi-GPU/NCCL execution are deferred. Metal remains
the macOS backend. CUDA parity is numerical, not a promise of bitwise identity
between GPU families. State handoffs must preserve canonical stored inputs exactly.
Completed tests and numerical/hardware limits are recorded in
[CUDA qualification](cuda-qualification.md).

Optional [SageAttention CUDA policies](attention.md) add native Sage2++ and
Sage3 attention for SM120. Their validation and runtime dependencies are
separate from [denoiser projection quantization](denoiser-quantization.md).

Build with CUDA runtime, cuBLAS/cuBLASLt, GCC/G++, ICU and json-c development
packages. Python is used by test/reporting helpers only, never inference.
`make CUDA_ARCH=auto` selects the visible device's architecture. Explicit values
are `86`, `89`, `90`, `100`, `120`; `make cuda-fat` requires CUDA 12.8 or newer and emits
all five architectures plus compute_86 PTX. SM100 selects B200 machine code;
it is distinct from SM120 and does not itself establish B200 hardware qualification.
Use `CUDA_PATH` for a nonstandard
installation. `CPPFLAGS` and `LDFLAGS` can locate dependencies outside system paths.

Remote helpers require SSH and rsync and accept `H3_CUDA_HOST`, optional
`H3_CUDA_PORT`/`H3_CUDA_SSH_KEY`, `H3_CUDA_DIR` (default `/workspace/bin/h3cli`), and
`H3_MODEL_DIR` (default `/workspace/models/MiniMax-H3`). These generic workspace
defaults are configurable; they do not identify a private host. The helpers do not copy models.
Use `H3_CUDA_CPPFLAGS` and `H3_CUDA_LDFLAGS` to pass dependency-prefix flags
to the remote Make invocation.
Temporary files and compiler caches default to `H3_CUDA_DIR`; use
`H3_CUDA_STORAGE_DIR` to choose another persistent directory. Both paths should
be absolute remote paths, since a quoted `~` is not expanded by the remote shell.
For a server using its home directory, for example:

```sh
export H3_CUDA_DIR=/path/to/bin/h3cli
export H3_MODEL_DIR=/path/to/models/MiniMax-H3
scripts/cuda_remote_probe.sh
scripts/cuda_remote_test.sh cuda-test
```

Run `scripts/cuda_remote_probe.sh` before the sync/test helpers. Returned logs
are placed in `outputs/cuda-validation/remote` on the development Mac.

The RTX 4090 qualification covers SM89; the
[RTX PRO 6000 qualification](cuda-pro6000-qualification.md) covers generic and
tuned SM120 execution. The [RTX 5090 qualification](cuda-5090-qualification.md)
adds SM120 with 32 GB VRAM and fast-mode partial weight caching.
The [RTX 3090 qualification](cuda-3090-qualification.md) covers SM86, 24-GB
streaming, fast/native/cuDNN execution, continuation and checkpoint restart.
The [H100 qualification](cuda-h100-qualification.md) adds SM90, full resident
weights, generic parity, fast execution and continuation/checkpoint restart.
The [H200 qualification](cuda-h200-qualification.md) records SM90 execution
with 139.8 GiB available VRAM, bounded comparisons and user-accepted video/audio
quality for the six matched pairs.
The [B200 qualification](cuda-b200-qualification.md) adds actual SM100
execution with 178.35 GiB visible VRAM, native/no-cuDNN/fat/project-PTX tests,
default regressions, all reference families and six bounded quality pairs.
The user separately accepted those six pairs' video/audio quality on 2026-09-18.
Compiling fat binaries is not hardware testing.

## Build and run

On Linux, install a CUDA toolkit (12.8+ for the complete fat target), a C/C++
compiler, Make, ICU development headers, json-c development headers, and FFmpeg.
The inference executable links only native libraries; Python 3 is needed for
the test drivers. For a persistent dependency prefix, for example:

```sh
export TMPDIR=/path/to/tmp
export XDG_CACHE_HOME=/path/to/.cache
export CUDA_CACHE_PATH=/path/to/.cache/cuda
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH"
make CUDA_ARCH=auto \
  CPPFLAGS='-I/path/to/deps/sysroot/usr/include -I/usr/local/cuda/include' \
  LDFLAGS='-L/path/to/deps/sysroot/usr/lib/x86_64-linux-gnu'
./bin/h3cli --info -d /path/to/models/MiniMax-H3
./bin/h3cli -d /path/to/models/MiniMax-H3 --cuda-device 0 \
  --cuda-weight-mode auto --ref-image inputs/2.jpg \
  --layers 50 --reuse 1 --core-reuse 1 --steps 20 \
  --width 864 --height 480 --frames 141 \
  -p 'The woman smiles and turns toward the camera.' -o output.mp4
```

`H3_CUDA_DEVICE` and `H3_CUDA_WEIGHT_MODE` provide the same selections through
the environment. Device selection precedes all GPU allocations. `auto` retains
the whole BF16 core when it fits; otherwise it keeps the largest safe leading
set of active blocks and streams the remainder through two slots. `resident`
requires the full eligible core and reports insufficient capacity before core
loading. `stream` retains no core matrices. Explicit `--ssd-streaming` selects
streaming and conflicts with `resident`.

Admission accounts for the actual packed sequence, already-live allocations,
future activations and scratch, two slots when needed, and a reserve of
`max(1 GiB, 10% of VRAM)`. All 50 BF16 core blocks use 35.89 GiB before activations,
so a 32-GB card needs partial residency or streaming, depending on headroom.
Automatic allocation failures reduce the
resident count and retry initialization down to zero; explicit resident requests
remain strict. FP8/NVFP4 select their existing full-resident or packed-streamed
paths. Placement does not change arithmetic or saved-state compatibility.

Only streamed BF16 matrices enter the optional pinned host cache. Its existing
40 GiB cap, host-memory reserve and bounded bounce-buffer fallback remain in
force. Resident counts depend on geometry and other live GPU allocations;
startup diagnostics show the admitted count and byte budget. Retained partial
automatic contexts replan on the next request. Decoder headroom checks can evict
the DiT while the full video decoder remains resident across tiles.

`make test-weight-residency` tests admission boundaries without a GPU.
`H3_MODEL_DIR=/path/to/MiniMax-H3 make cuda-memory-test` exercises the production shared path with 25 active
blocks, full/partial/stream placement, repeated slot generations and injected
allocation failures. Set `H3_TEST_MEMORY_LAYERS=50` for all blocks, including an
expected strict-resident rejection on smaller cards. Comparisons are bitwise.
See the [placement contract](weight-residency-design.md) and
[hardware qualification](weight-residency-results.md).

The portable attention implementation uses tiled online softmax and F32
accumulation. Large noncausal BF16 attention reuses key/value tiles across 16
queries and uses BF16 tensor-core products with F32 accumulation for QK;
softmax and value accumulation stay F32. Short sequences, causal attention,
and reference mode retain the scalar online-softmax implementation. Qwen
retains its explicit scale-mode contract. Noncausal BF16
attention rounds the scale constant to BF16, matching the Metal SDPA contract.
`H3_CUDA_REFERENCE=1` selects the portable dispatch and pedantic cuBLAS compute.
TF32 and lower-precision weight substitutions are not enabled.

SM90 and SM120 specialize noncausal BF16 attention with 128-wide heads and at least
512 tokens. Fixed-size shared tiles and register accumulators preserve the
portable tiled kernel's arithmetic order. H100 uses 512 threads per block;
SM120 keeps its existing 128-thread specialization. Other architectures and head sizes
retain their existing paths; reference mode disables the specialization.
`cuda-sm90-test` and `cuda-sm120-test` check exact BF16 outputs against the
unchanged portable tiled kernel in both layouts, including tail tiles and
batches, and skip devices outside their target architecture. Set
`H3_TEST_FIXED128_BENCH=1` to include the 2,281- and 18,225-token, 56-head
DiT benchmarks (one warmup and three timed iterations).
`H3_TEST_FIXED128_QUICK=1` limits the direct-kernel cases to at most 513 tokens
for sanitizer runs; it does not replace the ordinary qualification test.

Large VideoVAE encoder convolutions use F32 cuBLASLt GEMMs with at most 32 MiB
of temporary packed input. Small convolutions retain the direct kernel because
packing/dispatch can cost more than their computation. Reference mode retains
the direct convolution at every size; `cuda-conv-test` compares both paths,
including batch, stride, non-square shapes, and a representative large layer.

Weight uploads use two bounded pinned buffers by default. Set
`H3_CUDA_REGISTER_WEIGHTS=1` to try read-only file mapping/host registration;
unsupported registrations fall back to the pinned buffers. Per-tensor ready
and release events order uploads against the last compute-stream use of each
slot. No complete target latent is copied to the host between ordinary CUDA
sampler transitions; explicitly requested callbacks/checkpoints do read back
canonical F32 state.
CUDA uses device-resident sampler state by default for both ordinary generation
and continuation. `H3_CPU_SAMPLER=1` remains an explicit diagnostic override.
Metal retains its existing defaults, and checkpoint restart honors the stored
sampler mode on either backend.

## Long AudioVAE decode

Alias-free SnakeBeta splits its CUDA time-axis dispatch into at most 65,535
blocks per launch, retaining global sample coordinates and full-input filter
access across launch boundaries. The former single launch failed beyond
262,140 samples (about 8.19 seconds at 32 kHz), including `test2.sh`'s
482,400-sample audio. This fix applies to both default and reference CUDA modes.

`bin/audio_gpu_tests` covers the launch boundary, stereo input, partial tiles,
multiple launches, and samples on both sides of every split. It is included in
`make test`. To exercise all seven decoder stages at the failing duration with
released weights and synthetic latents, without denoising or video rendering:

```sh
make bin/audio_long_test
./bin/audio_long_test /path/to/models/MiniMax-H3
```

This checks completion, sample count, progress, and finite stereo PCM; synthetic
latents do not establish perceptual audio quality. For long generation jobs,
use `--save-sampler-state` as described in [sampler checkpoints](../features/sampler-state.md)
so a decoding failure does not discard completed denoising work.

## Validation tiers

| Target | Coverage | Released models |
| --- | --- | --- |
| `cuda-host-test` | Host API, containers, schedules, resume validation, continuation oracles | No |
| `cuda-test` | All GPU symbols, tensor storage/fences, BF16, Euler, prefix, reduction, audio, GQA, preview | No |
| `cuda-parity` | 41 deterministic Metal primitive fixtures | No |
| `cuda-tokenizer-test` | Exact Metal token IDs/decoded text for both vocabularies | Tokenizers |
| `cuda-real-parity` | Real vision/text, video/audio encoders, VAEs, block and complete DiT evaluation | Yes |
| `cuda-memory-test` | Resident/streamed equivalence and injected allocation-failure fallback | Yes |
| `cuda-conv-test` | Direct-versus-GEMM convolution accuracy and warmed timings | No |
| `cuda-attention-test` | Scalar-versus-tiled attention, layouts, tail tiles, 18,225-token sequences | No |
| `cuda-sm120-test` | Exact portable-versus-SM120 attention and warmed timings; skips other GPUs | No |
| `cuda-sm90-test` | Exact portable-versus-SM90 attention and warmed timings; skips other GPUs | No |
| `cuda-fat-test` | Execute Metal primitive comparisons from the multi-architecture CUDA object | No |
| `cuda-smoke` | 50 blocks, 128×128, 22 frames, two steps, reference image and synchronized audio | Yes |
| `cuda-features` | All reference modes, continuation, restart, reuse/core-reuse/reduction | Yes |
| `cuda-full-test` | 864×480/141-frame/20-step Ref2VA, 141/39 continuation, 20-step restart | Yes |
| `cuda-bench` | Fixed 864×480/141-frame/20-step single-image Ref2VA workload | Yes |

Set `H3_MODEL_DIR` to the original model root. `H3_CUDA_OUTPUT` selects the
integration artifact directory. Each integration run writes the command,
source digest, hardware/toolchain metadata, environment, elapsed wall time,
exit status, log, and checked FFprobe stream metadata. Each MP4 must contain
the expected frame count and dimensions plus 32-kHz stereo audio.
Continuation records retain the shared driver's geometry, seed, reference count,
and completed-boundary callback audit under `generation`.

`tests/cuda_integration.py references`, `chain`, `resume`, `cache`, and `optimizations`
allow individual integration groups to be rerun. `H3_CUDA_CASES` restricts the
reference cases (`t2va`, `first`, `last`, `first-last`, `images`, `video`,
`silent`, `video-audio`, `audio`). The chain uses different seeds across four
segments and changes both image references on the fourth segment. The shared
continuation harness verifies raw inherited tails and protected values at every
Euler boundary, as well as trimming delivered frames/audio.
Video fixtures are 2.5 seconds long, within the released pipeline's 2–15-second
range. Video cases request 56 target frames because reference decoding is
capped by the target length. The separate-audio case pairs its WAV file with an image, as required by
the existing Ref2VA API; audio-only references remain rejected on both backends.
The small reference-mode cases use two workers on Linux to overlap model reads;
`H3_TEST_REFERENCE_WORKERS=1` makes them serial. Reports record the worker count.
Full-quality, cache-pressure, continuation and benchmark runs remain serial.

The `cache` group additionally compares every decoded RGB byte on a cached
rerun using streamed weights, reserves enough device memory to leave only 4 GiB free, changes the
prompt, and verifies that cached GPU objects are evicted and generation still
completes. It then releases the reservation and checks checkpoint/restart.
Stream mode bounds the prepared cache: evicting resident weights on a large
card could otherwise release tens of GiB and remove the intended pressure.

The resume harness compares every completed transition, including a cached
rerun and a new process. Same-device comparisons require bit-identical F32
latents. Cross-backend comparisons require an exact restored starting boundary,
then measure divergence after subsequent computation. The checkpoint format
does not contain GPU pointers, events or library handles. Cross-backend loading
requires matching source identity, engine version, numerical options and model
content hashes; same-backend loading retains the stricter build checks.
The full/features tiers also pause and restart a continuation with a protected
39-frame prefix. To run that case separately, set
`H3_TEST_CONTINUATION` to an existing `.h3av` and run
`python3 tests/cuda_integration.py continuation-resume`; no sampler override is
set, so this exercises the backend's continuation default.

## Numerical fixtures and profiling

`tests/fixtures/cuda/manifest.json` records fixture hashes, the recording GPU,
shader hash and Metal baseline commit. Only Metal can record these goldens.
`tests/cuda_primitives.c` defines per-operation absolute and relative bounds.
Released vision and DiT tests use the existing repository's scale-aware bounds
from `test_real_qwen_vision.c` and `test_real_dit.c`, respectively. Reports expose
maximum/mean absolute error, relative L2 and cosine similarity, and reject
nonfinite values and mismatched element counts.

The default complete DiT comparison supplies the same Metal text conditioning
and unit-scale deterministic latent noise to both GPUs. The additional
`dit-stress` component supplies artificial low-amplitude conditioning and
latents; its stricter 5% L2 gate is a diagnostic stress case, not a production
acceptance tolerance. Its observed failures are retained in the qualification
record rather than overwriting its Metal golden or loosening that gate.

`--profile` reports logical device-memory peak, pinned-memory peak, streamed
bytes, source read/upload wall time, CUDA-event H2D/D2H time and bandwidth,
uncovered upload waits, GEMM/attention/convolution and elementwise timing.
The existing shared DiT streaming report also measures host waits for prefetched
blocks. Profiling does not introduce device-wide fences into ordinary compute.
The first sampler checkpoint may additionally hash the complete model tree;
subsequent checks use the existing identity/size/mtime/ctime-validated digest
cache under `outputs/.h3-model-hashes`.

The original `make parity` suite requires external `misc/fixtures` files. If
they are absent it cannot pass, even on the frozen pre-CUDA baseline. The new
checked-in Metal fixtures provide independent CUDA coverage; missing legacy
fixtures are reported explicitly and are not counted as passing tests.

`tests/cuda_analyze.py METAL_BASE CUDA_BASE --output REPORT.json` validates
AV checksums/geometry and finite latents, then compares latent tensors,
downscaled decoded RGB, and decoded stereo PCM. It reports numerical drift,
signal statistics and temporal frame changes without treating those metrics
alone as evidence of perceptual quality. `tests/cuda_metal_regression.py`
compares the frozen pre-port executable against the current Metal executable,
including Ref2VA, a 141/39 continuation and fresh-process sampler restart.

The opt-in [`--fast-cuda` workflow](cuda-fast.md) uses matched original weights
and settings, faster intermediate arithmetic and perceptual video/audio
acceptance instead of numerical parity. Default execution retains the tests
above. Its bounded quality preset is 288×384/56 frames/20 steps, continuation
uses two 90-frame targets with 39-frame overlap, and the detail check is
480×640/22 frames/20 steps. All use 50 layers and reuse/core-reuse one. Smoke
clips are never quality evidence. PRO renders are capped at five minutes each
and 45 minutes for the required round; representative M4 coverage is capped
at ten minutes each and 20 minutes total. See the linked guide for commands,
fallback/resume policy and [measured results](cuda-fast-validation.md).
