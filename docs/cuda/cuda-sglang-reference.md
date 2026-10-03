# Default CUDA reference rendering

CUDA video uses one native C/CUDA implementation of the pinned SGLang arithmetic.
Backend and operation derive that recipe; there is no public mode selector or
legacy video path. Attention, precision, cache/reuse and placement remain
explicit controls. The historical qualification reports linked below retain
past evidence; current acceptance is the complete recorded gate in
[CONTRIBUTING.md](../../CONTRIBUTING.md).

## Build and runtime

The original qualification used RTX PRO 5000 72GB, SM120, with CUDA 12.8
Update 1. [CUDA 13.0.3 also passes all 204 recorded hashes on SM120](cuda-13-qualification.md).
The oracle remains SGLang 0.5.20 / PyTorch 2.13.0+cu130. Python runs the recorded regression harness; native rendering does not invoke
Torch or a live SGLang oracle.

For a new Ubuntu 24.04 x86-64 installation, use the complete setup workflow:

```sh
bash scripts/setup_linux.sh
source outputs/setup/linux-env.sh
./bin/h3cli --help
```

Setup now builds with **CUDA 13.0 Update 3 / 13.0.3 (`nvcc V13.0.88`)**. The default pipeline
also needs **cuBLAS 13.1.1.3** (API identity `130101`) and **NVRTC 13.0.88** at
runtime; setup requires NVIDIA driver 580.126.20 or newer, matching Update 3.
The CUDA version shown by `nvidia-smi` does not select the compiler.

The script installs and exports `SGLANG_CUTLASS_PATH` at revision
`da5e086dab31d63815acafdac9a9c5893b1c69e2`, plus cuDNN frontend `v1.11.0`.
The build verifies the retained CUTLASS source manifest and uses precise math
flags for reference FlashAttention. Defaults are `CUDA_SGLANG=1 CUDA_CUDNN=1`;
minimal diagnostic builds cannot render ordinary CUDA video.

Encoding and decoding run in the `h3cli` process with the same cuDNN
**9.20.0.48** runtime. The single private runtime environment preserves the
provider's system packages and contains only CUDA 13 libraries.
The [shared-runtime qualification](cudnn-920-qualification.md) passes all 204
recorded hashes, encoder cancellation/recovery, and a conditioned CLI comparison.
The generated environment selects the exact cuBLAS and cuDNN libraries,
TurboJPEG **2.1.5**, and production FFmpeg/FFprobe **9.0.2**. Only the immutable
regression gate selects input FFmpeg/FFprobe **6.1.1** and output FFmpeg **4.2.2**
from `imageio-ffmpeg==0.5.1`. Install these isolated test tools with
`python3 scripts/setup_reference_media.py`; the gate verifies their profile and
content hashes before and after the run. Changing a codec can change recorded
hashes even when the CUDA toolkit version is correct. Normal inference, servers
and current-feature tests use 9.0.2.

Production MP4 delivery now defaults to CRF 18, explicit BT.709 conversion,
AAC at 192 kbit/s and faststart. The recorded gate privately retains the original
CRF 25, color conversion, audio defaults and MP4 layout. Denoiser and VAE
arithmetic remain unchanged. `--output-quality default` selects SGLang's CRF
value, while the complete historical encoding recipe belongs to the regression
harness. See [output encoding](../features/output-encoding.md) for compression
controls and lossless RGB delivery.

See the [source-build dependency table](../build/source-build.md#ubuntu-2404-lts-x86-64-and-nvidia-cuda)
and the checksum-pinned `scripts/requirements-sglang-runtime.txt`,
`scripts/requirements-reference-media.txt`, and `scripts/linux/lock.json` files. Native
rendering invokes neither Torch nor SGLang Python. Source `linux-env.sh` before
builds, inference, and the complete regression. `outputs/setup/runtime.json`
records version/loading checks; those checks are not a golden-regression pass.

`--steps` continues to count evaluations: use `--steps 50` for C2. SGLang's
equivalent sigma grids contain 7 and 51 points. `--frames 362 --steps 6` is C1.
Sage/SOL, FP8/NVFP4, LoRA and existing reuse/reduction controls require no mode
flag. Default dense BF16 without approximations retains SGLang parity; explicit
approximate choices may change content and are reported side-by-side.

Preview VAE is an independent approximate decoder selection outside full-output
SGLang parity. Full video decoding now always uses the SGLang recipe on CUDA;
`--full-vae-execution` and its alternatives have been removed. See the
[current VAE guide](../preview/fast-vae.md). Still/image decoding remains separate.

The library derives policy without modifying the caller's request. GPU contexts
capture the resolved arithmetic, attention and precision recipe. Saved files
use the [current-only schemas](../features/current-state-contract.md); old files
and missing presentation metadata are rejected.

## Precision and memory

Text and transformer execution preserve the oracle's BF16 storage and
FP32-sensitive boundaries, CPU Torch-compatible random draws, exact dense
attention, CUDA rotary trigonometry and ordered Euler updates. The original
full video VAE uses the upstream range-safe FP16 execution with sensitive
FP32 accumulation; the full audio VAE follows its separately audited FP32
convolution and activation recipe. No preview decoder or low-bit weights
are substituted.

Reference DiT host weight staging has a process-wide 40 GiB cap and uses
metadata identities. Failed optional admission falls back to bounded streaming.
Automatic BF16 placement retains the full core when admitted, otherwise a safe
leading block set; only the remaining matrices use host staging and two GPU
slots. Packed precision retains its resident/streamed choice. See the
[placement contract](weight-residency-design.md) for admission and lifetime rules.
Component contexts own their plans and scratch, and release inactive weights.
The video decoder keeps at most 256 MiB of spatial tile outputs, 128 MiB of
GPU stitching storage and 128 MiB of pinned readback; larger geometries use
the bounded CPU stitch path. Temporal CNN workspace is capped at 2 GiB. The earlier
17-frame encoder replay measured about 6.3 GiB peak device tensors; it used the
now-retired process isolation, so this is a historical measurement.
The existing process memory safety limit is unchanged.

Additional context-owned bounds are 32 MiB for the reference cuBLAS workspace
(Lt searches use at most 1 MiB of it), 512 MiB for audio convolution workspace,
64 MiB for each audio layout or math-attention buffer, and 1 GiB for each temporal encoder layout
buffer. VAE conversion scratch is capped at 256 MiB per tensor. These are
individual caps, not a claim that all buffers are simultaneously live.
The report distinguishes live tensor bytes, pinned host bytes, process RSS/swap,
and total device usage including driver/library overhead.

DiT patch projection also keeps its FP32 staging buffer within 1 GiB. Large
video/reference layouts are projected in independent row batches, then scattered
to BF16 output. At width 5,376, batches contain at most 49,920 rows. The cuBLAS
algorithm is selected using the original full row geometry and the reduction
dimension is unchanged. This fixes the former hard rejection of 1344×768 /
362-frame requests, whose generated-video projection alone needed 2.16 GiB
when materialized as one FP32 tensor. The existing memory safety cap is unchanged.
The [projection regression results](../../outputs/cuda-sglang/patch-batching/result.json)
cover that geometry with and without mapped reference rows; they do not claim
full-render quality qualification at this resolution.

Native generation timing includes lazy weight loading, conditioning,
denoising, full decoding and media delivery. SGLang loads its CPU-offloaded
weights before its reported generation region. Reports retain that difference
and show complete fresh-process time separately. Native loading is not
subtracted from generation to obtain a performance pass.

## State compatibility and measured scope

The reference arithmetic identity remains `H3_SGLANG_VERSION=4`. New fast uses
`H3_FAST_CUDA_VERSION=2` because preparation and sampling now share that base;
fast-v1 checkpoints must not be reinterpreted. Existing state sections can encode
both identities, with independent validation. Reference identities 1–3 and
legacy CUDA mid-sampler execution are rejected; restart rather than changing
arithmetic during resume.

Existing same-build/device/environment checks still apply. Decoder choice does
not change denoising state. Clean SGLang AV states restore their decoder identity
and may be delivered with explicit preview/FP32/balanced choices. Full legacy
container/continuation migration is M5; an archived old executable remains a
historical conversion aid, not an implicit video execution fallback.

Primary C0/C1/C2 and three held-out six-evaluation cases pass content gates;
C2 includes all 50 free-running and teacher-forced updates. First-frame,
last-frame, both-frame and one-image Ref2VA cases pass their complete
trajectories and media gates. Mixed image/video Ref2VA also passes all six
bit-identical video/audio updates and every decoded-media gate. No equivalent SGLang claim is made for h3-specific continuation
policies, 362-frame/50-evaluation rendering, other geometries/schedules, or
another GPU. Accepted speed/memory limitations remain visible; acceptance is
not a measurement of joint speed/VRAM parity.

## Mandatory short regression gate

Follow [CONTRIBUTING.md](../../CONTRIBUTING.md): run the complete M0 suite after
every code patch, preserving recorded fixtures and goldens. It includes a complete six-evaluation
C0 render, native media preparation, late C2 operator fixtures and exact decoded
pixel/audio checks. Use `make test-cuda-reference-regression` with a fresh output
directory, the bundled input fixtures and recorded golden hashes. The hard test limit is
12 minutes after build. See [M0/M1 evidence](two-modes-m0-m1.md).

## Reproduce the historical full qualification

The frozen cases and limits are in `tests/cuda_sglang_contract.json`.
`tests/cuda_sglang_campaign.py` runs three serial interleaved pairs for every
primary case with operator tracing disabled. The historical
`tests/cuda_sglang_fast_campaign.py` compares the retained pre-change executable
against the candidate for BF16/full-VAE and NVFP4/preview-VAE production defaults;
its raw sampler comparison uses an explicitly labeled CPU-sampler diagnostic.
Reference startup avoids full-checkpoint hashing. Historical fast step captures avoided
legacy AV-state export and its full-checkpoint scans; fast preview's existing
tiny-model digest is preserved.
The full fast outputs differ; they are retained under the user's explicit
acceptance of behavior changes, and are not labeled unchanged. The separate
`cuda_sglang_fast_repair.py` repeats only the documented missing-preview-file
setup failure, retaining the original failed attempts and all successful runs.

`tests/cuda_sglang_closeout.py` audits the frozen numerical, timing, memory and
fast-regression records without loosening failed gates.
`tests/cuda_sglang_report.py` builds the local video/audio inspection pages,
including selected final pairs when `inspection-selection.json` is present.
`tests/cuda_sglang_artifacts.py` then checks their links, checksums, exact frame
counts and complete media decoding. Failed attempts remain visible. An MP4 or
a successful process exit does not by itself pass numerical, timing or memory
qualification. Run GPU campaigns serially on an otherwise idle node.
