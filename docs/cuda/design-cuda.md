# Technical Design: Complete CUDA Backend for h3cli

## 1. Objective

Add a complete Linux/NVIDIA CUDA execution backend to the existing continuation-enabled `h3cli` codebase without replacing, weakening, or behaviorally changing the existing Metal implementation.

The resulting source tree shall support:

```text
macOS / Apple Silicon
    → existing Metal backend

Linux / NVIDIA
    → new CUDA backend
```

The public H3 model logic, CLI behavior, Ref2VA handling, native AV continuation, `.h3av` state, sampler checkpoint/resume, schedules, prompt handling, tensor layouts, and VAE logic should remain shared.

The CUDA backend is an execution backend, not a separate H3 implementation.

Initial CUDA scope:

```text
single NVIDIA GPU
original BF16 model
F32 accumulation where currently required
50-block DiT supported
full Ref2VA functionality
existing continuation functionality
existing stop/resume functionality
Linux x86-64
```

Target GPUs:

```text
RTX 3090       SM86 / Ampere
RTX 4090       SM89 / Ada
H100           SM90 / Hopper
RTX 5090       SM120 / Blackwell
RTX PRO 6000   SM120 / Blackwell
```

H100/H200 can share the same SM90 implementation if H200 support is later required.

The first correct implementation must avoid:

```text
FP8
INT8
NVFP4
GGUF quantization
approximate sparse attention
architecture-dependent model behavior
multi-GPU sequence parallelism
```

Those can be added after BF16 CUDA parity is established.

---

# 2. Primary architectural rule

Keep all H3 semantics above the GPU abstraction.

Conceptually:

```text
                        src/engine.c
                            |
          +-----------------+------------------+
          |                 |                  |
      Ref2VA           continuation        sampler
                         .h3av            .h3sample
          |                 |                  |
          +-----------------+------------------+
                            |
                         src/gpu.h
                      /             \
                     /               \
               Metal backend       CUDA backend
                    |                   |
             src/metal/gpu.m              src/cuda/gpu_cuda.cu
             MPSGraph              cuBLASLt
             Metal kernels         CUDA kernels
```

The CUDA backend must not contain H3-specific policy such as:

```text
39-frame continuation selection
65-tick audio overlap selection
prompt/reference semantics
seed policy
sigma schedule construction
continuation state serialization
sampler-state serialization
```

Those remain in shared C.

The backend should receive already-resolved tensors, row maps, shapes, offsets and schedules.

---

# 3. Preserve the Metal build

The current Darwin build remains the reference implementation.

On macOS:

```text
src/metal/metal.m
src/metal/gpu.m
src/conditioning/tokenizer.m
Metal
MPS
MPSGraph
Accelerate
```

continue to build as they do now.

Do not introduce:

```text
CUDA headers
CUDA libraries
Linux portability shims
#ifdef CUDA
```

throughout the H3 model code.

Platform selection occurs only in the build system and small platform-abstraction modules.

The current upstream build already cleanly separates common C sources from the Metal-specific Objective-C sources, making this division practical.

---

# 4. Linux/CUDA source layout

Add approximately:

```text
src/cuda/cuda_probe.c
src/cuda/gpu_cuda.cu
src/cuda/cuda_kernels.cuh
src/cuda/cuda_attention.cuh
src/cuda/cuda_conv.cuh

src/conditioning/tokenizer_portable.c

optional:
    third_party/utf8proc/
```

Shared model files remain unchanged wherever possible:

```text
src/engine.c
src/host.c
src/denoise/dit.c
src/denoise/dit_schedule.c
src/conditioning/text_encoder.c
src/conditioning/vision_encoder.c
src/conditioning/multimodal.c
src/vae/video_encoder.c
src/vae/video_vae.c
src/vae/audio_vae.c
src/weights/weights.c
src/weights/safetensors.c
src/media/ffmpeg.c
```

This follows an architecture already demonstrated by an H3 CUDA fork, which selects CUDA platform objects on Linux while retaining the same common C model implementation.

---

# 5. Generic device abstraction

Remove direct model-level dependency on:

```c
#include "src/metal/metal.h"
```

where it is used only for device discovery.

Introduce a platform-neutral interface such as:

```c
src/device.h
```

with:

```c
int h3_device_query(h3_device_info *info,
                    char *error,
                    size_t error_size);
```

Implementations:

```text
Darwin:
    src/metal/metal.m

Linux:
    src/cuda/cuda_probe.c
```

Preserve existing `h3_device_info` fields and append CUDA information rather than deleting/changing existing public fields.

Possible additions:

```text
backend
device_index

cuda_compute_major
cuda_compute_minor

cuda_driver_version
cuda_runtime_version

device_memory
free_device_memory

multiprocessor_count
```

Metal fields retain their current meaning.

---

# 6. CUDA build selection

Default behavior:

```text
Darwin
    → Metal build

Linux
    → CUDA build
```

Linux CUDA dependencies:

```text
CUDA runtime
cuBLAS
cuBLASLt
libstdc++
libm
FFmpeg
```

Avoid requiring:

```text
PyTorch
Python
SGLang
ComfyUI
CUTLASS
cuDNN
```

for the initial implementation.

Those may later become optional acceleration dependencies.

---

# 7. Unknown-GPU build handling

Because we do not know which remote GPU is available ahead of time, support:

```text
CUDA_ARCH=auto
```

At build time detect the local NVIDIA device and map:

```text
8.6  → 86
8.9  → 89
9.0  → 90
12.0 → 120
```

Example:

```bash
make CUDA_ARCH=auto
```

should print:

```text
Detected CUDA device: NVIDIA RTX 5090
Compute capability: 12.0
Building for: sm_120
```

Also support explicit:

```bash
make CUDA_ARCH=86
make CUDA_ARCH=89
make CUDA_ARCH=90
make CUDA_ARCH=120
```

For development, compile only for the currently rented GPU.

---

# 8. Universal CUDA build

Later add:

```bash
make cuda-fat
```

which contains SASS for:

```text
sm_86
sm_89
sm_90
sm_120
```

plus an appropriate PTX fallback.

That binary should run on all currently targeted GPUs.

Because Blackwell SM120 support requires a sufficiently recent CUDA toolkit, require CUDA 12.8 or newer for `cuda-fat`.

A target-specific build may continue to work with an older toolkit if that toolkit supports the selected GPU.

---

# 9. Runtime GPU detection

At startup query:

```c
cudaGetDeviceProperties()
cudaMemGetInfo()
```

and report:

```text
Backend: CUDA
Device: NVIDIA GeForce RTX 4090
Compute capability: 8.9
VRAM: 24 GiB
CUDA runtime: ...
Driver: ...

Precision: BF16
Weight mode: streaming
Attention: portable BF16
GEMM: cuBLASLt
```

Support:

```text
--cuda-device N
```

and/or:

```text
H3_CUDA_DEVICE=N
```

Default:

```text
device 0
```

The selected device must be established before constructing any CUDA context or allocating GPU storage.

---

# 10. GPU-family independence

There must be one CUDA model implementation.

Do not create:

```text
h3_3090.cu
h3_4090.cu
h3_h100.cu
h3_5090.cu
```

Instead:

```text
                    common CUDA implementation
                               |
                 optional optimized dispatch
                 /           |          \
              SM86         SM90        SM120
```

The generic BF16 path must work on all supported architectures.

Architecture-specific paths are optional optimizations and must produce numerically equivalent results within established tolerances.

---

# 11. CUDA context

Map the existing GPU abstraction approximately as follows:

```text
Metal                         CUDA

MTLDevice                  → CUDA device
MTLCommandQueue            → cudaStream_t
MTLCommandBuffer           → ordered CUDA stream work
submit/wait                → cudaStreamSynchronize
MPSGraph GEMM              → cuBLASLt
MPSGraph SDPA              → CUDA online-softmax attention
shared Metal buffer        → managed/device CUDA memory
blit copy                  → cudaMemcpyAsync
```

An existing CUDA H3 implementation has already demonstrated this mapping, including a single CUDA stream, cuBLASLt and tiled online-softmax attention.

---

# 12. CUDA tensor representation

Implement:

```c
struct h3_gpu_tensor {
    void *device;
    void *host;

    size_t elements;
    size_t bytes;

    h3_gpu_dtype dtype;

    h3_gpu *owner;

    storage flags...
};
```

Support three storage classes:

### Host-visible managed tensor

```text
cudaMallocManaged
```

Used where existing shared-Metal semantics require direct host access.

### Device-only tensor

```text
cudaMalloc
```

Used for:

```text
hot activations
scratch
streaming weight slots
Q/K/V
attention workspace
velocity caches
```

### Pinned host tensor

```text
cudaHostAlloc
```

Used for:

```text
asynchronous H2D transfer
weight-streaming bounce buffers
future cross-GPU communication
```

The existing CUDA fork demonstrates that preserving host-visible tensor semantics via managed memory is practical while still using true device-only allocations for high-throughput staging.

---

# 13. Host-access contract

Audit every use of:

```c
h3_gpu_tensor_contents()
```

Shared Metal memory makes direct CPU access inexpensive.

Discrete NVIDIA memory does not.

For performance-sensitive paths, replace unnecessary direct accesses with:

```text
h3_gpu_tensor_read_f32
h3_gpu_tensor_read_f32_range

h3_gpu_tensor_write_f32
h3_gpu_tensor_write_f32_range

BF16 equivalents
```

Keep `h3_gpu_tensor_contents()` valid for host-visible tensors to preserve existing behavior.

Calling it on a CUDA device-only tensor must return NULL or produce a clear contract error.

---

# 14. CUDA command semantics

Implement existing:

```text
h3_gpu_begin()
h3_gpu_continue()
h3_gpu_submit()
```

using CUDA stream semantics.

Recommended mapping:

```text
begin:
    mark command sequence active

continue:
    preserve stream ordering
    optionally mark profiling boundary

submit:
    cudaStreamSynchronize()
    check all deferred CUDA errors
```

Every kernel launch should call:

```c
cudaGetLastError()
```

or equivalent launch validation immediately.

Execution errors discovered during synchronization must propagate through the existing `h3_gpu_error()` path.

---

# 15. Full-quality BF16 policy

Initial CUDA integration targets the same quality regime we use on Metal:

```text
original BF16 checkpoint
BF16 weight storage
BF16 activation storage where applicable
F32 accumulation where required
F32 normalization/softmax calculations where appropriate
```

Do not silently enable:

```text
TF32 replacement for F32 GEMMs
FP8
FP16
INT8
NVFP4
```

The CUDA implementation should use:

```text
BF16 inputs
F32 accumulation
BF16 outputs
```

for the primary transformer GEMMs.

The existing GPU contract explicitly describes the portable BF16 path as F32 accumulation with rounding at operation boundaries.

---

# 16. cuBLASLt backend

Use cuBLASLt for large:

```text
BF16 × BF16 → BF16
F32 × F32 → F32
```

matrix operations.

Implement an algorithm cache keyed by:

```text
M
N
K
dtype
transpose flags
bias
batch
leading dimensions
```

On first use:

```text
query heuristics
select valid algorithm
cache result
```

Do not re-run heuristic selection for every transformer block.

Provide a deterministic/reference configuration that restricts algorithm selection when investigating Metal/CUDA discrepancies.

---

# 17. Small matrix operations

For matrices too small for efficient cuBLASLt dispatch, provide simple CUDA tiled kernels.

The correctness path takes priority over aggressive specialization.

Thresholds may be selected empirically per architecture later.

---

# 18. CUDA attention

The initial attention implementation must not materialize the complete:

```text
sequence × sequence
```

attention matrix.

Implement tiled online softmax / FlashAttention-style processing:

```text
load Q block
iterate K/V blocks
maintain running max
maintain running denominator
accumulate output
```

Requirements:

```text
BF16 Q/K/V storage
F32 score/softmax accumulation
BF16 or F32 output according to h3_gpu contract
noncausal H3 DiT attention
causal text attention
GQA support
```

One portable kernel should support SM86+ initially.

Architecture-specific attention can come later.

---

# 19. RoPE and QKV operations

Implement CUDA equivalents for all H3 GPU API operations involving:

```text
QKV split
grouped H3 QKV
Q/K RMS normalization
video RoPE
text RoPE
vision RoPE
head-major conversions
```

Preserve exact existing memory layouts.

Do not alter shared H3 model code to accommodate CUDA-friendly layouts until parity is complete.

If later optimized layouts are introduced, they remain backend-private.

---

# 20. Normalization and elementwise kernels

Implement complete CUDA support for:

```text
RMSNorm
LayerNorm
AdaLN
AdaLN with offsets
gating
gate + AdaLN
SiLU
GELU
SwiGLU
GEGLU
Snake
alias-free Snake
add
subtract
scaled add
clamp
casts
BF16/F32 conversion
copy
embedding lookup
```

All row-map semantics must remain identical to Metal.

This is particularly important for continuation because preserved/generated target rows use distinct modulation classes.

---

# 21. Audio operations

Implement all GPU operations required by:

```text
audio encoder
AudioVAE
audio transformer/pooling
```

including:

```text
Conv1D
strided Conv1D
ConvTranspose1D
weight normalization
attention pooling
causal attention
```

End-to-end CUDA output must contain synchronized H3 audio, not video-only generation.

---

# 22. Video encoder and VAE

Implement CUDA support for:

```text
Conv3D
VAE padding
group normalization
SiLU
upsampling/downsampling
video encoder
video decoder
```

Tiled decode behavior must remain consistent with the existing H3 implementation.

Avoid introducing a separate Python or external VAE runtime.

---

# 23. Convolution strategy

The first implementation may use implicit-GEMM-style convolution:

```text
initialize output/bias
for each kernel tap:
    cuBLASLt GEMM
    accumulate
```

where appropriate.

This avoids introducing cuDNN as a hard dependency.

Custom optimized convolution kernels may be added later after profiling.

---

# 24. Linux tokenizer

The current macOS tokenizer is Objective-C/Apple-platform dependent.

Add a portable C tokenizer implementation for Linux.

It must produce byte-identical:

```text
token IDs
Unicode normalization behavior
special token handling
multimodal token sequences
```

to the existing macOS tokenizer.

A small portable Unicode dependency such as `utf8proc` may be vendored.

The Metal tokenizer stays unchanged.

Existing CUDA H3 work has demonstrated exactly this platform split.

---

# 25. Weight loading

Reuse the existing:

```text
safetensors parser
weight metadata
phase-specific component loading
```

as much as possible.

Do not introduce a separate CUDA checkpoint format.

The same original MiniMax-H3 model directory should work on:

```text
Metal
CUDA
```

without conversion.

---

# 26. VRAM-aware execution

CUDA must support all candidate GPUs, including:

```text
24 GB:
    RTX 3090
    RTX 4090

32 GB:
    RTX 5090

80 GB:
    H100

96 GB:
    RTX PRO 6000
```

Therefore do not assume the complete ~BF16 transformer can reside in VRAM.

Implement:

```text
--cuda-weight-mode auto
--cuda-weight-mode resident
--cuda-weight-mode stream
```

Default:

```text
auto
```

---

# 27. Memory planner

At model-phase initialization:

```text
query free VRAM
estimate:
    required weight storage
    peak activations
    CUDA workspace
    safety reserve
```

If everything safely fits:

```text
resident mode
```

Otherwise:

```text
streaming mode
```

Never attempt resident allocation until OOM and then leave CUDA in an uncertain state.

If resident allocation still fails, cleanly release partial allocations and retry in streaming mode.

---

# 28. Streaming weights

For 24/32-GB GPUs implement double-buffered BF16 weight streaming:

```text
CPU/model storage

block N+1
      |
      | async H2D
      v
device slot B

while

device slot A
      |
      v
compute block N
```

Use:

```text
copy stream
compute stream

slot-ready event
slot-release event
```

The reader may preload:

```text
slot 0
slot 1
```

and alternate them through transformer layers.

Existing CUDA H3 work uses this exact copy-stream/event architecture.

---

# 29. Host weight source

Support two streaming sources.

Preferred when RAM allows:

```text
mmap checkpoint
register/pin relevant pages
async H2D
```

Fallback:

```text
pread into pinned bounce buffer
async H2D
```

Do not require the entire H3 model to be copied into a second ordinary heap allocation.

---

# 30. Streaming phases

Apply the same generic mechanism where necessary to:

```text
Qwen text encoder
H3 DiT
```

VAEs and smaller components may remain resident when VRAM permits.

Phase transitions should release memory that will no longer be needed, preserving the current h3cli phase-oriented memory model.

---

# 31. GPU-resident sampling

After the CUDA DiT works correctly, keep:

```text
current video latent
current audio latent
last velocity
previous velocity
```

on the CUDA device across denoising passes.

Do not transfer the entire latent to CPU between every pass.

Implement existing Euler semantics with a CUDA kernel such as the `h3_gpu_euler_bf16()` abstraction already used in advanced GPU execution. The current GPU API explicitly supports updating an F32 sample range using BF16 velocity caches.

---

# 32. Native continuation on CUDA

The existing continuation implementation remains authoritative.

CUDA must support:

```text
.h3av load

39 / 90 / 141 ... frame context

video tail copy

audio tail copy

0.999 clean video +
0.001 current noise

clean audio prefix

preserved/generated timestep classes

zero/protected prefix update

trimmed delivered output

new .h3av
```

No separate CUDA continuation algorithm is allowed.

---

# 33. Efficient protected-prefix update

Because the continuation mask is a contiguous temporal prefix, the CUDA Euler path may update only the generated ranges rather than computing:

```text
velocity × mask
```

over the entire target.

However the optimized range update must be proven equivalent to the generic zero-velocity implementation first.

Maintain a diagnostic path that explicitly zeros prefix velocities.

---

# 34. Ref2VA continuation

Support the complete existing combinations:

```text
continuation + ref image
continuation + multiple images
continuation + ref video
continuation + ref video/audio
continuation + standalone ref audio
different references on later segments
```

The continuation latent must remain distinct from Ref2VA reference conditioning exactly as it is on Metal.

---

# 35. Stop-after/resume on CUDA

Implement `.h3sample` loading and saving without changing its canonical format.

Saving a CUDA sampler state requires:

```text
synchronize completed Euler boundary

copy:
    current video latent
    current audio latent
    velocity caches if required

serialize existing conditioning
serialize schedule
serialize continuation metadata
```

Resume performs the inverse.

Do not serialize:

```text
cudaStream_t
CUDA events
cuBLAS handles
device pointers
kernel state
```

Those are rebuilt.

---

# 36. Metal → CUDA sampler-state portability

The file format should remain backend-neutral.

This permits:

```text
Mac Metal:
steps 0..3
    |
    v
.h3sample
    |
    v
CUDA:
steps 4..19
```

The CUDA implementation should restore:

```text
exact F32 AV latent
exact conditioning
exact sigma schedules
exact continuation mask
next_step
```

Small numerical divergence after the first resumed CUDA DiT evaluation is acceptable.

The handoff itself must be exact.

---

# 37. CUDA → Metal portability

Also test the reverse:

```text
CUDA
   ↓
.h3sample
   ↓
Metal
```

This is less important operationally but proves that `.h3sample` remains a genuine backend-neutral representation instead of silently acquiring CUDA dependencies.

---

# 38. Existing optimization flags

The CUDA implementation must define behavior for every existing H3 option.

For the first CUDA release:

```text
layers                 supported
reuse                  supported
core-reuse             supported after validation
token reduction        supported after validation
SSD streaming          mapped to CUDA streaming
cache/session mode     supported
continuation           supported
sampler resume         supported
```

Metal-specific M5 TensorOps/INT8 flags should:

```text
report unsupported
```

rather than silently changing behavior.

The normal BF16 CUDA path remains valid.

---

# 39. Library-context caching

Repeated library requests can retain the following caches on CUDA:

```text
prompt conditioning cache
prepared DiT
video decoder
reference conditioning
```

Retaining cached GPU resources between generations must respect available VRAM.

If caching would exceed the memory planner's safety reserve, fall back to host-side cached state and reload GPU state as needed.

---

# 40. Error handling

Translate all CUDA and cuBLASLt failures into the normal H3 error model.

Errors should identify:

```text
phase
operation
tensor shape
CUDA error
GPU
available VRAM
requested allocation
```

Example:

```text
CUDA DiT block 17: unable to allocate 268435456-byte scratch buffer
device RTX 3090: 1.21 GiB free of 24.00 GiB
```

No CUDA error should call `abort()` for a recoverable allocation or kernel failure.

---

# 41. CUDA profiling

Extend `--profile` to report:

```text
H2D transfer time
D2H transfer time
weight-stream wait
weight-stream bytes

cuBLASLt GEMM time
attention time
normalization time
VAE time

peak VRAM
allocated device bytes
managed bytes
pinned-host bytes

DiT time / block
DiT time / denoise evaluation
```

Retain the existing Metal profiling output unchanged.

---

# 42. Reference CUDA mode

Provide:

```text
H3_CUDA_REFERENCE=1
```

or equivalent.

Reference mode should disable architecture-specific optimized kernels and select the most straightforward BF16 implementation.

Purpose:

```text
correctness debugging
Metal comparison
new-GPU bring-up
regression isolation
```

Production optimization must always be comparable against this path.

---

# 43. Numerical parity policy

Do not require Metal and CUDA to be bit-identical.

Require:

```text
exact shapes
exact layouts
exact schedules
exact token/reference ordering
exact continuation boundaries
exact serialized input state
```

and numerical comparison per operation using established tolerances.

Track:

```text
maximum absolute error
mean absolute error
relative L2 error
cosine similarity
```

For end-to-end outputs additionally inspect:

```text
latent statistics
decoded-frame similarity
audio waveform similarity
semantic/action continuity
```

---

# 44. Golden Metal fixtures

Extend the existing fixture approach.

Generate small reference tensors on the Mac for:

```text
linear
AdaLN
RMSNorm
RoPE
QKV
SDPA
GQA
SwiGLU
Conv1D
Conv3D
AudioVAE
VideoVAE
one real H3 block
one full DiT step
continuation prefix construction
Euler update
```

Copy those fixtures to the remote CUDA node.

CUDA tests consume them without needing to render a complete movie.

Upstream already maintains toy and real-component parity fixtures, making this consistent with the project's existing test philosophy.

---

# 45. Remote development workflow

Add helper scripts such as:

```text
scripts/cuda_remote_probe.sh
scripts/cuda_remote_sync.sh
scripts/cuda_remote_test.sh
scripts/cuda_remote_bench.sh
```

Typical development:

```text
Mac:

edit code
make test
make parity

        ↓ rsync

remote node:

detect GPU
make CUDA_ARCH=auto
make cuda-test
make cuda-real-parity

        ↓

return logs/results
```

The scripts should accept an SSH host rather than containing RunPod-specific APIs.

Example:

```bash
H3_CUDA_HOST=user@cuda.example.com \
    ./scripts/cuda_remote_test.sh
```

---

# 46. Remote environment probe

Before building, collect:

```text
uname
CPU architecture
RAM
GPU model
VRAM
compute capability
driver
CUDA toolkit
nvcc
gcc
FFmpeg
available disk
```

Write this to:

```text
cuda-environment.txt
```

alongside every benchmark.

This is especially important because we do not know which GPU model will be available.

---

# 47. Model storage

Remote scripts must allow the model directory to be configured:

```text
H3_MODEL_DIR=/path/to/MiniMax-H3
```

Do not automatically transfer hundreds of gigabytes over SSH.

Tests that do not require real weights must use fixtures.

Full integration tests should run only if `H3_MODEL_DIR` exists.

---

# 48. CUDA test tiers

Define:

```text
make cuda-test
```

Small synthetic tests, seconds.

```text
make cuda-parity
```

Fixture comparisons, usually seconds/minutes.

```text
make cuda-real-parity
```

Real released component tests.

```text
make cuda-smoke
```

Very small end-to-end H3 render.

```text
make cuda-full-test
```

Normal production-sized BF16 integration test.

```text
make cuda-bench
```

Performance benchmark.

This avoids using expensive GPU time unnecessarily.

---

# 49. GPU architecture qualification

Only one node is needed to develop the implementation.

After functional completion, qualify architectures separately.

Minimum matrix:

```text
SM86:
    RTX 3090

SM89:
    RTX 4090

SM90:
    H100

SM120:
    RTX 5090 OR RTX PRO 6000
```

RTX PRO 6000 does not require another implementation after a successful RTX 5090 SM120 qualification.

Likewise H200 would not require a new implementation after H100 SM90 qualification.

---

# 50. Architecture-specific optimization

Do not begin this work until generic CUDA is correct.

Potential later dispatch:

```text
SM86:
    Ampere attention tuning

SM89:
    Ada occupancy/tile tuning

SM90:
    Hopper-specific attention / memory movement

SM120:
    Blackwell-specific attention / tensor-core path
```

Only the hot CUDA kernels vary.

Shared H3 C logic remains untouched.

---

# 51. Single-GPU scope

The first complete CUDA backend is single GPU.

If the rented node exposes multiple GPUs:

```text
--cuda-device
```

selects one.

Do not make initial correctness depend on:

```text
NCCL
P2P
NVLink
sequence parallelism
```

Multi-GPU can be added later behind the same GPU abstraction.

Existing CUDA forks already show that sequence-parallel interfaces can be layered on top without affecting the Metal single-device path, so preserving that extension point is sufficient for now.

---

# 52. Backward compatibility

Completion requires:

```text
macOS make
macOS make test
macOS make parity
```

to behave exactly as before.

Existing:

```text
CLI options
public C API
Metal environment variables
Metal cache behavior
Metal output
continuation
.h3av
.h3sample
```

must remain operational.

No CUDA-specific source must be compiled into the Darwin binary unless explicitly needed for shared declarations.

---

# 53. CUDA acceptance criteria

CUDA integration is complete when a Linux NVIDIA system can perform:

```text
T2VA
FL2VA first frame
FL2VA last frame

Ref2VA image
Ref2VA multiple images
Ref2VA video
Ref2VA silent video
Ref2VA video + audio
Ref2VA standalone audio

full audio/video VAE
MP4 muxing

native masked AV continuation
three-segment continuation
changing Ref2VA references per segment

.h3av save/load

stop-after
.h3sample save
process restart
resume
Metal → CUDA .h3sample handoff
```

using the original BF16 checkpoint without quantization.

The generic implementation must work on at least one available NVIDIA node before GPU-family optimization begins.

---

# 54. Performance acceptance

Correctness is the first release gate.

After correctness, require that:

```text
no full target latent crosses PCIe per transformer block

no full target latent crosses PCIe per denoising step

weight streaming overlaps compute where possible

DiT activations remain device-resident

sampler latent remains device-resident

VAE decoding does not introduce unnecessary host copies
```

On 24/32-GB cards the expected dominant transfers should be model weights, not activations.

On sufficiently large cards resident mode should substantially reduce those transfers.

---

# 55. Recommended implementation order

The dependency order should be:

```text
build/platform abstraction
      ↓
tensor/memory API
      ↓
basic elementwise + GEMM
      ↓
attention
      ↓
text encoder
      ↓
vision/reference encoders
      ↓
VAEs
      ↓
DiT
      ↓
sampler
      ↓
continuation
      ↓
stop/resume
      ↓
streaming optimization
      ↓
architecture tuning
```

Do not start by optimizing H3's largest kernel before the complete GPU contract is executable and testable.
