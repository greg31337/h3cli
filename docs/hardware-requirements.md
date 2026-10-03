# Hardware and memory requirements

h3cli streams the text encoder and can stream the video transformer, so the
entire model does not need to fit in RAM or VRAM. Memory requirements depend on
resolution, frame count, references, decoder, and weight placement. There is no
single minimum that covers every supported command.

This guide describes inference with the current implementation, using source
analysis and existing measurements. Building release packages has separate
[build requirements](build/source-build.md). GB denotes decimal storage sizes
or advertised hardware capacities; GiB denotes binary memory measurements.

## Choosing a starting configuration

| Configuration | RAM or unified memory | Dedicated VRAM | Evidence and limits |
| --- | ---: | ---: | --- |
| Linux, small videos with streaming | 32 GB | 12–16 GB | Plausible lower-memory configuration; not qualified on physical hardware with these limits |
| Linux, practical starting configuration | 64 GB | 24 GB | Generation has passed on 24 GB GPUs; 64 GB RAM is a planning estimate allowing streaming/cache headroom |
| Mac, small videos with `--ssd-streaming` | 32 GB unified | Shared | Estimate; needs validation on a smaller Mac |
| Mac, default resident BF16, small videos | 64 GB unified | Shared | Estimate; larger workloads can exceed this |
| Mac, longer videos and larger references | 96–128 GB unified | Shared | Substantial validation exists on the 128 GB M4; 96 GB remains an estimate |

These are starting points, not guarantees for arbitrary settings. The
[RTX 4090 qualification](cuda/cuda-13-qualification.md) records generation on a
24 GB GPU. The [macOS qualification](build/macos-distribution-results.md) records
the tested 128 GiB M4 Max. Neither establishes an exact minimum for smaller
machines or guarantees bitwise SGLang parity across different GPUs.

Start with a short, small preview before increasing resolution or duration.
Use `--preview-vae` to reduce decoder costs. On a smaller Mac, explicitly add
`--ssd-streaming`; CUDA automatically adjusts weight residency to available
VRAM. Avoid `--show` when minimizing memory because live previews can overlap
decoder and transformer allocations.

## How weight placement changes memory use

The 50 main BF16 transformer blocks occupy **35.89 GiB**. Fully streamed
execution retains two block slots, totaling **1.44 GiB**, plus other weights,
activations, and workspace. Text encoding, denoising, and final decoding have
different memory peaks; adding their separate peaks overstates the requirement.

CUDA selects full residency, partial residency, or streaming automatically.
Metal uses resident weights by default and requires `--ssd-streaming` to select
streaming. Its resident planner rejects insufficient capacity rather than
automatically changing modes. See the [CUDA planner](../src/weights/residency.c),
[Metal planner](../src/metal/gpu.m), and
[streaming guide](usage.md#2-make-a-first-fast-video).

The following recorded runs used dense BF16, all 50 blocks, 90 frames, six
evaluations, and the full video decoder:

| Workload | Placement | Sampled whole-GPU peak | Peak process RAM |
| --- | --- | ---: | ---: |
| 640×480, PRO 5000 / 5090 | Fully streamed | 5.6–5.8 GiB | 36.7 GiB |
| 1344×768, PRO 5000 | Fully streamed | 8.8 GiB | 36.7 GiB |
| 640×480, 5090 | Automatic partial residency | 25.9 GiB | 15.1 GiB |
| 1344×768, PRO 5000 | Fully resident | 43.3 GiB | 1.9 GiB |

GPU sampling can miss brief peaks. Process RAM excludes the rest of the system
and does not represent all filesystem cache. The CUDA planner also reserves
activation capacity and at least 1 GiB or 10% of device memory for headroom.
Consequently, a 5.6 GiB observed peak does **not** establish a 6 GB GPU minimum.
See the [measurement report](cuda/weight-residency-results.md) for exact scope.

The approximately 37 GiB host allocation is optional caching. CUDA caches
streamed weights when RAM permits, with a 40 GiB process-wide cache ceiling.
When cache admission fails, it falls back to bounded file reads and staging
buffers. Less RAM can therefore work, but may increase storage traffic and
render time. See [host cache admission](../src/cuda/gpu_cuda.cu).

FP8 and NVFP4 reduce core weight storage to approximately **17.94 GiB** and
**10.09 GiB**, respectively. Activations, other model components, conversion
workspace, and runtime overhead still require memory. These modes have stricter
GPU requirements than ordinary BF16 generation; see
[quantized denoising](cuda/denoiser-quantization.md).

## Memory guards and larger workloads

Both platforms require at least **10 GiB of available physical memory**, plus
the reserve for checked allocations. This is available headroom, not the total
installed-memory requirement. The process budget is the smaller of:

- **110 GB decimal** (about 102.45 GiB).
- Physical RAM minus `max(10 GiB, RAM / 8)`.

On Metal, GPU reservations also participate in the check. An 8 GB machine cannot
satisfy the normal guard, and 16 GB leaves little room for inference and the OS.
Swap does not remove the physical-headroom requirement. The Linux implementation
reads `/proc/meminfo` but does not account for container/cgroup memory limits;
a container can reach its own limit before the guard detects pressure. See
[memory accounting](../src/memory.c) and [memory guards](stability/memory.md).

Increasing denoising steps mainly increases runtime. Increasing pixels or frame
count grows activations and media buffers. References, continuation, adaptive
cache, and attention workspace also add memory. Upscaling runs refinement at
the larger geometry, so fewer refinement steps do not guarantee a low peak.

For example, a bounded **1344×768, 362-frame adaptive-cache** run reached
**52.14 GiB GPU memory**. An injected 32 GiB allocation limit failed. This was a
two-evaluation prefix of a longer schedule, not a completed video or a physical
32 GB GPU test. See the [long-video result](cuda/adaptive-budget-reference-results.md).

Metal has a separate reference-processing limit: its causal attention kernel
needs threadgroup memory for text/vision presentation tokens. The tested M4
kernel supports **7,936 tokens**; the runtime derives the limit from the device
and compiled kernel. More unified RAM does not increase this limit. See
[long-reference limits](stability/memory.md#causal-gqa-limit-a-different-constraint).

## GPU, CPU, and operating system

| Platform or feature | Current requirement |
| --- | --- |
| Ordinary CUDA generation | NVIDIA compute capability **8.6 or newer**, with a build covering the GPU |
| CUDA FP8/NVFP4, adaptive cache, SageAttention, SOL, and SubBlock | **SM120**, such as RTX 5090 or RTX PRO Blackwell, plus the corresponding compiled feature |
| Linux release package | x86-64 CPU with **AVX2/FMA**, glibc **2.35+**, NVIDIA driver **580.126.20+** supporting the GPU, and an executable runtime-cache directory |
| macOS release package | Apple Silicon and **macOS 26+** |

On Mac, CPU and GPU share unified memory; do not add separate RAM and VRAM
budgets. CUDA selects one GPU for execution, so multiple cards do not pool
their VRAM. CPU-only generation and AMD GPU backends are not provided.

Linux packages include the CUDA runtime, and both platforms' packages include
media tools. A separate CUDA toolkit, Python, PyTorch, or SGLang installation
is unnecessary for packaged inference. Source builds have additional
[dependencies](build/source-build.md). The Linux driver requirement follows the
pinned [CUDA 13.0 Update 3 release](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html).
GPU feature checks are in [device selection](../src/cuda/cuda_probe.c),
[attention admission](../src/denoise/attention.c),
[adaptive-cache admission](../src/denoise/approximate.c), and
[quantization admission](../src/cuda/cuda_quant.cuh).

## Storage and transfer bandwidth

One main model mode requires approximately **144 GB**. All catalogued models
occupy **294 GB logically**, with **216 GB of unique download content**.
Copy-on-write filesystems can reduce duplicate physical storage; capacity
checks allow ordinary copies. Leave additional space for quantization caches,
LoRA variants, videos, and saved states. See the
[model inventory](features/model-download-dependencies.md).

A local NVMe SSD is strongly preferable, especially when RAM cannot retain
streamed weights. PCIe and host memory bandwidth also affect CUDA performance:
fully streaming all 50 BF16 blocks over 50 evaluations transfers approximately
**1.75 TiB** of core weights to the GPU. Host caching avoids repeated disk reads,
but not those GPU uploads. Partial residency reduces both demands. A slow disk
or restricted PCIe link can therefore make a memory-sufficient system much
slower than the recorded benchmarks.

Firm lower-memory requirements still need qualification on physical 12/16 GB
CUDA GPUs with constrained host RAM and a 32 GB Mac using SSD streaming.
Allocation-limit tests on larger machines do not establish those guarantees.
