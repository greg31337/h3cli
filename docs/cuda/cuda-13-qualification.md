# CUDA 13.0.3 setup qualification

This records the initial toolkit upgrade. Its separate encoder process and
CUDA 12 decoder dependencies have since been retired; use the
[current runtime instructions](cuda-sglang-reference.md#build-and-runtime) and
[shared cuDNN 9.20 qualification](cudnn-920-qualification.md).

The default Linux setup uses CUDA **13.0 Update 3 / 13.0.3**, with
`nvcc V13.0.88`. Validation was performed on 2026-09-26. The recorded oracle
remains SGLang **0.5.20 / PyTorch 2.13.0+cu130**; no golden hashes or numerical
tolerances were changed.

## Installation and dependencies

The installation test started from Canonical's Ubuntu Base **24.04.5 x86-64**
filesystem without GCC, Python, or CUDA. It ran in PRoot 5.4.1 with the RTX 4090
and host driver **580.173.02** exposed. Build tools, media libraries, and native
CUDA dependencies came from `scripts/setup_linux.sh`; the host's existing
toolkit and cuDNN packages were not exposed to the test filesystem.

Setup installed and verified the dependencies in the
[source-build dependency table](../build/source-build.md#ubuntu-2404-lts-x86-64-and-nvidia-cuda), then built
`bin/h3cli`, `bin/sglang_encoder_worker`, and `bin/libh3.a`. Re-running setup
successfully reused the installed toolkit and dependency checkouts. The working
GPU driver was preserved. No Torch or SGLang Python installation was needed.

CUDA 13 changed `cudaGraphNodeGetDependentNodes` to take an explicit edge-data
argument. A small compatibility overload in `src/cuda/cuda_cudnn.cu` lets the pinned
cuDNN frontend **v1.11.0** compile against this API without changing its graph
construction or the selected convolution engines.

The cuDNN **9.10.2.21** audio decoder retained its original CUDA 12 cuBLAS
**12.8.4.1** and NVRTC **12.8.93** dependencies in a private runtime environment.
They were required by that cuDNN build even when linking against CUDA 13.
The image/video worker used isolated cuDNN **9.20.0.48**. Both processes retained
reference cuBLAS **13.1.1.3** where the SGLang arithmetic calls for it.

## Results

| Check | Result |
| --- | --- |
| Setup script regression tests | 27 passed |
| Recorded regression runner checks | 8 passed |
| Clean Ubuntu / RTX 4090 build and runtime checks | Passed |
| RTX 4090 ordinary CLI with `--first-frame` and no worker override | Passed; 128×128, 22 frames, two evaluations, H.264 video and AAC audio |
| RTX 4090 `make -j8 test` | Passed; SM90 and SM120 tuning probes correctly report another architecture |
| RTX PRO 5000 72GB / SM120 full recorded regression | **204/204 exact hashes passed**; 124.9 seconds after build, 215.9 seconds including build |
| RTX 4090 full recorded regression | All 204 outputs produced in 118.6 seconds after build; 87/204 match the SM120 goldens |
| RTX 4090 CUDA 12.8 versus CUDA 13.0.3 | **204/204 output hashes identical** between the two installations |
| macOS Metal build and `make test` | Passed; optional model/fixture tests report their existing skips |

The SM120 run used driver **595.91.07**, the CUDA 13.0.3 compiler/runtime copied
from the clean installation, and the existing pinned cuDNN/media libraries.
It completed every component probe and the full **640×480, 124-frame, six-step**
render within the runner's 12-minute execution deadline.

PRoot's root-emulation mode changes metadata while checking a file with mode
`0000`, which made the CPU quantization-cache test fail. Running the unchanged
tests with the actual root UID and no root emulation resolved this harness
artifact. No product test or metadata validation was weakened.

The recorded suite remains an exact comparison against its SM120 golden state.
A CUDA 12.8 baseline on the RTX 4090 completed all probes and generation but
matched only **87/204** hashes. CUDA 13.0.3 produced exactly the same 204 output
hashes on that GPU, preserving this baseline without rebaselining the SM120
goldens. A supported compiler and successful generation do not establish
bitwise equivalence across GPU architectures.

The passing SM120 run recorded these identities:

```text
source:   87de0a3b16b3dc28e5fdbe41a145c361ea03a52f6c8b2a168b54f982e4af5fcd
goldens:  fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3
```

Use the ordinary instructions in [CONTRIBUTING.md](../../CONTRIBUTING.md) to
repeat the full recorded regression. Its model and fixture paths remain
configurable; it requires no SGLang service or external oracle.
