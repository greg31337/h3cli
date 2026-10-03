# Build and installation details

Run checkout commands from the repository root. For a shorter introduction,
start with the [README](../../README.md); see the [documentation index](../README.md)
for user guides.

To prepare public downloads, follow the [release guide](../release/README.md).
It covers cloning a fixed commit, building Linux and Mac packages, signing,
testing, and publishing a draft on GitHub Releases.

On Apple Silicon, `bash scripts/build_macos.sh` produces a single
`h3cli-macos-arm64` with bundled shaders and FFmpeg/FFprobe 9.0.2, plus a
source archive and development DMG. Runtime requires macOS 26+, with no
Homebrew, Python or developer tools. Local builds are ad-hoc signed and
unnotarized; Developer ID distribution uses a separate explicit workflow.
See the [macOS build/run guide](macos-distribution.md),
[qualification report](macos-distribution-results.md) and
[implementation tasks](../todo.md). Native source-build instructions remain below.

On Linux, the portable `h3cli-linux-x86_64` executable includes the native CUDA
runtime and FFmpeg/FFprobe 9.0.2. It runs without a toolkit, Python, system FFmpeg,
root access or a container engine. The host supplies the model files, an NVIDIA
driver (580.126.20 or newer, supporting the GPU), Linux x86-64 with glibc 2.35+
and an executable cache directory. The native SGLang CPU RNG requires AVX2/FMA.

The identical executable passed all 204 recorded outputs in Ubuntu 22.04,
Ubuntu 24.04 and Debian 12 userlands on RTX PRO 5000. These tests share the host
kernel/driver; other compiled GPU architectures remain unqualified for this
artifact. See the [release report](linux-distribution-pro5000.md) and
[build/run guide](linux-distribution.md) for full coverage and limits.
Prepared artifacts are under `bin/linux-release/`; no public download has been
published as part of this work. Given the executable and its checksum:

```sh
sha256sum -c h3cli-linux-x86_64.sha256
chmod +x h3cli-linux-x86_64
H3CLI_BUNDLE_INFO=1 ./h3cli-linux-x86_64
./h3cli-linux-x86_64 -d /path/to/MiniMax-H3 \
  -p 'A small boat on a quiet pond.' \
  --width 256 --height 256 --frames 22 --steps 2 -o preview.mp4
./h3cli-linux-x86_64 -d /path/to/MiniMax-H3 --server
```

To produce the portable release from a checkout on Linux, install Python 3.10+,
`skopeo` and `umoci`, then run `bash scripts/build_linux.sh`. It builds in a
pinned Ubuntu 22.04 filesystem and writes the executable, checksums and companion
`h3cli-linux-x86_64-sources.tar.gz` under `bin/linux-release/`. Distribute that
source archive alongside the executable; keep `linux-validation/` private.
The [source-material guide](../../scripts/linux/SOURCE-MATERIALS.md) explains component
licenses, exact source packages and relinking. The [task list](linux-distribution-tasks.md)
records completed acceptance.

For ordinary source builds, clone or download the repository and follow your
platform setup below from its root directory. `make` selects Metal on macOS and
CUDA on Linux and produces `bin/h3cli` and `bin/libh3.a`. Model weights are needed
for generation, not compilation. The existing Ubuntu setup remains supported.

Executables and the static library live in the gitignored `bin/` directory. Test
executables omit the `h3_` prefix, for example `bin/sampler_tests`.
Build an individual executable with `make bin/sampler_tests`; `make clean`
removes `bin/` and the object and dependency files.

The project and command are named `h3cli`. The CLI entry
point is `src/h3cli.c`, and the shared inference engine is `src/engine.c`. The library
remains `bin/libh3.a` with the `src/h3.h` API. Existing `H3_*` environment variables,
`h3` cache directories and saved-state formats keep their names for compatibility.
Historical validation artifacts, pinned third-party sources and upstream
repository links retain their original names and paths.

Implementation files are grouped by function under `src/`; the CLI, engine,
public interfaces and shared platform utilities remain directly in `src/`.
See the [source layout](../../src/README.md) for the directory map. Metal's runtime
shader is `src/metal/shaders.metal`.

The sample images in [inputs/](../../inputs) are AI-generated portraits.
Their prompts, dimensions and checksums are recorded alongside them. Historical
measurements using previous images remain tied to their original checksums.

For automated dependency installation, run the matching script from a checkout
or downloaded copy of this repository:

```sh
# Apple Silicon, macOS 26+; run as your normal user, without sudo.
bash scripts/setup_macos.sh
source outputs/setup/macos-env.sh
make -j8

# Ubuntu 24.04 x86-64; run as root or a user with sudo.
bash scripts/setup_linux.sh
source outputs/setup/linux-env.sh
./bin/h3cli --help
```

Both scripts accept `--dry-run` and `--help`. The Linux script installs all
required CUDA video dependencies, verifies the native runtime, and builds
`bin/h3cli` and `bin/libh3.a`. Use `--no-build` to
prepare dependencies only, or `H3_BUILD_JOBS=4` to limit compilation parallelism.
`--with-cudnn` remains an accepted compatibility option; cuDNN is required and
installed by default. The macOS script prepares dependencies; run `make` afterward.

Neither script downloads models or edits shell profiles. Source the generated
environment in each new shell before building, testing, or running inference.
Linux dependencies under `outputs/setup/` are part of the runtime installation;
retain the paths exported by the generated environment.
An existing dependency checkout must match its pinned revision and be unmodified.
A single private cuDNN 9.20 installation serves encoding and decoding, preserving
provider-held packages.

On macOS, finish Apple's Command Line Tools installer if it opens and rerun setup;
an SDK older than 26 needs updating first. On a bare Linux host without a working
NVIDIA driver, add `--install-driver`, then reboot and rerun if requested.
Containers use the host's driver. A working but too-old driver is reported without
replacing it. GPU-free installation builds `CUDA_ARCH=fat`; execution still needs
a supported GPU and a compatible driver.

### macOS: Apple Silicon and Metal

Use an Apple Silicon Mac with macOS 26 or later and a macOS 26-or-newer SDK.
The current source references Metal 4 SDK declarations even on Macs that use
the ordinary Metal path.

1. **Install Apple's Command Line Tools.** In Terminal, run:

   ```sh
   xcode-select --install
   ```

   Finish the installer before continuing. Install Command Line Tools for
   Xcode 26 or later; if an older version is already installed, update it in
   System Settings → General → Software Update, or use
   [Apple's Command Line Tools downloads](https://developer.apple.com/download/all/).
   Check the selected compiler and SDK:

   ```sh
   xcode-select -p
   xcrun clang --version
   xcrun --sdk macosx --show-sdk-version
   ```

   The tools supply Clang, Make, Git, and the SDK frameworks used by this
   project: Foundation, Metal, MPS/MPSGraph, Accelerate, and system ICU.
   Server mode also links SDK-provided SQLite and libcurl; no additional
   Homebrew database or HTTP server package is required.
   Full Xcode and its optional offline Metal compiler are unnecessary for
   this build; shaders compile at runtime.

2. **Install Homebrew**, if it is not already installed, using its
   [official installer](https://docs.brew.sh/Installation):

   ```sh
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

   For the default Apple Silicon installation, add Homebrew to your login
   shell and activate it in the current Terminal session:

   ```sh
   echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
   eval "$(/opt/homebrew/bin/brew shellenv)"
   ```

3. **Install media and test tools.** FFmpeg also installs FFprobe; both are
   needed for media inputs and MP4 output. Python 3.11 or later supports the
   test/report helpers.

   ```sh
   bash scripts/setup_macos.sh
   source outputs/setup/macos-env.sh
   ffmpeg -version  # 9.0.2
   ffprobe -version
   ```

4. **Build and check the executable** from the repository root:

   ```sh
   make -j8
   ./bin/h3cli --help
   ```

   To run the regression suite, use `make test`. Tests that need missing model
   weights or optional fixtures report their skips. When copying the executable,
   preserve the sibling `bin/` and `src/` directories containing the executable
   and `shaders.metal`.

### Ubuntu 24.04 LTS: x86-64 and NVIDIA CUDA

Use **CUDA 13.0 Update 3 / 13.0.3 (`nvcc V13.0.88`)** with the pinned
libraries below. This aligns the compiler with the recorded SGLang
0.5.20 / PyTorch 2.13.0+cu130 environment. The toolkit version alone does not
guarantee bitwise parity: cuDNN, cuBLAS, codec versions, and GPU architecture
also matter. `nvidia-smi` reports driver capability, not the installed compiler.

| Component | Required version | Purpose |
| --- | --- | --- |
| CUDA toolkit | 13.0.3 / `nvcc V13.0.88` | Native compiler, CUDA runtime and linked cuBLAS |
| NVIDIA driver | 580.126.20 or newer, with support for the GPU | Matches CUDA 13.0 Update 3 |
| Reference cuBLAS | `nvidia-cublas==13.1.1.3` / API identity `130101` | Recorded SGLang GEMM arithmetic |
| Reference NVRTC | `nvidia-cuda-nvrtc==13.0.88` | CUDA 13 library dependency |
| cuDNN | `nvidia-cudnn-cu13==9.20.0.48` | Shared native image/video encoder and audio decoder |
| cuDNN frontend | `v1.11.0` | Native build headers |
| Reference CUTLASS | `da5e086dab31d63815acafdac9a9c5893b1c69e2` | Pinned FlashAttention inference headers |
| Production FFmpeg / FFprobe | 9.0.2, built from pinned upstream source | All normal input, output and server inspection |
| Regression-only output FFmpeg | 4.2.2, from `imageio-ffmpeg==0.5.1` | Immutable SGLang regression encoding |
| Regression-only input FFmpeg / FFprobe | 6.1.1 | Immutable SGLang regression decoding |
| TurboJPEG | Ubuntu 24.04's 2.1.5 | Accurate reference JPEG decoding |

The [CUDA 13.0 Update 3 release notes](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html)
list these compiler, cuBLAS, NVRTC, and driver versions. SGLang's
[Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker/Dockerfile)
also defaults to CUDA 13.0.3; follow the pins here instead of a moving image tag.
The CUDA toolkit and library components have independent version numbers.

Both setup scripts build FFmpeg/FFprobe 9.0.2 under `outputs/setup/ffmpeg`;
source the generated environment to select them. The standalone Linux executable
bundles 9.0.2. Historical codecs are test-only: before a source regression run,
run `python3 scripts/setup_reference_media.py`. The regression runner selects
its verified profile automatically; ordinary rendering and current-feature
tests continue to use 9.0.2. See the [media version policy](ffmpeg-upgrade.md).
The script installs checksum-pinned native runtime wheels into one private
environment. Python is used for installation and test helpers; inference
requires neither PyTorch nor the SGLang Python package.

From a clean checkout on Ubuntu 24.04 x86-64:

```sh
# Run as root or a user with sudo; a working host NVIDIA driver is preserved.
nvidia-smi
bash scripts/setup_linux.sh
source outputs/setup/linux-env.sh
./bin/h3cli --help
```

The script installs GCC/G++, Make, ICU, json-c, OpenSSL, media tools, the toolkit,
pinned headers, and the shared runtime environment. It verifies cuBLAS/cuDNN versions,
media executables, and cuBLAS 13 GPU initialization when a GPU is visible, then
builds the ordinary CUDA video pipeline. `outputs/setup/runtime.json` records
those checks. No model download or generation is implied by setup success.

For a bare host with no NVIDIA driver, run setup with `--install-driver`.
After any required reboot, confirm `nvidia-smi` works and
rerun setup. Inside a container, install/expose the driver on its host instead.
An older working driver must be upgraded by the host administrator.

For subsequent builds and tests:

```sh
source outputs/setup/linux-env.sh
make -j8
make test

# Complete recorded regression; needs the original model and a fresh output path.
export H3_REFERENCE_MODEL=/path/to/MiniMax-H3
export H3_REFERENCE_REGRESSION_OUT=outputs/cuda-reference-regression/run-001
make test-cuda-reference-regression
```

The default build uses `CUDA_SGLANG=1 CUDA_CUDNN=1`. Encoding runs inside
`h3cli` using the same cuDNN 9.20 runtime as decoding. Retain the paths
exported by `linux-env.sh`. Re-run setup after moving the checkout to
regenerate its absolute dependency paths. Build-only checks and `--help` do not
prove generation or bitwise parity; the full regression checks all 204 recorded
outputs without tolerance or skipped cases. See [CONTRIBUTING.md](../../CONTRIBUTING.md)
and the [reference guide](../cuda/cuda-sglang-reference.md) for measured scope.
The [CUDA 13 qualification record](../cuda/cuda-13-qualification.md) records
the clean RTX 4090 setup and the 204/204 exact pass on RTX PRO 5000 / SM120.
The [shared cuDNN 9.20 qualification](../cuda/cudnn-920-qualification.md)
confirms all 204 hashes after moving encoding into the main process and
removing the CUDA 12 compatibility dependencies.
The frozen SM120 goldens are not a promise of identical hashes on other GPUs.

Supported architecture selections are `86` (3090), `89` (4090), `90` (H100/H200),
`100` (B200), `120` (5090/RTX PRO Blackwell), `auto`, and `fat`. Setup uses `auto`
with a supported visible GPU and `fat` otherwise. For a specific deployment
without a build-time GPU, run `make CUDA_ARCH=89`, for example. Different GPUs,
math libraries, media tools, or compiler versions require validation against the
recorded results; successful compilation alone does not establish parity.

After either platform's build, generation downloads required missing weights
to the selected models root. Use `--models-path /fast/models` for a local SSD
or `-d` for an existing main checkpoint. Compilation does not download models. Offline LoRA preparation and additional reference-generation tools have their
own Python environments; native runtime LoRA folding does not require Python; see [the LoRA guide](../../lora/README.md) and
[CUDA validation tiers](../cuda/cuda.md#validation-tiers).

Final cleanup and qualification are documented in
[the M6/M7 record](../cuda/single-pipeline-final.md). The unchanged default
parity suite must run after every coherent code change; see
[CONTRIBUTING.md](../../CONTRIBUTING.md).
