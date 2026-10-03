# Reproducible Linux builds and a single-file CUDA distribution

Status: **implemented; final qualification passed; 32/32 tasks complete**.
Updated 2026-09-29. See the [PRO 5000 results](linux-distribution-pro5000.md)
for exact artifact evidence, four passing 204-output gates, completed source
audits and two byte-identical builds. Earlier PRO 6000 results remain in the
[historical report](linux-distribution-results.md).
Implementation checklist: [completed checklist](linux-distribution-tasks.md). The preceding server work is
preserved in [its completed checklist](../features/server-tasks.md) and
[qualification report](../features/server-results.md).

## Goal and release contract

Make Linux installation a download-and-run operation, and make building the
same release from source one command. Produce one executable containing h3cli
and its redistributable runtime dependencies. On first use it extracts a
versioned runtime into a user-owned cache; later invocations reuse that cache.
No FUSE, root installation, Python environment, CUDA toolkit, compiler, FFmpeg
installation or container engine is required to **run** the release.

The host still supplies a supported NVIDIA GPU and driver, a compatible Linux
kernel/glibc, model files, and sufficient RAM, VRAM and disk space. Models are
too large and independently managed to include. A single download does not mean
a fully static CUDA program or zero extracted files.

Keep the current CLI, server API, inference defaults, native state validation
and all platform-appropriate features. In particular, the package must include
the optional CUDA implementations needed by existing quality/attention presets;
it must not silently become a reduced-feature build. Preserve `make`,
`bin/h3cli`, `bin/libh3.a`, and the existing macOS workflow.

## Existing constraints

The current [Linux setup](../../scripts/setup_linux.sh) installs Ubuntu 24.04
packages and CUDA 13.0.3, prepares pinned libraries in `outputs/setup/`, and
generates an environment file with absolute paths. Moving a checkout or copying
only `bin/h3cli` does not carry that runtime. The Makefile embeds a toolkit
RPATH; separate environment variables select reference math libraries, JPEG,
input/output FFmpeg and ffprobe. The server additionally uses SQLite and curl.

The preceding development-host inspection found `GLIBC_2.38` in its installed
CUDA executable's requirements. This is historical evidence, not a measurement
of the clean PRO 6000 node or a permanent source requirement. Lowering the floor
requires rebuilding and auditing the
entire dependency tree, including helper programs. `ldd` alone misses explicit
`dlopen` loads, cuDNN engine libraries, codec helpers and non-ELF resources.

The original [recorded parity runner](../../tests/cuda_reference_regression.py)
built an isolated source copy and invoked `ffmpeg` from PATH when hashing final
RGB/PCM. The new artifact mode exercises the packaged executable, matching
probes and pinned input decoder while retaining the complete immutable
204-output contract. Source-build mode remains available.

## Supported envelope

| Property | Initial target |
| --- | --- |
| CPU / userland | Linux x86-64, glibc 2.35 or newer; generic host CPU code, no `-march=native` |
| Builder baseline | Pinned Ubuntu 22.04 x86-64 container/sysroot; exact compiler and package inputs locked |
| Required distribution tests | Ubuntu 22.04, Ubuntu 24.04, Debian 12, with the same release bytes |
| CUDA compiler | 13.0 Update 3 / 13.0.3, `nvcc V13.0.88` |
| Host driver | At least the existing 580.126.20 requirement, and support for the installed GPU |
| Ordinary CUDA kernels | Existing fat set: SM86, 89, 90, 100, 120 plus the existing PTX fallback |
| Optional kernels | Preserve current architecture guards, including Sage's SM120a code |
| GPU qualification | RTX PRO 5000 / SM120 for final acceptance; initial PRO 6000 functional results retained separately. Other architectures are compiled but unqualified for this artifact. |

Ubuntu 22.04, Ubuntu 24.04 and Debian 12 appear in NVIDIA's CUDA 13.0.3
installation guide, with glibc 2.35, 2.39 and 2.36 respectively. Selecting 2.35
is our initial portability tradeoff; upstream toolkit support does not establish
that our complete package works there.
[NVIDIA installation guide](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-installation-guide-linux/index.html).

Do not claim support for every Linux distribution. Alpine/musl, ARM64, glibc
older than 2.35, Windows/WSL qualification, new GPU ports and new attention
implementations are outside this work. A newer GPU accepting PTX is not a claim
of feature availability or recorded numerical parity. Keep the actual tested
kernel and driver versions in the report; distro containers share the host
kernel/driver and do not independently qualify every kernel or the minimum
driver version. Do not change the working test host's driver for this project.

## Pinned build and dependency inputs

Use a digest-pinned builder with versioned, hash-checked downloads and an exact
package/source lock. A container tag by itself is insufficient. Include the
compiler, linker, libc headers/sysroot, build tools, codec configuration,
dependency sources, and packaging tools. Network access belongs to an explicit
fetch phase; a rebuild with a populated dependency cache must work offline.
Freeze actual digests during implementation rather than inventing them here.

Preserve the existing numerical pins from
[the runtime lock](../../scripts/requirements-sglang-runtime.txt):

| Component | Required recipe |
| --- | --- |
| CUDA runtime/toolkit libraries | CUDA 13.0.3; record the exact toolkit runtime files |
| Reference cuBLAS / cuBLASLt | `nvidia-cublas==13.1.1.3`, cuBLAS API identity `130101` |
| NVRTC and builtins | `nvidia-cuda-nvrtc==13.0.88` |
| cuDNN and required engine/graph libraries | `nvidia-cudnn-cu13==9.20.0.48`; one version for encode/decode |
| cuDNN frontend | `v1.11.0` |
| SGLang FlashAttention CUTLASS | `da5e086dab31d63815acafdac9a9c5893b1c69e2` |
| Sage CUTLASS | `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8` / v4.2.1, with existing vendored Sage pins |
| Production FFmpeg and ffprobe | Source-built 9.0.2 for all normal media operations |
| Regression-only codecs | 6.1.1 input/probe, exact 4.2.2 output from `imageio-ffmpeg==0.5.1`; validation kit only |
| TurboJPEG | 2.1.5, retaining accurate JPEG decoding |

Resolve the remaining native dependency versions from the qualified baseline:
ICU/data, json-c, OpenSSL, SQLite, curl, C++ runtime and their dependencies.
Preserve observable behavior when rebuilding against the older sysroot. Do not
substitute Ubuntu 22.04's default media or Unicode libraries merely because
they are easier to install. If a downloaded ELF exceeds the ABI floor, rebuild
the corresponding source with the required version/configuration and validate
its output. Rebuilding FFmpeg 6.1.1/TurboJPEG is a qualification risk, not an
assumption that matching version strings guarantee identical arithmetic.

Preserve precise compiler flags for ordinary CUDA kernels and separate flags
for approximate kernels. Do not apply fast-math globally to make a new compiler
work. Compare toolkit-linked and explicitly loaded cuBLAS identities; deduplicate
only identical or explicitly qualified files. Never accidentally load two
incompatible copies with the same SONAME.

Qualified contributor build entry point:

```sh
bash scripts/build_linux.sh
```

It builds in a digest-pinned OCI filesystem through PRoot on an x86-64 Linux
builder without requiring a GPU or host CUDA toolkit. The PRO 6000 provider
blocks user namespaces and exposes no Docker daemon. PRoot was tested with GPU
driver bindings; it translates filesystem access and is not a security sandbox.
Host build helpers are Python 3.10+, skopeo and umoci; the exact PRoot binary is
hash-pinned. Each build freshly unpacks the verified OCI image and installs only
the locked local packages. No container runtime is used by the shipped program.
It snapshots the checkout, uses bounded build parallelism,
does not mutate host packages, and writes caller-owned artifacts under ignored
`bin/`. Keep container recipes, locks and packaging helpers under
`scripts/linux/`. A GPU is needed for qualification, not compilation. Local M4
testing preserves Metal behavior; building a fat CUDA release through emulation
on the M4 is not a required workflow.

Expected outputs:

```text
bin/linux-release/h3cli-linux-x86_64 # one downloadable executable
bin/linux-release/linux-runtime/    # same payload, unpacked for validation
bin/linux-release/linux-validation/ # matching probes; developer-only artifact
bin/linux-release/libh3.a           # library from the same portable build
bin/libh3.a                         # ordinary make output remains unchanged
outputs/linux-distribution/<run>/   # logs, manifests and qualification evidence
```

Record a source snapshot hash even for a dirty checkout or downloaded source
archive. Normalize build paths, timestamps, archive ordering and ownership;
use a fixed `SOURCE_DATE_EPOCH`. Build twice in different directories with empty
build caches and require identical release bytes. Keep provenance separate
from the self-hashed payload to avoid circular digests. Changes to actual code,
compiler flags or runtime contents must still change their respective identity.

## Relocatable runtime first

Build and qualify an ordinary directory before adding the single-file wrapper:

```text
runtime/
  bin/h3cli
  lib/                       # bundled native libraries and required sublibraries
  tools/ffmpeg                # pinned 9.0.2 input/output tool
  tools/ffprobe               # pinned 9.0.2 probe
  share/h3cli/runtime.json
  share/licenses/
```

Inventory all `DT_NEEDED`, explicit `dlopen` targets, subprocesses, data files and
licenses. Check every ELF's architecture, glibc/GLIBCXX requirements, CPU ISA and
RPATH/RUNPATH. Bundle libstdc++/libgcc as needed, without bundling host glibc,
its loader/NSS stack or NVIDIA driver libraries. CUDA link-time stubs must never
enter the release. Treat required driver user-space libraries, host DNS files
and trust stores as documented host interfaces, not missing packaged CUDA SDK
dependencies. Validate actual loaded paths during inference as well as static
dependency listings.

Use relative `$ORIGIN` paths on all relevant ELF objects, including transitive
libraries. Establish an explicit search policy for `dlopen` and host driver
discovery; a RUNPATH on just the top executable does not solve every transitive
load. No runtime path may reference the build workspace, Python environment,
`outputs/setup/` or `/usr/local/cuda`.

Resolve bundled resources from the real executable/runtime manifest, never the
shell's cwd or the command's spelling. Centralize Linux resolution under
`src/runtime/` rather than scattering new fallbacks across encoders. Native
source builds keep their documented environment overrides. In a qualified
bundle, math/media libraries and helper programs always come from its payload;
an inherited setup environment must not silently replace them. Strip conflicting
library injection/search settings for the packaged child environment while
preserving supported device, model, rendering and test variables. Document the
policy and test stale setup environments and hostile PATH entries. A separate
runtime-override feature is outside this first release.

Keep caller cwd and argv intact. Default model locations remain:

- `models/MiniMax-H3`
- `models/image-vae/minimax_h3_t1_image_vae_step1597.safetensors`
- `models/latent-upscale/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`
- `models/preview-vae/taeh3.safetensors`

Inputs, outputs, server state and LoRA/quantization caches retain their current
CLI rules. Never put mutable job data or models in the extracted runtime.
Explicit model paths continue to work from arbitrary directories. Renaming the
outer executable and invoking it through a symlink must work.

For curl HTTPS imports, discover the target distro's host CA store, honoring
documented explicit CA overrides, and verify DNS and certificate validation on
all three distros. Do not bundle a silently aging replacement trust store or
disable verification. A missing trust store must produce a useful import error;
it must not prevent offline generation or local upload workflows. Include any
required OpenSSL providers or ICU data in the dependency audit.

## Single executable and runtime cache

Package the validated directory behind a small statically linked Linux launcher
with its decoder built in. A standalone musl-linked launcher is acceptable;
the inference payload still requires glibc. Pin the launcher toolchain and
decoder. Do not implement a new compression algorithm or require system tar,
unzip, a shell, Python, FUSE, mounts or runtime downloads.

AppImage's build guidance also requires avoiding absolute paths and building
against the oldest supported base. Its FUSE troubleshooting documents an
extract-and-run fallback and its repeated extraction cost. A persistent cached
payload suits this large CUDA runtime and server workload; a separate AppImage
format is deferred.
[AppImage build guidance](https://docs.appimage.org/reference/best-practices.html),
[FUSE and extraction behavior](https://docs.appimage.org/user-guide/troubleshooting/fuse.html).

Default cache root:
`${XDG_CACHE_HOME:-$HOME/.cache}/h3cli/runtime/<runtime-id>/`.
The runtime ID hashes both the payload digest and the file-table digest, so
changes to names or modes also get separate cache entries.
Add `H3CLI_RUNTIME_CACHE` to select a writable, executable cache root, including
for services with no usable home directory. It is a local startup setting, not
a new SGLang request field. Add a launcher-only `H3CLI_BUNDLE_INFO=1` diagnostic
which prints package identity, compatibility requirements and cache location
without loading CUDA; no new public CLI switches are needed.

The launcher must:

1. Validate its trailer, manifest, offsets and declared extraction sizes before
   allocating or writing. Check the embedded payload digest on extraction.
2. Create a private staging directory under the chosen cache, lock competing
   initializations, and reject path traversal, absolute entries, escaping links,
   device nodes and unexpected files. Flatten library SONAME symlinks into regular
   files during assembly; the archive format accepts no symlinks. Validate every
   extracted file's digest.
3. Publish the complete verified runtime atomically. An interrupted extraction
   must never look ready; a subsequent launch can safely retry.
4. Reuse a complete owner-controlled cache on warm launches without decompressing
   or hashing gigabytes again. Verify the manifest/ready marker, ownership,
   expected file types and recorded file metadata. Detect missing or changed
   entries and fail with explicit stop/remove/retry guidance; never mutate
   a live runtime in place. A replacement directory must not disturb an active
   server still using the old runtime; fail with recovery guidance if that
   cannot be guaranteed. Qualification separately verifies all content hashes.
5. Set the scoped child runtime environment and `exec` the real h3cli, preserving
   argv, cwd, stdio, exit status, signals and inherited private descriptors.

The cache protects against incomplete extraction and accidental corruption; it
is not a defense against a malicious process already running as the same user.
The payload digest detects damage, not publisher authenticity. Publish a SHA-256
checksum and provenance with the release; do not imply a signed release unless
signing is separately implemented.

Keep versions in separate immutable directories. No automatic eviction or
garbage collector is needed in this release. Document manual cache removal only
after all users of that version have stopped, including workers and FFmpeg.
Handle concurrent launches, read-only download directories, cache `noexec`, low
disk space, insufficient permissions and corrupt downloads with concise errors
and recovery instructions. Do not silently fall back to a system library.

## Server and saved-state behavior

The server already finds its executable using the platform helper, hashes it
and spawns private workers. After the launcher uses `exec`, those workers should
resolve the extracted **core**, not re-extract the outer package or redispatch
through a shell. Children inherit the same pinned runtime and preserve control
FDs, process-group cancellation, parent-death handling and FFmpeg cleanup.
Starting another package version must not change a running server's executable
or libraries. Leave the one-extra-string SGLang request format unchanged.

Runtime paths currently participate in conditioning/checkpoint environment
identity. Audit these explicitly: moving identical package contents to another
cache root must not invalidate a same-build/same-device checkpoint just because
an absolute library path changed. Represent verified bundled dependencies by
their stable content/recipe identity where needed. Continue checking actual
contents and preserve strict source/build, device, backend, arithmetic and
model compatibility. Never solve relocation by broadly ignoring the numerical
environment or accepting incompatible saved files. No old-format compatibility
layer or cross-build resume promise is part of this work.

## Qualification and acceptance

The initial clean-node bootstrap used RTX PRO 6000. The user subsequently
authorized the **local RTX PRO 5000 for completion**, including fresh Linux
builds and final CUDA qualification:

```sh
ssh cuda-test
```

Keep the local M4 for Metal/host regression. The SSH command is an operational
instruction for this plan, not a shipping default. Preserve both hosts' recorded
results and qualify final bytes on the PRO 5000 without replacing its driver.

Before bootstrap, record the clean node's OS, kernel, CPU, GPU model/compute
capability/VRAM, driver, disk capacity, mounts and available tools. Check whether
the provider session is itself a container and verify that the pinned builder
and clean distro test containers can run with the required GPU access; do not
assume a usable Docker daemon or nested-container privileges. Resolve missing
build/container tools on this node, preserving its working driver. Keep CUDA
toolkits and development dependencies in the build environments, separate from
the clean runtime test environments.

Stage the current source and recorded fixtures, discover or provision external
model/component files and required test assets, and record their actual paths.
Do not assume `/models`, `/path/to/models`, an existing checkout or preinstalled
h3cli libraries. Check a scratch filesystem's CoW support before the LoRA tests.
Build from source on this node and establish a fresh full recorded parity
baseline with the pinned dependencies before evaluating packaging changes;
earlier results from another GPU are historical evidence only. Do not transplant
a prebuilt development runtime as the clean-build result.

Use configurable work/model paths; no machine-specific home directory, server
address or model location belongs in production defaults.
Store evidence under `outputs/linux-distribution/` and publish a concise report
in `docs/build/`. Missing fixtures, capabilities, required tests or hardware are
unresolved tasks, not passes.

### The complete recorded gate

Follow [CONTRIBUTING.md](../../CONTRIBUTING.md): every coherent code/build/test
change requires the full recorded CUDA regression. Do not change any golden
hash or fixture, introduce case selection/tolerances, or run live SGLang as a
replacement. Keep the 12-minute post-build gate and full 640×480 / 124-frame /
six-step render plus all production probes.

Extend the runner with an explicit prebuilt-artifact mode while preserving its
default isolated source-build mode. Build matching probes in the same pinned
build and record them in a separate validation manifest; test executables stay
out of the public download. Verify source, probes, package, payload, core, runtime
libraries and fixtures before/after the run, and include build/container/lock
files in source provenance. A mismatched probe or artifact is a failure.

In package mode, the full C0 render must enter through the final outer executable;
component probes must use the identical bundled libraries. Final RGB/PCM decoding
must explicitly use the pinned validation-only 6.1.1 decoder, never a PATH
binary. The gate activates a hash-bound external profile selecting the historical
4.2.2 encoder and 6.1.1 decoder/probe. The core checks profile and tool hashes;
normal jobs and current-feature tests use the bundled 9.0.2 tools. This explicit
media split supersedes the original requirement to ship historical codecs. Preserve all existing artifact, trajectory, geometry and deadline checks
and require **204/204** matches. Record loaded library/tool paths to prove that
the host's development installation did not supply missing pieces.

Pass the complete gate for the relocatable directory, then for the final single
file. Run that same single-file artifact in clean Ubuntu 22.04, Ubuntu 24.04 and
Debian 12 userlands on the PRO 5000, with host driver access and external models
but no installed CUDA toolkit or application dependencies. A validation harness
may contain Python; a separate runtime smoke environment must prove it is not
needed by the application. Keep the test harness out of dependency resolution.

### Other required checks

| Area | Required evidence |
| --- | --- |
| Build | Clean GPU-free build, offline rebuild, ABI/closure audit, identical bytes from two fresh builds |
| Launcher | Cold/warm and simultaneous starts, interrupted/corrupt extraction, paths with spaces, rename/symlink, disk/permission/noexec errors, arguments and signals |
| Relocation | Move download and runtime roots; remove access to original checkout/toolkit/venv; stale setup environment and fake PATH tools cannot change runtime selection |
| CLI | Small T2VA, anchors and mixed image/video/audio references; full/preview decode; stills; conditioning and state save/load, pause/resume, continuation/bridge, upscale, LoRA/CoW and placement |
| CUDA features | Existing current-CUDA integration suite and compiled capability inventory; preserve adaptive/SubBlock, attention and quantization guards/defaults |
| Server | Real submit/poll/download, CLI/server agreement, cancellation then another job, restart/queued recovery, uploads and opt-in HTTPS import; workers and FFmpeg use the package |
| Native builds | Local M4 build and applicable retained tests; Linux source build/setup tests still work |

Reuse current harnesses and fixtures. Outside the fixed parity render, default
to 256×256, 22 frames, two steps, seed 42; use 56 frames for mixed reference-video
coverage, 39/56 for continuation, and 512×512/22 frames for 2× upscale. Existing
feature tests may use up to six evaluations to exercise cache warmups. Retain
bounded dimensions already required by those suites. No new high-resolution
benchmark or 50-step quality campaign is necessary for packaging.

Measure compressed download size, extracted footprint, peak extraction space,
build time, cold/warm startup time and extraction memory use. Do not promise a
small download: cuDNN/cuBLASLt and their dependencies dominate the core executable.
Account for file licenses and redistribution requirements; include notices and
the corresponding-source offer/materials required by the chosen media builds.

## Delivery and exclusions

Implement in order: locked builder and baseline ABI, relocatable runtime,
single-file launcher, artifact-aware gate, complete qualification, documentation.
Keep the Ubuntu source setup available and document it as the contributor
alternative; share dependency pins rather than maintain conflicting versions.
Release instructions should lead with the executable download, driver/model
requirements and normal CLI/server examples only after the artifact qualifies.

This work prepares local release artifacts and evidence. Publishing a release,
installing drivers on users' systems, adding a GitHub GPU workflow, an automatic
updater, a package-manager installer, a model downloader or packaging the Python
development/LoRA helper tools is outside scope. Native LoRA features remain in
the executable. Do not remove current source setup or claim other distros/GPUs
are qualified just because compilation succeeds.

## Baseline finding during implementation

The unchanged starting source completed all 204 outputs on PRO 6000, but 113
hashes differed from the immutable PRO 5000 goldens. The source fingerprint
matched the prior qualified source. The previous 5090 campaign documented the
same count of cross-device differences. The user has now authorized PRO 5000
for completion, and the FFmpeg 9.0.2 candidate passed all 204 immutable goldens
there. Retain the PRO 6000 mismatch as a hardware limit; same-device agreement
alone remains insufficient. Final package changes require fresh gates against
the final bytes. No golden or tolerance change is authorized.
