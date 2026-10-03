# Portable Linux build and runtime

The portable build provides a single executable with FFmpeg/FFprobe 9.0.2 for
production. The final artifact passed all 204 immutable parity checks in three
Linux userlands on the PRO 5000, and two clean offline builds are byte-identical.
Source-material and vendor-binary audits and final functional qualification are
complete; see the [task list](linux-distribution-tasks.md) and
[PRO 5000 report](linux-distribution-pro5000.md). No artifact has been published.
Historical codecs remain confined to private validation tools. Earlier PRO 6000
results are retained in the [original report](linux-distribution-results.md)
and [media migration report](ffmpeg-upgrade.md).

## Build locally

Use Linux x86-64. The host needs Python 3.10+, `skopeo`, `umoci`, and permission
to trace child processes. On Debian/Ubuntu, install the build helpers with:

```sh
sudo apt-get update
sudo apt-get install python3 skopeo umoci
bash scripts/build_linux.sh --jobs 8
```

The build fetches checksum-pinned inputs, freshly unpacks the digest-pinned
CUDA 13.0.3 / Ubuntu 22.04 builder, installs its locked packages, and compiles
inside it. It does not install CUDA or development libraries into the host.
No GPU, Docker daemon, FUSE or nested user namespace is required for compilation.
PRoot supplies filesystem translation, not security isolation: build trusted
source only. GPU execution requires a compatible NVIDIA driver and device.

The default output directory is `bin/linux-release/`; the work tree and
verified downloads are under `outputs/linux-build/`. Output and work directories
must be fresh. Build an existing checkout or source archive; dirty content is
recorded by hash. Existing `make` and `scripts/setup_linux.sh` workflows remain
available, and ordinary `make` still writes `bin/h3cli` and `bin/libh3.a`.
Budget roughly 30 GiB per clean build in addition to downloaded inputs, models
and inference caches. Running two clean builds concurrently needs room for both.
LoRA variants are separate from the runtime cache and can have tens of GiB of
logical data even when the filesystem supports reflinks.

To fetch once and then build without fetching anything:

```sh
bash scripts/build_linux.sh --fetch-only --cache outputs/linux-build/cache
bash scripts/build_linux.sh --offline --cache outputs/linux-build/cache \
  --work-dir outputs/linux-build/offline-1 --output bin/linux-offline-1
```

`--offline` requires every locked input to exist and pass its checksum. It does
not disable networking for arbitrary source code; the pinned build recipes
perform no network fetches in their compile/install phases. The native CUDA
wheel hashes come from the same requirements file as the Ubuntu setup.

## Run a prepared artifact

Only the executable is needed on the target. Models, the NVIDIA driver, Linux
x86-64 with glibc 2.35 or newer, and writable/executable cache storage remain
external. The native SGLang CPU RNG path also requires AVX2 and FMA.
CUDA 13.0.3 uses the existing minimum driver requirement 580.126.20;
the driver must also support the actual GPU. Compiled architectures and physical
qualification are separate; do not infer GPU support from a successful build.

```sh
cd bin/linux-release
sha256sum -c h3cli-linux-x86_64.sha256
chmod +x h3cli-linux-x86_64
H3CLI_BUNDLE_INFO=1 ./h3cli-linux-x86_64
./h3cli-linux-x86_64 --help
```

The diagnostic works without a GPU/driver and does not extract anything. Normal
execution loads the native CUDA runtime and therefore requires the host driver.
The checksum detects corruption; it is not a publisher signature.

Run from the directory containing your `models/` folder, or provide explicit
model paths. Default paths stay relative to the caller's working directory:

```text
models/MiniMax-H3
models/image-vae/minimax_h3_t1_image_vae_step1597.safetensors
models/latent-upscale/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors
models/preview-vae/taeh3.safetensors
```

For example, with the executable copied into that directory:

```sh
./h3cli-linux-x86_64 -p 'A small boat on a quiet pond.' \
  --width 256 --height 256 --frames 22 --steps 2 -o preview.mp4
./h3cli-linux-x86_64 --server --server-state-dir outputs/server
```

The [server API](../features/server.md) is unchanged. Workers execute the
extracted core directly, inherit the pinned runtime, and do not re-extract the
download. Inputs, output paths, LoRA caches and server state keep their normal
CLI rules. No Python, system FFmpeg, CUDA toolkit or compiler is needed to run.

## Cache and environment

First launch extracts verified files into
`${XDG_CACHE_HOME:-$HOME/.cache}/h3cli/runtime/<runtime-id>/`. The ID incorporates
both compressed content and the file table. Concurrent first launches share an
extraction lock; complete runtimes are published atomically. Warm launches check
ownership, file types and saved metadata without reading gigabytes again.
Every direct core/worker launch also checks readiness metadata.

Set `H3CLI_RUNTIME_CACHE` to an absolute writable directory on an executable
filesystem. A `noexec` cache cannot run the core. Keep enough space for the
compressed download plus an extracted runtime; interrupted attempts can leave
private `.extract-*` directories. A later launch safely starts a new attempt.
There is no automatic cache eviction or in-place repair of an active runtime.

If a cache entry is corrupt, stop **all** processes using that version—including
servers, workers and FFmpeg—then remove only that entry and run again. Likewise,
remove abandoned `.extract-*` directories only when no extraction is running.
`H3CLI_BUNDLE_INFO=1` reports the exact entry. Never delete a live server's files.

The launcher strips inherited dynamic-loader injection/search variables. The
core binds cuBLAS/cuDNN/JPEG and FFmpeg/FFprobe 9.0.2 to its payload. Native source builds retain their normal setup overrides;
packaged runs ignore stale values for these resources. Device/rendering flags
and explicit test controls remain available. The six-evaluation test ceiling is
not embedded in the released default environment.

Moving the same package to another cache root preserves verified dependency
identity for conditioning/checkpoint reuse on the same build and device.
Changed runtime content, source/build identity, arithmetic, model or device
still invalidates incompatible state.

For an unpacked developer runtime, verify content and refresh its machine-local
readiness marker after copying it:

```sh
python3 scripts/linux/package.py seal bin/linux-release/linux-runtime
```

The single-file executable performs extraction and sealing itself.

## HTTPS and offline operation

Local inputs and offline inference require no network access or CA store.
Automatic model downloads and opted-in server URL imports use the target
system's CA store. Discovery checks
Debian/Ubuntu and common RPM-distribution CA paths. `CURL_CA_BUNDLE` takes
precedence over `SSL_CERT_FILE` when explicitly set. An invalid/missing store
fails certificate verification; verification is never disabled. Minimal OCI
images may need a system CA bundle provisioned for HTTPS, independently of
h3cli. No aging CA bundle is embedded in the payload.

## Validation artifacts

`linux-runtime/` holds the exact native payload. `linux-validation/` contains
matching developer probes, historical regression codecs and provenance; these
are excluded from the single executable and remain private test materials.
`build.json` records source and lock identities, compiler/runtime versions,
capabilities and media configurations.

The builder also emits `h3cli-linux-x86_64-sources.tar.gz` and its checksum.
Publish this companion source archive beside the executable when publishing a
release. It includes the exact project source, pinned upstream sources and 28
Ubuntu source packages with their packaging patches, including dependencies
statically linked into the launcher. `source-materials/manifest.json` records
all file hashes; the runtime's `share/licenses/redistribution.json` maps every
binary to its source and retained license notice. See the
[source and relinking instructions](../../scripts/linux/SOURCE-MATERIALS.md).
NVIDIA libraries retain their original bytes and license terms. Their original
relative search paths stay inside the runtime; open-source ELF paths are
adjusted during packaging. FFmpeg/libx264 carry GPL terms, and the runtime
retains their notices alongside the matching sources. This software is based
in part on the work of the Independent JPEG Group. No public hosting happens
during a build.

The regression runner accepts `--runtime`, `--validation`, and `--artifact` to
validate existing bytes. It checks their identities before and after execution,
launches C0 through the final outer executable, and explicitly activates the
verified `linux-validation/media/profile.json`: FFmpeg 4.2.2 for encoding and
6.1.1 for decoding/probing. The package constructor verifies the profile and its
tool hashes before applying this test-only selection. Ordinary inherited media
overrides cannot replace the production tools. Current-feature validation uses
9.0.2; historical tools never enter the standalone payload. All 204 outputs, 12 fixtures and the 12-minute
post-verification execution deadline remain mandatory. A same-device baseline
comparison is diagnostic and never converts an immutable-golden failure into
a pass.

## Models and network access

The executable embeds the [pinned catalog](../../src/models/catalog.json) and
uses its packaged libcurl/CA store to download missing models automatically.
No Python, Hugging Face CLI or external curl is needed. Run from any directory
with `--models-path /writable/models`; `-d` overrides only the main model.
`--offline`/`H3_OFFLINE=1` requires existing local files. The extracted runtime
remains immutable; model locks, parts and receipts live beside model
destinations. See [download and recovery instructions](../features/model-downloads.md).
The source archive includes `src/models/`, its catalog and deterministic
embedding generator; no weights or credentials are bundled.
