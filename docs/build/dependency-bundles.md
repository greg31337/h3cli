# Reusable build dependencies

Dependency production and application releases are separate operations. Build
and qualify a Linux x86-64 or macOS arm64 dependency bundle when its inputs
change, then reuse its pinned bytes for application builds and release candidates.
The release coordinator requires qualified bundle references on both machines.
It never starts a dependency source build implicitly.

## Contents and identity

The Linux bundle contains a prepared OCI builder image: the locked CUDA
toolchain/userland, installed distro packages, ICU, TurboJPEG, FFmpeg/FFprobe,
NVIDIA runtime wheels, and the CUTLASS/cuDNN frontend development headers.
Application CUDA kernels and validation probes are compiled for each application
build. The producer retains notices and generated dependency configuration inside
the image, so consumers do not need the original build directory.

The macOS bundle contains the qualified FFmpeg/FFprobe tools, static x264/media
development files, configuration and license notices. Apple tools, SDKs and system
frameworks remain host prerequisites and are not included. Each application build
copies media tools before signing; the shared cache is never a signing target.

Both bundles include the relevant source archives and dependency recipes. Linux
historical regression codecs remain private validation inputs; normal packaged
media uses the production version. Bundle payloads contain no models, signing
credentials or generated qualification videos. Treat these as build artifacts,
separate from the public standalone executable and its companion source archive.

| Input/output | Linux x86-64 | macOS arm64 |
| --- | --- | --- |
| Compiled dependencies | ICU 74.2, TurboJPEG 2.1.5, production FFmpeg/FFprobe 9.0.2 | FFmpeg/FFprobe 9.0.2, static x264 r3222/b35605a |
| Development inputs | Installed distro headers/libraries, CUTLASS and cuDNN frontend headers, pinned NVIDIA wheels | x264 headers/static archive and media development files |
| Historical media | FFmpeg 4.2.2 encoder and 6.1.1 decoder inside the private builder/validation inputs | No historical CUDA media |
| External build prerequisites | Python, skopeo, umoci, GNU cp and ptrace support | Locked Apple compiler/linker/SDK, Python, pkg-config and disk-image/signing tools |
| Compiled for each application | Application, CUDA kernels, launcher, matching validation probes | Application, launcher, Metal/ANE and other validation probes |

The platform locks remain authoritative for exact hashes and configurations.
Linux's driver and models are runtime/qualification prerequisites; Apple SDKs
and frameworks are never copied into a dependency archive.

There are two identities:

- The **recipe ID** hashes the platform, dependency lock, compiler/SDK/sysroot,
  architecture/deployment options, source/vendor hashes and relevant build scripts.
  Application code, Git commit, release version, run name and installation path
  do not enter this key. Changing dependency build machinery conservatively
  invalidates the recipe, even if the resulting library bytes might be identical.
- The **archive SHA-256** identifies the actual bundle bytes. A canonical manifest
  inventories every payload file's digest, size and permissions. The Linux OCI
  blobs preserve their own filesystem metadata and are individually hash-bound.

A qualification report is separate from the archive to avoid circular hashes.
It binds the archive digest to a reference application/source identity and actual
test results. A small `bundle.json` pins both archive and report hashes and sizes.
Release configuration saves this reference; a readiness file cannot replace it.

## Build once

Use the qualified PRO 5000 for Linux and the M4 with the locked Apple tools for
macOS. Source/toolchain download caches can be reused from earlier builds. Work
and output directories must be fresh; failures preserve their logs.

Use an ordinary source checkout without symlinks or special permission bits.
On a shared volume that inherits setgid onto checkout directories, prepare a
private checkout with ordinary directory permissions before production. The
producer normalizes its own Linux payload root; archived recipe directories
still preserve source permissions and are checked by the archive policy.

```sh
python3 scripts/dependencies.py identity --platform linux-x86_64
python3 scripts/dependencies.py build --platform linux-x86_64 \
  --cache /path/to/shared/linux-downloads \
  --work-dir outputs/dependencies/linux-build-001 \
  --output outputs/dependencies/linux-bundle-001 --jobs 8

python3 scripts/dependencies.py build --platform macos-arm64 \
  --cache /path/to/shared/macos-downloads \
  --work-dir outputs/dependencies/macos-build-001 \
  --output outputs/dependencies/macos-bundle-001 --jobs 8
```

Add `--offline` to production when all locked inputs and tools are already in the
download cache. The producer emits an immutable `.tar` archive, `candidate.json`
and build evidence. A candidate is intentionally unusable as normal release input
until qualification succeeds. Nothing is uploaded by these commands.

## Qualify a bundle

Qualification builds the current application against the candidate, exercises its
actual packaged media, relocation and current features/server, and verifies that
the shared dependency bytes remain intact. Linux also runs the complete 204-output
reference comparison, all four distro/mode comparisons and runtime isolation tests.
macOS runs its native package tests and Metal feature matrix. The existing feature
suite includes a real model-download test, so qualification needs network access.
The selected model root must already contain all pinned model groups, including
the preview VAE, image VAE and latent upscaler. Provision them with an existing
development or installed executable before starting the qualification:

```sh
h3cli --models-path /path/to/models -d /path/to/models/MiniMax-H3 \
  --download-models all
```

```sh
python3 scripts/dependencies.py qualify \
  --bundle outputs/dependencies/linux-bundle-001/candidate.json \
  --cache /path/to/shared/linux-downloads \
  --dependency-cache /path/to/shared/dependency-cache \
  --models-path /path/to/models --model /path/to/models/MiniMax-H3 \
  --out outputs/dependencies/linux-qualification-001 --jobs 8
```

Use the macOS candidate and cache paths on the Mac. Only successful qualification
creates `qualification.json` and `bundle.json` in the output directory. Failed
attempts never receive a passing receipt. Qualification records the reference
commit and whether its checkout was dirty; normal releases still require a clean,
reviewed application commit. Final release artifact tests and Apple signing/
notarization remain separate and mandatory.

Passing an already qualified `bundle.json` to `qualify` builds and tests another
application revision through the normal qualified-consumer path. Its application
build remains offline and does not recompile the dependency libraries.

## Reuse in application builds

```sh
bash scripts/build_linux.sh \
  --dependencies /path/to/linux-qualification/bundle.json \
  --dependency-cache /path/to/shared/dependency-cache --fetch-only

bash scripts/build_linux.sh \
  --dependencies /path/to/linux-qualification/bundle.json \
  --dependency-cache /path/to/shared/dependency-cache --offline \
  --work-dir outputs/linux-build/app-001 --output bin/linux-app-001
```

Use `scripts/build_macos.sh` and the macOS reference for Apple Silicon. The
`--fetch-only` command acquires and verifies the qualified bundle; it does not
compile dependencies. Warm `--offline` application builds use the cache. A local
archive can seed an offline cache directly; an uncached HTTPS URL fails in offline
mode. Application `build.json` and source materials record the bundle identity.

An explicit `--source-dependencies` option retains the original source-build path
for development and reproduction. It is mutually exclusive with `--dependencies`.
The internal `--dependency-candidate` path exists for qualification and is never
selected by the release coordinator. Native `make` and setup workflows remain
available.

## Configure releases

For individual platform scripts, pin a qualified reference when configuring a
fresh candidate:

```sh
./release/configure.sh --repo OWNER/REPO --tag v0.6.0 --run v0.6.0-01 \
  --models-path /path/to/models \
  --dependencies /path/to/qualification/bundle.json \
  --dependency-cache /path/to/shared/dependency-cache \
  --build-cache /path/to/shared/tool-downloads
```

For the one-command Mac coordinator, add these fields to each platform's private
entry in `outputs/release/machines.json`:

```json
{
  "dependencies": "/path/to/qualification/bundle.json",
  "dependencies_sha256": "REPLACE_WITH_SHA256_OF_BUNDLE_JSON",
  "dependency_cache": "/path/to/shared/dependency-cache",
  "build_cache": "/path/to/shared/tool-downloads"
}
```

Calculate the reference-file digest with `shasum -a 256` on macOS or `sha256sum`
on Linux. The Linux paths refer to the remote machine. Place shared caches outside
individual candidate checkouts so later releases use the same downloads and
verified bundle extraction. Changed references/settings require a new candidate;
resuming an existing candidate rechecks its saved identities and receipts.

## Storage, recovery and verification

Bundle references support local paths, `file:` URLs and credential-free HTTPS
locations. To move an archive/report to durable storage, update their locations
in a copy of `bundle.json`, retain their exact hashes/sizes and pin the new
reference-file digest in machine settings. There is no implicit `latest` lookup
or upload. Keep required source materials alongside the stored artifacts.

The shared cache uses archive digests as directory names. Concurrent users lock
the same identity; staging is published atomically after extraction and complete
verification. Archive entries cannot contain symlinks, hardlinks, device files,
duplicates or path traversal. Warm reuse checks the archive, qualification and
payload inventory again, including the manifest's correspondence to the archive.

Inspect `CACHE/bundles/ARCHIVE_SHA256/manifest.json` for the recipe, file inventory
and producer audit. Run the platform builder with `--dependencies REF --fetch-only
--offline --dependency-cache CACHE` to reverify a provisioned cache. This checks
bytes rather than merely displaying a stored readiness status.

Corruption fails instead of triggering an invisible replacement. Preserve the
diagnostic, use a fresh cache or remove only the affected inactive cache entry,
then fetch from the trusted reference again. Never edit cached binaries or
manifests to accept mismatched content. Interrupted staging directories are not
ready bundles. Completed archives and unrelated candidates remain reusable.

Build children use owned process groups. The remote release runner watches its
SSH input stream and cancels its workers on EOF, Ctrl+C or termination. Canceled
commands produce no successful completion receipt. Tests cover cancellation,
offline misses, corruption, cache invalidation and concurrent reuse:

```sh
python3 tests/test_dependency_bundles.py
python3 -m unittest discover -s release/tests
```

Follow [CONTRIBUTING.md](../../CONTRIBUTING.md) for numerical and final-artifact
qualification. Dependency reuse does not relax those checks or change saved-state
compatibility. Corresponding-source, licensing and vendor binary requirements
continue to apply to the final package.
