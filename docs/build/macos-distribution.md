# Standalone h3cli for Apple Silicon

The macOS package targets arm64 Macs running macOS 26 or newer. The downloaded
`h3cli-macos-arm64` contains the application, Metal shader sources and
FFmpeg/FFprobe 9.0.2 with static x264. Models are separate. Runtime execution
requires only macOS system libraries/frameworks, not Homebrew, Python, Xcode,
Command Line Tools or a separately installed FFmpeg.

Local builds are **ad-hoc signed, unnotarized development artifacts**. Public
Developer ID signing and clean Gatekeeper qualification are separate release
steps; see [qualification results](macos-distribution-results.md).
The previously prepared [Developer ID release candidate](macos-distribution-results.md)
has accepted inner/outer notarizations and a stapled DMG. Clean-environment and
fresh-trust qualification remain pending at the user's request.

## Build

Install the Apple toolchain and Python build prerequisites described by
`scripts/macos/lock.json`. `pkg-config` is a build-time prerequisite. The lock
checks compiler/linker/tool hashes and the SDK content inventory; changing the
lock requires qualification. No script changes the global Xcode selection or
installs software. `DEVELOPER_DIR` selects an existing Apple installation.

```sh
bash scripts/build_macos.sh --doctor
bash scripts/build_macos.sh --dependencies /path/to/macos-qualification/bundle.json \
  --fetch-only --cache outputs/macos-build/cache
bash scripts/build_macos.sh --dependencies /path/to/macos-qualification/bundle.json \
  --offline --jobs 8 \
  --cache outputs/macos-build/cache \
  --work-dir outputs/macos-build/run-001 \
  --output bin/macos-release
```

Use fresh work/output directories. Sources are snapshotted, including dirty and
untracked code, without models or build outputs. The authoritative FFmpeg pin
comes from the Linux lock; the macOS x264 pin identifies revision r3222/b35605a.
Both media helpers are built from source once, with dependency autodetection
disabled, when [creating the dependency bundle](dependency-bundles.md). Application
builds verify and reuse those bytes. Use `--dependency-cache /path/to/shared/cache`
to share bundles across checkouts. For an explicit full source rebuild, replace
`--dependencies PATH` with `--source-dependencies` in both commands.
The ordinary `make`/`bin/h3cli`/`bin/libh3.a` workflow remains available.

## Run and cache

Copy the executable from the DMG to a writable executable location, unmount the
DMG, and use Terminal. Nothing modifies PATH, shell profiles or model locations.

```sh
./h3cli-macos-arm64 --help
./h3cli-macos-arm64 --models-path /Volumes/Models --quality preview \
  -p 'A small wooden boat on a quiet pond' --width 256 --height 256 \
  --frames 22 --steps 2 -o boat.mp4
```

`--models-path` defaults to `models/` in the current working directory;
`MiniMaxH3` (with existing `MiniMax-H3` fallback), preview VAE, image VAE and
upscaler lookup retain their existing rules. `-d` and explicit auxiliary paths
retain precedence. Missing catalog models download automatically; `--offline`
or `H3_OFFLINE=1` prohibits this. Existing tokens, proxy and CA controls work.

The first invocation verifies and extracts into the private user cache
`~/Library/Caches/h3cli/runtime/<identity>`. Warm starts verify readiness before
executing the core. `H3CLI_RUNTIME_CACHE=/absolute/private/path` overrides that
cache; use a user-owned directory with mode 0700, without symlink components.
`H3CLI_BUNDLE_INFO=1 ./h3cli-macos-arm64` prints the manifest and selected cache
without extraction, model/network activity or Metal initialization.

The executable can be renamed, moved or launched from a read-only mounted DMG.
The current directory and CLI/server output behavior are preserved. All workers
use the same verified resources. Packaged runs ignore `H3_SHADER_PATH`, PATH
media tools, `H3_FFMPEG`, `H3_FFPROBE` and `H3_SGLANG_INPUT_FFMPEG` substitutions.
These overrides still work in ordinary source builds. `DYLD_*` overrides are
removed from packaged processes. Model and output settings are preserved.

Saved-state compatibility includes source/configuration, content identities,
models and the native backend checks. Relocation of the same package/cache does
not change resource identities. Compatibility across arbitrary native builds,
packages or macOS framework updates is not promised.

A modified cache is rejected. Stop all processes using that runtime before
removing its specific identity directory and retrying. Never overwrite an active
runtime. Interrupted extraction can leave `.stage-*` directories; remove them
only when no launch/extraction is running. Older versions are retained, with no
automatic cache pruning. Low disk space or a non-executable cache produces an
error; select another private local cache location.

## Corresponding sources and relinking

Keep `h3cli-macos-arm64-sources.tar.gz` beside distributed artifacts. It includes
exact application, FFmpeg and x264 source materials, licenses, lock and build
recipes. Its `BUILD.txt` gives the offline rebuild command. Apple frameworks and
SDKs are system/build prerequisites and are not redistributed. Models, private
media and CUDA components are excluded. FFmpeg/x264 carry GPL source obligations;
links to upstream moving branches are not a substitute for these materials.

## Public release workflow

Use the [five-command release workflow](../release/README.md). Its one-time
`setup-signing` discovers or helps provision the Developer ID Application
identity and Keychain notary profile. An Apple Development identity cannot sign
a public Developer ID distribution. Ordinary builds never submit to Apple.

`build-packages` signs the inner executables with hardened runtime, notarizes
that runtime, embeds its accepted bytes, signs the launcher and DMG, notarizes
the outer image and staples it. Submission IDs, unique upload names and hashes
are retained so a retry resumes completed submissions without re-signing accepted
code. Missing credentials or rejected notarization fails the stage.

`test-packages` verifies the signatures, identity, notarization requirement and
stapled DMG, mounts the image and checks its executable against the standalone
bytes, then runs rendering/media/server tests outside the checkout with isolated
runtime caches. `publish` verifies downloaded packages before publication. These
checks are automatic; the low-level `scripts/macos/release.py` is an internal
entry point, not another manual release step.

The standalone CLI uses the `codesign` notarized requirement; the DMG uses
`spctl --type open`. The app-specific execute assessment is inappropriate for
bare CLI programs; see [Apple's product-type guidance](https://developer.apple.com/forums/thread/130560).
A ticket cannot be stapled to a bare executable, so the DMG is the stapled
artifact. Fresh installations/accounts, browser quarantine and physical offline
trust experiments are optional investigations. Passing on an existing account
is not evidence of fresh-machine trust, and release notes state that limitation.
