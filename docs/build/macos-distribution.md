# Standalone h3cli for Apple Silicon

The macOS package targets arm64 Macs running macOS 26 or newer. The downloaded
`h3cli-macos-arm64` contains the application, Metal shader sources and
FFmpeg/FFprobe 9.0.2 with static x264. Models are separate. Runtime execution
requires only macOS system libraries/frameworks, not Homebrew, Python, Xcode,
Command Line Tools or a separately installed FFmpeg.

Local builds are **ad-hoc signed, unnotarized development artifacts**. Public
Developer ID signing and clean Gatekeeper qualification are separate release
steps; see [qualification results](macos-distribution-results.md).
The locally prepared [Developer ID release candidate](../../bin/macos-public-release/)
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
bash scripts/build_macos.sh --fetch-only --cache outputs/macos-build/cache
bash scripts/build_macos.sh --offline --jobs 8 \
  --cache outputs/macos-build/cache \
  --work-dir outputs/macos-build/run-001 \
  --output bin/macos-release
```

Use fresh work/output directories. Sources are snapshotted, including dirty and
untracked code, without models or build outputs. The authoritative FFmpeg pin
comes from the Linux lock; the macOS x264 pin identifies revision r3222/b35605a.
Both media helpers are built from source with dependency autodetection disabled.
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

The release command accepts an installed **Developer ID Application** identity
and an existing Keychain notary profile. It never takes account passwords or
private keys. Ordinary builds do not submit to Apple.

These are separate prerequisites: a successful `notarytool store-credentials`
does not create a signing identity. `security find-identity -v -p codesigning`
must show **Developer ID Application** with its private key available. An
**Apple Development** identity does not qualify. Create the certificate for the
same team used by the notary profile, through Xcode's certificate management or
[Apple's Developer ID certificate workflow](https://developer.apple.com/help/account/certificates/create-developer-id-certificates/).

```sh
python3 scripts/macos/release.py --input bin/macos-release \
  --output bin/macos-public-release \
  --identity 'Developer ID Application: Example (TEAMID)' \
  --keychain-profile h3cli
```

This explicit command signs the inner executables with hardened runtime,
notarizes them in a supported container, embeds their accepted exact bytes,
signs the launcher, creates/signs the DMG, then notarizes, staples and validates
it. It retains submission logs, inner hashes and final checksums. It fails on
rejection or missing credentials; it never falls back to ad-hoc signing. A
failed run retains evidence and requires a fresh output directory for retry.
No public upload/hosting is performed.

To inspect a completed distribution on the current machine:

```sh
xcrun stapler validate bin/macos-public-release/h3cli-macos-arm64.dmg
spctl --assess --type open --context context:primary-signature --verbose=2 \
  bin/macos-public-release/h3cli-macos-arm64.dmg
codesign --verify --strict --verbose=4 --test-requirement '=notarized' \
  --check-notarization bin/macos-public-release/h3cli-macos-arm64
```

Use the `codesign` notarized requirement for the standalone CLI. The app-specific
`spctl --type execute` assessment can reject valid CLI code because it is not an
app bundle; the DMG uses `spctl --type open`. These checks follow
[Apple's product-type guidance](https://developer.apple.com/forums/thread/130560)
and do not replace testing on a fresh machine.

Actual release acceptance also requires fresh quarantine/trust tests of the
stapled DMG, cold online/offline extraction, helpers and Metal/server operation.
A local ad-hoc test or previously warmed trust cache does not satisfy that gate.
Apple cannot staple a ticket to a bare CLI executable, so independent inner-code
notarization plus outer DMG stapling requires physical offline trust validation.
See [Apple's custom notarization workflow](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow)
and [hardened-runtime requirements](https://developer.apple.com/documentation/security/resolving-common-notarization-issues).
