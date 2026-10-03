# macOS standalone executable and notarized distribution

Status: **implemented; local package qualified**. Updated 2026-09-30.
Developer ID signing, inner/outer notarization and DMG stapling are complete.
Clean-environment and fresh-trust qualification remain pending at the user's
request. See the [implementation guide](macos-distribution.md) and
[qualification record](macos-distribution-results.md).
Implementation checklist: [todo.md](../todo.md). The completed model-download
work is preserved in [its checklist](../features/model-downloads-tasks.md) and
[qualification report](../features/model-downloads-results.md).

## Outcome and scope

Make running h3cli on an Apple Silicon Mac a download-and-run operation. Produce
one `h3cli-macos-arm64` executable containing the application, Metal resources,
FFmpeg/FFprobe 9.0.2 and their required redistributable dependencies. Extract a
verified runtime into a private, versioned user cache on first use. A warmed
runtime needs no extraction. Provide that executable inside a signed, notarized
DMG for public distribution, with companion source materials and checksums.

Users need no Homebrew, Python, compiler, Xcode, Command Line Tools, separately
installed FFmpeg, administrator privileges or container engine to run it.
macOS supplies Metal, MPS/MPSGraph, CoreML and the other system frameworks.
Model weights remain external and use the existing automatic download policy.
Keep the CLI and queued server, all platform-appropriate features, cancellation,
model/output path semantics and current saved-state validation.

Preserve ordinary `make`, `bin/h3cli`, `bin/libh3.a`, the macOS setup script and
the Linux release workflow. This work adds packaging, not new inference kernels,
model revisions, quality defaults, an Intel backend, a GUI, automatic updates,
App Store distribution or a system-wide installer. Public hosting is separate
from building and qualifying release artifacts.

## Baseline and supported envelope

Inspection of the current local build found:

- An approximately 1.7 MB arm64 h3cli executable linked only to Apple system
  frameworks/libraries, including system ICU, SQLite and libcurl.
- `LC_BUILD_VERSION` declaring macOS 26.0 as the deployment target. The inspected
  binary used SDK 27.0; this observation is not a new required SDK pin.
- Approximately 21 MB each for FFmpeg and FFprobe, both linked to Homebrew's
  `libx264.165.dylib`. Copying these tools alone would retain that dependency.
- A required external `src/metal/shaders.metal`; other shader programs include
  generated sources embedded in the executable. Shader lookup currently permits
  cwd and environment overrides.
- A Linux-specific launcher and packaged-runtime implementation under
  `src/runtime/`, including an appended archive/footer, Linux cache conventions
  and ELF library handling. These need explicit macOS implementations.

The resulting package should be much smaller than the CUDA distribution because
Apple supplies the GPU frameworks. Measure final compressed/extracted sizes;
the current component sizes are estimates, not a release-size promise.

| Property | Initial contract |
| --- | --- |
| Architecture | Native arm64, Apple Silicon; no Rosetta or universal Intel slice |
| Deployment target | Explicit macOS 26.0 for every Mach-O, including launcher and helpers |
| Builder | Apple Silicon macOS with a recorded, pinned Apple compiler/linker/SDK selection |
| Physical qualification | Local M4 on its installed macOS version; record exact OS and hardware |
| Other chips/OS releases | Preserve existing capability guards; distinguish compiled support from physical testing |
| Production media | FFmpeg/FFprobe 9.0.2, existing codec policy and qualified x264 source |
| Model acquisition | Existing catalog, automatic missing downloads, `--models-path`, explicit overrides and offline policy |
| Runtime location | `~/Library/Caches/h3cli/runtime/<runtime-id>` by default |

Audit SDK availability and load commands so a newer SDK does not silently raise
the deployment target. A deployment target is not proof of successful operation
on every macOS 26 release or Apple Silicon GPU. Report the tested OS explicitly;
qualification of the oldest runtime requires an available compatible machine or
boot environment. Do not reinstall the local M4 or present a GPU-less VM as
Metal qualification. Preserve M5/Metal 4 and Metal/ANE guards;
hardware unavailable on M4 remains explicitly unqualified.

## Build entry point and pinned inputs

Add `scripts/build_macos.sh`, backed by focused helpers under `scripts/macos/`.
Proposed interface, matching the Linux builder where practical:

```sh
bash scripts/build_macos.sh --doctor
bash scripts/build_macos.sh --fetch-only --cache outputs/macos-build/cache
bash scripts/build_macos.sh --offline --jobs 8 \
  --cache outputs/macos-build/cache \
  --work-dir outputs/macos-build/run-001 \
  --output bin/macos-release
```

`--doctor` reports missing/wrong build prerequisites without changing the host.
The normal build needs an installed supported Apple toolchain and build-time
Python; it must not silently install Homebrew, switch the global Xcode selection
or download models. Use `DEVELOPER_DIR`/an explicit SDK selection locally to the
build. Validate arguments before fetching. Work/output directories must be
fresh; do not clean or overwrite native builds or earlier releases.

Record exact compiler, linker, SDK build identity/content inventory, OS, build
tools, deployment target, configuration, source snapshot, generated files and
dependency hashes in a macOS lock/provenance record. Resolve real versions and
hashes during implementation; do not invent SDK or x264 pins in this plan.
Reject an unrecognized toolchain unless its lock is explicitly updated and
requalified. Apple SDKs/frameworks are host prerequisites, not source payloads
to redistribute. An offline build requires those tools and all locked sources
already present. Signing/notarization networking is a separate phase.

Share the authoritative FFmpeg 9.0.2 source pin with the existing build setup.
Pin the actual qualified macOS x264 source/configuration; do not infer a source
revision from the dylib ABI number or substitute Linux's codec build. Prefer
static x264 linkage into the two media helpers. If a private dylib is necessary,
include its full closure and use confined `@loader_path`/`@rpath` references.
Disable dependency autodetection and reject Homebrew/MacPorts or developer-path
linkage. Preserve existing media capabilities, GPL configuration and encoder
settings; no codec upgrade is part of this work.

Use isolated source snapshots for Git checkouts and source archives, including
dirty/untracked source content and generated catalogs. Exclude models, output
media, caches, credentials, object files and prior packages, while retaining
`src/models/` and checked-in test fixtures. Apply explicit arm64/deployment flags
to all languages and dependencies without CPU-specific `-mcpu=native` or new
global fast-math flags. Retain the existing shader-generation and optional
backend feature set. Keep mutable CoreML/ANE and quantization caches outside
the immutable extracted runtime.

## Release artifacts and runtime layout

All executable, library and final distribution outputs remain in ignored
`bin/macos-release/`; build evidence/cache belongs in `outputs/macos-build/`.

| Artifact | Purpose |
| --- | --- |
| `h3cli-macos-arm64` | Portable single executable; locally ad-hoc signed or Developer ID signed, explicitly labeled |
| `h3cli-macos-arm64.dmg` | Public-distribution container after Developer ID signing, notarization and stapling |
| `h3cli-macos-arm64-sources.tar.gz` | Complete redistributable corresponding sources, configuration and build/relink recipes |
| `libh3.a` | Native macOS library; retains local-file inference semantics |
| `build.json`, checksums and manifests | Source/toolchain/runtime/signing identities and reproducibility evidence |
| `macos-runtime/`, `macos-validation/` | Unpacked runtime and private probes for audit/qualification; not required beside the executable |

The DMG contains the executable plus usage/source-material notices, not an app
that starts a GUI on double-click. Document copying it to a writable executable
location and running it in Terminal. There is no automatic PATH or shell-profile
modification. The source archive must accompany a distributed release.

Suggested extracted runtime layout:

```text
bin/h3cli
tools/ffmpeg
tools/ffprobe
lib/                         # only if non-system dylibs are required
src/metal/shaders.metal       # plus any other audited external resources
share/h3cli/runtime.json
share/licenses/...
```

Audit direct/weak/re-exported Mach-O loads, runtime loads, media subprocesses,
shader resources and generated CoreML resources. System references must be on
an explicit Apple-system allowlist. Every other dependency must resolve inside
the payload. Audit architecture, minimum OS, install names and rpaths for every
Mach-O; the h3cli executable alone is insufficient. Source builds may retain
their existing developer overrides.

## Mach-O embedding, extraction and execution

Prove the container/signature arrangement early with a small signed helper.
Embed a bounded compressed archive and canonical manifest in a dedicated,
read-only Mach-O data section, located through validated load-command/section
bounds. The signed executable must cover these bytes. Do not copy the Linux
end-of-file footer assumption: `codesign` adds signature data and can change
file layout. Never append or modify a payload after signing. Keep the archive
format/manifest validation shared where practical, with distinct platform and
schema identifiers; leave the existing Linux wire format compatible.

The archive contains regular files only, explicit modes/sizes, bounded compressed
and expanded lengths, and full SHA-256 hashes. Reject traversal, absolute names,
duplicates, ancestor conflicts, symlinks, special files, invalid arithmetic and
decompression bombs. The bootstrap links only to system APIs or deliberately
embedded audited code; it cannot require an installed JSON, crypto or archive
tool. Prefer existing parser/hash code or Apple APIs over new runtime packages.

The runtime identity binds the platform/schema, launcher protocol/code identity,
canonical manifest and final payload bytes. Avoid self-referential hashes that
include a signature inside the data it signs. Developer ID timestamps and DMG
tickets are recorded separately from reproducible build-content identities.
Validate actual signed files and extracted bytes, not an unsigned approximation.

Preserve `H3CLI_RUNTIME_CACHE` as the explicit cache override. Otherwise obtain
the user's macOS caches directory, giving the default shown above. Do not use
cwd, the DMG, the executable's directory, a shared temporary directory or the
models root for runtime extraction. `H3CLI_BUNDLE_INFO=1` should report platform,
identities, sizes and selected cache without extracting, loading Metal or
downloading models.

Create private user-owned directories and files with restrictive modes. Use
directory-relative operations, no-follow checks, free-space checks, a
cross-process extraction lock, unique staging directories, durable writes and
atomic publication. A crash or cancellation must never expose a partial ready
runtime. Validate ownership/type/content before reuse; maintain a fast verified
readiness record that detects changes to cached files. Refuse modified caches
with actionable recovery instructions. Never repair by overwriting a runtime
that another CLI/server/FFmpeg process may be using. Do not automatically remove
older runtime versions. Concurrent cold launches must converge safely.

Use the macOS executable-path implementation already in `src/platform.c`.
Preserve argv, cwd, stdio, environment policy, exit codes, process groups and
signals, then execute the verified core. Resolve server preparation/inference
workers from the same runtime. There must be no shell interpolation, mount,
downloaded bootstrap program or runtime dependency on `codesign`, `xcrun`,
`otool`, Python or developer tools; those are build/test tools only. macOS
performs its normal execution trust checks on the signed code.

## Packaged resource and state policy

Packaged execution selects shaders, FFmpeg and FFprobe exclusively from the
verified runtime. Bind paths from the canonical runtime root, including server
worker restarts. Existing `H3_SHADER_PATH`, media overrides, cwd files and PATH
must not replace packaged resources; document that these remain source-build
controls. Handle relevant `DYLD_*` injection/search variables deliberately
without discarding legitimate model, offline, proxy, token or user-output
settings. Do not make code depend on DYLD overrides surviving hardened runtime.

Keep runtime Metal compilation through the system APIs and include all required
source text/includes. Do not require Apple's optional offline Metal compiler on
the user's machine. The packaged resolver must preserve useful errors while
preventing a coincidental `src/metal/shaders.metal` in cwd from changing a render.
Retain native source lookup behavior outside package mode.

Use production FFmpeg/FFprobe 9.0.2 for CLI delivery, input media and server
inspection. Preserve relevant stdin/stdout, cancellation, media limits and
subprocess cleanup. Historical 4.2.2/6.1.1 tools remain exclusive to the private
CUDA golden-validation kit; they are not macOS runtime dependencies.

Represent packaged shader/media identity by verified content and recipe, not a
user's cache pathname. Moving the same launcher, using a symlink, or changing
the runtime cache root must not invalidate an otherwise compatible saved state.
Source/build flags, resource contents, codec recipe, models, selected backend
and existing native compatibility checks still matter. Do not promise that an
arbitrary native build can resume a packaged checkpoint, or that states survive
macOS framework changes. Keep the existing source-build state policy intact.

Package startup must not change cwd. `--models-path` continues to default to
cwd `models/`, including `MiniMaxH3` and the existing `MiniMax-H3` fallback;
`-d` and auxiliary overrides retain their precedence. Runtime cache location
must not relocate weights, states or outputs. Reuse the existing download
catalog, receipts, integrity/resume behavior and server preparation architecture.
Use macOS system libcurl/trust behavior with existing CA/proxy/token overrides;
do not embed an aging CA bundle or inherit the Linux CA-path search. Model
networking remains separate from notarization and build dependency fetching.

## Signing, notarization and distribution

Use two explicit workflows. A local/offline build produces runnable ad-hoc
signed artifacts and a DMG marked as unnotarized development output. A separate
release command accepts a Developer ID Application signing identity and a
Keychain notarization profile. It produces the public-distribution artifacts;
it must never silently downgrade to ad-hoc signing or hide a notary failure.

Apple requires Developer ID signing and hardened runtime for notarized software.
Retain hardened runtime for the core, bootstrap and helpers, adding only
demonstrably necessary entitlements after testing Metal/MPSGraph, CoreML/ANE,
server subprocesses and library validation. Do not disable Gatekeeper, remove
quarantine or apply broad signing exceptions as a packaging workaround.
[Apple signing and hardened-runtime guidance](https://developer.apple.com/documentation/security/resolving-common-notarization-issues).

The self-extractor contains nested executables that the notary service may not
discover inside a custom compressed section. Follow Apple's custom-installer
model: sign and submit the inner runtime code in a supported container first,
then embed those exact accepted bytes, sign the outer executable, and notarize
the distribution container. Keep explicit inner and outer submission records
and inspect all notary logs. No payload mutation follows final embedding/signing.
[Apple custom notarization workflow](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow).

Put the Developer ID-signed standalone executable in a signed DMG, submit that
DMG, staple its ticket, and verify the final distributed image and mounted
contents. Apple supports stapling DMGs but not bare standalone binaries.
An outer ticket does not by itself prove that newly extracted inner code works
on a clean offline machine: cold extraction and helper execution must be tested
with fresh trust/cache state. Treat this as an early feasibility gate. If the
single-executable scheme cannot meet the required trust behavior, record the
failure and propose a standard bundled layout explicitly; do not silently
change the deliverable or claim a previously warmed trust cache proves it.
[Apple ticket/stapling rules](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow).

Developer ID credentials and a usable notary account are prerequisites for
public-release qualification. Store secrets in Keychain or caller-provided
secure credentials, never source, command logs, manifests or archives. Ordinary
build/test commands must not upload anything. The explicit notarization phase
sends executable code to Apple, but no models, private media or credentials in
the payload. Hosting/uploading releases for users is outside this plan.

Unavailable credentials do not prevent implementing and testing the local
builder. They do leave actual signing/notarization/Gatekeeper qualification
pending; an ad-hoc build is not a notarized release and those tasks stay open.

## Source materials and reproducibility

Provide complete exact h3cli, FFmpeg, x264 and other redistributed dependency
sources, patches, configuration, licenses/notices and build/relink instructions.
Reuse the Linux source-material inventory approach without including CUDA
components or Apple's SDK/framework binaries. Preserve existing GPL media and
applicable dependency source obligations; do not replace the corresponding
sources with a link to a moving upstream branch.

Separate deterministic build stages from external signing services. With the
same locked Apple tools/SDK and dependency cache, two fresh offline builds from
different absolute directories must reproduce the normalized core/helper
content, generated shader/catalog data, manifest, development payload and source
archive. Make ad-hoc signing deterministic where it enters those artifacts.
Record and resolve timestamp, UUID, path and archive-order differences without
removing meaningful build/state identity. Developer ID secure timestamps,
notary tickets and DMG filesystem metadata can vary: record their final hashes
and reproducibility limits instead of promising identical notarized DMGs.

## Validation and acceptance

Use the local M4 for macOS work and short Metal samples. Use only the existing
local RTX PRO 5000 for Linux/CUDA regression protection:

```sh
ssh cuda-test
```

Preserve installed models and native binaries as reference inputs; use isolated
build/cache/model-download/output trees. No cloud nodes, driver changes or
large performance campaigns are needed. New test executables stay in `bin/`;
qualification evidence goes under `outputs/macos-distribution/`.

### Host and package failure tests

Test Mach-O section bounds, compressed/archive limits, invalid manifests,
truncation, hashes, executable modes and signed-payload mutation. Cover traversal,
duplicate/ancestor paths, symlink swaps, ownership/permissions, disk exhaustion,
partial writes, crashes, simultaneous first launches, stale locks/readiness,
changed cached files and concurrent old/new releases. Test damaged signatures
and source/dependency hash mismatches without contacting external services.
Use tiny fixtures and bounded memory/time measurements.

Check space/Unicode paths, symlinked or renamed launchers, changed cwd, read-only
launcher locations, mounted DMGs, explicit cache overrides, unavailable home
directories and a cache that cannot execute. Include stdin/stdout, exit status,
Ctrl-C/SIGTERM and child cleanup for CLI, server, model helpers and FFmpeg.
Check `--help`/bundle inspection, model prefetch and offline behavior without
requiring a model or compiler. Prove that hostile cwd shader/media files and
resource environment overrides cannot substitute packaged content.

Run the current host/Metal suites, downloader/server suites and package tests.
Use sanitizers where they meaningfully exercise the new parsers/extractor.
Audit the final application, helpers and library for compiler warnings and
unexpected loads; inspect actual running processes in addition to static load
commands. Keep release toolchain tests separate from runtime-only checks.

### Bounded real-model matrix on M4

Use a fixed prompt/seed and existing qualified weights. Compare ordinary native
and packaged execution on the same Mac/OS using the exact same pinned media
tools and shader contents. Check shapes, audio duration, full decode and exact
output/latent equality where the existing contract requires it. Do not invent
a Metal-versus-CUDA golden contract or replace a mismatch with relaxed checks.

| Case | Required coverage |
| --- | --- |
| Text video | Full and preview VAE, 256×256, 22 frames, two steps |
| Anchors | First/last images; identical crop behavior |
| References | Image plus video/audio, 256×256, 56 frames, two steps |
| Still | 256×256 generation, save latent and exact decode round trip |
| Saved AV/conditioning | Full/preview and minimal decoder-only loading; conditioning reuse |
| Continuation/bridge | 56-frame source and compatible 39-frame context; two steps, bounded bridge step |
| Sampler resume | Pause after one of two steps, resume and compare with uninterrupted output |
| Upscale | 256×256/56-frame source to 512×512, two refine steps; pause/resume comparison |
| Relocation | Resume compatible package state after moving launcher/changing runtime-cache root, with model identity unchanged |
| Server | Real cold/warm jobs, artifact fetch, cancellation/restart and worker resource identity |
| Optional Metal features | Existing bounded probes for exposed native/Q8/SOL/ANE paths where supported on M4; preserve unsupported-hardware errors |

Use fixtures for most fault and transfer permutations. Perform one genuine
automatic preview-VAE download into an empty auxiliary destination through the
packaged CLI and one through the server; verify bytes/progress, then repeat
offline. Exercise `--models-path`, all auxiliary overrides and existing complete
read-only models. Test offline use without usable CA/network settings. Do not
redownload the hundreds of gigabytes of already qualified main weights.

### Runtime-only and distribution qualification

Run the final artifact outside the checkout under a clean user/account and
minimal environment. A PATH-only check is insufficient: demonstrate via load,
file-access and subprocess evidence that Homebrew, development directories,
Python, SDK/compiler tools and system FFmpeg are not used. Do not uninstall the
developer's tools to simulate this. Use an available clean macOS installation
when possible; label an isolated account on a development host accurately and
record any limits of its isolation. Tool absence must be established by evidence.

For the public artifact, use a browser/quarantine-equivalent download of the
stapled DMG and fresh execution-trust state. Assess and mount it normally, copy
the CLI, unmount, then run from another directory with an empty runtime cache.
Test first-run extraction both online and offline; separately test an already
prepared runtime offline. Provision models beforehand for offline inference.
Verify helper/core signatures survive extraction, Metal shader compilation works
without developer tools, and server workers inherit the same runtime. Record
code-signing, notary, stapling and Gatekeeper evidence; do not strip quarantine
or reuse prior successful trust decisions as the sole test.

### Linux and numerical regression protection

Follow [CONTRIBUTING.md](../../CONTRIBUTING.md) after every coherent code/build/test
change. On the local PRO 5000 run the full recorded CUDA gate, offline, with
all **204 expected hashes and 12 fixtures unchanged**. For the final source,
also rebuild and check the Linux standalone package with its private validation
kit and run the full artifact gate. Retain historical 4.2.2/6.1.1 codecs only
there; normal samples remain on 9.0.2. Run affected Linux host/package/server
tests. Mac packaging must not change the Linux archive/cache contract or bypass
native saved-state checks. The current docs-only planning change needs link
validation, not GPU regressions.

## Completion and evidence

Write `docs/build/macos-distribution.md` with build/run/signing/cache/recovery
instructions, and `docs/build/macos-distribution-results.md` with exact source,
toolchain, OS/hardware, media, payload and final artifact identities. Include
compressed/extracted sizes, cold/warm launch timings, peak extraction RSS,
reproducibility results, sample links, model-download observations, source audits,
public-signing/trust evidence and the unchanged CUDA gates. List tested features
and deployment-only capabilities separately.

Provide a one-command local build and a documented explicit release workflow.
Only mark public packaging fully qualified after actual Developer ID signing,
inner/outer notarization and clean trust checks pass. Missing credentials,
hardware, evidence or source materials are recorded as pending, not successful
tests. Keep the source/library developer workflow usable, validate final links
and exclusions, and mark the fresh checklist complete only against retained
evidence. Do not publish a release as an incidental build step.
