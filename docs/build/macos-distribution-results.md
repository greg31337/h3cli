# macOS standalone qualification

The Apple Silicon package now has **Developer ID signatures, accepted inner and
outer notarizations, and a stapled DMG**. The signed rendering/server matrix
passes on the existing M4. Clean-environment and fresh-trust qualification remain
pending at the user's explicit request. The original development executable,
native library, payload and corresponding sources reproduce across fresh
offline builds in different absolute directories.

[Build/run guide](macos-distribution.md), [design](design-macos-distribution.md),
[task checklist](../todo.md). Local deliverables are in
[`bin/macos-release/`](../../bin/macos-release/) for development and
[`bin/macos-public-release/`](../../bin/macos-public-release/) for the notarized
release candidate. Nothing was publicly published.

## Qualified environment and inputs

| Item | Recorded value |
| --- | --- |
| Machine | Apple M4 Max, 128 GiB unified memory |
| Runtime OS | macOS 26.6.2, build 25G83 |
| Declared target | arm64, macOS 26.0 |
| Compiler | Apple clang version 21.0.0 (clang-2100.3.34.2) |
| Linker | 27037.1 |
| SDK | 27.0, build 26A425 |
| SDK content inventory SHA-256 | `848809186b88a3c94e83fe962506bb5bf6831c971e0a3a718c1b6ca9240a5ab7` |
| Media | FFmpeg/FFprobe 9.0.2, statically linked x264 r3222/b35605a |
| x264 revision | `b35605ace3ddf7c1a5d67a2eb553f034aef41d55` |
| Final source/test-tooling fingerprint | `742edad3fc61ed48905f08da2362adecd68c544f3c4cd59373572cc1e9bc9f28` |
| Runtime code/recipe source identity | `72fbae27dcd33b1e8c0a74fb6d8760b28f40e9e147cfd0262abfe61c47a745af` |

The dependency/tool hashes, source epoch, configurations and exact input hashes
are in [`scripts/macos/lock.json`](../../scripts/macos/lock.json) and the
[artifact build record](../../bin/macos-release/build.json). FFmpeg uses the
shared authoritative Linux lock. The x264 revision was recovered from the
installed qualified Homebrew formula/receipt; its source archive is pinned by
SHA-256. The builder restores its exact generated version string without
changing encoder arithmetic.

macOS 26.0 is a deployment declaration, not physical testing of that release.
Other Apple Silicon chips, M5-only paths and other macOS versions remain outside
this physical qualification. Existing capability guards are retained.

## Reproducible development artifacts

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| [Standalone executable](../../bin/macos-release/h3cli-macos-arm64) | 23,354,320 bytes | `0553b2f80e26c33c4aa7adf52792985747d0389c073174ce759ca49b8a75c1be` |
| [Development DMG](../../bin/macos-release/h3cli-macos-arm64.dmg) | 23,600,125 bytes | `30db00d6de01a89b06f708ffe0e1bd521831e633923839f2f7f7acc871e24939` |
| [Native libh3.a](../../bin/macos-release/libh3.a) | 2,086,392 bytes | `271781693d0f5683bb678f17f6c6ca5c6568e4c22542fa367f6c089d22095f7b` |
| [Corresponding sources](../../bin/macos-release/h3cli-macos-arm64-sources.tar.gz) | Recorded in build.json | Recorded in SHA256SUMS |

The archive checksum lives in [SHA256SUMS](../../bin/macos-release/SHA256SUMS)
and build.json so that this report can be included in the archive without a
self-referential hash. Documentation and the final test-only signature assertion
are included in the final source materials; the tested executable/core/media
bytes are frozen and unchanged. `macos-runtime/` and `macos-validation/` retain
the unpacked runtime and private native probes for audit.

The archive includes exact h3cli, FFmpeg and x264 sources, licenses, configurations,
generated-version recipe and offline rebuild/relink instructions. Small recorded
fixtures remain, including seven safetensors files below 100 KB each; they are
not model weights. No models, credentials, compiled objects, Apple SDK/framework
binaries or CUDA dependency caches are included. See the
[source audit](../../outputs/macos-distribution/source-audit.json).

## Build, extraction and resource checks

- `--doctor`, verified offline fetch, fresh-directory checks and native builds
  pass. Git and source-archive inputs retain `src/models/` and exclude private
  directories, model/build outputs, objects and generated Metal includes.
- Recursive Mach-O audits cover the launcher, core, media tools and private
  probes: arm64/macOS 26.0, no rpaths or Homebrew/MacPorts/developer dependencies.
  The core/media use only Apple system libraries/frameworks. See
  [audit.json](../../bin/macos-release/audit.json).
- Application, library, launcher and probes compile without warnings. The pinned
  upstream dependency builds emit 13 existing Clang diagnostics for unused x264
  declarations and FFmpeg fall-through annotations. These are recorded in
  [the build log](../../outputs/macos-build/release-final/build.log); no h3cli
  warnings or build errors are hidden.
- The signed-section prototype extracts a byte-identical signed helper and runs
  it. The final 17-test fixture suite covers payload-signature mutation, hashes,
  canonical JSON, path/ancestor/number/mode rejection, symlinks, metadata changes,
  parallel launch, signals/stdio, crash/partial writes and multiple versions.
  Twenty repeated eight-way cold-start runs pass after fixing exclusive lock
  creation on APFS. Eight malformed Mach-O cases pass ASan/UBSan checks.
- Real disposable-volume tests reject insufficient space (about 24.8 MB free),
  a noexec cache and wrong ownership. No existing host volume was changed.
- Runtime resources are fixed to the verified cache. Hostile cwd shaders, PATH
  media tools and resource overrides cannot substitute content. Model, offline,
  CA/proxy and output settings retain their existing behavior.

Evidence: [fault suite](../../outputs/macos-distribution/bundle-tests-final.log),
[parser sanitizers](../../outputs/macos-distribution/section-sanitizer-final.log),
[volume failures](../../outputs/macos-distribution/cache-volume-final/results.json),
[reproducibility](../../outputs/macos-distribution/reproducibility-final.json).
The final documentation/test-only source-material refresh is separately checked
for identical archives and unchanged runtime code in the
[final source record](../../outputs/macos-distribution/source-materials-final.json).
The first reproduction check found an inherited parent Git commit; explicitly
labeling source-archive builds removed that directory-dependent metadata while
preserving content/flags/model/backend checks. The final executable, payload,
manifest, library and sources match exactly. DMG filesystem metadata differs
between builds; notarization timestamps/tickets are not claimed reproducible.

The compressed payload is 23,202,024 bytes and the
expanded runtime is 49,575,062 bytes. One final cold
launch measured 1.437 seconds, with a sampled peak RSS
of 30.42 MiB. Four warm launches took
13.32–13.61 ms.
RSS was sampled through `proc_pidinfo` at 1 ms intervals because macOS time
statistics reset across exec. These are qualification observations on a working
development host, not controlled throughput benchmarks.
[Measurement record](../../outputs/macos-distribution/launch-profile.json).

## M4 rendering and integration

The packaged matrix completed 19 commands including its CLI-equivalent server
comparison. The native matrix completed the 18 common commands. All 16 delivered
native/package media outputs are **byte-identical**, using the same model files,
OS, shader contents and packaged FFmpeg tools. Pause commands produce state only.
All outputs were probed and fully decoded; shape/frame/audio-duration checks pass.

[Packaged gallery](../../outputs/macos-distribution/package-qualified/review.html),
[native gallery](../../outputs/macos-distribution/native-qualified/review.html),
[exact comparison](../../outputs/macos-distribution/native-package-comparison.json).
Samples use 256×256/two steps, 22 frames ordinarily, 56 for the saved reference
source and mixed references, 39 protected frames for continuation, and 512×512
with two refinement steps for upscaling.

| Case | Packaged wall time | Native wall time | Result |
| --- | ---: | ---: | --- |
| text-full | 32.119 s | 30.505 s | Pass |
| preview | 22.046 s | 21.711 s | Pass |
| first-last | 29.953 s | 27.330 s | Pass |
| references | 59.656 s | 57.732 s | Pass |
| still | 31.940 s | 32.689 s | Pass |
| still-decode | 8.796 s | 8.691 s | Pass |
| av-decode | 4.868 s | 4.837 s | Pass |
| av-preview-decode | 0.724 s | 0.741 s | Pass |
| conditioning | 20.060 s | 19.803 s | Pass |
| minimal-av-decode | 3.958 s | 3.968 s | Pass |
| continue-hard | 31.220 s | 30.358 s | Pass |
| continue-bridge | 30.873 s | 30.496 s | Pass |
| sampler-uninterrupted | 24.297 s | 24.087 s | Pass |
| sampler-pause | 19.868 s | 19.929 s | Pass |
| sampler-resume | 8.322 s | 8.307 s | Pass |
| upscale | 65.476 s | 65.463 s | Pass |
| upscale-pause | 24.648 s | 24.894 s | Pass |
| upscale-resume | 44.263 s | 44.484 s | Pass |

Still/AV/minimal-decoder round trips, conditioning reuse, first/last crop,
image/video/audio references, hard/bridge continuation, sampler pause/resume and
upscale pause/resume pass. A sampler resumed after copying/renaming/symlinking
the executable, changing cwd and using a fresh cache; output remained exact and
all extracted core/helper signatures verified.
[Relocation record](../../outputs/macos-distribution/relocation/results.json).

The packaged server passed genuine cold model preparation and offline reuse,
artifact fetch/full decode, exact CLI equivalence, queue cancellation and durable
restart. A modified shader cache prevents server startup before any job runs.
The current 23-test HTTP suite and host/downloader suites pass on M4 and Linux.
[Server matrix](../../outputs/macos-distribution/package-qualified/results.json),
[cancellation/restart](../../outputs/macos-distribution/server-cancel-final/results.json),
[changed-runtime rejection](../../outputs/macos-distribution/server-cancel-final/changed-runtime.json).

CLI and server each downloaded a genuine pinned preview VAE into a separate
empty auxiliary root. Both warm offline reruns pass; CLI output is identical
with unusable CA paths while offline. Existing main weights were reused. A further sandboxed render also denied all
writes to the complete existing model tree; it passed without changing model
permissions or contents. [Read-only model check](../../outputs/macos-distribution/runtime-readonly/results.json).
Model-root/group/auxiliary precedence is covered by the retained fixture suites.
[Additional qualification](../../outputs/macos-distribution/extra/results.json).

Ad-hoc hardened-runtime probes for native attention/layout/FP16, SOL, Q8, ANE
split, actual CoreML/ANE execution and ANE cache integrity all pass, without
additional entitlements. These retain the existing feature restrictions;
they do not qualify every combination or M5-only implementation. Native
`make test test-server-http test-metal-native-host` and the final host suite pass.

## Runtime isolation and public-release limits

The final executable also renders outside the checkout, with a new home/cache,
minimal PATH, hostile resource files/overrides, invalid CA paths and networking
disabled. A sandbox denies Homebrew, developer directories/tools and checkout
file contents except the pre-existing model tree; parent directory metadata is
allowed for safe model-path resolution. An explicit xcrun launch is denied.
Live mappings and open-file snapshots show the actual system/runtime/model
paths. This is stronger than PATH hiding, but it is **the existing developer
account on the development OS**, not a clean account or installation.
[Isolation record](../../outputs/macos-distribution/runtime-only-final/results.json),
[mappings](../../outputs/macos-distribution/runtime-only-final/vmmap.log),
[open files](../../outputs/macos-distribution/runtime-only-final/open-files.log).

The final development DMG was mounted read-only. Its executable matches the
recorded SHA-256, verifies its signature and launches with a fresh cache.
This is not fresh-quarantine Gatekeeper qualification.
[Final DMG check](../../outputs/macos-distribution/dmg-final.json).

`scripts/macos/release.py` implements explicit Developer ID signing, inner ZIP
notarization, exact accepted-byte embedding, outer signing, DMG notarization,
stapling and verification. Keychain profile names are accepted; account passwords
and private keys are not CLI inputs. Ordinary/offline builds never submit to
Apple, and release errors never fall back to ad-hoc signing. The user installed
the release team's Developer ID Application identity and configured the
`h3cli` Keychain profile. Both are verified; neither credentials nor private keys
were exported. Signing needed local Keychain authorization. No security settings
or quarantine were bypassed, accounts created, tools uninstalled or OS/driver
installations changed.

## Developer ID release candidate

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| [Signed executable](../../bin/macos-public-release/h3cli-macos-arm64) | 23,338,080 bytes | `371605f9704f54ecea7e02249348147087930a95d899d7f6d0c720722b5b7486` |
| [Signed, notarized, stapled DMG](../../bin/macos-public-release/h3cli-macos-arm64.dmg) | 23,595,394 bytes | `add02fc97d41d8a7b843d56102a9b6b926405d5d6caa6f9727f04acab7f56ccf` |
| [Corresponding sources](../../bin/macos-public-release/h3cli-macos-arm64-sources.tar.gz) | 24,926,864 bytes | `5648ecea721b7427cb41c1a7e3a12ac35c5562282a5072fa5100fffcde508a17` |
| [Native libh3.a](../../bin/macos-public-release/libh3.a) | 2,086,392 bytes | `271781693d0f5683bb678f17f6c6ca5c6568e4c22542fa367f6c089d22095f7b` |

The signing identity is retained in the private release records. All four runtime executables have
valid Developer ID signatures, secure timestamps and hardened runtime, with no
additional entitlements. The final dependency audit still finds only Apple
system loads, arm64/macOS 26.0 and no rpaths.
[Signature records](../../outputs/macos-distribution/public-signatures/executables.json),
[dependency audit](../../bin/macos-public-release/audit.json),
[build record](../../bin/macos-public-release/build.json).

Apple accepted both submissions with no reported issues:

- Inner runtime: accepted; submission identifier retained privately.
  [Submission](../../bin/macos-public-release/inner-submission.json),
  [notary log](../../bin/macos-public-release/inner-notary-log.json).
- Final DMG: accepted; submission identifier retained privately.
  [Submission](../../bin/macos-public-release/outer-submission.json),
  [notary log](../../bin/macos-public-release/outer-notary-log.json).

All 18 embedded/extracted runtime members match the accepted inner ZIP bytes.
The DMG ticket is stapled and validates; its Gatekeeper assessment reports
`Notarized Developer ID`. After mounting read-only, the copied executable matches
the recorded hash, launches with a cold cache after unmounting, and satisfies
the explicit notarized-code requirement along with all extracted helpers.
[Accepted-byte comparison](../../outputs/macos-distribution/public-inner-acceptance.json),
[DMG/copy check](../../outputs/macos-distribution/public-dmg.json),
[release log](../../outputs/macos-distribution/public-release.log).

An initial `spctl --type execute` check reported that the valid CLI was not an
app. The appropriate check for this non-app executable is
`codesign --verify --strict --test-requirement '=notarized' --check-notarization`,
which passes. The diagnostic is retained; no trust policy was bypassed. The
separate DMG `spctl --type open` assessment passes. See
[Apple's product-type guidance](https://developer.apple.com/forums/thread/130560).

The signed matrix completed all 19 commands, including server preparation and
offline reuse. All 17 delivered files match the development package byte-for-byte;
the 16 common native outputs also match. This covers references, conditioning,
continuation/bridge, sampler pause/resume, upscaling/resume and media round trips.
CLI and server automatic preview downloads and offline reuse pass.
[Signed gallery and timings](../../outputs/macos-distribution/public-qualified/review.html),
[matrix](../../outputs/macos-distribution/public-qualified/results.json),
[exact media comparison](../../outputs/macos-distribution/public-media-comparison.json).

Developer ID-signed probes for Metal layout/native attention/FP16, SOL, Q8,
ANE splitting, actual CoreML/ANE prediction and ANE cache integrity all pass.
No extra entitlements were added. The signed CLI's cold preview download and
offline rerun with unusable CA paths produce identical video. The signed server
passes active/queued cancellation and durable restart. A sampler resumes exactly
after renaming/symlinking the executable and changing its cache/cwd. The signed
runtime also passes the sandboxed outside-checkout render with developer tools,
Homebrew, source data and networking denied, with mappings/open files recorded.

[Signed probes/downloads](../../outputs/macos-distribution/public-extra/results.json),
[cancellation/restart](../../outputs/macos-distribution/public-server-cancel/results.json),
[relocated resume](../../outputs/macos-distribution/public-relocation/results.json),
[runtime isolation](../../outputs/macos-distribution/public-runtime-only/results.json),
[check sequence](../../outputs/macos-distribution/public-extra-sequence.json).

The checklist now records **33/36 completed tasks**. MAC029/MAC030 remain pending
because the user explicitly asked to leave clean-environment and fresh-trust
checks open; MAC036 therefore remains open as well. These local checks do not
establish cold offline trust on a Mac that has never seen the product.

Signing changed no application, media or build/test source. The corresponding
sources and native library are identical to the qualified development artifacts;
that archive retains the development qualification documentation. The current
report and signing evidence record the later release candidate separately.

## CUDA protection

Only the local RTX PRO 5000 was used. The unchanged 204-output/12-fixture gate
passed at baseline and implementation milestones, then on final native source
and the rebuilt Linux standalone package. The final test-only payload-signature
assertion is included in the acceptance reruns below. Ordinary rendering remains
on FFmpeg 9.0.2; historical 4.2.2/6.1.1 tools stay in the private validation kit.

The golden manifest remains
`fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3`.
The inference objects/dependencies reused for Linux packaging were checked
against the previous pinned build and changed headers invalidated their dependent
objects. No model files, golden hashes or recorded fixtures were changed.
Linux host/package/downloader/server checks passed. The final native gate took
105.635 seconds and the final standalone artifact gate took 123.968 seconds;
each passed all 204 outputs. Exact artifact identities are recorded below.
The subsequent Developer ID work changed only signing artifacts and documentation;
the current code/build/test fingerprint still matches both accepted gates. CUDA
tests were not unnecessarily rerun for these signing/documentation-only changes.

- [Final native gate](../../outputs/macos-distribution/pro5000/evidence/acceptance-native/result.json).
- [Final artifact gate](../../outputs/macos-distribution/pro5000/evidence/acceptance-artifact/result.json).
- [Linux host/server log](../../outputs/macos-distribution/pro5000/evidence/linux-host.log).
- [Linux artifact metadata](../../outputs/macos-distribution/pro5000/acceptance-artifact.json).

The Linux deliverable and private kit remain under
`/path/to/h3cli-macos-work/linux-acceptance/` on PRO 5000. The native macOS developer
binary/library and existing model-download checklist/results remain available.
