# macOS standalone package building

Status: **notarized package qualified on the local M4 — 33/36 tasks complete**.
Updated 2026-09-30. [Evidence and pending prerequisites](build/macos-distribution-results.md).
The release team's **Developer ID Application** identity and the `h3cli`
Keychain notary profile are verified. Apple accepted the signed inner runtime;
all 18 embedded/extracted members match the accepted bytes. The final DMG is
signed, notarized and stapled; signed-artifact rendering, server, download,
Metal/ANE and isolation tests pass on the existing M4 account.
The user explicitly left clean-environment and fresh-trust qualification
(MAC029/MAC030) pending; final public completion MAC036 remains open.
The local ad-hoc executable, development DMG, library and corresponding sources
remain ready.
Implement [the macOS distribution design](build/design-macos-distribution.md).
The preceding **36/36 completed model-download tasks** are archived in
[their checklist](features/model-downloads-tasks.md), with
[qualification results](features/model-downloads-results.md).

## Fixed scope and acceptance rules

- Deliver a single `h3cli-macos-arm64` executable containing the application,
  shaders and pinned FFmpeg/FFprobe 9.0.2 dependencies. Extract verified resources
  to `~/Library/Caches/h3cli/runtime/<id>`, with `H3CLI_RUNTIME_CACHE` override.
  Running it needs no Homebrew, Python, Xcode/CLT, system FFmpeg, sudo or container.
- Initial target: Apple Silicon arm64, explicit macOS 26.0 deployment target.
  macOS supplies its system frameworks. Distinguish the deployment target from
  the actual OS/chip qualified; preserve existing feature/capability guards.
- Provide a signed, notarized, stapled DMG for public distribution and complete
  companion source materials. Local/offline builds use explicitly labeled
  ad-hoc signing. Public qualification requires a Developer ID identity and
  working notarization credentials; unavailable credentials leave those tasks
  pending. Ordinary build/test commands never upload code or publish releases.
- Embed the payload in a signature-covered Mach-O section before signing;
  prove inner-helper and outer-executable trust behavior early. Preserve exact
  extracted signed bytes. Do not reuse Linux's EOF-footer assumption or disable
  Gatekeeper/quarantine checks as a workaround.
- Preserve CLI/server features, cwd and output behavior, current state checks,
  automatic model downloads and `--models-path`/`-d`/auxiliary precedence. Models
  remain outside the runtime. No codec/model upgrade or numerical-policy change.
- Keep native `make`, `bin/h3cli`, `bin/libh3.a`, setup scripts and Linux packaging
  working. Binaries/packages go under ignored `bin/`; work/evidence goes under
  `outputs/macos-build/` and `outputs/macos-distribution/`. Exclude weights,
  private media, credentials and Apple's SDKs from redistribution.
- Use **local M4** for macOS builds and small Metal samples. Use **local PRO 5000**
  for Linux/CUDA protection; no cloud nodes, host-driver changes or OS reinstall.
  Preserve existing models and use isolated build/cache/download/output trees.
- Follow [CONTRIBUTING.md](../CONTRIBUTING.md): full CUDA golden gate after
  coherent code/build/test changes, plus final source and rebuilt Linux artifact
  gates. Keep all **204 hashes and 12 fixtures unchanged**, with networking off
  and no skipped/relaxed cases. Docs-only edits need document/link checks.
- Real sample defaults: 256×256, 22 frames, two steps; 56 frames where references
  or continuation require them, and 512×512 upscaling with two refine steps.
  Preserve the CUDA gate's own 640×480/124-frame/six-step case. No new large
  performance campaign. Missing prerequisites or unrun checks are not passes.

PRO 5000 connection for the required regression protection:

```sh
ssh cuda-test
```

## Baseline and build pipeline

- [x] MAC001 Inventory the current macOS build: OS/chip/RAM, compiler/linker/SDK,
  deployment target, linked and runtime-loaded dependencies, shaders, CoreML
  resources, subprocesses and exposed features. Record native host/Metal/server
  results and small reference samples before changes. Establish the complete
  current CUDA baseline on PRO 5000; preserve existing models and artifacts.
- [x] MAC002 Define the macOS dependency/toolchain lock and provenance schema.
  Pin actual Apple tool identities, SDK inventory, build tools, deployment flags,
  source epoch and verified dependency sources/configurations. Share FFmpeg's
  authoritative 9.0.2 pin; determine the exact qualified macOS x264 source.
  Separate supplied Apple prerequisites from downloadable redistributable inputs.
- [x] MAC003 Implement `scripts/build_macos.sh` and helpers with `--doctor`,
  `--cache`, `--work-dir`, `--output`, `--jobs`, `--fetch-only` and `--offline`.
  Validate prerequisites/options; use fresh isolated directories and controlled
  tool selection. Do not modify global Xcode selection, install Homebrew, touch
  native build outputs or fetch model weights. Make warm-cache builds offline.
- [x] MAC004 Snapshot Git checkouts and source archives consistently, including
  dirty/untracked code, catalogs and shader inputs. Exclude generated objects,
  models, outputs, secrets and prior packages while retaining `src/models/` and
  source test fixtures. Normalize paths/order/timestamps and preserve meaningful
  source/compiler identity in saved-state compatibility checks.
- [x] MAC005 Build pinned arm64 FFmpeg/FFprobe 9.0.2 and qualified x264 with
  explicit deployment/configuration flags and no dependency autodetection.
  Prefer static x264; otherwise bundle audited private dylibs with relative
  load paths. Preserve input/output capabilities and codec settings. Eliminate
  all Homebrew/MacPorts/developer-prefix runtime dependencies.
- [x] MAC006 Build the complete macOS application/library/probes without warnings.
  Retain Metal/MPSGraph, compiled optional native/Q8/SOL/ANE implementations,
  capability guards, server and native downloads. Keep generated shaders/catalog
  deterministic. Preserve source builds, setup and Linux integration; do not
  introduce new fast-math flags or machine-specific compiler tuning.
- [x] MAC007 Add recursive Mach-O/resource audits for every shipped executable
  and dylib: arm64, minimum OS, install names, weak/re-exported loads and confined
  rpaths. Allow only audited Apple-system loads outside the payload. Include
  explicit loads, media helpers and generated resource use in the inventory;
  fail assembly on missing dependencies, unrecorded files or host leakage.

## Single-executable runtime

- [x] MAC008 Prototype the signature-covered Mach-O payload early, before
  committing to the full extractor. Embed a tiny signed helper, sign the outer
  executable, extract identical bytes and execute it. Verify section bounds and
  codesign mutations; establish the public inner/outer notarization sequence
  and fresh offline trust test required by MAC021/MAC030. Do not append data
  after signing or quietly substitute a multi-file deliverable.
- [x] MAC009 Implement the bounded canonical macOS manifest/archive and Mach-O
  section reader. Bind platform/schema, launcher and payload identity without
  signature/hash cycles. Reuse safe common parsing/compression logic where useful,
  preserving Linux format compatibility. Keep the bootstrap free of non-system
  installed dependencies and runtime archive/crypto/developer command helpers.
- [x] MAC010 Implement private versioned cache extraction with ownership/no-follow
  checks, process locking, staging, free-space bounds, hashes, sync and atomic
  readiness publication. Cover interrupted/racing launches and fast verified warm
  reuse. Refuse changed caches; preserve active/older versions and provide safe
  manual recovery. Use macOS cache conventions and the explicit cache override.
- [x] MAC011 Implement the macOS launcher with canonical executable discovery,
  bundle inspection and verified core execution. Preserve cwd, argv, stdio,
  exit status, signals and process groups. Support symlinks, renamed binaries,
  read-only launch locations and mounted DMGs without writing alongside them.
  Keep extraction network-free and independent of model/GPU availability.
- [x] MAC012 Add packaged macOS resource initialization and override policy.
  Bind runtime identity and all helper/resource paths before CLI or server work;
  ensure internal workers select the same runtime. Ignore resource substitutions
  from cwd/PATH and source-only overrides, handle relevant DYLD variables, and
  preserve legitimate models/offline/proxy/token settings. Retain native behavior.
- [x] MAC013 Package all required shader source/includes and integrate their
  verified lookup with every Metal component. Keep system runtime compilation;
  require no offline Metal compiler or source checkout on the user's machine.
  Test hostile cwd shaders and `H3_SHADER_PATH` while preserving source-build
  lookup rules and useful compilation/lookup diagnostics.
- [x] MAC014 Bind all CLI/server media operations to packaged FFmpeg/FFprobe
  9.0.2, including input decoding, output encoding, probes and child cleanup.
  Verify no system media tool is used and private dylib paths remain relocatable.
  Keep historical codecs in the CUDA validation kit only.
- [x] MAC015 Integrate content-based runtime/media/shader identities with current
  sampler, conditioning, AV, presentation and upscale compatibility checks.
  Prove relocation and cache-root changes work for the same compatible artifact
  and unchanged model tree. Retain source/flags/resource/model mismatch detection;
  do not broaden arbitrary native/package or cross-OS checkpoint compatibility.
- [x] MAC016 Preserve default downloads, the embedded catalog, receipts and all
  model-root/auxiliary overrides through CLI and server worker launch. Runtime
  extraction must not relocate weights/states/outputs or alter cwd. Keep macOS
  system trust/CA/proxy/HF-token behavior and strict offline policy; do not embed
  a CA bundle or import Linux-specific CA-path selection.

## Source materials and reproducibility

- [x] MAC017 Produce the complete companion source archive, licenses/notices,
  patches, dependency configurations and build/relink instructions for h3cli,
  FFmpeg, x264 and every redistributed component. Audit exact source-to-binary
  correspondence. Exclude Apple SDK/framework payloads, CUDA dependencies,
  model weights and secrets; keep small checked-in fixtures distinguishable.
- [x] MAC018 Perform two fresh offline builds in different absolute directories
  with the same locked inputs. Compare normalized core/helper bytes, generated
  sources, deterministic ad-hoc payload, manifest and source archive. Resolve
  unintended path/time/UUID/order variation and document which final signing/DMG
  fields are externally nondeterministic. Preserve meaningful build identities.

## Signing and macOS distribution

- [x] MAC019 Implement deterministic ad-hoc signing for local development output
  and an explicitly unnotarized development DMG. Exercise the signing order,
  signature-covered payload and extracted-code verification. Clearly separate
  this successful local workflow from public-release trust qualification.
- [x] MAC020 Implement Developer ID Application signing for inner code and the
  outer launcher with secure timestamps, hardened runtime and minimal justified
  entitlements. Verify every nested signature. Test Metal/MPSGraph, CoreML/ANE,
  library validation, downloads and server subprocesses under these settings;
  do not solve failures by disabling platform security or broad exceptions.
- [x] MAC021 Implement and execute inner-runtime notarization in a supported
  container before embedding the exact accepted signed bytes. Retain submission
  identity, logs and accepted hashes. Use the documented custom-installer order
  rather than assuming the notary scanner inspects custom compressed sections.
  Apple accepted the inner-runtime submission, with no
  reported issues; accepted bytes match the extracted signed executable.
- [x] MAC022 Create the release DMG with the signed standalone CLI and usage/source
  notices. Sign/notarize the final distribution, inspect logs, staple and validate
  its ticket, and emit final hashes/status metadata. Record inner/outer identities
  separately. Do not attempt to staple the bare executable or modify signed code
  afterward. Keep public publishing outside the build/release commands.
- [x] MAC023 Isolate signing credentials in Keychain/caller-controlled secure
  profiles. Add explicit release-command help, failure/retry handling and redacted
  logging; ordinary build/offline/test modes must never submit to Apple or fall
  back silently to weaker signing. Test these boundaries without real secrets.
  Record missing credentials as pending public qualification, not a pass.

## Tests and hardware qualification

- [x] MAC024 Add host-only archive/Mach-O/extractor fault tests using tiny
  fixtures. Cover malformed bounds/hashes, traversal/duplicates/ancestor conflicts,
  symlink races, wrong ownership/modes, disk-full/write failures, crashes, stale
  metadata, tampering, concurrent cold launches and multiple active versions.
  Measure bounded memory/cancellation and run appropriate parser sanitizers.
- [x] MAC025 Add relocation and process tests: spaces/Unicode, changed cwd,
  symlinks/renames, read-only or missing cache parents, mounted DMGs, bundle info,
  hostile shaders/tools/environment, argv/stdio/exit codes and SIGINT/SIGTERM.
  Assert package resource identity and safe failure without relying on a checkout
  or runtime developer utilities. Cover state resume after cache relocation.
- [x] MAC026 Exercise the packaged server's admission, preparation, inference,
  artifact probing/fetch, cancellation, shutdown/restart and idempotency paths.
  Verify workers and FFmpeg use the same verified runtime; preserve responsive
  queue behavior, resource snapshots, authorization and offline inheritance.
  Run current server suites, including changed-runtime rejection.
- [x] MAC027 Exercise packaged model-root/group/override behavior with fixtures,
  then perform real automatic preview-VAE acquisition through CLI and server
  into separate empty auxiliary locations. Verify pinned hashes, progress,
  warm offline reuse, read-only installed models and no usable CA/network
  requirement for offline inference. Reuse the already qualified main weights.
- [x] MAC028 Run the complete bounded M4 matrix from the design: text/full and
  preview, first/last, image/video/audio references, still/decode, AV/conditioning,
  minimal decode, continuation/bridge, sampler resume and upscale/resume. Compare
  source/package on identical OS, weights, shaders and media tools; retain exact
  round-trip comparisons, fully decoded media, timings and sample galleries.
  Run current Metal/host and supported optional-feature probes without warnings.
- [ ] MAC029 Qualify the final runtime outside the checkout under a clean user
  and minimal environment, preferably an available clean macOS installation.
  Prove no Homebrew, Python, system FFmpeg, SDK/compiler or hidden developer
  resource dependency via load/file-access/subprocess evidence; PATH alone is
  insufficient. Record actual OS/M4, isolation limits and deployment-only support.
  Do not uninstall tools or reinstall the developer's system to perform the test.
- [ ] MAC030 Qualify the actual Developer ID/notarized/stapled DMG under fresh
  quarantine/trust state. Copy out, unmount and run with an empty extraction cache
  online and offline; also test warm offline rendering with provisioned models.
  Verify inner code signatures, Metal compilation and server children. Preserve
  Gatekeeper/notary/stapling evidence; no quarantine removal or cached-trust-only
  substitute. Missing credentials or a suitable environment leave this open.
- [x] MAC031 On PRO 5000 run affected Linux host/package/server checks and the
  full offline 204-output gate at coherent milestones. For final sources rebuild
  the Linux standalone artifact and run its full artifact gate with the private
  historical codec kit. Preserve all 204 goldens/12 fixtures, normal FFmpeg 9.0.2,
  Linux archive/cache behavior and native state checks. Fix failures and rerun.

## Documentation and completion

- [x] MAC032 Add `docs/build/macos-distribution.md` and update README/setup/help
  guidance for one-command building, prerequisites versus runtime requirements,
  copy/run usage, source materials, signing/notarization, cache inspection/removal,
  model paths/downloads/offline use, deployment limits and troubleshooting.
  Clearly identify development artifacts versus publicly qualified releases.
- [x] MAC033 Freeze final source, dependency and signing inputs; build the final
  executable, DMG, source archive, library, checksums and private validation kit.
  Audit signatures/contents and rerun affected final-artifact checks after any
  mutation. Verify the copied/distributed executable and DMG contents match the
  tested artifacts, with no weights, secrets, host paths or accidental runtime tools.
- [x] MAC034 Write `docs/build/macos-distribution-results.md` with exact source,
  toolchain/SDK, OS/hardware, media and artifact identities; sizes, extraction
  memory/cold-warm timings, reproducibility evidence, sample links, automatic
  downloads, state/server checks, source audits, public trust results and both
  final CUDA gates. Keep failures, untested OS/chips and pending signing explicit.
- [x] MAC035 Review final diffs, links, catalog/shader regeneration, artifact
  exclusions and documentation consistency. Preserve the archived model-download
  checklist/results and native/Linux instructions. Keep executable/library outputs
  under ignored `bin/` and qualification media/evidence under ignored `outputs/`.
- [ ] MAC036 Reconcile every checklist item against retained evidence. Confirm
  the local/offline build and actual public-release workflow meet their respective
  contracts, and that no required test, source audit or signing prerequisite is
  silently skipped. Mark completion only after required qualification passes;
  prepare deliverables locally without publishing a release.
