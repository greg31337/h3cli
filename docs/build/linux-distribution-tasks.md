# Portable Linux build and single-file CUDA release

Status: **32/32 tasks complete; final qualification passed**. Updated 2026-09-29.
Implement [the Linux distribution design](design-linux-distribution.md).
The previous 34/34 completed server tasks are preserved in
[the server checklist](../features/server-tasks.md), with
[their validation report](../features/server-results.md).
The portable builder, runtime and launcher are implemented. Before the media
migration, two clean offline builds produced identical executables; see the
[earlier qualification report](linux-distribution-results.md). The current
build uses FFmpeg/FFprobe 9.0.2 for production and isolated historical codecs for
the regression gate; [migration validation](ffmpeg-upgrade.md) retains the
initial PRO 6000 build and its hardware-specific strict-golden limitation.

The final executable passed all **204/204 immutable goldens** in Ubuntu 22.04,
Ubuntu 24.04 and Debian 12 on the local PRO 5000; the unpacked Ubuntu 22.04
runtime passed too. Every loaded-library audit passed. Source-material and
NVIDIA binary-integrity audits are complete. The local M4 suite passed.
See [the completion report](linux-distribution-pro5000.md).

Both fresh offline builds are byte-identical, including the companion source
archive. All 18 CUDA feature stages, 26 server cases, 16 runtime smoke checks,
six HTTPS checks and concurrent-version validation passed. Final instructions,
source materials and cleanup audits are complete. The prepared executable and
source archive are under `bin/linux-release/`. Nothing has been published.
The earlier PRO 6000 golden differences remain a recorded hardware limitation.

## Fixed scope and acceptance rules

- Deliver one executable download containing the native CUDA application and
  pinned redistributable runtime. Extract once to a persistent user cache; no
  FUSE, sudo, Python, toolkit, compiler, host FFmpeg or container engine is needed
  to run it. Models, supported Linux/glibc and the NVIDIA driver remain external.
- Initial target: Linux x86-64, glibc >=2.35. Build against a pinned Ubuntu 22.04
  baseline and qualify the identical release on Ubuntu 22.04, Ubuntu 24.04 and
  Debian 12. Do not claim every Linux distro or GPU is supported.
- Keep CUDA 13.0.3 / nvcc 13.0.88, current cuBLAS/cuDNN/NVRTC/media pins and exact
  default arithmetic. Include existing optional CUDA features with their native
  architecture guards. No dependency upgrade or numerical recipe change is
  authorized merely to make bundling easier.
- The initial work used the clean PRO 6000; finish Linux builds and CUDA
  qualification on the **local PRO 5000**, as newly authorized by the user.
  Use **local M4 for Metal/host checks**. Distro userlands share the PRO 5000's
  working driver; do not replace it. Keep host paths out of shipping defaults.
- Follow [CONTRIBUTING.md](../../CONTRIBUTING.md): the complete recorded CUDA gate
  runs after coherent code/build/test changes. All **204 hashes and 12 input
  fixtures remain unchanged**, with no skip, tolerance or partial-run option.
  Add final artifact validation; a passing source rebuild alone is insufficient.
- Preserve CLI/server features, cwd/model/output semantics, cancellation and
  native state checks. Keep `make`, macOS builds and the Ubuntu source setup.
  All built executables and `libh3.a` remain under ignored `bin/`; evidence goes
  under ignored `outputs/linux-distribution/`.
- Aside from the fixed 640×480/124-frame/six-step parity render, use the design's
  small sample matrix: 256×256/22 frames/two steps by default, 56-frame mixed
  references, 39/56-frame continuation and 512×512/22-frame upscale. Existing
  feature harnesses may need up to six evaluations. No new large-render campaign.
- Missing dependencies, failed tests, unavailable fixtures or untested required
  cases prevent completion. Prepare artifacts locally; publishing, GPU CI,
  automatic updating and model/driver installation are outside this checklist.

PRO 5000 work/test connection:

```sh
ssh cuda-test
```

## Build baseline and reproducibility

- [x] LNX001 Bootstrap the clean PRO 6000 work/test node. Record OS/kernel,
  CPU/GPU/compute capability/VRAM, driver, tools, storage and mounts before setup.
  Verify container-engine availability, provider/nested-container constraints
  and GPU access for distro tests; prepare missing build tools without replacing
  the driver. Stage source/fixtures and locate or provision model/components and
  test assets without assuming a model path or preinstalled h3cli dependencies.
  Verify CoW scratch storage. Build the starting source with pinned dependencies
  in an isolated development environment and establish a fresh full 204-output
  parity baseline on this GPU. Record source/CLI inventory, flags, runtime/ABI
  identities, linked/explicit loads, subprocesses and data files, including
  toolkit/reference cuBLAS and cuDNN engines. Keep the prior GLIBC_2.38 finding
  and earlier hardware results labeled as historical; do not substitute them
  or a copied prebuilt runtime for this node's baseline.
- [x] LNX002 Create the versioned build/dependency lock under `scripts/linux/`:
  builder digest, compiler/linker/sysroot, CUDA 13.0.3, pinned runtime wheels,
  frontend/CUTLASS sources, ICU/json-c/SQLite/curl/OpenSSL/media dependencies and
  packaging tools. Preserve existing numerical pins. Fetch with checksums;
  include versions, configuration, provenance and licenses. Share authoritative
  pins with existing setup rather than allowing separate versions to drift.
- [x] LNX003 Implement the pinned Ubuntu 22.04 x86-64 OCI filesystem builder
  using hash-pinned PRoot (the provider prohibits nested namespaces). Compile
  without a GPU or host toolkit, use a controlled generic CPU baseline, and
  separate dependency fetching from offline build steps. Do not import arbitrary
  host headers/libraries or install into the host. Freeze all transitive inputs,
  including package repository snapshots or equivalent exact packages.
- [x] LNX004 Add the proposed `scripts/build_linux.sh` entry point with useful
  help/errors, bounded parallelism, dependency-cache control and caller-owned
  outputs. Snapshot a checkout or source archive consistently; record dirty
  content and VCS identity without depending on mutable `.git` during the build.
  Emit the design's runtime, release and validation artifacts under `bin/` and
  reproducible build metadata under `outputs/`.
- [x] LNX005 Build the core and native dependencies against glibc 2.35. Preserve
  ordinary and approximate CUDA flags separately; resolve compiler incompatibility
  without changing arithmetic. Audit every bundled ELF's GLIBC/GLIBCXX versions,
  architecture and CPU requirements, including helper executables and transitives.
  Fail the build when the declared ABI floor is exceeded.
- [x] LNX006 Provide production FFmpeg/FFprobe 9.0.2 and TurboJPEG 2.1.5 with
  pinned source/configuration and dependency closure. Keep exact FFmpeg 4.2.2
  output and 6.1.1 input/probe in the separate immutable-regression validation kit.
  Validate JPEG/video/audio decoding behavior through the complete recorded gate;
  keep input and output tool selection distinct.
- [x] LNX007 Enable the full existing CUDA feature set in the release: SGLang,
  cuDNN, Sage, SOL and SubBlock, preserving quantization/cache/placement controls.
  Build the current fat SASS/PTX set and architecture-specific optional kernels.
  Audit embedded code and runtime guards; unsupported feature/GPU combinations
  must report their existing errors. Record compiled versus physically tested
  architectures separately; do not add new kernel implementations here.
- [x] LNX008 Integrate the build without breaking ordinary Make targets,
  `bin/libh3.a`, setup scripts or macOS. Remove absolute toolkit RPATHs from
  distributed artifacts while retaining a usable developer workflow. Normalize
  paths/timestamps/ordering and compiler identity inputs for reproducibility;
  preserve meaningful source/flag/build changes in saved-state identity.

## Relocatable runtime and native integration

- [x] LNX009 Assemble `bin/linux-runtime/` with all required native libraries,
  math sublibraries, NVRTC builtins, media tools, ICU/provider data and notices.
  Generate an explicit bundled/host dependency classification. Exclude glibc,
  its loader/NSS stack, host driver libraries, CUDA stubs, models, Python and
  development tools. Audit redistribution and supply required notices/source
  materials for the selected media builds.
- [x] LNX010 Implement and verify relative ELF search paths for the core,
  transitives and media helpers, plus explicit `dlopen` resolution. Ensure driver
  libraries still resolve from the host, including container-injected drivers.
  Detect conflicting SONAMEs/duplicate providers and any fallback to build-host
  libraries. Record actual loaded paths during representative GPU/media work.
- [x] LNX011 Centralize bundled resource discovery under `src/runtime/` using
  the real executable and validated manifest. Bind cuBLAS/cuDNN/JPEG and production
  FFmpeg/ffprobe 9.0.2 to the bundle. Keep native source-build overrides;
  prevent stale setup variables, PATH, LD_LIBRARY_PATH or preload settings from
  silently substituting package dependencies. Preserve legitimate device and
  rendering controls and explicit regression capture variables.
- [x] LNX012 Preserve cwd-relative default models, user input/output paths,
  server state and LoRA/quantization cache locations. Add target-distro CA store
  discovery for curl with documented explicit trust overrides; test DNS, HTTPS
  verification and missing-store errors. No mutable data enters the payload,
  and offline/local media jobs do not depend on network access or a CA store.
- [x] LNX013 Audit conditioning and sampler identity for relocated bundled
  library paths. Use verified content/recipe identity where absolute extraction
  paths are currently treated as numerical identity. Prove same-build/same-device
  checkpoint and conditioning reuse after relocation; preserve rejection for
  changed runtime contents, build, arithmetic, model or device. Do not weaken
  current formats or add old saved-file compatibility.
- [x] LNX014 Integrate server workers with the extracted core and pinned
  runtime. Preserve executable identity, private descriptors, process groups,
  cancellation, parent-death detection and FFmpeg cleanup. Workers must not
  re-extract the download or invoke a shell. Concurrent package versions must
  leave running servers untouched; the SGLang-plus-`h3cli` request schema stays
  unchanged.
- [x] LNX015 Produce a deterministic runtime manifest with per-file hashes,
  core/source/build identities, dependency/tool versions, required ABI/driver,
  compiled capabilities and notices. Include separate validation-probe provenance
  bound to the same build/runtime. Distinguish package integrity from publisher
  authentication; no signing claim without signatures. Audit for embedded
  workstation paths, credentials or development-only assets.

## Single-file launcher and cache

- [x] LNX016 Implement a small static launcher under `src/runtime/` and a pinned
  compressor/decoder packaging path. Append the verified runtime payload and
  versioned manifest/trailer to `bin/h3cli-linux-x86_64`. Validate bounds and
  extraction size before processing. No shell, external unpacker, FUSE, network
  download, root access or runtime container requirement.
- [x] LNX017 Implement the user-owned payload-hash cache, XDG/home defaults and
  `H3CLI_RUNTIME_CACHE` override. Lock concurrent extraction, validate entries
  and file digests, and atomically publish only complete runtimes. Reject
  traversal, escaping symlinks and special files; flatten library SONAME aliases to regular files during assembly. Warm launches verify readiness/ownership/metadata and reuse files
  without decompressing or hashing the entire payload again.
- [x] LNX018 Handle interrupted/corrupt extraction, missing files, permissions,
  low disk space, cache `noexec` and read-only download directories. Recover
  without modifying a live runtime; otherwise fail with actionable recovery
  guidance. Keep package versions separate and implement no automatic cache
  eviction. Document safe manual removal only after all associated processes
  stop, including workers and FFmpeg.
- [x] LNX019 Add `H3CLI_BUNDLE_INFO=1` for GPU-independent package/cache diagnosis.
  Preserve argv/cwd/stdio/signals/exit codes through `exec`, including symlinked
  or renamed downloads and inherited worker descriptors. Report architecture,
  ABI and missing/old driver problems clearly. Keep test evaluation ceilings
  out of the shipped default environment; explicit harness controls still work.
- [x] LNX020 Add meaningful launcher/package tests covering cold/warm/concurrent
  starts, interrupted and malformed payloads, metadata corruption, extraction
  boundaries, cache failures, relocation, Unicode/spaces and shell-looking
  arguments. Exercise real child exit/signal/FD behavior. Use a small fixture
  payload for host fault tests and the real large payload for qualification;
  mock results never count as GPU/media validation.

## Artifact-aware regression and qualification

- [x] LNX021 Extend the CUDA reference runner with a prebuilt-artifact mode
  while retaining isolated source-build mode. Bind the complete probe set to
  the package's source/build/runtime manifest; verify artifacts and fixtures
  before/after execution. C0 must launch the final outer executable, probes
  must use packaged libraries, and RGB/PCM hashing must invoke the pinned
  validation-only FFmpeg 6.1.1 decoder. Include container/lock/packaging inputs in source
  fingerprints. Retain all 204 hashes, geometry/trajectory checks and the
  12-minute post-build deadline; no case filtering or tolerance switches.
- [x] LNX022 Extend CPU gate-policy tests for both modes: missing/extra outputs,
  changed sources/fixtures/package/probes, mismatched runtime, stale results,
  deadlines and unrelated PATH FFmpeg. Ensure package mode cannot secretly
  rebuild or select an installed binary and source mode retains its current
  guarantees. Record package/core/library/tool identities and loaded-path
  evidence without hashing model weights or changing goldens.
- [x] LNX023 Build on local M4 and run applicable retained host/Metal tests,
  server contract/lifecycle checks, setup tests and focused sanitizers for new
  parsing/extraction code. Build the Linux developer path and run its host
  suite. Record actual coverage and resolve regressions from shared path,
  environment or identity changes before final release qualification.
- [x] LNX024 Qualify the unpacked relocatable runtime on the PRO 5000 with the
  complete **204/204** recorded gate. Move it away from the build tree and deny
  access to checkout/toolkit/venv dependencies. Prove matching runtime/tool
  resolution. Preserve the unchanged golden manifest digest and fresh evidence;
  do not continue to final package acceptance with numerical drift.
- [x] LNX025 Qualify the final single-file executable on the PRO 5000 with the
  complete **204/204** artifact gate, entering C0 through the outer launcher.
  Exercise cold extraction and warm reuse, validate all payload hashes and
  bind the result to the exact downloadable file. Re-run the complete gate if
  release bytes or runtime dependencies change afterward.
- [x] LNX026 Test the identical release in clean Ubuntu 22.04, Ubuntu 24.04 and
  Debian 12 userlands on the PRO 5000. Pass the full 204-output package gate in
  each, exposing only driver, models and test inputs/harness as needed. Also
  smoke-test a non-root runtime with no Python, toolkit, host FFmpeg, sourced
  environment or build tree; test stale setup variables, fake PATH tools and
  a second cache root. Record host kernel/driver sharing and ABI measurements.
- [x] LNX027 Run the retained current-CUDA suite and bounded CLI sample matrix
  through the package. Cover references/anchors, preview/full media, stills,
  state/conditioning reuse, pause/resume across cache relocation, continuation/
  bridge, upscale, native LoRA/CoW and weight placement; verify compiled optional
  attention/cache/quantization features with existing harnesses. Inspect actual
  media dimensions/frames/audio and state loads. Keep required test budgets and
  existing hardware guards; do not substitute mocks for required real cases.
- [x] LNX028 Qualify packaged server execution with small real CUDA jobs:
  submit/poll/download and CLI agreement, reference uploads, cancellation then
  success, queued restart/recovery and opted-in HTTPS import. Verify worker/core
  hashes, runtime selection, private FDs, resource cleanup and operation from a
  renamed package. Exercise no-network local workflows and concurrent different
  package versions. Keep fake-worker fault tests separate from real render proof.
- [x] LNX029 Rebuild the final release twice from the same inputs in different
  directories with empty build caches; require byte-identical artifacts. Verify
  an offline rebuild after dependency fetching. Measure compressed/extracted
  size, temporary peak space, extraction memory, build time and cold/warm startup.
  Confirm no rebuild/codec/version metadata escaped the lock and no production
  test-budget environment was embedded.

## Documentation and completion

- [x] LNX030 Update README/Linux build and server instructions to lead with the
  qualified download/run flow, external model/driver requirements and supported
  distro/GPU limits. Document one-command source builds, retained Ubuntu setup,
  explicit model locations, cache override/diagnostics/removal, offline behavior,
  CA policy, source-build environment differences and checksums. Mark proposed
  commands available only when implemented; do not publish artifacts as part of
  this task.
- [x] LNX031 Publish `docs/build/linux-distribution-results.md` with build locks,
  exact release hashes, dependency/ABI/capability inventory, all distro results,
  full parity evidence, native build checks, real CLI/server samples and measured
  footprint/startup costs. Link playable small videos and machine-readable
  records. Separate compiled GPU coverage from tested hardware and document
  remaining platform limits without overstating cross-distro support.
- [x] LNX032 Audit all 32 tasks, links, output placement, license/source materials,
  current CLI/API behavior and final release identity. Confirm required suites
  pass against the delivered bytes, all 204 goldens/12 fixtures remain unchanged,
  no build/host dependency leaked into the package, and tests left no workers or
  bound server ports. Reconcile the checklist honestly; unresolved required
  failures block completion.
