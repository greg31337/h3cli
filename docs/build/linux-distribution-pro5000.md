# Portable Linux release: PRO 5000 qualification

Final qualification passed; **all 32 tasks are complete**. This report supersedes
the historical [PRO 6000 acceptance status](linux-distribution-results.md).
[Task list](linux-distribution-tasks.md), [build/run guide](linux-distribution.md),
[acceptance audit](../../outputs/pro5000-release/acceptance-summary.json).

## Changes being qualified

Production rendering, reference decoding, server jobs and media inspection use
FFmpeg/FFprobe 9.0.2. The complete SGLang regression selects the historical
4.2.2 encoder and 6.1.1 decoder/probe from the separate private validation kit.
Neither historical codec is in the downloadable executable.

The redistribution inventory now covers every packaged ELF and the static
launcher. The companion source archive contains the exact application source,
pinned upstream archives and **28 exact Ubuntu source packages**, including
Ubuntu patches and build recipes. It includes GCC 11/12 and glibc for the
static launcher and library dependencies. NVIDIA libraries retain their original
bytes. cuDNN keeps a nested directory layout so its original `$ORIGIN` paths
stay inside the package. Missing sources, versions or notices fail assembly.
Vendored dependency notices accompany the runtime.

Build source snapshots and regression fingerprints consistently exclude the
generated Metal attention include; its generator and input sources remain
covered. The real non-root smoke test can use the invoking user's credentials
or drop root privileges when started as root. No permission emulation is counted
as proof of a non-root host process.

## Environment

The local RTX PRO 5000 72GB Blackwell runs Ubuntu 24.04.5 LTS, kernel
6.8.0-142-generic and NVIDIA driver 595.91.07. The GPU reports SM120 and
73,415 MiB of VRAM. The host has two Intel Xeon Gold 6154 CPUs (72 logical
CPUs total); builds use 16 jobs. All new Linux builds and GPU qualification use this host;
local M4 provides Metal/host coverage. Build helpers are installed in a private
work directory without sudo or host toolkit changes. The XFS model partition
provides CoW scratch storage.

Clean Ubuntu 22.04, Ubuntu 24.04 and Debian 12 userlands share the PRO 5000's
kernel and driver. The pinned builder compiles without binding the driver.
The toolkit remains CUDA 13.0.3, nvcc 13.0.88; cuDNN remains 9.20.0.48. The
full existing SASS/PTX and optional Sage/SOL/SubBlock feature set is retained.
Other GPU architectures and the minimum supported driver are not physically
qualified by this campaign.

## Initial immutable-golden check

Before changing the final packaging, the FFmpeg 9.0.2 candidate with SHA-256
`f1253afd32ccdb5944cb1ce28395b40f6a718165902b44624485cec1998afdda`
passed **204/204** on PRO 5000 in 122.134 seconds of gate execution
(143.154 seconds including integrity checks). This confirms that the PRO 6000's
113 differing outputs are not reproduced on the original golden hardware.
It does not replace the final artifact gates below.

The immutable golden manifest remains
`fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3`.
All 204 expected hashes and 12 fixtures remain unchanged. The full gate has no
case-selection, tolerance or skip options.

## Final artifact identity

Prepared local files: [single executable](../../bin/linux-release/h3cli-linux-x86_64),
[checksum](../../bin/linux-release/h3cli-linux-x86_64.sha256),
[companion sources](../../bin/linux-release/h3cli-linux-x86_64-sources.tar.gz),
[source checksum](../../bin/linux-release/h3cli-linux-x86_64-sources.tar.gz.sha256),
[build and capability inventory](../../bin/linux-release/build.json),
[runtime/ABI manifest](../../bin/linux-release/linux-runtime/share/h3cli/runtime.json).
The existing candidates are preserved in separate ignored directories.

The first fresh offline build completed in **1,772.250 seconds** with 16 jobs.
The executable contains 355 regular files and 56 audited ELF objects. Its
unpacked payload is 1,944,714,980 bytes; the download is 1,313,678,376 bytes.
The companion source archive is 418,313,674 bytes.

| Record | SHA-256 |
| --- | --- |
| Executable | `9e3234ff923cc915781871d2f50daf6b8e8b67be3b69171fbc8d48ea5a500b8a` |
| Source archive | `acd93739ea30a0a520dc54d0a06c2675453e1865182f03ecbf6a7df5307ba302` |
| Code/build/test source fingerprint | `eab9249ccf15e3dde58244aeaa70b02a4f013fab29d9b0399c9a8a7865ee0d28` |
| Compressed runtime payload | `ac3a0a5a32823239d30f91493e9370547c42d91e9fb085b95cef293746182bff` |

The [source audit](../../outputs/pro5000-release/source-audit.json) verified all
1,114 source-material files, all 93 Ubuntu source-package files, their `.dsc`
checksums, all 56 component-to-notice mappings and the exact current source
fingerprint. All 16 NVIDIA runtime library files match their original input
hashes. The toolkit and reference cuBLAS/cuBLASLt providers in this lock also
have identical original hashes.

The normal launcher test pass lacked a no-exec `/dev/shm`. Running the unchanged
suite with the host's real no-exec `/run/lock` exposed at `/dev/shm` passed all
**16/16** checks with no skips. This mapping uses PRoot and does not alter host
mounts. [Result](../../outputs/pro5000-release/launcher-noexec.json),
[verbose log](../../outputs/pro5000-release/launcher-noexec.log).

## Reproducibility and startup

Two fresh offline builds in different directories produced **identical bytes**
for the executable, its metadata, the complete source archive, `libh3.a`,
`build.json` and both validation manifests. Build times were **1,772.250 s** and
**1,715.974 s**, each with 16 jobs. No compiler/dependency build cache was reused;
only verified downloads were shared.

Cold `--help` startup took **16.662 s**; warm startup took **0.154 s**.
Peak resident memory was **233,544 KiB** cold and **233,452 KiB** warm, including
the launched core. The observed peak cache footprint was **1,944,755,817 bytes**,
sampled every 0.1 seconds. Keep roughly 3.04 GiB for download plus extracted
runtime, with additional space for models, outputs and inference caches.

[Measurements and byte comparisons](../../outputs/pro5000-release/measurements/results.json).

## Final immutable-golden gates

All four gates passed, with unchanged goldens and fixtures:

| Final artifact execution | Exact outputs | Wall time including integrity checks |
| --- | --- | --- |
| Ubuntu 22.04, unpacked | 204/204 | 142.881 s |
| Ubuntu 22.04, single-file | 204/204 | 165.917 s |
| Ubuntu 24.04, single-file | 204/204 | 137.326 s |
| Debian 12, single-file | 204/204 | 142.934 s |

Every run audited 12 application/probe/media processes and found no unexpected
shared-library provider. The Python harness stayed on the host; application
programs executed inside the clean userlands with explicit driver/model/artifact
bindings and no host toolkit or FFmpeg. The same final executable was used in
all three single-file runs.

[Aggregate results](../../outputs/pro5000-release/distros/result.json),
[Ubuntu 22.04 unpacked](../../outputs/pro5000-release/distros/ubuntu2204-unpacked/result.json),
[Ubuntu 22.04 executable](../../outputs/pro5000-release/distros/ubuntu2204-single-file/result.json),
[Ubuntu 24.04 executable](../../outputs/pro5000-release/distros/ubuntu2404-single-file/result.json),
[Debian 12 executable](../../outputs/pro5000-release/distros/debian12-single-file/result.json).

## Local M4

`source outputs/setup/macos-env.sh; make -j8 all test test-server-http` passed,
including all 21 HTTP lifecycle tests, 10 build-policy tests, 18 regression-policy
tests and 27 setup tests. The configured media tools are FFmpeg/FFprobe 9.0.2.
The first sandboxed invocation could not open Metal; the complete suite was
rerun with GPU access and passed.

[Result](../../outputs/pro5000-release/macos-result.json),
[complete log](../../outputs/pro5000-release/macos-tests-full.log).

## Functional validation notes

The final package passed all **26 server cases**, including exact CLI/server
MP4 and AV-state agreement, queued jobs, preview VAE, both image anchors, mixed
image/video/audio references, conditioning reuse, pause/resume, state-only and
decode jobs, continuation/bridge, upscale inspection and refinement, stills,
LoRA, quality presets, Sage/NVFP4, cancellation and queued restart recovery.
Cancellation and interruption are expected outcomes in their dedicated cases.
The library audit observed 78 processes and found no unexpected provider.

[Server results](../../outputs/pro5000-release/server/results.json),
[playable gallery with requests and wall times](../../outputs/pro5000-release/server/index.html).
Examples: [mixed references](../../outputs/pro5000-release/server/M04/video.mp4),
[continuation](../../outputs/pro5000-release/server/M07-hard/video.mp4),
[bridge](../../outputs/pro5000-release/server/M07-bridge/video.mp4),
[refined upscale](../../outputs/pro5000-release/server/M08-refine/video.mp4),
[LoRA](../../outputs/pro5000-release/server/M10-lora/video.mp4).

All nine real regression-profile checks passed, including ordinary 9.0.2 tool
selection, relocated historical tools, missing/changed profile and tool rejection,
path-boundary validation and recovery.
[Profile results](../../outputs/pro5000-release/profile-checks/result.json).

All **16 runtime smoke checks** passed. Checkpoint resume and conditioning reuse
across separate cache roots produced byte-identical AV state and MP4 output.
Stale loader/media variables and fake PATH tools could not substitute package
dependencies; a changed media helper was rejected by the direct worker entry.
Clean distro inventories contained no Python, toolkit or system FFmpeg. The
real unprivileged host user (UID 1000) completed a 256×256/22-frame render in a
stock Ubuntu 22.04 userland.
[Runtime results](../../outputs/pro5000-release/smoke/results.json),
[non-root sample](../../outputs/pro5000-release/smoke/nonroot/video.mp4).

The complete **18-stage CUDA feature suite passed** against the final executable
and matching prebuilt probes. It covered CoW LoRA, bridge CLI validation, real
reference/anchor/continuation renders, sampler resume, reuse/core/reduction,
the option matrix, attention/SOL/SubBlock contexts, conditioning invalidation,
weight placement and cancellation/recovery, full VAE and stills, patch operators,
injected memory exhaustion, current-state cropping, VAE lifetime, decode-failure
recovery and policy validation. Production media remained 9.0.2 throughout.

[CUDA suite results](../../outputs/pro5000-release/features/result.json),
[integration log](../../outputs/pro5000-release/features/feature-integration.log).

All **six HTTPS checks** passed: each of Ubuntu 22.04, Ubuntu 24.04 and Debian 12
rejected an invalid CA path and completed an HTTPS reference-image render using
the provisioned system trust store. Each dedicated library audit passed.
[HTTPS results](../../outputs/pro5000-release/https-distros/results.json),
[Debian 12 sample](../../outputs/pro5000-release/https-distros/debian12/video.mp4).

Two renamed package versions ran concurrently using one cache parent and
different runtime IDs. Both completed real video jobs; each server kept its
original core executable path while the other version started and ran.
[Concurrent-version results](../../outputs/pro5000-release/server-versions/results.json).

The [complete functional record](../../outputs/pro5000-release/qualification/results.json)
ties every stage to the final artifact. The server matrix took 1,674.167 s,
the full CUDA feature suite 1,988.424 s, runtime smoke 166.683 s, HTTPS 158.651 s
and concurrent-version validation 100.378 s, including each stage's wrapper and
library sampling where applicable.

The first server-matrix attempt hit that harness's fixed 10-second startup
wait while the cold standalone payload was still extracting; no server job ran.
After a successful ordinary `--help` launch populated the runtime cache, the
matrix was restarted. Cold extraction is tested independently by the artifact
gates, launcher suite and real-runtime smoke suite. The failed first attempt
is retained in the evidence; it is not counted as a pass.

Loaded-library evidence is taken from the four dedicated distro gates, the
server/smoke/feature stages and each HTTPS helper's own sampler. The aggregate
sampler watches only the release and qualification directories, so helpers with
separate cache roots can report no observed math libraries there. Those empty
observations are not counted as successful dependency audits. Profile-policy
checks instead verify selection, hashes and rejection behavior; the two-version
check verifies distinct runtime IDs and stable running core paths.

The final cleanup audit found no remaining application/media workers and all six
test server ports closed. No CUDA compute process remained, and GPU memory
returned to 22 MiB. The temporary LoRA weight cache and abandoned extraction
from the first cold-start attempt were removed after their processes stopped.
[Cleanup audit](../../outputs/pro5000-release/final-host-audit.json),
[unchanged source/golden/fixture check](../../outputs/pro5000-release/final-source-check.json).

## Redistribution handoff

The source-material inventory is an engineering record of the selected files,
licenses and matching sources. The executable and its companion source archive
must be supplied together on the release download page, with both checksums.
The historical validation kit stays private. Nothing is published by this work.
See [source/relink instructions](../../scripts/linux/SOURCE-MATERIALS.md) and
the component license copies supplied with the runtime.
