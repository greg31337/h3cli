# Portable Linux distribution qualification

Updated 2026-09-30: **the production package includes FFmpeg and FFprobe 9.0.2**
for rendering, reference decoding, media inspection, server jobs and
current-feature tests. The historical 4.2.2 output encoder and 6.1.1 input
decoder/probe are retained only for the recorded SGLang parity regression, in
the separate private validation kit. Neither historical codec is shipped in
the standalone executable.

| Use | Version and location |
| --- | --- |
| Production input/output encoding and decoding | FFmpeg 9.0.2, packaged as `tools/ffmpeg` |
| Production media inspection | FFprobe 9.0.2, packaged as `tools/ffprobe` |
| Recorded SGLang regression output | Exact FFmpeg 4.2.2 encoder, private `linux-validation/media` kit |
| Recorded SGLang regression input/probe and output verification | FFmpeg/FFprobe 6.1.1, private `linux-validation/media` kit |

**Current release qualification passed on PRO 5000**, including all 204 recorded
outputs in Ubuntu 22.04, Ubuntu 24.04 and Debian 12. The source-material audit
is complete for the 9.0.2 package. See the
[PRO 5000 completion report](linux-distribution-pro5000.md) for artifact hashes,
current measurements and acceptance evidence, and the
[FFmpeg migration report](ffmpeg-upgrade.md) for media selection and test isolation.
All 204 expected hashes and 12 fixtures remain unchanged. Nothing has been
published. [Build guide](linux-distribution.md),
[completed checklist](linux-distribution-tasks.md).

The PRO 6000 measurements below describe the **pre-migration artifact** and
remain historical evidence. That artifact completed functional validation but
failed release acceptance because of a cross-device golden mismatch and an
incomplete source-material audit for its old encoder. Those results must not
be read as the current package's status or measurements.
[Historical acceptance summary](../../outputs/linux-distribution/final/acceptance-summary.json).

## Implementation

`scripts/build_linux.sh` fetches verified inputs and builds in a fresh, pinned
Ubuntu 22.04 filesystem using PRoot. Compilation needs no GPU or host toolkit.
The output includes a single executable, its unpacked runtime, matching validation
probes, `libh3.a`, checksums, dependency notices and available source materials.
The normal Makefile and Ubuntu setup remain supported.

The static launcher extracts into a user-owned, content-addressed cache, then
executes the native core. Extraction is locked, bounded, hash-checked and atomic.
Warm starts check readiness metadata; direct server workers perform the same
runtime checks. The current package pins both cuBLAS providers, cuDNN, NVRTC,
JPEG and source-built FFmpeg/FFprobe 9.0.2. Input and output use the same packaged
FFmpeg executable. Historical regression codecs are selected through the
hash-verified private validation profile outside the payload. Models, mutable
caches, driver, glibc/NSS and system certificate trust stay outside the payload.

Conditioning and sampler identities use verified bundled resource content rather
than cache paths. CLI cwd semantics and model defaults are preserved. Server
publication now honors `H3_FFPROBE`, including the packaged tool selection.
The existing SGLang-plus-`h3cli` request schema is unchanged.

## Historical PRO 6000 environment and locks

The original Linux compilation and CUDA execution used the authorized RTX PRO 6000 node.
The host was Ubuntu 24.04.3, kernel 6.8.0-138-generic, NVIDIA driver 610.57.04,
RTX PRO 6000 Blackwell Workstation Edition (SM120, 97,887 MiB), and an AMD EPYC
9575F CPU. The provider prohibits nested user namespaces and supplies no Docker
daemon. PRoot provides filesystem translation, not a security sandbox. The
three tested userlands share this host's kernel and driver.
The [initial inventory](../../outputs/linux-distribution/run01/node-preflight.log)
also records the provider's preinstalled CUDA 12.8 libraries. The build used
its pinned CUDA 13.0.3 filesystem; loaded-path audits exclude those host toolkit
libraries from packaged execution.

The original lock recorded exact OCI digests, 55 Debian package files, six source
archives and the PRoot executable hash. The current
[lock.json](../../scripts/linux/lock.json) also pins the production FFmpeg 9.0.2
source; native wheel pins remain authoritative in
[requirements-sglang-runtime.txt](../../scripts/requirements-sglang-runtime.txt).
The historical builder used CUDA 13.0.3 / nvcc 13.0.88 on Ubuntu 22.04, reference
cuBLAS 13.1.1.3, cuDNN 9.20.0.48, input FFmpeg 6.1.1, TurboJPEG 2.1.5 and ICU
74.2. Its output encoder was the exact imageio-ffmpeg 0.5.1 wheel's FFmpeg 4.2.2
binary. These two historical media versions now belong only to the regression kit.

Every bundled ELF is audited for x86-64, relative RPATH, dependency closure and
GLIBC imports no newer than 2.35: 57 ELF files passed. The
[runtime manifest](../../bin/linux-release-before-ffmpeg902/linux-runtime/share/h3cli/runtime.json)
records each object's required symbol versions, search paths and host dependencies;
[build.json](../../bin/linux-release-before-ffmpeg902/build.json) records packages and embedded CUDA
code. The release compiles the existing SM86, 89,
90, 100 and 120 SASS/PTX coverage, including optional Sage/SOL/SubBlock support.
Only SM120 was physically exercised in this work. Older drivers, other kernels,
other GPU architectures, musl and ARM64 are not qualified by these measurements.

## Historical PRO 6000 numerical acceptance

The unchanged starting source completed all 204 outputs on PRO 6000, but only
91 matched the immutable PRO 5000 golden hashes; 113 differed. This is the same
cross-device limitation previously documented for the 5090. It predates the
packaging changes. The [full baseline record](../../outputs/linux-distribution/baseline/result.json)
contains every output hash. Its identifiers are:

| Record | SHA-256 |
| --- | --- |
| Starting source | `4302af383ac5b1c0b5b81580007a2ec3a6f8b5b01026f178dbba80e085ec7de8` |
| Immutable golden manifest | `fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3` |
| Baseline executable | `17a5b32bb1bf13f4cea1176985e730e5a3606df07785aa5dd5b8f6bbd2fc36fc` |

The final portable executable reproduced all 204 unchanged-source PRO 6000
outputs exactly in Ubuntu 22.04, Ubuntu 24.04 and Debian 12, including the
unpacked runtime and outer executable. This comparison is diagnostic: it does
not turn the failing immutable-golden gate into a pass. All 204 expected hashes and
12 input fixtures remain unchanged; there are no tolerance or skip options.

| Final artifact execution | Complete outputs | Immutable goldens | Same-device baseline | Full gate wall time |
| --- | --- | --- | --- | --- |
| Ubuntu 22.04, unpacked runtime | 204/204 | 91 match, 113 differ — failed | 204/204 identical | 45.031 s |
| Ubuntu 22.04, single file | 204/204 | 91 match, 113 differ — failed | 204/204 identical | 54.545 s |
| Ubuntu 24.04, single file | 204/204 | 91 match, 113 differ — failed | 204/204 identical | 48.587 s |
| Debian 12, single file | 204/204 | 91 match, 113 differ — failed | 204/204 identical | 47.585 s |

All four loaded-path audits passed, observing 15–18 application processes per
run and no unexpected libraries. The gate still returned failure each time.
The [distro record](../../outputs/linux-distribution/final/distros/result.json)
links each execution to the exact artifact; per-run directories also contain
all output hashes, commands, render logs, videos and library maps. The Python
test harness stayed on the host; application programs ran inside clean
userlands without host toolkit, Python or FFmpeg dependencies.

NVIDIA limits cuBLAS bitwise reproducibility guarantees to the same architecture
and SM count. That is a plausible explanation for the hardware-dependent
outputs, not proof of the exact cause of each difference.
[NVIDIA reproducibility documentation](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility).

The user subsequently authorized the local PRO 5000 for completion; its new
results are recorded in the linked completion report. This historical report
retains the PRO 6000 outcomes without treating them as strict-golden passes.

## Historical functional validation

Host, policy and sanitizer checks preceded the final artifact freeze. The final
artifact then passed the complete CUDA feature, server, relocation and additional
HTTPS/version tests. Its strict numerical gate remains failed as recorded above.

| Check | Result |
| --- | --- |
| Local M4 `make all test` | Passed |
| M4 HTTP lifecycle suite | 21/21 passed |
| M4 server contract and HTTP sanitizers | Passed; HTTP 21/21 |
| Native Linux Make test recipes, executed outside PRoot | 63/63 commands passed |
| Locked-build policy checks | 8/8 passed |
| Recorded-gate policy checks | 15/15 passed |
| Linux launcher lifecycle/fault tests | 16/16 passed |
| Linux launcher ASan/UBSan/LSan | 16/16 passed |
| Prebuilt current-CUDA suite | 18/18 stages passed |
| Real packaged CUDA server matrix | 28 cases passed, including expected cancellation/interruption/rejection |
| Final real server loaded-library audit | 86 application processes, 57 shared-object paths, no unexpected provider |
| Clean-userland server library audit | Passed on Ubuntu 22.04, Ubuntu 24.04 and Debian 12 |
| Real non-root render | UID/GID 65534; 256×256, 22 frames, two steps |
| Relocated checkpoint and conditioning | Exact AV bytes; resumed MP4 bytes identical |
| Concurrent versions, shared runtime cache | Both servers completed real jobs; separate cache IDs, running core paths unchanged |
| HTTPS and invalid-CA rejection | Passed in all three clean userlands; six checks |
| Final worker/listener cleanup | No active application processes or listening servers |

The current-CUDA suite covers references, anchors, continuation, resume, reuse,
core reuse, reduction, all twelve attention/precision combinations, SubBlock,
conditioning, residency, stills, full VAE, operator checks, memory lifetimes and
failure recovery. The server matrix additionally covers preview delivery,
upscale, native LoRA, uploads/downloads, CLI/server equality, queued restart,
HTTPS import and rejection of an unavailable CA store.

PRoot interfered with a chmod/ctime assertion and a SIGSTOP-based source-mutation
test. Both passed natively, as did the complete native Linux test recipe set.
Sanitizers were also run natively because tracing and ASan are incompatible in
this environment. These harness limitations were not bypassed by changing
application behavior or weakening assertions.

The real LoRA cache recorded 15 cloned files and zero full-copy fallbacks. Its
logical size was approximately 62 GiB; provider free-space accounting fell
sharply during this test. Derived weights were removed after the server stopped,
while cache manifests and rendered evidence were retained. Models were preserved.

The [offline sample gallery](../../outputs/linux-distribution/final/index.html)
contains 22 hash-verified video/image files from the final server's 28 cases,
with requests and client wall times. These small, bounded jobs verify behavior;
they are not a visual quality benchmark. Raw evidence includes the
[server results](../../outputs/linux-distribution/final/server/results.json),
[final CUDA feature suite](../../outputs/linux-distribution/final/features/result.json),
[15 runtime smoke checks](../../outputs/linux-distribution/final/smoke/results.json),
[simultaneous server versions](../../outputs/linux-distribution/final/server-versions/results.json),
[HTTPS userland checks](../../outputs/linux-distribution/final/https-distros/results.json),
[final cleanup](../../outputs/linux-distribution/final/cleanup.json),
[63 native Linux test recipes](../../outputs/linux-distribution/preflight/native-all.json),
[M4 host/HTTP log](../../outputs/linux-distribution/run01/m4-server-probe-fix.log),
[M4 sanitizer log](../../outputs/linux-distribution/run01/m4-server-final-sanitize.log),
and [Linux launcher sanitizer log](../../outputs/linux-distribution/preflight/launcher-sanitize-native.log).
Generated evidence and binaries are intentionally gitignored; these relative
links refer to this working checkout, not hosted downloads.

The HTTPS runs provisioned the host's CA certificate file at the userland's
standard system path. Each invalid explicit override was rejected; restarting
without the override enabled discovery and a real HTTPS-reference render.
No CA bundle was added to the application payload. Both concurrent servers used
renamed executable files containing spaces and one shared runtime-cache parent.

## Historical reproducibility and footprint

Two clean offline builds of the final source produced byte-identical single-file
executables. Their SHA-256 is
`1d1a2eb08692990adae1ed1541cbb8c63e88c6dd7f07b3dff035fb2074a01409`.
The artifact's source fingerprint is
`e9833df81385a62b909ad43ddda4c97d4199434bad446d9651f5c083cad88937`,
which matched that build's code, build recipes and tests. The source
was transferred as an archive; its Git identity is therefore recorded as
unknown rather than inferred from the local checkout. The outer snapshot record
also captures documentation and other checkout files without making report text
part of numerical state identity.

All twelve compared CUDA/host policy objects in the preceding normalized pair
also matched. Every qualification record in that historical campaign names the
executable above; its checks ran against the frozen code, build recipes and tests.
The later FFmpeg migration produced separately identified artifacts. The historical
[executable](../../bin/linux-release-before-ffmpeg902/h3cli-linux-x86_64) and
[checksum](../../bin/linux-release-before-ffmpeg902/h3cli-linux-x86_64.sha256) are available locally.

| Measurement | Final artifact |
| --- | --- |
| Compressed executable | 1,338,741,316 bytes (1.25 GiB) |
| Payload | 344 files; 2,014,313,850 bytes (1.88 GiB) |
| Observed peak cache allocation by file length | 2,014,352,451 bytes, including readiness metadata |
| Cold runtime-cache `--help` | 5.323 seconds; peak RSS 231,652 KiB |
| Warm runtime-cache `--help` | 0.054 seconds; peak RSS 231,520 KiB |
| Clean offline build times, eight jobs each | 720.774 and 718.799 seconds |

Cold means an empty extraction cache, not an empty OS page cache. These startup
measurements include the native core's initialization through `--help`, not
model loading or rendering. Peak cache size was sampled every 100 ms and checked
again after completion. The downloaded file plus this cache require about
3.12 GiB; model, LoRA, quantization and application outputs are additional.
[Machine-readable measurements](../../outputs/linux-distribution/final/measurements.json).

NVCC's `--frandom-seed` fixes generated symbol names, but its host object symbol
tables still contained PID-bearing temporary filenames. The wrapper removes
host debug/file metadata with `objcopy --strip-debug` before linking. The
executable sections and CUDA fatbins, including device line information, remain
unchanged; no numerical compiler flags changed.
[NVCC option documentation](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-compiler-driver-nvcc/index.html#frandom-seed-frandom-seed).

## Redistribution audit

The current package builds FFmpeg 9.0.2 from pinned source and supplies that
archive, its build recipe and matching dependency sources/patches in the
companion source archive. The final PRO 5000 source-material audit is complete;
the companion archive must accompany the executable when published. See the
[source-material guide](../../scripts/linux/SOURCE-MATERIALS.md) and
[completed audit](linux-distribution-pro5000.md#final-artifact-identity).

The old 4.2.2 binary is no longer a production redistribution dependency. Its
unresolved source mismatch is retained here as historical evidence for the
private regression encoder, not as an open audit item for the 9.0.2 package.
That encoder has SHA-256
`700073daef5c23bbcb18c2eae60553a454a5221ec19b4a88c8c367a664671a7c`.
The encoder in the upstream 4.2.2 archive inspected during that audit hashed to
`a05b380d4f336eac5a027f90ee9f60073175ae60ca65ca3ea1001585385b4836`.
Their executable sections differ, so matching version/configuration text does
not establish corresponding source for the pinned binary.

The pre-migration package included GPLv3 text and labeled the upstream readme
as a comparison, not provenance for the wheel binary. Its historical
`share/licenses/output-ffmpeg/source-audit.json` set `publication_ready` to false.
That artifact's available source archives and build recipes did not complete
the audit. Production moved to the source-built 9.0.2 package; the exact old
encoder remains unchanged in the isolated regression kit.

## Historical acceptance blockers and subsequent completion

The checklist then recorded 25 completed tasks and seven open tasks.
LNX006 and LNX024–026 required a passing immutable-golden gate, which the
PRO 6000 did not satisfy even with unchanged source. LNX009 required completion
of the redistribution source-material audit. LNX030 and LNX032 required a
qualified release and all acceptance gates to pass, so were also open then.

The subsequent FFmpeg 9.0.2 package and PRO 5000 qualification completed all
32 tasks, including the source audit and immutable-golden gates. The historical
PRO 6000 mismatches remain recorded; they were not accepted as parity passes.
See the [completed checklist](linux-distribution-tasks.md) and
[final qualification report](linux-distribution-pro5000.md). Nothing has been published.
