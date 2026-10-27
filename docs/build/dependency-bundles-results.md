# Dependency bundle qualification

Updated 2026-10-22. Both platforms passed qualification and reuse across two
application revisions on the local M4 and RTX PRO 5000.
The [operating guide](dependency-bundles.md) describes production, qualification,
cache verification and release configuration. The
[archived milestone](../release-tooling-history.md) tracks acceptance for both
platforms.

## Reference applications

Validation uses isolated Git checkouts. Revision B adds a comment to
`src/h3cli.c`: the application commit and source identity change while the
dependency inputs remain unchanged. These are validation revisions, not published
release tags. The Linux pair includes the producer directory-permission fix.

| Platform | Revision A | Revision B |
| --- | --- | --- |
| macOS arm64 | `69a0e4a08f61d17a9034b756803852743a3eac47` | `f4b8331df7ad6e6f1d9099cdc3d15b584f22751c` |
| Linux x86-64 | `6671a0f0b1fb69ed4b90ad94b6f871889211faf3` | `084a33e0ada43c322c3036cbe12fc45838e07d8a` |

Revision A qualifies the dependency archive. Revision B uses that qualified
reference through the ordinary release runner, including an offline application
build and fresh artifact qualification. The shared cache is outside both
application checkouts.

## macOS evidence

- Recipe ID: `46679d1bffd32a2e353be57e094b3122fd839a194b048cd8824afa55d4c85074`.
- Archive SHA-256: `229d5264615d37fea7d84ad1b2ea355baeaafb7f40747b91597ab0a3898c06b1`.
- Qualification SHA-256: `2b7e23e1bed98bffae48565b6f4970434e3b19cb5ad7f0f34e386ac6b0d67272`.
- Cold dependency production: 56.3 seconds, eight jobs, provisioned offline inputs.
- Revision A qualification: 582.8 seconds; all 19 feature/server cases passed,
  alongside package fixtures, media, architecture/deployment and relocation checks.
- Revision B build sequence: 53.2 seconds including prerequisites, cache
  verification and package fixtures; `dependency_compilation` is false.
- Both application builds used the same archive and recipe. All 359 shared
  payload files retained their hashes and modification times after building and
  Developer ID signing. Candidate media copies received the signatures.
- Apple accepted the inner runtime and outer DMG; stapling, Gatekeeper and
  executable notarization checks passed. All 19 final signed feature/server cases
  passed, and the release runner wrote its successful test receipt.
- The companion source archive was checked against all 1,127 application source
  files, both exact upstream archives and the dependency identity.

Independent cold compilations produced identical FFmpeg, FFprobe and x264 static
library bytes. Signing and notarization use per-candidate copies of these inputs.
The existing clean-environment/fresh-trust tasks in the macOS milestone remain
separate; this work uses the existing M4 account.

## Linux evidence

Recipe ID: `efcb352743c8faa8ad78a7e5b71d6787ff3136e75b02f2cae5f2ccaa18258a91`.
Archive SHA-256: `5f87462c6832264e78e98ab98d1b6045d482214c655da9c85761955fc5683b13`.
Qualification SHA-256: `807a3f5c8a8dd741591366df8617ac34b40fc5e598357388ff70f2183e39e04d`.
The archive is 5,475,706,880 bytes with 275 inventoried payload files, including
the hash-bound OCI layers. Cold production took 1,028.1 seconds with 16 jobs and
provisioned offline inputs. Two fresh application builds took 925.1 and 934.7
seconds with dependency compilation disabled. Their standalone executables,
source archives, provenance and packaged media tools are byte-identical. Revision A
qualification passed in 2,707.3 seconds, including all 204 numerical outputs,
four distribution/mode checks with 204 outputs each, all 16 runtime checks and all
19 feature/server cases. The distro matrix covers Ubuntu 22.04 unpacked and
single-file execution, plus single-file execution on Ubuntu 24.04 and Debian 12.
The final producer
source passed the complete 204-output CUDA gate in 156.1 seconds with golden
manifest SHA-256 `fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3`.

Revision B's ordinary release build completed offline in 938.8 seconds with
`dependency_compilation` false and the same qualified archive, recipe and report
hashes. Its complete release sequence, including prerequisites, model checks,
artifact tests and private export, took 2,692.6 seconds. All 204 numerical outputs,
four distribution/mode comparisons, 16 runtime checks and 19 feature/server cases
passed again on this final artifact.
All 275 shared bundle payload files retained their hashes and modification times
through both builds and test suites. The 55 packaged library/media files are
byte-identical between the two application revisions. Build, test and export
completion receipts passed the release runner's resume verification.

The source archive audit checked all 1,231 inventoried files, including 1,127
application files, the exact upstream/source-package archives and the dependency
identity, for both application revisions. An initial feature run stopped because
the selected model root lacked the auxiliary VAEs; it received no qualified
reference. After provisioning all
pinned model groups, all 19 feature/server cases passed on that artifact. The
fresh qualification run passed on the independently rebuilt, identical artifact.

Two independent source compilations produced identical ICU data/common/
internationalization libraries, TurboJPEG and all four production/historical
FFmpeg/FFprobe executables. The first compilation's
archive was rejected for inherited setgid directory permissions; it was retained
as failed evidence and never given a qualified reference. Production now uses an
ordinary-permission source checkout and normalizes the Linux payload root.

## Cache, cancellation and release checks

The 18 dependency tests and 55 release-runner tests passed on both hosts. The
13 Linux build-host tests and 18 CUDA gate-host tests also passed. Coverage
includes immutable pins, incompatible recipes/platforms, qualification binding,
offline misses, corrupted archives and manifests, interrupted downloads, unsafe
archive entries, concurrent consumers and repeated cancellation.

The Linux launcher fixture suite passed 15 cases; its hardcoded `/dev/shm`
no-exec case was unavailable because that mount permits execution on this host.
The same assertion was then exercised on the existing `/run/lock` no-exec mount:
an executable cache succeeded and a no-exec cache failed with the expected
actionable error. No mount settings or test acceptance criteria were changed.

An actual canceled Linux dependency build stopped all 25 owned processes and
created no candidate/completion record. Closing an actual SSH input stream
stopped nested workers with exit status 130 while an unrelated test process
remained alive. Completed cached dependencies were retained.

The final Linux export was collected and verified on the Mac against all six
exported file digests, release identity, qualification reports and the pinned
dependency reference. An intentionally different reference digest was rejected.
The private machine settings now select these qualified bundles and shared
dependency/download caches; the previous settings are backed up with the evidence.
The current working-tree code fingerprint and both dependency recipe IDs still
match the completed CUDA gate and qualified bundles. Documentation checks passed
for all 106 local links in the 13 changed Markdown files.

Raw reports, source snapshots, archives, receipts and failed-attempt logs are
retained under ignored `outputs/dependency-bundles-20261022/` on the Mac and the
private dependency-validation workspace on the PRO 5000. A local archive also
retains 612 checksum-verified Linux evidence files. Machine addresses,
Apple identifiers, submission IDs and raw configurations remain private.
