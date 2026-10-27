# Five-command release acceptance

Validation run: 2026-10-27. **Complete:** all five commands passed, including native
builds/tests, signed packages, final package tests and interrupted/resumed GitHub
publication followed by a verified published retry.
This records the evidence for [REL001–REL008](../todo.md), not a production release.

## Candidate and systems

The application and packaging candidate uses isolated validation commit
`b7498fc5c0f7c8af496d420f87c560cb36c9e7f2`. Subsequent caller-side failure reporting,
draft discovery, process cleanup, noexec-check collection and the platform-specific
runtime-cache lookup are covered by 101 release-tool tests on both hosts, actual
failure/cancellation invocations and the real-host launcher check. Package
verification uses the current test driver
with the selected source's scripts and records the driver's hash; changing it
invalidates previous test results. These refinements do not change application
binaries or the selected source archives.

| Target | Tested system |
| --- | --- |
| macOS arm64 | macOS 26.6.2, Apple M4 Max |
| Linux x86-64 | Linux 6.8.0-146, glibc 2.39, RTX PRO 5000 72GB Blackwell, driver 595.99.02, compute capability 12.0 |

The candidate uses the already qualified
[dependency bundles](../build/dependency-bundles-results.md). Their compatibility
and contents are verified automatically; application builds do not rebuild those
dependencies. Linux native tests use the locked dependency libraries on the real
host, with host filesystem semantics. Configuration automatically selects cloning
scratch on the cache volume when the workspace volume does not support reflinks.

The later shared process-cleanup fix changes a recipe-tracked helper. The first
release selecting a commit with that fix will automatically build and qualify a
replacement dependency bundle for each platform. Subsequent releases with the
same recipe reuse it. The artifact candidate above used the earlier qualified
recipes; its bytes are unchanged by the caller-side cleanup fix.

## Completed checks

- Signing setup imported existing settings, reused credentials, and saved an
  encrypted disposable Minisign key's unlock password in Keychain. Unattended
  signing and verification passed. The existing project key was also verified
  without replacement. Configuration checks both selected hosts and credentials.
- `build-local` passed on both hosts. A warm retry reused both builds and kept
  the completion receipt unchanged. Temporarily changing one byte in a disposable
  test executable made `test-local` fail; restoring the exact bytes restored reuse.
- All 101 release-tool tests passed on macOS and Linux. They cover configuration,
  stale inputs, partial drafts, conflicting public assets/tags, failed-upload
  placeholders, cancellation, notarization resumption and failure reporting.
  The CUDA gate's 18 fixture/contract tests also passed.
- Mac native host/API tests, 42 LoRA tests without skips, all 19 real-model
  rendering/state cases, and online/offline server checks passed. The unsigned
  Mac package build also passed, and its source archive's 1,141 tracked files
  match the selected commit exactly.
- Linux native host/API tests and 42 LoRA tests without skips passed. The full
  **204-output CUDA reference gate** and all 19 rendering/state cases, including
  online/offline server checks, passed. All 18 additional CUDA feature/lifecycle
  checks passed, including continuation chains, restart equivalence, placement,
  cancellation recovery, VAE lifetime and memory/error handling. `test-local`
  ended with PASS for both hosts.
- Actual Ctrl+C cancellation returned 130 and left no owned remote workers.
  Closing the SSH lifeline also stopped its remote child. Failure attempts and
  logs were retained; retries used fresh work directories automatically.
- Publication testing found that the GitHub tag endpoint omitted the existing
  draft; discovery now also checks the authenticated, paginated release listing.
  A Mac cleanup race returned a permission error for an unreaped zombie group;
  cleanup now retries that probe while still propagating errors sending TERM/KILL.
  Five real local cancellations passed after the fix. A real interrupted draft
  retained its completed first asset and returned 130.
  The 18 dependency/process tests passed on both hosts, and the complete
  204-output CUDA gate passed again after the shared cleanup change.
- `build-packages` passed. Apple accepted both the inner runtime and outer DMG
  submissions; the DMG was stapled and verified. All 11 collected assets and
  their Minisign-signed checksum list verified.
- `test-packages` passed against those exact final bytes. Both platforms passed
  all 19 rendering/state cases and online/offline server checks outside the
  source checkout with isolated caches. The final Linux executable passed all
  204 reference outputs. The Mac executable ran from the mounted stapled DMG
  and matched the standalone asset. Both source archives matched every tracked
  application file in the selected commit.
- The Linux build fixture skipped its `/dev/shm` noexec case because that mount
  permits execution. The final executable's exact launcher was extracted and
  tested with a small payload on `/run/lock`, which is noexec. Executable-storage
  startup passed; noexec-cache startup failed with the required diagnostic. The
  final-package command collects this check automatically, without a rebuild.
- All 85 relative links checked in the revised release/build guides resolved.

The original golden manifest is unchanged:
`fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3`.

## Tested package artifacts

The final signed/checksummed assets have these SHA-256 identities:

| Asset | SHA-256 |
| --- | --- |
| Linux standalone executable | `c45d42150a6398675f474ba6c06f28c3d4863c4314b3b149ce58409f988fdb08` |
| Mac standalone executable | `15932ff1e1dd3f4c932ecea04db887c1b4f849747e6c0789fd0633d8dc36c706` |
| Mac stapled DMG | `14a2cd42307f3d70e018ce3a338b210915a8f66c862cc0f25dd10ca2668eb60e` |

## GitHub publication acceptance

The real checks used a private disposable repository and a prerelease label.
Cancellation was injected between uploads, after the first asset completed.
The draft retained that asset. The normal `publish` command verified it, uploaded
the remaining assets, downloaded and checked all 11 files, verified Apple and
Minisign signatures, and rendered with the downloaded Linux executable on the
GPU host. The render's dimensions, 22 frames and decoding were checked.

GitHub confirmed a published (non-draft) prerelease at the exact selected source
tag. The original uploaded asset kept its ID and bytes. Running `publish` again
downloaded and verified the published assets and repeated the Linux render.
Every asset ID and SHA-256 remained unchanged, and the retry made no upload or
release-edit calls. Both invocations ended with `PASS publish`.

Private raw logs, receipts and failure-injection evidence remain under ignored
`outputs/release-validation-20261027/`, with remote reports identified by their
worker receipts. No credentials, personal host paths or Apple account records
belong in this report.

The tested systems use existing maintainer accounts. Fresh installations,
browser-quarantine experiments, other GPUs/operating systems and human visual
review are outside this acceptance scope; the release notes state those limits.
