# Simple build, test and release tooling

Status: **complete — 8/8 tasks implemented and validated**.
Updated 2026-10-27.

Make releases practical for a single maintainer working on an experimental/hobby
project. After signing setup and environment configuration, the entire process
must require only the five commands below. Reuse the working builders, dependency
bundles and tests; simplify the public interface and remove manual handoffs.

The commands are implemented and documented in the
[current release guide](release/README.md). Native and signed-package acceptance
passed on both hosts, followed by interrupted/resumed publication and a verified
published retry in a private disposable repository. See the
[acceptance record](release/results.md) for the checks and tested scope.

## The only manual steps

1. **Initial signing prerequisites:** `./release/release.sh setup-signing`.
   Reuse or help provision GitHub login, the existing Minisign key, and the
   Developer ID identity/Keychain notary profile for Mac packages. Include Apple
   account, certificate and credential-access setup here. This must work before a release
   config exists; never replace working credentials just to start another release.
2. **Configure the release environment:**
   `./release/release.sh configure --version v0.6.0`.
   Ask for missing machine/model paths and the GitHub repository, discover tools
   and existing credentials, and save one ignored configuration. Remember host
   settings across releases; subsequent configuration normally changes only the
   version. Default to a GitHub prerelease, with a stable label selectable here.
   Check credentials and SSH access, provision the required tools/models with
   progress, and save environment settings so later commands need no shell exports.

Then run these five commands in order:

| Manual action | Command | Work performed automatically |
| --- | --- | --- |
| Build local binary on macOS / Linux | `./release/release.sh build-local` | Resolve dependencies and compile the native binary and its test helpers on each selected host. Print binary paths. |
| Test local binary | `./release/release.sh test-local` | Run the required host and GPU tests against those builds and save results automatically. |
| Build release packages | `./release/release.sh build-packages` | Build standalone deliverables and matching sources, collect both platforms, sign/notarize/staple the Mac distribution, and generate signed checksums. Print package paths. |
| Test release packages | `./release/release.sh test-packages` | Extract/mount and run the actual final packages, check signatures, contents and hashes, and save results against those exact bytes. |
| Release packages on GitHub | `./release/release.sh publish` | Generate notes, create the tag/draft, upload, download and verify the assets, then publish and print the public URL. |

For the existing two-machine setup, run this sequence from the Mac: each command
runs its macOS work locally and its Linux work over the configured SSH connection.
Compilation and GPU tests run natively on their respective hosts. The same
interface supports configuring just one platform. A selected target must pass;
an unreachable Linux host cannot silently become a Mac-only success. No manual
SSH commands, source copies, artifact transfers or matching run names are needed.

## Work that belongs inside the commands

- Configuration selects an existing source commit, defaulting to `HEAD`, and
  clearly shows whether the working tree differs. Build an automatic snapshot of
  that commit on both hosts; exclude uncommitted changes explicitly. Cloning,
  source synchronization, version metadata and candidate directories are internal
  details. Ordinary development commits are not extra release preparation steps.
- Keep [reusable dependency bundles](build/dependency-bundles.md). Discover and
  verify compatible cached bundles automatically. If none is available,
  `build-local` obtains one or builds and validates it once from the locked inputs,
  then caches it. Announce this work and its progress. Unchanged application
  releases reuse it; operators never copy bundle hashes or invoke separate
  identity/build/qualify/pin commands. Offline misses fail with a useful reason.
- Use one small shared implementation behind the commands and a simple saved
  state file under `outputs/release/`. Record source/configuration, dependency and
  artifact identities and test results automatically. No hand-written reports,
  `record-check.sh`, placeholder editing, manual checksum signing or extra
  collect/stage/verify/publish commands are part of the normal workflow.
- Each stage uses the preceding stage's matching output. Source, build settings
  or artifact changes invalidate affected later results; packaging-time rebuilds
  and signing are covered by `test-packages`. Tests never rebuild the tested
  artifact silently, and `publish` never rebuilds or re-signs it.
- `build-packages` includes the existing Apple submissions and waits for their
  results. Other build/test commands do not submit to Apple. Only `publish` writes
  tags or release assets to GitHub; invoking it is the publication action, with
  no separate approval file, browser button or final publish command. Use system
  credential stores configured during setup, not passwords in config or logs.
- Retrying the same command resumes verified completed work and allocates fresh
  attempts for failed work. Preserve failure logs; do not require deleting folders
  or inventing a new run ID. Ctrl+C, SIGTERM and SSH disconnects stop owned workers
  on both machines. Resume notarization/upload attempts without duplicating them.

Publication may use a draft internally, but must finish with a published release.
Generate usable notes from commits and actual test results; custom notes are
optional configuration. Verify the remote tag and the downloaded bytes against
the tested packages before publication. On retry, reconcile a matching draft or
already-published release; never overwrite a conflicting public version. Confirm
the final public state before reporting success. GitHub CLI already supports
[generated notes and explicit target commits](https://cli.github.com/manual/gh_release_create)
and [publishing a draft](https://cli.github.com/manual/gh_release_edit).

## Tests and final results

Keep the existing correctness checks automated inside the two test commands:

- `test-local`: required host/release-tool tests and bounded real-model tests on
  the selected native backends, including the complete Linux **204-output CUDA
  reference gate** required by [CONTRIBUTING.md](../CONTRIBUTING.md). Prepare test
  fixtures and saved states automatically; operators do not run setup renders.
- `test-packages`: run the final standalone binaries outside the checkout with
  isolated runtime caches; exercise rendering, CLI/server startup and packaged
  media dependencies; verify Mac signatures/notary tickets/stapling, source
  materials, asset lists and checksums. Include the existing final Linux artifact
  CUDA gate. Generated reports identify the exact tested binaries/packages.
- Automate the current downloaded-Linux check within `publish`, using the same
  configured GPU host. Keep source tests and downloaded-asset verification tied
  to the release without manual report attachment.

Fresh Mac accounts/installations, browser-quarantine experiments, broader OS/GPU
matrices and human video review are optional development investigations, not
recurring publication requirements. Record the actual tested systems and known
limits in release notes; never describe an existing developer account as a fresh
trust environment. This replaces the old manual stable-release gates for both
release labels. Automated signing, artifact and numerical failures still block
publication; a required missing prerequisite or skipped test is not a pass.

Every one of the five commands must end with one unambiguous summary line:

```text
PASS build-local: macos-arm64, linux-x86_64; binaries: <paths>
PASS test-local: all required tests passed; report: <path>
PASS build-packages: signed release packages ready; packages: <path>
PASS test-packages: all final package tests passed; report: <path>
PASS publish: <public GitHub release URL>
FAIL test-packages: linux-x86_64: <reason>; log: <path>; retry: ./release/release.sh test-packages
```

Exit zero only for success and nonzero for failure or cancellation (130 for
Ctrl+C). Print progress and the failing command while running. A successful
subtask, existing file or stale report cannot produce an overall `PASS`. A
`--dry-run` may print `PLAN`, but must never claim a build, test or release passed.

## Implementation milestone

- [x] REL001 Add the five named commands to `release/release.sh`, using a shared
  driver, consistent help, progress, exit codes and final PASS/FAIL summaries.
- [x] REL002 Consolidate signing setup and environment configuration. Import
  existing machine/signing settings, discover tools/bundles, and save one usable
  profile without manual JSON copying, checksum entry or environment sourcing.
- [x] REL003 Implement `build-local` with automatic source snapshots, native
  host selection, SSH execution and dependency provisioning/reuse. Keep normal
  development builds working and keep private configuration out of artifacts.
- [x] REL004 Implement `test-local` against the recorded local builds, including
  the required host/current-feature suites and unchanged full CUDA gate. Save
  results automatically and reject missing, mismatched or failed inputs.
- [x] REL005 Implement `build-packages` around the existing packagers, Apple
  signing/notarization, Linux collection, source materials and signed checksums.
  All intermediate steps and retry directories stay internal.
- [x] REL006 Implement `test-packages` against the final signed/distributable
  bytes. Replace manual report gates with automated checks and documented test
  scope; keep optional fresh-machine research out of the release path.
- [x] REL007 Implement `publish` as upload, download/verification and public
  release in one invocation. Generate complete notes, create the exact source
  tag, preserve tested bytes and support safe retries without a second command.
- [x] REL008 Test the whole sequence on the Mac and PRO 5000, including first
  setup, warm dependency reuse, failures, cancellation, stale results and partial
  publication. Verify GitHub behavior with a disposable test repository. Update
  the release/build/dependency guides to this workflow and retire numbered/manual
  runbooks from the normal path. Accept only when the seven manual actions above
  suffice and all five stage commands report accurate final PASS/FAIL results.

## Research informing the plan

Reviewed upstream project files on 2026-10-27. These are examples of automation
patterns; the five-stage local interface above is the choice for h3cli.

| Project and primary source | Relevant pattern to adopt |
| --- | --- |
| [fzf release configuration](https://github.com/junegunn/fzf/blob/master/.goreleaser.yml) | One configuration describes platform builds, archives, GitHub releases and macOS signing/notarization. Keep credentials in setup and signing inside packaging. |
| [ripgrep release workflow](https://github.com/BurntSushi/ripgrep/blob/master/.github/workflows/release.yml) | Version validation, platform builds, archives, checksums and uploads are scripted. Hide these mechanics behind explicit maintainer commands. |
| [yt-dlp release workflow](https://github.com/yt-dlp/yt-dlp/blob/master/.github/workflows/release.yml) | Preparation, reusable platform builds, artifact collection, generated notes and GitHub publication form an automated sequence. Avoid manual handoff and note-template completion. |
| [whisper.cpp release workflow](https://github.com/ggml-org/whisper.cpp/blob/master/.github/workflows/make-release.yml) | One dispatch selects a commit, runs release checks and automates tag/description/release creation, with a dry-run path. Keep source selection and release mechanics explicit but automated. |

Use these patterns with the existing Python/shell builders and local GPU hosts;
this milestone does not require adopting GoReleaser or moving GPU tests to hosted
CI. Keep the implementation small enough for one maintainer to understand.

## Previous work

The [previous checklist](release-tooling-history.md) is archived intact:
dependency bundles remain **12/12 complete**, with
[qualification evidence](build/dependency-bundles-results.md). The macOS package
milestone remains **33/36 complete**, with
[recorded evidence and limits](build/macos-distribution-results.md).
MAC029/MAC030 were not executed, and MAC036's old closure criteria remain unmet;
none is relabeled as passed. Their former manual publication requirements are
superseded by this milestone's automated checks and stated support scope.
