# Making an h3cli release

Use the numbered scripts in [release/](../../release/README.md). They turn the
Linux and Mac runbooks into a repeatable process. Configure each machine once;
the scripts remember its paths and keep logs and results under `outputs/`.
No shell variables need to be carried between Terminal windows.

## Recommended path

Use the [one-command coordinator](#one-command-from-the-mac) for a complete draft,
or the numbered scripts below for individual steps.

Build and test Linux on the qualified PRO 5000, and build, sign and test Mac on
Apple Silicon. Use the same reviewed commit, version and run name on both.
The signing Mac collects the Linux result and prepares the GitHub release.

```text
One clean commit
    ├── Linux run.sh → build → models → CUDA tests → private handoff
    └── Mac run.sh   → build → Apple signing → models/render tests
                              ↓
                   collect → stage/sign checksums → draft → verify
                              ↓
                   clean-machine checks → explicit publish step
```

`release/linux/run.sh` runs Linux steps 1–5. `release/macos/run.sh` runs Mac
steps 1–4. `release/macos/run.sh --through 8` continues through draft upload and
download verification. The combined runner **never publishes**; publication is
always the separate `step9-publish.sh` command.

| Guide | What it covers |
| --- | --- |
| [Linux](linux.md) | Prerequisites, portable build, all 204 parity checks, handoff |
| [Mac](macos.md) | Locked Apple tools, signing, notarization, small renders, clean Mac |
| [GitHub](github.md) | Linux collection, signed checksums, draft, verification, publication |
| [Release notes template](release-notes-template.md) | Notes copied automatically into the private run directory |

Ordinary `make` remains the development build. These scripts call the existing
portable packaging recipes in `scripts/linux/` and `scripts/macos/`.

## One command from the Mac

After one-time machine/signing setup, start a release from a clean, committed
checkout on Apple Silicon:

```sh
./release/release.sh --version v0.2.0
```

Copy `release/machines.example.json` to `outputs/release/machines.json` and set:

| Setting | Value |
| --- | --- |
| `macos.models` | Absolute local model-root path |
| `macos.model` (optional) | Main model override; otherwise detects `MiniMax-H3` or uses `MiniMaxH3` |
| `macos.apple_identity` | Developer ID Application identity; empty selects the only valid identity |
| `macos.notary_profile` | Existing Keychain notarization profile, normally `h3cli` |
| `macos.secret_key` | Existing Minisign secret-key path outside the checkout |
| `linux.host` | SSH alias or `user@host` for the RTX PRO 5000 |
| `linux.identity_file` | Optional SSH key path on the Mac, also used for downloads |
| `linux.root` | Parent directory for new isolated release candidates on Linux |
| `linux.models` | Existing Linux model root; missing supported model groups may be downloaded |
| `linux.model` (optional) | Main model override |
| `jobs` in either machine | Build jobs, 1–32, default 8 |

Keep this settings file ignored: it contains private paths and connection details.
The SSH key and signing keys are never uploaded or copied to Linux. Linux paths
must be absolute and contain no spaces or shell characters. SSH must already work
without a password prompt and the host key must already be trusted.

Prepare the tools listed below on each machine, including `skopeo` and `umoci`
on Linux and `gh`/`minisign` on the Mac. Use the numbered prerequisite scripts
with `--install` once if necessary. Complete
[signing setup](github.md#one-time-release-key-setup), log in to GitHub, and commit
the public key at `docs/release/h3cli-release.pub` along with the coordinator.
The remote machine needs no GitHub login: source is transferred as a Git bundle
through the configured SSH connection. GitHub tag/release writes happen on the Mac.

The coordinator pins `HEAD` (or `--commit REVISION`) on both machines. It builds
in `outputs/release/candidates/RUN/source` on the Mac and `linux.root/RUN/source`
on Linux; existing development checkouts are not modified. It verifies Mac tools
and signing access before starting the Linux build. The Linux runner builds,
provisions models, runs all qualification tests and exports the handoff. The Mac
then builds, signs/notarizes, tests, collects Linux, signs the checksums, generates
notes, uploads a draft and verifies the downloaded assets. The final URL is
printed only after verification succeeds. It never calls the publication step.

```sh
./release/release.sh --version v0.2.0 --dry-run
./release/release.sh --version v0.2.0 --notes /path/to/change-description.txt
./release/release.sh --version v0.2.0 --run v0.2.0-02
```

`--settings PATH` selects another machine profile. Dry run reads local settings
and Git only; it performs no writes or network calls. `--notes` supplies an
optional change description; automated results and pending source/clean-machine
checks are appended from verified records. Without it the notes identify the
source commit. Notes are regenerated on each coordinator run. Signing may still
prompt locally for Keychain access or the Minisign password.

Rerun the same command to reuse hash-verified completed work and finish a partial
upload. A changed commit or machine profile requires a new `--run`. Failed partial
builds are preserved and also require a new run. Command logs and receipts live
under `outputs/release/RUN/` inside each candidate checkout. After a draft is
complete, run individual check-recording/publication commands from the Mac's
candidate checkout. Stable publication still requires the five reports below;
human video review is not a release gate.

## What you need

- Git, Python 3.10+, the GitHub repository name (`OWNER/REPO`), an agreed full
  commit ID and version such as `v0.1.0`.
- Linux x86-64 for compilation; the qualified RTX PRO 5000 and installed models
  for exact CUDA tests. Get connection details from the maintainer.
- An Apple Silicon Mac, the exact Apple tools in
  [the lock file](../../scripts/macos/lock.json), a Developer ID Application
  identity, and a working Keychain notarization profile.
- GitHub write access and a maintainer-held Minisign key for signing the release
  checksum list. See [one-time signing setup](github.md#one-time-release-key-setup).
- A clean Mac/account for runtime independence, plus a clean Mac/installation
  for fresh-trust checks. These cannot be simulated by a completion file.
- Disk space for builds and [models](../features/model-downloads.md#offline-storage-and-recovery).
  The automated model step provisions all groups and can download hundreds of GB.

A checksum detects changed bytes. A signature proves that the holder of the
trusted publisher key approved those bytes. Apple's notarization is separate;
its ticket is stapled to the Mac DMG. Users verify the detached Minisign
signature and checksums for Linux downloads. The Mac DMG is also Developer ID
signed and notarized.

## Prepare the source on each machine

First review and merge the release scripts, release changes and public signing
key. From an existing checkout, create a fresh release clone:

```sh
./release/clone.sh --repo OWNER/REPO \
  --commit REPLACE_WITH_FULL_COMMIT_ID --destination ../h3cli-release
cd ../h3cli-release
```

If you do not yet have a checkout, obtain one using GitHub's **Code → Clone**
instructions, then use the helper above. It refuses to overwrite a destination
and checks out the exact commit without following a moving branch.

Run the platform's `step1-prerequisites.sh --install` to bootstrap missing
helpers, including Python, before configuring. The prerequisite step needs no
configuration and does not change the Apple toolchain. Configure each clone
with [the Linux](linux.md) or [Mac](macos.md) example.
Configuration is saved in `outputs/release/config.json`, which is gitignored.
It contains paths and identity names, not passwords or private-key contents.
The default run name is `VERSION-01`; use the same `--run` on both platforms.
Local model paths can differ. An existing `MiniMax-H3` directory is detected;
`--model` overrides the main directory explicitly.

## Running, previewing and retrying

Every numbered script accepts `--help`, `--config PATH` and `--dry-run`.
Dry run prints commands without building, downloading, signing or uploading.
Steps after prerequisites need configuration first. Except for the standalone clean-Mac helper,
scripts locate the checkout from their own location and work from any cwd.

```sh
./release/linux/run.sh --dry-run
./release/macos/run.sh --through 8 --dry-run
./release/status.sh
```

Completed builds and tests are reused only after their recorded hashes and
configuration are checked. Successful steps have JSON completion records under
`outputs/release/RUN/`; command logs are in its `logs/` folder. A nonzero exit
stops the pipeline. Failed tests get a fresh attempt directory on retry; an
interrupted Linux transfer also gets a fresh attempt. Draft uploads compare
existing remote bytes and upload only missing assets.

The underlying builders require fresh outputs. If a build fails after creating
output, keep it for diagnosis and configure a new `--run` and `--config` for a
fresh attempt. Both platforms must ultimately use the same run name. Apple and
Minisign signing retries use fresh attempt directories and publish their final
local folder only after successful verification. Inspect a failed Apple
submission's logs before retrying. Existing outputs are never deleted to force
a retry. Reuse the verified dependency download cache.

Do not edit the configured checkout during release work. Build and publication
steps require its exact commit and an empty `git status --porcelain`. Store notes
and logs in ignored `outputs/` or `bin/`. Changing an existing configuration is
rejected; use a new configuration and run instead of silently changing a candidate.

## Before calling a release ready

The scripts check build identities, test reports, checksums, accepted Apple
submissions, the remote tag and the exact public asset list. Record actual source
and clean-machine evidence using [record-check.sh](../../release/record-check.sh).
The scripts cannot establish that a Mac was actually clean. Human video review
is optional and is not recorded as a required publication check.

Stable publication requires these five reported checks:

| Check ID | Evidence to supply |
| --- | --- |
| `source-tests` | Required source tests for the release commit, under [CONTRIBUTING.md](../../CONTRIBUTING.md) |
| `macos-clean-runtime` | Execution without developer dependencies in a clean environment |
| `macos-fresh-online` | Browser-downloaded DMG on a Mac with fresh trust, online |
| `macos-fresh-offline` | Separate first launch offline, with models already provisioned |
| `linux-downloaded` | Downloaded executable starts on the qualified GPU host |

The [current Mac report](../build/macos-distribution-results.md) still has clean
environment and fresh-trust qualification pending. These scripts do not mark
those checks as passed. A maintainer can publish a prerelease with those gaps
stated explicitly; stable publication waits for their reports.
