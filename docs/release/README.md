# Making an h3cli release

Run these commands from the project checkout. The release driver snapshots the
selected committed source, remembers machine settings, and handles native Mac
and Linux builds, tests, signing, transfers and GitHub publication. Private
configuration, attempts and logs stay under ignored `outputs/release/`.

## One-time signing setup

```sh
./release/release.sh setup-signing
```

This works before a release configuration exists. It reuses GitHub login, the
Minisign key and the Mac's Developer ID Application identity and Keychain notary
profile. It never rotates an existing key. If needed it starts GitHub login,
Minisign key creation and `notarytool store-credentials`. A Developer ID
Application certificate and its private key must be installed through Keychain
Access or Xcode's account/certificate settings; the command explains missing
Apple prerequisites. An Apple Development certificate is insufficient.

An encrypted Minisign key is unlocked here once and its password saved in the
system credential store (macOS Keychain, or libsecret on Linux). Passwords are
never written into the profile or logs. A headless Linux environment can use an
existing unencrypted Minisign key stored outside the checkout. The public key is
included automatically in packages; no public-key commit is a release step.
Use `setup-signing --help` to select an identity, key or notary profile when
several exist. Initial access prompts belong to this setup step.

## Configure the environment

```sh
./release/release.sh configure --version v0.6.0
```

Existing `outputs/release/machines.json` settings are imported automatically.
The ignored `outputs/release/profile.json` remembers host/model/cache paths and
credential names. Subsequent configuration normally changes only the version.
The default label is **prerelease**; choose `--channel stable` here for a stable
release. The same automated checks apply to both labels.

Configuration verifies GitHub write access, signing, notarization and selected
hosts, installs missing host helpers and provisions missing locked model files.
It retains existing model files of the expected size; it does not hash all model
weights on every release. Network/provisioning failures are failures, not skipped
prerequisites. `--offline` requires already installed models and cached bundles.
Operating-system tool installation may request administrator access during setup.

For a new two-machine environment, supply paths once:

```sh
./release/release.sh configure --version v0.6.0 --targets macos,linux \
  --repo OWNER/REPO --macos-models /path/to/models \
  --linux-host cuda-test --linux-root /path/on/reflink-disk/releases \
  --linux-models /path/to/models --linux-model /path/to/models/MiniMaxH3
```

Configuration chooses Linux test scratch on a volume supporting file cloning
(for example XFS with reflinks or Btrfs), trying the work, dependency-cache and
model volumes automatically. It fails early if none supports the required LoRA
tests. The Mac filesystem must support clones as well.
Choose `--targets macos` or `--targets linux` for one platform. Linux can
be local when starting on Linux, or remote when starting on the Mac. A selected
unreachable host fails the command; it is never silently omitted.

`--commit REVISION` selects an existing commit (`HEAD` by default). Configuration
prints uncommitted differences and explicitly excludes them. Snapshots and
source transfer are automatic; no separate clone, run name or tag preparation is
needed. The selected commit must contain this release tooling. Use `--notes FILE`
only if custom notes are wanted; otherwise notes come from commit subjects and
actual test results. `--profile PATH` supports a separate test release environment.

## The five release commands

```sh
./release/release.sh build-local
./release/release.sh test-local
./release/release.sh build-packages
./release/release.sh test-packages
./release/release.sh publish
```

| Command | Result |
| --- | --- |
| `build-local` | Compiles the native application and test helpers on every selected host. Uses compatible qualified dependency bundles; obtains or builds and qualifies missing bundles once. Prints binary paths. |
| `test-local` | Tests those recorded binaries, host/API and release-tool suites, real-model rendering/state/server features and the full 204-output CUDA gate plus current CUDA suite on Linux. Fixtures and continuation states are prepared automatically. |
| `build-packages` | Builds corresponding standalone packages/sources, signs and notarizes the Mac inner runtime and DMG, staples the DMG, collects Linux assets and signs the complete checksum list. Prints the package directory. |
| `test-packages` | Tests exact final executable bytes outside the checkout with isolated caches, including rendering/server/media, the final Linux 204-output gate, launcher cache diagnostics, signatures/notarization, mounted DMG contents, sources and checksums. |
| `publish` | Generates notes, creates the exact source tag and an internal draft, uploads missing assets, downloads/verifies them, runs the downloaded Linux binary on the GPU host, publishes and verifies the resulting release. Prints its GitHub URL. |

Every stage ends with `PASS <command>: ...` and exit zero, or `FAIL <command>:`
with the reason, log and retry command. Ctrl+C exits 130 after stopping owned
workers, including remote workers; SSH disconnects also stop the remote work.
`--dry-run` is read-only and prints `PLAN`, never `PASS`.

Run the same command to retry. Completed work is reused only after its inputs
and recorded output hashes verify. Failed attempts and logs are retained, and
fresh attempt directories are allocated automatically. Changing source/config
selects a new run; changing an artifact invalidates downstream results. Tests do
not rebuild the binaries they test, and publication never rebuilds or re-signs.

Apple uploads occur only in `build-packages`; stored submission IDs allow retries
to wait for the same notarization. GitHub tags/assets change only in `publish`.
An interrupted draft upload resumes matching assets; an identical published
release can be verified again. Conflicting tags, notes, channels or asset bytes
are never overwritten. No manual collect, checksum, report, note-template,
approval-file or final browser action is required.

Retries remove only empty failed-upload placeholders in a matching draft, as
documented by [GitHub's release asset API](https://docs.github.com/en/rest/releases/assets#upload-a-release-asset),
then upload those missing assets again. Completed assets must match the tested bytes.

## Test scope

The required automated tests must actually pass. Missing dependencies, changed
artifacts, failed signatures and numerical mismatches block publication. Notes
name the tested systems and exact source commit. Raw logs contain local paths
and remain private; follow the [publication notes](../publishing.md).

Fresh Mac installations/accounts, browser quarantine experiments, broader OS/GPU
matrices and human visual review are optional investigations. Existing developer
accounts and warmed trust stores are not described as fresh environments. Apple
cannot staple a ticket to a bare CLI; the DMG is the stapled distribution. This
experimental project's automated acceptance makes no fresh-machine trust claim.

## Maintainer internals

[Dependency bundle internals](../build/dependency-bundles.md),
[Mac packaging](../build/macos-distribution.md) and
[Linux packaging](../build/linux-distribution.md) describe the lower-level tools.
Their standalone commands are useful for development, not additional release
steps. The old numbered wrappers, `driver.py` and `coordinator.py` remain for
historical diagnostics; they are retired from the normal release path. Their
manual stable-release receipts are not gates in the five-command workflow.
