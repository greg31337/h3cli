# Build and sign the Mac download

Use the clean clone from the [release guide](README.md) on Apple Silicon.
The runtime minimum is macOS 26. The build must use the exact installed Apple
tools in [scripts/macos/lock.json](../../scripts/macos/lock.json).

## Configure once

```sh
./release/macos/step1-prerequisites.sh --install
./release/configure.sh --repo OWNER/REPO --tag v0.1.0 \
  --commit REPLACE_WITH_FULL_COMMIT_ID --models-path /absolute/path/to/models \
  --apple-identity 'Developer ID Application: Example Name (TEAMID)' \
  --linux-host h3cli-linux-builder --linux-root /absolute/remote/path/to/h3cli-release
```

Use the same commit, version and run as Linux. The SSH alias must already select
the maintainer-provided host, login, port and key. The remote path must be absolute
and contain no spaces or shell characters. Alternatively, use
`--linux-handoff /absolute/path/to/copied/linux-handoff-RUN` for a local copy.
Do not copy the Linux configuration onto the Mac; its local paths differ.

## Apple tools and credentials

Step 1 runs the packaging doctor's exact SDK/tool checks. The current lock
requires SDK 27.0 build `26A425`, Clang 21.0.0 (`clang-2100.3.34.2`), linker
27037.1 and `pkg-config` 3.0.7. The newer SDK does not change the macOS 26 runtime
minimum. Install the matching tools from
[Apple Developer Downloads](https://developer.apple.com/download/all/).
`DEVELOPER_DIR` can select an existing Xcode installation; the scripts do not
change the global Xcode selection. Do not edit the lock merely to bypass a failure.

Use a **Developer ID Application** certificate with its private key available
in Keychain. `Apple Development` is not the public distribution identity this
packager requires. See [Apple's certificate setup](https://developer.apple.com/help/account/certificates/create-developer-id-certificates/).
Certificate creation/installation is the team's responsibility and cannot be
completed by these scripts without the team's account access.

The helper checks the identity list, notarization profile and GitHub login:

```sh
./release/macos/setup-signing.sh
```

If the notary profile has not been stored, use the interactive setup once:

```sh
./release/macos/setup-signing.sh --store-credentials \
  --apple-id release-owner@example.com --team-id TEAMID --github-login
```

The app-specific password is prompted locally and stored in Keychain. The
default profile name is `h3cli`; configure `--notary-profile` to use another.
A working profile does not create a signing certificate. The profile and
identity must belong to the same team. Reuse the working profile for later
releases. [Apple's notarization workflow](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow)
explains the account-side requirements.

## Run the build, signing and local tests

```sh
./release/macos/run.sh --install
```

With Homebrew already installed, `--install` supplies Python 3.10+ if needed and
missing `pkg-config`, `gh` and `minisign` helpers. Step 1 works before configuration
or Python. It does not upgrade an installed locked tool or install Xcode. Install
Git/Apple tools first if needed to obtain the checkout. Omit `--install` on a
prepared Mac.

| Step | Script | What it does |
| --- | --- | --- |
| 1 | [step1-prerequisites.sh](../../release/macos/step1-prerequisites.sh) | Checks tools and runs the SDK doctor |
| 2 | [step2-build.sh](../../release/macos/step2-build.sh) | Fetches inputs, builds offline, verifies sources/checksums and runs package fixture tests |
| 3 | [step3-sign.sh](../../release/macos/step3-sign.sh) | Signs inner code, notarizes it, signs the launcher/DMG, notarizes and staples the DMG |
| 4 | [step4-test.sh](../../release/macos/step4-test.sh) | Checks signatures/team/notarization, provisions models and runs the small feature/server matrix |

The development build is `bin/macos-dev-RUN/`. The public candidate is
`bin/macos-signed-RUN/`. Step 4 tests the signed executable; signing changes
bytes, so a test of the development binary is insufficient.

Keychain may prompt locally for permission to use the signing key. Apple
submissions can take several minutes. The signing script saves submission logs
and fails on rejection. It never falls back to ad-hoc signing. Failed signing
outputs are preserved; rerunning step 3 starts a fresh signing attempt, as
explained in the main guide.

Step 4 uses mostly 256×256, two-step clips, plus 512×512 upscale cases. It tests
references, stills, continuation, bridge, saved state, upscale and the server.
Its server download case requires network access. Watch the generated videos
using the run's `features/review.html` and retain the test results.

For a separately configured native development environment, run:

```sh
./release/macos/test-source.sh --environment outputs/setup/macos-env.sh
```

This wraps `make all test test-server-http test-macos-package`. It supplements
the signed-package tests. Use a trusted setup file and follow
[CONTRIBUTING.md](../../CONTRIBUTING.md) for required source coverage.

## 7. Check a clean Mac and a fresh download

After [the draft upload](github.md), use a browser to download the DMG on a clean
Mac/installation. The tester needs repository access to see a draft. Preserve
quarantine information. Open the DMG, copy the executable to a writable folder,
and eject the DMG. Do not disable Gatekeeper or remove quarantine attributes.

Copy [check-clean-machine.sh](../../release/macos/check-clean-machine.sh) to that
machine too. It uses only macOS system tools, not Python, Homebrew or Xcode:

```sh
bash check-clean-machine.sh \
  --binary /absolute/path/to/h3cli-macos-arm64 \
  --models-path /absolute/path/to/models --output /absolute/path/to/fresh-check
```

The script runs `--help`, an offline 256×256/56-frame/two-step render, and a
queued server render with status polling and video download. It leaves a log
and two MP4s to review, and stops its own server. Add `--model` for a custom main
model directory or `--port` if the default local port is in use.

A clean account helps prove runtime independence; it does not reset machine-wide
trust. For a separate cold offline trust test, first download the DMG and
provision models, then disconnect networking **before** first opening or
assessing the DMG. A previously accepted online launch is not an offline
first-launch test. Follow [Apple's fresh-machine procedure](https://developer.apple.com/forums/thread/130560).

The automated signature checks use the DMG's `spctl --type open` assessment and
`codesign`'s notarized requirement for the bare CLI. An app-bundle-only execute
assessment is not substituted. The stapled DMG is the recommended download; a
bare CLI cannot itself have a ticket stapled to it.

Record actual results with [record-check.sh](../../release/record-check.sh),
following the GitHub guide. The clean-machine helper cannot certify that your
machine had fresh trust; write those conditions in the report yourself.
Current clean-environment and fresh-trust checks remain pending in the
[Mac report](../build/macos-distribution-results.md).

Continue with [GitHub publication](github.md). The source archive, signed DMG and
signed standalone executable are selected automatically; private probes and
notarization work files stay in the build record.
