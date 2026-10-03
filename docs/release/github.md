# Sign the downloads and publish on GitHub

Run these steps on the signing Mac after [Linux](linux.md) and [Mac](macos.md)
qualification. Configuration comes from `release/configure.sh`; there is no
need to export paths or hand-copy checksum commands.

## One-time release key setup

The publisher needs [GitHub CLI](https://cli.github.com/) and
[Minisign](https://jedisct1.github.io/minisign/). Mac step 1 installs missing
helpers when called with `--install`. The Apple identity is a separate key from
the Minisign key used for the public checksum list.

For the first release, do this in a development checkout before freezing the
release commit. Configure it using the Mac guide, then run:

```sh
./release/macos/step1-prerequisites.sh --install
./release/macos/setup-signing.sh --github-login --generate-key
```

`--github-login` opens GitHub's browser login and configures Git's credential
helper. `--generate-key` creates a password-protected Minisign key at the configured
`--secret-key` path (default `~/.config/h3cli-release/h3cli-release.key`) and copies
only its public key to `docs/release/h3cli-release.pub`. It refuses to overwrite
existing keys or keep the secret key inside the checkout.

Keep the secret key and password private and make a secure backup. Review and
commit the public key, publish it on a trusted project page, then make the final
clean release clones/configurations. Reuse the key for later releases; omit
`--generate-key` once it exists. If the team already has a key, install its
reviewed public key through the normal review process and configure the existing
secret-key path. Never create a new key simply to bypass a verification failure.

The helper also checks Apple identities, the notary profile and GitHub access.
For first-time Apple setup, use [the Mac instructions](macos.md#apple-tools-and-credentials).
Passwords are entered locally, never put in the config or command arguments.

## 1. Collect the files on the signing Mac

```sh
./release/macos/step5-collect.sh
```

This pulls the six files from Linux's `bin/linux-handoff-RUN/` using the configured
SSH alias, or reads `--linux-handoff` if a local handoff was configured. It checks
the exact source commit, a clean build, public-file hashes, passing Linux reports,
all 204 parity outputs and the four distro gates. A failed transfer stays in its
own attempt directory, so rerunning can start a fresh transfer.

The handoff's `build.json` and `qualification.json` stay private. A binary without
its matching successful qualification record is rejected. Detailed original logs
and videos remain on the build machine for review.

## 2. Make and sign the final checksum list

```sh
./release/macos/step6-stage.sh
```

This verifies both platforms' completed tests, compares the Mac source inventory
with the clean checkout, and verifies the signed Mac build's link to its input.
It copies only the public files to `bin/publish-RUN/`, writes `release-info.txt`
and `SHA256SUMS`, signs the checksum list with Minisign, then verifies both the
signature and all file hashes. Enter the signing-key password at the local prompt.

| Public asset | Purpose |
| --- | --- |
| `h3cli-linux-x86_64` | Linux executable |
| `h3cli-linux-x86_64.sha256` | Individual executable checksum |
| `h3cli-linux-x86_64-sources.tar.gz` | Matching Linux application/dependency sources and recipes |
| `h3cli-linux-x86_64-sources.tar.gz.sha256` | Individual source checksum |
| `h3cli-macos-arm64.dmg` | Recommended signed, notarized, stapled Mac download |
| `h3cli-macos-arm64` | Signed standalone Mac alternative |
| `h3cli-macos-arm64-sources.tar.gz` | Matching Mac application/dependency sources and recipes |
| `release-info.txt` | Repository, version and exact source commit |
| `h3cli-release.pub` | Convenience copy of the public key |
| `SHA256SUMS` | Hashes of the nine files above |
| `SHA256SUMS.minisig` | Detached publisher signature over the list |

Unexpected files and files at or above GitHub's 2 GiB per-asset limit are rejected.
No wildcard upload is used. Private probes, historical regression codecs, model
weights, keys, notarization submissions, build records and `libh3.a` are not in
this list. The source archives are required: GitHub's automatic repository ZIP
lacks the dependency sources. See [Linux sources](../../scripts/linux/SOURCE-MATERIALS.md)
and [Mac sources](../build/macos-distribution.md#corresponding-sources-and-relinking).

Step 6 also creates `outputs/release/RUN/release-notes.md` from the template,
filling the version and commit. Edit the remaining `REPLACE_...` fields. The
next step rejects unfinished placeholders. Describe actual results and pending
checks. Do not change the staged files after signing.

## 3. Create the tag and draft release

```sh
./release/macos/step7-draft.sh
```

This is the first step that writes to GitHub. It checks `origin`, creates the
version tag at the configured commit if needed, pushes only that tag, creates
an unpublished draft and uploads the eleven named files. Existing tags must
point to the same commit. Existing public releases are never modified.

Rerunning a partial draft upload downloads and checks existing asset bytes,
then uploads only missing files. Different bytes or unexpected assets stop the
step; it never uses `--clobber`. Draft notes can be updated by editing the notes
file and rerunning this step. See the official
[create](https://cli.github.com/manual/gh_release_create) and
[upload](https://cli.github.com/manual/gh_release_upload) commands.

## 4. Download and verify the draft

```sh
./release/macos/step8-verify.sh
```

The script checks the remote tag, downloads into a fresh directory, checks the
exact asset list, verifies the Minisign signature using the key in the reviewed
checkout, checks every hash against the tested staging files, and checks the
Mac signature/team, notarization and stapled DMG. It saves a completion record
with the downloaded file hashes. It never trusts a key just because that key
was downloaded beside the signature.

To automate steps 1–8 after configuration and notes preparation, use:

```sh
./release/macos/run.sh --through 8
```

Successful earlier steps are verified and reused. On a first run it will stop
at the draft step if the notes still contain placeholders. Edit them, then rerun.
This command uploads a draft but cannot publish it.

A general verification helper is available on either build machine for a folder
containing all eleven assets:

```sh
./release/verify-downloads.sh --directory /absolute/path/to/downloaded \
  --public-key /absolute/path/to/trusted/h3cli-release.pub
```

It verifies the signature and all nine file checksums, and fails on missing or
changed files. It needs Python 3.10+ and Minisign as verification tools; neither
is a requirement for running the released h3cli executable.

## Record the remaining real-world checks

Complete the [clean-Mac procedure](macos.md#7-check-a-clean-mac-and-a-fresh-download)
using a browser download from the draft. `gh`/`scp` transfers do not prove
browser quarantine behavior. Confirm a downloaded Linux executable runs on
PRO 5000, and review the sample videos. Keep reports with machine/OS details,
commands, results and the artifact checksum.

Attach each actual report with this helper; change the check ID and report path
for each completed check:

```sh
./release/record-check.sh --check source-tests --report /absolute/path/to/source-test-report.txt
./release/record-check.sh --check visual-review --report /absolute/path/to/video-review.txt
./release/record-check.sh --check macos-clean-runtime --report /absolute/path/to/clean-runtime.txt
./release/record-check.sh --check macos-fresh-online --report /absolute/path/to/fresh-online.txt
./release/record-check.sh --check macos-fresh-offline --report /absolute/path/to/fresh-offline.txt
./release/record-check.sh --check linux-downloaded --report /absolute/path/to/linux-download.txt
```

Each record is tied to the staged files and names the reporting user. These are
human-reported checks, not automatic claims that a test passed. Supply existing
results; do not create an empty “passed” note to get through a gate. The script
rejects empty reports and refuses to overwrite existing recorded evidence.

## 5. Publish deliberately

For a fully qualified stable release:

```sh
./release/macos/step9-publish.sh --channel stable
```

This requires all six reported checks, checks the completed automation records,
re-downloads and verifies the draft again, updates its notes and makes it public
as the latest stable release. It never rebuilds or re-signs the approved files.

For an early preview, use this **instead**:

```sh
./release/macos/step9-publish.sh --channel prerelease
```

A prerelease still requires both platforms' automated tests plus `source-tests`
and `visual-review`. Its notes must explicitly name every missing check ID,
for example `macos-clean-runtime`, `macos-fresh-online`, `macos-fresh-offline`
or `linux-downloaded`, with an explanation that it is pending. The script marks
the release as a prerelease and does not mark it latest. The currently deferred
clean-Mac checks remain deferred until real reports are supplied.

Publication is never included in `run.sh` and needs an explicit `--channel`.
[GitHub's publication command](https://cli.github.com/manual/gh_release_edit)
is called only after verification succeeds. Check the public page and retain
the exact artifacts and private build/test/signing records in the team's archive.
For a published binary correction, make a new version; do not replace bytes under
an existing version.
