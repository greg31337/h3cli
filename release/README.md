# Release scripts

Start with [the release guide](../docs/release/README.md). These are executable
entry points for all build, signing and publication commands in that guide.
Python 3.10+ and Git are required for the main runner. Each platform's first
step is a shell bootstrap that can install Python before configuration. The
Mac clean-machine helper is standalone Bash and uses only macOS system tools.

For a single Mac command coordinating both machines, copy
`machines.example.json` to ignored `outputs/release/machines.json`, fill in the
machine paths and credentials, then run `./release/release.sh --version v0.2.0`.
See [one-command setup](../docs/release/README.md#one-command-from-the-mac).
It creates isolated checkouts, runs Linux over SSH, generates release notes,
and finishes with a verified GitHub draft. `--dry-run` prints the sequence.

1. `clone.sh` creates a fresh checkout of an explicit reviewed commit.
2. `configure.sh` saves repository, version, model and machine paths once.
3. `linux/run.sh` performs Linux steps 1–5; `macos/run.sh` performs Mac steps 1–4.
4. `macos/run.sh --through 8` collects, signs and uploads a draft, then verifies it.
5. `macos/step9-publish.sh --channel stable` publishes only after all required
   evidence is recorded. `--channel prerelease` permits explicitly documented
   pending clean-machine checks.

Every numbered script accepts `--help`, `--dry-run` and `--config PATH`. Commands
use the saved configuration from `outputs/release/config.json` by default and
do not depend on the calling directory. `status.sh` lists saved step records.
`record-check.sh` attaches source-test and clean-machine evidence to the staged files.
`verify-downloads.sh` verifies a complete download using an independently trusted
public key. `macos/setup-signing.sh` performs the one-time interactive credential
and Minisign setup. Each platform's `test-source.sh` wraps the native source suite
when its development environment is already installed.

The shell files share `driver.py` so checksum lists, identities and failure
handling cannot diverge between steps. It invokes the existing packagers and
regression runners; it does not change inference or the recorded golden state.
No runner automatically publishes or bypasses missing qualification.
Human video review is not a publication requirement.

Test the automation without a GPU, network or real signing credentials:

```sh
python3 -m unittest discover -s release/tests -v
```

These tests use temporary files and mocked external services. They do not
constitute physical release qualification or submit anything to GitHub/Apple.
