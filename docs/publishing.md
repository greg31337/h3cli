# Keeping private information out of published files

Use repository-relative paths or `/path/to/...` examples in documentation.
`cuda-test` and `cuda-pro6000` are example SSH aliases, not public hostnames.
Configure the aliases in your own `~/.ssh/config`; keep the real hostname, user,
port and key location there. Hardware models, software versions and artifact
checksums can remain in public qualification reports.

Historical reports in this repository have had personal paths, server addresses,
GPU UUIDs, Apple team/certificate identifiers and notary submission IDs removed
or normalized. Recorded artifact checksums still refer to the original test
artifacts. They are not checksums of the sanitized reports. The SGLang golden
manifest and input fixtures are unchanged.

Keep credentials and raw machine records under ignored `private/` or `outputs/`,
or outside the checkout. The ignore rules also cover common environment, SSH
and signing-key files; ignore rules do not protect files already tracked by Git.
Review `git diff` and `git status` before committing. Do not force-add raw run
logs, release configuration, Apple account records or private signing keys.
Retain third-party license notices and their upstream author attribution.

## Configuring historical test runners

Run from the repository root. The affected runners use `H3_MODEL_DIR`, defaulting
to `models/MiniMax-H3`; a supported `--model` argument takes precedence. Point it
at your installed model if it uses another location, such as `models/MiniMaxH3`.

- `H3_TEST_QUANT_CACHE` selects the quantization cache. Defaults are under
  `outputs/quant-5090/packed` or `outputs/cuda-sol/packed` for the respective runners.
- `H3_TEST_REFERENCE_ROOT` selects the checkout containing the original
  multi-reference evidence needed by the SOL campaign; it defaults to the current
  directory. Existing fixture checksums and model-identity checks still apply.
- `H3_TEST_IMAGE_VAE_DIR` selects the image-VAE inventory directory, defaulting
  to `models/image-vae`.

The SOL campaign records device zero's GPU UUID in its private frozen identity
and rejects a different GPU or model location on subsequent runs. The public
manifest contains no device serial. Multi-reference preparation and reports
write fresh manifests only under ignored `outputs/`.

## Repository history and release files

Editing the current files does not erase older commits, Git author metadata,
existing clones or previously built source archives. If publishing for the first
time and the old history must remain private, create a separate clean repository
from the sanitized tracked files. Otherwise, review history before pushing it.
Rebuild release source archives from the sanitized revision before uploading.
Actual signed Apple binaries necessarily expose their signing certificate and
team identity; never include private keys or account credentials alongside them.
