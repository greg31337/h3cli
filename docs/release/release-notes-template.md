# h3cli REPLACE_VERSION

REPLACE_SHORT_DESCRIPTION_OF_CHANGES

## Downloads

- **Linux/NVIDIA:** `h3cli-linux-x86_64`. Requires Linux x86-64, glibc 2.35+,
  an AVX2/FMA CPU, and NVIDIA driver 580.126.20+ with support for your GPU.
- **Apple Silicon:** `h3cli-macos-arm64.dmg` is the recommended download.
  Requires macOS 26+. Copy the executable out of the DMG and eject it before
  use. The separate `h3cli-macos-arm64` is also available.
- Model weights are separate. Missing supported weights download automatically;
  use `--models-path /path/to/models` to choose their location or `--offline`
  to require installed weights. Allow space for models and the runtime cache.

Both packages include FFmpeg/FFprobe 9.0.2. They do not require a separate Python,
FFmpeg or developer-tool installation to run. Linux still needs the NVIDIA
driver; the driver is not included.

## Verify the downloads

The release includes `SHA256SUMS` and `SHA256SUMS.minisig`. Get the project's
trusted public signing key from REPLACE_TRUSTED_PUBLIC_KEY_LINK. Verify the
manifest signature with Minisign, then verify the downloaded files against it.
Do not trust a new signing key only because it appears among the download files.

The Mac executable and DMG are Developer ID signed; the DMG is notarized and
stapled. Signing team: REPLACE_TEAM_ID.

## What was tested

- Source commit: REPLACE_FULL_COMMIT_ID.
- Linux hardware, driver and tested userlands: REPLACE_LINUX_DETAILS.
- Complete recorded SGLang gate: REPLACE_ACTUAL_RESULT_AND_EVIDENCE.
- Current-feature rendering and server tests: REPLACE_ACTUAL_RESULTS.
- Mac hardware and OS: REPLACE_MAC_DETAILS.
- Clean-account runtime and fresh-trust online/offline checks:
  REPLACE_ACTUAL_RESULTS_OR_PENDING_CHECKS.
- Known limitations or deferred checks: REPLACE_LIMITATIONS.

Compiled GPU support is not the same as physical testing on every GPU. Describe
only the machines actually tested for this candidate. For a prerelease, clearly
state what is still pending before stable release acceptance.

## Source materials

The attached `h3cli-linux-x86_64-sources.tar.gz` and
`h3cli-macos-arm64-sources.tar.gz` contain matching application/dependency sources,
notices and build recipes. GitHub's automatic repository source ZIP/tarball does
not replace these companion archives. Component licenses remain in force,
including those for FFmpeg/x264 and NVIDIA libraries.
