# FFmpeg 9.0.2 and isolated regression codecs

2026-09-29 policy: use **FFmpeg/FFprobe 9.0.2 for all ordinary rendering,
reference input, media inspection, server jobs and current-feature tests**.
Keep the historical **4.2.2 output encoder and 6.1.1 input decoder/probe** only
for the immutable 204-output SGLang regression. No recorded golden is changed.

The final PRO 5000 package has passed all 204 outputs in each of three Linux
userlands and completed the source-material audit. See the
[current qualification report](linux-distribution-pro5000.md). The PRO 6000
experiment and migration measurements below remain historical evidence.

## Implementation

Both platform setup scripts build the checksum-pinned 9.0.2 upstream source into
`outputs/setup/ffmpeg`; their environment scripts put its `bin` first on PATH and
set `H3_FFMPEG`, `H3_SGLANG_INPUT_FFMPEG` and `H3_FFPROBE`. The Linux standalone
package includes just `tools/ffmpeg` and `tools/ffprobe`, both 9.0.2. Ordinary
packaged runs continue to ignore stale PATH/media overrides.

For source regression runs, install the private test tools with:

```sh
python3 scripts/setup_reference_media.py
make test-cuda-reference-regression
```

This writes `outputs/setup/reference-media/profile.json`. `H3_REFERENCE_MEDIA`
can select a relocated profile. The portable builder supplies the equivalent
profile under `linux-validation/media`, outside the standalone payload. It binds
that profile's hash into the validation manifest. The regression runner checks
all tool hashes before and after execution, selects the historical codecs for
all probes and the full CLI render, and uses 6.1.1 for final RGB/PCM decoding.
Missing tools fail instead of falling back to production or PATH FFmpeg.

The harness also sets the private `H3_TEST_REFERENCE_ENCODING=1` control to
retain the recorded CRF 25, AAC defaults, color conversion, thread count and
MP4 layout. CUDA arithmetic alone no longer selects historical compression.
Production uses CRF 18, explicit BT.709 conversion, AAC 192 kbit/s and faststart;
see [output encoding](../features/output-encoding.md). The test profile rejects
explicit output overrides, and the recorded golden manifest remains unchanged.

The packaged core recognizes the harness's `H3_TEST_REFERENCE_MEDIA` and
`H3_TEST_REFERENCE_MEDIA_SHA256` pair, verifies the profile and media files, then
sets only the three media selections. This is a private test control, not a CLI
mode. Input dependency identity follows the selected tool, preventing reuse of
conditioning prepared with different decoding. Current-feature validation
explicitly retains production media.

The old static encoder is no longer a production redistribution dependency.
The final release materials include the 9.0.2 source archive, build recipe and
matching transitive dependency sources and patches. The companion source archive
must accompany the executable when it is published; nothing has been uploaded.
The historical validation kit remains private qualification material.

## Experiment

The [official download page](https://ffmpeg.org/download.html) lists 9.0.2,
released 2026-09-18, as the latest stable version in the newest release branch.
The official source archive's SHA-256 is
`8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e`.
Its detached signature verified against the FFmpeg release key, fingerprint
`FCF986EA15E6E293A5644F10B4322F04D67658D8`, downloaded from FFmpeg's site.

This initial experiment used the authorized PRO 6000 node. Building in the existing
Ubuntu 22.04 development filesystem with GCC 11 and eight jobs took 105.524 s.
The build enabled GPL, libx264 and zlib, with automatic dependency detection
disabled. It used the builder's locked x264 library; the historical encoder
contains its own older static x264. These measurements compare the two complete
encoder builds, not an isolated change to FFmpeg alone.

Both encoders received identical fixed 640×480/124-frame RGB and 32 kHz stereo
PCM files. These inputs were previously decoded from a recorded render; this
is a codec comparison, not another H3 inference or numerical-parity run. Both
outputs were decoded with the same pinned FFmpeg 6.1.1.

| Check | Result |
| --- | --- |
| Old and new output encoding | Both completed successfully |
| Decoded output RGB | Same dimensions/frame count; hashes differ; pairwise PSNR 43.140 dB |
| Decoded output PCM | Same sample count; hashes differ; relative L2 difference 0.02474 |
| Reference fixture decoded at 44.1 kHz, stereo | 6.1.1 returned 45,159 samples/channel; 9.0.2 returned 44,100 |
| Existing native CUDA CLI using the 9.0.2 output encoder | 256×256, 22 frames, two steps; completed in 18.297 s |

The pairwise metrics describe differences between codecs, not a quality ranking.
The modern decoder produces a different reference-audio sample count in this
fixture; normal media lifecycle checks cover the migrated path. This experiment is not a full 204-output regression pass.

Evidence: [build/signature input identities](../../outputs/ffmpeg-upgrade/downloads.json),
[signature log](../../outputs/ffmpeg-upgrade/signature.log),
[build result](../../outputs/ffmpeg-upgrade/build-result.json),
[codec comparison](../../outputs/ffmpeg-upgrade/comparison.json),
[native render metadata](../../outputs/ffmpeg-upgrade/h3cli-new-encoder.json),
and [native render video](../../outputs/ffmpeg-upgrade/h3cli-new-encoder.mp4).
Evidence files are gitignored and available in this working checkout.

## Initial migration validation on PRO 6000

All functional checks passed on the local M4 and the authorized PRO 6000.
The fresh offline Linux build took 739.18 seconds with eight jobs.

| Check | Result |
| --- | --- |
| M4 setup, `make all test`, HTTP lifecycle | Passed; 21 HTTP cases |
| M4 real media/server jobs | Six passed; MP4 and AV state exactly matched the supported MPSGraph CLI |
| Host setup/build/regression policy | 27 / 8 / 18 tests passed |
| Linux normal runtime and private test-codec installer | Passed, including GPU library initialization |
| Linux launcher tests | 16 passed |
| Final package regression-profile checks | Nine passed: selection, relocation, missing/changed files and recovery |
| Normal CUDA server cases | 22 passed, including expected cancellation/interruption |
| Runtime smoke checks | 16 passed, including non-root rendering, relocation and stale-environment isolation |
| Current CUDA suite | All 18 stages passed, including real reference media, conditioning, residency, VAE and recovery |
| Immutable SGLang gate | Four complete runs; 204 outputs each; zero differences from the unchanged PRO 6000 baseline |

**The strict golden gate failed on PRO 6000:** each run matches 91 recorded hashes
and differs from the same 113 hashes already failing on the unchanged PRO 6000
baseline. The runs cover Ubuntu 22.04 unpacked and single-file modes, plus
Ubuntu 24.04 and Debian 12 single-file modes. Baseline agreement is diagnostic;
it does not turn the recorded-golden failure into a pass. No fixture, golden,
tolerance or case set changed.

The M4 server harness's explicit native-Metal CLI comparison hits its existing
performance gate. The comparison was completed on the supported MPSGraph
backend used by the server, with byte-identical MP4 and AV-state results.

The verified executable is [preserved migration candidate](../../bin/linux-release-ffmpeg902-candidate/h3cli-linux-x86_64),
SHA-256 `f1253afd32ccdb5944cb1ce28395b40f6a718165902b44624485cec1998afdda`:
1,313,606,957 bytes; 340 payload files. Source fingerprint:
`485992d7c6c8b6dd0277b9913654b82946f02d1270742ca4f4831df85b386b1c`.
The earlier local release is retained under `bin/linux-release-before-ffmpeg902`.

Evidence: [aggregate result](../../outputs/ffmpeg-upgrade/result.json),
[M4 results](../../outputs/ffmpeg-upgrade/macos-result.json),
[CUDA server results](../../outputs/ffmpeg-upgrade/linux-server/results.json),
[CUDA stages](../../outputs/ffmpeg-upgrade/linux/qualification/features/result.json),
[runtime smoke results](../../outputs/ffmpeg-upgrade/linux/qualification/smoke/results.json),
[profile isolation](../../outputs/ffmpeg-upgrade/linux/profile-checks/result.json),
and [four full regressions](../../outputs/ffmpeg-upgrade/linux-distros/result.json).
Sample mixed-reference videos: [M4](../../outputs/ffmpeg-upgrade/macos-server/M04/video.mp4)
and [CUDA](../../outputs/ffmpeg-upgrade/linux-server/M04/video.mp4).
These evidence files are gitignored and available in this working checkout.
