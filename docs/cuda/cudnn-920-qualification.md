# Shared cuDNN 9.20 qualification

Encoding and decoding now run inside `h3cli` with **cuDNN 9.20.0.48**.
The encoder subprocess, its IPC implementation and build target, and the
private CUDA 12 compatibility dependencies have been removed. Setup installs
one checksum-pinned native runtime environment. The compiler remains
**CUDA 13.0 Update 3 / 13.0.3 (`nvcc V13.0.88`)**; cuDNN frontend remains
**v1.11.0**.

Validation on 2026-09-26 used an **RTX PRO 5000 72GB / SM120**, driver
**595.91.07**, the original model weights, and the bundled recorded fixtures.
All comparisons retained the existing golden manifest and exact hashes.
No SGLang service or regenerated oracle output was used.

## Results

| Check | Result |
| --- | --- |
| cuDNN 9.20 throughout, before removing the subprocess | **204/204 recorded hashes passed** |
| In-process encoding with shared cuDNN 9.20 | **204/204 recorded hashes passed** |
| Ordinary CUDA build and `make -j8 test` | Passed; the SM90-specific tuning probe reports a different architecture |
| Encoder cancellation and recovery | Exact recorded image moments; unchanged surrounding GPU output and cuDNN version |
| Image-conditioned CLI before/after subprocess removal | Identical video/audio latents, decoded RGB frames and decoded audio samples |
| Native runtime verification, including cuBLAS GPU initialization | Passed |
| Setup script and recorded-regression runner unit tests | 27 and 8 passed, respectively |
| macOS Metal build and `make test` | Passed; ten existing optional model/fixture tests skipped |

The complete recorded regression includes image and video encoding, audio
encoding and decoding, component probes, and a **640×480, 124-frame, six-step**
render. Its 204-output comparison has no tolerance or skip switches.
The final run took **147.2 seconds after build**, or **237.6 seconds including
build**, within the 12-minute execution deadline. Its identities were:

```text
source:   bf37e76c1f4abe60693a2af37164f2832b660d14f0689e0571ace96b22a0c894
goldens:  fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3
h3cli:    b86539b12509edf68dd30fc4a1f2a4971662690dd374f0148caf81577b15ec25
```

The additional conditioned comparison used `inputs/1.jpg`, seed 42,
128×128, 22 frames and two steps, with the prompt
“A woman turns toward the camera.” Both builds used the same cuDNN 9.20
runtime. Saved AV checksums were validated before comparing the latent
payloads; both MP4s were decoded with the same FFmpeg executable.

Process-library inspection during conditioned generation found one cuDNN
9.20 library family and CUDA 13 cuBLAS/runtime libraries, with no CUDA 12
compatibility libraries loaded. The reference context also checks for the
cuDNN API identity `92000` before executing convolutions.

The ordinary tests needed their full recorded fixture tree copied into the
isolated regression build directory. That directory intentionally contains
only the fixtures needed by the 204-output runner. The initial ordinary test
attempt failed on absent fixture files; copying the existing fixtures resolved
it without changing code or expected results.

## Reproduction

Use the [source-build setup instructions](../build/source-build.md#ubuntu-2404-lts-x86-64-and-nvidia-cuda)
and the [complete recorded regression](../../CONTRIBUTING.md). For an existing
installation, rerun setup and source the regenerated `linux-env.sh`. The old
private `outputs/setup/cuda-runtime` environment is no longer used; the shared
runtime lives in `outputs/setup/sglang-runtime`.

The supplemental cancellation/recovery check accepts the image moments from
a passing recorded regression:

```sh
make bin/cuda_sglang_encoder_recovery
./bin/cuda_sglang_encoder_recovery \
  "$H3_REFERENCE_MODEL/FL2VA/video_vae/source" \
  tests/fixtures/cuda-reference/first.png \
  "$H3_REFERENCE_REGRESSION_OUT/data/image-moments.f32"
```

Historical qualification documents retain the dependency versions and
subprocess layout actually used for their measurements. These new results
qualify shared cuDNN 9.20 on SM120; they do not establish bitwise equivalence
on other GPU architectures.
