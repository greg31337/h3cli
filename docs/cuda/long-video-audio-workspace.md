# Extended-duration CUDA audio workspace correction

A 640×480/719-frame render completed denoising but failed at AudioVAE's pre
convolution with `reference audio cuDNN workspace exceeds 512 MiB`. The frame
extension retained that old cap in both the cuDNN planner and the tracked
scratch allocator. This was an internal workspace limit, not evidence of
exhausted physical VRAM. The original 464-frame runtime test did not cross it.

Descriptor-only reproduction with cuDNN 9.20 on SM120 finds the first failing
aligned duration at **566 frames / audio T=943**. The pinned FFT workspace
jumps from 286,883,840 bytes to **2,176,942,080 bytes (2.027 GiB)**. The same
larger requirement applies to 719 and 736 frames. Raising the cap only to
1 GiB is insufficient.

The planner and allocator now share an explicit **3 GiB** ceiling, allocate
only the queried amount, and retain normal memory admission. The original
convolution algorithms, FP32 arithmetic, media limits and tensor layouts are
unchanged. Errors above the ceiling now include the requested bytes, limit and
convolution shape. The existing 64 MiB per-layout cap is sufficient: the largest
layout through 736 frames is 62,822,400 bytes. The existing 7 GiB decoder device
estimate remains above the audio workspace and delivery allocations.

`make test-cuda-audio-plan` performs **4,136 descriptor-only convolution checks**
over all 44 aligned durations from 5 through 736. It covers projection, pre
convolution, all seven upsampling stages, residual kernels/dilations and output
convolution, without allocating activation, weight, workspace or movie tensors.
The old implementation fails this test at 566 frames; the corrected one passes.

Validation completed **2026-11-23**:

| Check | Result |
| --- | --- |
| All aligned-duration convolution plans | 4,136 passed; maximum workspace and layout sizes above. |
| Local macOS bounded host suite | Passed, including duration rejection, 41 script tests and four guard tests. |
| Corrected native CUDA binary | 256×256/464-frame generation, byte-identical full replay and preview passed. Seeded video/audio latents match the previous binary exactly. |
| Replacement Linux standalone | Offline 256×256/464-frame generation, byte-identical full replay and preview passed. |
| Final unchanged CUDA reference gate | **204/204 exact hashes**, 121.20 seconds, verified native binaries. |
| Package launcher suite | 15 passed; one skipped for the unavailable noexec mount. |

Native generation took 62.78 seconds; standalone generation took 85.11 seconds.
The native run's sampled device peak was 5,568 MiB. Those executions retain the
256×256/464-frame limit. The unchanged regression uses its separately authorized
historical fixture. Full 719/736-frame rendering is not performed in this
correction; those durations have descriptor-only coverage.

| Identity | SHA-256 |
| --- | --- |
| Replacement Linux standalone | `72bfa76a0345a8cca993e4c8e4a54cd4c1a543da201d43691b1e8611095080da` |
| Corrected native executable | `be2bbb27e6f1c5a01e22d152b715537d1306664e46e73fd677acda82e28f1fc3` |
| Final regression source fingerprint | `84f73ef3ca7c8ab6346d11ad6042066c46329d67eb12f6412a00bb001d80859b` |
| Package source snapshot | `b09065e518321ee6bf894febd5a280c60ac2dd195d7a6e1f44d7eb9057ebda01` |
| Unchanged golden manifest | `fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3` |

All 216 production source files plus the Makefile match between the package
snapshot and tested checkout. The standalone is retained under
`outputs/long-video/audio-workspace-fix/linux-package/h3cli-linux-x86_64` in the
test checkout, alongside private build and validation logs. The final native
gate record is under `outputs/cuda-reference-regression/audio-workspace-fix-204-001/`.

For expensive renders, `--save-sampler-state shot.h3sample` saves completed
denoising **before** audio/video decode. Resume that checkpoint with the same
compatible build to retry delivery without repeating denoising.
`--save-av-state` alone is written after decoding and does not provide this
protection. See [checkpoint recovery](../features/sampler-state.md).
