# Single-still qualification

Local M4 Max / 128 GiB, 2026-09-22. Implementation baseline:
`31d8ee834197b19b4f2ac1d4dc9f8d3174ce3a7c`, plus the single-still changes.
Codec qualification passed before the generation integration gate opened.
See [usage](single-still.md), [provenance](single-still-sources.md), and the
[sampling contract](single-still-generation-contract.md).

- [Codec comparison HTML](../outputs/single-still/review.html)
- [Generation examples HTML](../outputs/single-still/generation.html)
- [Machine-readable results](../outputs/single-still/results.json)
- [Artifact checksums](../outputs/single-still/checksums.json)

These generated assets are retained in this workspace, outside Git. Python
helpers and the pinned metadata fixture are included in the repository.

## Independent codec qualification

The independent reference evaluates the original VAE equations in PyTorch F32
with F16 weights expanded to F32. It does not call the C decoder. The frozen
contract is `tests/still_contract.json`: unclamped candidate/reference RGB MAE
≤0.01 and PSNR ≥35 dB; source/reconstruction PSNR >25 dB on the fixed photo
crops when the independent baseline passes that floor. All three passed.

| Crop | Candidate/reference MAE | Candidate/reference PSNR | Source/reconstruction PSNR |
| --- | ---: | ---: | ---: |
| 256×256 | 1.24e-7 | 135.46 dB | 29.93 dB |
| 512×512 | 8.24e-8 | 139.00 dB | 34.20 dB |
| 640×480 | 7.69e-8 | 139.66 dB | 35.71 dB |

The image checkpoint encoder agrees with the independent posterior-mean
reference (latent MAE about 1.1–1.3e-6). All original encoder tensors equal the
image tensors after F16 rounding; this is not byte equality. Round trips load
the image checkpoint's own encoder and release it before decoder loading.
They do not share an original encoder cache.

All four output slices match the corresponding independent slices. Supplying
raw latent values as normalized values gives 14.60 dB against the proper
reconstruction, confirming that normalization matters. A whole-image 512×512
reference differs from released tiling by 24.18 dB and reconstructs the source
at 23.87 dB; larger tiles remain unavailable as an equivalent optimization.
Wrong checkpoint roles, missing metadata, invalid slices, malformed architecture,
nonfinite whitening/weights/latents, truncated files and overlapping component
selection are rejected by tests.

At 640×480, a retained decoder took 1.25/1.23/1.23 seconds across three calls.
GPU live allocation stayed at 9,746,566,432 bytes; peak was 9,746,574,624 bytes.
Loading and validation took about 5–6 seconds in those measurements. The public
path additionally validates compatibility before weight allocation; timings vary
with file cache state. Cancellation during loading, a decoder block and final
callback, retry after cancellation, and failed PNG encoding passed. These
checks preserve the destination and release owned allocations.

## Bounded generation qualification

Three serial 640×480 renders used six evaluations each, seed 42, dense BF16,
all 50 blocks, original model weights and one actual video latent frame. No
1344×768 generation or long video denoising was performed.

| Case | Wall | Denoising | Image load | Image decode | Delivery | Sampled process peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt-only teapot | 35.55 s | 6.13 s | 6.02 s | 1.25 s | 0.24 s | 39.07 GB |
| Same prompt/seed repeat | 32.97 s | 6.12 s | 5.99 s | 1.25 s | 0.25 s | 39.07 GB |
| Image-reference portrait | 49.25 s | 16.27 s | 5.98 s | 1.25 s | 0.26 s | 39.28 GB |

Wall time includes model/conditioning initialization and teardown; phase times
are not a cross-hardware benchmark. Peak footprint is sampled at progress and
latent callbacks, not a continuously measured upper bound. Startup/conditioning
account for most of the remaining wall time. Full phase logs are linked in HTML.

The repeated prompt matched initial noise, all six latent transitions, final
latent and PNG exactly. Re-decoding the saved prompt latent also reproduced the
PNG checksum. Each generation returned one final callback, seven latent
callbacks (initial plus six), one image, no AV/sampler state and zero output
audio. Auxiliary audio had exactly 128 F32 values and was discarded.

Independent tests cover T=1 packed positions, zero/one/two ordered image
reference slots, four audio rows, sigma schedules and seed-42 noise. Retained
Euler velocities are derived from successive sampler states; they are regression
fixtures, **not independent full-model velocity parity**. The semantic screen
shows the requested red teapot and an image-conditioned portrait. This small
corpus does not establish broad prompt adherence, identity fidelity or 50-step
quality. SOL/FP16/Q8/ANE/Turbo compositions inherit no acceptance from it.

## Video, lifecycle and platform coverage

Original and updated video decoders produced bit-identical F32 RGB at both
22 and **243 frames**. The latter reused the retained B6 BF16 video state and
streamed 14 batches; no denoising was needed. Its checksum is
`9bbb551a890e62d4ffdfab132f06ad8f07878b0163cd68bf77f11cb77d3f03b7`.
Decode times were 111.39 seconds before and 111.42 seconds after the shared-core
change; these are preservation checks, not a speedup claim.

Existing suites passed: 49,206 tile/overlap checks, 103,204 continuation checks,
1,658 sampler-state checks and 1,777 core checks. Still host tests cover strict
parsing, selection and ownership; synthetic GPU tests cover 25,166 T=1
RoPE/unpacking/conversion checks under ASan/UBSan and Metal API/shader validation.
The sanitizer compiles the included tile core with instrumentation; the linked
GPU library is a normal build, with Metal validation covering GPU dispatches.

Image/video roles are separate objects, and the image decoder is never placed
in the video cache. Tests cover operation-key separation, wrong-role selection,
repeated codec calls. A full same-context
still→video→still denoising sequence was not added to the bounded corpus.

CUDA has no new backend-specific kernel in this feature: shared C uses the
existing GPU interface and the file is included in the common build list.
The initial local run only checked portable C syntax. SS021 has since passed
real NVCC compilation and default CUDA execution on the user-provided RTX PRO
6000 Blackwell node: see [CUDA results](single-still-cuda-results.md) and its
[comparison gallery](../outputs/single-still-cuda/review.html). Codec comparison
uses the same retained latents; generation determinism is qualified separately
on each backend. No full-render cross-hardware speed claim is made.
