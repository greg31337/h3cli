> Historical qualification/provenance record. The original-checkout VAE comparison
> was removed during [current-state cleanup](../cuda/legacy-cleanup-results.md).
> Its commands below describe the archived run, not the current test suite.

# Single-still CUDA qualification

Node: `cuda-pro6000`, build directory `/path/to/h3-still`, original
models at `/path/to/models/MiniMax-H3`. Hardware is NVIDIA RTX PRO 6000
Blackwell Server Edition, SM120, 97,887 MiB reported VRAM (96 GB class), driver
595.91.07. Toolkit is CUDA 12.8; independent reference uses PyTorch 2.8.0+cu128.
The local qualification run is dated 2026-09-22 (the node's UTC logs use
2026-09-23).

This record covers the default CUDA backend, F16 image-checkpoint weights
expanded to F32, and dense BF16 still generation. It does not qualify Sage,
NVFP4, fast CUDA, cuDNN, Turbo or other accelerated compositions. The original
[still contract](single-still-generation-contract.md) and frozen codec gates
remain unchanged. The feature uses the existing CUDA operators; no new CUDA
kernel was needed for the T=1 path.

## Build and retained evidence

The current working tree, including the uncommitted single-still implementation,
was uploaded as source. Input image, normalized Metal codec latents, independent
RGB fixtures and the retained 243-frame video state were copied separately.
Downloaded original weights were reused through a symlink; model files were
not modified. The pinned image VAE was downloaded to `/path/to/models/image-vae`
and verified against SHA-256
`6c3d0bfa055986a803a566a862fcde283a1e63db62829e5ef4a2a5aebf50bb86`.

The node required `libicu-dev`, `libjson-c-dev` and `pkg-config`. Build:

```sh
cd /path/to/h3-still
export PATH=/usr/local/cuda/bin:$PATH
make -j8 CUDA_ARCH=120 bin/h3cli bin/still_probe bin/still_tests bin/still_generate \
  bin/still_controls bin/still_lifecycle bin/still_video_regression \
  bin/still_video_baseline bin/still_layout_fixture
```

The binary is `/path/to/h3-still/bin/h3cli`. Source/input SHA-256 manifests and
compiled-binary hashes are retained in `outputs/single-still-cuda/`. The build
completed; GCC reports indentation/style warnings, without compile/link errors.

Run real-model phases serially:

```sh
python3 tests/still_cuda.py codec
python3 tests/still_cuda.py video
python3 tests/still_cuda.py generation
```

`tests/still_cuda.py` records every command, exit status, controlled environment
and wall time. The generation phase enforces six evaluations per render. Video
checks decode saved latents only, at 22 and 243 frames. The independent CUDA
reference explicitly disables TF32 matmul/convolution and cuDNN benchmarking.
The candidate retains the default CUDA execution policy.

Host checks use the small metadata-only fixture and sparse temporary files.
The original continuation validator exercises invalid requests whose nominal
schedule is 20 steps; its pure host run omits the six-evaluation override so it
can reach the intended validation error. It does not execute a denoiser. All
actual denoising tests keep the six-evaluation ceiling.

## Qualification results

The host suite and CUDA synthetic tests pass, including 25,166 T=1 checks
under Compute Sanitizer with **zero errors**. Existing suites pass with 1,778
core checks, 1,658 sampler-state checks, 103,204 continuation checks and 49,206
tile/overlap checks. Host ASan/UBSan passed. Compute Sanitizer instruments the
actual CUDA kernels; the host sanitizer includes the shared tile core.

All three real codec sizes pass the frozen MAE ≤0.01 / PSNR ≥35 dB gate.

| Crop | CUDA/retained reference PSNR | CUDA/Metal PSNR | Fresh CUDA/reference PSNR | Source/round-trip PSNR |
| --- | ---: | ---: | ---: | ---: |
| 256×256 | 131.72 dB | 131.74 dB | 139.35 dB | 29.93 dB |
| 512×512 | 133.73 dB | 133.71 dB | 143.10 dB | 34.20 dB |
| 640×480 | 133.71 dB | 133.70 dB | 143.79 dB | 35.71 dB |

The same-latent test uses the exact retained Metal inputs. Fresh round trips use
the CUDA encoder and a separate PyTorch CUDA F32 reference. The latter uses the
node's FFmpeg crop, so small source-pixel differences from the Mac are not mixed
into a claimed same-input comparison. Encoder latent MAE is 8.77e-7–1.08e-6.

Repeated decode GPU allocation was exactly 9,746,566,432 bytes for all three
calls at each size, with peak 9,746,574,624 bytes. Warm 640×480 decode measured
0.514/0.511/0.511 seconds. Loading including checkpoint inspection and F16
conversion took about 31 seconds on this node; the public API validates identity
before loading as well. Startup hashing and conversion dominate these small
renders. These timings are not a full-render speedup claim against Metal.

All four slice outputs match the retained independent reference. Wrong
normalization, cancelled loading/decoding/final callback, retry after a cancelled
block, failed PNG encoding and exactly one PPM/frame callback are covered.
Existing destinations remain intact on cancellation/encoder failure.

The execution log's `vae=tf32` field is the inactive fast-CUDA preset; the
qualified run reports `effective=default`, zero fast decoder dispatches and no
cuDNN route. It does not select the fast-CUDA TF32 attention branch.

## Video preservation

The baseline and updated CUDA decoders have identical F32 RGB checksums at
both 22 and **243 frames**, including callback counts/order. The 243-frame
fixture uses T=72 and 14 streamed batches, with no new video denoising.

| Frames | Baseline decode | Updated decode | F32 RGB SHA-256 |
| --- | ---: | ---: | --- |
| 22 | 8.08 s | 7.87 s | `ec105aa6186cfc1d7cfdbffd7bba6aa302f81884663e6f7f38dda72f805692ae` |
| 243 | 108.20 s | 108.08 s | `574260cea1fa869538a5dbb7eb158009767c969ee08b0bd14c7d384156fea162` |

This is a within-CUDA regression comparison; it does not claim bitwise identity
between CUDA and Metal video pixels. The retained state checksum is recorded
in the build manifest.

## Bounded still generation

All three serial renders passed at 640×480, seed 42, six evaluations, original
BF16 weights, dense attention and all 50 blocks. Each used one video latent
frame plus two auxiliary audio ticks generated and discarded. Each returned
one final image callback, seven finite latent states (initial plus six), no
AV/sampler state and zero output audio. No audio VAE was loaded.

| Case | Wall | Denoising interval | Image identity/load | Image decode | PNG/delivery | Peak GPU tensors | Sampled host RSS + swap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt-only | 89.54 s | not collected | 47.17 s | 0.559 s | 0.109 s | 36.055 GiB | 0.803 GB |
| Same-seed repeat | 74.53 s | 0.867 s | 47.68 s | 0.557 s | 0.110 s | 36.055 GiB | 0.803 GB |
| Image reference | 73.41 s | 1.219 s | 47.80 s | 0.513 s | 0.105 s | 36.183 GiB | 0.810 GB |

The first generation used the original test harness. A timing-only revision
added an explicit progress-callback interval before repeat/reference testing;
the inference executable/library and arithmetic did not change. Initial and
final test-source/binary manifests are retained. CUDA's existing GPU profile
marks are cumulative since context creation, so their “Euler denoise” wall
values are not reported as standalone denoising duration. The explicit interval
includes the per-step latent capture used by this test. Host RSS and GPU tensor
allocation are separate measurements and are not comparable to unified-memory
Metal footprint by treating either alone as total memory.

The prompt-only repeated run matched the initial state, all six transitions,
normalized latent and final PNG **exactly**. Re-decoding its saved latent also
matched the PNG checksum. The images show the requested red teapot and a
reference-conditioned portrait. They are bounded semantic smoke tests, not a
broad identity/prompt benchmark or 50-step quality qualification. Same prompt
and seed across CUDA and Metal do not imply identical diffusion trajectories.

## Completed evidence

SS021 is complete for this SM120 node and default CUDA execution. All tasks in
[the archived single-still plan](single-still-tasks.md) are now closed; accelerated compositions retain
their previous independent qualification requirements.

- [CUDA comparison gallery](../../outputs/single-still-cuda/review.html)
- [Results and measurements](../../outputs/single-still-cuda/results.json)
- [Exact test commands](../../outputs/single-still-cuda/commands.jsonl)
- [Source, fixture and binary manifest](../../outputs/single-still-cuda/build-manifest.json)
- [Retained-file checksums](../../outputs/single-still-cuda/checksums.json)

Generated assets are retained in this workspace; source, test binaries and logs
also remain on the node. No model payloads were downloaded back from the node.
The reproducible harness is `tests/still_cuda.py`, and the CPU-only report builder
is `tests/still_cuda_report.py`. The original codec contract was not relaxed.


