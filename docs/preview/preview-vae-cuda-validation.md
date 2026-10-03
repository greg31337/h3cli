# TAEH3 preview validation: RTX 5090

Validated on 2026-09-18 (local date; 2026-09-19 UTC). Native CUDA
`--preview-vae` is implemented and the available bounded automated checks pass.
Both optional cuDNN and a separate build without cuDNN work. The original
full decoder remains the default; tiny reconstruction works with or without
`--fast-cuda`.

**The user accepted the reviewed CUDA videos on 2026-09-18, completing T044:**
"all videos have acceptable quality". This separately approves the RTX 5090
cuDNN/cuBLAS comparisons and continuation joins. Open the
[CUDA review gallery](../outputs/preview-vae/cuda-5090/quality/review.html).
See also [usage](preview-vae.md), [tasks](todo.md),
[structured evidence](preview-vae-cuda-validation.json), and the
[previous Metal report](preview-vae-validation.md).

## Environment and reproducibility

- Baseline commit: `a3abbba57b34b8ff142523b460d66f9aa4849c65`. The structured
  evidence fingerprints final sources/binaries; individual ledger entries
  retain the executable hashes used during development.
- NVIDIA GeForce RTX 5090, SM120, 170 SMs, 32,607 MiB reported by `nvidia-smi`
  (31.36 GiB usable through the CUDA runtime). Driver 595.91.07, CUDA toolkit
  12.8.93, driver API 13.2, Ubuntu 24.04.3, GCC 13.3, FFmpeg 6.1.1.
- Native cuDNN **9.8.0.87**, runtime version 90800, linked from the system
  libraries. The separately installed Python package contains cuDNN 9.10.2;
  it is not the native runtime used for these results. Existing optional
  cuDNN frontend headers are pinned to 1.11.0, commit
  `8801fd7b31c2f798732ed1e1dd4711a9cde4217d`.
- All remote work and fixtures use local storage under `/path/to/models`:
  `h3-preview-vae` for the main build, `h3-preview-vae-nocudnn` for the
  fallback build, `h3-preview-deps` for dependencies/cache and
  `h3-preview-tmp` for temporary files. `/workspace` was not used.
  Full models are `/path/to/models/MiniMax-H3`. The container memory cap is
  91,999,997,952 bytes, smaller than the host's reported 186 GiB; no swap.
- TAEH3 checkpoint revision
  `62f7591f59dfbb4c3c02b7a621d180a9eeaba26c`, complete SHA-256
  `4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13`.
  The same pinned checkpoint and saved AV states used for the accepted Metal
  review were reused. Transfer of the reference-derived corpus was explicitly
  approved by the user. No new denoising was performed for the quality gallery.

The [persistent ledger](../outputs/preview-vae/cuda-5090/budget.json) includes
all GPU attempts, failures, instrumentation, warmups, generation, media mux and
validation within the **1,800-second CUDA allowance**. Component reservations
are at most 120 seconds, complete workflows at most 300 seconds. The final total is **1366.40 seconds across 69 attempts**
(65 passed, two retained failures and two timeouts); detailed totals
are in the structured evidence. CPU builds, source synchronization and host-only
inspection are separate. No `test2.sh` or production-length render was run.
The existing Metal ledger additionally records two short shared-contract
regressions, bringing that round's cumulative use to 976.43/1,200 seconds.

## Implementation

`src/cuda/tiny_vae_cuda.cu` owns its device, nonblocking stream and library handles.
Weights and activations use FP16 in NHWC layout, with FP32 convolution
accumulation. cuDNN uses bounded, shape-specific forward plans; unsupported
plans use tensor-core cuBLAS with a bounded im2col panel. A plain CUDA build
requires neither cuDNN nor Python. The implementation follows the native
[cuBLAS API](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html)
and installed cuDNN headers; it does not require a Python inference framework.

The decoder implements the trained input transform, spatial convolutions,
nine temporal memory blocks, nearest-neighbor upsampling, temporal expansion,
pixel shuffle and RGB clamping. Three reusable activation buffers and nine
history buffers avoid per-layer allocation/readback. Only the input and a
bounded completed RGB batch cross the device boundary. Global valid-frame
selection, finite checks, RGB8 conversion, resize, trimming and mux use the
shared delivery code; frame selection remains independent of batch size.

At most 64 cuDNN plans and 32 MiB convolution scratch are used by default.
`H3_PREVIEW_CUDA_WORKSPACE_MB=1..128` controls scratch;
`H3_PREVIEW_CUDA_CONV=auto|cublas|cudnn` selects automatic, forced fallback or
strict cuDNN behavior. Device, checkpoint and policy identity participate in
cache matching. Buffers grow only to the largest requested valid geometry.
Preflight reserves a conservative geometry-dependent allowance. Cancellation
and handled failures drain outstanding work, reset history and preserve the
existing output file. Fatal CUDA context errors are still reported normally.

No reference encoder, AudioVAE, sampler or authoritative state format changed.
The CUDA decoder replaces the prior explicit unsupported stub. The later
source cleanup also removed the unused stub from the remote source archive;
source-identity metadata was explicitly rebuilt after that deletion.

## Automated coverage

| Area | Result and scope |
| --- | --- |
| Host/CLI | `make test-preview-vae` passes on CUDA and local Metal: default selection, missing/malformed/truncated models and states, sidecar association, early failure, legacy-state overrides, frame/trim arithmetic and decode-only dispatch. |
| Decoder structure | Real weights, color/pixel-shuffle patterns and a sparse temporal-history oracle pass with cuDNN and cuBLAS. Tests cover 7/17/27 latent frames, batches 1–5, non-five-aligned boundaries, tails, geometry changes, selected-frame context and reset. The temporal oracle checks current-plus-previous input independently of another backend. |
| Memory and policies | Bounded 1-MiB scratch, allocation-budget injection and recovery, memory preflight rejection, cache-policy changes, repeated requests, cancellation and invalid policy rejection pass. These are not forced physical device-OOM stress tests. |
| Optional dependency | A separately linked no-cuDNN build passes real/pattern/temporal contracts and saved-state replay. `ldd` confirms no cuDNN dependency. Its MP4 exactly matches forced cuBLAS from the cuDNN-enabled build. Strict cuDNN selection on this build fails explicitly. |
| Instrumentation | Compute Sanitizer memcheck passes for both cuDNN and cuBLAS with zero reported errors/leaks. Racecheck restricted to the new custom kernels passes; unrestricted vendor-kernel instrumentation timed out. See limitations below. |
| Authoritative state | Controlled two-step CPU-sampler cases retain exact raw Euler boundaries and final AV latents across full/tiny, noisy and denoised live previews, paused tiny output, resumed full output and full-after-tiny. Paused output remains silent. |
| Continuation | Hard and bridge full/tiny trajectories match exactly, including tiny live previews. Untrimmed 90-frame states deliver 51 new frames after removing 39 video frames and 52,000 audio samples. |
| Cache and selection | Real/pattern/missing/real model changes and explicit cache clear pass. Tiny selection with ordinary CUDA produces the exact same pilot MP4 as tiny plus fast CUDA. |
| Functional generation | All ten cases pass: first anchor, last anchor, both anchors, image, multiple face/body images, video, image plus audio, video plus separate audio, reduction and reuse. T2VA is covered by invariance and cold-workflow cases. |
| Finalization | Tiny-generated states finalize with the full decoder without DiT calls; original state hashes, dimensions, audio, resize and continuation trims are preserved. Exact RGB equality is retained for the default contract; fast-CUDA cross-process equality is diagnostic. |
| Media | All 26 completed generated MP4s, all 40 gallery videos, three 141-frame joins and the finalization outputs pass full decoding and presentation checks. Expected dimensions/counts, 24 fps, 32-kHz stereo audio and durations are verified. Full/tiny/cuBLAS gallery audio decodes identically. |
| Streaming and errors | Full streaming/buffered F32 output matches exactly within one decoder context at both short presets and the bridge fixture, including all seams/tails. Public API cancellation and encoder-failure tests preserve an existing target and reap FFmpeg. |
| Regressions | Linux `make test`, `make cuda-test`, `make cuda-fast-test` and focused preview tests pass. The normal suite includes 41 primitive comparison records and 70 exact tokenizer fixtures. SM90-specific tuning is explicitly skipped on SM120. Shared Metal real-weight and temporal-oracle regressions pass. |

Functional cases use two steps, usually 64×64 internal / 128×128 delivered and
22 frames. Video inputs require 56 target frames under the existing reference
contract. These renders prove integration, not useful visual quality. Folded
Turbo/realism model trees were unavailable on this node and were not transferred;
their existing Metal coverage is not presented as CUDA preview qualification.

The original default full-VAE MP4 before/after this change is byte-identical:
`5f9fd21972ef01ad9b61a29a19c6982d9d3d0d649b5a7aa031be86d74283a828`.
The no-cuDNN/forced-cuBLAS pilot MP4 is also byte-identical:
`002d54f3ea0f2423bbcc93522dba2e292649d059cbbcf144109da9f24c0aaffd`.
These exact checks protect existing behavior; tiny-versus-full numerical
similarity is not a quality acceptance gate.

## Performance

These are bounded single-device measurements with one warm repeat per preset,
not statistical estimates. Cold means a fresh decoder object/plan; OS file and
CUDA caches may already be warm. The full comparator uses `--fast-cuda`, so
improvements are measured against the existing accelerated full VAE.

| Preset | Full cold | Full warm | cuDNN tiny cold | cuDNN tiny warm | cuBLAS tiny cold | cuBLAS tiny warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 288×384 / 56 frames | 2.841 s | 2.716 s | 0.181 s | 0.0317 s | 0.139 s | 0.0549 s |
| 480×640 / 22 frames | 2.192 s | 2.077 s | 0.194 s | 0.0573 s | 0.174 s | 0.0829 s |

The warmed speedups are **85.6× / 36.2× with cuDNN** and
**49.5× / 25.1× with cuBLAS**, exceeding the 3× decoder target. These intervals
include compute, RGB transfer and sink checks, but exclude audio, mux and
model-compatibility hashing. Full loading takes 0.815/0.781 seconds;
tiny loading is about 0.18 seconds for either route.

Warmed cuDNN compute/transfer is 0.00760/0.00320 seconds at 288×384 and
0.01237/0.00363 at 480×640; cuBLAS compute is 0.03071/0.03522 seconds.
The remaining measured decoder time includes host finite/sink work.
cuDNN is faster once warmed; cuBLAS can start faster because it avoids cuDNN
initialization/planning. The current optional-cuDNN/default-fallback policy
preserves that choice without adding another public CLI flag.

| Preset / route | Decoder-owned device peak | Pinned host readback | Observed device allocation | Maximum host RSS |
| --- | ---: | ---: | ---: | ---: |
| 288×384 cuDNN tiny | 283.3 MiB | 25.3 MiB | 817.2 MiB | 714.2 MiB |
| 288×384 cuBLAS tiny | 283.3 MiB | 25.3 MiB | 809.2 MiB | 577.2 MiB |
| 480×640 cuDNN tiny | 696.5 MiB | 70.3 MiB | 1239.2 MiB | 723.8 MiB |
| 480×640 cuBLAS tiny | 696.5 MiB | 70.3 MiB | 1231.2 MiB | 667.3 MiB |

The full decoder observes about 9.99 GiB device allocation at either preset,
with 9.365 GiB logical GPU peak. Owned buffers, device-wide allocation
observations and Linux host RSS are different metrics; library/context
allocations explain part of the gap. Linux RSS excludes device VRAM. These
standalone measurements exclude generation's cached transformer/encoders.

| Saved-state replay | Full video delivery | cuDNN tiny delivery | Full total CLI | Tiny total CLI |
| --- | ---: | ---: | ---: | ---: |
| image05, 288×384/56 | 3.008 s | 0.340 s | 9.20 s | 5.97 s |
| image12, 288×384/56 | 3.045 s | 0.353 s | 9.47 s | 5.97 s |
| face/body, 288×384/56 | 3.009 s | 0.345 s | 9.30 s | 5.95 s |
| detail, 480×640/22 | 2.350 s | 0.370 s | 8.17 s | 5.56 s |
| chain1, 288×384/90 | 4.901 s | 0.408 s | 11.58 s | 6.44 s |
| chain2, 288×384/51 delivered | 4.832 s | 0.350 s | 11.50 s | 6.41 s |
| face20, 480×640/22 | 2.373 s | 0.361 s | 8.23 s | 5.58 s |

Video delivery includes decoding, conversion/resize, pipe writes and mux.
For image05 tiny, the breakdown is 0.181 seconds decode/transfer pipeline,
0.054 conversion, 0.055 pipe writes and 0.050 mux completion. First-frame
latency improves from 1.047 to 0.165 seconds; trimmed chain2 improves from
2.846 to 0.176 seconds. CLI totals also include compatibility hashing,
AudioVAE and loading. Ordinary CUDA's portable compatibility hashing is much
slower here than the fast-CUDA/OpenSSL route; that is separate from tiny VAE
arithmetic. Tiny replay without `--fast-cuda` passed at about 30 seconds.

The matched fresh-process, two-step T2VA workflow takes **34.078 seconds full
versus 33.367 seconds tiny (1.021×, 2.1% less wall time)** through MP4 and saved
state. Its small canvas and model startup dominate. Both final states hash to
`266d578a42a576aa9fd7eb797353e7cf13d46a14731072c54ffc7b0031c74184`.
Do not extrapolate decoder-only gains to complete generation, or compare a
cached tiny request against a cold full request.

Warmed 64×64/22 live previews take about 3.7 ms per displayed step, including
context and readback. Two continuation previews take about 14.7–14.9 ms
together. The source-only continuation helper does not timestamp latent
callbacks; its zero preview interval is excluded from performance claims.
Readback, conversion and fixed startup now matter more than the tiny network;
no further quality-reducing shortcuts were added in this round.

## Quality, retained failures and limits

The gallery compares eight identical states through CUDA fast full VAE,
cuDNN tiny, cuBLAS tiny, accepted Metal tiny and the pinned development
reference. The [manifest](../outputs/preview-vae/cuda-5090/quality/manifest.json)
records commands, source/state/input identities and MP4 hashes. Three joined
CUDA videos contain 90+51 frames with the continuation boundary at 3.75 seconds.
The corpus covers `1.jpg`, `2.jpg`, face/body, multiple references, motion,
detail and continuation. The older six-step 128×128 face fixture is broken
with the original full VAE too; it is retained only as a smoke comparison.
The separate 20-step 480×640 face fixture is the relevant face-quality example.

Static first/middle/last inspection shows corresponding composition, broad
color and poses across native CUDA tiny, Metal tiny and the pinned reference,
with expected loss of fine detail compared with full VAE. Static inspection
and exact audio checks cannot establish perceived motion, identity or listening
quality. Following review, the user separately accepted all CUDA gallery
videos on 2026-09-18: "all videos have acceptable quality". T044 is complete
for this corpus, including both native convolution routes and continuation
joins. Other CUDA GPUs and long, high-resolution clips are not qualified by
these results.

Four unsuccessful attempts remain charged and recorded:

- Vendor-wide racecheck reached its 120-second limit. Targeted racecheck first
  failed on incorrect filter syntax, then passed in 49.875 seconds after using
  `regex=`. Filtering covered all new custom kernel names; it did not requalify
  vendor-library kernels. Racecheck mainly detects shared-memory hazards and
  does not prove every global-memory ordering property. Both routes also passed
  memcheck, and temporal/reset/cancellation contracts exercise stream ordering.
- The initial fast-CUDA finalization test incorrectly demanded cross-process
  RGB byte equality. Bridge output differed in 6,621 of 626,688 decoded RGB
  bytes, mostly by one level (observed range −5..4 after H.264). The root cause
  of that small variation was not isolated. Identical source states and exact
  within-context full streaming/buffered output were verified. The test now
  retains exact default-path, state and audio gates while recording fast-mode
  cross-process equality diagnostically, consistent with the existing visual
  equivalence contract. The revised finalization test passes.
- The ten-case feature batch reached its 300-second limit after seven complete
  cases. Only the final three were rerun, passing in 96.191 seconds. All ten
  resulting MP4s passed the final media audit; the interrupted attempt was not
  erased or counted as a pass.

No automatic failure was converted to a quality approval. Full-VAE
quantization, TensorRT, reduced-resolution tiny latents, reduced-frame-rate
output and new decoder training remain independent follow-up ideas.
