# Bounded long Sage3 qualification

All 16 LSG tasks completed on **2026-11-24**. The final standalone Linux
artifact passes the unchanged **204/204 exact-output regression**. The
[machine-readable acceptance record](sage3-long-acceptance.json) retains bounded
run results, sampled timings, placement, finite-state checks and artifact identities.

The [contract](sage3-long-contract.md) supports native SM120 CUDA generation
through 1,450 aligned frames, with independent BF16/NVFP4 projections. Runtime
tests use only 256×256, at most 464 frames, two evaluations and all 50 blocks.
Maximum-duration/canvas evidence is metadata-only. No 60-second speed, quality
or memory-fit claim follows from this campaign.

The CUDA target is an NVIDIA RTX PRO 5000 72GB Blackwell, SM120, with
73,415 MiB reported device memory and driver 595.99.02. The build uses CUDA
13.0, cuBLAS 13.1.1.3 and cuDNN 9.20.0.48. GPU work is serial and guarded by
the 60 GiB development ceiling plus the existing physical-device reserve.

## Baseline, operator and host evidence

- Fresh baseline and post-implementation regressions passed **204/204**, with
  the original fixture and manifest hashes unchanged. The code gate took
  152.57 seconds after its isolated build.
- Shared host planner tests cover six canvases through 1,450 frames, the
  one-million-row boundary, padding tails, 64-bit element offsets and bounded
  correction descriptors. Allocation-free capability probes reject missing
  Sage builds, non-SM120 devices and old CUDA runtimes. Near-limit reference
  rows accept exactly one million joint rows and reject the next row.
- Six short binary comparisons are exact: 1, 129 and 2,281 rows in both layouts,
  comparing the preserved pre-change binary to the new native operator.
- 37 standard and 19 forced-chunk functional cases pass, including zero,
  constant, random and nonfinite input. Inputs and guard regions are checked.
- Bounded cancellation/allocation/reuse tests pass. CUDA memcheck reports no
  errors for 257 rows with 128-row query chunks and three-head groups.
- All 100 pinned Sage3 oracle comparisons pass the unchanged relative-L2
  threshold of 0.01. Standard plans (70 comparisons) match exactly; forced
  chunks (30) reach at most 0.000016229 relative L2. The final two-head group,
  query tails and complete logical key range are exercised.

The oracle is upstream SageAttention commit
`d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`, with the pinned CUTLASS dependency,
built separately from inference. This validation environment uses Torch
2.13.0+cu130 and Triton 3.7.1; the native arithmetic recipe retains its original
explicit reduction order. The oracle applies the existing independent logical
key-tail mask without changing the upstream binary.

The independent blocked FP32 comparison describes approximation error, not a
port regression. Maximum relative L2 across the standard fixtures was 0.191
for normal inputs, 0.393 for adversarial outliers, 1.0 for tiny values, zero
for zeros and 0.03125 for constants. Tiny values can underflow to zero; their
maximum absolute error was 0.000433. Constant absolute error was 0.015625,
within the predeclared 0.02 contract. These measurements do not establish
perceptual equivalence to dense BF16 attention.

## Native and workflow checks

| Check | Sage3 + BF16 | Sage3 + NVFP4 |
| --- | --- | --- |
| 464-frame prompt-only, first/last anchors, mixed image/audio/silent-video references | Pass | Pass |
| Video references with embedded soundtrack | Pass | Pass |
| Full replay, preview replay, dimensions and A/V timestamps | Pass | Pass |
| Same-build pause/resume and cached conditioning | Exact latents | Exact latents |
| 447-frame source/context → 464-frame hard and bridge continuation | Exact protected history; exact full replay; 464-frame merges | Same |
| Long script, seed variants, inherited and segment-local references, continuation workers | Pass | Pass |
| API repeat, memory denial, load cancellation, attention/precision switches | Exact restored latents and PCM | Same |
| HTTP native worker, saved policy, early limit rejection and legacy still compatibility | Pass | Pass |
| Full/preview cancellation, encoder failure, checkpoint/output preservation and recovery | Pass | Pass |

Default-attention BF16 and NVFP4 controls also pass prompt-only, anchor and
mixed-reference cases. Native Sage3 dispatch counters show no dense-attention
fallback. Extended NVFP4 selects projection recipe 5 and reuses the existing
recipe-2 packed-weight cache; warm loads report 200 cache hits.

The API repeat test held BF16 RSS at 39,560,159,232 bytes. NVFP4 RSS changed
from 11,873,263,616 to 11,873,439,744 bytes. Each repeat changed free VRAM by
2 MiB, below the fixed 512 MiB plateau allowance; final teardown reports zero
tracked tensor allocations. Returning from default/BF16 to the original Sage3
selection reproduces both video and audio exactly.

An initial recovery attempt used an old test object and failed before saving
the expected checkpoint. Recompiling the API helpers against current headers
resolved that harness issue; the complete recovery case and both protected
prefix checks then passed. The failed attempt and diagnostic are retained
separately from accepted results. No inference-code change or relaxed numerical
tolerance was used to turn it into a pass.

A separate rejection test found that CLI resume precision conflicts reached
model lookup before the engine rejected them. CLI checkpoint preflight now
rejects both BF16→NVFP4 and NVFP4→BF16 overrides first, with permanent lifecycle
coverage. Both lifecycle cases were rerun, and the post-fix code regression
passed 204/204 in 119.84 seconds. This changes rejection handling only; the
inference paths used by the benchmark matrix are unchanged.

FL2VA→Ref2VA continuation with different normalization/VAE signatures remains
an explicit error. The supplemental mixed-reference continuation fixtures use
matching Ref2VA sources and targets. Both hard and bridge continuation pass in
both projection formats with exact protected history; the incompatible-source
attempt is retained as rejection evidence.

All 30 supplemental exclusion checks return exit code 2 before model lookup or
output creation. They cover incompatible resume attention/precision, AV transfer,
source capture, FP8, other attention implementations, reuse/depth/token changes,
adaptive caching, Metal and LoRA. Host/script/state tests additionally cover
indirect checkpoint and `#upscale` requests.

Automatic placement selects resident weights in both projection formats and
reproduces streamed native latents exactly. Injected allocation limits also
preserve exact results after retry: BF16 settles at 11 resident blocks with
streamed remaining weights, while NVFP4 switches to compressed streaming. The
NVFP4 planner log still describes its initial resident proposal on retry; the
subsequent quantization configuration and execution counters confirm the actual
compressed-stream mode. The failed partial loads are released before retry.

The matched 464-frame prompt-only controls have the following latent differences
from default attention. These are approximation measurements on one seed and
two evaluations, separate from the exact same-recipe contracts above.

| Projection format | Video relative L2 / maximum absolute error | Audio relative L2 / maximum absolute error |
| --- | ---: | ---: |
| BF16 | 0.4613 / 4.4770 | 0.4025 / 0.7345 |
| NVFP4 | 0.3202 / 1.8801 | 0.6355 / 2.3168 |

They do not establish perceptual equivalence or predict long-video quality.

## Instrumented performance and memory

Each configuration has one warmup and three measured fresh CLI runs, with the
same FL2VA model, teapot prompt, seed 42, 256×256 canvas, 464 frames, two
evaluations and all 50 blocks. The actual joint sequence is 10,328 rows.
All runs enable `H3_PROFILE=1`; these are diagnostic timings, not an unprofiled
production-speed qualification. Times below are medians for both evaluations
together. Actual placement matches the requested resident or streamed mode.

| Placement / projections / attention | Denoise s | Whole attention s | Projection GEMM s |
| --- | ---: | ---: | ---: |
| resident / BF16 / default | 7.347 | 1.787 | 5.003 |
| resident / BF16 / Sage3 | 6.788 | 1.237 | 4.992 |
| resident / NVFP4 / default | 4.250 | 1.785 | 1.908 |
| resident / NVFP4 / Sage3 | 3.692 | 1.238 | 1.902 |
| stream / BF16 / default | 7.528 | 1.782 | 5.029 |
| stream / BF16 / Sage3 | 7.073 | 1.339 | 5.003 |
| stream / NVFP4 / default | 16.980 | 1.590 | 1.705 |
| stream / NVFP4 / Sage3 | 16.706 | 1.145 | 1.705 |

Whole-attention time is 25–31% lower with Sage3 in these measurements. The
corresponding denoise reductions are 7.6% for resident BF16, 13.1% for resident
NVFP4, 6.0% for streamed BF16 and only 1.6% for streamed NVFP4. No Sage3 denoise
regression appears in this matrix. The small streamed NVFP4 difference should
not be generalized from three samples to longer videos.

The following peaks are maxima across the three measured runs. Tracked peaks
are DiT tensor counters; whole-device peaks include preparation and decoding.
RSS sums resident pages across descendants and may count shared pages twice.

| Placement / projections / attention | H2D GiB during steps | Tracked tensor peak GiB | Whole-device peak GiB | Process-tree RSS GiB |
| --- | ---: | ---: | ---: | ---: |
| resident / BF16 / default | 0.007 | 38.385 | 38.951 | 1.330 |
| resident / BF16 / Sage3 | 0.007 | 38.883 | 39.455 | 1.420 |
| resident / NVFP4 / default | 0.007 | 12.597 | 13.275 | 1.344 |
| resident / NVFP4 / Sage3 | 0.007 | 13.095 | 13.779 | 1.435 |
| stream / BF16 / default | 71.066 | 3.932 | 5.438 | 36.796 |
| stream / BF16 / Sage3 | 71.066 | 4.430 | 5.438 | 36.886 |
| stream / NVFP4 / default | 56.531 | 2.584 | 5.439 | 11.014 |
| stream / NVFP4 / Sage3 | 56.531 | 3.082 | 5.439 | 11.105 |

Sage3 adds about 0.5 GiB to the tracked denoiser peak. The full decoder sets the
whole-device peak for streamed cases, so equal overall peaks do not imply that
Sage3 has no memory cost. These measurements do not replace the separate
maximum-duration capacity estimates.

Complete Sage3 component costs include centering/reduction, packing, correction,
the attention kernel and output handling. The whole-attention column above
also captures gaps around those regions; do not substitute kernel time alone.

| Sage3 placement / projections | Center/reduce s | Pack s | Correction s | Kernel s | Output s | Complete component sum s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| resident / BF16 | 0.049 | 0.123 | 0.132 | 0.873 | 0.059 | 1.236 |
| resident / NVFP4 | 0.049 | 0.123 | 0.135 | 0.872 | 0.059 | 1.237 |
| stream / BF16 | 0.057 | 0.136 | 0.162 | 0.869 | 0.073 | 1.297 |
| stream / NVFP4 | 0.047 | 0.114 | 0.128 | 0.797 | 0.059 | 1.145 |

Streamed NVFP4 remains much slower than resident NVFP4 for this fixture despite
its lower GEMM time. It uploads 56.531 GiB during the two evaluations, with
packed-weight copies on the compute stream and repeated bounded projections.
This existing overhead dominates the small attention saving; changing it is
outside this milestone. Conversion is included in GEMM, and transfer event
accounting has the limitations described below.

The separate cold-cache NVFP4/Sage3 run prepares all 200 projection weights
with zero cache hits. Core loading/preparation takes 49.68 seconds, versus
14.79 seconds for the earlier warm streamed native case with 200 hits and no
conversions. Complete native execution takes 117.46 versus 83.43 seconds;
these are individual observations, separate from the warm benchmark medians.
The cold run peaks at 5,570 MiB whole-device usage and 11.10 GiB process-tree
RSS, and passes full and preview replay. Its video/audio latents exactly match
the warm-cache run. Packing does not change the stored
weight recipe or require a complete resident BF16 copy of the transformer.

## macOS checks

The macOS package passes offline 256×256 / 22-frame / two-evaluation generation.
The previously authorized 256→512, 22-frame, K=0 upscale compatibility case
also passes. Host checks include 51 script tests, 23 sampler test groups
(265 adversarial cases), resource guards and early long/CUDA-only rejection.
Metal retains its 362-frame ceiling and does not execute Sage3.

The executable is an ad-hoc signed, unnotarized development build. Artifact
SHA-256: `cf277c50bbca3c6060f7177fb74cf044b9d54c67ac1b6f4cfd011c78966c9559`.

## Reproducing bounded checks

Use the existing qualified CUDA build on `cuda-test`. Run GPU work serially.
`tests/long_video_campaign.py` owns the shared process lock, plans every child,
checks attention/projection provenance and monitors the 60 GiB device ceiling.
Historical attention suites with larger implicit inputs are outside this scope.

```sh
python3 tests/long_video_campaign.py native --attention sage3 --precision bf16 \
  --model /path/to/MiniMaxH3 --preview-model /path/to/taeh3.safetensors \
  --out outputs/sage3-bf16-native
python3 tests/long_video_campaign.py native --attention sage3 --precision nvfp4 \
  --quant-cache /path/to/packed --model /path/to/MiniMaxH3 \
  --preview-model /path/to/taeh3.safetensors --out outputs/sage3-nvfp4-native
```

The same runner provides conditioning, cached conditioning, pause/resume,
continuation, script, API, HTTP, recovery and benchmark cases. Benchmark mode
uses a warmup and three subsequent fresh CLI processes with matched explicit
placement. These are filesystem/packed-cache-warm runs, not retained API-context
runs. All enable `H3_PROFILE=1`: their timings are instrumented diagnostics,
not an unprofiled production-performance qualification. The API tests separately
exercise retained contexts.

Cold preparation is reported separately. Component categories overlap and must
not be added to wall time. In particular, pending preparation transfer events
can be collected during the first denoising step; its H2D event seconds and byte
delta need not cover the same interval. Do not derive bandwidth from that pair.
Projection conversion is already included in the GEMM category.

Compressed NVFP4 streaming has another accounting limitation: its packed-weight
copy is enqueued on the compute stream before the timed GEMM region. The byte
counter includes that copy, while the H2D event timer does not. Small reported
H2D seconds therefore do not mean negligible transfer cost. Recipe 5 also
retains its bounded slabs and MLP recomputation; the 464-frame fixture executes
1,000 native projections over two evaluations. This milestone does not change
that streaming or recomputation behavior. See the
[packed projection path](../../src/cuda/cuda_quant.cuh) for the copy and timer
boundaries.

The [capacity report](sage3-long-memory-estimates.md) is a conservative planning
estimate. Sage3's 512 MiB scratch is additional to retained BF16 full-sequence
storage. No high-resolution or maximum-duration runtime is used to validate it.

## Final artifacts and regression provenance

The standalone Linux executable passes offline Sage3 native generation,
full replay and preview replay in both projection formats, with exact
video/audio latents relative to the source build. It also rejects conflicting
resume precision before model lookup. The delivered executable is
`linux-package/h3cli-linux-x86_64` on `cuda-test`.

Linux executable SHA-256: `6aa1694d4a2bad7f29cdf3accfce3f1a6ddcdf726ab8881ab3b5dac9d34177e6`.
Final code/test source SHA-256: `1665c21fba134640f7f595b115e65f1595f50587e83544b405d9c0971bf18d8f`.

| Gate | Exact outputs | Execution seconds |
| --- | ---: | ---: |
| Fresh pre-change baseline | 204/204 | 178.00 |
| Coherent Sage3 implementation | 204/204 | 152.57 |
| Resume preflight correction | 204/204 | 119.84 |
| Final delivered Linux executable | 204/204 | 123.91 |

Every gate uses the original fixtures and unchanged golden manifest
`fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3`.
Gate timings exclude build and artifact-verification time. The archived
736-frame, large-canvas and empty-still task lists retain their original hashes.
Final source archives include the completed documentation; documentation-only
refreshes preserve the tested executable bytes and runtime payloads.

The acceptance record separates failed diagnostic attempts from passing
qualification. Raw logs, commands and machine paths remain in ignored outputs.

## Task evidence

| Task | Implementation and acceptance evidence |
| --- | --- |
| LSG001 | Frozen contract, baseline source/artifact identities and fresh 204-output gate. |
| LSG002 | Attention-aware campaign/resource guard, child/state provenance and bounded runtime plans. |
| LSG003 | Native default/Sage3 admission; retained source/upscale exclusions and early rejection cases. |
| LSG004 | Extended Sage3/NVFP4 recipe 5; bounded slabs, unchanged global scales and packed cache recipe. |
| LSG005 | Widened imported V offsets, shared checked descriptors and metadata crossings beyond INT_MAX. |
| LSG006 | Shared plan 1 within 512 MiB; full key coverage, forced query/head tails and short binary parity. |
| LSG007 | Thirty host capacity plans over six canvases, matching macOS/Linux reports and measured bounded peaks. |
| LSG008 | Early sampler policy validation, same-recipe resume/replay, conflicting override rejection and context switches. |
| LSG009 | CLI/API/HTTP/script policy propagation, conditioning, anchors, seeds and continuation workers. |
| LSG010 | Resident/stream/auto placement, injected retries, cancellation/recovery, repeat plateaus and teardown. |
| LSG011 | Host frame/canvas/row/offset boundaries, unavailable-capability probes and retained exclusions. |
| LSG012 | Standard/forced native operators, 100 pinned-oracle comparisons, FP32 characterization and memcheck. |
| LSG013 | Four native attention/projection controls, anchors/mixed references, finite latents and dispatch counters. |
| LSG014 | Exact resume/history/replay, mixed-reference hard/bridge continuation, public workers and macOS compatibility. |
| LSG015 | Eight matched instrumented benchmark groups, warmup plus three samples, cold preparation and overhead analysis. |
| LSG016 | Offline macOS/Linux packages, four unchanged 204-output gates, final artifact provenance and documentation checks. |

All maximum-duration and production-canvas claims remain planning/descriptor
coverage. Runtime qualification is 256×256 through 464 frames, plus the
explicit unchanged-regression and short BF16 upscale exceptions. No 60-second
runtime, perceptual equivalence or universal 72 GB fit is claimed.
