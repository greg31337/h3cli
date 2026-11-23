# CUDA large-canvas memory estimates

Planning date: **2026-11-22**. Implementation is tracked in
[the archived checklist](../large-canvas-todo.md). The tables below are
**estimates, not guarantees of admission**. See the separate
[short-run measurements](large-canvas-results.md)
for actual 22/39-frame evidence. No 5-, 10- or 15-second real-model videos were run
at the new maximum canvases. The separately authorized 204-output regression
passes with its original 640×480 / 124-frame case. Large-canvas development
must fit the existing 72 GB CUDA card and remain
at most three seconds. Use 22 frames by default, 39 only after preflight.
Continuation/bridge must not execute above **1344×768**. Upscaling must be
tested at **3840×2176**, including refinement and decoding, using 22 frames
or 39 after admission. The 5/10/15-second figures remain host-only estimates.

## Scope and assumptions

- Native generation: **1920×1088**, 2,088,960 pixels.
- Existing learned spatial 2× upscaler: **1920×1088 → 3840×2176**,
  8,355,840 output pixels, unchanged time and source audio. This is both the
  implementation target and the required short runtime test canvas, including
  its transpose; only the longer durations remain estimation-only.
- CUDA, batch one, 24 FPS, dense BF16, all 50 transformer blocks, one prompt
  budgeted at 512 text rows, no image/video references, no live previews,
  no LoRA, quantization or approximation caches. References require additional
  accounting; these numbers are not a worst-case reference budget.
- Full stock VAE with the existing 256-pixel tiles and streamed frame delivery.
  Source generation, learned transfer, refinement and decoding release their
  completed resources between phases. The job peak is the largest phase peak,
  not the sum of nonoverlapping phase peaks.
- **K=0** means learned transfer plus full decode, without a transformer
  refinement pass. **K=2/3/4** adds transformer sampling at the upscaled size.
  The current default is K=4; fewer positive steps mainly reduce runtime,
  rather than the dominant activation peak. Schedule storage still varies.
- GiB means 2^30 bytes. Physical capacity, free device memory, process RSS,
  tracked tensor bytes and whole-device peaks are distinct quantities.

## Duration and tensor counts

The [temporal helpers](../../src/host.c) align requested frames upward to
`F = 5 + 17*k`, with `T = 2 + 5*k` video latent positions and
`A = round(F*40/24)` audio latent positions. Audio contributes `2*A` packed
rows. Native video contributes `T*(W/32)*(H/32)` rows.

| Requested duration | Requested frames | Aligned frames | Actual duration | Video T | Audio A | Native video rows | Upscaled video rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 s | 120 | 124 | 5.167 s | 37 | 207 | 75,480 | 301,920 |
| 10 s | 240 | 243 | 10.125 s | 72 | 405 | 146,880 | 587,520 |
| 15 s | 360 | 362 | 15.083 s | 107 | 603 | 218,280 | 873,120 |

These rows describe the existing alignment policy, not exact-duration trimming.
The native latent is `[1,24,T,68,120]`; the upscaled latent is
`[1,24,T,136,240]`. Upscaling doubles each spatial axis and quadruples video
rows. Text/reference and audio rows must also fit every downstream limit.

For development use **22 frames / 0.917 seconds / T=7** by default, and
**39 frames / 1.625 seconds / T=12** only after admission. The sole 56-frame
exception is continuation/bridge at no more than 1344×768: it needs 39
preserved context frames and a strictly longer target, so 56 is the minimum
valid target. Its 2.333-second duration still fits the user's limit. Do not
upscale that output or use 56 frames for ordinary/upscale acceptance.
Requesting 72 frames or three seconds rounds to 73/24 = 3.042 seconds and is
forbidden. Long-duration checks and continuation above 1344×768 are host-only
calculations or metadata checks, without full-size GPU allocation, sampling
or decoding. Upscaling to 3840×2176 is executed at 22/39 frames.

## Development matrix for the physical 72 GB device

A read-only check on 2026-11-22 reported an RTX PRO 5000 72 GB with
**73,415 MiB total (71.69 GiB)** and **72,787 MiB free (71.08 GiB)**.
Host memory was approximately 187.5 GiB total and 182.2 GiB available.
These are snapshots; recheck both memories before each job. No new render
was executed for this check.

The same conservative formula below gives the following rounded capacities
for allowed test workloads, before extra reference rows. Native includes
512 text rows and the appropriate audio rows; refinement uses the equivalent
target geometry. These capacity budgets remain distinct from measured peaks.

| Workload | Frames | Video T / target rows | Resident / fully streamed capacity |
| --- | ---: | --- | ---: |
| Native 1920×1088 | 22 | 7 / 14,280 | 49 / 11 GiB |
| Native 1920×1088 | 39 | 12 / 24,480 | 53 / 15 GiB |
| Upscale refinement to 3840×2176 | 22 | 7 / 57,120 | 65 / 27 GiB |
| Upscale refinement to 3840×2176 | 39 | 12 / 97,920 | 80 / 42 GiB |
| Continuation/bridge target at 1344×768 | 56 | 17 / 17,136 | 50 / 12 GiB |

Learned transfer plus decode at 3840×2176 without refinement (K=0) has an
approximately **8 GiB** planning budget for either 22 or 39 frames, excluding
source generation. Use `H3_CUDA_WEIGHT_MODE=stream` for the required K=2/3/4
baseline and its resumed workers. Native can use auto placement, selecting
explicit streaming before launch if actual conditions need it. Resident
maximum-canvas refinement exceeds the 60 GiB development budget at both
durations, so it is not a required run. Optional auto-placement comparisons
must independently fit the budget. Lower K reduces runtime but does not
remove the high-resolution activation requirement; K=0 is a separate path.

The completed short runs provide the following whole-device peaks, sampled
every 0.5 seconds. They include the device baseline and can miss brief peaks:

| Workload | Frames | Measured whole-device peak |
| --- | ---: | ---: |
| Native 1920×1088, six steps | 22 | 5.60 GiB |
| Native 1920×1088, two steps | 39 | 7.92 GiB |
| K=0, 2176×3840 full delivery | 22 | 5.42 GiB |
| K=0, 3840×2176 full delivery | 39 | 5.42 GiB |
| K=2/3/4, 3840×2176 full delivery | 22 | 15.67–15.71 GiB |
| K=2/3/4, 2176×3840 full delivery | 22 | 15.67–16.04 GiB |
| K=2, 3840×2176 full delivery | 39 | 24.71 GiB |

The initial 22-frame landscape K=0 delivery also passes but lacks a sampled
whole-device peak. Reference-heavy 22-frame K=2 reaches 20.67 GiB. Cached
native/refined jobs use approximately 37 GiB process RSS; the 39-frame K=0 job
uses 7.93 GiB. These measurements do not justify lowering the planning reserves.
The 39-frame portrait and K=3/K=4 extensions remain unrun; all required
22-frame orientations and refinement levels pass.

Set a **60 GiB required-capacity ceiling for development jobs**, including
their modeled reserve, and independently enforce the real allocator's free-
memory check. The physical card's 10% reserve is about **7.17 GiB**; that
actual reserve, current live allocations and other processes must be used at
launch, rather than assuming the capacity table's scaled reserve is the card's
reserve. Include actual reference/text rows, context caches and each phase's
working buffers. Host cache/working-memory allowances below fit this server's
reported RAM, but remain subject to the host guard. Run GPU jobs serially and
release completed phase resources.

If a 39-frame case does not pass preflight, select a 22-frame source/target
before running it. Keep required quality/reference/lifecycle cases at 22
frames. Preserve full depth, requested K and references. An OOM is not an
accepted result. The required allowed-resolution cases must work on this
card; larger hardware is not a prerequisite. The test guard must reject
continuation above a 1344 long axis, 768 short axis and 1,032,192 pixels even
when memory would fit. Upscale admission instead allows the full 3840×2176
target and transpose, but only at 22/39 frames and within the memory/index
budget. Apply those rules consistently to K=0, direct API/operator execution,
resume and decode-only work. Do not confuse the continuation cap with the
separate upscale ceiling.

## Conservative VRAM capacity budgets

The following budgets retain the current conservative weight-admission formula,
allow 2 GiB for other live device allocations at planning time, and reserve
10% of device capacity. The checked shared estimator is implemented in
[`src/geometry.h`](../../src/geometry.h); reproduce these values with
`make test-large-canvas && bin/large_canvas_tests --report`. The extended
profiles admit these shapes arithmetically, then perform separate memory
admission. Long-duration execution remains untested. Round upward to whole GiB; do not interpret them as measured
minimums or as physical GPU recommendations.

| Requested duration | Native generation: resident / fully streamed | 2× upscale K=0, later job only | 2× upscale K=2/3/4: resident / fully streamed |
| --- | ---: | ---: | ---: |
| 5 s | **72 / 34 GiB** | **8 GiB** | **156 / 118 GiB** |
| 10 s | **99 / 60 GiB** | **13 GiB** | **262 / 224 GiB** |
| 15 s | **125 / 87 GiB** | **18 GiB** | **368 / 330 GiB** |

K=0 excludes generating its source: an end-to-end generation→upscale workflow
needs the larger of its native-generation and K=0 budgets. The refined upscale
already dominates its source-generation budget. Partial residency lies between
the resident and fully streamed cases. Streamed weights do not remove the
large target activations. The present 72 GB CUDA server cannot be assumed to
admit any of the 5–15 second **refined** 3840×2176 cases under this planner;
the 15-second native case also exceeds that device's streamed planning budget.
The implementation must report resource limits instead of silently shrinking
the canvas, dropping refinement or shortening the video.

### Reproducible transformer capacity calculation

From [DiT dimensions and allocation planning](../../src/denoise/dit.c):
`HIDDEN=5376`, `INNER=56*128=7168`, `FFN=14336`.

```text
S = video_rows + 2*A + 512                 # no reference rows
block_bytes = 2*(4*7168*5376 + 3*14336*5376)
resident_core = 50*block_bytes = 35.888671875 GiB
streamed_core = 2*block_bytes = 1.435546875 GiB
future = S*2*(12*5376 + 8*7168 + 4*14336) + 1 GiB
       = S*358400 bytes + 1 GiB
capacity_GiB = ceil((future_GiB + core_GiB + 2) / 0.90)
```

The real [weight planner](../../src/weights/residency.c) and
[CUDA wrapper](../../src/cuda/gpu_cuda.cu) use current free memory and reserve
`max(1 GiB, device_total/10)`; 2 GiB is an explicit planning allowance for other
live allocations, not a value hardcoded in those functions. Text/reference
encoding, driver workspaces, other processes and allocator fragmentation can
require more. The current planner is deliberately more conservative than
recorded peaks. Any tighter estimate must follow an allocation-lifetime audit
and short-run measurements, not simply lower the guard to admit a request.

### Empirical peak cross-check

The [existing 90-frame comparison](../experiments/latent-upscale-results.md)
records direct dense BF16 peaks of 44.70 GiB at 1344×768 and 51.33 GiB at
1920×1088. K=4 refinement peaks are 43.80 and 50.43 GiB. The video-row counts
are 27,216 and 55,080. A simple linear extrapolation gives:

```text
slope = (51.33 - 44.70) / (55080 - 27216) GiB per video row
direct_peak = 38.224186 + slope*video_rows
refinement_peak = 37.324186 + slope*video_rows
fully_streamed_peak ≈ resident_peak - 35.888672 + 1.435547
```

| Requested duration | Native estimated peak: resident / streamed | Refined 2× estimated peak: resident / streamed |
| --- | ---: | ---: |
| 5 s | 56 / 22 GiB | 109 / 75 GiB |
| 10 s | 73 / 39 GiB | 177 / 143 GiB |
| 15 s | 90 / 56 GiB | 245 / 211 GiB |

These lower numbers **exclude extra capacity headroom**. They extrapolate two
older measurements with one prompt/seed; text/audio differences, allocator
behavior, kernel dispatch and row-batching boundaries are not modeled. In
particular, streaming subtraction is a hypothesis supported by the weight
sizes, not a new long-duration streamed measurement. Use the conservative
capacity table for initial admission planning. Neither table qualifies long
videos under the three-second test restriction.

### Learned transfer without refinement

The [existing network](../../src/upscale/upscale_network.c) keeps three
512-channel BF16 feature volumes at the target resolution, plus the pinned
approximately 0.643 GiB model, small convolution packing buffers and an output
latent. For `V = T*136*240*512*2 / 2^30` GiB:

| Requested duration | One feature V | Approximate explicit transfer tensor/weight peak | Existing plan reserve `4*V + 2` |
| --- | ---: | ---: | ---: |
| 5 s | 1.152 GiB | 4.16 GiB | 6.61 GiB |
| 10 s | 2.241 GiB | 7.48 GiB | 10.96 GiB |
| 15 s | 3.331 GiB | 10.80 GiB | 15.32 GiB |

The explicit estimate is `3*V + model + target_BF16_latent + packing`, without
backend/context overhead. The [upscale plan](../../src/upscale/upscale_plan.c)
uses the larger `4*V + 2 GiB` reserve. The previous CUDA K=0 job peaked at about
6.01 GiB including full VAE decoding at the smaller target. For the capacity
table, allow a provisional 7 GiB decoder/runtime budget, take
`ceil(max(7, 4*V+2)/0.90)`, and verify the new canvas with short full-decode
tests. These are separate phases; adding their peaks would double-count memory.

## Host RAM and decoded output

Use separate host allowances: approximately **8 GiB** application working
memory for native generation and **16 GiB** for the upscale job, plus optional
streamed-weight caching of up to **40 GiB** and at least **10 GiB** physical
headroom. These are conservative planning allowances, not measured minima.
They imply roughly 18/26 GiB without that cache and 58/66 GiB with it.
Practical budget envelopes for each of the 5/10/15-second cases are therefore
**24–32 GiB host RAM without the weight cache**, **64 GiB for cached native
generation**, and **80–96 GiB for cached refined upscaling**. Cache admission
is optional and may use less; installed RAM and maximum process RSS differ.

The duration-independent decoder allowance requires
[streamed delivery](../../src/media/delivery.c): the current full VAE emits
up to 22 frames per chunk with five overlap frames. At 3840×2176, the default
256/64 spatial plan has 20×11 tiles. Retained tile RGB, one stitched F32 chunk
and temporal overlap alone total approximately 6.1 GiB of host storage before
other decoder, latent, RGB8 and encoder buffers. Bound this lifetime explicitly.
One upscaled F32 latent costs only 0.108/0.210/0.312 GiB for the three durations,
but sampler/transfer/state copies must still be included.

Retaining a complete F32 RGB movie would add the following, instead of the
bounded chunk behavior assumed above:

| Requested duration | Entire 1920×1088 F32 RGB movie | Entire 3840×2176 F32 RGB movie |
| --- | ---: | ---: |
| 5 s | 2.89 GiB | 11.58 GiB |
| 10 s | 5.67 GiB | 22.69 GiB |
| 15 s | 8.45 GiB | 33.80 GiB |

The [process guard](../../src/memory.c) remains applicable independently of
VRAM; see [hardware accounting](../hardware-requirements.md). Do not route
normal high-resolution CLI delivery through a whole-movie allocation, or add
cached source-generation weights to a supposedly independent K=0 process.

## Checked geometry, indexing and delivery

The [implemented contract](large-canvas-contract.md) separates arithmetic
validity from memory admission:

- CUDA source axes extend to 1920 and learned output axes to 3840, admitting
  **240×136** latent output and transpose. Metal retains its existing bounds.
- The old `5376*3` QKV proxy is removed. The actual width is **7168*3**;
  dense CUDA uses wide QKV addresses and checked launch limits. MLP FC1/SwiGLU
  work exceeding U32 elements uses independent row slabs with complete K
  dimensions. Global attention and full-context normalization are preserved.
- With the table's audio/text allowance, native QKV element counts are
  **1,643,034,624 / 3,186,935,808 / 4,730,836,992**. The 15-second case exceeds
  UINT32_MAX. Upscaled counts are **6,512,400,384 / 12,662,458,368 /
  18,812,516,352**: all three exceed it. FC1 outputs and other intermediates
  receive separate checks; even a native 10-second case is close to U32 FC1
  elements before references. These long boundaries are tested with host
  arithmetic and logical offsets, without long-video GPU allocations.
- CUDA's fast VAE stitcher selects its bounded host fallback for more than
  16 tiles per axis, before accessing fixed device arrays. The conservative
  shared host estimate is 3.644 GiB for native and 8.253 GiB for upscaled
  delivery, including tiles, chunk/overlap, conversion and encoder allowance.
- Existing reference encoders and optional modes have independent limits.
  Admit them from their actual rows and live allocations; do not infer support
  from target pixels alone.

These tables were regenerated from the final shared planner and compared with
the allowed 22/39-frame CUDA measurements and the bounded continuation case.
Maximum-resolution transfer/refinement/decode passes at 22 frames, with the
admitted 39-frame extensions identified above. Longer shapes/counts receive
host-only checks. Keep 5/10/15-second entries and continuation above 1344×768 labeled
**estimated and not end-to-end tested** until the user changes the corresponding
restriction.
