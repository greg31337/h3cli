# CUDA large-canvas acceptance

Date: **2026-11-22**. **All 24 tasks are complete under the bounded acceptance
contract.** Validation gaps and test corrections are disclosed below. See the
[implementation contract](large-canvas-contract.md),
[archived task list](../large-canvas-todo.md) and
[memory derivation](large-canvas-memory-estimates.md).
The [per-case measurements](large-canvas-metrics.json) contain 116 passing
guarded case records and three separately identified failed attempts that were
corrected and rerun, plus the subsequent **204/204 immutable regression pass**.
Host checks and macOS legacy renders are recorded below.

## Scope and hardware

The implementation adds ordinary CUDA profile 2 through 1920×1088 and learned
2× profile 3 through 3840×2176, including transpose. It preserves the pinned
network, full-context normalization, global attention, all 50 transformer
blocks and exact source audio. K=0 skips refinement; K=2/3/4 retain their full
requested schedules. CUDA tests explicitly stream BF16 transformer weights.

The test device is an NVIDIA RTX PRO 5000 72GB Blackwell (SM120), driver
595.99.02, with 73,415 MiB total VRAM (71.69 GiB),
with 72,787 MiB initially free. Host physical RAM is approximately 187.5 GiB.
Each guarded job checks current free memory, a physical-device reserve of
max(1 GiB, total/10), at least 32 GiB available host RAM, and a conservative
capacity estimate below 60 GiB. GPU jobs on this device run serially.
After the final checks, no campaign workers remain and free VRAM returns to
72,787 MiB, matching the initial snapshot.

The development checkout starts from
`2b78e5944d7d7903983710d981c29d7401da81a6`; changes are uncommitted.
The final Linux package snapshot SHA-256 is
`027f035b5941e4d60050ab18b2483608ea6c12dc071d79316bbaa58da3fb2235`.
Its 216 production source files match this workspace, including the final
explicit API rejection of LoRA on extended native resume. Report-only edits
follow that snapshot.
The isolated server snapshot has no Git metadata, so the build
records its commit as `unknown`; the snapshot hash identifies the actual input.
The final standalone binary SHA-256 is
`c40b1d3be70237c58d87403ec9a15cb236b20a8f7b949a5baa3d6102f9073055`.
It was built offline with GCC 11.4.0, CUDA 13.0.88, FFmpeg 9.0.2 and the pinned
SGLang/Sage/SOL/SubBlock/cuDNN dependencies. Its launcher tests pass (16 tests,
one skip because the builder lacks a `/dev/shm` noexec mount). Offline short
native render/pause/resume, maximum K=2 upscale/pause/resume and saved-state
decode all pass, with exact uninterrupted/resumed AV comparisons. Embedded
CUDA code covers SM86/89/90/100/120; this campaign tests SM120 only.
Private commands, paths, device identifiers and raw logs remain under ignored
`outputs/large-canvas/`.

| Test executable | SHA-256 |
| --- | --- |
| Final Linux `h3cli` | `0dfe99272591ab2a223f3e5765c33a8b1476a08ca1420425e3aa91bb723fe9c6` |
| Final API rejection probe | `cf85ec34f008d180870ee7db2b39c02d7ad6f7f37e2fbce94639b0a634049233` |
| Linux `h3cli`, mixed-reference/replay build before final resume guard | `fe48fcd51bd22553463a8bd49fbe47b8468a68cf7b7257c0daa86cabd4b2fd19` |
| Linux lifecycle probe | `f34380812ae339c87568f336b5b92bc8022da54b65058e9041888b8d302f3446` |
| Linux condition export probe | `d8cdc9aecf851f9f312c0b54aaf25cea8719e4b3345e39f04070e21eaa6142f0` |
| Final macOS `h3cli` | `1578e462e378c079d52b910a8e34c4789be9cdaedf1cdaad1775a21cd4e6454b` |

The matrix was collected during implementation. Early-validation guards were
added after the initial matrix and K=4 lifecycle build; they do not change
the arithmetic of these accepted 22/39-frame cases. The K=2/K=3 lifecycle
build precedes the final API LoRA-resume rejection guard. The replacement
standalone package and targeted API rejection probe cover that final guard.
These short matrix runs are separate from the subsequent immutable gate,
which validates the final standalone artifact as recorded below.

The native source's local model-metadata identity is
`098cfd26c53e143b2f73ad588e7292b2d3b026a3abbaba73dc804f7bed285087`
and its AV-metadata identity is
`617d923095290e644d0e51c1847112de4838dd5f186965104d1bfc9395e1e505`.
These are the application's local metadata fingerprints, not full transformer
weight hashes. The learned artifact and independent source have separately
verified content hashes below.

## Completed maximum-canvas matrix

Every row below delivers 22 frames at 24 FPS, 0.917 seconds. Each upscale row
starts from a native maximum-canvas source. Times include cold model work and
full MP4 delivery; they are observations, not stable performance benchmarks.
Whole-device peaks are sampled by `nvidia-smi` every 0.5 seconds, including the
device baseline. Tracked tensor peaks omit driver and external allocations.

| Operation | Output | Seconds | Whole-device peak GiB | Tracked peak GiB |
| --- | --- | ---: | ---: | ---: |
| Native, six steps | 1920×1088 | 100.9 | 5.60 | 5.01 |
| Native, two steps | 1088×1920 | 84.7 | 5.44 | 4.98 |
| K=0 | 2176×3840 | 49.7 | 5.42 | 4.98 |
| K=2 | 3840×2176 | 162.2 | 15.67 | 15.16 |
| K=3 | 3840×2176 | 203.2 | 15.71 | 15.18 |
| K=4 | 3840×2176 | 246.1 | 15.71 | 15.20 |
| K=2 | 2176×3840 | 165.9 | 15.67 | 15.16 |
| K=3 | 2176×3840 | 210.8 | 15.71 | 15.18 |
| K=4 | 2176×3840 | 252.3 | 16.04 | 15.20 |

The initial landscape native and K=0 cases also passed (112.0 and 40.0 seconds),
but preceded the whole-device sampler; they do not supply a whole-device peak.
Native/refinement processes retained about 36.8 GiB RSS, principally the
streamed host weight cache. K=0 portrait RSS peaked at about 6.8 GiB. Sampling
may miss short peaks. The final runner sums descendant RSS for script/server
workers; shared pages can be counted more than once. Earlier script RSS measured
only its launcher and is not a complete process-tree peak.

The conservative short budgets remain 11/15 GiB native streamed and 27/42 GiB
refined upscale streamed for 22/39 frames. The measured short peaks do not
justify lowering reserves or claiming the long-duration estimates as measured.

The admitted 39-frame landscape extension also passes, delivering 1.625 seconds
at 24 FPS. It exercises decoding across a temporal chunk boundary:

| Operation | Output | Seconds | Whole-device peak GiB | Process-tree RSS GiB |
| --- | --- | ---: | ---: | ---: |
| Native, two steps | 1920×1088 | 94.1 | 7.92 | 36.81 |
| K=0 | 3840×2176 | 74.5 | 5.42 | 7.93 |
| K=2 | 3840×2176 | 325.7 | 24.71 | 36.85 |

The 39-frame portrait and K=3/K=4 extensions are not run. Their required
22-frame counterparts pass. Reference-heavy cases stay at 22 frames. Ordered
maximum-size images produce 39,728 actual native sequence rows and 82,568
upscale rows; the refined upscale peaks at 20.67 GiB whole-device usage and
37.15 GiB process-tree RSS. Native `match`, `high` and `max` fixtures retain
their respective reference preprocessing; changing this flag never changes
the requested video canvas.

Delivered 22-frame video lasts 0.916667 seconds and its audio lasts 0.925
seconds, a difference of 8.33 ms (less than one video frame). At 39 frames,
both streams last 1.625 seconds. Streams start at zero. Source/target latent
audio hashes match in every upscale matrix case; the API test also compares
decoded PCM exactly.

## Independent arithmetic evidence

The frozen artifact SHA-256 is
`4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6`;
the upstream source SHA-256 is
`744063b43e0f3eec23e2485cb7c65503069946ca9690906ecb548d7515cb89e2`.
The independent oracle uses PyTorch 2.13.0+cu130, CUDA 13.0 and cuDNN 9.20,
BF16, TF32 disabled, full-volume normalization and chunking disabled.
Native and oracle GPU forwards run separately under the same campaign lock.

| K=0 transfer | Relative RMS error | Maximum error / reference RMS | Result |
| --- | ---: | ---: | --- |
| 1920×1088 → 3840×2176, T=7 | 0.00298317 | 0.0497083 | Pass |
| 1088×1920 → 2176×3840, T=7 | 0.00334815 | 0.0377568 | Pass |

The unchanged frozen network tolerances are 0.04 and 0.35, respectively.
Both comparisons also verify finite target values and exact source audio bytes.
All 210 bounded adversarial operator comparisons pass, covering normalized
input/output, interpolation and isolated layers. The worst end-to-end relative
RMS error is 0.0112999 and maximum error/reference RMS is 0.100569. Raw-condition
comparisons also pass for `match`, `high`, `max`, first/last anchors and mixed
video/audio/image references. The worst learned condition comparison has
relative RMS error 0.00416132 and maximum error/reference RMS 0.0615154.
High/max image tensors and the mixed case's fixed-size video/image tensors are
byte-identical across retargeting; all condition audio is byte-identical.
The second `match` image grows from latent 104×78 to 114×86, demonstrating
the upstream target-dimensions scale embedding rather than assuming every
reference grows by exactly 2×.

## Full immutable regression follow-up

On 2026-11-22, the user explicitly requested the complete 204-output regression,
authorizing its unchanged 124-frame render after the bounded campaign.
The final standalone binary and its matching packaged probes pass **204/204
exact golden hashes**, with no missing or unexpected artifacts. Collection,
rendering, decoding and comparison took **101.066 seconds**; total elapsed time,
including artifact identity checks, was **122.420 seconds**.

This runs the full 640×480, 124-frame, six-evaluation C0 case and every component
probe, using the hash-verified historical FFmpeg 4.2.2 encoder and 6.1.1 decoder.
The runner checks source, fixture, runtime and tool identities before and after
execution. Goldens and fixtures remain unchanged. The final executable is the
`c40b1d3b…` standalone artifact identified above.

The local workspace, server checkout and packaged probes share regression
source fingerprint
`55adfee5a04f1933bddaf87828b33b8004a55ad7b5fe6acdb02622448b31c4af`.
The private result record SHA-256 is
`7529c66bb19fc91712e0ad924e5b00b7748a89a7714299cb1dc1c79d4cc33dbf`;
its sanitized summary is in the [measurement record](large-canvas-metrics.json).
This closes the earlier immutable-gate gap. It does not qualify 5/10/15-second
videos at the new maximum canvases or continuation above 1344×768.

## Other completed checks

- Native small-canvas seed repeats and pause/resume produce identical AV values.
- Native maximum full-VAE replay, preview output and preview callback
  cancellation pass. All reported dimensions/frame counts come from FFprobe.
- Two independent 22-frame maximum-canvas script segments merge to 44 frames
  (1.833 seconds). Worker diagnostics confirm inherited `max` and overridden
  `high` reach actual image preprocessing.
- Native hard/bridge continuation passes using a 256×128, 39-frame source and
  a 56-frame target retaining the 39-frame prefix. Both decode-only replays pass.
- Public API native → K=2 upscale → native reuse passes with identical seeded
  AV values and source/target PCM. Free VRAM is exactly 71,746,256,896 bytes
  after both native calls; RSS changes from 39,456,595,968 to 39,431,405,568
  bytes. Injected memory denial returns a resource error.
- HTTP maximum-canvas video and legacy still jobs complete; an oversized
  canvas is rejected before a worker is launched.
- Ordered `match`, `high`, `max` references and first/last anchors pass native
  generation and maximum K=2 upscale after deleting their original files and
  relocating each clean source bundle.
- A mixed ordered video/audio, standalone audio and image case passes native
  generation and maximum K=2 upscale. Its video fixture has 39 frames and its
  standalone audio fixture is two seconds; output has 22 frames. The upscale
  uses a relocated bundle after all original reference files are deleted and
  preserves the generated source audio exactly.
- Maximum K=2/K=3/K=4 landscape and portrait checkpoints match the uninterrupted AV
  trajectory exactly at every refinement boundary. Cancellation at step 1,
  checkpoint recovery and full saved-state decoding also pass.
- A conditioned maximum landscape K=2 trajectory matches at both resumed
  boundaries after deleting the original images and relocating the source.
  This checks that saved reference augmentation is not repeated on resume.
- Maximum full-VAE cancellation and injected encoder failure preserve the
  original destination; subsequent 22- and 39-frame decodes succeed with
  FFprobe-confirmed 3840×2176 dimensions and exact frame counts. A deterministic tensor
  allocation denial recovers without exhausting physical VRAM. CUDA memcheck
  passes on repeated full-context learned-network forwards using a small
  `[1,24,7,2,4]` input. Sanitizer coverage is bounded to this operator fixture;
  maximum canvases are covered by ordinary runtime tests.
- Local macOS builds and the bounded host target pass. Apple M4 Max, 128 GiB,
  legacy Metal native 256×128 and learned K=0 512×256 delivery pass at 22 frames,
  with identical audio bytes. New CUDA profiles are rejected by Metal host/CLI
  and container checks.
- The final standalone package passes offline native 1920×1088 generation and
  exact resume, then 3840×2176 K=2 upscale and exact resume, all at 22 frames.
  Its full saved-upscale decode and source-audio identity checks pass.
- Final-source bounded host suites pass on both macOS and Linux. The targeted
  API probe loads an extended native checkpoint and rejects an attached LoRA
  before adapter preparation or transformer loading.

## Visual observations

Frame 11 of the six-step native landscape output shows a recognizable red
teapot, coherent lighting and tabletop texture. Maximum landscape and portrait
K=4 frames are recognizable but soft, with distorted small details inherited
from their two-step source fixtures. No obvious tile borders are visible in
these inspected frames. Portrait composition leaves substantial empty space.
These observations are separate from numerical and lifecycle acceptance;
they do not establish temporal quality or normal-schedule production quality.

## Task acceptance record

All tasks pass the bounded criteria in the checklist. The subsequently
authorized immutable gate also passes. Long videos at the new maximum canvases
and larger continuation remain unqualified.

| Task | Evidence | Status |
| --- | --- | --- |
| CLC001 | Frozen profile/backend contract, pinned network and unchanged archive/golden identities | Pass |
| CLC002 | Shared checked phase estimator; 22/39 and host-only 124/243/362-frame reports; measured short peaks above | Pass |
| CLC003 | Fixed serial runner, duration/geometry/source checks, capacity/current-free admission and bounded host target | Pass |
| CLC004 | Shared native/delivery policies; legacy Metal delivery and new-profile rejection | Pass |
| CLC005 | CLI/API/HTTP and two-segment `--render`; actual inherited/overridden reference-size worker diagnostics | Pass |
| CLC006 | AV/sampler/source/presentation round trips, malformed and forged profile rejection | Pass |
| CLC007 | Checked sequence/text rows, signed GEMM dimensions, wide offsets and logical overflow boundaries | Pass |
| CLC008 | Independent MLP row slabs preserve complete dot products; host boundary/dispatch checks | Pass |
| CLC009 | Actual conditioned sequence accounting, ordered references and first/last anchors; unqualified modes rejected | Pass |
| CLC010 | Streamed BF16 maximum K=0/2/3/4 acceptance, separate host/device admission and measured headroom | Pass |
| CLC011 | Maximum native entry points and independent script merge; bounded hard/bridge continuation; large host-only continuation | Pass |
| CLC012 | Maximum native full/preview decode, replay, portrait and preview cancellation | Pass |
| CLC013 | Maximum clean-source capture; deleted original references and relocated bundles used for K=2 | Pass |
| CLC014 | Maximum learned transfer in both orientations; independent full-context oracle within frozen tolerances | Pass |
| CLC015 | Image/video/keyframe raw-condition retarget oracle, ordering, audio and conditioned resume | Pass |
| CLC016 | All refinement schedules, exact checkpoints at each boundary, cancellation/recovery and state-only API | Pass |
| CLC017 | Maximum full-VAE delivery in both orientations, 39-frame cross-chunk case, saved decode and failure recovery | Pass |
| CLC018 | Backend/shape/container/index/admission and duration-guard host tests on macOS/Linux | Pass |
| CLC019 | Native/reference/anchor/API/HTTP/script/continuation matrix, mixed references and visual observations | Pass |
| CLC020 | Maximum transfer oracle plus bounded adversarial operators and raw-condition comparisons | Pass |
| CLC021 | Maximum K=0/2/3/4 landscape/portrait, admitted 39-frame K=0/K=2, every refinement boundary | Pass |
| CLC022 | Reused-context plateaus, memory denial, allocation/encoder failure, cancellation and CUDA memcheck | Pass |
| CLC023 | Bounded host/legacy CUDA checks, macOS compatibility, offline standalone native/upscale/decode/resume and subsequently authorized 204/204 immutable gate | Pass |
| CLC024 | Guides, final memory tables, source/package identities, task evidence, links and privacy review | Pass |

## Reproduction and limits

Use installed offline models and the configured CUDA build environment. Fresh
output paths are required. The wrappers enforce frame, geometry, memory and
serialization guards before each launch:

```sh
make all test-large-canvas-host bin/upscale_lifecycle bin/upscale_network \
  bin/large_canvas_api bin/large_canvas_conditions bin/large_canvas_recovery
python3 tests/large_canvas_guard.py native --model /path/to/MiniMax-H3 \
  --frames 22 --output outputs/large-canvas/example-native.mp4
python3 tests/large_canvas_guard.py upscale --model /path/to/MiniMax-H3 \
  --source outputs/large-canvas/example-native.h3up \
  --upscale-model /path/to/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors \
  --frames 22 --refine 4 --output outputs/large-canvas/example-upscale.mp4
bin/large_canvas_tests --report
python3 tests/large_canvas_campaign.py package \
  --binary /path/to/h3cli-linux-x86_64 --model /path/to/MiniMax-H3 \
  --weights /path/to/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors \
  --out outputs/large-canvas/example-package
```

The fixed `large_canvas_campaign.py` modes cover portrait, crosschunk, native,
references, continuation, script, API, HTTP, lifecycle, recovery, sanitizer,
operators, raw conditions, mixed references, the standalone package and the
early `lora-resume` rejection probe.
`--refine 2|3|4` selects the lifecycle schedule. Oracle modes need the pinned
upstream source and an independently provisioned Torch Python environment.
The existing operator generator accepts explicit `--short`; this changes its
fixture temporal length to at most T=7 without changing the frozen contract
or tolerances. The original long campaign is not invoked.

**Validation gaps and deviations:**

- The immutable gate was initially deferred under the three-second limit.
  The user's subsequent explicit request authorized the unchanged full gate;
  it now passes all 204 hashes as recorded above.
- No 5/10/15-second real-model videos or full-duration GPU tensors were run
  at the new maximum canvases. Their 124/243/362-frame geometry/index/resource
  cases remain host-only; the authorized 124-frame regression uses 640×480.
  Continuation above 1344×768 is host-validated only; its runtime quality,
  performance and memory remain unverified.
- Early invocations of the broad host suite inadvertently executed existing
  long synthetic FFmpeg fixtures, including 361/362-frame clips and merges.
  This violated the requested development duration cap. Those invocations are
  excluded from bounded acceptance. `test-large-canvas-host` now excludes both
  synthetic-media runners; subsequent real-model tests use the short guard.
- The first independent oracle invocation failed before GPU work because its
  Python environment lacked NumPy. The retained failure record is separate
  from the passing reruns with the complete Torch environment.
- The first mixed-reference fixture failed before denoising because standalone
  reference audio requires at least two seconds. Its audio fixture was padded
  to two seconds for a fresh rerun; the reference video remains 39 frames and
  the requested output remains 22. No production behavior was changed for this
  fixture correction.
- The first raw-condition oracle incorrectly supplied scale 2 for a `match`
  reference whose target was only about 1.10× its stored size. The corrected
  test follows the pinned upstream target-dimensions formula: the average of
  the height and width ratios. Production code and frozen tolerances are
  unchanged; the failed comparison is retained separately from the rerun.
