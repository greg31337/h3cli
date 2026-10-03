> Archived full-matrix plan. The user replaced its unfinished renders with a one-hour sample capped at two steps. See [current tasks](../todo.md).

# CUDA multi-reference testing tasks

Design: [CUDA multi-reference test campaign](design-multi-reference-testing.md).
Execution in progress, 2026-09-22–23. Initial CUDA build and host checks pass. Report bugs and
retain reproductions; **product fixes are on hold**. Completed single-still
work is preserved in [its archived plan](../single-still-tasks.md).

Campaign limits: CUDA node `cuda-pro6000`, models under
`/path/to/models/MiniMax-H3`; 640×480 and 1344×768; at most ten evaluations
per render; video-bearing outputs cover the aggregate input duration, rounded
up to the next supported frame count: 73 frames / 3.0417 seconds for one
three-second clip, 158 frames / 6.5833 seconds for two three-second or three
two-second clips. Image-only outputs retain 226 frames / 9.4167 seconds.
Controls match each output length. At most nine images, three videos and
**six seconds of video input in total** per request.
Image-bearing cases require both `match` and `max`.

Added scope: **nine images and 362 output frames at both 640×480 and
1344×768**, both sizing modes, ten steps. These four extra reference cases and
six matching controls produce 15.0833-second clips and are the explicit
exception to the original ten-second output cap.

The user raised every final matrix run to **`--steps 10`** before main rendering began.
The original six-step manifests are retained; `final-main.json` freezes the
76 replacement ten-step definitions with separate baseline keys. Two-step
smoke runs remain preflight evidence.

The default video pipeline requires two seconds per clip. The revised
six-second budget supports proper three-video generation with three 48-frame
clips. All video cases now test the default `released-v1` pipeline; the earlier
planned rejections and legacy diagnostics are replaced with generation tests.

Initial evidence: `outputs/multi-reference-cuda/` contains the source/model/asset
identities, frozen 76-case base manifest plus ten-case 362-frame extension, build log, 175,340 existing Ref2VA host checks,
236 modality-tokenizer checks and twelve runner/audit checks. All ten smoke cases now
have validated complete outputs. The maximal 1344×768 nine-image/three-video
case passed its authorized retry at 975.71 s / 52.33 GiB; its earlier 900.28 s
timeout during VAE decode remains retained as a separate attempt. A few NVML
sampling gaps exceed 100 ms and are explicitly reported. All 33 regular 640×480 final ten-step
renders have passed; the 1344×768 matrix is now running. The ten additional
362-frame attempts remain scheduled after that matrix. All three 640×480
control groups have <1% wall-time spread; the unchanged X91-max repeat
produced an identical MP4 with a 0.63-second wall difference. The maximal
nine-image/three-video case took 1,256.70 s and peaked at 46.67 GiB. Test-monitor corrections are retained
separately; no product fixes.

## M0 — Freeze the environment and fixtures

- [x] MR001 Re-probe the supplied CUDA node, GPU UUID/model/VRAM, CUDA/driver,
  disk and host/cgroup limits; record availability and exclusive GPU use.
- [x] MR002 Upload a frozen source snapshot into `/path/to/h3-multi-reference`,
  build for the detected GPU, record source/diff/binary/toolchain hashes and
  verify both original model routes without modifying model files.
- [x] MR003 Inventory and checksum the nine existing image assets; inspect and
  freeze their order/roles and 1/3/6/9 subsets. Record source and resolved
  `match`/`max` geometry at both target resolutions and any coverage gap.
- [x] MR004 Prepare distinct V1/V2/V3 silent clips (1×3 s, 2×3 s, 3×2 s)
  plus the bounded A1 embedded-audio variant. Verify actual decoded frames,
  all stream durations and aggregate duration ≤6 s; retain source provenance.
  Each V3 clip must retain exactly 48 normalized frames, never fewer.
- [x] MR005 Freeze the prompt, seed, dense BF16/full-VAE settings, residency
  policy, media encoder settings, frame limits, reference order and all matrix
  IDs and duration-derived frame counts in a machine-readable manifest.
  Enumerate 56 core + 10 supplemental main attempts, ten added 362-frame
  attempts and up to ten two-step smoke attempts at 73/158 frames. Preserve
  the original manifest and freeze the addition in `extension-362.json`.
- [x] MR006 Verify that the default validator accepts all planned durations,
  including three 48-frame clips, using bounded host checks. Check separate
  `<Picture N>`/`<Video N>` numbering, count limits, released VAE geometry and
  73/158/226-frame alignment. Record MR-L001 as resolved by the revised budget.

## M1 — Implement bounded measurement and reporting tools

- [x] MR007 Add a serial CLI runner with fresh processes, exact argv/env
  records, per-attempt directories, incremental JSON/CSV and resumable state
  that rejects stale source/configuration/asset results.
- [x] MR008 Enforce the ten-evaluation, ten-second output, reference-count and
  six-second aggregate-input limits before launch. Derive video-bearing
  output length from supplied durations; use 226 frames for image-only main
  cases and 73/158 for smoke. Permit 362 frames only for the explicit nine-image
  extension and its no-reference controls. Reject other oversized requests.
- [x] MR009 Measure full CLI wall time on the node using a monotonic clock,
  including loading through final mux/teardown. Exclude transfer/build and
  postprocessing; never use cumulative CUDA phase marks as duration evidence.
- [x] MR010 Add external NVML telemetry at 50 ms target / 100 ms maximum:
  device and process VRAM, idle usage, UUID, timestamps, gaps and sampled
  peaks. Record RSS/swap/cgroup separately and invalidate missing GPU metrics.
- [x] MR011 Add bounded timeouts, resource headroom checks and process-group
  cleanup; retain existing memory guards. Wait for GPU release between jobs,
  stop affected unsafe branches and preserve rejected/aborted/partial records.
- [x] MR012 Implement baseline lookup and wall delta/ratio/percent plus VRAM
  delta calculations. Retain all B00 samples, expose >10% timing spread,
  distinguish output lengths and smoke/bound-prompt comparisons; exclude
  failures from successful-render ratios. Verify calculations and failure handling with
  small synthetic records before GPU work.

## M2 — Run bounded smoke checks

- [x] MR013 Run smoke cases at 640×480, then 1344×768: B00-F073, I09-max
  and V01 at 73 frames; B00-F158 and X93-max at 158 frames. Use two steps
  throughout, measure against the matching-length smoke B00, and retain
  outputs and capacity failures with exact settings.
- [x] MR014 Validate telemetry coverage, completed evaluation counts, decoded
  media and cleanup. Freeze finite main-run timeouts and capacity thresholds
  from these observations; document blocked branches without altering cases.

## M3 — Execute the default pipeline matrix

- [ ] MR015 Run three ten-step B00 controls for each resolution/frame-count
  pair (73, 158, 226 and 362 frames), bracketing matching candidates at beginning,
  middle and end. Record each wall time, peak VRAM and output; compute the
  matching medians and variability.
- [ ] MR016 Run I01/I03/I06/I09 at both resolutions and both sizing modes
  at 226 frames (16 candidates), plus I09-F362-match/max at both resolutions
  (four added candidates). Retain wall/VRAM comparisons, geometry and media.
- [ ] MR017 Run V01/V02/V03 at both resolutions (six candidates) through the
  default pipeline, using 73 frames for V01 and 158 for V02/V03. Retain
  completed outputs, wall/VRAM comparisons and any actual failures.
- [ ] MR018 Run X11/X91 at both resolutions and both sizing modes (eight
  candidates). Capture the practical cost of one video plus 1/9 images.
- [ ] MR019 Run X32/X93 at both resolutions and both sizing modes (eight
  candidates), at 158 output frames with six seconds total video input.
  Retain completed mixed-reference outputs, baseline comparisons and any
  actual capacity or generation failures.

## M4 — Verify reference consumption, audio and ordering

- [ ] MR020 Audit the existing multi-video run records to confirm every input
  reaches preprocessing/conditioning with the expected normalized/VAE/latent
  geometry: 72/56/17 for V1/V2 clips and 48/39/12 for V3. Verify reference
  order and count; report dropped/truncated inputs or missing diagnostics.
  This audit adds no renders and must not switch to the legacy pipeline.
- [ ] MR021 Run A01/A11 at both resolutions (four cases, A11 uses `match`)
  through `--ref-video`; verify embedded audio handling and compare to the
  corresponding silent case and no-reference control.
- [ ] MR022 Run O03 and O91 at both resolutions (four cases), verifying image
  reversal/interleaved modality ordering and explicit prompt bindings. Record
  the binding-clause difference from B00 in every timing comparison.
- [ ] MR023 Repeat X91-max once per resolution after intervening jobs; retain
  all timings, outputs and GPU-release traces. Report repeat instability or
  retained-memory anomalies without adding automatic unlimited retries.

## M5 — Deliver evidence and report bugs without fixes

- [ ] MR024 Fully decode every output and verify dimensions, frame count,
  24 fps, audio/container validity and authorized duration: ≤10 s in the base
  matrix; 362 frames / 15.0833 s in the added group, with at most one frame of
  mux/audio rounding. Preserve failed partial
  outputs and distinguish heuristic visual warnings from confirmed failures.
- [ ] MR025 Maintain `bugs.json`/`bugs.md` with stable IDs, severity/category,
  exact reproductions, hashes, observed/expected behavior and supporting logs
  or media. Mark fixes deferred; keep known duration rules separate from bugs.
- [ ] MR026 Create `review.html`, `review-640x480.html` and
  `review-1344x768.html` with candidate/no-reference playback, input previews,
  contact sheets, settings, wall overhead and sampled peak VRAM. Clearly label
  smoke, rejected and incomplete cases, and show each duration group.
- [ ] MR027 Download playback assets, fixture previews, manifests, metrics,
  logs and bug records into local `outputs/multi-reference-cuda/`; verify
  checksums and all relative gallery links without remote dependencies.
- [ ] MR028 Write the final per-case report and CSV with total wall time,
  baseline identity/time, overhead seconds/percent/ratio, peak VRAM and delta,
  status and output/bug links. Include absolute baseline metrics, all failures
  and telemetry limitations; avoid extrapolating from ten-step quality.
- [ ] MR029 Reconcile all 76 scheduled main attempts and up to ten smoke
  attempts, plus any separately recorded retries. Explain every blocked or
  unrun row; qualify default multi-video generation only where evidence exists.
- [ ] MR030 Close the test campaign with local gallery/report links, key
  time/VRAM findings and the deferred bug list. Completion requires an honest
  coverage account and retained evidence, not product bug fixes or human-review
  approval.
