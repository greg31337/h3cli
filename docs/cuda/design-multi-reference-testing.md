# CUDA multi-reference test campaign

Status: sampled closeout complete and evidence verified, 2026-09-23. Execution tasks: [todo.md](../todo.md).
This campaign measures image-only, video-only and mixed Ref2VA generation,
retains playable outputs, and reports defects. Product fixes are on hold.
The live results and playback pages are retained under
`outputs/multi-reference-cuda/`; the final coverage and transfer audits passed.

## Current execution amendment: one-hour remainder

The user replaced the unfinished ten-step matrix with a sample that must finish
within one hour and use **at most two steps**. This amendment supersedes the
exhaustive coverage, repeat count and ten-step requirements below for work that
had not completed. Retain all earlier results and original manifests.

At the change, 36 ten-step main cases and all ten two-step smoke cases were
complete. `R1344-I03-match` was interrupted after three steps had already
completed; its partial timing, telemetry and logs remain recorded as a
user-requested cancellation, not a product defect. The 39 unstarted original
main cases are explicitly deferred in `blocked.json` for this scope change.

The frozen `sample-one-hour.json` selects four **one-step** runs, within the
two-step cap: a no-reference control and nine-image `max` render at each of
640×480 and 1344×768, all with **362 frames**. Using one step preserves both
requested long-output resolutions within the hour. Execution prioritizes the
high-resolution control, both nine-image outputs, then the low-resolution
control if time permits. If that last control is deferred, report the candidate's
absolute measurements without substituting an unmatched timing comparison.
Steps, frame count and resolution must match within each comparison; existing
ten-step and two-step results are never used as one-step timing controls.
Each pair has one control, so repeated-control variability is unavailable.
Few-step outputs support functional inspection, not quality qualification.

The original BF16/full-VAE settings, seed and inputs remain unchanged. Existing
two-step smoke evidence retains the maximal nine-image/three-video mixture at
both resolutions. The other unrun audio, ordering, sizing and repeated-control
combinations remain deferred; do not describe this sample as full coverage.

`tests/cuda_multi_reference_sample.py` runs serially with the retained memory
guards and a shared wall deadline. The frozen render deadline is
**2026-09-23 15:45 UTC**, leaving time for local packaging before the requested
hour ends. It records skipped/aborted attempts if the time or resource bound
is reached. No automatic rerenders are allowed. The ordinary runner rejects
resuming the superseded ten-step matrix after this amendment is frozen.

The final sample completed the high-resolution control and both nine-image
renders; the low-resolution control was deferred at the budget boundary.
Across retained and new runs, 49 outputs completed, one case was cancelled and
40 cases were explicitly deferred. See the [results summary](multi-reference-results.md)
for measurements and evidence links.

## Original matrix contract (superseded above for the remaining work)

| Setting | Campaign contract |
| --- | --- |
| Output sizes | 640×480 and 1344×768; identical coverage at each size |
| Main renders | Ten denoising evaluations, seed 42; output frame count selected from the input durations below |
| Output duration | Original matrix: at most 226 frames / 9.4167 seconds; added nine-image cases and their controls: 362 frames / 15.0833 seconds, explicitly requested by the user |
| Startup smoke renders | Two evaluations, 73 or 158 frames with matching controls, at the two requested sizes |
| Images | Zero through nine distinct image assets; image cases run with both `--ref-image-size match` and `max` |
| Videos | Zero through three; **sum of supplied video durations ≤6 seconds per request**; each default-pipeline clip is at least two seconds |
| Combined references | At most 12: nine images plus three videos |
| Execution | One CUDA render at a time, fresh process per attempt |
| Deliverables | Per-attempt wall time and peak VRAM, no-reference comparisons, MP4s, HTML galleries, reproducible bug ledger |

`h3_align_frame_count()` rounds upward to `5 + 17n`. `--seconds 10` requests
240 frames, which becomes 243 frames / 10.125 seconds. **226 frames** is the
largest supported frame count below the ten-second cap. Earlier 243-frame
qualification gates belong to other campaigns. The ten-evaluation ceiling
applies at both resolutions in this campaign, including repeats and failures.

For a request containing videos, choose the smallest supported output length
that covers the **sum of their actual supplied durations**, which also covers
each individual clip. Compute `required = ceil(24 * sum(duration_seconds))`
using exact media time bases, then `frames = h3_align_frame_count(required)`.
Reject a video-bearing case if the aligned result exceeds 226 frames; never
truncate references to make them fit. This duration choice is a conservative campaign
policy, not a claim that Ref2VA requires the output to concatenate its inputs.
The six-second aggregate input cap supports these duration groups:

| Input/reference group | Aggregate video duration | Output frames | Output duration |
| --- | ---: | ---: | ---: |
| V1 and mixtures using V1 | 3 s | 73 | 3.0417 s |
| V2 and mixtures using V2 | 6 s | 158 | 6.5833 s |
| V3 and mixtures using V3 | 6 s | 158 | 6.5833 s |
| Image-only main cases | None | 226 | 9.4167 s |
| Added nine-image cases and matching controls | None | 362 | 15.0833 s |

Run no-reference controls at each frame count used, not one shared-duration
control. Smoke tests use 73 frames for image-only/single-video cases and 158
frames for the three-video mixture, accommodating the full supplied clips.
The 362-frame extension is image-only, covers both output resolutions and both
`match`/`max` modes, and retains ten evaluations. It is the explicit exception
to the original ten-second output cap; all original cases retain their limits.

Use the original BF16 checkpoint, default CUDA dense attention, full video VAE,
all transformer blocks and no step/core reuse. Freeze these choices across
references and controls. Turbo, Sage, NVFP4, preview VAE, continuation, first/last
frame anchors and still generation are outside this comparison. Ten steps
provide functional and visual inspection evidence, not final-render quality
qualification. Record the effective settings, including any environment
overrides, rather than trusting the command line alone.

The final-run requirement was raised to **`--steps 10` for every main case**
before any main render started. The original six-step base/362-frame contracts
remain retained to preserve the completed two-step smoke identities. A hashed
`final-main.json` overlays all 76 main attempts with ten steps and separate
`S10` baseline keys. The runner enforces the new ten-step ceiling. The renderer's
optional `H3_TEST_MAX_EVALUATIONS` diagnostic hook only accepts the literal `6`,
so final runs leave it unset and use the normal ten-step CLI; two-step smoke
runs retain `H3_TEST_MAX_EVALUATIONS=6`. Production memory guards are unchanged.
No six-step main results are reused. Smoke outputs remain preflight evidence.
The user raised the smoke timeout to 3,600 seconds after the maximal high-resolution
mixture hit its original 900-second limit during VAE decode. The retained
`retries.json` records the explicit retry; the original attempt remains in
`attempt-1` and the replacement uses `attempt-2`.

Standard final timeouts are at least 1,500 seconds or 14× the longest completed
smoke baseline at that resolution; the 362-frame group uses at least 3,000
seconds or 40× that baseline. Each exact timeout is retained in its request.

## Known constraints found while planning

The current [duration validator](../../src/media/refvideo.c) requires **48–360
normalized frames per video** in the default `released-v1` path: at least two
seconds per clip, including silent clips. The user's revised six-second total
budget accommodates three distinct two-second clips, so multi-video cases now
target successful default-pipeline generation. This replaces the earlier
short-clip rejection matrix and legacy diagnostics; no legacy fallback is
needed or scheduled. Do not concatenate the clips into one reference or alter
the validator. A rejection of a valid fixture is a finding to investigate and
report, not a planned pass.

Exactly two seconds is the minimum: V3 must contain **48 normalized frames per
clip**, not 47 after trimming or resampling. The released pipeline selects 39
VAE frames and 12 temporal latents from each such clip. V1 and each V2 clip
contain 72 normalized frames, selecting 56 VAE frames and 17 temporal latents.
Verify these distinctions so expected VAE prefix selection is not confused
with dropping an input clip. These are source-derived expectations; CUDA
runtime validation remains pending.

The current [request validator](../../src/engine.c) permits nine images, three videos,
three audio inputs and 12 references overall. Embedded video audio counts
toward the audio-input limit. Separate [audio validation](../../src/media/ffmpeg.c)
also requires at least two seconds. Core video cases use `--ref-silent-video`
to isolate visual conditioning; a three-second audiovisual fixture covers
embedded audio separately. See [Ref2VA preprocessing](../minimax/refvideo.md).

Image sizing is not a request to stretch every reference to the output canvas.
In [h3_reference_image_canvas()](../../src/host.c), `match` scales toward the
target pixel area without upscaling; `max` retains native size up to a
2048-pixel short edge. Dimensions are rounded to the canvas multiple. Record
resolved dimensions and latent rows for every image. A small input may have
identical geometry in both modes; that is not evidence of a sizing bug.
Video sizing is independent of this image flag.

## Node, build and reproducibility

Use the supplied node:

```sh
ssh cuda-pro6000
```

The [previous CUDA record](../features/single-still-cuda-results.md) identifies an RTX
PRO 6000 Blackwell Server Edition, 97,887 MiB VRAM, CUDA 12.8 and models at
`/path/to/models/MiniMax-H3`. Re-probe the GPU UUID, actual free VRAM, driver,
toolkit, host and cgroup limits, disk capacity and other GPU processes before
execution. These are previously observed values, not an availability check.

Create an isolated source/build directory `/path/to/h3-multi-reference` and
reuse the existing model files read-only. Pin the source revision plus any
working-tree diff, source manifest, binary checksum, compiler/build flags,
dependency versions and model identities. Archive the nine inputs and prepared
clips with checksums. Keep credentials and unrelated environment values out of
the records. Existing `scripts/cuda_remote_*.sh` can supply transport/probing;
inspect their defaults before reuse.

Use one explicit weight residency policy for the entire comparison. Start
with `--cuda-weight-mode resident` on the reported 96 GB device and confirm
capacity during smoke tests. Do not silently switch an expensive reference
case to streaming. If resident mode is infeasible, report the affected rows;
any separately selected streaming matrix needs its own matching controls and
configuration identity.

Illustrative main command, with ordered reference arguments appended before
`-o` for each candidate:

```sh
./bin/h3cli -d /path/to/models/MiniMax-H3 \
  -p 'A cinematic wide shot in a sunlit courtyard. Adults wearing everyday clothing turn toward the camera and gesture naturally. The camera moves slowly, with coherent lighting and quiet outdoor ambience.' \
  --width 640 --height 480 --frames 226 --steps 10 --seed 42 \
  --reuse 1 --core-reuse 1 --cuda-device 0 --cuda-weight-mode resident \
  --cuda-attention default --cuda-denoise-quant off \
  -o outputs/multi-reference-cuda/R640-B00-F226-r1/output.mp4
```

The runner validates flags against the frozen executable and records its
effective configuration. The CUDA build selects CUDA; do not add a Metal
backend flag. Run with a controlled, recorded `H3_*` environment and retain
the existing memory guards.

## Fixtures, prompts and reference ordering

Inventory the existing nine files as a stable ordered pool:
`1.jpg`, `face1.jpg`, `face2.jpg`, `body1.jpg`, `body2.jpg`, `2.jpg`,
`3.jpg`, `5.png`, `4.jpg`, all under `inputs/`. Verify that these
are distinct decodable assets, inspect their content, and freeze role labels
before rendering. Use nested subsets of size 1, 3, 6 and 9 so count comparisons
add known assets. Preserve original bytes. Include a sufficiently large source
early in the ordering to exercise different `match`/`max` geometry at both
resolutions; if the pool cannot do that, report the coverage gap explicitly.

Prepare distinguishable video excerpts from existing user-provided or retained
generated footage. If suitable motion footage is absent, deterministic
pan/zoom fixtures from the supplied images can exercise transport and encoding;
label their limited motion evidence. Asset preparation uses FFmpeg, outside
render timing, and requires no new model generation. The frozen fixture set uses one retained
garden-motion excerpt and two explicitly labeled zooms of clothed supplied
portraits; the latter provide limited real-motion evidence.

| Video set | Supplied clips at 24 fps | Aggregate duration | Primary audio policy |
| --- | --- | ---: | --- |
| V1 | One 72-frame clip | 3.0 s | Silent |
| V2 | Two distinct 72-frame clips | 6.0 s | Silent |
| V3 | Three distinct 48-frame clips | 6.0 s | Silent |
| A1 | V1 with a known embedded soundtrack | ≤3.0 s | Embedded audio included |

Validate actual decoded frame counts, timestamps and stream/container durations
with FFprobe/FFmpeg. Trim/mux audio so padding does not put supplied media over
the cap. Preserve the exact frame counts of the silent V2/V3 fixtures; in
particular, never shorten a two-second V3 clip below 48 normalized frames.
If the A1 soundtrack requires a small duration adjustment, record it and keep
both streams at least two seconds long. Freeze video dimensions, codec, frame
rate and motion across output resolutions. Record source, normalized, selected VAE and latent frame counts
separately; model preprocessing is not the supplied duration.

The core matrix uses the **same literal prompt** and seed for every reference
count and its no-reference control. The multimodal preamble supplies the visual
conditions without requiring a different user prompt for every count. This
measures the cost of adding conditioning to a fixed request. A no-reference
run uses the T2VA/FL2VA route, while reference runs use Ref2VA; their different
checkpoint/loading and packed-sequence costs are part of the measured workflow
overhead. Do not describe the difference as isolated reference-encoder time.

Persist the exact ordered argument list. `<Picture N>` and `<Video N>` use
separate per-modality ordinals in [src/conditioning/multimodal.c](../../src/conditioning/multimodal.c), not
the global reference index. Verify this mapping with host/tokenizer checks.
Ordering diagnostics below add explicit binding clauses to the core prompt;
retain the unbound core prompt as their baseline and disclose that prompt
change in their timing rows. Those rows measure practical bound-prompt cost,
not the strictly identical-prompt comparison used by the core matrix.

## Test matrix

Prefix each ID with `R640` or `R1344`. Suffix image-bearing cases with `match`
or `max`; both modes are required. Every video case uses `released-v1`.
All main attempts request ten steps. I01/I03/I06/I09 use 226 frames;
single-video cases use 73 frames; two/three-video cases use 158 frames.
B00 has separate `F073`, `F158` and `F226` variants at each resolution, plus
`F362` controls for the added long-output group below.

| ID | Images | Videos / total source duration | Image modes | Expected default-path outcome |
| --- | ---: | --- | --- | --- |
| B00 | 0 | 0 | N/A | Complete no-reference control |
| I01 | 1 | 0 | match, max | Complete |
| I03 | 3 | 0 | match, max | Complete |
| I06 | 6 | 0 | match, max | Complete |
| I09 | 9 | 0 | match, max | Complete or documented capacity failure |
| V01 | 0 | V1: 1 / 3 s | N/A | Complete with `--ref-silent-video` |
| V02 | 0 | V2: 2 / 6 s | N/A | Complete with both videos consumed |
| V03 | 0 | V3: 3 / 6 s | N/A | Complete with all three videos consumed |
| X11 | 1 | V1: 1 / 3 s | match, max | Complete |
| X91 | 9 | V1: 1 / 3 s | match, max | Complete or documented capacity failure |
| X32 | 3 | V2: 2 / 6 s | match, max | Complete with all references consumed |
| X93 | 9 | V3: 3 / 6 s | match, max | Complete or documented capacity failure |

This is **22 configurations per resolution**, including all three B00 lengths:
44 overall. Run each B00 length three times per resolution, bracketing its
matching candidates (beginning, middle, end): **56 core attempts** (38 reference
candidates and 18 controls). Every core row targets a completed render.
Record any actual failure with time/VRAM and diagnostics, but set its
successful-render overhead comparison to N/A.

Required supplemental rows at each resolution:

| IDs | Purpose | Additional attempts per resolution |
| --- | --- | ---: |
| A01, A11 | A1 via `--ref-video`, respectively zero images and one image (`match`); compare with V01/X11-match | 2 |
| O03 | Reverse I03-match image order and bind the three `<Picture N>` roles explicitly in the prompt | 1 |
| O91 | Interleave the video among X91-match images; explicitly bind `<Picture 1>`, `<Picture 9>` and `<Video 1>` | 1 |
| R91 | Repeat X91-max unchanged after other completed jobs; check repeat timing, output and GPU cleanup | 1 |

Together this is **66 scheduled main attempts**, including the 56 core
attempts. Up to ten separate smoke attempts run at two steps: B00-F073,
I09-max and V01 at 73 frames, plus B00-F158 and X93-max at 158 frames, at each
resolution. This checks the nine-image/three-video combination before the main
ten-step run. Every smoke attempt reports timing/VRAM against its resolution's
matching-length smoke B00; never compare smoke timings with ten-step controls.
Run up to **76 total scheduled attempts**; avoid unbounded automatic retries.
Retries needed to distinguish a transient failure must be counted and
documented separately.

### Added 362-frame group

The user additionally requested nine-image tests at 362 frames. At **each**
resolution run `B00-F362-r1`, `I09-F362-match`, `B00-F362-r2`, `I09-F362-max`,
and `B00-F362-r3`, all at ten steps. These add four reference renders and six
matching controls: **76 main attempts plus ten smoke attempts, 86 total**.
The original 76-attempt count above describes the base matrix only. Calibrate
long-run timeouts separately (conservative 40× the longest completed matching
resolution smoke baseline, at least 3,000 seconds), record each chosen timeout,
and retain all resource guards.

Freeze this addition in `extension-362.json`, linked to the original immutable
manifest identity. Validate every resumed case definition exactly. Existing
cases, model/binary/asset identities and their measurements remain unchanged;
362-frame comparisons use only F362 controls, never shorter baselines.

Audio rows use the same core prompt and matching B00-F073. Order rows disclose
the extra binding clause as described above; O03 uses B00-F226 and O91 uses
B00-F073. R91 also uses B00-F073. A failed or rejected candidate has no
successful-render ratio even when its baseline
exists. Do not infer equivalent output across reference permutations.

## Measurement and resource controls

The external runner is [tests/cuda_multi_reference.py](../../tests/cuda_multi_reference.py),
with frozen contracts under `tests/archive/multi-reference/multi-reference*.json`.
[The report generator](../../tests/cuda_multi_reference_report.py) creates the
tables and galleries; [the offline auditor](../../tests/cuda_multi_reference_audit.py)
checks media hashes, raw telemetry totals and all relative playback links.
These are new test tools, not changes to inference behavior. Launch the actual
CLI executable, not a reused in-process model session. Reuse report/transport
patterns from `tests/still_cuda.py` and the remote scripts without inheriting
their fixture sizes, precision settings or clocks.

**Wall time:** use the node's monotonic clock immediately before process
creation and after process/owned media-child completion. Include model load,
reference decode/encode, vision/text, denoising, full VAE/audio decode, muxing,
output writes and normal teardown. Exclude source transfer/build, prepared
fixture creation, telemetry startup, post-run validation and downloading.
Fresh-process timings are not necessarily cold-filesystem-cache timings.
Keep the first baseline, all repeats and run order visible. CUDA profile
marks currently report cumulative context time; do not relabel their values
as standalone phase durations or substitute them for total wall time.

**Peak VRAM:** start external NVML sampling before launching the child and
continue through process exit and GPU release, at a target interval of 50 ms
(100 ms maximum, gaps recorded). Record device UUID, whole-device used VRAM,
idle usage, per-process bytes/PID where available, and the process-tree mapping.
Include CUDA contexts, libraries, temporary buffers and allocation caches, not
only h3cli's tracked tensors. Report `max_vram_gib` as the **sampled whole-device
maximum** on the exclusive GPU, with per-process peak and idle usage alongside
it. Do not subtract idle usage from that primary maximum. Tensor high-water
counters can supplement it but must remain separately labeled. GPU sampling
is a lower bound on an instantaneous peak; record cadence and telemetry gaps.
Missing counters are unavailable, never zero. A failed telemetry collector
invalidates the memory measurement and must stop further timed jobs until the
collector is restored.

Keep the GPU exclusive during each case, and monitor CPU RSS/swap and cgroup
usage independently. Host memory is not VRAM. Keep the existing 110 GB process
guard; additionally respect a smaller cgroup allowance. Retain raw cgroup usage and
subtract only clean inactive file cache when estimating reclaimable headroom;
checkpoint hashing can fill the page cache without exhausting the working set.
Record conservative
external stop thresholds before rendering: at least 4 GiB device headroom and
10 GiB host/cgroup headroom, with stricter limits if initial probing requires
them. Sampled monitoring cannot guarantee protection from instantaneous OOM.
Use gradual count escalation and smoke tests, and stop an affected branch
after an OOM, guard abort or loss of node responsiveness. Signal/reap only the
runner's process group and preserve its partial records; never reset the GPU
or kill unrelated processes. Wait for GPU usage to return near its recorded
idle band before the next job. Record unexplained retained allocations as a
possible cleanup bug and stop launching additional renders on that device.

Run smoke tests first at 640×480, then 1344×768. Start the main matrix at the
lower resolution, progressing through increasing image counts before mixtures.
Do not bypass guards, shrink inputs, drop references, change precision or
shorten a main case to turn a failure into a pass. Record exact failing cases
and continue independent safe branches. Calibrate a finite timeout from the
smoke/no-reference controls, log its value before each batch, and distinguish
timeout, external guard termination, application rejection and CUDA OOM.
Capture stderr faithfully; a clean exit alone is insufficient validation.

For every completed case, let `T0` be the median of its three matching B00
wall times and `G0` the median of their individual sampled VRAM peaks. Report:

```text
wall_seconds                 = this attempt's full wall time
wall_delta_seconds           = wall_seconds - T0
wall_over_baseline            = wall_seconds / T0
wall_overhead_percent        = 100 * (wall_seconds / T0 - 1)
max_vram_gib                  = max whole-device used bytes / 2^30
vram_delta_gib                = max_vram_gib - G0
```

Smoke controls have one sample and must be labeled accordingly. Link every
comparison to the baseline attempts and configuration key: source/binary,
model/policy, GPU, output size, frames, steps, seed and core prompt. Show the
baseline range; flag >10% wall-time spread as unstable and keep the comparison
provisional instead of hiding variability. Different resolutions never share
a control. Avoid claims about small speed differences from single candidates.

## Records, visual outputs and bug handling

Write results incrementally under `outputs/multi-reference-cuda/`:

```text
manifest.json, extension-362.json, final-main.json
source-and-assets.sha256, environment.json, build.log
commands.jsonl, retries.json, results.json, results.csv, bugs.json, bugs.md
evidence-audit.json
fixtures/                     # staged inputs, short clips, thumbnails, metadata
<case-id>/<attempt-id>/        # argv/env, stdout/stderr, metrics, telemetry, output.mp4
review.html                   # index and summary
review-640x480.html
review-1344x768.html
report.md
```

On the prepared node, resume with `.test-venv/bin/python
tests/cuda_multi_reference.py run --phase main`. Completed records are skipped
only after their frozen case definitions are checked. Generate reports with
`python3 tests/cuda_multi_reference.py report`, then run `python3
tests/cuda_multi_reference_audit.py --final` once every row has an outcome.
The reporter and auditor also run locally without CUDA or model files.

Every result records case/attempt IDs, reference order and modality counts,
source checksums, image policy and resolved dimensions, video/audio timings,
pipeline, prompt text/hash, requested/completed steps, output frame geometry,
configuration/baseline identity, UTC timestamps, monotonic wall duration,
VRAM and host peaks, telemetry health, exit status/signal, output checksums,
validation status and bug IDs. Capture packed sequence length and individual
conditioning row counts when existing diagnostics expose them; unavailable
diagnostics are recorded as unavailable rather than invented. A small host
probe may independently compute expected geometry without modifying inference.

Fully decode every resulting MP4 with FFmpeg after timing. Verify output
resolution, 24 fps, exactly the manifest's frame count (226 for image-only
main cases, 73 for single-video main cases, 158 for two/three-video main cases;
73 or 158 for smoke; 362 only for the added image-only group). Verify expected
audio/container streams and duration ≤10 seconds in original cases. Added
cases require exactly 362 video frames / 15.0833 seconds, allowing at most one
frame of container/audio rounding. Check for truncated files, decode errors, repeated/frozen frames and
obviously blank output; label heuristic findings
for visual review rather than declaring a quality bug automatically. Preserve
decodable partial outputs with a conspicuous failed/incomplete label.

The report screens every validated output with FFmpeg `blackdetect`: at least
0.5 seconds with 98% of pixels below 2% luma raises an unconfirmed visual
finding. Retain the detector command, log, intervals and input checksum;
invalidate cached screening if the media changes. Frame hashes independently
flag nearly frozen output, including uniform bright frames. Neither heuristic
alone declares a renderer regression or changes successful completion status.

The two resolution pages show synchronized candidate/control playback, input
image thumbnails and video previews, reference numbering/order, sizing mode,
effective geometry, wall time/overhead and peak VRAM. Include contact sheets
from several timestamps for motion and artifact inspection. Separate core,
audio/order, smoke and failed/rejected sections. A rejected case gets an error
card linking its log and bug/limitation record, not a fabricated output.
Use relative links and local assets; download the completed HTML, playback
media, fixture previews and test records into this workspace. Verify the pages
and links work without access to the CUDA node. Human notes may be added, but
delivery of the pages does not wait for human approval.

After the serial worker exits, regenerate the report on the node and run
`tests/cuda_multi_reference_audit.py --final`, then
`tests/cuda_multi_reference_audit.py --write-transfer-manifest`. The transfer
manifest records the size and SHA-256 of every retained artifact, including
logs, telemetry, fixtures, media, reports and tool snapshots. It excludes only
its own manifest/verification record and the empty runner lock, and refuses
to snapshot an active or unfinished campaign. After downloading, run
`tests/cuda_multi_reference_audit.py --verify-transfer` before regenerating any
derived files locally, then repeat the final evidence/link audit. This makes
the remote-to-local integrity check distinct from media decode validation.

Maintain a bug ledger with stable IDs, category (product/environment/harness/
visual/known limitation), severity, affected cases, exact reproduction argv,
source/binary/model/input hashes, expected versus observed behavior, logs,
memory timeline and minimal evidence. Record the two-second-per-video minimum
and frame-rounding rule as known constraints, not newly discovered defects.
The earlier MR-L001 planning limitation is resolved by the authorized
six-second input budget; it is not an outstanding campaign blocker or a runtime
bug fix. Runtime qualification remains pending. Separate existing restrictions,
suspected visual issues and confirmed regressions. Record memory/latency anomalies even
when the render completes, with the actual baseline evidence.

Do not modify CUDA kernels, preprocessing, limits, scheduling or production
defaults to address discoveries. Keep fixes deferred in the ledger for a later
request. Test-tool corrections must retain the invalidated attempt and explain
why its metrics were excluded. Do not start old regression suites that generate
20-step or >10-second clips; their helpers need bounded use in this campaign.

Close testing when every scheduled row has a measured outcome or an explicit
blocked/not-run reason with evidence, all valid outputs and failed-case records
are available locally, and the report covers both resolutions and sizing modes.
Summarize time and VRAM by reference family, capacity limits and actual
default multi-video coverage, including any unresolved failures. Completion of
the campaign does not imply that blocked scenarios work or that ten-step images
prove production quality.
