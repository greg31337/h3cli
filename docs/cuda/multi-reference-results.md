# CUDA multi-reference results

These measurements use the original images identified by the frozen campaign
manifest. The [current input images](../../inputs/README.md) were replaced with
AI-generated portraits on 2026-09-26. These recorded results do not qualify the
new images; a new campaign must prepare and hash its own fixture copies.

The remaining campaign was sampled under the user's one-hour budget and
two-step maximum. The new 362-frame cases use **one step**; earlier ten-step
renders and two-step smoke results remain separate evidence. Product fixes
remain on hold.

Full deliverables: [playback index](../../outputs/multi-reference-cuda/review.html),
[640×480 playback](../../outputs/multi-reference-cuda/review-640x480.html),
[1344×768 playback](../../outputs/multi-reference-cuda/review-1344x768.html),
[per-case report](../../outputs/multi-reference-cuda/report.md),
[CSV](../../outputs/multi-reference-cuda/results.csv), and
[findings](../../outputs/multi-reference-cuda/bugs.md).

## Scope and measurement

The node has an RTX PRO 6000 Blackwell Server Edition with 95.59 GiB VRAM.
Tests use the original resident BF16 checkpoint, default CUDA dense attention,
full VAE, seed 42 and no step/core reuse. All media and telemetry are retained.
Wall time runs from renderer launch through completion of its owned children;
download and post-render validation are excluded. Memory values below are
sampled whole-device peaks; separate process peaks and telemetry gaps are in
the CSV/JSON. Sampling targets 50 ms, with occasional longer gaps, so reported
peaks are lower bounds on the true instantaneous maximum.

Controls match resolution, output length and step count. The image-only
ten-step group uses 226 frames; single-video groups use 73; two/three-video
groups use 158. The explicit long-output extension uses 362 frames. Reference
videos total at most six seconds. Two of the three video fixtures use synthetic
zoom motion, limiting real-motion coverage. Reference-versus-control deltas
include different conditioning routes and the entire rendering workflow.

## One-hour closeout

Rendering stopped at 15:42:39 UTC on 2026-09-23, before the frozen 15:45 deadline.
Three selected one-step cases completed with all 362 frames (15.0833 seconds)
and audio. The low-resolution no-reference control was deferred because the
remaining time could not accommodate it. No further renders were launched.

| One-step, 362-frame case | Wall | Device peak | Process peak | Versus matched no-reference control |
| --- | ---: | ---: | ---: | --- |
| 1344×768, no references | 1134.43 s | 58.64 GiB | 57.47 GiB | Control |
| 1344×768, nine images, `max` | 1388.53 s | 62.09 GiB | 60.91 GiB | +254.11 s / +22.4%; +3.45 GiB device peak |
| 640×480, nine images, `max` | 390.23 s | 47.07 GiB | 46.44 GiB | Unavailable: matching control deferred |

The high-resolution pair has only one control; timing variability is unknown.
Do not substitute earlier two-/ten-step or shorter-output controls for the
missing low-resolution measurement. One-step videos are strongly blurred and
are functional/timing evidence, not quality qualification.

Final coverage is **49 complete outputs**: 36 ten-step main renders, ten
two-step smoke cases and three one-step samples. One ten-step case was
cancelled by the user. Forty cases are explicitly deferred: 39 original main
rows and the final sampled control. One earlier timed-out smoke attempt is
retained separately from its successful retry. There are no pending rows.

## Retained ten-step findings

All 33 regular 640×480 cases completed, covering up to nine images, three
videos, mixtures, both image sizing modes, embedded audio, ordering and a
same-input repeat. Three high-resolution cases completed before the budget
change: the 226-frame control and one-image `match`/`max` cases.

| 640×480 case | Frames | Wall | Matched control | Wall increase | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nine images, `match` | 226 | 522.25 s | 386.55 s | 35.1% | 43.01 GiB |
| Nine images, `max` | 226 | 966.73 s | 386.55 s | 150.1% | 45.42 GiB |
| Nine images + three videos, `match` | 158 | 724.70 s | 241.02 s | 200.7% | 43.72 GiB |
| Nine images + three videos, `max` | 158 | 1256.70 s | 241.02 s | 421.4% | 46.67 GiB |

The three repeated 640×480 control groups each had less than 0.61% wall-time
spread. The repeated nine-image/one-video `max` case produced identical MP4
and decoded-frame checksums, with wall times differing by 0.63 seconds.
Whole-device peaks varied despite matching process peaks; the evidence does
not establish a leak or its cause. Adding audio to the same silent video cost
about 5.2 seconds without images and 5.6 seconds with one image.

The 1344×768 ten-step control took 2706.24 seconds at 51.06 GiB peak. One image
took 2842.03 seconds / 51.45 GiB with `match`, and 3058.00 seconds / 52.01 GiB
with `max`: provisional increases of 5.0% and 13.0% against a single control.

## Findings and limits

No confirmed renderer defect has been identified in completed cases. The
initial high-resolution maximal-mixture smoke timed out during VAE decoding
at 900 seconds; its explicitly authorized retry completed in 975.71 seconds.
Both attempts remain recorded. Test-tool cache accounting and tokenizer
fixture corrections are distinguished from renderer defects in the ledger.

Few-step outputs are blurred or show periodic texture and support functional
inspection only. The ten-step contact sheets show recognizable scenes, with
imperfect faces/hands and some prompt-adherence concerns. Some supplied
references contain nudity, and some outputs do too. These observations do not
establish a reference-path regression or qualify preservation of every identity.
Full media decoding, frame hashes and blank-output screening complement the
contact sheets; human perceptual qualification is not claimed.

The unstarted original ten-step rows remain deferred, not passed. The interrupted
ten-step case retains its partial metrics as a user-requested cancellation.
The sampled closeout does not qualify the original exhaustive matrix.

## Verification

All 49 completed outputs passed full decoding, expected dimensions/frame count,
24 fps, duration and audio checks. Frame hashes and nearly-black-span screening
produced no warnings; this does not negate the visible few-step quality issues.
The final audit verified 16 input assets, every recorded attempt's telemetry,
step count and identities, and 592 local gallery links. Thirteen host checks
passed. GPU processes had exited after the final render.

The [transfer verification](../../outputs/multi-reference-cuda/transfer-verification.json)
matched **670 files / 1,010,830,239 bytes** against the frozen remote checksum
manifest. Both remote and local [final evidence audits](../../outputs/multi-reference-cuda/evidence-audit.json)
passed with zero pending cases. The one earlier smoke retry and user-requested
cancellation remain independently inspectable; neither was silently discarded.
