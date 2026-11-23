# CUDA rendering through 736 frames

Status: **planned — 0/24 tasks complete**.
Created **2026-11-23**.

Extend native **CUDA** video generation from the current 362-frame ceiling to
**736 aligned frames at 24 FPS (30.667 seconds)**. Metal retains its existing
**362-frame ceiling**. Cover the CLI, public API,
HTTP service and `--render`, including first/last-frame conditioning,
supported references, continuation, saved states and delivery. Preserve the
existing backend-specific canvas limits and model arithmetic. A valid duration
still needs to pass sequence, allocation and available-memory checks.

The completed **24/24 CUDA large-canvas milestone** is
[archived byte-for-byte](large-canvas-todo.md), SHA-256
`bf366b522ba261e92adac1a9b2c7acba57c1f3b2b539498be51fa05613f539a3`.
Its [acceptance report](cuda/large-canvas-results.md) and
[memory estimates](cuda/large-canvas-memory-estimates.md) remain historical
evidence. The [empty-latent still milestone](empty-still-todo.md) is unchanged.

## Scope and production contract

- Keep the existing upward alignment to `5 + 17*n` and the 24 FPS timebase.
  Ordinary video keeps its existing minimum of 22 aligned frames. Accept a
  request only when its aligned result is at most **736 on CUDA or 362 on
  Metal**; reject overflow and out-of-range requests before model work. Do not
  clamp or truncate requests.
- Derive backend-specific limits centrally: CUDA's 736 output frames correspond
  to **217 video latent steps** and **1,227 audio latent steps**; Metal keeps
  362/107/603. Preserve valid existing 22–362-frame behavior, still-image
  behavior, seeds, schedules and numerical recipes.
- The extended duration is **CUDA-only**, including generation, continuation,
  resume, full/preview decode, source capture and learned upscaling. Metal must
  reject effective durations above 362 before model loading or large allocation,
  including requests reached through saved states, scripts and HTTP. Read-only
  metadata inspection may describe a CUDA long state without executing it.
  Existing backend-specific canvas limits remain unchanged. Qualify CUDA's
  dense BF16/default-attention path first; audit optional modes and explicitly
  reject unqualified new-duration combinations instead of silently changing
  requested precision, depth, attention or caching.
- Cover prompt-only FL2VA, first/last anchors and supported Ref2VA inputs.
  Frame anchors remain incompatible with full references and continuation.
  Preserve `--render` global references, segment-local additions, comma lists,
  `#ref-image-size` inheritance and seed-set behavior.
- Keep reference-input duration limits separate from output duration. Existing
  2–15-second reference-media rules and pinned SGLang reference behavior are
  not expanded by this work. Helpers receiving the longer *target* frame count
  must still accept it while enforcing the independent input-media cap.
- Keep CUDA source capture and the existing learned K=0/2/3/4 upscaler temporally
  compatible where their existing shape, sequence and memory guards allow it.
  Metal's source/upscale/decode paths retain the 362-frame limit, including K=0.
  The shared upscale planner currently checks sequence length even for K=0.
  Do not promise every canvas/duration/refinement combination or raise kernel
  ceilings as a side effect of extending the frame ceiling.
- Preserve full-context attention and normalization. No temporal tiling of
  denoising, independently generated windows, temporal interpolation, frame
  duplication, silent reference removal or model replacement.
- **Motion quality and scene consistency are excluded from acceptance.** Do
  not add visual-review gates, aesthetic rankings, subject/motion consistency
  scores or long-range narrative checks. Validate execution, numerical
  correctness, state integrity, media dimensions/timing and resource behavior.

## Development test limits

The user's new limits supersede the archived three-second/max-canvas test
matrix **for this milestone only**:

1. All runtime video canvases must be **256×256**: generated clips, continuation
   sources/targets, saved-state replay, preview/full decode, server/script jobs,
   synthetic video fixtures and final merged files. Use 256×256 image fixtures
   for references and first/last anchors too. Do not invoke suites with larger
   hidden media defaults. Normal internal encoder preprocessing is unchanged.
2. Treat **20 seconds as a hard upper bound** for every video artifact, including
   sources, references, intermediate output and final concatenated videos.
   The main CUDA extended-duration case is **464 frames / 19.333 seconds**.
   `--seconds 20` requests 480 frames and aligns to **481 / 20.042 seconds**;
   it is a host-only planning case, not an authorized runtime test. Use explicit
   `--frames 464` for the representative near-20-second render. Shorter cases
   are allowed for regression, failure handling and integration checks.
3. **719/736 frames, all larger canvases, and above-limit boundary cases are
   host-only**: arithmetic, logical indexing, allocation plans and bounded
   metadata checks. Do not render/decode these cases or allocate their complete
   temporal GPU/CPU movie tensors. Do not generate a longer clip and trim it.
4. No learned 2× upscale runtime jobs in this campaign: a 256×256 source would
   produce 512×512, and a 128×128 source would also violate the specified
   runtime canvas. Cover extended upscaling through host planning, bounded
   metadata/index tests and existing numerical contracts. Report the missing
   long-upscale runtime validation explicitly.
5. Run real-model jobs serially, using all 50 model blocks, **two denoising
   evaluations by default**, and no more than the existing six-evaluation
   development ceiling. Keep the existing native memory headroom and test
   capacity guards; do not weaken them to admit a job.
6. Use local macOS for host/build work, short legacy Metal compatibility and
   early rejection checks; no Metal job may execute above 362 frames. Use the
   already authorized remote 72 GB CUDA machine for extended-duration execution.
   Use a configured alias such as `cuda-test` in committed examples; keep private SSH details
   and raw machine logs in ignored outputs.

For CUDA continuation/bridge, a **447-frame source**, **464-frame target** and
**447-frame preserved context** exercise context beyond the old limit while
delivering 17 new frames. The merged source plus suffix is exactly 464 frames,
so all artifacts remain below 20 seconds. Smaller continuation fixtures may
also be used. Never merge two full 464-frame clips during development.

[CONTRIBUTING.md](../CONTRIBUTING.md) normally requires the immutable
204-output CUDA regression after coherent code changes. Its fixed **640×480**
render conflicts with the user's current **256×256-only** test instruction.
Defer that unchanged gate under this instruction; do not resize its fixtures,
edit golden hashes, add skips or describe host checks as a numerical gate pass.
The historical 204/204 result does not qualify the new code. The prior explicit
request to run that gate authorized the completed run, not a new exception to
this milestone's testing restriction. Record this regression gap at closeout.
Documentation-only changes require document/link checks, not model execution.

## CUDA host planning baseline

These are extrapolated **capacity budgets**, not measured minimums or promises
that the current binary accepts these durations. Reuse the checked planner
formula with dense BF16, all 50 blocks, 512 text rows, no extra reference rows,
2 GiB for other live allocations and 10% capacity headroom. Actual conditioning,
live contexts and free device/host memory must be included before admission.

| Canvas | Aligned frames / duration | Video T / audio T | Resident / fully streamed VRAM capacity | Runtime test policy |
| --- | --- | --- | ---: | --- |
| 256×256 | 464 / 19.333 s | 137 / 773 | 48 / 9 GiB | Required representative case |
| 256×256 | 481 / 20.042 s | 142 / 802 | 48 / 10 GiB | Host-only: exceeds test duration |
| 256×256 | 736 / 30.667 s | 217 / 1,227 | 50 / 12 GiB | Host-only |
| 1344×768 | 719 / 29.958 s | 212 / 1,198 | 124 / 86 GiB | Host-only |
| 1344×768 | 736 / 30.667 s | 217 / 1,227 | 126 / 88 GiB | Host-only |
| 1920×1088 | 736 / 30.667 s | 217 / 1,227 | 209 / 171 GiB | Host-only |

Use `H3_CUDA_WEIGHT_MODE=stream` for bounded CUDA baseline jobs. Retain the
60 GiB development capacity ceiling, actual-free-memory admission and
`max(1 GiB, device_total/10)` device reserve; account for headroom once.
Plan about 64 GiB system RAM when retaining the existing streamed-weight cache,
subject to actual host admission and bounded decoder delivery.

At 736 frames, 1920×1088 has 442,680 video rows and 445,646 total rows under
the table's text/audio assumptions, below the current CUDA one-million-row
ceiling. Its 3840×2176 upscale has **1,770,720 video rows alone** and exceeds
that ceiling regardless of VRAM. Preserve such rejections, including shared
upscale-plan guards; frame support is not an unlimited sequence-length claim.

## M0 — Freeze temporal policy and bounded development

- [ ] LVD001 Inventory every output-frame, latent-time, context and media-input
  limit, including literals 362/107/604/15 and indirect helper callers. Freeze
  the backend/feature support matrix, baseline identities and legacy behavior.
  Distinguish output-duration limits from intentional reference-media limits;
  record the excluded quality criteria and runtime-validation gaps above.
- [ ] LVD002 Introduce shared backend-aware temporal limits and conversions in
  the host/geometry policy. Derive CUDA's 736/217/1,227 while retaining Metal's
  362/107/603 and backend-appropriate legal continuation contexts;
  validate seconds conversion, upward alignment, signed/unsigned casts and
  overflow. Keep still-image and existing short-video shapes unchanged.
- [ ] LVD003 Extend phase-specific host/device estimates to the new ceiling,
  using actual text, references, anchors and audio. Reproduce the table above
  without allocating long-video tensors; retain backend sequence/index bounds,
  capacity reserves, streamed placement and actionable rejection diagnostics.
- [ ] LVD004 Add a dedicated guarded test entry point and bounded host target.
  Check effective source/target/delivery canvases, aligned duration, reference
  media, resume/decode inputs, child workers and every merged output before
  launch. Reject 481/719/736-frame runtime jobs, all non-256×256 video canvases
  and Metal execution above 362 frames. Allow early rejection tests without
  model work. Audit inherited suite defaults and enforce the evaluation limit.

Exit: the shared policy and test runner separate CUDA production support
through 736 frames, Metal's 362-frame ceiling and the 256×256/under-20-second
development envelope.

## M1 — Generation, conditioning and public entry points

- [ ] LVD005 Update native validation in [engine](../src/engine.c) and
  [script planning](../src/cli/render_plan.c) to use the shared policy.
  Support CLI/API `frames` and `seconds` through each backend's aligned ceiling,
  preserving defaults, option precedence and script inheritance. Reject invalid
  durations before model loading or creating outputs, with consistent diagnostics.
- [ ] LVD006 Update explicit native HTTP requests and the
  [SGLang-compatible target-duration adapter](../src/server/sglang.c).
  Extend its 15-second *output* ceiling consistently with native
  alignment on CUDA, while retaining Metal's existing ceiling and the
  reference-media/minimum-duration contracts. Validate explicit versus derived
  duration, resolved execution backend and contradictory request fields.
- [ ] LVD007 Audit [conditioning/layout](../src/host.c), positional encodings,
  first/last-frame indices and audio/video row offsets through 736 frames.
  Ensure helpers such as [soundtrack preparation](../src/sglang/sglang_media.c)
  accept long target requests without expanding/truncating the input-media
  contract accidentally. Preserve reference order, image-size policy and seeds.
- [ ] LVD008 Audit CUDA allocation products, kernel launch dimensions, GEMM
  shapes, packed audio rows, attention sequence bounds and independent MLP
  slabs at the new maximum. Use checked arithmetic and bounded logical-index
  probes. Preserve existing global attention and numeric order; retain explicit
  failure when sufficient VRAM cannot overcome an existing kernel limit.
- [ ] LVD009 Preserve Metal's 362-frame policy across generation, continuation,
  saved-state resume, full/preview decode and K=0/2/3/4 upscaling. Audit shared
  helper callers so raising CUDA limits cannot enable long Metal execution.
  Reject effective durations above 362 before model loading or large allocation,
  independently of canvas or available memory; retain legacy Metal behavior.

Exit: supported entry points share one backend-aware duration policy and preserve
all conditioning semantics, with separate resource/backend admission.

## M2 — State, delivery and connected workflows

- [ ] LVD010 Extend AV-state and presentation shape validation to the shared
  ceiling. Audit element counts, file-length bounds, trim offsets and timestamps.
  Preserve legacy format/identity rules; ensure old readers safely reject new
  long states and malformed files before large allocation. Add a required
  format field/version only if the existing metadata cannot express the policy.
  Identify extended CUDA durations independently of canvas profile and reject
  their execution on Metal, even when their canvas is valid there.
- [ ] LVD011 Extend sampler checkpoints, clean source bundles and their
  inspectors to represent longer temporal shapes safely. Preserve ownership of
  references/audio, RNG state and exact same-build pause/resume behavior. Audit
  container limits and any serialized fixed-size temporal structures without
  weakening provenance or existing build/runtime compatibility checks.
- [ ] LVD012 Extend CUDA [full VAE streaming decode](../src/vae/video_vae.c)
  beyond its `T<=107` guard using the backend-aware limit; Metal retains that
  guard. Preserve 22-frame chunk decoding, five-frame overlaps, chunk order and
  bounded stitching/delivery memory.
  Validate final-chunk counts, cancellation, failure cleanup and decode-only
  paths; never retain the whole RGB movie to implement the longer duration.
- [ ] LVD013 Validate CUDA generated audio decoding through 1,227 latent steps:
  checked buffer/sample counts, audio/video timestamps and complete delivery.
  Keep the 15-second reference-audio encoder limit independent. Verify no
  target-duration truncation, integer overflow or stale audio after context reuse.
- [ ] LVD014 Extend CUDA preview decoding and shared frame-count helpers, including
  [h3_tiny_vae_frames](../src/vae/tiny_vae.c), which also gates ordinary full-VAE
  delivery. Cover selected-frame bounds, tails and saved-state preview/decode.
  Enforce Metal's existing temporal ceiling at execution boundaries. Preserve
  the existing prohibition on preview-VAE upscale delivery.
- [ ] LVD015 Extend CUDA hard continuation and bridge temporal plans/masks,
  including [fixed bridge arrays](../src/sampling/bridge.h), validation and serialization.
  Derive the largest legal preserved context (**702 frames** at this ceiling)
  and require a longer target. Metal retains its existing target/context bounds.
  Preserve exact prefix bytes, audio alignment and suffix trimming. Keep
  first/last-anchor and upscale-continuation exclusions.
- [ ] LVD016 Extend the CUDA learned upscaler's temporal guards and
  source/condition planning where existing sequence/memory limits admit the result; keep Metal's
  362-frame source/target ceiling, including no-refinement jobs.
  Preserve K=0/2/3/4 schedules, full-context normalization, clean audio and raw
  condition retargeting. Cover the new limits with host-only plans/index tests;
  explicitly retain rejections such as 736 frames at a 3840×2176 target. Do not
  run a 2× video job or claim long-upscale runtime qualification in this campaign.

Exit: longer CUDA native jobs retain functional save/resume, decode and
continuation paths; connected upscale paths have explicit, checked compatibility limits.

## M3 — Bounded tests and acceptance

- [ ] LVD017 Add host tests for old and new boundaries, including 362/363,
  464/480/481, 719/720/736/737, huge integers and nonfinite seconds. Check
  temporal alignment, audio counts, anchor positions, 702-frame contexts,
  state metadata and resource/index rejection at supported maximum canvases.
  Test CUDA acceptance versus Metal rejection of aligned results above 362,
  including indirect saved-state and shared-upscaler entry points.
  Use small allocations/logical extents for cases outside the runtime envelope.
  Test the runner itself against forbidden dimensions, durations and merges.
- [ ] LVD018 Run guarded CUDA native cases at **256×256 / 464 frames** for
  prompt-only generation, both first/last anchors and supported references.
  Use short 256×256 reference videos and bounded audio fixtures under their
  existing input limits. Verify output dimensions, frame count, finite values,
  timestamps and audio duration; record per-phase time, whole-device/tracked
  peaks and host RSS. Do not assess motion or scene consistency.
- [ ] LVD019 On CUDA at 256×256 within 464 frames, exercise pause/resume,
  cancellation recovery, full/preview decode-only and reused API contexts. Compare seeded
  latent/audio results and replay outputs under the existing numerical
  contracts. Test allocator/encoder failure cleanup and repeated context use
  without increasing duration or resolution to induce failures.
- [ ] LVD020 Exercise real CUDA `--render` workers and HTTP generation with an
  extended-duration 256×256 case. Verify effective commands and preserve global
  versus local references, comma lists, image-size inheritance and seed offsets.
  Use shorter separate fixtures for independent-shot merging/seed sets; ensure
  every final file stays below 20 seconds and no subprocess bypasses the guard.
- [ ] LVD021 Run CUDA hard and bridge continuation at 256×256 with the
  **447-frame source → 464-frame target / 447-frame context** case. Check exact
  protected prefix, 17 delivered new frames, audio trimming, saved-state replay
  and a 464-frame final merge. Exercise the 702/736 boundary through host-only
  planning and mask/index checks, never a long render or full movie allocation.
- [ ] LVD022 Build on local macOS, run the bounded host suite, and verify
  256×256 / 22- or 39-frame legacy Metal rendering and saved-state/preview
  behavior. Verify early rejection above 362 frames across CLI/API/script/HTTP,
  resume, full/preview decode and upscale paths without loading models or
  allocating long tensors. Preserve existing Metal canvas rejection too.
- [ ] LVD023 Run compatible regression checks through the guarded entry point;
  build final macOS/Linux artifacts and verify offline 256×256 rendering and
  saved-state delivery within each backend's limits: CUDA extended-duration
  checks, Metal short compatibility and long-request rejection. Keep all 204
  golden hashes unchanged. Record the immutable CUDA gate as **deferred: fixed 640×480
  conflicts with user test limits**, and exclude noncompliant legacy suites
  rather than invoking their hidden defaults. Capture final artifact identities.
- [ ] LVD024 Update CLI help, README/usage, sequence rendering, HTTP, state,
  continuation and upscale documentation with the CUDA-only duration extension
  and Metal's unchanged 362-frame ceiling. Publish an acceptance report
  separating measured CUDA 256×256/464-frame behavior, Metal compatibility/rejection,
  host-only 736-frame and larger-canvas checks, deferred upscale execution and
  the deferred immutable gate. Include resource estimates/measurements and
  retained failures, validate links/archive hashes, and mark completion only
  when all required bounded checks pass. No quality/consistency review is needed.

## Completion criteria

The implementation accepts supported native CUDA requests through 736 aligned
frames. Metal retains its 362-frame ceiling, including saved-state execution
and upscaling. Unsupported Metal durations fail before model loading or large
allocation; invalid or unaffordable requests fail before unsafe work.
All 24 tasks have recorded evidence within the user's testing envelope.
**No 30-second or high-resolution runtime test is required or authorized by
this plan.** CUDA end-to-end evidence reaches 464 frames at 256×256; 736-frame
execution, larger-canvas long execution, long upscale execution and the new
code's immutable 204-output regression remain explicitly unverified under
these restrictions. Motion and scene consistency are outside the milestone.
