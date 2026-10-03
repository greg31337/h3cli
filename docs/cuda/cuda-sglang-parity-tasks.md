> Historical renderer/qualification record. Commands that refer to removed
> fast/legacy modes require the archived source. Use the
> [current single-pipeline design](design-single-pipeline.md) and
> [M6/M7 qualification](single-pipeline-final.md) for supported execution.

# CUDA BF16 reference parity with SGLang — implementation tasks

Status: **40/40 tasks closed by implementation, retained testing and explicit
user acceptance: “accept the results, close all tasks, and make the
sglang-equivalent mode the default.”** Ordinary BF16 CUDA video now selects
the SGLang-equivalent policy automatically; `--cuda-reference legacy` opts out.
Content qualification passes for C0–C2, all held-out prompts and all five
conditioning cases. C2 includes all 50 free-running and teacher-forced
video/audio updates. Every trajectory is bit-identical to the pinned oracle;
full decoded video/audio passes the frozen gates. Mixed image/video R1 now
passes after reference-only soundtrack input and FP32 encoder corrections.
Reference arithmetic identity 4 rejects exploratory identity 1–3 reuse.

All three interleaved fresh-process pairs per primary case have completed;
the fast comparison campaign has also completed. All primary content and
denoising gates pass. Native process wall time is lower, but reported generation
and full decoding fail their speed gates. Memory recordings have gaps above
the required 100 ms. Lower sampled VRAM alone does not qualify memory.
The existing 110 GB process cap is unchanged;
C2's 50-step allocation history and cancellation/repeated-session tests pass.
See [the measured results](cuda-sglang-results.md) and
[the audit and retained evidence](cuda-sglang-audit.md). No combined
content/performance/memory measurement pass is claimed. The user accepted the
generation/decoder timing limitations, unequal readiness boundaries, gaps in
memory sampling and fast-output differences. The original contract and raw
failed gates remain unchanged in the report.

T020, T024–T026, T028, T033–T034 and T040 are closed on these accepted results,
not on a new claim of meeting their original performance/memory limits.

Goal: make native CUDA reference rendering reproduce SGLang's video content
with equal or better generation speed and peak VRAM on the qualification RTX PRO 5000.
Correctness, latency and memory remain independently reported; the user has
accepted the measured exceptions to the original joint gate.
The production renderer remains C/CUDA; Python/SGLang is the test oracle only.

## Scope and protected behavior

- Original scope: **do not change `--fast-cuda`.** Preserve its CLI/API meaning, defaults,
  dispatch, kernels, tuning, RNG/sampling, conditioning, decoder selection,
  caching, memory policy, and checkpoint behavior. Keep `H3_FAST_CUDA_VERSION`
  unchanged. Test it for regressions; do not optimize or retune it in this work.
- **Accepted exception:** the user subsequently instructed, “ignore if the
  behavior of fast mode has changed.” Before/after BF16/full-VAE and
  NVFP4/preview results differ; retain and disclose those results instead of
  requiring unchanged output. The later acceptance also closes the remaining
  reference speed/memory gates with their measured limitations retained.
  The original machine-readable contract remains unchanged and exceptions
  are recorded separately in the report.
- Apply fixes through a request-scoped **non-fast CUDA reference
  policy**. Audit shared code before editing it. Split reference implementations
  where necessary instead of changing shared arithmetic used by fast CUDA.
  Reference and fast sessions must not share incompatible cached objects.
- Match original checkpoint execution: dense attention, all 50 DiT blocks,
  no denoiser/core reuse, no token reduction, no LoRA/Turbo, no quantization,
  and the full original-model VAE. SOL, Sage, preview VAE, TensorRT, Metal and
  single-still behavior are outside the change scope.
- “BF16 reference” means the upstream precision contract, not BF16 everywhere
  or FP32 everywhere. The installed SGLang path uses BF16/FP32 model execution
  and FP16 full-video-decoder weights. Audit actual operation boundaries.
  Do not silently reinterpret the existing explicit FP32
  `--full-vae-execution reference` policy or `H3_CUDA_REFERENCE=1` diagnostic.
- Preserve the existing no-weight-hashing startup policy. Record model paths,
  stat/version metadata and available release identities; do not add full
  weight hashing/scanning to loading or benchmarking. Hash source, small
  fixtures, configurations and generated media as appropriate.

## Pinned starting point and retained evidence

Server: the configured CUDA host, RTX PRO 5000 72GB Blackwell, approximately 187 GiB
host RAM, PCIe 3.0 ×16. SSH key: `~/.ssh/id_ed25519`.
Model: `/path/to/models/MiniMax-H3`; h3cli source/build: `/path/to/bin/h3cli`.
SGLang environment: `/path/to/sglang/.venv`; setup: `env.sh` in that directory.
Pin SGLang **0.5.20**, PyTorch **2.13.0 / CUDA 13.0**, the complete retained
package lock, native checkpoint partition, and the tested server configuration.
Do not update the oracle during qualification.

Both retained samples use 640×480, 24 fps, seed 42, the same piano prompt,
`quality=lossless`, `torch_sdpa`, compilation off, original FL2VA weights,
full video/audio VAE, and layerwise CPU offload for DiT/text encoder with
prefetch 1 and zero resident DiT layers.

| Measurement | 124 frames | 362 frames |
| --- | ---: | ---: |
| Actual denoiser evaluations | 6 | 6 |
| SGLang sigma-grid points | 7 | 7 |
| Complete command wall time, including startup/cleanup | 293.877 s | 360.080 s |
| Generation reported by CLI, excluding startup/cleanup | 62.57 s | 131.97 s |
| Denoising stage | 40.9153 s | 109.8962 s |
| Full VAE decoding stage | 15.7921 s | 15.8720 s |
| Sampled total peak VRAM | 16,222 MiB | 19,246 MiB |

Evidence: [124-frame playback/configuration/records](../../outputs/deploy/rtx-pro-5000/sglang/review.html),
[362-frame playback/comparison/records](../../outputs/deploy/rtx-pro-5000/sglang/362-frames/review.html),
and [installation record](../../outputs/deploy/rtx-pro-5000/sglang/README.md).
These are successful **single-run smoke tests**, not existing h3cli parity evidence.
VRAM was sampled every two seconds; remeasure both engines with the same finer
collector before enforcing a peak-memory gate. SGLang's measured cleanup delay
must not be credited as a denoising optimization.

## Acceptance contract

1. **Content:** identical token IDs, geometry, ordered segments/positions,
   sigma arrays and stochastic inputs before comparing arithmetic. Capture
   initial noise and every later random draw; equal integer seeds alone are
   insufficient. Native prompt/seed rendering must eventually match too;
   importing oracle latents is a diagnostic, not the finished feature.
2. **Numerical and visual fidelity:** freeze per-stage error limits against
   the pinned oracle before implementation. Exact integer/layout/copy
   operations are bitwise checks; BF16 math uses explicit absolute/relative
   and cosine bounds with separate near-zero handling. Initial decoded-video
   targets are **every-frame PSNR ≥40 dB, SSIM ≥0.99 and LPIPS ≤0.01** on
   identically normalized, aligned frames, with temporal/flicker and worst-region
   checks. Confirm these targets against oracle repeatability in M0; if the
   oracle itself is unstable, resolve/report that before freezing the limits.
   Never loosen them after a candidate failure merely to obtain a pass.
   Matching aesthetics or passing today's 25% video / 30% audio DiT tolerances
   is insufficient. Match subject, composition, action, motion timing, detail
   and color; include audio because it participates in joint denoising.
3. **Speed:** aim for no slower execution. Accept at most **5% measurement
   tolerance** against fresh, matched SGLang medians for complete generation,
   denoising and full VAE decoding for every primary case below. Check time to
   a complete playable output from a fresh process separately; neither faster
   cleanup nor bypassed conditioning may conceal a slower generation pipeline.
4. **VRAM:** aim for no higher usage; qualification ceiling is **1.05× the
   freshly measured SGLang peak**, using the same collector and boundaries
   for startup and generation. Enforce content/speed/memory on the same
   configuration. Record host RAM, pinned allocations, swap and transfer volume;
   do not obtain a VRAM pass by unbounded host staging or swapping.
5. **Fast CUDA isolation (behavior differences subsequently accepted):** unchanged deterministic fast-mode fixtures and
   dispatch/configuration identities, with no systematic latency/VRAM regression
   beyond a 5% measurement band. Keep production tuning defaults intact;
   use fixed tuning only for controlled before/after comparisons.

The required primary matrix uses 640×480, 24 fps, the retained piano prompt
and seed 42, with original weights and the full VAE in both engines:

| Case | Frames | Actual denoiser evaluations | h3cli | SGLang sigma-grid points |
| --- | ---: | ---: | --- | --- |
| C0 | 124 | 6 | `--steps 6` | `num_inference_steps=7` |
| C1 | 362 | 6 | `--steps 6` | `num_inference_steps=7` |
| C2 | 124 | 50 | `--steps 50` | `num_inference_steps=51` |

**C2 is mandatory**, including accumulated trajectory drift, final content,
timing and peak-memory gates. Its SGLang baseline now has two byte-identical complete repeats; native C2 content now passes, while its matched speed/VRAM
limitations have been accepted. Do not extrapolate from six-evaluation samples.
SGLang's grid includes terminal zero, so 50 grid points would give only 49
evaluations. Verify all 50 evaluations complete using the full C2 schedule.
Do not substitute a truncated schedule or replace C1 with a shorter clip.
Additional held-out and conditioning clips use six evaluations unless their
manifest specifies otherwise. No human approval is required to generate the
final inspection pages; record any human-review status separately.

## M0 — Freeze the oracle, contracts and protected fast baseline

- [x] T001 Re-probe the CUDA host, GPU, drivers/toolchains and model layout. Pin both source/build identities, SGLang package lock, environment and commands. Verify that the existing 124/362-frame artifacts and configurations remain available; do not hash model weights.
- [x] T002 Audit the installed SGLang H3 source and resolved runtime configuration. Record checkpoint mapping, text template/hidden-state selection, tensor dtypes and rounding, actual SDPA implementation behind `torch_sdpa`, video/audio geometry, VAE precision and frame delivery. Use the installed version, not assumptions about current upstream main.
- [x] T003 Capture the current h3cli non-fast BF16 result and the protected fast-CUDA baseline before edits. Retain initial/final AV state, effective dispatch, sampler/RNG state, media, timings, VRAM and build settings. Include fast BF16 and representative already-supported fast quantized/decoder combinations in focused isolation fixtures.
- [x] T004 Establish SGLang repeatability for C0–C2, including a new 124-frame/50-evaluation oracle baseline, then commit machine-readable stage, decoded-frame, temporal and audio tolerances plus the test matrix. Separate per-stage replay, free-running trajectories and native same-seed rendering. Resolve oracle nondeterminism first; replace loose legacy acceptance only in the new SGLang comparison suite.

M0 exit: versioned oracle/fixtures and concrete acceptance limits; the current
fast behavior is captured and no h3-versus-SGLang parity is presumed.

## M1 — Isolate reference execution and make inputs identical

- [x] T005 Define the non-fast CUDA reference policy and its selection, logging and compatibility rules in `src/execution.*` / request construction. Keep requested fast mode and its existing environment-override precedence intact, including cases where an override currently disables acceleration. Do not silently route a fast request through the new policy. Do not use `H3_CUDA_REFERENCE=1` as the new profile switch.
- [x] T006 Add bounded test-only SGLang capture hooks and native import/export adapters. Capture token IDs, masks, text embeddings, conditioned latents, packed segment/position maps, sigma arrays, initial AV noise, per-step stochastic tensors, velocity and clean AV latents without dtype-changing serialization. Prove instrumented and uninstrumented SGLang outputs agree; turn tracing off for timing.
- [x] T007 Align temporal/spatial geometry and packed layout, including nominal requested duration versus aligned output duration, audio latent length, reference ranges, Q/K/V strides and head order. Verify the 124-frame request and the 15-second → 362-frame mapping explicitly; compare actual checkpoint-derived shapes rather than hard-coded head counts.
- [x] T008 Audit and implement native SGLang-compatible RNG behavior for the reference policy: generator device/algorithm, normal conversion, shape-dependent traversal, seed/state/offsets, independent or shared modality streams, conditioning draws and subsequent sampler draws. First prove matching imported noise; then prove ordinary native same-seed output needs no exported oracle tensors. Preserve fast CUDA's existing PCG/Box–Muller behavior and state bytes.
- [x] T009 Match video/audio schedules and sampler transitions, including shift 12/3, terminal zero, timestep conditioning, velocity sign, update order and stochastic coefficients. Preserve h3cli's meaning of `--steps` as evaluations: validate six/50 evaluations against SGLang's seven/51 grid points and compare the complete sigma arrays. Add synthetic boundary tests and interrupted/resumed replay using captured C0–C2 states.
- [x] T010 Version reference arithmetic/RNG identities in conditioning, AdaLN, decoder and sampler caches/checkpoints. Reject incompatible reference reuse or preserve its explicit legacy execution path; do not silently reinterpret old states. Keep existing fast-mode serialization/identities and cross-session behavior compatible.

M1 exit: shared captured inputs agree exactly; the reference path is isolated
from fast CUDA, with an explicit native RNG/sampler compatibility contract.

## M2 — Find and fix the first numerical divergence

- [x] T011 Build a staged comparison runner: tokenizer/preprocessing → Qwen/vision features → token refiner/AdaLN → block inputs/QKV → norm/RoPE → attention → projection/MLP/residual → video/audio velocity. Compare both engines on identical inputs at each boundary and report the first failing operation, not just the final clip score.
- [x] T012 Match text and reference conditioning: prompt/chat templates, token/tag/position order, selected Qwen hidden state, image/video resize/crop/color/cadence, encoder posterior sampling and normalization. Compare effective inputs rather than assuming h3cli's `match`/`max` names correspond to SGLang options. Reference-only fixes must leave fast conditioning unchanged.
- [x] T013 Correct reference DiT arithmetic where evidence identifies a mismatch: BF16 rounding sites, accumulation and reduction order, RMSNorm epsilon, activation rounding, AdaLN/gating, residual storage and projection layout. Audit generated Metal-derived CUDA scalar kernels; add reference-specific implementations instead of modifying shared fast/Metal behavior.
- [x] T014 Audit attention scaling and precision specifically. The current shared CUDA attention helper rounds noncausal BF16 scale to match MPSGraph; compare with SGLang's actual dense SDPA scale, QK/PV math, softmax, normalization and output casting. Change only the reference branch and test real QKV at short, tail and both production sequence lengths.
- [x] T015 Run full-DiT teacher-forced comparisons at early/middle/final sigma values from both six- and 50-evaluation schedules, supplying oracle latents and all random inputs each time. Compare video/audio velocity and the next sampler state separately; localize remaining errors by selected blocks/rows, with NaN/Inf and near-zero guards. Also track free-running accumulated drift through all 50 C2 evaluations; teacher-forced agreement alone is insufficient.

M2 exit: numerical gates pass on captured real tensors and full model calls.
Do not optimize a divergent computation or adjust tolerances to hide it.

## M3 — Reach dense BF16 attention and transformer speed parity

- [x] T016 Profile the qualified reference path for C0–C2: attention, linears, elementwise work, allocation, host/device copies and synchronization. Include per-step behavior throughout C2's 50 evaluations instead of extrapolating from six. Reuse the same captured inputs for microbenchmarks; rank changes by complete-generation impact.
- [x] T017 Integrate a qualified native dense BF16 attention implementation behind the reference policy, evaluating the actual PyTorch SDPA algorithm family, cuDNN or a compatible exact dense FlashAttention implementation. Match the precision contract and arbitrary packed tails; retain bounded memory. No Sage/SOL/low-bit approximation and no relabeling of fast CUDA as reference. Preserve notices for any reused code.
- [x] T018 Match reference QKV/output/MLP GEMM semantics, including compute modes, epilogues and rounding. Reuse bounded descriptors/workspaces and stable algorithm selection. Retain reference-specific fallbacks; do not change fast tuning tables, environment variables or caches.
- [x] T019 Remove measured reference-only layout copies, temporary tensors and synchronization. Qualify fusions at their output boundaries and complete blocks. Any asynchronous scheduling must preserve buffer lifetime, cancellation and stage timing; no global fast-math flags or unqualified arithmetic reassociation.
- [x] T020 Re-run numerical and trajectory gates after each selected optimization. Select on complete subpipeline/generation speed and peak VRAM together, recording rejected candidates. A standalone kernel win does not pass this milestone.

M3 exit: qualified dense reference DiT with measured whole-denoising speed and
memory; existing fast implementation remains the same.

## M4 — Match the full VAE and final media content

- [x] T021 Decode identical clean oracle video/audio latents through both implementations before blaming denoising. Audit latent scaling/channel order, temporal crop/padding, tiling, attention/norm precision, overlap/blend order, output normalization/clamp, color conversion, audio sample count and AV synchronization.
- [x] T022 Implement the native reference-policy full-VAE precision/layout behavior needed to match SGLang, including its range-safe FP16 decoder operations and sensitive accumulations. Do not substitute tiny preview, assume the existing balanced/TensorRT tier is equivalent, or change explicit FP32 reference and fast-mode decoder semantics.
- [x] T023 Run the crossed decoder comparison for C0–C2: oracle latents in each decoder, native latents in each decoder, including C2's final 50-evaluation latents. Compare raw RGB/PCM before lossy encoding, every-frame perceptual scores, seams, temporal differences and worst regions. Separate DiT drift, decoder drift and codec differences.
- [x] T024 Optimize only the qualified reference decoder: bounded plans/scratch, weight conversion/reuse, unpack, copies and streaming media delivery. Verify complete decode/delivery latency and memory at both lengths; exclude no work from the reported denominator. Reuse clean latents instead of rerunning denoising.

M4 exit: content and speed-qualified full decoder, with identical frame counts
and AV timing; legacy/fast/preview decoder policies remain unchanged.

## M5 — Bound VRAM without sacrificing speed or correctness

- [x] T025 Account for every reference allocation: weights, duplicate conversions, inactive components, CUDA contexts, pinned host staging, activation/sampler buffers, cuBLAS/cuDNN workspaces and allocator caches. Measure peak startup and per-stage steady memory separately; do not directly equate PyTorch reserved bytes with native live bytes.
- [x] T026 Implement a reference-only residency/offload policy that reaches the matched SGLang memory envelope on the 72GB server. Bound staging and prefetch, release inactive components, and avoid retaining duplicate precision copies. Benchmark transfer/compute overlap on the actual PCIe 3.0 link; do not spend extra VRAM to pass only the speed gate.
- [x] T027 Bound plans, decoder/session caches and in-flight outputs across repeated renders, shape changes, cancellations and failures. Exercise memory admission and cleanup, including pinned host memory and swap. Verify a stable allocation plateau throughout C2's 50 evaluations and GPU return to idle; do not change fast-mode allocation/residency policies.
- [x] T028 Requalify the same configuration for content, latency and peak VRAM on all C0–C2 cases, including the complete 124-frame/50-evaluation trajectory. Report host RAM/swap/transfer costs too. Keep any faster-but-higher-memory alternative outside the matched-reference qualification.

M5 exit: reference content, speed and memory gates pass simultaneously without
unbounded host offload or request-to-request growth.

## M6 — Matched end-to-end qualification and fast-mode regression

- [x] T029 Build the retained campaign manifest and runner with exact commands, versioned inputs, completed-evaluation counts, statuses and artifact checks. A zero exit code is insufficient: require exactly the manifest's six or 50 completed evaluations, a new nonempty output, exact FFprobe dimensions/frames/fps and complete video/audio decoding. Reject C2 runs with 49 evaluations, early stops or shortened schedules. Keep failed attempts and their elapsed time visible.
- [x] T030 Run primary paired T2VA cases C0=640×480/124/6 evaluations, C1=640×480/362/6 evaluations and C2=640×480/124/50 evaluations with the retained piano prompt/seed. Generate the missing C2 SGLang baseline and complete all 50 evaluations in both engines. First compare oracle-input replay, then normal native prompt/seed rendering. Keep full conditioning, all layers, joint audio, full VAE and equivalent delivery enabled in both engines; no cached-conditioning shortcut for only one side.
- [x] T031 Add held-out six-evaluation 640×480/124 cases with different seeds and prompts covering faces/hands, fine textures, camera motion and multiple moving subjects. Freeze selections before fixing the implementation; compare full trajectories and every decoded frame, not hand-picked screenshots.
- [x] T032 Qualify equivalent supported conditioning cases: first/last-frame FL2VA, one Ref2VA image under matched effective image-size policies, and a short reference video plus image. Preserve input order and audio treatment. Use cheap captured-input diagnostics before full renders; document h3-specific continuation policies without claiming an unsupported SGLang equivalent.
- [x] T033 Measure cold-process time to playable output and process exit separately; measure generation/stage time after model readiness. Use at least three interleaved pairs per primary case, including C2, no concurrent GPU jobs, and the same tracing-free inputs/settings. Report medians/ranges; keep sustained-session warm runs separate from fresh-process runs. Synchronize at measurement boundaries, not every operator.
- [x] T034 Collect total GPU usage on both engines at ≤100 ms intervals plus available allocator peak counters, host/pinned memory and swap. Retain per-step memory history through all 50 C2 evaluations to expose cumulative growth. Preserve collector overhead and measurement boundaries, compare both load/runtime maxima, and apply the joint performance/VRAM gates to fresh baselines instead of the old two-second samples alone.
- [x] T035 Compare pre-change and post-change fast CUDA with identical source-independent fixtures, fixed-algorithm diagnostics and normal production defaults. Verify unchanged seed/conditioning/latents/media where deterministic, dispatch choices, version/flags, decoder/quantized combinations, checkpoints and timings/memory. Do not regenerate golden outputs to conceal a regression.
- [x] T036 Test reference→fast→reference and fast→reference→fast sessions in one process, cache hits/misses, save/load conditioning and sampler resume, cancellation/error paths, and builds with/without optional cuDNN. Run relevant existing host/CUDA/preview/Metal compatibility checks for touched shared code; hardware qualification remains limited to the measured GPU.

T035 is closed by the completed before/after comparison and the user's explicit
acceptance of behavior differences. Its checkbox is not an unchanged-output
claim. Timing/sampled-memory medians pass for both tested fast variants.

T033 has all nine fresh-process pairs and complete stage records, but native
lazy loading and SGLang preloaded CPU weights still give different generation
readiness boundaries. It is closed by explicit acceptance alongside the failed
generation/decoder and memory-observation gates; the report does not subtract
native loading.

M6 exit: C0–C2 and held-out/conditioning content passes, including the complete
50-evaluation trajectory. Speed/memory limitations and fast behavior differences
are accepted by the user; raw gate outcomes remain in the report.

## M7 — Deliver inspection reports and the qualified reference path

- [x] T037 Generate a local HTML report with full-length SGLang/native players for C0–C2 and additional cases, synchronized seeking, difference/worst-frame views, temporal metrics, audio playback, commands and per-case content/latency/VRAM results. Label frame/evaluation counts and include C2's per-step drift and memory history. Link the unchanged fast regression artifacts separately; do not present fast CUDA as the reference candidate.
- [x] T038 Download all playback assets and records from the server and validate local links, checksums, media readability and frame counts. Keep source/configuration identities, metrics and small replay fixtures. Bound raw diagnostic captures to selected operations/steps under an explicit storage budget; never remove models, adapters, accepted outputs or required fixtures to make room.
- [x] T039 Document the final non-fast CUDA reference command/default selection, effective precision, memory policy and supported comparison scope. Preserve existing fast CLI/API behavior, explicit FP32 diagnostics and old state compatibility. State any new reference-policy identity/migration rule and limit parity claims to the tested geometry/evaluation combinations and RTX PRO 5000; C2 does not establish 362-frame/50-evaluation or other-schedule parity.
- [x] T040 Audit completion against the frozen manifest and all five acceptance gates, including mandatory C2 content, speed and peak-memory evidence. Publish the first-divergence/root-cause findings and fixes, final paired metrics, unchanged-fast evidence and remaining limitations. Close on the retained evidence and explicit user acceptance; preserve failed speed/memory checks and fast behavior differences without relabeling them as passes.

Completion records **SGLang content parity and user acceptance of the measured
speed, memory-observation and fast-mode differences**. Closing tasks does not
turn those measurements into passes. The accepted reference policy is the
default for ordinary BF16 CUDA video; explicit accelerated modes, Metal and
saved execution identities retain their selection rules.
