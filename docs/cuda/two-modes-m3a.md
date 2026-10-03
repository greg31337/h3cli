# M3A — Port exact fast-mode improvements to reference CUDA

Status: implemented and qualified; see the [M3A results](two-modes-m3a-results.md).
Mapped DiT staging and one-lane shared CUDA Qwen prefetch are enabled; reference
VAE cast fusion remains deferred. The assessment below records the original
evidence and qualification contract, not the final performance result. Tasks are in [M3A](single-pipeline-tasks.md#m3a--port-qualified-exact-improvements-to-reference).
This extends the [two-mode design](design-two-modes.md) using the retained
[M2/M3 results](two-modes-m2-m3.md).

## Recommendation and evidence

Port private file-mapped pinned weight staging to the shared SGLang pipeline,
subject to reference-specific qualification. Evaluate fused full-VAE casts
separately; their numerical evidence supports eligibility, but their performance
does not yet justify promotion. Neither requires changing reference arithmetic.

The final three interleaved 640×480 / 243-frame / six-evaluation pairs on the
RTX PRO 5000 produced byte-identical dense reference/fast videos:

| Median measurement | Reference | Fast dense | Interpretation |
| --- | ---: | ---: | --- |
| Complete wall time | 158.360 s | 134.380 s | 15.14% lower wall time |
| Transformer core loading | 44.185 s | 24.794 s | Main observed saving |
| Denoising | 64.764 s | 66.036 s | No denoising acceleration |
| Conditioning | 21.002 s | 22.400 s | 6.65% regression; fails the 5% limit |
| Full AV decode | 17.751 s | 17.534 s | Small difference, not isolated fusion evidence |
| Peak VRAM | 8,737,193,984 B | 8,737,193,984 B | Unchanged |

Source: [final timing record](../../outputs/two-cuda-modes/m2-m3/m3-final-timing/result.json).
These are medians per boundary; their sums need not equal median wall time.
An earlier campaign measured 6.11% lower total wall time, while three isolated
clean-latent full-VAE decode pairs had a fast/reference wall ratio of **1.00464**
(no meaningful gain), with identical media. See the
[earlier measurements](../../outputs/two-cuda-modes/m2-m3/m2-qualification/result.json).
The observed 6–15% whole-request savings are evidence for a candidate, not a
guaranteed reference-mode speedup or proof that every substitution contributed.

The final unchanged reference suite passed all 204 artifacts, but the two
substitutions were fast-only during that run. That pass does not qualify their
execution in reference mode. T021 remains open for conditioning performance;
T027 remains open because Sage2++, Sage3 and SOL failed the fixed C0/R1 quality
gates. M3A neither waives those failures nor waits for approximate attention to
qualify: its scope is exact shared execution.

## Candidate decisions

| Change | Code/evidence | Reference decision |
| --- | --- | --- |
| Private mapped pinned weights | `sglang_host_weight` in `src/cuda/gpu_cuda.cu`; 200 mapped DiT entries, unchanged tensor/media results | Primary candidate: removes allocation/zeroing and the `pread` staging copy, retaining streamed GPU slots |
| Fused full-VAE Q/K/V casts | `fast_vae_cast_qkv` in `src/cuda/cuda_sglang_vae.cuh`; same `__float2half_rn` conversions and sticky finite checks | Conditional candidate: measure independently; retain separate casts if gain is absent |
| Full-VAE buffer reuse | Already enabled for shared SGLang decoding in `src/vae/video_vae.c` | Already shared; do not count as a new port |
| Common attention validation/counters | Already shared; cancellation repairs concern Sage/SOL contexts | Keep existing correctness fixes; no new reference speed claim |
| Sage2++, Sage3, SOL; FP8/NVFP4; old blanket TF32 tuning | Approximate arithmetic, failed quality gates or future unqualified work | Fast-only; outside reference parity |
| Full DiT GPU residency | Roughly 36 GiB additional weights | Excluded from this port; conflicts with current VRAM objective |

## Implementation boundaries

Expose exact substitutions through captured, component-specific shared policy,
not by setting reference `fast_v2` or old fast math flags. Retain reference
arithmetic identity 4, fast recipe 2, dense attention, library/math choices,
sampler, state compatibility and original full-decoder precision boundaries.
Keep preview VAE and Metal behavior independent. Use internal diagnostic
switches for same-reference-mode A/B tests, not a third public CUDA mode.

For weight staging, preserve metadata keys, range validation, process-wide
40 GiB host budget, memory admission with 16 GiB reserve, streamed slots and
copy-stream release fences. Audit page alignment and account for actual pinned
pages as well as payload bytes. Unsupported mapping/registration must retain
the copied pinned/bounded bounce fallback. Model weights must never be written
or hashed. A writable `MAP_PRIVATE` registration cannot write back to the file,
but is **not an immutable snapshot of a concurrently edited file**. Audit the
immutable-model lifetime assumption, replacement/in-place mutation handling,
and metadata invalidation using small synthetic files; do not claim the existing
post-registration `fstat` covers all later edits. Reject detected changes safely.
Test partial initialization, unregister failures, cancellation and repeated
release so neither mappings nor the process-wide budget leak or double-release.

For VAE fusion, preserve rounding, signed zeros, overflow/nonfinite detection,
sticky faults, scratch bounds and stream ordering. Verify kernel tails and
graph replay. Test identical saved latents independently of denoising; retain
the original casts unless a measured benefit meets the existing promotion rule.

## Qualification and rollout

Predeclare candidate toggles, run order, cache conditions, timing boundaries and
statistics before measurement. Use at least three interleaved same-reference-mode
A/B pairs at 640×480 / 243 frames / six evaluations, with serialized GPU work and
unchanged libraries. Attribute weight staging and cast fusion separately. Keep
the existing promotion contract: target at least 5% total-wall or denoising gain,
smaller effects only with explicit measured combined benefit, no unexplained
stage regression, and bounded memory. Resolve the conditioning ratio above
1.05 before promotion; do not substitute a more favorable statistic afterward.
Retain invalid telemetry attempts; the existing one-second sampling-gap limit
must not be relaxed. Report host RSS/pinned residency as well as peak VRAM.

Supplement the frozen suite with mutable tests for new lifecycle/fallback paths,
default and explicit `sglang` selection, mixed reference/fast contexts, image
`match`/`max`, ordered video/audio conditioning, continuation and clean-latent
decoding. Reuse bounded fixtures and six-evaluation 124-frame clips; no new long
50-evaluation campaign. Keep large-shape coverage in the existing operator gate.
Retain paired playable output, decoded RGB/PCM comparisons and commands in HTML.

After **every code change**, run the whole unchanged
`make test-cuda-reference-regression` gate before the next unrelated patch.
The final task repeats it on the final source with the promoted reference paths
actually exercised, evidenced by separate mutable dispatch diagnostics. A pass
that only takes copied-weight/unfused fallbacks cannot qualify their replacements.
Keep all 204 expected artifacts, fixtures, golden hashes, case selection and
evaluation counts unchanged. No rebaseline or waiver.

Enable only individually qualified candidates in reference by default. Record
any rejected/deferred fusion result. Later fast comparisons must use the newly
qualified reference performance baseline while retaining the historical parity
oracle and prior results; a shared loading improvement is no longer a fast-only
advantage. Publish exact source/build identities and all passing/failing evidence.
