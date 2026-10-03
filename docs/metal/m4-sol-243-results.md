# Conservative mixed SOL: 243-frame experiment

M2 closed on explicit user acceptance of the measured dense performance and
B6 quality. The user deferred the remaining M2 conditioning/continuation corpus
and chose conservative SOL next. This experiment implements T155/T156 on that
baseline; it does not transfer the dense quality exception to approximation.

## Implementation

Attention checkpoint version 4 adds `--sol-dense-steps` and
`--sol-dense-sigma`. Dense evaluations use the absolute schedule index, including
on resume; the sigma rule uses the maximum video/audio sigma with an inclusive
threshold. Both fields round-trip and affect prepared-object identity. The
persisted sampler index/schedule supplies the counter; no restartable separate
counter is introduced.

SOL recipe 1 uses the existing scaled FP16 exact attention arithmetic, BF16
centroid storage, and FP32 routing, centroid attention, softmax and merging.
Recovery heads are forced fully dense before exact/approximate counters are
reported. The merged result passes all-head validation before any output commits.
Protected query and key blocks, local temporal neighborhoods and partial tails
remain exact. Original weights, residual state, conditioning, projections and
sampler arithmetic are unchanged. Dense mixed recipe 4 remains selectable.

## Operator and host validation

Ten Metal operator cases pass, covering all-exact, mixed routes, constant keys,
extreme logits, FP32 recovery, rejected nonfinite input, input/output layouts,
tails and scratch reuse. Small cases use an independent scalar double-softmax
oracle for exact tokens and weighted centroids. Protected rows and recovery
heads are compared with the same dense arithmetic; input hashes and destination
guards remain intact. Protected-row L2 is zero throughout the real-QKV sweep.

Metal API validation passes. Full shader instrumentation exceeded the device
resource limit for `h3_sol_2_32`; this failure is retained in
`outputs/metal-native-m4-sol-243/validation-notes/`. It is not reported as a shader
validation pass. Host tests include 130,447 layout/policy assertions covering all
Ref2VA modalities, anchors and 39/90/141/192-frame prefixes; 1,616 sampler state
checks; malformed-container and stale-record checks. Sampler ASan/UBSan passes.

## Bounded real-QKV sweep

Frozen gates and sweep: `tests/metal_sol_mixed_limits.json`. Inputs are retained
640×480×243 H3 captures, S=22,426, H=56, D=128. The operator harness overlays
synthetic protected ranges; it is not a substitute for model layout validation.
Each timing uses three warmed alternating dense/SOL pairs with all adapters,
summaries, routing, merging and output checks inside the timer. Fixed Q/K blocks
are 32/64, tau=1 and local radius=1. These are preliminary component screens,
not whole-step B5 acceptance or perceptual B6 claims.

| Layer | Minimum exact | Dense time / SOL time | Attention relative L2 | Combined screen |
| --- | ---: | ---: | ---: | --- |
| block0 | 90% | 0.953× | 0.012432 | FAIL |
| block0 | 75% | 1.096× | 0.024436 | FAIL |
| block0 | 50% | 1.400× | 0.046268 | PASS |
| block24 | 90% | 0.975× | 0.025228 | FAIL |
| block24 | 75% | 1.107× | 0.041963 | PASS |
| block24 | 50% | 1.428× | 0.074389 | FAIL |
| block49 | 90% | 0.959× | 0.004427 | FAIL |
| block49 | 75% | 1.055× | 0.009370 | FAIL |
| block49 | 50% | 1.250× | 0.031314 | PASS |

No single setting passes both the 1.10× component speed screen and L2≤0.05 /
cosine≥0.998 across every sampled layer. The 90% floor is slower; the 75% floor
has small gains; the 50% floor has larger gains but excessive middle-layer
approximation error under the preliminary screen. Limits were frozen before
these measurements and were not refitted. This bounds the current experiment;
B5/B6 production qualification is not opened by these results.

## Integrated diagnostic

The matched dense/SOL B2 experiment retains 243 frames and all 50 layers. It
uses the 75% floor, one dense initial evaluation, one dense initial layer and
sigma threshold 0.99. Per-step input/velocity/latent captures make this a
diagnostic timing run. Playback uses the preview VAE and is explicitly separate
from six-evaluation production-VAE qualification. Both runs pass all 100-block execution and range checks. SOL uses 34 recovered
head evaluations with **zero invalid heads**. The first video/audio input,
velocity and latent are byte-identical to dense, as are both second-evaluation
inputs. The dense baseline's first velocities also match the accepted M2 B6
executable byte for byte.

At the second evaluation, video/audio sigma is 0.923076928/0.75. Layer zero stays
dense; the other **49 layers route**. Actual block pairs are **83.398% exact**;
1,426 conditioning/anchor rows are protected. Recovery-forced routes are
included in the reported counts.

| Diagnostic measurement | Dense FP16 | SOL (75% minimum exact) |
| --- | ---: | ---: |
| First evaluation | 125.281 s | 125.082 s |
| Second evaluation | 125.060 s | 115.099 s |
| Peak tracked Metal memory | 39.905 GiB | 41.153 GiB |

The routed evaluation improves by **1.08655×**. These two-evaluation capture
runs are diagnostic evidence, not a B5 steady-state pass. The combined two-step
time includes the deliberately dense first evaluation. Incremental SOL versus
dense video/audio velocity L2 is **0.059224 / 0.010825** and final latent L2 is
**0.079977 / 0.022847**. These are numerical diagnostics, not a perceptual verdict.
No production-VAE B6, Ref2VA or continuation render qualification is claimed.

[Open the paired diagnostic playback](../../outputs/metal-native-m4-sol-243/review/review.html)
and [checksummed measurements](../../outputs/metal-native-m4-sol-243/review/report.json).

**Disposition:** T155/T156 implementation is complete. M4 production qualification
stays open; retain the accepted dense baseline. The bounded sweep found no preset
passing all preliminary screens, and integrated diagnostic speedup is below the
1.10× target. B5/B6 and held-out render qualification remain conditional on an
improved conservative preset. The existing BF16 SOL merge regression also passes
its CPU oracle and protected-row/guard checks after the shared merge refactor.

## Evidence

* Frozen executable, complete compiled source snapshot and checksums:
  `outputs/metal-native-m4-sol-243/frozen/`.
* [Operator and component records](../../outputs/metal-native-m4-sol-243/validation/index.json).
* [Host tests](../../outputs/metal-native-m4-sol-243/host-tests.log) and
  [final policy sanitizer results](../../outputs/metal-native-m4-sol-243/sampler-policy-sanitize.log).
* [Accepted M2 B6 playback](../../outputs/metal-native-m2-243/continued/accepted/review.html).

No mandatory human review, longer render, ANE work or production-default change
is introduced. Ref2VA/continuation render qualification remains outstanding;
host protection correctness is distinct from generated-clip quality.
