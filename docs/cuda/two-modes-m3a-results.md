# M3A reference CUDA exact substitutions — validation

Status: **complete — T053–T061**. This report supplements the
[M3A assessment](two-modes-m3a.md); it does not waive T021/T027.

[Playback gallery](../../outputs/two-cuda-modes/m3a/review.html) ·
[final qualification](../../outputs/two-cuda-modes/m3a/evidence/m3a-p3/result.json) ·
[closing frozen regression](../../outputs/two-cuda-modes/m3a/evidence/frozen-m3a-final/result.json)

The accepted combination lowers median request wall time by **8.77%** on the
placement-controlled workload, with identical media and unchanged peak VRAM.
All 204 unchanged frozen artifacts pass on final source. Reference VAE cast
fusion is deferred, and mapped Qwen weight retention was rejected.

## Implemented scope

Reference CUDA can pin private file-backed weight pages directly, sharing the
exact staging optimization with fast mode. Context creation captures storage
and cast policy independently of fast arithmetic. Reference remains recipe 4,
with the original dense attention, sampler, BF16/FP32 math and full decoder.
Fast remains recipe 2. Reference full-VAE Q/K/V casts remain separate pending
independent performance evidence for fusion.

Pinned accounting includes page alignment, with the original 40 GiB
process-wide admission cap and 16 GiB memory reserve. Unsupported registration
uses copied pinned storage; memory pressure uses bounded streaming buffers.
Successful releases are removed immediately so retry after a later unregister
failure cannot double-release them. New per-component diagnostics report actual
mapped/copied entries and fused/separate casts for either CUDA mode.

Model files must remain immutable while a request uses them. Metadata is checked
before lookup and after filling a new entry. Replacement or edits between
completed reads invalidate the cache key; detected changes during filling fail.
Private mappings prevent writeback but do not snapshot concurrent external edits.
No weight hashing, GPU residency expansion, library upgrade, approximate
attention or decoder-precision change is part of this port.

## Reproduction

Use the existing CUDA host the configured CUDA host, RTX PRO 5000 72GB, driver
595.91.07, model `/path/to/models/MiniMax-H3`, and pinned M0 dependencies.
The retained predeclared plan fixes three interleaved 243-frame / six-evaluation
same-reference-mode pairs, ratio-of-medians statistics, natural OS cache state,
and the existing timing/memory limits. A copied baseline uses
`H3_TEST_EXACT_NO_MMAP=1`; the final baseline also sets `H3_QWEN_PREFETCH=8`
to retain the original loader concurrency. Normal reference uses mapped DiT
staging and one Qwen prefetch lane. These diagnostics
are captured at context creation, not mutable runtime mode switches.

Full-VAE cast comparison uses the identical retained C0 AV state and three
interleaved decode pairs. `H3_TEST_EXACT_VAE_FUSION=1` enables only the diagnostic
reference fusion candidate. The normal full decoder remains separate from
`--preview-vae`.

The mutable probe `make bin/cuda_reference_exact` compiles backend internals with
fault injection confined to the test translation unit. It checks aligned budget
accounting, unaligned tails, cache hits, mutation/replacement between synchronized
reads, registration fallback, memory pressure, partial-cleanup retry, repeated
release/context lifetime, file-byte preservation, conversion bits, rounding
boundaries, signed zero, overflow/nonfinite faults and captured graph replay.
The production binary contains no injected failure switches.

The gate remains `make test-cuda-reference-regression`, using recorded golden
hashes, bundled fixtures and a fresh output directory. Run the entire gate after
every code change. Retain the expected outputs, complete case coverage and
evaluation counts; do not regenerate expected outputs to hide regressions.

## Initial evidence

The first patch passed all 204 unchanged reference artifacts in 142.387 seconds
after build, including complete native C0 rendering, conditioning/decoder
probes, large patch shapes and bounded late 50-evaluation fixtures. The source
digest is `07b65413e05e4bae2fadb31b1b63ae10cfc24e8841af42d59dcbd58510fd7bde`.
The new exact-substitution probe, existing mixed-policy probe and fast-v2 probe
all passed before performance runs began.

Local Metal builds successfully and exports all 152 declarations applicable to
its preprocessed header. Preview numerical checks (653,390 checks) and cache
isolation passed with device access. The broader `make test-preview` target
failed afterward because its sampler harness requests 20 evaluations while the
existing test-budget guard permits six or explicitly 50; its log is retained
separately. The preview numerical/cache checks themselves passed.
`tests/check_gpu_contract.py` also has a pre-existing stale inventory, including
CUDA-only declarations in its platform-independent list. Neither that inventory
nor the frozen reference suite was modified to make these checks pass.

## Primary timing campaign

All three 243-frame / six-evaluation pairs produced identical complete MP4s.
The ratio-of-medians result is retained as a **failed promotion attempt**:

| Boundary | Copied | Mapped | Mapped/copied |
| --- | ---: | ---: | ---: |
| Total wall | 153.185 s | 138.590 s | 0.90473 |
| Core loading | 40.316 s | 27.564 s | 0.68371 |
| Conditioning | 22.057 s | 23.975 s | **1.08699 — fail** |
| Denoising | 64.658 s | 65.158 s | 1.00774 |
| Full AV decode | 17.714 s | 17.965 s | 1.01416 |
| Peak VRAM | 8,737,193,984 B | 8,737,193,984 B | 1.00000 |
| Peak host RSS | 39,570,845,696 B | 39,571,562,496 B | 1.00002 |
| Peak component pinned bytes | 38,568,722,432 B | 38,569,541,632 B | 1.00002 |

Mapped staging first executes during transformer core loading, after the measured
conditioning phases; Qwen counters show no mapped/copied host-weight entries.
This makes run-to-run timing variation a plausible explanation, but does not
turn the failed statistic into a pass. The predeclared single complete timing
repeat uses the same source, boundaries, three-pair order and acceptance limits.

## Decoder candidate decision

Three identical-latent decode pairs produced identical complete MP4s. Fused
casts / separate casts median wall ratio was **0.97899** (2.10% lower wall),
which does not reach the existing 1.05× promotion target. Fusion is numerically
eligible but **deferred for reference defaults**. The separate conversion path
remains the reference default; the diagnostic candidate and existing fast-mode
fusion remain available for future measurements. No combined benefit is inferred
from unrelated loading measurements.

## Supplemental content and state comparisons

The copied/default and mapped/explicit-`sglang` reference runs matched all 117
preparation/trajectory captures in each of four complete 640×480 / 124-frame /
six-evaluation cases: C0 text-only, R0 image `match`, R1 ordered portrait plus
video/audio in `max`, and continuation from the retained reference AV state.
Decoded RGB and PCM streams matched exactly, and all four paired MP4 files were
byte-identical. Full AV states and raw captures remain on the server; playback
assets and checksum-bearing test records are retained for local inspection.
These captured runs are correctness evidence, not promotion timing samples.

## Conditioning investigation and protocol amendment

The single planned unrestricted repeat also failed conditioning: median wall
156.149 → 141.850 s (ratio 0.90843), but conditioning 22.496 → 28.975 s
(ratio 1.28799). All media remained identical, VRAM unchanged, and denoising /
decode ratios were 1.00641 / 1.00109. No failed statistic was converted to a pass.

Topology inspection found two Xeon Gold 6154 sockets / NUMA nodes; the GPU is
attached to node 0 (CPUs 0–17,36–53). Short input-only probes pinned separately
to nodes 0 and 1 did **not** establish a remote-socket penalty: text medians
13.090 / 12.928 s. A second, predeclared matched diagnostic compared unrestricted
workers with node-0-bound workers, keeping identical input-only capture settings.
Its order was unrestricted, bound, bound, unrestricted. Text medians were
14.593 / 13.035 s, a 1.11947 ratio; all captured inputs were identical and no
mapped staging or denoising executed. This supports worker-placement variability
as a conditioning confound, rather than a change to Qwen arithmetic. It does not
explain every second of variability in unrestricted complete requests.

After exhausting the one unrestricted repeat, the protocol was explicitly
amended for **one controlled three-pair campaign**, conditional on that diagnostic
showing a >5% placement effect. Both copied and mapped requests run under
`numactl --cpunodebind=0 --membind=0`; case, six evaluations, 243 frames, C/M–M/C–C/M
order, ratio-of-medians statistics, telemetry and acceptance thresholds remain
identical. This is a new placement-controlled experiment, not another unrestricted
retry or a relaxation of the frozen suite. Both unrestricted failures and both
NUMA diagnostics are retained. Application defaults and OS-wide affinity are not
changed. Performance claims from this experiment must identify its NUMA binding.

## Rejected second implementation candidate: shared Qwen staging

The controlled core-only candidate also failed conditioning, so its mapped
reference default was **not accepted** and the queued final promotion gate
stopped. The gains above remain evidence for core loading only, not a passing
whole-request qualification.

The second code patch extends the same exact host staging to large (at least
1 MiB) Qwen BF16 weight reads. This targets conditioning directly instead of
running further timing retries of the first candidate. The Qwen tensors remain
non-streamed; only their source upload uses the shared host cache. Tiny weights
keep the existing path, the global 40 GiB cap and admission fallback remain,
and the context releases its cache before the DiT component takes over. No
conditioning arithmetic, prefetch concurrency, checkpoint data or GPU residency
policy changes. `H3_TEST_EXACT_NO_TEXT_MMAP=1` isolates the new text-loading
substitution; `H3_TEST_EXACT_NO_MMAP=1` disables both text and core mapped staging.

Patch 2 has its own predeclared campaign: the unchanged frozen gate first,
updated lifecycle/non-streamed-byte tests, two interleaved input-only Qwen pairs,
one complete three-pair NUMA-0-bound 243-frame/six-evaluation campaign with all
original limits, then current mapped conditioning/continuation captures against
the retained copied baseline. Cast-fusion source and its deferral remain unchanged;
its numerical probe runs again. This is a new implementation candidate, not a
reclassification of the first candidate's failed measurements.


The second candidate passed all 204 frozen artifacts (143.109 s after build),
but isolated Qwen staging was **2.35319× slower** than the original bounce path.
It reached the 40 GiB cached-weight cap for weights read only once, adding
pinning/retention cost without DiT-style reuse. Further candidate renders were
stopped; the active baseline was allowed to finish. This extension was removed.

## Final implementation candidate: exact prefetch scheduling

A bounded comparison of the existing `H3_QWEN_PREFETCH=8` and `=1` settings used
original bounce uploads and identical node-0 placement in order 8,1,1,8. All
captured conditioning inputs matched. Text medians were 13.25194 s with eight
lanes and 12.61641 s with one (1.05037× throughput). The diagnostic's legacy
`node` field denotes **prefetch lanes**, not NUMA nodes; all four commands bind
to CPU/memory node 0. Its script and commands are retained to make this explicit.

Patch 3 removes Qwen weight retention and combines mapped **DiT** staging with
one Qwen prefetch lane for shared CUDA. CUDA uploads already serialize on one
copy stream; the change reduces competing loader threads while retaining the
same two-layer prefetch ring, GPU tensors, arithmetic and explicit override.
Metal keeps its existing eight-lane default. Logs expose actual lanes/depth.

The final comparison uses the original copied/eight-lane policy
(`H3_TEST_EXACT_NO_MMAP=1 H3_QWEN_PREFETCH=8`) versus the candidate defaults.
Both use node-0 placement, three interleaved 243-frame/six-evaluation pairs, the
original statistics and every original threshold. There are no further tuning
trials if this candidate fails. The frozen gate runs before qualification and
again on final source; current conditioned/continuation captures are compared
against the retained original copied/eight-lane outputs. VAE fusion stays deferred.


## Final candidate timing

The final candidate passes every timing/memory threshold in the predeclared
three-pair comparison. Both policies use NUMA-0 CPU/memory binding on this
server, 640×480 / 243 frames / six evaluations and the original full VAE.
These results do not establish an unrestricted-host or denoising speedup.

| Median boundary | Original copied / eight lanes | Mapped DiT / one lane | Candidate/original |
| --- | ---: | ---: | ---: |
| Complete wall | 143.254 s | 130.689 s | 0.91229 |
| Core loading | 30.745 s | 21.934 s | 0.71341 |
| Conditioning | 20.383 s | 20.671 s | 1.01412 |
| Denoising | 65.827 s | 65.966 s | 1.00212 |
| Full AV decode | 17.809 s | 17.784 s | 0.99862 |
| Peak VRAM | 8,737,193,984 B | 8,737,193,984 B | 1.00000 |
| Peak host RSS | 39,570,829,312 B | 39,571,181,568 B | 1.00001 |
| Peak component pinned bytes | 38,568,722,432 B | 38,569,541,632 B | 1.00002 |

Total wall is **8.77% lower (1.096× throughput)**, primarily from core loading.
Conditioning is 1.41% slower, within the unchanged 5% limit. Each paired complete
MP4 is byte-identical. Page-rounded pin accounting explains the 819,200-byte
reported pinned difference; GPU residency is unchanged. Medians are calculated
per boundary, so their sums need not equal median wall time. The mutable runner
labels any environment override as “instrumented”; the baseline selectors do
not enable profiling/captures or disable decoder graphs in these timing runs.


All four **final-source** supplemental replays also match the retained original
copied/eight-lane outputs: 117 preparation/trajectory captures per case, decoded
RGB and PCM, and complete MP4 bytes. Cases cover explicit `sglang`, image `match`,
ordered image/video/audio `max`, and continuation; the 243-frame timing requests
exercise default selection. Existing policy and fast-v2 probes pass alongside
the new staging/cast probe. No visual approximation was accepted for this port.

## Default policy and limitations

- Promote private mapped **DiT** staging and one Qwen prefetch lane in shared
  CUDA. Original metadata admission, copied/bounce fallbacks and streamed GPU
  slots remain. Explicit `H3_QWEN_PREFETCH` still overrides the lane count.
- Retain separate reference full-VAE casts. The 2.1% isolated fusion gain did not
  reach the promotion threshold; fast mode retains its existing fusion.
- Reject retained mapped Qwen weights: exact output, substantially slower loading.
- Keep arithmetic identities, dense reference attention, dependency pins,
  preview VAE, Metal's eight-lane default and reference goldens unchanged.

Use this qualified reference source for future fast-mode timing comparisons;
do not replace historical parity goldens or erase the failed M2/M3 evidence.
T021/T027 remain open independently. Performance is established only for this
server and the stated placement/workload; unsupported host registration may
fall back without the mapped-loading gain. Models must remain immutable during
use. No broader resolution/conditioning speedup is inferred from these samples.


## Closing gate and retained artifacts

The final clean-build `make test-cuda-reference-regression` passes **204/204**
original artifacts with no diagnostic overrides or NUMA binding: 136.599 s for
tests, 249.589 s including build. All recorded fixtures and golden outputs
remain unchanged. Every implementation candidate had a whole-suite
run; the final promoted source had another closing run after qualification.

| Identity | SHA-256 |
| --- | --- |
| Original pre-M3A source | `4c51f122af96a6462d726533655f22c45a0dfe83d31562a620d5996c5f5ee59e` |
| Final source | `6154af2d168eebe0154fc34178d0c7a25f79e8ec33a2921abcc83a8625bd5926` |
| Final clean-build CUDA `h3cli` | `17695e00f439781b8c0fe8ccf769be98fd6627efbb0dc878ac7b5dc417b282db` |
| Frozen suite manifest | `4ac45c94f1cbec276333102676ce1600ead71da025df4ca60b9ae2343098e413` |

The closing native C0 log proves actual default dispatch: reference mode, **200
mapped DiT entries**, Qwen **one lane / depth two**, and **2,268 separate VAE
Q/K/V cast groups**, with zero fused reference groups. The result records binary
hashes, commands, source files and every artifact; these are build/test hashes,
not checkpoint-weight hashes. Final local Metal builds and exports all 152
applicable API declarations; unrelated existing host-test failures are recorded
above rather than treated as passes.

The [local gallery](../../outputs/two-cuda-modes/m3a/review.html) pairs original
and final-source complete videos for all four content cases and links timing
clips, fusion evidence and unsuccessful attempts. Downloaded **461** records,
logs, drivers and playback assets were verified against their
[checksums](../../outputs/two-cuda-modes/m3a/m3a-review-manifest.json);
[all 92 local HTML links](../../outputs/two-cuda-modes/m3a/playback-verification.json)
resolve. No raw latent states, models or adapters were downloaded.

Server evidence is retained under `/path/to/qualification/two-modes/`:
`m3a-p1`, `m3a-p2` (stopped/rejected), `m3a-p3`, the bounded diagnostics and timing
attempts, and `frozen-m3a-{p1,p2,p3,final}`. The final tested executable is
`frozen-m3a-final/build/bin/h3cli`; raw captures and AV states remain there and in the
qualification directories. The archived campaign drivers reproduce selectors,
ordering, telemetry and exact comparisons; retained paths refer to that server.
No additional rendering or model downloads are needed to inspect the report.
