# B200 qualification

Historical record: renderer modes and legacy test launchers described below have been
retired. Retained artifacts preserve the original results. Use the
[single-pipeline design](design-single-pipeline.md) and
[current qualification](single-pipeline-final.md) for supported commands.

B200 passes native SM100, no-cuDNN, fat-binary and project-PTX execution,
default/model regressions, fast reference/state workflows, memory and bounded
sanitizer checks. The six matched 20-step pairs take **47–51 seconds with
`--fast-cuda` versus 119–158 seconds by default (2.4–3.1×)**. The existing
cuDNN-first implementation remains selected; no new SM100 inference change
was justified by these measurements.

B200 video/audio quality is **accepted** for the six matched 20-step pairs.
On 2026-09-18 the user confirmed: "B200 videos look and sound good"; see
[the acceptance record](cuda-fast-acceptance.json). This is overall approval;
individual clip scores and playback speed were not separately supplied.
The existing-GPU state handoff remains pending transfer permission and a
follow-up rendering allowance.
Evidence lives in `outputs/cuda-validation/b200`. Inference source is unchanged
from checkout `b4759a021c90eb07697440b1520fd4423acdf877`; the workflow source
digest is `2fc8a48f75a1453e88de06489519ff17b67a175973adb3c51394e510fadb43b9`.
Only qualification test helpers, their build target, benchmark instrumentation
and documentation change in this round. `--fast-cuda` remains opt-in and
default numerical tolerances are intact.

## Node and dependencies

The GPU reports **NVIDIA B200, SM100, 148 multiprocessors**, 183,359 MiB in
`nvidia-smi` and 178.35 GiB through CUDA. MIG is disabled and the power cap is
1,000 W. The host is x86-64 Ubuntu 24.04.3 with driver 580.105.08, CUDA 12.8.93,
GCC 13.3 and FFmpeg 6.1.1. The Xeon Platinum 8568Y+ exposes 192 logical CPUs;
its container has a 20.4-CPU quota, a 250,999,996,416-byte memory limit and no
swap. The host's physical-memory figure in `--info` is not that container limit.

Source, dependencies and caches use `/path/to/h3.c-b200`. Both original BF16
models are in `/path/to/models/MiniMax-H3`. `/workspace` is backed by the
local overlay filesystem, with about 231 GiB free after setup, rather than a
network mount. The node was idle before qualification began.

The working driver, toolkit and held provider packages were preserved.
`scripts/setup_linux.sh` completed without replacing them. The qualified
**cuDNN 9.10.2.21 / cudnn-frontend 1.11.0** combination is isolated under
`outputs/setup/cudnn-9.10.2/usr` and `outputs/setup/cudnn-frontend`; installed
cuDNN 9.8 remains untouched. `env.sh` and `prepare-cudnn.sh` record the exact
build/link/runtime configuration and package extraction commands. This tests
one qualified stack; it does not claim that the provider's untouched cuDNN
9.8 stack or arbitrary newer releases are qualified on B200.

## Baseline and selected routes

Default SM100 uses the generic BF16 path. SM90 and SM120 fixed-head kernels
are not substituted for SM100. Fast mode selects cached cuDNN DiT attention,
cooperative QKV normalization/RoPE, cached/timed cuBLASLt projections and the
existing TF32 VideoVAE attention. Unsupported head dimensions and causal/GQA
attention retain their documented fallback. `H3_CUDA_REFERENCE=1` wins over
fast mode; execution failures abort rather than reusing partial output.

The six matched pairs use all 50 layers, 20 steps, reuse/core-reuse one,
original BF16 weights and no token reduction. Prompts, references, seeds,
schedules and geometry match within each pair. Times are **process wall time
through final MP4 generation**. Full-media validation adds 0.33–0.42 seconds
per invocation and is separately recorded and charged to the ledger.

| Case | Resolution / target frames | Default | Fast CUDA | Speedup |
| --- | --- | ---: | ---: | ---: |
| `1.jpg` | 288×384 / 56 | 146.42 s | 47.45 s | 3.09× |
| `2.jpg`, second seed | 288×384 / 56 | 124.21 s | 46.98 s | 2.64× |
| `2.jpg` + `body1.jpg` | 288×384 / 56 | 128.03 s | 51.19 s | 2.50× |
| Detail | 480×640 / 22 | 118.88 s | 49.73 s | 2.39× |
| Continuation segment 1 | 288×384 / 90 | 157.68 s | 50.90 s | 3.10× |
| Continuation segment 2 | 288×384 / 90 | 156.16 s | 51.23 s | 3.05× |

Segment two consumes a 39-frame overlap and delivers 51 new frames. All twelve
files pass full FFmpeg decoding, geometry/frame-count and 32-kHz stereo-audio
checks. The first pair is also the budget pilot and includes colder host/model
caches. Different corpus cases are not repeat measurements. These numbers
measure the existing fast path's benefit over default CUDA, not an additional
B200 tuning gain or an unmeasured production-length speedup.

An unchanged-inference repeat of `1.jpg` takes 48.78 seconds versus the
pilot's 47.45 seconds, about 2.8% slower. This bounds the observed short-run
variation, not a statistical distribution or an optimization gain. Its extra
output is outside the frozen six-pair quality decision. The corpus binary hash
is `7dab808ccd42e9e4a5004f7d1f94181754fed1a19a20f1a89a517d73cfd8f911`;
the final build/repeat hash is
`d5c0239db510bfeae76ca2f4018bae4f99cd79f1669cd34fddde1eccf62e2626`.
The added test build target changes build provenance; the inference source
digest and selected fast implementation remain unchanged.

The local gallery is `outputs/cuda-validation/b200/quality/review.html`.
`quality-acceptance.json` freezes all twelve MP4 and metadata hashes, with
commands, source/input/binary identity and timings in each case's JSON file.
Static sheets show coherent scenes without gross spatial corruption in the
inspected frames. The user's separate video/audio approval is recorded in the
frozen corpus manifest; per-clip checklist scores are not inferred. Empty sheet
cells are padding, not black frames in the videos.

For the `1.jpg` pilot, cumulative DiT load/denoise profile time falls from
37.70 to 13.65 seconds. Loading is included in those totals (19.71/10.19 seconds),
not a separate phase to add again. The denoise-phase increments are therefore
about 17.99/3.46 seconds. VideoVAE decoding takes 17.84/4.70 seconds; audio
0.60/0.76 seconds. Peak DiT tensor allocation is 36.69 GiB with no streamed
weights. Each context's pinned bounce peak is 32 MiB, not total host RSS.

Host costs dominate this short fast run: compatibility hashing takes 11.41
seconds and Qwen text processing 12.32 seconds, including 11.84 seconds of
source read/upload work. The default pilot's corresponding figures are
62.62 and 22.75 seconds. Default/fast use their existing hashing policies;
host/model cache state also differs. Do not attribute the complete speedup to
GPU arithmetic or use these cold single-process runs to rank B200 against
H200/H100 hosts. Reusing loaded state across requests and ordinary uploads
remain the established workflow; no new transfer policy is promoted here.

Isolated attention uses BF16, 56 heads, dimension 128, ordinary output layout,
one warmup and three timed calls:

| Tokens | Default generic | Native fast | cuDNN fast |
| ---: | ---: | ---: | ---: |
| 2,281 | 13.452 ms | 1.340 ms | 0.150 ms |
| 18,225 | 823.911 ms | 78.803 ms | 6.270 ms |
| 18,945 | 890.902 ms | 85.026 ms | 6.760 ms |

These results favor the existing cuDNN-first policy. A custom SM100 native
kernel is not required to complete qualification and has not been introduced.
The fast QKV operation takes 0.100/0.700 ms at 2,281/18,225 rows versus
1.925/15.251 ms by default. Decoder attention at 2,048 rows takes 0.650 ms
with TF32, versus 28.022 ms by default and 7.695 ms with the tiled candidate.
Component gains cannot be multiplied into a complete-render prediction.

## Production shapes and tuning decisions

Both ordinary and head-major output layouts pass at 33,757 and 110,212 tokens,
derived from 480×640 and 1344×768 at 362 frames with the quality prompt's 151
text rows and adapted `1.jpg` reference. `production-shapes.json` records all
video/audio/image/text row counts. No corresponding long video is rendered.

| Tokens | Native fast, warmed | cuDNN fast, warmed | cuDNN first operation |
| ---: | ---: | ---: | ---: |
| 32,768, rounded control | 249.19 ms | 20.35 ms | 927.37 ms |
| 33,757, derived | 264.81 ms | 22.02 ms | 1,025.31 ms |
| 110,212, derived | 2,815.38 ms | 249.12 ms | 1,232.02 ms |
| 110,592, rounded control | 2,831.84 ms | 249.50 ms | 1,302.54 ms |

First-operation measurements include cold library/plan work and submission;
they are distinct from warmed reuse. Three warmed samples per case are retained
in JSON. For example, the 18,225-token cuDNN samples span 6.262–6.280 ms,
native samples 78.801–78.805 ms, and 2,048-token TF32 decoder samples
0.646–0.654 ms. These are isolated-operation repeat ranges, separate from
complete-process variability. The large native/cuDNN gap supports using the library implementation;
profiling does not justify a custom SM100 attention rewrite in this round.

At 18,225 tokens, attention workspace limits of 0/32/128/512 MiB all select a
**zero-workspace cuDNN plan** and take about 6.27–6.28 ms. Unlike the H200 plan,
this B200 plan remains usable at a zero limit. That is not a recommendation to
set the global limit to zero: decoder attention also uses the scratch policy.
The native/no-cuDNN and unsupported-shape fallbacks are tested separately.
Keep the current 64-MiB scratch default.

F32 decoder projections (1,797 rows, K=2,048, N=6,144/16,384) are measured with
0/32/64/128-MiB GEMM workspace and tuning on/off. At N=6,144, existing tuning
improves roughly 0.94 to 0.83 ms; at N=16,384 it changes about 2.15 to 2.14 ms.
Larger workspace supplies no useful warmed gain. BF16 DiT sweeps cover 2,281
and 18,225 rows, four hot projections, 0/32/128-MiB workspace and tuning on/off.
Tuning helps some output/MLP projections, with extra first-call cost; larger
workspace does not establish a consistent gain. Keep the existing cached/timed
32-MiB policy. Cache keys include shape/type/bias/alignment within an immutable
device/library/workspace context, and bias addresses refresh per call.

Six decoder variants reuse the exact same saved 56-frame fast AV state.
Component wall times are 19.47 seconds for default, 6.44 for F32 GEMM, 9.10 for
tiled attention, and 6.19/7.14/6.44 for TF32 with 32/64/128-MiB GEMM workspace.
All are finite and fully decodable with the expected video/audio geometry.
Those component times exclude conversion/mux; the complete 62.28-second
experiment is charged to the shared rendering ledger. No alternate decoder
is promoted. The main quality pairs exercise the existing selected TF32 path;
decoder videos are outside their frozen review scope.

Nsight Compute cannot access this provider's performance counters
(`ERR_NVGPUCTRPERM`, exit 1). The attempted capture is preserved but is not
counted as a successful counter profile. Resource reports, unprofiled kernel
timings, cumulative phase profiles and sanitizer checks remain usable.

No alternate stack, larger default workspace, registered-upload policy or new
SM100 kernel is promoted. These are measurements of the existing fast path;
there is no claimed gain over the pre-tuning fast baseline.

## Regression evidence

The native cuDNN-enabled build passes `make test`, all 41 default primitive
comparisons, 70 exact tokenizer fixtures and 23 released-model comparisons:
vision, text, video encoder, audio decoder, video decoder, one DiT block and a
small full DiT evaluation. Architecture-specific SM90/SM120 tests correctly
skip this SM100 device. Fast shape, tail, batch/layout, reference-precedence,
unsupported-shape and zero-scratch checks pass.

Both 25- and 50-layer resident, streamed, automatic and injected-allocation-
failure fixtures pass. Default comparisons stay exact; fast fixtures check
valid outputs and memory behavior. The bounded default/native-fast racechecks
report zero hazards. Runtime/cache memcheck reports zero errors and zero
leaked allocations, including 64 asynchronous handoffs, 16 cancellations,
in-flight teardown and file-replacement/modification/pressure checks.

At 50 layers, resident execution peaks at 38,574,523,160 tensor bytes. Default
forced streaming peaks at 1,597,895,680 bytes and transfers 39,305,871,360 weight
bytes. Fast forced streaming retains 38,535,168,000 bytes in its bounded weight
cache, with total peak tensor bytes of 40,115,929,880. Injected allocation
failure releases partial residency and falls back to streaming in both modes;
it does not keep an unbounded cache. These are small-shape memory fixtures,
not measurements of the largest possible video.

The first fast sampler capture completes in 196.86 seconds, including its
uninterrupted control and cold full-model fingerprinting. Fresh-process restart
takes 23.29 seconds using the content-fingerprint cache. Explicit fast-to-default
checkpoint handoff also passes. Default two-step restart checks every completed
boundary exactly; the final independent audit confirms byte-identical AV files.
Ordinary and continuation restart use fresh processes. Continuation audits
cover protected prefixes, source-state immutability, changed face/body references,
39-frame overlap and delivery of 51 new frames. These fixture-operation times
include more than a single render and are not entries in the matched performance table.

Fast/default/fast mode switching, conditioning/prepared-DiT/decoder cache reuse,
reuse/core-reuse and token reduction pass. The pressure case
leaves only 4 GiB free, evicts the prepared cache while retaining the decoder,
completes another render and saves a checkpoint. The deliberately reserved
pressure allocation is separate from per-context tensor peaks.

The native fast attention kernel reports 205 registers per thread, 69,632
shared bytes, zero local bytes and one active 256-thread block per SM. These
resource figures do not constitute a hardware-counter profile; instrumented
sanitizer timings are not used as performance measurements.

The no-cuDNN native SM100 build passes fast dispatch/cache tests, all 41
primitive comparisons and a complete two-step image smoke (41.28 seconds
including full-media validation). Native and cuDNN-enabled fat builds each
pass all 41 primitive comparisons on B200. Native objects contain SM100 SASS
and compute_100 PTX; both fat objects contain SM86/89/90/100/120 SASS and
compute_86 PTX. Non-SM100 architectures are cross-compilation evidence only
in this round.

Global `CUDA_FORCE_PTX_JIT=1` fails at cuBLASLt handle creation and is not a
runtime pass. A separate **project-object compute_86 PTX-only build**, linked
normally to the vendor libraries, passes all 41 primitive comparisons and the
fast suite on B200. Object inspection confirms no project SASS. Its first
inventory check had a helper bug: a substring match treated cuobjdump's
"No ELF file found" diagnostic as an ELF entry. Matching actual inventory
records fixes the helper; the failed attempt and corrected runtime results
are both retained. This required no CUDA inference change.

The final default CUDA suite passes after the test-target changes. The local
Metal suite also passes with GPU access; its initial sandboxed attempt could
not initialize Metal and is retained separately. Optional local MLX/released
model fixtures are absent and report skips, rather than passes. Linux/macOS
setup/build-selection checks pass all 21 tests, and the budget helper passes
all four tests. The isolated cuDNN runtime still executes 9.10.2 plans with
`LD_LIBRARY_PATH` removed, using its embedded library search path.

Independent inspection validates geometry, finite latents and checksums for
**33 AV states**, plus whole-file and section checksums for **five sampler
states**. Their Ref2VA model fingerprint is
`2eed3a9f8c38f909ca8919cb6b744ceebb1bfe070bc4937244bfece87970f4d1`.
Default checkpoints retain default execution provenance; fast checkpoints
record implementation version one. All eight reference families and the
separate fast bridge pass full-media/state checks. The hard-continuation quality
pairs are included in the user's overall video/audio approval.

## Reproduction and measurement helpers

On this node, `source outputs/cuda-validation/b200/env.sh` selects the isolated
qualified libraries. `stage1.sh` records the baseline build/default/component
checks and paired pilot. `stage2.sh` covers attention/QKV, memory/sanitizer,
remaining quality and state/cache tests. `stage3.sh` contains the workspace,
saved-decoder and initial reference session. `stage3b.sh` preserves that batch
timeout, runs the targeted remainder and bridge check; `stage4.sh` contains
no-cuDNN/native/fat checks. `stage4b.sh` resumes the corrected PTX inventory
check and final regression/state checks. Keep the shared ledger
when running affected subsets; reruns are not a new 45-minute allowance.

`tests/cuda_fast.c` now records the first attention operation separately from
three warmed calls. `tests/cuda_fast_projection.c --dit ROWS` measures the
actual BF16 DiT dimensions: K/N 5,376/21,504 (QKV), 7,168/5,376 (output),
5,376/28,672 (MLP input) and 14,336/5,376 (MLP output), with first-call tuning
and repeated-context timings. Existing F32 decoder experiments remain intact.

The added `bin/cuda_references_test` runs eight two-step cases through one
reusable public-API context. This preserves reference-family coverage without
paying repeated process/model startup costs and exercises reference changes
between requests. The first batch hits its 300-second cap after completing
seven cases, while working on image-plus-audio. Its full 300.43 seconds and
partial log are retained. The helper now accepts an optional case name, so
only the unfinished audio case is rerun under a 60-second cap and passes in
41.52 seconds including validation. The wrapper
charges both attempts to the same ledger and independently checks all complete
MP4/AV states. Additional default FL2VA smoke duplicates and the extra
same-GPU default-to-fast AV handoff are omitted to preserve budget. Default
FL2VA released-model comparisons, six default Ref2VA quality renders and the
explicit fast-to-default sampler handoff remain covered. This is functional
evidence; its cached-session case times are not independent CLI-render benchmarks.

## Scope and pending gates

Per-render and per-component limits remain 300 and 120 seconds. All rendering,
including pilots, restart controls, failed attempts, decoder replay and mux,
shares the B200 ledger, which finishes at **2,697.21 / 2,700 seconds**,
including the timed-out reference batch and its targeted remainder. No `test2.sh` or 362-frame render is part
of this round; production shapes are covered by isolated operations.

B200 human video/audio acceptance covers the six frozen matched pairs, excluding
two-step functional clips, decoder outputs and the timing repeat.
Cross-GPU state transfer requires a separate decision: automatic approval review rejected uploading the
existing H200-derived state because that transfer had not been explicitly
authorized. No H200 state was uploaded. The earlier user-requested skip of the
Metal-to-H200 check remains in force. No folded LoRA/Turbo model is supplied,
and soft continuation is not implemented; supported hard/bridge behavior is
the applicable scope. Extra default FL2VA smokes and a same-GPU AV handoff
were omitted as documented above; no extended render or long-form quality
claim is made. T187/T192/T194 are complete with the B200 approval. T188 retains
the unperformed cross-GPU handoff; all other tasks in this qualification round
are completed within the stated scope.
