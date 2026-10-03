# Two-mode CUDA: M0/M1 implementation record

This record accompanies [the design](design-two-modes.md) and [checklist](single-pipeline-tasks.md).
Qualification records distinguish the unchanged reference gate, mutable policy checks and later fast-mode work.

## Starting point and ownership inventory

The runtime baseline is `84d1555` (bounded FP32 patch projection). The planning
commit is separate from the unchanged arithmetic-4 baseline. The local test device
is RTX PRO 5000 72GB Blackwell at the configured CUDA host, driver 595.91.07. Models
remain under `/path/to/models/MiniMax-H3`; no model-weight digests are generated.

| Owner | Existing responsibility | Migration boundary |
| --- | --- | --- |
| `src/h3cli.c`, `src/sglang/sglang.c`, `src/engine.c` | Mode resolution, CLI, host preparation, request scopes | Resolve one policy before folding/preparation; default reference, explicit fast |
| `src/execution.c`, `src/cuda/cuda_policy.c` | Thread-local fast request and immutable policy captured by constructors | Restore nested scopes; keep policy dependencies out of thin host tools; retain old low-level entry points until consumers migrate |
| `src/cuda/gpu_cuda.cu` | Per-context handles, arithmetic dispatch, stream/cache lifetime | Capture shared base separately from enabled fast substitutions; reference handles never borrow fast tuning |
| `h3_cuda_sglang*.cuh`, FlashAttention source | Accepted linears, norms, RoPE, attention and full VAE | Shared reference primitives; never delete merely because their names contain reference |
| `src/cuda/cuda_reference_wrappers.cuh`, generator | Mixed portable and SGLang dispatch | Update generators with wrappers; remove only proven dead legacy branches in M6 |
| `src/denoise/dit.c`, `src/denoise/dit_schedule.c` | Batch grouping, FP32 heads, attention/weight descriptors | Keep shared base semantics; explicit main-DiT overrides only |
| `h3_cuda_sage*`, `h3_cuda_sol*`, `src/cuda/cuda_quant.cuh` | Existing attention and packed projection kernels | Reuse behind fast policy; full combination qualification remains M3/M4 |
| `h3_cuda_fast*`, older dispatch/kernels | Old fast implementation candidates | No automatic promotion into the shared baseline; individual M2 measurements |
| `src/conditioning/conditioning.c`, keys in `src/engine.c` | Prepared/conditioning caches | Include base and fast identities; never share incompatible mutable contexts |
| `src/sampling/sampler_state.c`, `src/sampling/sampler_file.c` | RNG/recipe/build/version checks | Distinguish shared base from exact policy; old states must not silently become new fast |
| `src/media/decode.c`, presentation/AV state | Saved clean data and delivery identity | Decoder selection independent of denoising; data readers are not a legacy renderer |
| Image VAE/still, LoRA, continuation | Consumers with existing restrictions | Inventory now; full migration/qualification M5, before M6 deletion |
| Metal, portable host helpers | Non-CUDA execution | Preserve behavior, serialization security and shared primitives |

Legacy reachability includes automatic mode-zero fallback, explicit
`--cuda-reference legacy`, `H3_CUDA_REFERENCE=1`, old fast orchestration and
state restoration. M1 removes implicit video fallback and rejects the legacy
request spelling; final removal of internal consumers is M6. Generic helpers
still needed by Metal, image decode or production shared kernels are not dead code.

## Immutable regression contract

The M0 gate is `make test-cuda-reference-regression`, supplied with a new output
directory. The manifest records fixture and numerical output hashes. Models and
library locations come from caller configuration. Source and binary identities
are recorded for each run. The runner has no rebaseline, threshold, case-filter
or skip flags.

**Run the whole suite after every coherent code patch before the next unrelated
patch**, preserving recorded fixtures and golden outputs. This includes generators, build
configuration and test tooling. A failed gate requires a code fix and complete
rerun. Documentation-only edits need document validation. Missing GPU, timeout,
missing fixtures and stale source records are not passing results.

Each run builds an isolated source snapshot, records its digest and binary
identities, runs fixed production probes under a 720-second GPU/test deadline,
and checks the original source again. Tests hash source, binaries, small inputs
and outputs, never model weights. Integer/BF16/FP32 artifacts, decoded RGB and
PCM use exact hashes against the qualified native baseline; MP4 metadata is not
a numerical golden. The original SGLang full contract is separately retained.

Coverage includes CPU normal streams and 6/50 schedules, every Euler boundary,
real C2 steps 25/49, packed conditions, norm/linear/RoPE/dense attention,
six patch-batching geometries, image match/max preprocessing, vision, image and
video CNNs, soundtrack input and audio encoder, full VAE tile/audio decode,
complete C0 native preparation/trajectory/full media, and mixed contexts,
repeated allocation, cancellation and overflow recovery.

Mutable mode/CLI/state migration tests are separate from the frozen gate.
The separately frozen `tests/cuda_reference/fast-quality.json` sets future fast
quality and performance limits before optimization. Passing reference tests does
not qualify any Sage/SOL/quantized preset; paired visual evidence remains required
at the corresponding milestone. Historical speed/memory waivers are retained in
[the accepted results](cuda-sglang-results.md), not reused for new changes.

## Qualification records

The recorded golden state covers 12 bounded input fixtures and 204 exact
output artifacts. The accepted source/binaries and dependency identities remain
in `two-modes/baseline.json` and the preserved source directories; the original
accepted corpus and its contract are untouched.

Calibration measured 189.01 seconds for the fixed probes and full C0 render
(excluding build). C0 decoded RGB/PCM and all 36 one-based trajectory artifacts
match the accepted corpus exactly. Late zero-based C2 steps 25/49 use captures
numbered 26/50 and reproduce their accepted outputs exactly. The initial
pre-freeze harness expected 38 trajectory artifacts and used the wrong late-step
indexing; both harness errors were corrected before freezing. The initial failed
record is retained, alongside the repaired calibration; no production arithmetic
changed and the successful C0 render was reused for that calibration repair.

Fault injection rejects changed/missing fixture content, changed outputs and
missing/extra cases. The complete fresh-build replay and M1 results are recorded
below after execution. Current portable invocation is documented in
[CONTRIBUTING.md](../../CONTRIBUTING.md).


M0's complete unchanged fresh-build replay passed in **193.78 seconds** after
build (`two-modes/frozen-m0/result.json`), preserving all 204 artifacts exactly.
The recorded manifest digest above has not changed during M1. Build time is reported
separately and is not hidden inside the short-test target.

## M1 behavior and scope

M0 and M1 are complete: **T001–T014**. Later milestones remain unchecked.
The final source fingerprint is
`71263343ebcd7413f0dcfa579c2ca56f7e572b6c5431c37da240a36277baa937`.
The local working source was checked against the tested snapshot after download.

| Validation | Result |
| --- | --- |
| Final unchanged reference gate | PASS: 204 exact artifacts; 154.24 s after build, 262.58 s including isolated build |
| Complete native C0 render | PASS: 640×480, 124 frames, six evaluations, 120.95 s process wall; all updates, decoded RGB and PCM unchanged |
| CUDA policy host tests | PASS: 74 checks with optional engines; 76 checks in minimal/Metal builds |
| Sampler state tests | PASS: 1,780 checks on CUDA and Metal host builds |
| CLI and decoder options | PASS: 14 checks, including reversed conflicts, preview/full delivery and unchanged saved latents |
| GPU policy capture/isolation | PASS: reference → fast → reference, exact shared operators, allocation plateau and cancellation |
| Unsupported runtime/build | PASS: absent pinned cuBLAS fails; builds without SGLang/cuDNN reject both video modes before text/core loading |
| Thin host tools and cache | LoRA host target links; quant-cache and preview/presentation host tests pass |
| Local Metal | Final executable builds; BF16 primitives pass; 129-symbol compiled GPU contract passes |

[Open the playback report](../../outputs/two-cuda-modes/review.html),
[final gate record](../../outputs/two-cuda-modes/records/frozen-m1-verified/result.json),
[CLI/delivery records](../../outputs/two-cuda-modes/delivery/result.json), and
[context/build records](../../outputs/two-cuda-modes/records/m1-extra-validation.json).
The M0 and final M1 MP4 files also happen to be byte-identical; acceptance uses
the decoded pixel/PCM and trajectory checks, not MP4 container hashes.
Playback is supplied for inspection; no human visual review is claimed.

Seven fresh-build frozen runs passed, including M0 and each subsequent coherent
M1 revision. Post-build durations were 193.78, 178.32, 163.59, 155.80, 167.31,
160.75 and 154.24 seconds. Build-inclusive final duration is below five minutes;
the test deadline remains 12 minutes, with no case removal or relaxed threshold.
Their individual source identities, commands, timings and logs are retained
under the local report's `records/frozen-*` directories and on the CUDA server.

`h3_cuda_policy` resolves mode, shared base, fast identity, attention, weight
precision and presentation without mutating caller-owned parameters. Request
scopes restore it on return, error and cancellation; GPU contexts keep a copy.
Both video modes use shared SGLang preparation, RNG/sigma/Euler and sensitive
heads. The shared-base predicate is distinct from exact-reference intent.

Fast now has identity 2. Its old blanket tuning is disabled on the shared base;
explicit main-DiT attention and quantized-weight descriptors retain their
independent dispatch, with full preset quality/performance qualification deferred
to M3/M4. Dense protected/fallback attention uses the shared dense primitive.
Approximate attention retains the base's FP32 scale rather than legacy BF16
scale rounding. The reference arithmetic identity remains 4.

Preview and explicit full-VAE policies are orthogonal. Generation and decode-only
label nonstandard delivery outside full-output parity; they do not change saved
denoising. Unsupported dependencies/options fail before request preparation and
LoRA folding. Legacy mid-sampler execution is rejected before folding. Fast-v1
states fail version validation; shared-base fast-v2 states serialize both
identities. Full container/feature migration remains M5.

The older internal low-level helpers, still/image execution and generic data
readers remain pending their planned migration/removal. M1 does not claim that
M6's legacy-code deletion has already happened. Metal keeps its existing default
and passes the local host/policy/serialization checks.

## Retained development failures and limits

- The first mixed-context launch omitted the pinned cuBLAS runtime environment
  and failed clearly. Repeating under the frozen manifest's environment passed;
  the failed launch is retained in `m1-context.log`.
- The first decoder option smoke attempted full FP32 delivery of all 124 C0
  frames and exceeded its 90-second limit. It is a failed attempt, retained in
  `m1-cli.log`. Explicit FP32/balanced option coverage now uses a cropped
  64×64 / 22-frame saved-latent diagnostic; preview still delivers all 124
  frames. This crop is not an additional generation or quality qualification.
  The immutable gate still renders and fully decodes complete C0.
- A new presentation roundtrip test initially left nonzero bytes after an empty
  digest terminator and failed its whole-struct comparison. The mutable test
  now clears the field fully. The failure is retained in
  `m1-delivery-preview.log`; it did not require any golden change.
- Local Metal primitives pass. The broader Metal fixture executable cannot run
  without the absent `misc/fixtures/h3_dit.safetensors`. The old Python GPU
  declaration audit reports 46 missing declarations in its coverage list;
  `src/gpu.h` was not changed here, and the compiled contract check links all
  129 GPU symbols. These unrelated checks were not modified to hide failures.
- Actual GPU execution is qualified on the local SM120 only. Other GPU
  generations and the full new-fast preset matrix remain later milestones.
  Full fast-generation equivalence, optimization and performance are M2;
  the M1 mixed-context probe establishes shared operator isolation only.
