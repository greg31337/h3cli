# Legacy cleanup qualification

The completed task list is archived in [legacy-cleanup-todo.md](../legacy-cleanup-todo.md).

Status: **complete**. CUDA/Metal builds, the retained suite, 54 render/state jobs
and the final **204/204 exact SGLang gate** passed. Old saved files are deliberately
unsupported.

This change removes the public CUDA mode selector, continuous Ref2VA pipeline,
old-state readers, sidecarless decode, retired request fields, unused identity
wrappers/component hashing and no-op execution and VAE shims. CUDA video derives arithmetic identity 4 internally. Metal/still
math, explicit attention/precision/approximation controls and their combination
guards remain supported. Current AV decode requires its matching backend recipe.

The saved-file compatibility break is intentional. Fresh files use AV schema 3,
presentation 8, sampler 2, conditioning 2 and upscale envelope 2. See the
[current contract](../features/current-state-contract.md). Geometry profile 0/1
and source/refinement records remain distinct active semantics. There is no
migration or automatic interpretation of older saved files.

## Baseline and protected contract

Initial commit: `c8bd8403a5e44b9bb50d68c48c2b5b6cb76cddfb`, initially clean.
The recorded SGLang manifest SHA-256 remains
`fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3`.
All 12 fixtures and 204 expected output hashes are unchanged. The unchanged
baseline passed 204/204 in 95.77 seconds after building. Intermediate cleanup
candidates also passed all 204 outputs (99.80, 96.88, 96.37, 97.03, 94.94 seconds).
The final fresh gate passed **204/204** after the full suite and render matrix,
in **102.38 seconds** after building (**194.62 seconds** including the isolated
build). Its manifest matches the initial Git revision byte for byte; all twelve
fixture checksums are unchanged.

- [Final result, all 204 hashes and source/binary/config identities](../../outputs/legacy-cleanup/final-gate08/result.json)
- [Exact gate commands and durations](../../outputs/legacy-cleanup/final-gate08/commands.json)
- [Final parity video](../../outputs/legacy-cleanup/final-gate08/video.mp4)
- [Fixture audit](../../outputs/legacy-cleanup/protected-fixture-audit.json)
- [Local/server source comparison](../../outputs/legacy-cleanup/source-sync-final.json)

Final gate source SHA-256: `356e04eaa24eeafb72272fae3ff4f24015b034fda02bda496eca21f7ad0a6da8`. The render matrix uses
binary `4344868b47da72f77820b42436b902cceee3d8e3c4ec4863039dd4d08a3ece80`.
The final isolated build records binary `657aef2462d93543591a60e1690b4b4b9be8ad6dc95d7869cbc8add65d2279bd`.
Their production sources and sampler build identity are identical; isolated CUDA
builds include their own debug paths. The final server executable and library
are installed in `~/h3cli/bin/`. Host configuration and existing server-only user
scripts were preserved; the old source/binaries are backed up under the ignored
`outputs/legacy-cleanup/root-before-cleanup/`.

The recorded `final.h3av` is schema 1. Only
[`tests/reference_av_fixture.h`](../../tests/reference_av_fixture.h) can load
that exact checksum-bound fixture for probes. Every numerical operation still
uses current production code. No expected output is calculated in the adapter,
and the adapter is absent from the CLI/library. The VAE probe now requests
`default`; the retired bit is removed from its lifetime assertions. Coverage,
strict comparison, source-change checks and 12-minute post-build deadline remain.

## Environment

CUDA qualification uses the authorized RTX PRO 5000 at `cuda-test`, 72 GB,
SM120. The GPU identity is retained in private run records. Work and evidence are under
`~/h3cli/outputs/legacy-cleanup/`, sourced from the existing `~/h3cli/cuda-env.sh`.
Models resolve to `/path/to/models/MiniMax-H3`; component paths are recorded by
each render command. This does not change portable production defaults.

The existing qualified stack is CUDA 13.0.3 / nvcc 13.0.88, cuBLAS 13.1.1.3,
cuDNN 9.20.0.48, pinned CUTLASS and cuDNN frontend, with Sage, SOL and SubBlock
enabled. No dependency was upgraded. Sage's architecture-specific code now uses
an explicit `compute_120a`/`sm_120a` gencode pair so generic SM120 PTX is not
incorrectly emitted for scaled MMA instructions. The baseline used the same
compiler override before the Makefile fix.

Local qualification uses the M4: a clean Metal executable/library build,
current host/API checks and GPU primitive checks passed. All 117 retained
CUDA-compatible targets compiled; eleven Metal/Mach/ANE diagnostic targets
compiled locally. The fat multi-architecture executable is an optional packaging
target, outside this native SM120 qualification. The first sandboxed GPU
attempt could not initialize Metal; the authorized direct retry passed.

## Retained tests and exclusions

`make test` includes setup-script mocks, current CLI, sampler and malformed-file tests, continuation,
bridge, reference media/layout, posterior math, conditioning, still, preview,
upscale, attention/precision policies, LoRA runtime, residency and platform GPU
operators. `make test-current-host` is the host subset. Shared prerequisites
avoid repeating whole suites for each feature target. The tokenizer conformance
fixture generator was ported from Objective-C to C; the same collision and
multimodal assertions run on both backends. Verbose tokenizer logging is a
Metal diagnostic; portable CUDA remains quiet while preserving identical IDs.

`make test-current-cuda` runs the explicit supplementary inventory in
[`tests/current_cuda_suite.py`](../../tests/current_cuda_suite.py): bounded
real-model reference/continuation/resume/reuse integration, the twelve-option
attention/precision matrix, Sage failure recovery, SOL/SubBlock contexts,
conditioning replacement, placement/cancellation/resume, still and full-VAE
operators, memory, and decoder failure/retry/lifetime checks. A fresh current
CUDA AV state is required. The numerical gate runs separately.

The final aggregate `make -j1 test` passed in **284.77 seconds**, with the
CoW-dependent tests using scratch on `/models`. All **18 supplementary check
groups** passed after the affected bridge, continuation and matrix probes were
ported. The initial failures remain recorded. The nine successful reference
cases were retained, and every remaining integration tier reran; no assertion
or case was dropped. Eleven additional quantization operator/failure checks
passed. See the [suite summary](../../outputs/legacy-cleanup/current-suite-summary.json),
[aggregate log](../../outputs/legacy-cleanup/root-full-suite08.log),
[initial supplementary record](../../outputs/legacy-cleanup/current-cuda07/result.json)
and [retest commands/status/durations](../../outputs/legacy-cleanup/remaining-qualification.json).

Setup mocks passed 27 cases; bridge CLI checks passed 41 on CUDA and Metal;
adaptive CLI passed eight and VAE tile policy passed three. The CPU gate-runner
checks also passed. The [build inventory](../../outputs/legacy-cleanup/build-inventory.json)
records every retained target and platform classification.

Long performance/quality campaigns, archived oracle generation and benchmark
report reproduction are not part of current functional acceptance. Optional
standalone diagnostics remain available. No old missing-fixture skip in the
retained aggregate is counted as a pass. SM90 tuning is compiled but excluded
on SM120 hardware; its target explicitly reports that architectural exclusion.
The CUDA suite excludes the Metal-only CLI guard, which passes on the local M4.
All 37 LoRA runtime tests passed with scratch under `/models`, including the
CoW failure check skipped on the root filesystem. Offline LoRA training tooling is a
separate optional Python project; native folding and runtime LoRA are covered
here using a bounded synthetic nonzero rank-one adapter, not a trained model.

Focused ASan/UBSan checks passed 3,178 sampler assertions, 222 malformed sampler
containers, reference-video preprocessing/layout, 48 conditioning assertions,
36 malformed conditioning containers and 430,919 SOL layout assertions.
Tokenizer ASan/UBSan passed 247 conformance and 240 multimodal checks. Decoder
failure checks passed 37 runs covering partial reads, allocation failures,
integer overflow, cancellation, stalled children and successful cleanup. CUDA
decoder lifetime/recovery checks also passed streamed/materialized equality,
allocation plateaus, cancellation, failed admission, sink/mux failure and clean
retry. The real adaptive/SubBlock checkpoint passed 17 warmup corruption and
override checks. The sanitizer-discovered impossible allocation was fixed by bounding sampler element
counts by `PTRDIFF_MAX / sizeof(float)` before allocation.

## Render qualification

All R01–R18 cases passed: 54 jobs, 46 delivered media files, and 18 exact
state/content comparisons. The matrix uses 640×480, 124 frames, six steps, seed 42 by default.
State/feature controls use freshly generated current files. The larger reference
case is 1344×768. Its automatic planner retained all 50 BF16 blocks
(38,535,168,000 bytes), with no streaming slots. Partial and streamed plans were
separately qualified by the exact residency and memory-pressure tests. Upscale
sources are 320×256, refined to 640×512 with two steps. Audio reference coverage
uses an image plus audio; audio-only generation is unsupported. Adaptive plus
SubBlock is text-only; reference SubBlock is tested separately to preserve the
existing adaptive-reference guard. Every delivered file passed ffprobe, hash
verification and strict complete FFmpeg decoding. Video quality acceptance is
not inferred. Contact-sheet inspection found no blank/corrupt samples; some six-step cases have tight subject crops
or abrupt first/last-anchor transitions; the six-step still also has visible
ghosting. All 46 media files passed the structural/decode audit. Every video has
finite, non-silent PCM with zero clipped samples; audio durations match the
delivered frames within the declared tolerance. These checks are not a listening
review or a subjective sound-quality approval.

The [local review gallery](../../outputs/legacy-cleanup/renders07/review.html)
contains every delivered sample, exact commands and wall times. The
[matrix record](../../outputs/legacy-cleanup/renders07/result.json),
[media audit](../../outputs/legacy-cleanup/renders07/media-audit.json),
[input identities](../../outputs/legacy-cleanup/renders07/inputs.json) and
[inspection notes](../../outputs/legacy-cleanup/renders07/inspection.json)
retain the detailed evidence. Large sampler/AV/upscale artifacts remain on the
PRO 5000 under `~/h3cli/outputs/legacy-cleanup/renders07/`.

## Render timings

All 54 jobs passed, with 46 delivered files and 18 exact state/content comparisons.
Wall time includes the complete CLI invocation (loading, generation, state I/O,
decode and mux). Post-run validation and download time are separate. The table
uses the final successful attempts; previous attempts remain in `attempts/`.

| Job | Wall (s) | Sampled peak GPU (MiB) | Media / log |
| --- | ---: | ---: | --- |
| R01-text | 89.57 | 40380 | [MP4](../../outputs/legacy-cleanup/renders07/R01-text.mp4) |
| R01-upscale-source | 71.94 | 40380 | [MP4](../../outputs/legacy-cleanup/renders07/R01-upscale-source.mp4) |
| R02-first | 67.79 | 40526 | [MP4](../../outputs/legacy-cleanup/renders07/R02-first.mp4) |
| R03-last | 68.18 | 40526 | [MP4](../../outputs/legacy-cleanup/renders07/R03-last.mp4) |
| R04-anchors | 70.24 | 40652 | [MP4](../../outputs/legacy-cleanup/renders07/R04-anchors.mp4) |
| R05-image | 117.74 | 42692 | [MP4](../../outputs/legacy-cleanup/renders07/R05-image.mp4) |
| R06-images | 121.20 | 42986 | [MP4](../../outputs/legacy-cleanup/renders07/R06-images.mp4) |
| R07-audio | 75.56 | 40560 | [MP4](../../outputs/legacy-cleanup/renders07/R07-audio.mp4) |
| R08-video | 103.90 | 41734 | [MP4](../../outputs/legacy-cleanup/renders07/R08-video.mp4) |
| R09-silent | 105.51 | 41708 | [MP4](../../outputs/legacy-cleanup/renders07/R09-silent.mp4) |
| R09-soundtrack | 101.14 | 41754 | [MP4](../../outputs/legacy-cleanup/renders07/R09-soundtrack.mp4) |
| R10-mixed | 103.34 | 42243 | [MP4](../../outputs/legacy-cleanup/renders07/R10-mixed.mp4) |
| R11-full | 14.36 | 5760 | [MP4](../../outputs/legacy-cleanup/renders07/R11-full.mp4) |
| R11-preview | 3.78 | 1100 | [MP4](../../outputs/legacy-cleanup/renders07/R11-preview.mp4) |
| R12-reference-cache | 92.85 | 42680 | [MP4](../../outputs/legacy-cleanup/renders07/R12-reference-cache.mp4) |
| R12-reference-pause | 109.38 | 42692 | [Log](../../outputs/legacy-cleanup/renders07/R12-reference-pause.log) |
| R12-reference-resume | 62.21 | 42680 | [MP4](../../outputs/legacy-cleanup/renders07/R12-reference-resume.mp4) |
| R12-text-cache | 53.58 | 40380 | [MP4](../../outputs/legacy-cleanup/renders07/R12-text-cache.mp4) |
| R12-text-pause | 71.76 | 40380 | [Log](../../outputs/legacy-cleanup/renders07/R12-text-pause.log) |
| R12-text-resume | 41.40 | 40380 | [MP4](../../outputs/legacy-cleanup/renders07/R12-text-resume.mp4) |
| R13-bridge | 78.67 | 39344 | [MP4](../../outputs/legacy-cleanup/renders07/R13-bridge.mp4) |
| R13-hard | 100.62 | 38842 | [MP4](../../outputs/legacy-cleanup/renders07/R13-hard.mp4) |
| R14-anchors-inspect | 0.22 | 22 | [Log](../../outputs/legacy-cleanup/renders07/R14-anchors-inspect.log) |
| R14-anchors-pause | 22.14 | 38176 | [Log](../../outputs/legacy-cleanup/renders07/R14-anchors-pause.log) |
| R14-anchors-resume | 16.92 | 38176 | [MP4](../../outputs/legacy-cleanup/renders07/R14-anchors-resume.mp4) |
| R14-anchors-source | 39.56 | 37772 | [MP4](../../outputs/legacy-cleanup/renders07/R14-anchors-source.mp4) |
| R14-anchors-upscale | 26.31 | 38176 | [MP4](../../outputs/legacy-cleanup/renders07/R14-anchors-upscale.mp4) |
| R14-sampler-import | 10.10 | 5764 | [MP4](../../outputs/legacy-cleanup/renders07/R14-sampler-import.mp4) |
| R14-text-inspect | 0.17 | 22 | [Log](../../outputs/legacy-cleanup/renders07/R14-text-inspect.log) |
| R14-text-pause | 21.89 | 37944 | [Log](../../outputs/legacy-cleanup/renders07/R14-text-pause.log) |
| R14-text-resume | 16.72 | 37944 | [MP4](../../outputs/legacy-cleanup/renders07/R14-text-resume.mp4) |
| R14-text-sampler-source | 41.10 | 37690 | [MP4](../../outputs/legacy-cleanup/renders07/R14-text-sampler-source.mp4) |
| R14-text-source | 36.55 | 37690 | [MP4](../../outputs/legacy-cleanup/renders07/R14-text-source.mp4) |
| R14-text-upscale | 25.91 | 37944 | [MP4](../../outputs/legacy-cleanup/renders07/R14-text-upscale.mp4) |
| R15-lora | 740.70 | 40380 | [MP4](../../outputs/legacy-cleanup/renders07/R15-lora.mp4) |
| R15-still | 175.98 | 37570 | [PNG](../../outputs/legacy-cleanup/renders07/R15-still.png) |
| R15-still-decode | 79.30 | 9716 | [PNG](../../outputs/legacy-cleanup/renders07/R15-still-decode.png) |
| R16-adaptive-subblock | 82.60 | 41252 | [MP4](../../outputs/legacy-cleanup/renders07/R16-adaptive-subblock.mp4) |
| R16-adaptive-subblock-resume | 36.14 | 40072 | [MP4](../../outputs/legacy-cleanup/renders07/R16-adaptive-subblock-resume.mp4) |
| R16-reuse-2 | 81.92 | 40380 | [MP4](../../outputs/legacy-cleanup/renders07/R16-reuse-2.mp4) |
| R16-reuse-2-resume | 28.87 | 37366 | [MP4](../../outputs/legacy-cleanup/renders07/R16-reuse-2-resume.mp4) |
| R16-reuse-3 | 76.29 | 40380 | [MP4](../../outputs/legacy-cleanup/renders07/R16-reuse-3.mp4) |
| R16-reuse-3-resume | 24.15 | 37366 | [MP4](../../outputs/legacy-cleanup/renders07/R16-reuse-3-resume.mp4) |
| R16-subblock-reference | 160.24 | 43204 | [MP4](../../outputs/legacy-cleanup/renders07/R16-subblock-reference.mp4) |
| R16-subblock-reference-resume | 30.51 | 38872 | [MP4](../../outputs/legacy-cleanup/renders07/R16-subblock-reference-resume.mp4) |
| R17-fp8 | 123.65 | 23394 | [MP4](../../outputs/legacy-cleanup/renders07/R17-fp8.mp4) |
| R17-fp8-decode | 11.30 | 5760 | [MP4](../../outputs/legacy-cleanup/renders07/R17-fp8-decode.mp4) |
| R17-fp8-pause | 54.61 | 23394 | [Log](../../outputs/legacy-cleanup/renders07/R17-fp8-pause.log) |
| R17-fp8-resume | 46.33 | 23394 | [MP4](../../outputs/legacy-cleanup/renders07/R17-fp8-resume.mp4) |
| R17-nvfp4 | 78.67 | 15326 | [MP4](../../outputs/legacy-cleanup/renders07/R17-nvfp4.mp4) |
| R17-nvfp4-decode | 11.70 | 5760 | [MP4](../../outputs/legacy-cleanup/renders07/R17-nvfp4-decode.mp4) |
| R17-nvfp4-pause | 39.46 | 15326 | [Log](../../outputs/legacy-cleanup/renders07/R17-nvfp4-pause.log) |
| R17-nvfp4-resume | 32.39 | 15326 | [MP4](../../outputs/legacy-cleanup/renders07/R17-nvfp4-resume.mp4) |
| R18-large | 310.58 | 49690 | [MP4](../../outputs/legacy-cleanup/renders07/R18-large.mp4) |

## Qualification notes

- The functional matrix uses the clothed AI-generated `inputs/face1.jpg`, with
  `inputs/2.jpg` as its second image/last anchor. The recorded soundtrack is
  looped to 2.5 seconds to satisfy the public reference-audio minimum. The frozen
  parity inputs are unchanged.
- Source export and resumable checkpoint capture are separate public operations.
  The upscale matrix creates each independently and compares their source latents
  before testing completed-sampler import.
- The launcher clears `H3_REFERENCE_MODEL` and `H3_REFERENCE_REGRESSION_OUT` before
  state creation/replay. An initial text-cache replay correctly rejected a
  bookkeeping-environment mismatch; its control is regenerated with a consistent
  environment. No identity check was bypassed.
- Bridge rejection tests explicitly request six steps. The suite's evaluation
  ceiling originally intercepted their default 20-step request; all 41 retained
  assertions pass with the bounded request on CUDA and Metal.
- The continuation integration probe now initializes its expected noise with the
  current CUDA RNG and packed-audio layout; Metal keeps its existing RNG. Its
  initial-state, prefix, callback and exact-resume assertions remain.
- Quantization operator probes supply a complete H3 block size to the current
  admission API while retaining small actual test tensors. The former arbitrary
  byte counts correctly failed the geometry guard.
- Failed requests and retries remain in the evidence directory. They are not
  counted as successful feature runs.

The memory placement probe uses deterministic nonzero embeddings and compares
all plans against its own streaming result. Historical multi-reference campaign
metadata moved unchanged to `tests/archive/multi-reference/`; it is not a retained
aggregate prerequisite. Preview replay requires the writer-produced presentation
sidecar. Current same-build CUDA resume checks are byte-exact at each latent
boundary; the old cross-backend tolerance mode was removed.

The cold LoRA case passed in 740.70 seconds end to end. Native CoW cloning took
0.002 seconds, folding 5.52 seconds and untouched-byte validation 44.44 seconds;
source/cache hashing dominated cold setup. These costs remain in total wall time.

The adaptive+SubBlock case exercised six adaptive decisions, retained
372,208,152 bytes of cache history and dispatched 196 sparse calls. Conservative
policy refreshed every step on the coarse six-step schedule; no adaptive cache
hit or speedup is claimed. Reuse 2/3 executed four/three transformer evaluations.
Current policy unit tests cover cache reuse decisions.

## Remaining intentional occurrences

- Qwen `legacy` names identify a supported BF16-scaling diagnostic on current
  kernels; the race-fixed direct/tiled kernels remain live and tested. This is
  unrelated to a removed renderer or saved-file format.
- cuDNN's legacy API mention explains an active engine-selection boundary.
- Strict content hashing remains for explicit quantization-cache verification
  and LoRA cache construction;
  ordinary model/state identity uses metadata, including effective LoRA identity.
- Geometry profiles, RNG versions and source/refinement record versions describe
  current distinct execution contracts, not backward migrations.
- Historical reports and their evidence-rendering scripts retain past commands
  and results. Removed option names in current tests assert unknown-option errors.

## Removal inventory

Files removed or replaced, with their reasons:

| File | Reason |
| --- | --- |
| `scripts/cuda_remote_bench.sh` | Wrapper for the removed historical 20-step benchmark tier. |
| `src/execution.c` | No-op execution bit or removed decoder-policy shim. |
| `src/vae_policy.c` | No-op execution bit or removed decoder-policy shim. |
| `src/vae_policy.h` | No-op execution bit or removed decoder-policy shim. |
| `tests/cuda_primitives.c` | Runner for retired pre-SGLang CUDA/Metal golden corpus. |
| `tests/cuda_real.c` | Runner for retired pre-SGLang CUDA/Metal golden corpus. |
| `tests/cuda_tokenizer.py` | Runner for retired pre-SGLang CUDA/Metal golden corpus. |
| `tests/fixtures/cuda/manifest.json` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/multi-reference-362.json` | Moved unchanged to `tests/archive/multi-reference/`: historical campaign metadata, not golden tensors. |
| `tests/fixtures/cuda/multi-reference-final.json` | Moved unchanged to `tests/archive/multi-reference/`: historical campaign metadata, not golden tensors. |
| `tests/fixtures/cuda/multi-reference-sample.json` | Moved unchanged to `tests/archive/multi-reference/`: historical campaign metadata, not golden tensors. |
| `tests/fixtures/cuda/multi-reference.json` | Moved unchanged to `tests/archive/multi-reference/`: historical campaign metadata, not golden tensors. |
| `tests/fixtures/cuda/primitives.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/audio-decoder.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/audio-encoder.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/block-adaln.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/block-attention-residual.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/block-attention.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/block-final.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/block-qkv.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/block-query.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/dit-audio.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/dit-realtext-audio.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/dit-realtext-video.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/dit-video.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/image-encoder.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/text-multimodal.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/text.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/video-decoder.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/video-encoder.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/vision-1-deep0.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/vision-1-deep1.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/vision-1-deep2.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/vision-1-merged.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/vision-2-deep0.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/vision-2-deep1.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/vision-2-deep2.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/real/vision-2-merged.bin` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/cuda/tokenizer-metal.json` | Retired pre-SGLang CUDA/Metal numerical baseline. |
| `tests/fixtures/refvideo-layout/layout-legacy56.safetensors` | Removed continuous Ref2VA layout. |
| `tests/refvideo_ab.py` | Historical pre-release/legacy Ref2VA comparison campaign. |
| `tests/refvideo_regression.py` | Historical pre-release/legacy Ref2VA comparison campaign. |
| `tests/refvideo_render_regression.py` | Historical pre-release/legacy Ref2VA comparison campaign. |
| `tests/test_metal.c` | Standalone archived MLX numerical fixture comparison; current operators remain covered. |
| `tests/test_real_audio_encoder.c` | Standalone archived MLX numerical fixture comparison; current operators remain covered. |
| `tests/test_real_audio_vae.c` | Standalone archived MLX numerical fixture comparison; current operators remain covered. |
| `tests/test_real_dit.c` | Standalone archived MLX numerical fixture comparison; current operators remain covered. |
| `tests/test_real_dit_block.c` | Standalone archived MLX numerical fixture comparison; current operators remain covered. |
| `tests/test_real_dit_schedule.c` | Standalone archived MLX numerical fixture comparison; current operators remain covered. |
| `tests/test_real_multimodal_text.c` | Standalone archived MLX numerical fixture comparison; current operators remain covered. |
| `tests/test_real_prompt.c` | Standalone archived MLX numerical fixture comparison; current operators remain covered. |
| `tests/test_real_qwen_vision.c` | Standalone archived MLX numerical fixture comparison; current operators remain covered. |
| `tests/test_real_ref_video_text.c` | Standalone archived MLX numerical fixture comparison; current operators remain covered. |
| `tests/test_real_video_encoder.c` | Standalone archived MLX numerical fixture comparison; current operators remain covered. |
| `tests/test_real_video_vae.c` | Standalone archived MLX numerical fixture comparison; current operators remain covered. |
| `tests/test_text_metal.c` | Standalone archived MLX numerical fixture comparison; current operators remain covered. |
| `tests/still_video_regression.c` | Historical checkout VAE comparison; current decode/round-trip coverage remains. |
| `tests/test_tokenizer_conformance.m` | Ported conformance and collision assertions to portable C. |

The corresponding Makefile targets and links were removed. Mixed BF16 tests
retain analytic Euler, reduction, protected-prefix, AdaLN and patch-projection
checks; only the archived model-fixture comparison branch was removed.


## Final audits

The portable source/tool files match between the local workspace and the final
CUDA snapshot. Active documentation links resolve locally, and every removed or
replaced file has a reason in the inventory above. No task renderer, test worker
or GPU process remained after qualification. Executables and the library remain
under ignored `bin/`; evidence is under ignored `outputs/legacy-cleanup/`.
The diff audit records one cosmetic trailing blank line in `src/sampling/sampler_state.c`;
the qualified production source was retained byte for byte. No functional or
validation failure remains outstanding.
