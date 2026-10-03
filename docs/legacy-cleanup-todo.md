# Remove obsolete execution and saved-file compatibility

Status: **complete — 32/32 tasks complete**. Implementation, current format
contracts, retained tests, clean CUDA/Metal builds, sanitizer checks, 54 render/state
jobs and the final **204/204 exact SGLang gate** passed. See
[the results record](cuda/legacy-cleanup-results.md) for evidence and media links.
The separate server project remains in [the server checklist](features/server-tasks.md) and
[its design](features/design-server.md); implementing the server is not part of
this work.

## Scope and acceptance rules

- Remove `--cuda-reference` entirely, including `auto`, `sglang`, and `legacy`.
  Ordinary CUDA video always selects the current native SGLang arithmetic.
  Preserve explicit attention, quantization, reuse, adaptive-cache, SubBlock,
  preview, and weight-placement controls and their current combination guards.
- Remove `--legacy-ref2va-video-pipeline` and its entire historical execution
  path. Released Ref2VA preprocessing becomes the only reference-video path.
- Remove backward readers, migrations, fallback defaults, retired API fields,
  and dead checks retained solely for old h3cli states or execution modes.
  **Previously saved h3cli files need not remain loadable.** Current generation,
  save/load, pause/resume, continuation, bridge, decoding, conditioning reuse,
  stills, and upscaling must continue working with newly written files.
- SGLang parity is the only historical numerical contract to preserve. Keep
  the **12 recorded fixtures and all 204 golden hashes** in
  [the manifest](../tests/cuda_reference/manifest.json) unchanged. Do not
  regenerate goldens, introduce tolerances/skips, narrow coverage, or add an
  independent/live SGLang oracle. Other obsolete numerical/migration tests and
  their exclusive fixtures, scripts, build targets, and ABI shims may be removed.
- A full test run means the **complete retained current-feature suite plus the
  complete SGLang parity gate**. Do not delete current functional coverage just
  because it resides beside a legacy test; port those assertions to the current
  API/file format. Sample renderings supplement this suite.
- Preserve current Metal/still arithmetic, model-weight formats, and supported
  external media/latent inputs. A low version number or the word `legacy` does
  not establish dead code: some versions currently distinguish active geometry,
  backend, and feature profiles. Consolidate those profiles deliberately.
- Source remains in `src/`, executables and `libh3.a` in ignored `bin/`, and
  render/test evidence in ignored `outputs/legacy-cleanup/`. Avoid unrelated
  numerical optimizations and GPU dependency upgrades during this cleanup.

## Work host and validation environment

Use the **RTX PRO 5000 server at `cuda-test`** for implementation builds,
CUDA tests, SGLang parity, and sample renderings:

```sh
ssh cuda-test
```

Use `~/h3cli` as the server checkout and the model installation under `/models`.
Resolve and record the actual model/component paths during preflight; the
existing checkout may expose them through `models/MiniMax-H3`. Source the
checkout's qualified `cuda-env.sh` and verify its settings rather than replacing
the installed CUDA/cuDNN/CUTLASS stack. Keep host-specific paths in this work
record or environment variables, never in production defaults or test code.

This instruction supersedes the earlier prohibition on PRO 5000 testing for
this cleanup. The server project's M4-only restriction belongs to that separate
project. Use the local M4 for a Metal build and applicable host checks when
shared code changes; all CUDA qualification and the render matrix below run on
the PRO 5000. Do not use the 4090/5090 nodes.

Follow [CONTRIBUTING.md](../CONTRIBUTING.md): run the complete recorded CUDA
regression after each coherent implementation/test-tooling change, with a fresh
output directory. Run it again on the final candidate after the full suite and
sample renders. Hardware/dependency failures, timeouts, skipped cases, and stale
outputs are not passes. Documentation-only planning needs link/content checks,
not remote execution.

## Inventory and parity boundary

- [x] LGC001 Record the initial commit, dirty files, build configuration, and
  full source/test dependency inventory. Classify each legacy occurrence as
  obsolete production behavior, active feature, parity dependency, or historical
  documentation. Trace callers through generators, shared GPU code, tests,
  Makefile targets, and the planned server option inventory before removal.
- [x] LGC002 Preflight the PRO 5000, checkout, qualified runtime, model variants,
  image VAE, latent upscaler, preview VAE, FFmpeg, media inputs, and a bounded
  LoRA fixture. Record versions, device identity, available resources, and exact
  paths. Run the unchanged 204-output gate as the baseline and preserve logs.
- [x] LGC003 Specify the new current-only state contracts before editing their
  readers/writers: AV, presentation, sampler sections, conditioning, and upscale
  source/refinement records. Inventory what today's writers actually emit,
  including Metal/CUDA and ordinary/upscale geometry. Assign new schema versions
  where required; define explicit unsupported-version errors and exact current
  round-trip/provenance invariants. No migration or implicit defaulting of old
  fields is required.
- [x] LGC004 Isolate the parity suite's compatibility dependencies. Its frozen
  `final.h3av` fixture and the `legacy` argument used by `cuda_sglang_vae` must
  not force production backward readers or alternate policies to survive.
  Where needed, move minimal fixture loading/probe adapters under `tests/` and
  pass their unchanged tensors to current production operators. Preserve
  reference math, fixture integrity checks, capture coverage, and all 204 hashes.
  Update harness/API wiring only; document why each retained test-only adapter
  is necessary. Do not compute expected outputs inside an adapter.

## CLI and execution cleanup

- [x] LGC005 Delete `--cuda-reference` from the long-option table, enums, help,
  parser, conflict checks, examples, and active launchers. All values become
  ordinary unknown-option errors; do not keep a compatibility alias. Remove
  caller-selectable public request/decode fields and explicit-set bits that
  served this flag, updating all current callers.
- [x] LGC006 Derive CUDA video arithmetic internally from the backend and
  operation across generation, resume, AV decode, state-only, continuation,
  and upscale. Preserve the current internal arithmetic identity and scoped
  numerical behavior where required by real consumers. Remove obsolete
  reference/legacy selection and environment overrides, including the
  codec-version-1 decode validation bypass. Keep Metal and still dispatch valid.
- [x] LGC007 Remove `--legacy-ref2va-video-pipeline`, its request fields,
  `H3_REFVIDEO_LEGACY`, legacy branches, and exclusive helpers throughout
  `refvideo`, FFmpeg extraction, posterior/encoder handling, layout preparation,
  and execution. Simplify signatures to the sole released path while retaining
  correct cadence, chunking, posterior sampling, condition rows, and soundtrack
  behavior for video, silent video, and video with an external soundtrack.
- [x] LGC008 Remove the old Ref2VA pipeline selector from conditioning keys,
  checkpoint serialization, compatibility checks, telemetry, and saved-state
  inspection. Keep any current preprocessing recipe identity needed to detect
  incompatible files; do not accidentally alter ordered mixed references or
  reference image sizing/patch limits.

## Current-only saved files

- [x] LGC009 Replace AV backward loading with the new current contract. Remove
  sidecarless legacy delivery, `--decode-full-state`, `legacy_full_state`, and
  automatic legacy codec/geometry inference. Require complete current metadata
  for public saved-state delivery and retain precise corruption, integrity,
  model, geometry, and source/output-collision checks.
- [x] LGC010 Consolidate presentation readers and writers. Remove the historical
  version ladder and implied defaults after moving all current writers to an
  explicit contract covering full/preview decode, attention, quantization,
  adaptive/SubBlock warmups, trimming, and upscaling. Keep both supported backend
  delivery recipes; old version numbers must not be confused with active math.
- [x] LGC011 Remove historical sampler sections, retired fast-mode fields,
  old Ref2VA fields, fallback warmups/budgets, and old execution-version migration.
  Simplify section-version handling around the new writer contract, preserving
  RNG/schedule state, device/build/model validation, ordered references, cache
  history, placement policy, and current resume override rules.
- [x] LGC012 Update conditioning serialization and fingerprints with no legacy
  `fast-cuda`/`legacy-refvideo` entries or backward interpretation. Test current
  save/load with and without schedule tensors and reject mismatched current
  prompts, references, shapes, recipes, and models.
- [x] LGC013 Remove old upscale/continuation import and conversion branches.
  Preserve current `.h3up` source/refinement save/load and current completed
  sampler import if it remains a supported current-format operation; a function
  or error named `legacy import` is not by itself grounds to remove that useful
  operation. Ensure ordinary and upscaled AV states keep their valid geometry
  and audio-preservation contracts. Update inspection and resume paths together.
- [x] LGC014 Remove obsolete model/content-signature compatibility entry points
  and duplicate identity domains where no active consumer remains. Update
  current producers and consumers together; preserve integrity and model/VAE
  mismatch detection and current LoRA identity behavior. Do not hash model
  weights just to replace a removed compatibility path.

## Remaining code and test cleanup

- [x] LGC015 Remove `fast_cuda`, `resume_default_cuda`, retired execution bits,
  zero-only statistics, public ABI padding/shims, and old CLI/environment
  handling that exist solely for deleted behavior. There is no requirement to
  preserve the old library ABI for legacy tests. Isolate any genuinely needed
  parity-probe adapter in tests, rather than keeping fake production policy.
- [x] LGC016 Audit `vae_policy.*`, Qwen legacy diagnostic variants, old benchmark
  controls, obsolete state wrappers, and generated kernels. Delete those whose
  only remaining consumers are removable legacy tests. Preserve primitives
  used by SGLang parity, current Metal/still execution, or supported diagnostics;
  simplify misleading names/comments where appropriate.
- [x] LGC017 Remove unreachable CLI/library branches, including retired-option
  resume checks, always-false presentation checks, duplicate policy resolution,
  and impossible conflict combinations. Centralize current validation where it
  removes duplication without changing current error/ownership behavior.
- [x] LGC018 Remove newly orphaned files, declarations, includes, allocations,
  stats, generator entries, and build/link dependencies. Check source generators
  as well as generated output. Retain third-party notices for live consumers
  and avoid deleting shared kernels just because a caller was once called fast.
- [x] LGC019 Delete tests, fixtures, oracle-generation scripts, benchmarks, and
  Makefile targets that only preserve removed compatibility or obsolete numerical
  baselines. Audit older `cuda-parity`/`cuda-real-parity` and migration suites
  separately from the protected SGLang gate. Split mixed tests so current
  behavior coverage remains; record every deletion and its reason.
- [x] LGC020 Update retained CLI/API/state tests for removed flags and new file
  contracts. Cover unknown-option rejection, current-format round trips,
  unsupported-version rejection, malformed/truncated files, mismatched metadata,
  reference ordering, cancellation/cleanup, and valid generation/resume/decode.
  Test new state variants for all currently supported feature families without
  adding a historical migration suite.
- [x] LGC021 Update parity harness build/API wiring only where cleanup requires
  it. Run `tests/test_cuda_reference_gate.py`; verify the manifest and 12 fixture
  hashes are unchanged and that collection still requires exactly 204 outputs.
  Preserve strict comparisons, source-change detection, the 12-minute post-build
  deadline, and production execution for every measured numerical operation.
- [x] LGC022 Make the retained full test suite runnable through clear aggregate
  targets and an explicit inventory of supplementary targets. Remove stale
  prerequisites instead of leaving skipped/broken legacy tests in `make test`.
  Include current host, CLI, serialization, reference media, CUDA operators,
  conditioning, saved-state, still, preview, upscale, cache/attention, LoRA, and
  weight-residency coverage; the parity gate remains a required separate run.

## Build, full suite, and render qualification

- [x] LGC023 Clean-build the CUDA executable/library and every retained test
  binary on the PRO 5000 with the qualified dependencies and supported optional
  kernels. Confirm outputs stay under `bin/`. Clean-build Metal locally on M4
  and run applicable host checks to catch shared-header/API breakage; record
  backend-specific results separately.
- [x] LGC024 Run the complete retained Linux/CUDA suite on the PRO 5000,
  including `make test`, current integration/feature targets, and supplementary
  targets recorded in LGC022. Fix failures without weakening coverage. Capture
  commands, exit status, duration, and per-target results; there must be no
  unexplained exclusions or hidden reliance on historical golden files.
- [x] LGC025 Run focused sanitizer/failure-path checks for changed current file
  readers and refactored ownership/dispatch paths. Exercise early rejection,
  partial reads, allocation failure where injectable, interrupted writes, and
  failed decode/resume followed by a successful operation. Keep these checks
  bounded and relevant to the cleanup.
- [x] LGC026 Run media cases R01–R10 below on the PRO 5000 using fixed prompts,
  seeds, and recorded input identities. Save each required MP4 and its command,
  complete log, state/presentation files where applicable, wall time, and memory
  observations. Confirm the sole released reference-video route is selected.
- [x] LGC027 Run state/feature cases R11–R15 using files freshly written by the
  final candidate. Verify exact same-build pause/resume and conditioning reuse
  against their uninterrupted/current controls, decoder output consistency,
  continuation trimming, bridge handling, upscale audio preservation, and
  still-latent round trips. These are current functional comparisons, not new
  recorded numerical goldens.
- [x] LGC028 Run optimization cases R16–R18. Use explicit short warmups so
  adaptive and sparse paths execute during six steps, and prove activity from
  counters rather than flag acceptance. Exercise current precision/attention
  metadata and saved-state round trips; do not require approximate outputs to
  match dense SGLang hashes or expand the numerical oracle.
- [x] LGC029 Validate all delivered media with ffprobe and a complete FFmpeg
  decode: dimensions, frame count, fps, audio presence/duration, delivery trims,
  and absence of truncation. Produce a local HTML review page with every video
  and still, exact settings, and total wall times. Inspect for gross visual/audio
  regressions and report the evidence without assuming user visual acceptance.
- [x] LGC030 Update README, CLI/library/state documentation, setup/test commands,
  and active tools to describe the single CUDA recipe, single Ref2VA path, and
  current-only saved formats. Reconcile the server design/checklist's option
  inventory with removed flags without implementing server work. Clearly mark
  old reports as historical; do not rewrite past evidence to imply new passes.
- [x] LGC031 Run the final complete SGLang regression on the PRO 5000 in a fresh
  directory after the full suite and renders. Require **204/204 exact matches**
  and unchanged fixture/manifest hashes. Save the result, source/binary/config
  identities, command log, and duration. Any subsequent source/test-tooling fix
  requires rerunning its affected checks and the complete parity gate.
- [x] LGC032 Publish `docs/cuda/legacy-cleanup-results.md` with the removal
  inventory, remaining justified legacy/test-only occurrences, new format
  contracts, backward-compatibility break, full-suite results, render links,
  timing table, Metal build status, and final parity evidence. Audit active
  references, clean builds, worker/process cleanup, document links, and git diff.
  Mark tasks complete only when their required evidence exists.

## Required sample matrix

Default: **640×480, 124 frames, six steps, seed 42**, dense BF16, full VAE.
Use fixed family-appropriate prompts and the same inputs/settings for controls.
Each row is required; rows with named variants require each variant. Inputs must
be available before marking preflight complete. Do not substitute a rejected
request, mock result, or an old MP4 for a successful current render.

| Case | Request / variants | Required evidence |
| --- | --- | --- |
| R01 | Dense text-only, default settings | MP4 plus fresh AV/presentation, sampler, conditioning, and upscale source artifacts in compatible invocations; the final parity render may supply the matching baseline. |
| R02 | First frame only | Anchor handling and delivered frame count. |
| R03 | Last frame only | Anchor handling and delivered frame count. |
| R04 | First and last frames | Both anchors, current state provenance, playable result. |
| R05 | One image, `--ref-image-size max` | Image preprocessing/patch handling and released Ref2VA render. |
| R06 | Two ordered images, `--ref-image-size high` | Ordering and distinct conditioning for both inputs. |
| R07 | Image plus standalone reference audio | Audio encoder, reference layout, complete generated audio/video. |
| R08 | Reference video with embedded audio | Released cadence/chunking, posterior, and soundtrack route. |
| R09 | Silent reference video; video with external soundtrack | Two renders proving both current video/audio semantics. |
| R10 | Mixed image, video, and standalone audio | Ordered multimodal conditioning and current save/load provenance. |
| R11 | Fresh AV decode with full VAE and preview VAE | Two decoded MP4s; full decode matches the same-build full control's decoded media. |
| R12 | Pause after step 3 of 6, then resume; conditioning save/reuse | Completed MP4s match uninterrupted/current controls at latent or decoded-content level as appropriate; use text and one reference-conditioned case. |
| R13 | Hard continuation and bridge from a new source | Two renders; 640×480, 56 internal frames, six steps, 39-frame context; verify actual delivered prefix trimming. |
| R14 | Fresh latent upscale, text-only and first+last sources | 320×256 / 22-frame / six-step source renders, then 640×512 / two-step refinement; include refinement pause/resume, source inspection, and current completed-sampler import. |
| R15 | Still generation/decode; one LoRA video | 512×512 still at six steps plus saved-latent decode; default-size six-step video with a recorded LoRA fixture. |
| R16 | `--reuse 2`; `--reuse 3`; adaptive conservative + SubBlock 0.75 text-only; SubBlock 0.75 with one max-size reference image | Four renders; combined case uses both warmups set to 2, reference case uses SubBlock warmup 2. Record actual reuse/cache/sparse activity and verify current state serialization. |
| R17 | FP8 + Sage2++; NVFP4 + Sage3 | Two current supported precision/attention renders, with fresh state decode and bounded pause/resume coverage. Additional supported policy combinations remain covered by the full suite. |
| R18 | 1344×768, 124 frames, six steps, one max-size reference image, SubBlock 0.75, warmup 2 | Larger layout/state coverage and active sparse attention; use current supported placement and record residency/streaming decisions. |

Keep denoiser execution at or below the existing six-evaluation test ceiling per
invocation, except that fresh resume jobs execute only their remaining steps.
Upscale refinement uses its own two-step schedule. These are functional sample
renders, not a new image-quality or performance campaign. Full-precision SGLang
numerical acceptance comes exclusively from the unchanged 204-output gate.
