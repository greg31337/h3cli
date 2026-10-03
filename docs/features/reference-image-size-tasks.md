> Archived: all 24 reference-image sizing tasks are complete.
> Active work is tracked in [todo.md](../todo.md).

# Uniform and extended reference-image sizing

Status: **complete; 24/24 tasks complete**. Design:
[Uniform reference-image sizing and capacity](design-reference-image-size.md).
The completed 31-task weight-residency checklist is preserved in
[its archive](../cuda/weight-residency-tasks.md), with
[qualification results](../cuda/weight-residency-results.md).

Goal: make `--ref-image-size max` use the same 2048-pixel short-edge geometry
on Metal/CUDA and still/video paths, add `high` for a 2048-pixel long edge,
and admit SGLang-sized reference images/batches without the artificial
32,768-patch aggregate cap. Keep `match` as the default.

## Execution contract

- Follow the design's formulas, ties-to-even 32-pixel alignment and inclusive
  1:4–4:1 aspect-ratio range for `high`/`max`. Both fixed-edge modes upscale
  small sources. Preserve `match` and non-image conditioning behavior.
- Admit up to 65,536 patches **per image**, without substituting a new fixed
  aggregate patch cap. Preserve existing reference-count limits and checked
  memory/index safeguards. Bound CUDA patch-projection packing by chunking
  independent rows; keep complete per-image attention groups.
- Do not expand CUDA SGLang parity tests or the recorded reference regression.
  Do not run new SGLang, cross-backend or old-versus-new numerical-parity
  comparisons. Add separate functional, geometry, state and robustness tests.
- Follow [CONTRIBUTING.md](../../CONTRIBUTING.md): after every coherent source,
  build or test-tool change, run the complete unchanged **204-output golden
  regression** on the qualified RTX PRO 5000 environment. Preserve all fixtures,
  hashes, cases, runner behavior and the 12-minute post-build deadline. No
  skipped cases, updated goldens or substituted baseline counts as a pass.
- Build/test local Metal and the authorized local RTX PRO 5000 server using
  CUDA 13.0.3/cuDNN 9.20. No 5090, FP8/NVFP4 or additional quality campaign is
  required. Keep machine-specific paths and connection details untracked.
- Use metadata identities for model weights; do not hash/scan the base model.
  Keep source under `src/`, build products under ignored `bin/`, and generated
  fixtures, captures and results under ignored `outputs/`. Run GPU jobs
  serially per device; retain failed attempts and distinguish them from passes.
- Every completed implementation/validation task needs recorded evidence.
  Evidence and preserved failed attempts are recorded in the
  [validation record](reference-image-size-results.md).

## M0 — Freeze behavior and validation

- [x] RI001 Record the starting revision, source identity, Metal/CUDA build
  environments and baseline unchanged 204/204 golden result. Record golden
  manifest/fixture identities so completion can prove they were not modified.
- [x] RI002 Audit sizing callers, enum handling, batch guards, GPU index ranges,
  conditioning keys, sampler records and upscale planning. Map every required
  change to the shared design; distinguish unrelated `32768` constants and
  existing image/reference-count limits.
- [x] RI003 Freeze a small functional manifest with the design's six rows per
  backend, exact fixture geometry, prompt/seed, settings, expected patch counts,
  output requirements and per-stage timeout/resource records. Generate owned
  deterministic fixtures; do not alter the parity fixture set or `inputs/`.

Exit: explicit behavior, independent test scope and a passing existing baseline.

## M1 — Shared sizing and public controls

- [x] RI004 Implement one backend-independent host resolver for `match`, `high`
  and `max`, with checked arithmetic and deterministic fixed-edge alignment.
  Retain old helper symbols only as thin wrappers needed by existing callers;
  preserve existing CUDA-video `max` geometry and `match` behavior.
- [x] RI005 Append `H3_REFERENCE_IMAGE_HIGH=2` without changing old enum values.
  Update CLI parsing/help, API comments and request validation. Reject invalid
  strings/enums before model work; keep `match` as the default.
- [x] RI006 Route all reference-image preparation through the shared resolver,
  including Metal and CUDA stills. Feed Qwen and the visual VAE the same canvas,
  keep ordered references, and add actionable geometry/capacity diagnostics.
  Leave backend pixel arithmetic and anchor/video/audio policies intact.
- [x] RI007 Add host/API/CLI tests for the documented example table, small and
  large inputs, square/portrait/landscape, 4:1 endpoints and rejection outside
  them, rounding ties, invalid/overflow inputs, internal render dimensions and
  output-independent `high`/`max`. Update ordinary historical host expectations
  affected by the intentional change; do not modify the frozen parity suite.

Exit: one sizing contract on both backends and output paths; full unchanged
golden gate passes after this coherent code/test change.

## M2 — SGLang-compatible image capacity

- [x] RI008 Replace image/batch 32,768 guards with checked per-image admission
  through 65,536 patches and checked aggregate row/byte/index accounting. Keep
  image admission separate from video batches and preserve existing count and
  memory protections. Fail with the real limiting resource or invalid geometry.
- [x] RI009 Implement bounded CUDA patch-projection packing for larger batches:
  pack the filter once, reuse at most 512 MiB input scratch, process complete
  chunks and tails with checked offsets, and preserve the current path for
  fitting requests. Keep bias handling and full per-image attention groups.
- [x] RI010 Audit and repair larger-count consumers through vision positions,
  attention, QKV/MLP, merger/deepstack slicing, text insertion, VAE conditioning
  and DiT layout. Preserve image order and prevent cross-image attention.
  Exercise Metal/CUDA-still sequential paths without requiring common batching.
- [x] RI011 Add separate host capacity and chunk-planning tests around 32,768,
  65,536 per image, 43,520/49,152 aggregate rows, aggregate totals over 65,536,
  nine maximum-size images, overflow and insufficient-budget rejection. Verify
  row coverage and offsets structurally, without a numerical reference model.
- [x] RI012 Add bounded GPU completion/memory-safety tests for chunk boundaries
  and tails, including 32,764/32,768/32,772 and 65,540 rows. Verify finite output,
  expected shapes, complete writes, guard regions and cleanup; include actual
  vision encoding at the 65,536-patch single-image boundary. Do not add oracle
  tensor comparisons or numerical tolerances.
- [x] RI013 Test allocation failure, cancellation and small/large/small requests
  in a retained context. Account for live activation/output memory, release
  scratch after its GPU consumers, and leave no stale image-group metadata or
  partial outputs. Preserve existing process/device safety caps.

Exit: required large inputs execute within the documented resource contract;
separate robustness checks and the unchanged golden gate pass.

## M3 — Conditioning and saved-state compatibility

- [x] RI014 Extend conditioning keys/identities for `high` and the new geometry
  policy. Reject stale Metal/still `max` conditioning for fresh requests while
  avoiding unrelated invalidation of text-only/`match` data. Test alternating
  modes and retained-context hits/misses using structural identity assertions.
- [x] RI015 Extend sampler/state validation and serialization for `HIGH=2`.
  Preserve old enum values and exact-resume use of already prepared conditions;
  do not resize/reopen saved references or reuse legacy geometry as fresh new
  policy conditioning. Add versioned metadata only where needed, with explicit
  legacy interpretation and unknown-value rejection.
- [x] RI016 Make upscale planning treat `high` and `max` as intrinsic stored
  canvases; preserve `match` retargeting. Add high-mode save/load/planning and
  legacy stored-geometry tests, including media-unavailable resume/planning.
  Test serialization/geometry, not rendered numerical equivalence.

Exit: fresh requests, cached requests, prepared resume and upscale planning
retain explicit sizing semantics; the unchanged golden gate passes.

## M4 — Build and functional validation

- [x] RI017 Build Metal CLI/library and affected standalone test tools. Run the
  relevant host, CLI, conditioning, sampler and upscale-state tests plus the
  new sizing/capacity targets. Retain exact commands and results.
- [x] RI018 Build CUDA CLI/library and affected tools on the RTX PRO 5000. Run
  relevant host/GPU/state tests and the unchanged complete 204-output regression
  in the qualified environment. Record source/binary/runtime identities.
- [x] RI019 Complete all six Metal functional rows from the design: video
  `match` one image, `high` two images, `max` two images, `max` one 4:1 image;
  still `high` one portrait and `max` one portrait. Use two evaluations, with
  videos at 640×480/90 frames and stills at 640×480. Verify completion, actual
  conditioning geometry/counts, full media decoding and cleanup.
- [x] RI020 Complete the same six CUDA functional rows with the same fixture
  and output contracts. Prove the former 32,768 rejection is gone for both a
  multi-image request and a 65,536-patch single image. Preserve failed attempts;
  fix OOM/timeout/failures rather than treating a missing row as passing.
- [x] RI021 Audit all twelve functional outputs and robustness evidence. Record
  actual time/memory and any limitations without numerical-parity, perceptual
  quality or performance claims. Confirm that no new SGLang parity case or
  comparison script was introduced.

Exit: both builds and the required functional matrix pass; new behavior has
coverage separate from the unchanged numerical regression.

## M5 — Documentation and closeout

- [x] RI022 Update README/help/API documentation and ordinary tool option lists
  for `match|high|max`, upscaling, rounding, aspect ratio, per-image capacity,
  aggregate resource handling and cache compatibility. Correct unconditional
  down-only descriptions and explain the intentional Metal/still `max` change.
- [x] RI023 After the final coherent code/test change, retain a fresh complete
  **204/204** golden pass within the existing deadline. Verify unchanged
  manifest, fixtures and regression coverage against RI001; never regenerate
  expected outputs. Recheck builds/tests only as needed for subsequent changes.
- [x] RI024 Publish a concise validation record, check links and task evidence,
  and close this list only when all required results exist. State that the new
  feature passed functional coverage and the existing golden gate, with no new
  numerical-parity qualification. Keep artifacts reproducible and untracked.
