# Native queued server and SGLang video API

Archived 2026-09-29. Active work is tracked in [todo.md](../todo.md).

Status: **complete — 34/34 tasks passed**.
Updated 2026-09-29. See the [validation report](server-results.md).
M04 passed at the approved 256×256/56 frames/two steps. The complete recorded
SGLang parity regression passed all 204 hashes on the PRO 5000 with unchanged
goldens; the tested source matches the delivered tree.
Implement [the server design](design-server.md). This is a fresh checklist;
the prior completed work is preserved in
[adaptive-budget/reference tasks](../cuda/adaptive-budget-reference-tasks.md) and
[its qualification report](../cuda/adaptive-budget-reference-results.md).

## Fixed scope and validation limits

- Add `h3cli --server` with durable job submission, polling, results and media
  downloads. Job requests use only SGLang-defined fields and one added string,
  `h3cli`, parsed as CLI flags. Its settings override corresponding SGLang
  data/settings, including media, geometry, quality and operation selection.
  Preserve every supported CLI feature through the shared parser/execution
  path; retain effective native combination guards. No structured `h3.options`,
  references/LoRA arrays or operation selector in the request.
- Pin compatibility to SGLang
  `7ee7bef79decaf78f7669994f74e6f75c7784c14`. Cover canonical H3 `task`/
  `conditions`/`target`, JSON/multipart, sigma-point versus evaluation counts,
  variants and the video lifecycle. Map standard `quality` directly to native
  `--quality`; native preview levels are available in the `h3cli` string. Follow
  the design's explicit preset/individual-option precedence and document native
  caching differences. Do not silently accept unsupported knobs.
- Use the **local M4 for server qualification**. The user's follow-up also
  requires the full recorded SGLang parity regression on the previously
  authorized PRO 5000; this is the sole CUDA exception to the original local-only
  test scope. No live SGLang inference or changes to the 204-output golden state.
- Real render defaults: **256×256, 22 frames, two evaluations**, seed 42.
  At most six evaluations per invocation. Exceptions are only the design's
  107-frame canonical four-second test, approved 56-frame M04, 39/56-frame continuation, and
  512×512/22-frame 2× upscale. No large-resolution/performance campaign.
- Exercise the real M4 engine for every required operation; fake workers
  cover transport/lifecycle faults, not render qualification. Missing fixtures,
  OOMs, timeouts and unsupported platform behavior are not passes.
- Group service files in `src/server/`, CLI parsing in `src/cli/`, and common
  request/dispatch interfaces at `src/`. Keep vendored code in `third_party/`,
  executable/library output in ignored `bin/`, and evidence in ignored
  `outputs/server-validation/`.
  Keep the existing CLI working and save all published output/state sidecars.

## Contract and shared request path

- [x] SRV001 Freeze the implementation baseline and evidence manifest: source,
  M4/OS/toolchain, model/component metadata and current CLI inventory. Archive
  small pinned SGLang contract fixtures and record provenance without importing
  SGLang/PyTorch. Include upstream quality values and JSON/multipart/SDK wrappers;
  classify expected compatibility differences from the design.
- [x] SRV002 Extract the typed CLI option registry and owned request structure.
  Account for every current long/short option, path role, repeatable value,
  explicit-set bit, default and platform restriction. Removed CLI names have no
  server aliases; current-only state schemas also apply to server artifacts.
  Preserve exact 64-bit seeds. Generate a coverage report that fails if any CLI
  feature lacks its `h3cli` string
  mapping or named process/presentation equivalent. Include `quality` and
  `no-preview-vae`, path arity and operation/preset dependency groups.
- [x] SRV003 Refactor one-shot operation dispatch and preflight into reusable
  functions. Replace parser `exit()` paths with structured errors while keeping
  CLI exit codes/messages/defaults. Preserve checkpoint omission/override
  semantics, reference/LoRA order, numerical recipes and model lifetime.
- [x] SRV004 Define/version the SGLang-plus-one-string request schema and implement
  the bounded `h3cli` argument lexer through the shared CLI parser. Cover quotes,
  escapes, empty values, Unicode, aliases, `--name=value`, repeated flags,
  reference/LoRA order and two-argument video/audio pairs. Reject non-strings,
  malformed quoting, stray operands, NUL, unknown/startup/internal-worker flags
  and excess bytes/tokens. Never invoke a shell or expand variables/globs.
  Keep typed requests/operation IDs internal to worker IPC; version response
  metadata/capabilities separately. No alternative public options object.
- [x] SRV005 Implement canonical SGLang H3 lowering and explicit unsupported
  field handling: `task`, conditions/roles/keyframe indices, target geometry,
  nearest-grid rounding, timing/alignment, `N` sigma points to `N-1` native
  evaluations, fixed shifts and current native limits. Map `quality` values
  `lossless`/`extra-high`/`high` to the shared native preset implementation;
  allow `preview`/`fast-preview` only via `--quality` in the string. Preserve
  omitted-quality defaults and operation restrictions. Test tiny and large
  arithmetic cases on CPU without allocating large video tensors.
- [x] SRV006 Implement JSON/multipart and SDK extras normalization. Parse
  `extra_body`, `extra_json`, `extra_params` once; normalize multipart's text
  `h3cli` without a second JSON decode. Reject conflicting transport aliases,
  unknown keys and invalid types. Implement the design's priority layers:
  defaults, SGLang preset, SGLang explicit values, native preset, native explicit
  flags. Merge scalars/anchor slots, atomically replace ordered reference lists,
  infer overridden operations, and resolve dependent geometry/timing only from
  effective inputs. Never import/probe replaced media/model paths. Validate
  effective settings after merging; persist provenance and truthful metadata.

## Service, storage and queue

- [x] SRV007 Add native HTTP/SQLite build dependencies. Pin and vendor the
  required CivetWeb subset with hashes/licenses, disable generic file serving
  and scripting, and update setup instructions. Reuse/harden bounded native
  JSON handling without adding a Python runtime service. Build locally on M4.
- [x] SRV008 Add `--server` startup parsing and defaults from the design.
  Resolve cwd/resources, model IDs, port/bind address, state directory, auth,
  read roots, upload/storage/queue/time limits. Reject startup job flags,
  unavailable resources, busy ports and second owners of the same state root.
- [x] SRV009 Implement durable schema/versioning for jobs, variants, assets,
  leases, artifacts, events and idempotency records. Use local SQLite WAL with
  checked migrations, constraints and atomic admission. Test corrupt/unsupported
  schemas and failed writes; acknowledge only durable jobs/assets.
- [x] SRV010 Implement FIFO scheduling with exactly one active inference worker.
  Count/reserve variants atomically, bound waiting work, return 429 plus retry
  information, and preserve queue order across concurrent submissions/restarts.
  Status/download/upload handlers must remain responsive while work runs.
- [x] SRV011 Implement fresh worker spawning via `posix_spawn`, executable-build
  identity pinning, private request/progress descriptors and process groups.
  Parent must not initialize Metal or mutate per-job environment/cwd. Invoke
  shared native execution without a shell or unchecked flag forwarding; isolate
  options/memory between jobs. Send validated typed IPC, not the raw flag string.
- [x] SRV012 Implement callback-driven structured phase/progress/result IPC.
  Distinguish queued/running/paused-result/failed/cancelled/interrupted states,
  variant progress and actual output geometry. Drain logs continuously; never
  infer completion from terminal text or an enqueued GPU counter.
- [x] SRV013 Implement cancellation, deadlines and shutdown: cooperative callback
  cancellation, bounded termination escalation, FFmpeg/process-group cleanup,
  parent-death handling, worker reaping and queue preservation. Resolve races
  with completion/deletion and prove a subsequent job can execute.
- [x] SRV014 Implement restart recovery and idempotency. Preserve queued inputs
  and order; mark interrupted work without automatic regeneration; reconcile
  durable completion manifests after publication crashes. Same key/body returns
  the same job, changed effective request fails, and explicit resume uses valid
  checkpoints. Canonicalize equivalent JSON/multipart/SDK transport spellings;
  preserve effective override/seed identities across restarts.

## Assets, native features and APIs

- [x] SRV015 Implement streamed upload/import and immutable managed inputs.
  Enforce path containment, symlink/regular-file checks, body/media limits,
  asset IDs/digests and queue leases. Support repeated image/video/audio inputs,
  external soundtracks, model/component selections and permitted local imports.
- [x] SRV016 Implement opt-in URL imports with bounded redirects, time/size caps
  and connection-level address policy. Cover disabled mode, private/metadata
  addresses, DNS/redirect escape and failed-import cleanup. Local upload paths
  must support all media workflows with URL imports disabled.
- [x] SRV017 Implement managed output names and atomic artifact publication.
  Register MP4/PNG/frames/logs, `.h3cond`, `.h3sample`, `.h3av` plus presentation
  sidecars, `.h3up` and still latents. Preserve bundle associations on download,
  import and artifact reuse; never serve partial output as completed.
- [x] SRV018 Route every native operation/feature from the registry, including
  LoRA/CoW caches, reference modes/sizing, continuation/bridge, checkpoint pause/
  resume, conditioning, preview/full decode, stills and latent upscale. Map
  `show` to preview events/artifacts and `zoom` to display metadata. Preserve
  CUDA/Metal optimization flags and reject unavailable/invalid combinations.
  Infer operation from flags on both submit endpoints; support native-only
  bodies with `h3cli`, optional SGLang task, and no invented operation/options fields.
- [x] SRV019 Implement SGLang video submit/retrieve/list/delete/content routes,
  response/error shapes and exact status projection. Native non-video or paused
  jobs use `/v1/h3/jobs`; a completed video must have a published MP4. Test
  model replacement, inferred tasks, aliases and effective quality/geometry/
  operation projection. SGLang-only and overridden jobs use the same envelope.
- [x] SRV020 Implement sequential fan-out with 1–10 variants, immutable resolved
  seeds after native overrides, consistent count aliases, per-variant results
  and `content?variant=N`. A native scalar seed replaces a SGLang seed list.
  Preserve FIFO and quotas; define partial failures/cancel behavior and retain
  explicitly completed variants without marking the parent fully successful.
- [x] SRV021 Implement native jobs, cancellation, assets, artifact, discovery,
  readiness and schema endpoints. Enforce auth consistently, strict HTTP methods,
  safe relative links, correct media types, HEAD/range downloads and 404 behavior
  for unpublished/deleted artifacts. Never expose absolute server file paths.
- [x] SRV022 Implement bounded SSE with reconnect/event IDs and preview links.
  Slow or disconnected clients cannot block/cancel a worker. Report phase
  counters, queue wait/run time, error codes, final recipe/build identity and
  delivered frame/audio metadata; cap progress below 100 until publication.
- [x] SRV023 Implement disk/asset/event/log quotas, explicit retention/deletion
  and lease-safe garbage collection. No automatic eviction of queued inputs or
  active downloads. Test quota exhaustion, orphan cleanup and deletion races;
  retain jobs by default until deletion rather than losing results on restart.

## Local M4 tests and delivery

- [x] SRV024 Add exhaustive host contract/option tests: CLI versus HTTP normalized
  requests, all registry descriptors, signedness/overflow/nonfinite values,
  omission versus explicit overrides, ordered repetition, unknown/removed names
  and backend guards. Cover each override family, all quality presets and
  preset/explicit order permutations, quoted paths/prompts, short aliases,
  native operation changes, inherited checkpoint values and mixed media list
  replacement. Assert replaced missing/forbidden paths and URLs are never
  accessed; validate effective derived geometry. Reject the old structured
  request shape and malformed strings; prove shell-looking data stays inert.
  Include ordinary CLI regression checks after refactoring.
- [x] SRV025 Add real HTTP/fake-worker lifecycle tests on loopback: concurrent
  admission, FIFO/one-worker invariant, status/list/download, multipart/extras,
  variants, idempotency, SSE, auth/path/URL restrictions, limits, disconnects,
  crashes, timeout/cancellation, restart, publication/GC races and cleanup.
  Repeat conflicting-field/quality cases across JSON, SDK extras and multipart
  with key-order permutations and verify identical effective requests/results.
  Run locally; label fake outputs so they cannot count as H3 render evidence.
- [x] SRV026 Build the Metal CLI/library/service and run applicable existing
  local CLI/host checks plus focused sanitizer checks for new request/HTTP/
  queue/worker code. Validate setup/dependency changes locally. Keep CUDA
  server execution marked untested. The follow-up requires the complete
  recorded CUDA CLI reference regression on the PRO 5000.
- [x] SRV027 Execute real M4 cases M01–M04: queued T2VA, the small canonical
  SGLang-only request, first+last anchor replacement, and mixed reference list
  replacement. M01 must override conflicting SGLang target/steps/quality/seed
  through `h3cli`; M02 uses standard `quality=extra-high`. Verify the
  actual worker/recipe, async response and playable downloaded outputs. Record
  the canonical duration/frame alignment rather than claiming 22 frames there.
- [x] SRV028 Execute M05–M06: conditioning reuse, preview VAE and preview/frame
  artifacts, state-only/checkpoint jobs, bounded pause/resume and AV decode.
  Download/reuse the actual committed bundles through the API and verify the
  same loader/provenance rules as the CLI.
- [x] SRV029 Execute M07–M09: short continuation and bridge, 2× latent upscale,
  source inspection, and still generation/latent decode. Observe the matrix's
  39/56-frame and 512×512 exceptions; compare expected delivered trimming,
  audio preservation and artifact types without a quality/performance campaign.
- [x] SRV030 Execute M10–M12: bounded LoRA/native reuse controls, one explicitly
  opted-in Metal attention job, active cancellation and a successful next job.
  Exercise standard `quality=high` and native `--quality fast-preview` with
  explicit six-step limits; no preset may accidentally launch a 50-step test.
  Cover remaining optimization mappings with registry/host fixtures. Record
  unsupported local capabilities; do not replace missing real tests with mocks.
- [x] SRV031 Validate all real artifacts with FFmpeg/ffprobe and native state
  loaders: dimensions, frame count, 24 fps, expected audio, complete decode,
  repeated downloads and bundle reuse. Compare one small same-build CLI/server
  job plus normalized requests; make no upstream SGLang numerical-parity claim.
- [x] SRV032 Publish README/server usage, API/schema reference and runnable curl/
  Python client examples for submit, poll, download, references, variants,
  cancellation and saved-state workflows using the single request envelope.
  Document string quoting, each precedence rule, input-list replacement,
  `quality`/`--quality` mapping, native preview presets and overridden metadata.
  Clearly label compatible subset, cold starts, retention and platform limits.
- [x] SRV033 Publish `docs/features/server-results.md` with source/dependency
  identities, exact requests/commands, M01–M12 artifacts, queue/run/total times,
  resource observations, failures and coverage classification. Distinguish
  the recorded CUDA CLI regression from untested CUDA server execution;
  link the evidence and playable small-video outputs.
- [x] SRV034 Audit the complete option-to-API inventory, documentation links,
  single-extra-field schema, precedence/quality fixtures and required real tests.
  Confirm no leaked workers, bound ports or asset leases, and working ordinary
  CLI behavior. Reconcile this
  checklist honestly; unresolved required operations/tests prevent completion.
