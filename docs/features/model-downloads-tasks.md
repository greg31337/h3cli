# Native automatic model downloads

Status: **complete — 36/36 tasks**. Updated 2026-09-30.
Implement [the native model download design](design-model-downloads.md).
The previous 32/32 completed Linux distribution tasks are preserved in
[their archived checklist](../build/linux-distribution-tasks.md), with
[the qualification report](../build/linux-distribution-pro5000.md).

## Fixed scope and acceptance rules

- Missing supported models download **by default**, without a flag or prompt.
  Use native libcurl on Metal, CUDA and the standalone Linux executable.
  Do not add a `--download-missing` flag or a Python/HF runtime dependency.
- Add `--models-path PATH` (default `models/`) as the shared main/auxiliary root.
  Without `-d`, select `PATH/MiniMaxH3`; reuse an existing `PATH/MiniMax-H3`
  only when the canonical entry is absent. Explicit `-d` wins for the main
  directory and explicit auxiliary flags win for their respective files.
  Preserve existing installations, custom weights, numerical behavior and
  saved-state compatibility checks.
  Complete local installations perform no model-network requests. Never silently
  overwrite, upgrade or mix a partial custom/different-revision model tree.
- Deliver `--models-path PATH`, `--download-models GROUPS`, `--list-models`, `--offline` and
  `H3_OFFLINE=1`, with the path, inspection and server semantics in the design.
  Automatic server preparation must keep HTTP responsive and precede GPU work.
- Main H3 revision, exact file closure and all artifact sizes must be established
  from qualified models and published artifacts. An incomplete catalog blocks
  completion; do not substitute a mutable branch or guessed hash.
- Use **local M4** for Metal/host checks and **local RTX PRO 5000** for Linux/CUDA
  qualification. Use isolated writable download trees and existing qualified
  models read-only. Do not use the previous cloud nodes or replace drivers.
- Follow [CONTRIBUTING.md](../../CONTRIBUTING.md): run the complete recorded CUDA
  gate after coherent code/build/test changes, and on the final portable
  artifact. All **204 expected hashes and 12 input fixtures stay unchanged**.
  The gate must run with model networking disabled; no skipped/relaxed cases.
- Tiny local transfer fixtures cover faults and concurrency. Real downloads
  prove actual source/redirect/resume behavior. Small renders are 256×256,
  22 frames, two steps unless the feature needs 39/56 frames or a larger target.
  The existing parity gate retains its own 640×480/124-frame/six-step case.
- Executable/test outputs and `libh3.a` stay in ignored `bin/`; transfer and
  rendering evidence goes in ignored `outputs/model-downloads/`. Do not put
  model weights, credentials or generated evidence in Git or release payloads.
- Completed implementation is covered by the native host suites. Do not mark unrun checks,
  missing artifacts, unavailable sources or missing hardware as passes.

Required lookup/download destinations when the corresponding override is absent:

| Component | Relative to `--models-path PATH` | Override |
| --- | --- | --- |
| Main model | `MiniMaxH3/` | `-d/--model-dir` |
| Preview VAE | `preview-vae/taeh3.safetensors` | `--preview-vae-model` |
| Image VAE | `image-vae/minimax_h3_t1_image_vae_step1597.safetensors` | `--image-vae` |
| Latent upscaler | `latent-upscale/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors` | `--upscale-model` |

The same explicit-override, local-lookup, download-if-missing and offline rules
apply to every component. Only components required by the effective operation
are fetched; no auxiliary may fall back to cwd `models/` under a different root.

PRO 5000 work/test connection:

```sh
ssh cuda-test
```

Implementation evidence: [catalog and closures](model-download-dependencies.md),
[CLI/server contract](model-downloads.md), [25 downloader tests](../../tests/model_downloads.py)
and [23 HTTP/server tests](../../tests/server_http.py), passed on M4 and PRO 5000.
Hardware/package evidence is recorded in the
[qualification report](model-downloads-results.md).

## Catalog and effective dependency contract

- [x] MD001 Inventory current loaders and operation dispatch for FL2VA, Ref2VA,
  all reference types, stills, preview/full decode, state-only, continuation,
  bridge, sampler resume and upscale/resume. Record the exact files needed for
  computation versus identity validation and broad startup inventory. Include
  quality expansion, negations and explicit overrides in the dependency table.
- [x] MD002 Pin the main H3 source revision against the qualified local models.
  Enumerate required tokenizer/config/shard-index/weight files, exact sizes,
  full SHA-256 values, shared identical artifacts and provenance/notices.
  Resolve upstream layout differences explicitly; exclude unused alternative
  layouts and executable repository code. Record a verified transfer/disk budget.
- [x] MD003 Verify the existing preview, image-VAE and BF16 upscaler pins,
  complete-file hashes, sizes, source availability and notices. Preserve the
  qualified artifacts and default filenames; do not substitute newer models.
- [x] MD004 Implement the versioned catalog and deterministic executable
  embedding under `src/models/`. Validate schema, relative destinations,
  duplicate identities, dependency cycles and pin completeness. Ensure native
  and packaged builds consume the same catalog and include it in source materials.
- [x] MD005 Implement the shared pure dependency resolver and checked-in
  file-level operation table. Select from effective settings and current
  saved-state metadata, including decoder/preview overrides. Refactor broad
  startup inventory only as necessary to support the proven minimal closures;
  preserve model identity checks and inference arithmetic.

## Local storage, integrity and installation

- [x] MD006 Implement `--models-path PATH` and shared destination resolution.
  Derive `ROOT/MiniMaxH3` when `-d` is absent and auxiliary component paths
  under the same root using the table above, including preview VAE, image VAE
  and latent upscaler. Apply explicit `-d`/auxiliary precedence regardless of
  flag order; explicit relative paths remain cwd-relative. Reuse an existing
  `ROOT/MiniMax-H3` only when the canonical entry is absent, prefer the canonical
  entry when both exist, and never search outside the selected root or rename
  existing trees. Cover empty/missing roots, symlinks and read-only installs.
  Place private receipts/locks/staging on the destination filesystem outside
  native model/state scans and the immutable release runtime. Validate ancestors
  and enforce confined publication using directory-relative operations.
- [x] MD007 Implement existing-model classification and safe adoption of partial
  unmanaged trees. Preserve complete custom installations; verify overlapping
  official files before extending a partial tree. Refuse custom/folded/revision
  conflicts and existing corrupt files without overwriting or deleting them.
  Treat explicit missing auxiliary paths as pinned-component destinations.
- [x] MD008 Implement verified receipts and warm local reuse. Record catalog,
  file hash/size and stat identity; rehash modified files, recover lost receipts
  safely and avoid whole-model hashing/network probes on ordinary warm runs.
  Preserve complete older compatible installations; reject mixed-revision repair.
- [x] MD009 Implement streaming checksum/size verification, same-filesystem
  staging, sync/atomic publication, readiness records and crash recovery.
  Publish new checkpoint discovery/config only after its closure is ready.
  Recheck racing existing destinations and never expose a partial weight file.
- [x] MD010 Implement cross-process locks, deterministic lock ordering,
  cancellable waiting and shared-artifact transfer deduplication. Reuse verified
  identical files with CoW/copy fallback, without writable hard links. Estimate
  space for staging/copy fallback and handle mid-transfer disk exhaustion.

## Native HTTP transfer

- [x] MD011 Implement bounded libcurl streaming with four-file concurrency,
  checked sizes/writes, progress and cancellation callbacks, and structured
  component errors. Centralize safe libcurl initialization/lifetime across
  CLI/server threads. Compile against the currently pinned curl API versions.
- [x] MD012 Implement HTTPS-only redirects, CA discovery/overrides, proxy
  support and optional `HF_TOKEN` authentication. Restrict credential forwarding
  to the intended origin and redact signed URLs/tokens in diagnostics/events.
  Reuse packaged CA behavior without inheriting media-import size/time limits.
- [x] MD013 Implement persistent partial metadata and HTTP Range resume for
  pinned content, including prefix verification, correct 206/Content-Range,
  ignored ranges, 416/complete partials, expired CDN URL refresh and source
  changes. Verify actual large HF artifacts work through direct HTTP.
- [x] MD014 Implement five-attempt retry/backoff, bounded Retry-After handling,
  connect/inactivity limits, prompt cancellation and useful terminal/non-TTY
  progress. Distinguish permanent HTTP/TLS/hash/storage errors from transient
  failures. Report download duration separately and in operation wall time.

## CLI and offline behavior

- [x] MD015 Add `--models-path` and the prefetch/list/offline operations and
  environment policy to the shared option contract/help. Validate group syntax
  and option conflicts; honor the resolved root and corresponding path overrides
  in generation, decode/resume, inspection, prefetch and list operations.
  Prefetch/list need no GPU/prompt;
  list, help, model/state inspection and validation-only paths never download.
- [x] MD016 Integrate default preparation into every execution branch after
  effective request validation and before weight/GPU allocation. Cover all
  operations from MD001, including decode-only and resume. Invalid requests,
  missing media/states/LoRAs and unsupported backends must not trigger unrelated
  transfers. Keep low-level inference APIs local-file based.
- [x] MD017 Enforce offline behavior with zero outbound model requests, including
  metadata/auth probes. Return precise missing-component diagnostics. Keep
  complete local inference usable without network/DNS/CA stores. Add explicit
  offline execution to model-dependent tests so default downloads cannot hide
  missing prerequisites or make negative tests access the internet.

## Server preparation and resource lifetime

- [x] MD018 Allow empty configured managed roots at server startup and split
  admission from model availability. Accept startup `--models-path`, derive
  main/auxiliary paths with startup `-d` precedence, and pass resolved absolute
  destinations to workers. Reject per-job `--models-path`; retain permitted
  per-job explicit model overrides. Resolve missing authorized paths without
  weakening read-root checks. Existing readable custom paths remain usable;
  request-selected missing paths outside administrator-owned destinations do
  not grant download/write authority. Unknown `model` values still fail.
- [x] MD019 Add persisted dependency plans/readiness and a supervised CPU-only
  preparation helper using the current executable. Keep transfers out of HTTP
  handlers and DB transactions; preserve FIFO inference and queue reservations.
  Share installed files across variants/jobs and keep inference workers offline.
- [x] MD020 Integrate effective dependency-file resource snapshots. Preserve
  identities of existing inputs through preparation, snapshot newly installed
  files before worker startup, and ignore downloader metadata/unrelated groups.
  Adding Ref2VA/auxiliaries must not invalidate ready FL2VA jobs; modifying a
  required checkpoint must still fail. Preserve native saved-state fingerprints.
- [x] MD021 Add bounded preparation events/byte progress under `h3`, using queued
  standard status and the design's preparation phases. Implement startup
  `--server-model-download-timeout`, separate inference timing, cancellation,
  shutdown/restart and idempotency behavior. Retain resumable partials and avoid
  deleting shared completed files when one job is cancelled.
- [x] MD022 Apply startup/environment/per-job offline policy consistently,
  including URL asset import. Update discovery/schema/option coverage without
  adding SGLang request fields or a download endpoint. Keep credentials and
  network enablement outside client-controlled request data; model storage uses
  its own free-space checks rather than the server's result-storage quota.

## Automated fault and integration tests

- [x] MD023 Add host-only catalog/resolver tests covering every MD001 operation,
  all quality presets/negations, explicit paths, shared dependencies, conflicting
  catalog entries, resume metadata and minimal closures. Verify full decoder
  `--show`, preview overrides and state-only plans do not overfetch auxiliaries.
  Test `--models-path` with absolute/relative/spaced paths, trailing slashes,
  `MiniMaxH3`/`MiniMax-H3` selection, `-d`/auxiliary precedence in either flag
  order, no fallback to cwd models, and identical lookup/download destinations.
  For each of the four components, test root-derived existing/missing/offline
  paths and explicit override paths; prove one override does not relocate the
  other components or trigger an unused model download.
- [x] MD024 Add deterministic tiny HTTP(S) fixtures for successful fetch, TLS/CA,
  redirects, token containment, proxies, 401/403/404/429/5xx, timeouts, truncation,
  wrong size/hash and all MD013 Range cases. Inject fixtures only through test
  interfaces; production has no arbitrary catalog/HTTP bypass. Assert bounded
  memory, retry counts and cancellation latency.
- [x] MD025 Add filesystem/process tests for no-clobber publication, disk-full,
  permissions, partial custom-tree conflicts, symlink swaps, missing/corrupt
  receipts, crashes at publication boundaries, interrupted resume, lock recovery,
  simultaneous processes and CoW/copy fallback. Prove warm read-only use succeeds.
- [x] MD026 Add CLI/server tests for automatic defaults, offline/inspection
  network absence, responsive queue admission, byte progress, cancellation,
  restart, duplicate submissions, shared downloads, preparation timeouts and
  resource mutation detection. Verify additions of unrelated components do not
  invalidate ready jobs or change saved-state identities. Verify server workers
  inherit the startup models root despite different cwd, startup-only flag
  rejection in jobs, and offline/custom-root operation without stray downloads.

## Builds, real downloads and GPU qualification

- [x] MD027 Build application/library/test targets without warnings on local M4
  and PRO 5000. Run the current host/Metal suite and downloader tests. Keep
  outputs under `bin/`, preserve dependency pins and avoid new runtime tooling.
  Run complete recorded CUDA gates at coherent implementation milestones.
- [x] MD028 Build the final standalone Linux executable with embedded catalog
  and existing runtime. Audit dependency/source materials and confirm no Python,
  HF CLI, external curl, model payload or credentials are required/included.
  Validate deterministic catalog embedding and relocatable model destinations.
- [x] MD029 On PRO 5000, use that executable and an isolated empty destination
  to install every catalog group from the real pinned sources. Record exact
  file/catalog hashes, transferred/reused bytes, disk use, timings and peak host
  memory. Interrupt/resume at least one real large transfer. Preserve installed
  files for subsequent tests rather than repeatedly fetching the main weights.
- [x] MD030 On M4, prove a real automatic preview-VAE download into an empty
  auxiliary location, then run short Metal samples. Run the design's bounded
  text/full, preview, first-last, references, still/decode, AV decode,
  continuation/bridge, sampler resume and upscale matrix across both backends.
  Use 56 frames for mixed-reference cases where required; retain videos/stills.
- [x] MD031 Run real server jobs on M4 and PRO 5000 with initially absent managed
  components, proving preparation progress, successful short rendering and
  offline warm reuse. Use fixtures/CoW staging for root/queue/crash permutations;
  include a genuine network-backed auxiliary acquisition. Compare managed versus
  manually provisioned identical weights with the same build and settings.
- [x] MD032 Run the existing current CUDA feature suite and server suites on
  PRO 5000, with models installed and outbound model access blocked. Run the
  unchanged complete **204/204 SGLang parity gate** on the final source build
  and final standalone executable. Preserve all hashes/fixtures and historical
  gate codecs; normal samples keep production FFmpeg/FFprobe 9.0.2.
- [x] MD033 Exercise the same standalone artifact under Ubuntu 22.04, Ubuntu
  24.04 and Debian 12 userlands on PRO 5000. Verify HTTPS/CA behavior, an actual
  small model transfer, warm offline rendering, non-checkout execution, custom
  paths and read-only complete models. No host Python/HF/external curl dependency.

## Documentation and completion

- [x] MD034 Replace the preview Python helper with native prefetch guidance and
  remove obsolete callers/default paths. Update README, preview/still/upscale
  guides, server guide/design addendum, CLI help and portable installation docs
  for default downloading, groups, `--models-path`/`-d` precedence, auxiliary
  overrides, existing `MiniMax-H3` discovery, offline use, progress,
  storage requirements and recovery. Keep setup/build-only operations offline.
- [x] MD035 Write `docs/features/model-downloads-results.md` with source/catalog/
  binary identities, actual upstream pins, transfer/resume measurements, all
  test outcomes, sample links and complete 204-output parity evidence. Distinguish
  real network bytes from fixture/CoW reuse and record platform limits/failures.
- [x] MD036 Validate documentation links, final diff, catalog reproducibility
  and secret/artifact exclusions. Mark tasks complete only with linked evidence;
  ensure default cold/warm/offline CLI and server behavior matches the design.
  No publishing or model-revision update is implied by completing this list.
