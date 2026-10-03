# Native model downloads

Status: **implemented and qualified on M4 and RTX PRO 5000**. Written 2026-09-29;
updated 2026-09-30. All 36 implementation tasks are complete; see the
[qualification report](model-downloads-results.md).
Implementation checklist: [completed model-download tasks](model-downloads-tasks.md).

## Outcome

A normal h3cli command downloads missing, supported model components before
loading weights, then continues with the requested operation. **Downloading
missing models is the default; there is no `--download-missing` flag or
confirmation prompt.** Implement this with native libcurl on Metal, CUDA and
the standalone Linux release. Python, `hf`, Git and a separately installed
`curl` executable are not runtime dependencies.

Keep model data outside the executable and its immutable extracted runtime.
Preserve existing numerical behavior, quality presets, existing model
installations, custom weights and saved-state validation. The models root still
defaults to `models/`; the directory selection rules below retain existing
`MiniMax-H3` installations while using `MiniMaxH3` for new installations.
A complete local installation makes
no model-network requests, including version checks. New weights are pinned to
reviewed revisions and verified before use; this is not an automatic updater.

This design supersedes the earlier explicit-download-only policy for managed
models. The SGLang request surface remains unchanged: standard fields plus the
existing `h3cli` string. Server clients cannot choose download URLs or repositories.

## Current implementation and integration points

- [The option registry](../../src/cli/options.def) supplies main and auxiliary
  model paths; [h3cli.c](../../src/h3cli.c) dispatches generation, inspection,
  still decoding, AV decoding, resume and upscale through different branches.
- [engine.c](../../src/engine.c) recognizes the `FL2VA` and `Ref2VA` trees.
  Startup currently inventories available checkpoints and common components,
  which must be accounted for when defining minimal download closures.
- The former Python preview helper has been replaced by native prefetch.
  The [catalog](../../src/models/catalog.json) retains its exact qualified pin;
  all component defaults now follow the selected shared root.
- [The Makefile](../../Makefile) already links libcurl into the CLI. The server's
  [asset importer](../../src/server/assets.c) uses it for short, bounded media
  imports. Model downloads need a separate transfer policy for very large files.
- [Server startup](../../src/server/server.c) permits missing managed roots;
  [admission](../../src/server/queue.c) persists a dependency plan and
  [resource snapshots](../../src/server/resources.c) before queuing. Preparation
  refreshes required file identities and writes readiness before inference;
  installing an unrelated group does not change the job's snapshot.

## Command and path contract

Implemented commands:

```sh
# Missing FL2VA dependencies and the preview VAE download automatically.
./bin/h3cli --quality preview -p 'A small boat on a quiet pond.' \
  --width 256 --height 256 --frames 22 --steps 2 -o preview.mp4

# Prefetch named groups, then exit without requiring a GPU or a prompt.
./bin/h3cli --download-models base,preview
./bin/h3cli --download-models all

# Set the shared root: main model is /data/models/MiniMaxH3 when -d is absent.
# Auxiliary models also use /data/models/<component>/ unless overridden.
./bin/h3cli --models-path /data/models --download-models all

# Override only the main model directory; auxiliary models still use the root.
./bin/h3cli --models-path /data/models -d /other/custom-h3 --download-models all
./bin/h3cli -d /data/MiniMax-H3 --download-models base,references

# Inspect the embedded catalog and local availability without networking.
./bin/h3cli --list-models

# Strictly local execution; report missing components instead of downloading.
./bin/h3cli --offline -p 'A small boat on a quiet pond.' -o video.mp4

# Queued jobs prepare their required models automatically.
./bin/h3cli --server
./bin/h3cli --server --models-path /data/models
./bin/h3cli --server --offline
```

`--download-models` accepts a comma-separated set of `base`, `references`,
`preview`, `image-vae`, `upscale` or `all`. Reject unknown/empty entries and
generation options in this standalone operation. Deduplicate repeated groups.
`--models-path` and main/auxiliary path overrides apply to their corresponding groups.
Prefetch verifies the selected catalog artifacts; it does not overwrite a
different existing checkpoint. `--download-models ... --offline` verifies local
files and fails if installation would require network access.

`--list-models` reports groups, pinned revisions, destinations, expected sizes,
local availability and estimated missing bytes. It performs only local reads;
unverified files are labeled unverified. Shared hashes count once for transfer
size, while the disk estimate accounts for the available materialization method.
Do not scan/hash hundreds of gigabytes merely to display this list.

`--offline` and `H3_OFFLINE=1` disable outbound model requests. The environment
setting cannot be weakened by a job. On the server, offline also rejects URL
asset imports and conflicts with `--server-allow-url-inputs`; HTTP listening and
local uploaded assets remain available. A per-job `--offline` in `h3cli` can
only tighten the startup policy. `--download-models` and `--list-models` are
process operations, not accepted job flags.

Help, version/build information, `--info`, saved-state inspection, list mode,
and validation-only invocations never download or create model directories.
Missing local data produces an actionable diagnostic. Invalid requests fail
before downloads whenever their invalidity can be determined without weights.
Build/setup scripts also continue to avoid model downloads.

### Destinations and existing installations

Add **`--models-path PATH`** to specify the shared directory containing the main
and auxiliary models. Its default is `models`, relative to the invocation cwd.
An absolute path stays absolute; resolve a relative root against the original
CLI/server startup cwd once, before entering any worker directory. Reject an
empty value. Repeated values use the last value, matching other scalar options.

With `ROOT` denoting that resolved directory, lookup and automatic download use
the same destinations:

| Component | Destination under the shared root | Explicit override |
| --- | --- | --- |
| Main model | `ROOT/MiniMaxH3` | `-d/--model-dir` |
| Preview VAE | `ROOT/preview-vae/taeh3.safetensors` | `--preview-vae-model` |
| Image VAE | `ROOT/image-vae/minimax_h3_t1_image_vae_step1597.safetensors` | `--image-vae` |
| Latent upscaler | `ROOT/latent-upscale/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors` | `--upscale-model` |

Apply one policy to every row: its explicit override wins; otherwise use its
path under `ROOT`. Look for the component there first and, when required and
missing, automatically download it to that same location. Offline mode reports
that location as missing. Selecting a root does not by itself download unused
components. Auxiliary paths must not retain hard-coded cwd-relative `models/`
prefixes after a different root is selected. The main directory's historical
spelling fallback below is the only additional compatibility lookup; do not
introduce an auxiliary search outside the selected root.

Resolve paths independently of command-line ordering, using these rules:

1. An explicit `-d/--model-dir` is the exact main checkpoint directory and takes
   precedence over `--models-path`. Do not append `MiniMaxH3` to `-d`. Changing
   `-d` does not relocate auxiliary defaults.
2. Without `-d`, select `ROOT/MiniMaxH3`. For compatibility with current
   installations, if that entry is absent and `ROOT/MiniMax-H3` exists, reuse
   the latter. If both exist, select `MiniMaxH3`; if neither exists, install
   into `MiniMaxH3`. A present but incomplete, invalid or dangling canonical
   entry is an installation error/adoption case, not a reason to silently
   switch to the other tree. Never merge the two trees or rename an existing
   installation. `--list-models` and startup diagnostics show the selected path.
3. An explicit `--preview-vae-model`, `--image-vae` or `--upscale-model` wins
   over its derived path. Relative explicit paths retain their existing
   meaning relative to invocation cwd; do not rebase them under `ROOT`.
4. An explicitly selected root is authoritative. Do not search the old cwd's
   `models/` or the executable/runtime directory if a file is absent there.
   A missing required managed component downloads to the selected destination,
   or fails locally under `--offline`.

For example, `--models-path /data/models` with no `-d` looks for
`/data/models/MiniMaxH3`; `--models-path /data/models -d /other/custom-h3`
uses `/other/custom-h3` and still finds the default preview VAE at
`/data/models/preview-vae/taeh3.safetensors`. With neither flag, a new main
installation uses `models/MiniMaxH3`, while an existing `models/MiniMax-H3`
installation is reused through rule 2. These rules apply to generation,
decoding, resume, inspection, list mode and prefetch on both platforms and in
the standalone package. They do not relocate media, saved states, outputs,
LoRAs or derived caches.

Keep `-d/--model-dir`, `--preview-vae-model`, `--image-vae` and
`--upscale-model`. For CLI execution, a missing explicit path is a destination
for the same pinned component that would be installed at the default path;
it is not a repository identifier or a request to locate a similarly named
checkpoint. Print the source and destination before transferring. This applies
to an empty main directory and to an absent auxiliary file with any basename.
Users needing strict existence checks select offline execution.

Existing complete custom installations continue through the current loaders
without enforcing official-model hashes or importing them into a cache.
Existing auxiliary files are never replaced automatically. Missing LoRAs,
reference media, saved states, quantization caches and folded/custom checkpoints
are not downloadable catalog components.

For a partially populated main tree with no downloader receipt, compare the
existing files needed for safe adoption against the pinned catalog before
adding anything. Hash all overlapping files in the affected checkpoint and
shared dependency closure; do not mix revisions or append official shards to
custom/folded weights. If that cannot establish a consistent installation,
stop and name the conflicts, suggesting a separate empty destination. Do not
delete, repair or upgrade existing files as a side effect of rendering.

Managed installations have receipts tying files to hashes and catalog identity.
For unchanged files, use the verified receipt and file metadata to avoid full
rehashing on every render. Rehash changed files before trusting them. Corrupt
or conflicting present files fail explicitly; missing files can be restored
from the same catalog. A new executable must not silently combine its newer
catalog with an older partial installation. Complete compatible installations
remain usable; completing a different revision requires a separate destination
or an explicit future migration feature.

No global second copy of the model is required. Reuse identical verified files
within the selected installation, using CoW copies where supported and ordinary
copies otherwise. Avoid writable hard links between user-visible checkpoints.
Existing model-root symlinks remain supported: resolve the selected root once,
then confine publication beneath that resolved directory.

## Catalog and dependency planning

Keep one reviewed, versioned catalog in the repository under `src/models/` and
embed it in every executable with a deterministic build rule. Runtime discovery
must not read a checkout-relative JSON file or fetch a mutable remote catalog.
Each artifact records a stable ID, group memberships, upstream repository and
full commit, repository-relative source path, destination-relative path,
complete-file SHA-256, exact byte size and source/license provenance. Reject
duplicate/conflicting destinations, unsafe paths and dependency cycles.

The first catalog matches the models already qualified by h3cli. Main revision
`42ed227ee7df40d41602854ae760620d6eb651fe` was verified by full-file SHA-256
against all 115 existing main files. The [file inventory](model-download-dependencies.md)
records the 119-artifact closure, including tokenizer/configuration files,
shard indexes, every required shard, notices and the three auxiliaries. No
model-repository Python is downloaded for execution. The catalog is a flat
artifact/group manifest without dependency edges, so cycles are not representable.

The existing auxiliary pins are the starting contract:

| Group | Source | Revision | Complete-file SHA-256 |
| --- | --- | --- | --- |
| `preview` | `madebyollin/taehv`, `safetensors/taeh3.safetensors` | `62f7591f59dfbb4c3c02b7a621d180a9eeaba26c` | `4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13` |
| `image-vae` | `Mamad8/MiniMax-H3-Image-VAE`, default filename above | `c7b9252c73707dba494cf4d99ca45d3f33f561b3` | `6c3d0bfa055986a803a566a862fcde283a1e63db62829e5ef4a2a5aebf50bb86` |
| `upscale` | `LBH-123-AI/Minimax_h3_latent_Upscaler`, default filename above | `3f941d5d182014dd5c0a5e16330420ee2d4aa0c6` | `4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6` |

These pins are preserved in the [native catalog](../../src/models/catalog.json),
with provenance from the [preview qualification](../preview/preview-vae.md),
[image-VAE qualification](design-single-still.md) and
[upscale contract](../../tests/upscale/contract.json). All three complete-file
hashes and sizes have been checked against the qualified local files and pinned
publisher metadata; tensor payload size alone is not used as the file size.

Implement a shared, side-effect-free resolver from the **effective operation**
to required artifacts. Run it after quality expansion, explicit overrides,
reference ordering, and bounded parsing of saved-state metadata. CLI and server
must use the same resolver; do not infer requirements from flag presence alone.

| Operation | Required closure |
| --- | --- |
| Text / first-last video | FL2VA transformer, conditioning, tokenizer and required video/audio components |
| Image, video or audio references | Ref2VA transformer and required conditioning/encoding/decoding components |
| Preview decoding or preview quality | Add tiny decoder only when effective preview decoding is enabled; retain full VAE files required for reference encoding or model identity |
| `--show` with ordinary full decoding | Follow the actual selected decoder; `--show` alone does not imply the tiny model |
| Still generation | Selected main checkpoint/conditioning plus image VAE and any genuinely required main-model metadata |
| Still latent decoding | Image VAE and saved input; no unrelated main transformer download |
| AV latent decoding | Selected video decoder, audio decoder, and exact files required by current identity checks; no unused transformer/text weight transfer |
| Continuation / bridge | Requirements of the effective generation mode plus saved-state identity dependencies |
| Sampler resume | Mode and required weights recovered from the checkpoint, plus current permitted delivery overrides |
| Fresh latent upscale | Saved source's main-model closure, learned upscaler and required delivery components |
| Upscale sampler resume | Audit saved-stage requirements; do not redownload the learned network if transfer is already represented in the checkpoint |
| State-only / stopped jobs | Include only components actually used for computation, previews or identity validation |

The [file-level dependency table](model-download-dependencies.md) records the
implemented closure and the native inventory/identity constraints.
Where startup inventory currently requires unrelated components, separate
operation-specific requirements from broad model inspection without changing
inference or relaxing saved-state identity checks. Do not advertise a smaller
closure until the real loader can execute with only those files.

`base` and `references` prefetch full supported FL2VA and Ref2VA closures
respectively. The three auxiliary groups fetch their own artifacts; `upscale`
does not guess a saved source's checkpoint. `all` is the union. Quantization and
LoRA folding still derive their caches locally from the existing supported base
weights; this feature does not add remote quantized or LoRA catalogs.

## Transfer and installation implementation

Use themed native modules under `src/models/` for catalog access, dependency
planning, HTTP transfer and installation/receipts. Expose progress/cancellation
callbacks and structured errors; keep terminal rendering in the CLI adapter.
The preparation layer runs before weight loading and GPU memory allocation.
Low-level inference APIs remain local-file APIs; networking is explicit inside
the shared CLI/server orchestration layer.

Use immutable HF `/resolve/<commit>/<path>` and pinned GitHub raw URLs.
Implement ordinary HTTPS downloads; Xet protocol support is outside this
version. HF documents backwards compatibility through its LFS bridge, but
qualify direct HTTP and Range requests for our actual pinned large artifacts
instead of assuming optimized Xet-client performance.
[HF storage documentation](https://huggingface.co/docs/hub/xet/using-xet-storage).

Required transfer behavior:

1. Initialize libcurl once with a lifetime safe for concurrent server requests.
   Use APIs available in the existing pinned Linux libcurl; preserve current
   dependency versions and zero-warning builds on both platforms.
2. Require HTTPS and certificate/hostname verification, including redirects.
   Reuse packaged/native CA discovery and `CURL_CA_BUNDLE` / `SSL_CERT_FILE`.
   Support conventional HTTPS proxy configuration. Credentials come from
   `HF_TOKEN` in the process environment, never a job flag, catalog or log.
   Send HF authorization only to the intended HF origin; signed CDN redirects
   do not receive that header. Bound redirects and redact query strings/tokens.
   Use host/protocol checks compatible with the pinned libcurl version.
   [libcurl redirect contract](https://curl.se/libcurl/c/CURLOPT_FOLLOWLOCATION.html).
3. Stream to disk with bounded buffers and bounded file concurrency (initially
   four transfers). Do not buffer weights in RAM. Check size arithmetic and
   write results; cap each response at the pinned size. Estimate free space
   before starting, including partial files, staging and copy fallback.
4. Store a `.part` file and resume metadata containing artifact/catalog/hash
   identity outside loader-scanned directories. Resume through HTTP Range with
   validated status and Content-Range. If a server ignores Range, restart that
   file safely; never append a full response. Handle complete-part/416 cases
   by verifying size/hash, and refresh expired signed URLs from the original
   pinned URL. Do not treat an HTTP ETag as a universal file SHA-256.
   [libcurl resume API](https://curl.se/libcurl/c/CURLOPT_RESUME_FROM_LARGE.html).
5. Bound attempts (five per transfer), use exponential backoff and bounded
   Retry-After handling for transient errors, and distinguish 401/403/404,
   transport/TLS failures and storage failures. Use connect and inactivity
   limits suitable for large downloads, not the media importer's 30-second
   whole-transfer timeout. Cancellation interrupts I/O/backoff promptly.
6. Verify exact bytes and full SHA-256, including resumed prefixes, before
   publishing. Sync data, rename on the same filesystem without clobbering a
   racing destination, then sync the parent. Recover cleanly from failures
   between file publication and receipt publication.
7. Use advisory filesystem locks keyed by canonical destination and artifact
   identity, with a deterministic lock order. Recheck after acquiring locks;
   another process may already have installed the file. Process death releases
   ownership; never steal a live lock because a download is slow. Cancellation
   releases locks and retains resumable data.
8. Publish complete dependency groups with a readiness record. Consumers wait
   for the entire required closure before loading. Stage a newly introduced
   checkpoint tree and publish its discovery marker/config only when its
   dependencies are ready, so another invocation cannot inventory a half-built
   variant. Complete files already in use remain immutable.

Keep downloader metadata and staging outside directories scanned by native
model/state signatures. Use a private sibling `.h3cli-downloads/` directory on
the destination filesystem, keyed by canonical destination and catalog ID;
never embed staging paths in state or server resource identities. Handle a
missing root by resolving and validating its existing ancestors before mkdir.
Use directory-relative operations to prevent destination substitution during
publication. Temporary plain-HTTP fixture servers are a test-only transport
injection, unavailable through shipping flags or environment variables.

Report component, bytes transferred/total, reused bytes, verification phase,
retry/wait state and elapsed time. CLI progress goes to stderr and works without
a TTY; completion includes download time separately and in total wall time.
No stdin question or pause is needed before an automatic download. On failure,
name the component, destination and remedy, retain useful partial data, and do
not start inference or publish media. Do not recursively chmod/chown existing
model trees or require write access on a complete read-only installation.

## Server lifecycle and ownership

Start an empty configured model root without downloading every group before
binding the service. Startup validates configuration/authorized destinations;
each valid job determines its missing dependencies. Default downloads are
enabled here too. Administrators disable them with startup `--offline` or the
environment setting. This is independent of `--server-allow-url-inputs`.

`--models-path` is accepted at server startup and derives the managed main and
auxiliary destinations by the same rules as CLI execution. A startup `-d`
overrides only the main directory. Persist absolute resolved paths for jobs;
workers never derive model locations from their private cwd. This is a process
configuration option, not a per-job `h3cli` option. Discovery documents that
restriction; per-job `-d` and auxiliary overrides remain available under the
existing authorization rules.

The write destinations are the administrator's configured main directory and
default auxiliary model locations under the resolved `--models-path` root.
Explicit per-job paths
outside these destinations retain existing read-root rules but cannot cause
new writes/downloads merely because they are readable. A missing such custom
path fails with an administrator-facing prefetch instruction.
An unknown SGLang `model` value is still rejected, never resolved against HF.

Split admission and execution into these phases:

1. Normalize and validate the effective request without model network access.
   Import/probe authorized user inputs, validate saved-state headers as needed,
   resolve the dependency plan, reserve queue capacity and persist the job.
   Complete existing resource identities can be captured immediately.
2. The queued job's preparation phase downloads missing managed components in
   a cancellable, supervised CPU-only helper using the same executable. Keep
   HTTP request threads and DB transactions out of transfers. Initially one
   preparation job at a time is sufficient; it may use the four-file downloader.
   Preserve FIFO inference and do not reserve GPU memory during preparation.
3. Validate the installed closure, finalize its resource snapshot atomically
   and persist readiness before starting the inference worker. The worker runs
   with model networking disabled; any remaining missing resource is an error.
4. Preserve standard SGLang statuses. Preparation remains `queued`, with
   `h3.phase` reporting `waiting_for_models`, `downloading_models` or
   `verifying_models`; byte progress belongs under `h3`, not denoising percent.
   Emit bounded progress events through the existing event mechanism. Network
   failure terminates the job with a structured component error and no GPU work.

Separate a configurable startup `--server-model-download-timeout` (default
21,600 seconds) from `--server-job-timeout`. Start the former when preparation
begins; queue waiting does not consume it. Start the latter with inference.
Both waits and transfers are cancellable. Model bytes and partials live outside
the server's result-storage quota; check destination free space separately and
report expected model disk use. The finite embedded catalog limits what jobs
can install; there is no arbitrary model import endpoint.

Deduplicate preparation across variants, jobs and other CLI/server processes.
Cancelling one job must not remove a file another consumer is using. Retain
partials on cancellation. Server shutdown/restart follows the existing
interrupted-job policy; a fresh/retried job can resume the retained download.
Keep idempotent submissions tied to one persisted job during preparation.

Replace the resource snapshot's whole-root identity with a snapshot of the
**effective required files**, including custom paths/LoRAs where applicable.
Ignore receipts, partials and unrelated components. Installing Ref2VA or an
auxiliary must not invalidate a ready FL2VA job, but editing any file that job
actually requires must still fail the existing resource check. Retain early
identities of existing custom files throughout preparation to detect mutation.
Do not change native sampler/continuation fingerprints or their compatibility
rules as part of this server bookkeeping change.

## Packaging, documentation and validation

Embed the catalog in native and packaged executables, include its provenance
in the companion source materials and preserve all current CUDA/media pins.
The packaged downloader uses the existing CA selection and bundled libcurl;
it must work without a checkout, Python, `hf` or external curl. Existing local
inference must also work without DNS, network access or an installed CA bundle.
Model payloads, credentials and download staging never enter the release.

Replace the preview Python downloader with documented native prefetch commands;
remove it and its stale destination references after updating callers/docs.
Update README, preview/still/upscale guides, server help/schema/discovery and
portable installation instructions. Clearly distinguish automatic first-use
installation, explicit prefetch, local inspection, `--models-path` versus `-d`,
auxiliary overrides, existing `MiniMax-H3` discovery and offline use.

Use local M4 for Metal/host validation and the local RTX PRO 5000 for CUDA and
portable Linux validation. Keep downloaded test trees isolated from existing
qualified models. Ordinary model tests use tiny fixture files and a controlled
HTTP(S) service, without internet access or GPU requirements. Fault injection
covers Range/redirect/auth/retry/TLS errors, truncation, hash mismatch, disk-full,
permissions, symlinks, cancellation, crashes and concurrent processes. Add
request-plan tests for every operation/override combination in the table above.
Add path-resolution cases for absolute/relative roots, spaces, trailing slashes,
symlinks, missing roots, both main directory spellings and precedence independent
of flag ordering. Verify auxiliary overrides, no fallback outside the selected
root, read-only/offline lookup, and inheritance of resolved roots by server
workers. Reject per-job `--models-path` as a startup-only option.

Real acceptance requires one verified clean installation of all catalog groups
on the PRO 5000 through the shipping executable, using actual upstream URLs.
Measure transfer/disk sizes, timings, peak host memory and retry/resume behavior.
Reuse verified artifacts or CoW copies for later cases instead of repeatedly
downloading large weights; record which cases exercised real network bytes.
Exercise a real pinned auxiliary download on M4 as well. No existing model file
needs to be removed to simulate a cold start.

Run these bounded sample cases with the prepared models and retained outputs:

| Case | Scope |
| --- | --- |
| Automatic default generation | Text video, 256×256, 22 frames, two steps, full decoder |
| Automatic preview | Same geometry/steps, preview VAE initially absent; test preset/override selection separately |
| References | One image with `--ref-image-size max`, plus bounded video/audio-reference coverage at 56 frames where required |
| First/last | Two anchors, 256×256, 22 frames, two steps |
| Still and saved decoding | Supported small still, decode-still, save/decode AV with full and preview VAE |
| Saved workflows | Small continuation/bridge and sampler resume; create states against the prepared model tree |
| Upscale | 256×256 source to 512×512, 22 frames, two refinement steps; include a resume planning check |
| Server | Empty managed root, queued download/progress, duplicate submissions, cancellation/restart, then successful short render |
| Offline / warm | Repeat selected renders with outbound requests blocked; compare with manually provisioned identical model bytes |

Run the existing host/Metal and current CUDA/server suites, plus the complete
**204-output recorded SGLang regression** on PRO 5000 as required by
[CONTRIBUTING.md](../../CONTRIBUTING.md). Keep all 204 expected hashes and 12
input fixtures unchanged. Force model networking off in the gate; do not make
missing fixtures/models trigger downloads or fallback. Compare manual versus
managed preparation with the same build/backend/settings; no new upstream
SGLang inference or golden regeneration is needed.

Rebuild and exercise the final standalone artifact, including HTTPS and warm
offline use under the existing Ubuntu 22.04, Ubuntu 24.04 and Debian 12 coverage.
The final artifact must pass the recorded gate on PRO 5000. Report actual tested
platforms, catalog identity, source/binary identities and remaining failures;
an unrun case or missing source pin is not a pass.

No inference algorithm changes, new model formats, model auto-updates, arbitrary
repository/URL loading, remote LoRA distribution, general cache eviction UI or
HF/Xet client embedding are part of this feature.
