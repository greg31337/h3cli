# Runtime LoRA folding and cached model variants

Status: implemented and validated on September 19, 2026. The existing offline
workflow remains available. All implementation tasks in [../todo.md](../todo.md)
are complete; [runtime-validation.md](runtime-validation.md) records the host,
Metal, and RTX 5090 acceptance results and their scope.

## Objective

Allow users to select ordinary H3 LoRAs directly on the `h3cli` command line.
Before inference, H3 resolves the requested adapter combination to a persistent
folded BF16 transformer. A valid cached variant is reused. On a cache miss, H3
folds the adapters, validates and publishes the variant, then continues the
original generation request automatically.

The startup operation runs inside the native H3 library on macOS and Linux.
It requires no Python environment, subprocess folding tool, model download, or
manual model assembly. The resulting transformer uses the existing inference
and weight-streaming paths; there are no LoRA matrix multiplications in the
denoising loop.

```text
base model + ordered --lora arguments
                  |
       validate request and select mode
                  |
       derive folded-variant identity
                  |
       +----------+-----------+
       |                      |
 verified cache hit      miss / invalid entry
       |                      |
       |              native FP32 folding
       |              one BF16 rounding
       |              validate + publish
       +----------+-----------+
                  |
       effective BF16 transformer
                  |
       optional FP8 / NVFP4 preparation
                  |
         existing H3 generation
```

## Current implementation and integration points

- [../../lora/fold_lora.py](../../lora/fold_lora.py) implements strict adapter
  mapping, alpha/rank scaling, bounded matrix folding, one BF16 rounding,
  source protection, and numerical validation. It is the compatibility
  reference and independent test oracle for the native implementation.
- [../../lora/workflow.py](../../lora/workflow.py) assembles a complete model
  around an offline transformer. Runtime folding resolves the transformer
  directly and needs neither assembly nor persistent base-model symlinks.
- [../../src/weights/lora.c](../../src/weights/lora.c) owns adapter selection, strict planning,
  deterministic CPU folding, validation, cache publication, and reader leases.
  [../../src/weights/lora_json.c](../../src/weights/lora_json.c) supplies strict JSON and
  locale-independent numeric parsing.
- [../../src/engine.c](../../src/engine.c) adds `h3_load_dir_with_lora`, inventories available
  modes, and resolves the effective transformer before allocating inference
  resources. Shared encoders and VAEs retain their base-model paths. A saved
  sampler checkpoint selects its original mode before preparation.
- [../../src/sampling/sampler_state.c](../../src/sampling/sampler_state.c) fingerprints the selected
  model tree, including its files and relative names. Resolving only the DiT
  loader path would leave resume and quantization provenance referring to the
  wrong weights. The additive effective-model fingerprint entry points now
  substitute the same transformer while retaining its logical relative names.
- [../../src/weights/safetensors.h](../../src/weights/safetensors.h) provides native tensor
  inventory and reading, including the adapter metadata needed by folding.

This design adds runtime preparation to the scope of the earlier
[offline design](design-lora.md). The offline tool and its documented behavior
remain supported. Runtime artifacts have their own recipe and schema versions.

## Command-line contract

| Flag | Behavior |
| --- | --- |
| `--lora PATH[:SCALE]` | Append an adapter in command-line order. Repeat for multiple adapters. Default scale: `1.0`. |
| `--lora-cache DIR` | Override the persistent folded-transformer cache root. |
| `--lora-memory-mib N` | Positive integer scratch-memory budget for folding; default `512` MiB. This does not set the inference memory budget. |

The default cache is `$XDG_CACHE_HOME/h3/lora` when `XDG_CACHE_HOME` is an absolute
path, otherwise `$HOME/.cache/h3/lora`. An explicit flag takes precedence. If
neither an explicit path nor a usable default is available, report an error.
Do not fall back to writing inside the model directory. Reject empty cache
paths, invalid memory values, and cache/memory flags without any `--lora`.

Example using Turbo and the requested fast preview settings:

```sh
./bin/h3cli -d ./models/MiniMax-H3 \
  --lora ./lora/downloads/minimax_h3_turbo_v4_step600_ema.safetensors:1.0 \
  --lora-cache ./outputs/lora-cache \
  --ref-image ./inputs/face1.jpg \
  -p 'The person walks through a forest and sings.' \
  --width 1344 --height 768 --steps 8 \
  --fast-cuda --cuda-denoise-quant nvfp4 --preview-vae \
  -o ./outputs/turbo-preview.mp4
```

Multiple adapters use the same syntax as the offline tool:

```sh
./bin/h3cli -d ./models/MiniMax-H3 \
  --lora ./lora/downloads/minimax_h3_turbo_v4_step600_ema.safetensors:1.0 \
  --lora ./lora/downloads/h3-realism-people-t2v-i2v-r2v.safetensors:0.7 \
  --ref-image ./inputs/face1.jpg \
  -p 'r34l1sm, the person smiles and turns toward the camera.' \
  --steps 8 -o ./outputs/combined.mp4
```

The combined example illustrates syntax, not a quality guarantee. Adapter
profiles may report guidance; they never change steps, schedules, prompts,
reuse, CUDA policy, quantization, or decoder selection automatically.

Parsing must preserve the offline tool's existing-literal-file rule: if the
whole argument names a file, use that path with scale 1.0, even if its name
contains a colon. Otherwise split only the final colon. Paths with spaces and
shell punctuation are ordinary argument strings; never interpret them as shell
commands. Require a readable regular file after resolving supported symlinks.

Scales are locale-independent finite numbers within the FP32 finite magnitude
range. Accept zero, negative, fractional, and greater-than-one scales. Parse
and retain the multiplier as binary64, derive effective scales in binary64, and
convert each effective scale once to FP32 for arithmetic, matching the offline
contract. Reject invalid syntax, NaN, infinity, and overflow. Preserve adapter
order and intentional duplicate adapter arguments; do not merge or sort them.

No-LoRA invocations retain their current behavior and do not inspect, create,
or write the LoRA cache. `--info` with LoRAs validates the arguments and reports
the requested adapters and available modes without folding or creating cache
files; mode-specific mapping remains a generation-time check. Decode-only
invocations reject LoRA flags because decoding an AV state uses no transformer.

## Native API and request lifecycle

Add a native preparation module, provisionally `src/weights/lora.c` / `src/weights/lora.h`, with
separate parser/mapping, arithmetic, and cache helpers where useful. Expose an
additive model-loading API such as `h3_load_dir_with_lora(base, options)` while
keeping `h3_load_dir(base)` equivalent to a request with no adapters. Options
contain an ordered adapter list, cache root, and scratch-memory limit. The
context owns copies of its configuration; caller strings may be released after
loading. Do not put mutable adapter selection into per-step sampler parameters.

The context keeps the base model directory for tokenizer, encoders, VAEs, and
other unchanged assets. Add an effective-transformer resolver per mode. Each
resolved mode holds the verified transformer path, content identity, and cache
lease. It is an override of the transformer component, not a replacement base
directory. This avoids copying large unchanged components and stale cached
symlinks when a base model moves.

The adapter list is fixed for the lifetime of a context. Library users create a
new context to change it. CLI invocations use the supplied flags and prepare
each required mode on its first request. Runtime adapter-editing commands and
simultaneous requests on one context are outside the initial scope.

For each request:

1. Complete CLI/API validation that can run before preparation, including
   backend/quantization checks and incompatible flag combinations. Load and
   validate a supplied continuation or sampler state before expensive folding.
2. Select the actual transformer mode. Ordered references select Ref2VA;
   prompt-only and first/last-frame requests select FL2VA. Sampler resume uses
   the checkpoint's saved mode, not the new command line's empty reference list.
3. Require that mode in the base model. Normalize every adapter against that
   mode and validate all inputs before allocating output shards. Do not fold
   both modes for an ordinary one-shot request.
4. Resolve or build the cached transformer, before loading encoders, reference
   conditioning, inference weights, or preparing quantized projections. Release
   an incompatible prepared DiT before folding another mode in a reused context.
5. Bind the effective transformer to every downstream consumer. Refresh the
   selected transformer's inventory from the resolved artifact and include its
   identity in prepared-DiT cache keys. Shared components still use the base.
6. Perform model fingerprint/resume compatibility checks against the effective
   model, then run normal generation. Release folding scratch buffers before
   inference allocations. Keep the read lease for as long as weights may be
   mapped or reused; release it on context cleanup.

The initial model inventory and device capability query may precede folding;
large inference allocations must not. Preparation reports progress through the
request callback and checks cancellation at bounded hashing, copy, and matrix
block boundaries. Cancellation returns through normal cleanup without starting
generation or damaging a previously completed cache entry.

## Supported adapters and folding semantics

Port the strict supported subset from `fold_lora.py`:

- BF16 H3 base matrices and BF16/F16/F32 A/B tensors, with shapes
  `W[out, in]`, `A[rank, in]`, `B[out, rank]`, and positive rank.
- Native and supported PEFT prefixes, A/B and down/up names, supported
  Diffusers names, and unambiguous ComfyUI flattened names.
- Fused H3 QKV targets and explicit Q/K/V row slices, including token-refiner
  attention. Validate equal H3 Q/K/V widths against available configuration;
  do not infer an orientation or import Qwen's GQA layout.
- Scalar alpha tensors and the offline tool's supported alpha/rank metadata.
  Use `effective_scale = user_scale * alpha / rank`; if alpha is absent,
  explicitly report `alpha = rank`. Conflicting metadata is an error.
- All mapped ordinary transformer targets, including token refiners and
  modulation matrices, rather than a hard-coded list limited to attention.

Require complete pairs, unique non-overlapping targets within each adapter,
valid ranks and dimensions, and finite input values. Multiple adapters may
update the same target. Reject unknown tensors, orphan alpha values, ambiguous
names, non-BF16 base targets, unsupported layouts, invalid shard indexes, and
architectural adapters such as PDD, VDN, DoRA, ControlNet, or unsupported RS-LoRA
semantics. Rejection applies even at scale zero. Never silently ignore part of
an adapter or substitute base generation after a folding error.

Preserve strict safetensors validation: duplicate JSON/tensor keys, dtype and
shape validity, checked integer arithmetic, payload bounds, overlapping or
missing payload ranges, truncation, and exact shard-index correspondence.
Use bounded header/metadata sizes and safe tensor-name handling. Preserve
supported unaligned payloads using byte-safe reads, not aligned pointer casts.

For a target matrix, the arithmetic contract is:

```text
accumulator = FP32(base BF16 values)
for adapter in command-line order:
    delta = FP32(B @ A) for its mapped rows
    delta = FP32(delta * FP32(effective_scale))
    accumulator = FP32(accumulator + delta)
output = BF16_round_to_nearest_even(accumulator)
```

Round to BF16 once after all applicable adapters. Reject non-finite
intermediates and BF16 overflow. Zero updates leave the original weight bytes
unchanged. Preserve untouched tensor bytes, shard headers, offsets, ordering,
file lengths, configuration, and indexes. Never modify source shards or adapters.

Recipe 1 uses a bounded, blocked native CPU FP32 kernel with a specified
ascending rank-reduction order, separate multiply/add operations, and no
fast-math or FP contraction. Vectorization across independent output elements
is allowed. Set and restore the thread's required floating-point environment:
round-to-nearest-even with gradual underflow, including conversion/scaling.
Results must not depend on the caller's rounding/flush mode, scratch budget, or
worker scheduling.
This gives deterministic regeneration on supported hosts without a new mandatory
BLAS dependency. Faster arithmetic implementations may be added under a new
recipe unless they prove byte-identical results. Python/NumPy remains a test
oracle, not a runtime dependency; real-world BLAS reduction differences mean
native/Python parity is numerical rather than a universal byte-equality claim.

The memory planner includes A/B conversion, base accumulators, delta buffers,
I/O, and validation scratch. Tile additional dimensions when a whole A matrix
does not fit. Reject a budget that cannot accommodate the smallest tile before
copying the checkpoint. The budget bounds owned folding allocations; document
file-cache/RSS overhead separately and retain H3's existing memory-floor checks.

Validate FP32 and stored-BF16 results using independent bounded matrix/vector
probes and the offline tool's error-bound methodology. Re-read written bytes,
record output hashes, and verify every unchanged byte range. Validation failure
prevents publication and generation.

## Cache identity and on-disk layout

Use a dedicated cache, separate from the existing CUDA quantization cache:

```text
<lora-cache>/
  locks/<key>.lock
  variants/<key>/
    manifest.json
    transformer/
      config.json
      model.safetensors.index.json       # when present in the source
      *.safetensors
      h3_lora_manifest.json
  staging/<key>.<unique>/
  invalid/<key>.<unique>/                 # quarantined invalid entries
```

All directories used for publication must be on the same filesystem. A key is
SHA-256 over a versioned, domain-separated, unambiguous serialization containing:

1. Cache schema, mapping/scaling version, arithmetic recipe, output dtype and
   rounding policy.
2. Selected mode and the sorted inventory of original transformer files:
   relative name, length, and content SHA-256, including config, shard indexes,
   and any existing fold manifest.
3. Every adapter's file-content SHA-256 and canonical scale in its original
   argument order. Encode binary64 scale bits explicitly in a fixed byte order;
   normalize negative zero to positive zero. Equivalent decimal spellings such
   as `1` and `1.0` have the same identity.

Use explicit lengths and byte encodings; never hash padded C structs, locale
dependent strings, or unordered JSON. Paths, mtimes, creation timestamps,
scratch size, prompt, seed, resolution, step count, CUDA execution policy,
quantization mode, and decoder choice do not identify the folded transformer.
Different adapter orders are distinct even where their ideal real-number sums
are equal. Record resolved mappings/effective scales in the manifest and verify
them against the recipe and inputs.

Base transformer identity covers the original bytes supplied with `-d`.
An already folded base is allowed: new flags apply additional updates to those
bytes, and its parent manifest is retained as provenance. No existing adapter
is subtracted or applied implicitly. Report that a folded base was selected.

The manifest records schema/recipe, key, mode, ordered adapter digests/scales,
source file hashes, mapping and scale decisions, output inventory/hashes,
validation results, and copy method. Human-readable source paths and timings
belong in `manifest.json` outside the effective model tree. The generated
`transformer/h3_lora_manifest.json` contains only deterministic identity fields,
including the parent manifest identity where applicable. It must not include
timestamps, absolute paths, timings, copy methods, or memory-dependent metrics.
These would make identical rebuilt variants fail sampler fingerprint checks.

Moving an unchanged base or adapter to another path reuses the same transformer
entry. The current base supplies shared assets at runtime; changing a VAE or
text encoder changes full-model compatibility without unnecessarily refolding
unchanged transformer bytes. Base files and adapters must still be present and
validated on every new invocation; the cache is not a standalone model export.

## Lookup, publication, and recovery

1. Open and validate the inputs, hash their contents, derive the key, and retain
   file identity/change guards. A stat stamp alone is not a persistent content
   identity. Reuse of verified hashes within one context must detect changed
   files before later requests.
2. Acquire a shared lock for an existing entry. A hit requires the expected
   schema/key/mode, complete inventory, valid metadata, and matching output
   content hashes. Revalidate source identity while resolving the request.
   Directory existence alone is never a hit.
3. On a miss or corruption, release the read lock and acquire the exclusive
   per-key lock, then recheck. If another process completed the entry, verify
   and reuse it. Independent keys may prepare concurrently. Lock waits must
   support cancellation and report that preparation is waiting.
4. With the exclusive lock, quarantine an invalid final entry and create a
   unique staging directory. Clone/copy the original transformer there, fold
   all mapped updates, and validate the complete result. Check for source
   replacement or mutation and rehash the sources before publication.
5. Flush output files and deterministic identity metadata, write the completed
   outer manifest last, fsync the staging directory, and atomically rename it
   into the final entry. Fsync affected parent directories. Generation starts
   only after publication and validation succeed. Downgrade to a shared lease
   before making the entry available to the context.

A cache hit must perform no fold writes. It may work with a read-only cache
when the existing entry and persistent lock file can be read and shared-locked.
A miss in such a cache fails with an actionable error. Report hashing/verification
time separately from folding time; a warm hit still reads substantial data and
does not promise constant-time startup.

Cache entries are immutable after publication. Never patch a completed entry
in place. A shared lease prevents another H3 process from replacing it while
its weights are in use. Keep lock files at stable paths and do not unlink them
after unlock. Test the shared-to-exclusive recheck and exclusive-to-shared
transition. If the platform cannot convert a lock atomically, reacquire the
shared lease and revalidate the final entry before opening it for model use;
never assume a previous validation survived an unlocked interval.

Try APFS cloning on macOS and reflink cloning on supporting Linux filesystems.
If unavailable, copy automatically after a conservative disk-space check and
report the expected cost. Modified shards must always be independent of the
source; never open a hard link or symlink to source weights for writing. Own all
transformer files in the entry so later source writes cannot mutate a hit.
Budget for worst-case modified CoW pages, staging, and any quarantined entry.
When logical-size quotas charge successful clones and leave insufficient
reported CoW headroom, replace modified unpublished clones with checked full
copies one shard at a time. Removing that staged clone releases its quota before
the independent copy; original inputs are never removed.
Disk-space preflight is advisory; short writes and ENOSPC must still cleanly
abort publication.

Restrict cache-created permissions to the user, validate manifest-relative
paths, reject traversal and unexpected symlinks inside entries, and refuse cache
locations that overlap input trees/files in ways that could overwrite inputs.
Cleanup may remove only owned staging paths after obtaining the relevant key's
exclusive lock. A killed writer's staging directory is never treated as a hit.
Do not steal a live lock based on a PID file or a timeout.

Corrupt entries are rebuilt once under the lock if sources and storage permit;
otherwise fail without producing media. Retain quarantined data for explicit
cleanup and report its location. Automatic eviction, a cache daemon, network
cache sharing, and cache management commands are outside v1. Document manual
pruning while no H3 process is using the cache and the disk cost of many strengths.

## Effective model identity, continuation, and resume

Every operation that depends on transformer content must use the resolver:

| Consumer | Required behavior |
| --- | --- |
| DiT loading, streaming, and in-memory reuse | Load the selected cached transformer; key reuse on effective identity and existing execution settings. |
| Model inventory and generation diagnostics | Report base/shared assets and the selected resolved transformer distinctly. |
| Sampler capture and compatibility | Fingerprint the effective model's actual files using original logical relative names. |
| Quantization preparation and delivery provenance | Read/hash folded BF16 values and report the effective model fingerprint. |
| Tokenizer, encoders, VAEs, and AV signature | Resolve unchanged components from the current base model. |

Extend fingerprint traversal to substitute the transformer's physical root
while preserving its logical `transformer/...` names. Keep the existing mode
domain, ordering, and byte-hashing behavior for no-LoRA and legacy offline model
trees. Metadata outside the logical tree is not fingerprinted. Never use only
the cache key or claimed manifest hashes as proof of the effective model bytes.
Runtime effective-model traversal reads content afresh, with optional CPU SHA
acceleration, because identical file timestamps can survive rapid same-size
writes on coarse filesystems.

Sampler resume initially requires the user to repeat the original LoRA flags,
or supply the exact effective model through an existing compatible model path.
Treat LoRA flags as model selection analogous to `-d`, separate from the CLI's
ban on new generation settings. Read the checkpoint to determine mode; resolve
the folded variant before `h3_sampler_state_compatible` checks the model.
The checkpoint remains authoritative for all saved sampling parameters.

```sh
./bin/h3cli -d ./models/MiniMax-H3 \
  --lora ./lora/downloads/minimax_h3_turbo_v4_step600_ema.safetensors:1.0 \
  --resume-sampler-state ./outputs/paused.h3sample \
  -o ./outputs/resumed.mp4
```

With identical inputs and the same folding recipe, cache eviction followed by
regeneration must reproduce the same effective fingerprint and pass the
existing resume checks. A wrong base, changed adapter/scale/order, or different
resulting bytes fails the ordinary exact-model check; never resume with base
weights as a fallback. Missing flags do not cause automatic adapter discovery
or download. Log the original selection/key when saving a checkpoint so the
invocation can be reproduced. No sampler binary-format change is required for
this first version; automatic recipe restoration can be a later extension.

Python-folded and runtime-folded model trees may have different manifests or
rounding results. Numerical parity does not grant cross-model sampler resume;
the existing exact fingerprint remains the authority. Recipe/build compatibility
rules also remain in force when a newer executable opens an older checkpoint.

Complete `.h3av` continuation retains the existing VAE/latent compatibility
contract. It can change LoRA selection between segments where those checks
permit it. Do not impose sampler-style transformer equality on AV continuation.
Save optional LoRA provenance as informational metadata without changing that
acceptance rule. Hard and bridge continuation must both be covered.

## CUDA quantization and preview integration

Always fold original BF16 weights first. Then feed the resulting BF16 tensors
to the existing FP8/NVFP4 preparation path and its independent cache. Never add
LoRA deltas to packed or dequantized quantization artifacts. The packed cache
already keys on source tensor content; prove that changed folded content cannot
reuse base-model packed weights, while truly unchanged tensors may share them.

`--fast-cuda`, `--cuda-denoise-quant`, and `--preview-vae` retain their existing
validation, device support, and independent meanings. LoRA selection does not
enable any of them. Switching these options need not refold a BF16 transformer.
Unsupported quantization requests should fail during capability checks before
expensive preparation. Quantization failure does not invalidate a fully
published LoRA variant and must not silently select another precision.

## Diagnostics and failure behavior

Use existing progress conventions to report selected mode, adapter count/order,
effective scales, cache root/key, hit/miss/repair, lock waits, copy strategy,
current tensor/block, bytes processed, and validation/publication completion.
Summarize preparation, cache verification, and inference times separately. Avoid
per-element logging and include enough identity information to reproduce a run.

On invalid inputs, mapping errors, cancellation, memory/disk exhaustion, source
mutation, validation failure, or publication failure, return an actionable error
before generation. Preserve sources and valid entries, release locks/buffers,
and clean only this attempt's staging files. A cache repair may fail, but H3 must
never silently omit the requested adapters. Default no-LoRA execution and the
offline tools must continue to work without any cache-related dependency.

## Validation and acceptance

Host tests should use small generated safetensors fixtures and temporary cache
roots, with no model download or GPU requirement for parsing, folding, and
cache lifecycle checks. Keep the Python implementation independent as an oracle.

| Area | Required evidence |
| --- | --- |
| Parsing and mapping | Literal-colon/spaced paths, valid/invalid scales, duplicates/order, all supported conventions, QKV slices, alpha/rank conflicts, malformed files and unsupported architectures. |
| Math | Independent FP32 and BF16 references; zero/negative/multiple scales; one-rounding counterexample; finite/overflow checks; invariant bytes across tile sizes and repeated native builds under the same recipe. |
| Native/offline parity | Tiny exact fixtures plus representative real matrices compared numerically using existing error bounds; exact untouched bytes. |
| Cache identity | Hits survive path moves and equivalent scale spellings; changes to any source content, scale, order, mode, or recipe produce the appropriate miss. |
| Cache integrity | Missing/truncated/changed output, invalid manifest/schema, source mutation with preserved mtime, path traversal, and symlink attempts cannot produce false hits. |
| Publication and concurrency | Same-key writers converge to one verified entry, different keys proceed independently, reader leases protect mapped files, cancellation/crash/fault injection never exposes a partial variant. |
| Resource handling | Enforced scratch budget, bounded RSS accounting, clone and full-copy paths, disk-full/permission errors, read-only hits, cleanup and descriptor/lock leak checks. |
| Mode and API lifecycle | Standalone FL2VA/Ref2VA models, both-mode models, missing selected mode, mode changes between library calls, and immutable context configuration. |
| State compatibility | Cold/warm/rebuilt runtime variants resume identically under existing deterministic settings; changed effective weights reject; no-LoRA/legacy fingerprints remain compatible; hard/bridge AV continuation preserves its rules. |
| Inference composition | Metal BF16 and supported CUDA BF16/fast/FP8/NVFP4, full/preview VAE, and source-content-based packed cache separation. |

Measure cold hash/copy/fold/validation/publication time, warm verification time,
scratch/RSS, disk use, and subsequent inference separately. Cover Turbo at
strength 1.0 and eight steps, an ordinary adapter at its normal step count, and
a multi-adapter request. Use the same inputs/settings in runtime/offline
comparisons and state numerical limits explicitly. Include a 1344x768 CUDA
preview smoke run when suitable hardware is available; keep small repeatable
fixtures as the correctness gates.

Acceptance requires passing relevant existing LoRA, model-loading, sampler,
continuation, preview, and quantization checks in addition to the new suite.
Report unavailable hardware cases as unvalidated. Documentation and task
completion must distinguish functional correctness from any human-assessed
quality or performance claim.
