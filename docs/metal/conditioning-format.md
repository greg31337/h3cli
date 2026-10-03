# Conditioning cache and M0/M1 benchmark contract

Current container schema is 2 on both Metal and CUDA. Older files are rejected.
Model/recipe identity uses the current metadata contract; see
[all current state formats](../features/current-state-contract.md).

`--save-conditioning PATH` atomically creates an `.h3cond` file after conditioning
and AdaLN preparation. `--load-conditioning PATH` validates identity before loading
encoders and imports reusable conditioning. Both can be specified to refresh a
cache. Normal generation does not use disk conditioning unless requested.

The identity includes exact prompt bytes, an empty negative-prompt hash (there is
no negative-prompt API), ordered reference content hashes and kinds, first/last
frame hashes, output and render geometry, aligned frame count, reference sizing,
FL2VA/Ref2VA, audio conventions, model/component signatures, the AV compatibility
signature, the Ref2VA pipeline recipe, and conditioning arithmetic environment.
Model signatures reuse the checkpoint implementation, including effective folded
LoRA transformer identity. References are identified by contents rather than path.
The encoder platform/device/architecture are recorded from checkpoint device
metadata, so copying a cache to another GPU cannot silently change the encoder
arithmetic contract. Main DiT backend selection remains independent.
A mismatch names the first differing field and requests regeneration. Files are
checksummed, not authenticated: use caches from trusted sources.
Explicit disk-cache requests also include this identity in the live context's
conditioning and prepared-DiT keys. A reference content change therefore
invalidates those objects even if path, byte length and mtime were preserved.

Seed is deliberately absent: reference augmentation and latent initialization
run again for the requested seed. Steps are absent from invariant identity.
`--conditioning-schedule` saves optional AdaLN records and, on load, requires the
complete matching video/audio sigma arrays, reference topology, continuation
prefix and bridge profile. Without that option, a schedule miss drops scheduled
records and rebuilds them while retaining invariant conditioning. This is not a
fallback for an identity mismatch, which always fails.
Scheduled caches require all 50 transformer blocks when saving. Pruned-layer
runs may load a complete cache, then prune the imported schedule normally.

## Lifetime and dependency map

| Data | Lifetime / dependency | Cache treatment |
| --- | --- | --- |
| Token IDs, presentation positions/spans, text modality tags | Prompt and ordered references | Persist diagnostic metadata and final BF16 embeddings |
| Vision patch/deep-stack outputs and text-encoder KV scratch | Used while producing the final multimodal Qwen embedding | Release; their effect is already present in persisted embedding |
| Raw reference VideoVAE/image/Ref2VA rows | Reference bytes, geometry and encoder recipe | Persist F32 rows before seeded augmentation |
| Raw reference AudioVAE rows and stereo ordering | Reference bytes and audio conventions | Persist F32 rows before seeded augmentation |
| Ordered reference layout descriptors and first/last anchors | Reference ordering and geometry | Persist scalar descriptors; rebuild positions deterministically |
| Refined text consumed by DiT | Encoded text, transformer/refiner weights and arithmetic recipe | Persist BF16 output independent of denoising steps |
| Timestep embeddings and block/final AdaLN | Full video/audio schedule, conditioning classes, prefix/bridge | Optional schedule-keyed BF16 records |
| Seeded reference augmentation, random target latents | Seed and raw conditioning | Recompute after loading raw cache |
| Continuation source latents / bridge insertion | Requested AV source and seed | Keep in AV state; never import from conditioning cache |
| Per-step hidden state, Q/K/V, attention scratch, velocities | Current evaluation / block | Never persist in `.h3cond` |
| Model weights and backend pipeline objects | Context/model lifetime | Existing model/prepared cache, not `.h3cond` |

Refiners and encoder outputs retain existing arithmetic. Planned M1 backend
selection affects main DiT attention only and is therefore not part of
invariant conditioning identity. Checkpoint section 36 separately records the
nondefault Metal backend and its attention implementation version, is required
when present, and changes prepared checkpoint identity. Current containers
serialize the active policy explicitly. A conflicting explicit resume selection
fails; an omitted selection restores the checkpoint policy.
The linked [M0/M1 results](m0-m1-results.md) describe an earlier prototype;
current Metal options and guards are documented in the main README.

## Version 2 binary representation

All scalars are little endian; payload floats are IEEE F32 or BF16. Other host
endianness is rejected. The 128-byte header contains `H3COND\0\0` at 0, version
`u32=2` at 8, header size at 12, total `u64` byte length at 16, record count at 24,
entry size `u32=96` at 28, SHA-256 of the table at 32, SHA-256 of header bytes
0–63 at 64, and zero reserved bytes 96–127.

Each 96-byte entry contains ID, dtype, rank and required flag (four u32 fields),
three u64 shape axes at 16, u64 payload offset/length at 40/48, payload SHA-256 at
56, and eight zero reserved bytes. Payloads follow the table contiguously in
entry order. Unknown IDs/types, duplicates, incorrect shapes, gaps, overlaps,
truncation, trailing bytes, invalid reference geometry/tags, and nonfinite
conditioning tensors fail. File size is bounded to 8 GiB, record count to 80,
identity to 1 MiB, references to 4096. Checksums cover every payload before use.

Dtypes: U8=1, BF16=2, U32=3, U64=4, F32=5. Unused dimensions are zero.

| ID | Payload / shape |
| --- | --- |
| 1 | Canonical UTF-8 identity, U8 bytes without NUL |
| 2 | Raw Qwen BF16 `[tokens,5120]` |
| 3 | U8 modality tags `[tokens]` |
| 4, 5 | Raw F32 video/audio conditioning `[elements]` |
| 6 | U32 `[references,5]`: kind, video T/H/W, audio T |
| 7 | U32 `[4]`: conditioned flag, keyframe count, first/last indices |
| 8 | U32 token IDs `[tokens]` |
| 9 | U32 presentation positions `[3,tokens]` |
| 10 | U64 presentation spans `[count,2]` |
| 11 | U8 full-schedule key `[32]` |
| 1001 | Refined BF16 text, flattened `[tokens*5376]` |
| 1002 | Final AdaLN BF16, flattened; exact logical extent checked by schedule importer |
| 1100–1149 | Block AdaLN BF16, flattened; exact logical extent checked by schedule importer |

IDs 1, 2 and 7 are mandatory; the rest are conditional on the workload. Scheduled
records require ID 11. AdaLN records use the existing versioned prepared-tensor
import/export contract. Writes use a private temporary file, flush/fsync, and
atomic rename, so an interrupted writer cannot expose a partial destination.

## Reproducible evaluation limits

`scripts/metal_native_bench.py` runs B1=1, B2=2, B5=5 or B6=6 evaluations with all
50 blocks, dense attention, BF16 weights, reuse=1 and CPU Euler. `--blocks 1|5`
exists only for explicit B1 diagnostics; those outputs are not quality samples.
`--components` fences individual GPU components and is also B1-only. Compare
ordinary B5 runs without those fences. `--trace` captures a B1 Metal System Trace
using scoped `DEVELOPER_DIR`; it does not change the global Xcode selection.
The trace attaches during denoising for a bounded window (20 seconds by default,
`--trace-window 1..60`), while the complete B1 run continues. Launching the entire
model through Instruments caused excessive memory pressure on the validation Mac.

The runner and Makefile test recipes set `H3_TEST_MAX_EVALUATIONS=6`. The library
checks fresh and resumed denoising bounds and rejects longer test runs. Host
fixtures may describe longer schedules without executing them. New tests must
use the runner or inherit that environment; manually launching historical test
commands without it is outside this benchmark contract. Normal user step counts
and default VAE remain unchanged. Preview VAE is only the runner default.

Each run retains exact command, environment, binary/source hashes, OS/compiler,
Xcode, device/model path, conditioning/media/AV hashes, evaluation counters,
per-step timings and memory observations. B5 steady statistics use steps 2–5;
B6 uses 2–6, with step 1 reported separately. Profile fences measure completed
work, and component-fenced/trace runs must be labeled as diagnostics. Memory
includes process resident/physical footprint/compressed bytes, system compressor
and swap, tracked tensor bytes and the Metal device's current allocation count.

For the recorded M4 Max baseline, `H3_DIT_COMMAND_BLOCKS=5` bounds retained MPS
temporaries using the existing command-buffer split override. Production B1 was
byte-identical to the default split; the selected policy is part of cache and
benchmark identity and must match on both sides of a performance comparison.
Ordinary user defaults are unchanged. A fresh repeatable suite can be launched as:

```sh
H3_DIT_COMMAND_BLOCKS=5 python3 scripts/metal_native_suite.py \
  --output outputs/metal-native/repeat --cases B1 B2 B5 B6
```

The model-backed long-lived-context regression uses three preparation-only
requests and zero denoising evaluations:

```sh
make bin/conditioning_context
./bin/conditioning_context models/MiniMax-H3 outputs/metal-native/context-regression-new
```
