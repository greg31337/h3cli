# Current state contracts after legacy cleanup

Implementation contract, 2026-09-28. Old h3cli saved files are intentionally
unsupported. Reject their schema before interpreting payloads; there is no
production migration route. Model safetensors and supported external inputs
are unaffected.

| Container | Current schema | Required interpretation |
| --- | --- | --- |
| AV `.h3av` | 3 | Explicit ordinary/upscale geometry profile; normalized video/audio tensors, seed, model/VAE identity, checksum. |
| `.presentation` | 9 | All delivery, recipe, precision, attention, adaptive threshold/hit ceiling, warmup, geometry and upscale fields serialized explicitly; canonical text bound to AV fingerprint. |
| Sampler `.h3sample` | 2 | Current sections only; adaptive section 40 v3 stores effective threshold/hit ceiling and budget, sections 41/42 retain BF16 history; explicit warmups and execution/device identity; no fast or historical Ref2VA selector. |
| Conditioning `.h3cond` | 2 | Current prompt/reference/model/recipe identity and optional schedule tensors; no retired policy selectors. |
| Upscale `.h3up` | 2 envelope | Current source or refinement record with explicit stage; existing stage-specific mathematical recipes remain distinct. |

Both CUDA and Metal writers emit these contracts. Backend math and ordinary
versus upscaled geometry remain explicit active profiles, not backward format
versions. CUDA video chooses SGLang arithmetic internally; saved execution
identity checks detect mismatches but cannot select an older renderer.

The [extended CUDA canvas contract](../cuda/large-canvas-contract.md) adds
profile 2 for native video through 1920×1088 and profile 3 for learned 2×
output through 3840×2176, including transposes. Legacy profiles 0/1 retain
their meanings. Required sampler/source section 46 records the new identities;
old readers reject them. Metal rejects profiles 2/3 before tensor allocation.
Profile 3 requires upscale provenance; profile 2 cannot masquerade as refinement.

Presentation upscale recipe 3 records learned 2× transfer from a native AV
state, with zero refinement and the parent AV fingerprint. It preserves the
source's coupled video/audio prefix trim, including hard/bridge continuation
delivery. Existing `.h3up` recipes 1/2 remain untrimmed. This adds no container
fields and does not change ordinary saved-state bytes; readers without recipe 3
support reject it. See [AV upscaling](latent-upscaling.md).

CUDA duration through 1,450 frames is independent of canvas profile. AV schema 3
expresses it directly; sampler/source envelope 2 uses required section 15 v2
for 363–736 frames and v3 for 737–1,450. Short section 15 v1 remains
byte-compatible. Native long NVFP4 records execution recipe 5 while retaining
packed-weight recipe 2. Metal rejects
long execution even for profile 0/1. See the
[duration contract](../cuda/long-video-1450-contract.md).

Native long Sage3 retains attention recipe 1 / plan 1 and composes with BF16 or
NVFP4 projection recipe 5. Early sampler readers validate both policies before
payload allocation; presentation and prepared-context keys preserve them.
Resume rejects conflicting attention overrides. Completed AV replay only needs
decoder assets. Sage3 `.h3up`, AV transfer and upscale remain excluded; see the
[Sage3 contract](../cuda/sage3-long-contract.md).

Require matching current files for decode, continuation, bridge, conditioning
reuse, checkpoint resume, and upscale. Retain bounds, finite-value, checksum,
canonical encoding, model/device/build provenance, and output-collision checks.
Current same-build save/load and pause/resume must preserve latent content.

The recorded SGLang fixture `tests/fixtures/cuda-reference/final.h3av` remains
byte-for-byte unchanged. A bounded test-only loader reads its known original
schema for numerical probes; it is not linked into the CLI or library.

Adaptive section 40 v1/v2 and presentation 8 are unsupported. Threshold zero is
an explicit saved value, never an omitted default. Resume omission restores
controls; explicit overrides must match the saved FP32 threshold and integer
ceiling. Reference recipe 3 requires ordered media-kind/audio provenance and
reconstructed layout/RoPE validation before cache allocation; source media are
not reopened. Text arithmetic recipes 1/2 and all other envelope versions remain
unchanged. Disabled presentation controls are zero.

BF16 continuation and bridge use adaptive mathematical recipe 4 with the same
section-40 v3, sections 41/42, sampler envelope 2 and presentation 9. Readers
reconstruct generated suffixes and active bridge classes from checked saved
geometry, prefix and bridge metadata before allocating cache payloads. Forged
mode/context, mismatched class tables, anchors, upscale and quantized recipe 4
are rejected. Resume restores ready/streak/phase/step and both cache tensors
without reopening source AV or reference media. CUDA uses CPU-state F32 Euler;
GPU-state adaptive checkpoints remain unsupported.

Presentation recipe 4 records continuation provenance even when debug delivery
keeps the prefix and trimming is zero; presentation has no separate continuation
flag. A complete AV state carries no adaptive history into the next segment.
It can feed an ordinary or approximate hard/bridge request with compatible
geometry and latent space. `--keep-continuation-prefix` changes delivery only.
