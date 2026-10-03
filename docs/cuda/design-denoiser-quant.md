# CUDA denoiser quantization

Implementation contract: [T001–T064](../todo.md). Starting source revision:
`f4893c70a1c86ebe2c8f801a64ea72091d6ddd4f`, also recorded in the validation
environment manifest. Target: RTX 5090 / SM120;
native cuBLASLt FP8 E4M3 and NVFP4 E2M1, BF16 projection outputs and FP32
accumulation. The policy is off by default, scoped to the four repeated DiT
projection families, and independent of fast CUDA and VAE selection.

Each packed weight owns its data and scales. The existing BF16 projection
interfaces dispatch by this explicit weight descriptor, so refiners, encoders,
heads, attention and VAEs continue using their original weights and kernels.
Preparation reads one original BF16 matrix at a time. The default metadata cache
avoids weight hashing during both preparation and loading. Artifacts include
recipe, dimensions and scale layout; source files are immutable while in use.
Folded transformers have distinct source metadata identities. The optional
strict cache retains content addressing and checksums.

Activation quantization, scales, optional streaming storage and GEMM plans
belong to the GPU context. No mutable process-global precision mode controls
existing contexts. Admission uses compressed weights plus activation/scratch
reserve, and the compressed streaming fallback preserves compute/copy ordering.

FP8 starts with tensorwide dynamic activation scaling and offline weight
scaling. NVFP4 uses 16-value UE4M3 block scales in cuBLASLt's tiled layout and
a tensorwide factor. Scale direction and supported arithmetic are verified
with known-value native matrix tests before real-weight generation. Sensitive
matrix exceptions or rotations become explicit versioned recipes, selected
from the bounded calibration corpus and independently reviewed on held-out
video/audio outputs.

CLI: `--cuda-denoise-quant off|fp8|nvfp4` and
`--cuda-denoise-quant-cache DIR`. API fields mirror these selections. A sampler
checkpoint records the mode and recipe version; ordinary resume restores it
and rejects incompatible explicit changes. Completed AV states retain their
latent arrays and carry denoising provenance in presentation metadata. Decode
replay needs no quantization artifacts and cannot undo quantized denoising.

Validation follows the short presets, separate format acceptance and shared
experiment ledger in the checklist. The initial round had a 60-minute cumulative
allowance. On September 19 the user authorized an unlimited cumulative follow-up;
`quant_run.py --unlimited` records that extension without resetting prior costs.
The 120/300/600-second per-job caps still apply. Kernel format/memory safety
checks are distinct from perceptual acceptance; BF16 trajectory equivalence
is not a quantized quality criterion. Quantized attention, distillation and
other GPU/Metal qualification are outside this implementation.

Kernel contracts follow [NVIDIA's cuBLAS narrow-precision documentation](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html#narrow-precision-data-types-usage).
No vendor sample implementation is copied into the project.

## Integration and artifact contract

`bin/quant_projection()` is the recipe's per-matrix selector. `src/denoise/dit.c` uses
it only while loading `blocks.N.attn.{qkv_proj,out_proj}.weight` and
`blocks.N.mlp.{fc1,fc2}.weight`; the existing text refiner and AdaLN loaders
stay on their original routes. `h3_gpu_linear_bf16()` dispatches by the
weight descriptor, not a process-wide precision flag. The original BF16
stream loader and fused BF16 MLP buffers are bypassed for a quantized core.
Request-local policy is scoped through thread-local save/restore; packed
resources and plans belong to the prepared DiT's GPU context. Mode/recipe
enter its cache key without changing conditioning or decoder cache keys.
Metal retains its existing INT8 implementation and rejects these CUDA options.

The arithmetic remains recipe 1. Normal loading uses a format-2 metadata key:
canonical source path, device/inode, file size, nanosecond mtime and ctime, tensor
offset, dimensions, precision and recipe. SHA-256 compresses this small metadata
record into a filename; no weight bytes enter it. The artifact itself supplies
the versioned manifest/header, so no source scan is necessary to discover it.
Source identities distinguish base/folded models and model variants. Relocation,
replacement or edits invalidate entries; byte-identical copies do not share
metadata entries. Publication uses the existing lock/staging/fsync/rename policy.

`--cuda-denoise-quant-verify` / `H3_QUANT_VERIFY=1` selects the legacy strict cache,
whose key contains the original tensor's content digest and whose payload is
checksummed on every load. This option is diagnostic, not an arithmetic policy;
changing it invalidates a live prepared DiT but not sampler arithmetic identity.
Strict and metadata entries coexist without fallback between them. Old strict
entries cannot be associated with source matrices without hashing: the first
metadata load therefore repacks once, without deleting old entries.

Recipe 1 has no rotations, corrections or projection exceptions. It uses
row-major source/packed weights, E4M3 scalar dequantization factors for FP8,
and E2M1 nibbles plus positive E4M3 VEC16 block scales for NVFP4. Native
matmul treats the weight as a transposed column-major operand. Activation
rows are padded to 128 and cropped after GEMM. NVFP4 block-scale indices
follow the documented 128-row × four-block tiling. The descriptor records
mode, N/K dimensions, scale/global offsets, resident versus host-owned
packed storage and its owning GPU; rotation/correction are absent in this
version. Any change to these semantics or the selector requires a recipe bump.

The default metadata header is 128 bytes, little endian: magic `H3QMT002` at 0,
cache format and arithmetic recipe uint32 values at 8/12, mode/N/K at 16/20/24,
zero at 28, payload size/scale offset/global offset uint64 values at 32/40/48,
metadata key at 56 (32 bytes), source tensor offset at 88, and zero reserved bytes
at 96–127. The filename is `m2-MODE-NxK-METADATAKEY.h3q`. Readers reconstruct and
compare the entire header, require exact file length, check source stability,
and retain scale/nonfinite validation. Same-size payload corruption can escape
these checks; local cache files are trusted by this mode. No payload checksum
is generated or checked, including on cache creation.

The strict artifact header is 128 bytes, little endian: magic `H3QWT001` at 0,
recipe/mode/N/K as four uint32 values at 8, payload size/scale offset/global
offset as three uint64 values at 24, source SHA-256 at 48, payload SHA-256 at
80, and 16 reserved zero bytes at 112. Payload length and all offsets are
recomputed from the requested shape before reading. Physical layouts and
converter semantics are versioned together for this first native backend.
The source hash, recipe, mode and dimensions also form the filename.

Sampler files use required extension 34 only for non-off mode. Legacy off
files and prepared keys remain unchanged. Presentation version 2 records
mode/recipe/model identity, bound to the complete AV-state fingerprint;
version 1 remains supported. Normal mode records a metadata-derived model
identity and the trailing `quant_identity metadata-v1` tag, avoiding a second
whole-model content scan solely for provenance. The new reader supports old
untagged content identities; old readers reject the new trailing field. Strict
mode emits the original content identity/representation. Sampler, conditioning,
LoRA and VAE compatibility checks keep their separate contracts. Decode-only
replay never consults packed weights.
