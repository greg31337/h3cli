> Historical kernel/qualification guide. M3B/M4 remove `--fast-cuda` and use
> [one CUDA pipeline with explicit attention/precision options](design-single-pipeline.md).
> Commands and old acceptance statements below describe their recorded recipes.
> Use the current design for supported commands and the
> [single-pipeline report](single-pipeline-results.md) for new comparisons.
> Explicit approximate options may degrade quality; default reference parity remains frozen.
> The later [adaptive-cache/quantization experiment](adaptive-quant-experiment.md)
> defines conservative mixed-precision execution recipe 3 on RTX PRO 5000,
> including its BF16 first-block exception; the historical recipe below is unchanged.


# RTX 5090 denoiser quantization for previews

`--cuda-denoise-quant fp8` and `--cuda-denoise-quant nvfp4` select native
cuBLASLt weight-and-activation quantization for the repeated DiT projections.
The default is `off`. These preview options change both video and audio.
Both formats received separate human acceptance for the six matched base-model
cases on September 19, 2026. The user also accepted the separate 20-step
hard/bridge continuation and eight-step Turbo strength-1.0
[gallery](../../outputs/quant-5090/followup/review.html). Other GPUs
are not qualified. See the
[validation report](denoiser-quantization-validation.md) and
[implementation contract](design-denoiser-quant.md).

## Build and run

Use the normal Ubuntu/CUDA build from the README, with CUDA 12.8 Update 1 or
newer and **cuBLASLt 12.8.4.1 or newer**. Checking only “CUDA 12.8” is insufficient
for GeForce NVFP4. The tested node uses nvcc 12.8.93, cuBLASLt version query
120804, driver 595.91.07 and an RTX 5090 with 32 GB. Runtime selection currently
requires SM120. Other GPU families and Metal are rejected, and other SM120
cards have not been qualified. cuDNN is optional and is used for attention,
not for quantized projections. No additional inference framework is required.

```sh
make -j8 CUDA_ARCH=120
./bin/h3cli -d /path/to/models/MiniMax-H3 --fast-cuda \
  --cuda-denoise-quant fp8 --cuda-denoise-quant-cache /path/to/h3-packed \
  --ref-image inputs/2.jpg -p 'The person turns toward the camera and smiles.' \
  --width 288 --height 384 --frames 56 --steps 20 --reuse 1 --core-reuse 1 \
  --preview-vae -o outputs/preview.mp4
```

Change `fp8` to `nvfp4` for the four-bit candidate. `--fast-cuda` remains an
independent option; omitting it retains default CUDA execution outside the
selected projections. `--preview-vae` is also independent and requires the
existing TAEH3 model. Explicit `H3_CUDA_REFERENCE=1` conflicts with non-off
quantization. `--ssd-streaming` or `H3_CUDA_WEIGHT_MODE=stream` can force packed
streaming for diagnostics. Automatic admission also falls back to packed
streaming if the compressed core cannot fit; an allocation failure during
loading permits one retry. Streaming stores compressed weights in host RAM
and uploads them on the compute stream without expanding them to BF16.

Repeated library requests can retain the prepared DiT, which avoids substantial
fresh-process verification/loading costs. Very short one-shot clips may finish
more slowly even though denoising itself is substantially faster.

API users set `h3_params.cuda_denoise_quant` to `H3_QUANT_OFF`, `H3_QUANT_FP8`
or `H3_QUANT_NVFP4` from `src/weights/quant.h`, and optionally set the borrowed
`cuda_denoise_quant_cache` path. Use `h3_cache_set_enabled(ctx,1)` for repeated
requests. Mode/recipe changes invalidate prepared DiT resources; conditioning
and decoder policies keep their existing identities. The context is used
sequentially, as in the existing API. A context's model directory is immutable:
load a new context for a different folded adapter or strength. Clear/free the
previous context's caches before loading another resident core on a 32-GB card.
Ref2VA/FL2VA selection can change between requests in the same context when both
model variants are available.

## Preparation and identity

The default cache is `$HOME/.cache/h3/denoise-quant`. Use a local SSD. Preparation
is lazy, one original BF16 matrix at a time; the original model is preserved.
Normal FP8/NVFP4 preparation and loading **do not hash weight payloads**. Format-2
entries use a small metadata identity: canonical source path, device/inode, file
size, nanosecond mtime/ctime, tensor offset/shape, format and recipe. Each load
checks the version, complete header, exact artifact length, and scale validity.
The source is checked again before accepting or publishing the packed data.
These checks trust local files; silent data corruption that leaves metadata
unchanged is outside this policy.

`--cuda-denoise-quant-verify` (or `H3_QUANT_VERIFY=1` for the API/preparation helper)
selects the original strict content-addressed cache and hashes source and packed
bytes. It is an optional diagnostic, and does not change quantization arithmetic.
Strict and metadata artifacts occupy separate namespaces in the same directory.
There is no hidden strict fallback during ordinary loading.

**Upgrade/move behavior:** old `v1-*.h3q` files contain only content hashes, not
the source-file mapping needed for metadata lookup. They remain available to
strict mode. The first normal run prepares new `m2-*.h3q` entries without hashing;
subsequent runs reuse them. Moving/copying a model, editing its files, or using a
different folded transformer invalidates the relevant metadata keys. Existing
artifacts are never silently matched by shape alone or automatically deleted.

An explicit helper prepares all 200 projection matrices without loading
encoders or generating media:

```sh
make bin/quant_prepare CUDA_ARCH=120
./bin/quant_prepare fp8 /path/to/models/MiniMax-H3/Ref2VA/transformer /path/to/h3-packed
./bin/quant_prepare nvfp4 /path/to/models/MiniMax-H3/Ref2VA/transformer /path/to/h3-packed
```

Repeat for FL2VA when needed. A complete core requires approximately **17.94
GiB FP8** or **10.09 GiB NVFP4**, plus small headers and lock files. Strict mode
can share byte-identical matrices across model variants; metadata mode keeps
distinct source-file identities. Source mutation during preparation is rejected. Publication
uses an advisory writer lock, temporary file, fsync and atomic rename. A killed
preparation may leave a `.tmp.*` file, which is never treated as a valid cache.

Corrupt artifacts fail with their path; remove that artifact and prepare again.
Old recipe files can be removed when no process uses them. Do not remove or
replace original model files: warm cache loading still verifies them.
Only explicit strict verification incurs the large tensor checksum scans.
`--profile` logs `verification=metadata-v2 (no weight hashing)` or `sha256-strict`,
plus per-matrix `cache-hit`/`prepared` timing. The metadata cache still retains
the existing per-matrix transfer/synchronization policy.

Use the repository's [offline LoRA folding workflow](../../lora/README.md)
first. Quantization consumes the resulting BF16 matrices, after adapter order,
strength and folding have been applied in the original basis. Different folded
files receive different metadata keys, while the existing model/adapter
manifest remains authoritative. A raw LoRA or prequantized third-party checkpoint
is not a replacement for a complete folded H3 model. The 5090 follow-up tested
the `minimax_h3_turbo_v4_step600_ema` Ref2VA adapter at strength 1.0 with eight
steps in both formats. Strength 0.5 has full NVFP4 generation and all four
FP8/NVFP4 projection-family/cache checks. Model and packed identities stay
distinct across base/1.0/0.5. The matched strength-1.0 Turbo triplet received
separate human playback/listening acceptance. Strength 0.5 has functional
coverage only; arbitrary adapters remain unqualified.

## Recipe 1 and limitations

All 50 blocks' QKV, attention-output, SwiGLU FC1 and FC2 matrices use native
W8A8 or W4A4 GEMMs with BF16 outputs and FP32 accumulation. There are currently
**no projection exceptions, rotations or correction branches**. Other matrices,
encoders, refiners, AdaLN precomputation, normalization, RoPE, attention,
residuals, sampler and VAEs retain their existing policies. Profile logs state
the actual native-call count, cache hits/preparations, residency, conversion
time and existing GPU timing categories. Conversion is included in GEMM time;
those categories must not be added as disjoint wall times.
`H3_QUANT_DIAGNOSTICS=1` additionally counts over-range and nonzero-to-zero
packing values on the device, reported with `--profile`. It adds a diagnostic
scan and is intended for calibration, not performance comparisons. Cold totals
include weight preparation; warm totals count activation packing.

FP8 uses E4M3 with tensorwide weight scales and dynamic activation amax. NVFP4
uses E2M1 packed nibbles, 16-value E4M3 block scales in cuBLASLt's 128×4 tiled
layout, and a tensorwide factor. Activations are padded to 128 rows and output
tails are cropped. Scales and scratch belong to the context. All-zero inputs
are supported; nonfinite inputs fail when the compute stream synchronizes.
The public selection does not change requested evaluation counts or enable
reuse, reduction or a distilled schedule.

Sampler checkpoints store a required quantization mode/recipe extension.
Normal resume restores it; explicit incompatible mode overrides are rejected.
Set `cuda_denoise_quant_set=1` to make an API `off` value an explicit override.
The cache location may change because verified packed content has the same
identity. Existing model/backend/build compatibility checks still apply.

Completed AV states preserve their original float latent arrays. Version-2
presentation sidecars bind the quantization mode, recipe and model identity to
the state fingerprint. Normal quantized generation uses a metadata-derived model
identity and adds `quant_identity metadata-v1` to the presentation sidecar; strict
mode retains the original content fingerprint and legacy representation. This
avoids a separate whole-model hash scan for quantization provenance. Explicit
sampler/conditioning compatibility, runtime LoRA validation, and VAE validation
retain their separate contracts. Decode-only full/tiny replay needs neither a packed
cache nor denoiser initialization; use the ordinary decode options without a
quantization flag. **Full-VAE replay cannot recover original-precision
denoising.** An original-precision final result requires another denoising run
with quantization off. Legacy version-1 presentation files and unquantized
sampler files retain their existing interpretation.

Short-preset residency is a major part of the measured speedup over BF16 on a
32-GB card. Large token counts increase attention and activation costs and may
force streaming. Short-clip results are not production-length speed guarantees.
The two modes remain opt-in. Their accepted scope is the tested short base-model
previews, hard/bridge continuations and eight-step Turbo strength-1.0 outputs
on the tested RTX 5090 library configuration. The validation report records
the exact scope and separate Turbo/continuation playback decisions.
