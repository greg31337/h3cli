> Historical qualification/provenance record. The original-checkout VAE comparison
> was removed during [current-state cleanup](../cuda/legacy-cleanup-results.md).
> Its commands below describe the archived run, not the current test suite.

# Single-still provenance

Implementation baseline: h3cli `31d8ee834197b19b4f2ac1d4dc9f8d3174ce3a7c`.
The C implementation extends this repository's existing VAE operators. It does
not vendor VPIPE's model runtime or a new Metal/CUDA kernel. The tensor schema
contains tensor names/shapes from the pinned artifact, not weight data.

| Source | Pinned identity | Use |
| --- | --- | --- |
| [VPIPE](https://github.com/tgo-app-dev/vpipe/commit/351b20c10761eca8ba2f07f923323ca78bd7da15) | `351b20c10761eca8ba2f07f923323ca78bd7da15` | Direct T=1 decoder capability, explicit temporal slice, released spatial tiling; conceptual reference, no C++ source imported. VPIPE is Apache-2.0. |
| [Image VAE](https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE/tree/c7b9252c73707dba494cf4d99ca45d3f33f561b3) | `c7b9252c73707dba494cf4d99ca45d3f33f561b3` | Explicit model artifact; model licensing remains separate from h3cli code. |
| [Original VAE equations](https://github.com/MiniMax-AI/MiniMax-H3/tree/d21241f0a4b3acbb34c97dae47fa417b7065e438/FL2VA/video_vae) | `d21241f0a4b3acbb34c97dae47fa417b7065e438` | Independent test equations: `vae_vit.py`, `base_module.py`, `attention.py`, `conv.py`, `vae_cnn.py`, `func.py`, `klvae.py`. These sources declare Apache-2.0; tests acknowledge MiniMax's implementation. |
| Community sampler references | See [generation contract](single-still-generation-contract.md) | Independently corroborated T=1 + auxiliary audio T=2 recipe; no sampler code imported. |

The complete downloaded artifact has SHA-256
`6c3d0bfa055986a803a566a862fcde283a1e63db62829e5ef4a2a5aebf50bb86`,
size 5,207,808,784 bytes, and 562 F16 tensors. Full checksum verification preceded
any model tests. Header-declared decoder/base digests are provenance, not a
replacement for this checksum.

Audit against the local original F32 VAE: all 116 encoder tensors equal after
F16 rounding (76 already exactly equal as values); both quant-convolution tensors
are equal; both post-quant tensors differ; 439 of 440 decoder tensors differ.
Whitening metadata is equal. Full per-tensor results are retained in
`outputs/single-still/checkpoint-audit.json`. No byte-equality claim is made
between different source dtypes. The image encoder compatibility digest includes
its actual F16 encoder/quant payload and F32 metadata whitening.

Reference arithmetic expands F16 weights to F32, uses explicit PyTorch
matmul/softmax, independently computes T=1 RoPE and all four output slices, and
retains unclamped RGB. Its T=1 encoder uses the final temporal kernel plane of
the causal zero-padded convolution; it does not replicate the input in time.
The reference's initial padding/crop adapter errors were corrected from the
original source before encoder qualification. Numeric acceptance thresholds
were unchanged. `tests/still_contract.json` fixes the Lanczos crop recipe.

Reproduce locally (GPU commands serial):

```sh
make bin/h3cli bin/still_probe bin/still_tests bin/still_generate bin/still_controls bin/still_lifecycle
python3 tests/still_audit.py
python3 tests/still_codec_matrix.py
make test-still-host
make test-still-sanitize
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 bin/sanitizers/single-still/still gpu
```

Python real-model tests require NumPy, PyTorch with MPS, and Pillow. The actual
validation used `outputs/continuation-validation/venv/bin/python3`. Full command
arrays are retained with the fixture logs; artifact/source/report checksums are
retained under `outputs/single-still/`. Models and test outputs are not committed.

Host tests use a sparse file built from the pinned metadata-only fixture and
need no model weights. Real codec/reference/generation tests require the model
files. The exact earlier reference script used for 256/640 fixtures is retained
as `source/still-reference-initial.py`; the later script only separates the
whole-image trace directory from the tiled trace directory. Reference records
identify both script digests explicitly.

Additional reproducible checks (GPU invocations serial):

```sh
./bin/still_controls models/image-vae/minimax_h3_t1_image_vae_step1597.safetensors outputs/single-still/fixtures/256/latent.safetensors outputs/single-still/fixtures/256
./bin/still_lifecycle
make bin/still_video_baseline bin/still_video_regression
./bin/still_video_baseline outputs/metal-native-m6-q8/runs/B6-reference-plus/result.h3av 72
./bin/still_video_regression outputs/metal-native-m6-q8/runs/B6-reference-plus/result.h3av 72
python3 tests/still_report.py
```

The baseline target extracts the original VAE source from the pinned Git
commit. Passing latent time 7 instead of 72 produces the short 22-frame check.
`tests/still_generate.c` captures the six-step prompt/reference fixtures and
all latent transitions; its command takes an output directory, prompt and an
optional third argument to select `inputs/2.jpg`. Create the output directory
first. Use `H3_PROFILE=1 H3_TEST_MAX_EVALUATIONS=6` to retain phase timing and
enforce the test ceiling. Final generation prompts are in the usage examples.

CUDA follow-up qualification is documented in [the CUDA record](single-still-cuda-results.md).
The independent reference disables CUDA TF32 explicitly; its earlier Metal
version is retained as `outputs/single-still/source/still-reference-metal-qualified.py`.
The CPU-only report scripts link the two evidence sets without changing the
frozen codec thresholds. See `outputs/single-still-cuda/` for initial/final
source manifests, binary hashes, original model header inventory, fixture
checksums and exact serial commands.
