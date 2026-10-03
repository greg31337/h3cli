# Single full-video decoder

The CLI `--full-vae-execution` and the corresponding fields in `h3_params`,
`h3_decode_options` and internal CUDA policy are removed. Full CUDA video
rendering always uses the previously qualified SGLang-compatible decoder.
Preview TAEH3 and single-still image VAE remain separate and unchanged.

The retained CUDA recipe uses range-safe FP16 block matrices and attention,
FP32 residuals and sensitive operations, the original weights, and the existing
spatial/temporal blend and output conversion order. CUDA graph replay, bounded
GPU stitching, pinned readback and streamed FFmpeg delivery remain enabled.
Metal retains its original FP32 full-video decoder.

## Removed implementation

Removed the alternate explicit FP32 CUDA policy, balanced BF16 linears and
attention, TensorRT engine loading/execution/build tooling, Metal mixed VAE
kernels, mixed-only plans/scratch, policy scopes and decoder-cache mode keys.
The CUDA convolution plan helper remains because the original encoder uses it.
The graph on/off diagnostic remains for testing the same arithmetic.

The frozen suite still calls two historical internal interfaces. A stateless
benchmark shim accepts only the name `legacy` for the default, and the GPU
setup function accepts only a reserved zero argument. These cannot select an
alternate decoder. All sixteen frozen test files and their manifest remain
unchanged; no golden, dependency, tolerance or case was relaxed.

Existing approximate candidates were qualified against looser visual criteria,
not the immutable SGLang contract. This change keeps the fastest previously
parity-qualified recipe; it does not claim a new speedup or an exhaustive new
benchmark of the removed candidates.

## Validation

Final source fingerprint:
`8cec1032d0c6d032b630f767e3b571b3d7f5a6bc702b5a0b5523a9af7191ed3d`.

Frozen suite manifest:
`4ac45c94f1cbec276333102676ce1600ead71da025df4ca60b9ae2343098e413`.

CUDA qualification uses RTX PRO 5000 72 GB, model
`/path/to/models/MiniMax-H3`, and compatible CUDA/media libraries.
Model weights are not hashed.

The unchanged gate passes **204/204 artifacts**, including six complete
640×480 / 124-frame denoising evaluations and decoded RGB/PCM. Test execution
was 119.67 seconds after the isolated build. This is an exact frozen
regression match, not a new comparison against another SGLang release.

Additional CUDA checks pass: raw unpack and nonfinite recovery; policy checks;
13 CLI/delivery cases (all removed selector values rejected, preview and full
output decoded with expected frame/audio streams); streamed/materialized
bitwise equality; repeated allocation plateau; cancellation, memory admission
failure, sink failure and successful retry. These focused checks take about
18 seconds excluding their build.

[Playback and validation summary](../../outputs/vae-single/review.html) ·
[Final gate](../../outputs/vae-single/evidence/gate4/result.json) ·
[Focused checks](../../outputs/vae-single/evidence/focused/result.json) ·
[Local checks](../../outputs/vae-single/local/result.json) ·
[Source/removal audit](../../outputs/vae-single/audit.json).

The earlier attempts caught missing compatibility arguments at compile time.
Their logs are retained. The subsequent gate passed, and the closing gate was
repeated on the final source after mutable benchmark cleanup.

Local Metal build, tile/overlap and posterior contracts, still host/GPU,
preview host and sampler checks pass. The short Metal GPU suite passes
25,166 checks; tile/overlap passes 49,206; posterior/RNG passes 184,304;
sampler passes 1,772. No long Metal render was needed.

The mutable preview host fixture contained an older hardcoded quantization
recipe value (1); the current validator already requires recipe 2. The fixture
now uses `H3_QUANT_VERSION`, without changing production validation. The mutable
parallel-tile benchmark also needed its existing stitch call updated for the
current argument list. These repairs do not change the frozen suite.

See the [current VAE guide](../preview/fast-vae.md) for CLI and C API migration.
