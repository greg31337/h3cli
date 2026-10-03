# Original-model video VAE

Full video decoding now has one execution path per backend. CUDA keeps the
SGLang-compatible decoder already used by default rendering: FP16 block
matrices and attention, FP32 residuals and sensitive operations, and the
original checkpoint, tiling, blending and byte conversion. It retains CUDA
graph replay, bounded streaming, GPU spatial stitching and pinned output
readback. Metal keeps its original FP32 decoder.

`--full-vae-execution` is removed. The alternate explicit FP32 CUDA, balanced
BF16 and TensorRT implementations, engine builder and Metal mixed-precision
variants are removed with it. No TensorRT dependency remains. The retained
CUDA recipe is the fastest previously qualified implementation under the
frozen SGLang parity contract; older approximate candidates were tested under
different quality criteria and are not substitutes for that contract.

`--preview-vae` still selects TAEH3. Its weights, precision, live-preview
behavior and defaults are unchanged. Single-still image decoding remains a
separate implementation. Attention and denoiser quantization options do not
select a different full video decoder.

## Usage

Use the full decoder by omitting `--preview-vae`:

```sh
./bin/h3cli -d models/MiniMax-H3 -p "A pianist playing a grand piano." \
  --width 640 --height 480 --frames 243 --steps 6 \
  --save-av-state outputs/piano.h3av -o outputs/piano.mp4
```

A completed `.h3av` state contains the original denoised latents regardless of
its presentation decoder. Finalize a TAEH3 preview without denoising again:

```sh
./bin/h3cli -d models/MiniMax-H3 \
  --decode-av-state outputs/draft.h3av -o outputs/final.mp4
```

Keep the `.presentation` sidecar to preserve output dimensions and continuation
and audio trimming. Current AV schema 3 and presentation schema 8 are required.
The saved backend delivery recipe must match the decoding build.

CUDA generation and decode-only delivery stream completed frame ranges to
FFmpeg. The materialized float-frame API still allocates the requested full
video. Existing memory admission and cancellation remain in place. CUDA graph
replay is enabled by default where supported, with the existing fallback on
capture failure. `H3_FULL_VAE_CUDA_GRAPH=0` remains a diagnostic switch for the
same arithmetic, not another precision mode.

## Migration and validation

Remove `--full-vae-execution VALUE` from commands. Remove `full_vae_execution`
from C parameter and decode-option initializers, and recompile C callers after
updating the headers. The old selection is not part of sampler arithmetic, so
the current cleanup separately replaces saved-file contracts; write fresh states.

`make test-full-vae` checks the remaining decoder host contracts. CUDA tests
also check raw unpack, nonfinite failure and recovery, streamed versus
materialized equality, repeated allocation stability and default/preview CLI
delivery. The complete immutable `make test-cuda-reference-regression` gate
must run after every coherent code change, as required by
[CONTRIBUTING.md](../../CONTRIBUTING.md).

The no-op execution/VAE shims are removed. Parity probes load the one frozen
old-format AV fixture through a checksum-bound test-only adapter, then call
current production operators. No backward loader remains in the library.

See the [single-decoder validation record](../cuda/vae-single-results.md).
The earlier [design](design-fast-vae.md), [CUDA results](fast-vae-results.md)
and [Metal results](fast-vae-metal-results.md) are historical records of the
removed experiments, not current usage instructions.
