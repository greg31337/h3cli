# Released Ref2VA video VAE: T027–T056

The native encoder now exposes the complete released video-conditioning API:
`h3_ref2va_video_vae_encode()`. It returns owned, normalized F32
`[24,T,H,W]` latents, following [design-refvideo.md](design-refvideo.md).
T027–T056 are implemented and tested on M4 Max.

Generation now calls this API by default for Ref2VA video references. Images,
keyframes and explicit legacy videos retain the historical encoder. These encoder
results cover T027–T056; [integration validation](refvideo-integration.md) covers
condition rows, layout, whole DiT and generation. The old MLX encoder tests remain
low-level continuous CNN/posterior-mean checks, not a released Ref2VA oracle.

## Encoder contract

The API in `src/vae/video_encoder.h` takes both `source_frames` and `vae_frames`.
The former is the channel stride of the borrowed normalized RGB `[3,T,H,W]`
allocation; the latter is its legal `17*n+5` prefix. A 60-frame buffer can
therefore supply a 56-frame VAE prefix without changing Qwen's RGB view.
Spatial axes must be multiples of 16 and at least 32 pixels. Media duration
validation remains in the preprocessing layer.

The encoder loads weights and latent normalization once per call, then:

1. Extracts independent 17-frame RGB chunks, repeating the last selected frame
   for the final chunk's padding.
2. Runs the shared causal CNN and quant convolution, retaining all 48 channels.
3. Stitches raw moment tiles at 256 pixels with at least 64 pixels of overlap.
   Vertical blending precedes horizontal blending, using the original neighboring
   tiles as in the official implementation.
4. Concatenates `[48,5,H,W]` chunk moments and removes exactly three trailing
   temporal slices. It checks the final length against `h3_video_latent_t()`.
5. Splits mean/log-variance, clamps log-variance to `[-30,20]`, and computes
   `mean + exp(0.5*logvar)*epsilon` in separate F32 operations.
6. Rounds the sample through IEEE FP16 using round-to-nearest-even, then applies
   the existing F32 latent mean/std normalization.

The RNG initializes independently to seed 42 for every call. It implements
PyTorch's CPU MT19937 contiguous F32 normal draw, including eight-pair blocks,
the replacement of the last 16 values for non-multiple lengths, and the small
double-precision fallback. It neither consumes nor changes the request's
`h3_rng`. F32 operation boundaries are preserved in posterior math to avoid
fused multiply/add moving samples across FP16 midpoints.

`h3_video_vae_encode_moments()` exposes raw continuous CNN output for tests;
`h3_ref2va_video_vae_moments()` adds the released temporal wrapper. The internal
posterior primitive also accepts explicit epsilon, separating numerical checks
from RNG checks. Pair each owned output with `h3_video_moments_free()` or
`h3_video_latent_free()`. Progress counts completed spatial tiles across all
chunks, and GPU statistics accumulate over the complete encode.

The historical `h3_video_vae_encode()` retains its continuous causal,
posterior-mean path and normalization before stitching. It shares CNN execution,
weight loading and the channel-count-independent stitch loop, while keeping
its postprocessing separate. Image/keyframe behavior is preserved.

## Oracle and numerical bounds

Fixtures use official Diffusers 0.40.0 with PyTorch 2.14.0 on CPU in F32.
Only the encoder and quant-convolution tensors from local
`models/MiniMax-H3/Ref2VA/video_vae/source/model.safetensors` are materialized; the unused
decoder stays on the meta device. Original weight names are mapped to Diffusers
module names and loaded strictly, with no encoder operation replaced.

The reference implementations are
[AutoencoderKLMiniMaxH3](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py),
[conditioning preparation](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/modular_pipelines/minimax_h3/encoders.py),
and [PyTorch CPU distributions](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/cpu/DistributionTemplates.h).
The installed encoder source SHA-256 is
`4c3c9745ee27d16ff343c4998244bad41cd8f4213f0029cf7ce11ebb6d72ca1b`.
Package/source hashes, model identity, fixture hashes and commands are retained
with the results. The oracle is an optional test dependency; native generation
and ordinary host tests require none of these Python packages.

| Check | Result on M4 Max |
| --- | --- |
| 17-frame 64×64 raw moments, all 48 channels | Max abs `3.55e-5`, relative L2 `7.36e-7` |
| Temporal geometry | 39→12, 56→17, 73→22, 107→32, 124→37 |
| 60-frame buffer / 56-frame prefix | Same moments as tightly packed 56-frame input; RGB unchanged |
| Combined temporal + spatial moments, 39 frames at 288×304 | Max abs `1.85e-4`, relative L2 `5.34e-7` |
| Complete normalized latents across the matrix | Max abs at most `0.002714`, relative L2 at most `3.33e-5` |
| Seed-42 epsilon, optimized build, including 131,071-value draw | Bit-identical to checked PyTorch CPU samples |
| Explicit-epsilon posterior | FP16-rounded samples and normalized outputs exactly match the math fixture |
| Legacy image, continuous video and tiled output | Byte-identical to frozen pre-change native encoder |
| Repeated reference encode after request seeds 72 and 987 | Byte-identical output |

Raw CNN and temporal moment comparisons require maximum absolute error ≤`2e-3`
and relative L2 ≤`2e-4`. Synthetic stitching requires ≤`1e-6` and ≤`1e-7`.
Complete normalized latent comparisons require ≤`3e-3` and ≤`3e-4`.
Small F32 CNN differences can cross an FP16 rounding midpoint, producing a full
quantization step after normalization; the final comparison therefore uses both
an absolute and a whole-tensor relative bound. It does not claim bitwise GPU/CPU
CNN parity. With identical oracle moments and epsilon, rounding/normalization
are exact. Sanitized builds allow epsilon error ≤`1e-6`; the observed maximum
was `1.19e-7`, reflecting scalar math/compiler differences.

The raw 17-frame test passed before temporal concatenation was added. Further
fixtures cover a 288×304 raw encode, synthetic 512×704 tile layouts including
both overlap axes and corner blend order, a real CNN continuous-vs-chunked
comparison, every supported legal chunk-plan length, and a combined tiled
multi-chunk encode. Host FP16 tests exhaust all 65,536 bit patterns and finite
midpoints; the full oracle also checks adjacent F32 values on either side.

## Reproducing validation

The checked-in 66 KiB fixture contains CPU epsilon samples, explicit posterior
inputs/outputs and FP16 boundary cases. Run the independent native host tests with:

```sh
make -j8 all bin/refvideo_encoder_test
make test-video-posterior
make test-video-posterior-sanitize
make test-refvideo test-sampler
make -j8 test
```

For the full numerical oracle, use a separate Python environment and local
weights. Run model jobs sequentially:

```sh
uv venv --python 3.12 outputs/refvideo-encoder-validation/venv
uv pip install --python outputs/refvideo-encoder-validation/venv/bin/python \
  -r tests/requirements-refvideo-oracle.txt

outputs/refvideo-encoder-validation/venv/bin/python tests/refvideo_encoder_oracle.py \
  --stage all --output outputs/refvideo-encoder-validation/oracle

outputs/refvideo-encoder-validation/venv/bin/python tests/test_refvideo_encoder.py \
  --fixtures outputs/refvideo-encoder-validation/oracle \
  --output outputs/refvideo-encoder-validation/new-native

# Also exercise native allocations and indexing under ASan/UBSan:
outputs/refvideo-encoder-validation/venv/bin/python tests/test_refvideo_encoder.py \
  --fixtures outputs/refvideo-encoder-validation/oracle \
  --binary bin/sanitizers/refvideo-encoder-validation/refvideo_encoder_test \
  --smoke --output outputs/refvideo-encoder-validation/new-sanitized
```

The optional `--baseline PATH` points to a frozen pre-change encoder test driver
and requires exact bytes for all three legacy cases. The recorded run used the
retained library from commit `1d0caff3e31f14914fcf2f2e2e879e7bbee85545`.
To regenerate the compact host fixture, add
`--compact-host-fixture tests/fixtures/refvideo-posterior.safetensors` to the
oracle's `--stage math` or `--stage all` command. Its adjacent JSON records provenance.

Results are under `outputs/refvideo-encoder-validation/`: `native-final/results.json`
records 25 encoder comparisons, `posterior-tests.log` records the full 567,036
host/oracle checks, and the compact ordinary suite runs 184,304 checks. Sanitizer
results are in `sanitizers.log` and `native-sanitized/results.json`.
The full repository suite and focused preprocessing/sampler suites passed.
Eleven optional checks in `make test` skipped their absent legacy fixture paths;
the 25 new encoder comparisons used the available local weights and all ran.
The staged CLI check confirmed the T057–T070 diagnostic and no checkpoint output.
Only M4 execution is certified by these runs; M5 and complete Ref2VA generation
are outside this batch.
