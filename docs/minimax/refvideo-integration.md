# Ref2VA generation integration: T057–T090

`released-v1` is now the default video-reference path in `h3cli`. Images and
keyframes continue through the historical one-frame encoder. Explicit legacy
videos use the unchanged cadence-trimmed, continuous causal encoder and posterior
mean. That implementation remains available for at least one compatibility cycle.

Each visual has independent normalized-frame, VAE-frame and latent-T bookkeeping.
The recorded latent T controls allocation, ordered references, validation,
patchification, augmentation and RoPE. For a 56-frame VAE prefix, released T is 17
and legacy T is 14. The existing 0.999/0.001 augmentation processes the entire
larger span with the same request RNG behavior. Posterior draws independently
restart at seed 42 for every released video. Qwen and soundtrack timing retain
the full normalized duration; VAE snap-down never changes their RGB buffer.

Decoded-video diagnostics include pipeline, normalized frames, selected VAE
frames, latent T and posterior seed. Legacy selection prints a geometry warning.
Sampler provenance and cache separation from T001–T026 remain enforced.

## Validation on M4 Max, 128 GiB

All tests use this machine; no M5 result is claimed.

* 10,424 geometry, duration, Qwen and cache checks; 25 actual media cases.
* 164,916 official layout, allocation and augmentation checks. All positions and
  reference/target row offsets agree with Diffusers within 1e-12. Cases include
  released/legacy 56-frame references, image+video, video+audio, two videos,
  three videos, and audio+video+image with different spatial aspect ratios and
  soundtrack spans.
* 31 native VAE comparisons, including legal lengths 39, 56, 73, 107, 124 and
  normalized lengths 48, 60, 80, 110. A 288×320 temporal fixture exercises spatial
  tiling and patchification together. Raw non-temporal tiling also retains the
  288×304 fixture from the earlier encoder tests.
* Official fixtures retain actual per-chunk raw moments, concatenated moments,
  the three-token drop, epsilon, sampled and FP16-rounded posterior, normalized
  latents and patchified rows. Normalized rows have maximum absolute error at
  most 0.002714; limits are 0.003 absolute and 0.0003 relative L2. Packing itself
  introduces no additional error. Legacy image/video/tiled encoder results
  remain byte-identical to the retained pre-T027 binary.
* One full 50-block transformer forward and a 20-evaluation trajectory pass on
  identical official condition rows, layouts, text and target noise. Relative
  L2: video/audio velocity 0.04428/0.00934; final latents 0.02766/0.06145.
  Predeclared whole-DiT limits are 0.15 relative maximum and 0.10 relative L2,
  matching the existing semantic DiT test's precision allowance.

**Production readiness is on hold (T090).** The real-media 256×256, 124-frame,
20-evaluation test completed in both implementations, but failed its predeclared
trajectory bounds after the AdaLN precision correction: final video/audio
relative L2 0.25587/0.26361, against 0.10. The decoded frame MAE is 0.05768
against 0.05; PSNR is 21.46 dB (minimum 20).
The independent real-media encoder/augmentation comparison passes (max absolute
0.002022, relative L2 0.00002259), and the complete layout agrees within 1e-12.
A further whole-DiT forward on this exact real-media fixture also passes
(video/audio relative L2 0.03207/0.01161). Investigation found that native AdaLN
timestep preparation rounded to BF16
before SiLU; the official path activates in F32 and casts afterward. This is now
corrected only for released video references. Images, FL2VA, legacy videos and
bridge precomputation retain their historical operation order. The official
block-0 AdaLN comparison at timesteps 0 and 0.999 confirms that this correction
reduces relative L2 from 0.001903 to 0.0001554. It does not resolve the complete
trajectory mismatch. Both original and corrected full-generation measurements
are retained; numerical bounds have not been relaxed.

A separate, unmerged shader experiment restored explicit BF16 rounding after
AdaLN, residual products, SwiGLU activation and RoPE products. Five small CPU
and MPS oracle comparisons became byte-exact, but the full real-media DiT with
identical official condition rows still failed (video/audio final relative L2
0.26114/0.36822). This does not establish the remaining error's cause. The
experiment stays under `outputs/refvideo-integration-validation/bf16-prototype/`; the production shaders
retain their previous behavior. A broader DiT precision investigation is still
needed before T090 can close.

T081's complete-generation test is implemented and exposed this precision issue.
The new default remains available for validation, with the legacy
option retained. `tests/refvideo_rollout.py` writes a negative decision before
checking evidence and can only emit `production_ready: true` after every gate
passes. It cannot leave a stale approval after a failed recheck.
Both failed and missing generation-evidence cases were checked against a
previous positive decision; each cleared it and returned a failing exit status.

## Reproduction

Cheap tests, including checked-in official layout fixtures, need no Python ML
packages or model weights:

```sh
make -j8 all
make test-refvideo test-video-posterior test-refvideo-sanitize
make -j8 test
```

The optional official oracles use `tests/requirements-refvideo-oracle.txt`:

```sh
python3 -m venv outputs/refvideo-oracle-venv
outputs/refvideo-oracle-venv/bin/pip install -r tests/requirements-refvideo-oracle.txt
# Use that environment's Python for the following commands.
python tests/refvideo_encoder_oracle.py --stage all --output outputs/refvideo-oracle
make -j8 bin/refvideo_encoder_test bin/refvideo_dit_test
python tests/test_refvideo_encoder.py --fixtures outputs/refvideo-oracle \
  --output outputs/refvideo-native
```

The transformer oracle loads original shards with the **official checkpoint
converter**, including per-head QKV reordering and SwiGLU half swapping. It
checks every target key and shape and hashes source weight contents. Download
the pinned converter outside the source tree:

```sh
curl -L --fail \
  https://raw.githubusercontent.com/huggingface/diffusers/v0.40.0/scripts/convert_minimax_h3_to_diffusers.py \
  -o outputs/convert_minimax_h3_to_diffusers.py
python tests/refvideo_dit_oracle.py \
  --converter outputs/convert_minimax_h3_to_diffusers.py \
  --encoder-fixture outputs/refvideo-oracle/temporal56.safetensors \
  --output outputs/refvideo-dit.safetensors --full
python tests/test_refvideo_dit.py --fixture outputs/refvideo-dit.safetensors \
  --output outputs/refvideo-dit-native --full
```

The default device is MPS; `--device cpu` is also accepted. These optional tests
require the full Ref2VA model and substantial memory. Twenty native evaluations
correspond to 21 official scheduler grid points, terminal zero included. Both
schedules are compared explicitly. Close-reference settings use all 50 layers,
no token reduction or core/velocity reuse, BF16 matrix paths and reference RoPE (scale 1).
Changing this accounting or using the native 256-pixel RoPE adaptation would
invalidate a controlled comparison.

`tests/refvideo_generation_oracle.py` accepts a released, single silent-video
sampler checkpoint at step zero. It independently decodes the same media,
encodes the selected prefix with Diffusers, applies shared augmentation draws
and rebuilds the entire official layout. It fixes Qwen embeddings and target
noise from the checkpoint to isolate this change from existing text-encoder and
request-RNG differences. Its separate `prepare`, `denoise`, and `decode` stages
retain auditable inputs, latents, RGB, PCM and MP4. This is a controlled
conditioning-boundary oracle; it does not certify a separate Python Qwen run or
identical outputs from independently seeded native/PyTorch RNGs.

For an existing step-zero close-reference checkpoint and its source clip:

```sh
python tests/refvideo_generation_oracle.py --stage prepare \
  --checkpoint outputs/start.h3sample --reference inputs/reference.mp4 \
  --output outputs/refvideo-generation
python tests/refvideo_generation_oracle.py --stage denoise \
  --converter outputs/convert_minimax_h3_to_diffusers.py \
  --output outputs/refvideo-generation
python tests/refvideo_generation_oracle.py --stage decode \
  --converter outputs/convert_minimax_h3_to_diffusers.py \
  --output outputs/refvideo-generation
python tests/refvideo_generation_render.py --checkpoint outputs/start.h3sample \
  --oracle outputs/refvideo-generation --output outputs/refvideo-generation-native
python tests/test_refvideo_generation.py \
  --native-checkpoint outputs/refvideo-generation-native/final.h3sample \
  --native-video outputs/refvideo-generation-native/native.mp4 \
  --native-prefix outputs/refvideo-generation-native/native \
  --oracle outputs/refvideo-generation --output outputs/refvideo-generation-results.json
```

The retained checkpoint uses seed 72, 256×256, 124 frames, 20 evaluations,
all 50 blocks, reuse/core-reuse 1, reference RoPE, and the BF16 diagnostic flags.
The native oracle additionally sets `H3_CPU_SAMPLER=1`,
`H3_DISABLE_FUSED_MLP=1` and `H3_DIT_F32_FINAL=1`. The render wrapper reads the
exact saved request, instruments an isolated build, re-encodes the media and
asserts unchanged text, conditions, layout, schedules, provenance and original
noise before comparing final outputs.

`tests/refvideo_render_regression.py` builds a frozen baseline and current
sources in output directories. It encodes visual/audio conditions afresh,
replays only text to isolate known Qwen cold-run variability, then compares
conditions, final AV latents and MP4 bytes. The default suite includes video,
face/body/2.jpg images, and image+video with an embedded soundtrack. It adds
no replay controls to the production executable. All three cases pass exactly;
the mixed case also validates the legacy branch after the AdaLN correction.
The script now generates its static and pan/zoom video fixtures from
`inputs/body1.jpg`; `--reference` can select an existing audiovisual clip.
The retained video-only comparison used
`outputs/continuation-validation/reference.mp4`. The image case uses
`inputs/face1.jpg`, `inputs/body1.jpg` and `inputs/2.jpg`.

After encoder/layout parity passed, `tests/refvideo_ab.py` rendered the video
and mixed cases with the same media, prompt, seed and settings and the legacy
flag removed. Both produce 90 frames at 24 fps with exactly 3.75-second video
and audio tracks. Review of frames 0, 44 and 89 found consistent short reddish
hair, glasses, dress and necklace cues in both paths, a stable prompted garden,
and turning/walking motion. Both mixed outputs raise a hand and retain the
video-reference appearance with the additional `2.jpg` reference.

These smoke comparisons establish no clear fidelity advantage. The references
are photo-derived: the mixed clip adds synthetic pan/zoom, while the prompt
requests a steady camera and replaces the indoor source scene with a garden.
They cannot establish articulated-motion transfer or improved scene retention.
The 220 Hz synthetic soundtrack validates duration handling and mux timing,
not semantic audio preservation or lip sync. Multi-reference ordering has
independent numerical coverage; this single-seed, 256-pixel review is limited
perceptual evidence. Exact artifact hashes and observations are in
`ab/results.json` and `ab/review.json`.

Results and exact command/binary/fixture identities are retained in
`outputs/refvideo-integration-validation/`. The older MLX raw encoder tests
remain low-level continuous causal CNN/posterior-mean checks. They are not the
correctness oracle for released temporal chunking or posterior sampling.
