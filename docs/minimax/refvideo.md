# Ref2VA video conditioning

Video references use one released preprocessing recipe. The historical
continuous video-conditioning route and its CLI selector have been removed.
Images, first/last keyframes, output decoding and audio VAE keep their respective
current paths.

```sh
./bin/h3cli -d models/MiniMax-H3 -p 'Continue the motion and ambience.' \
  --width 640 --height 480 --frames 124 --steps 6 \
  --ref-video reference.mp4 -o outputs/reference.mp4
```

`--ref-video` includes embedded audio, `--ref-silent-video` excludes it, and
`--ref-video-audio VIDEO AUDIO` supplies a replacement soundtrack. Mixed image,
video and separate audio references preserve their declared order.

## Geometry and timing

Decode normalizes video to 24 fps within the target duration and bounded input
allocation. Metal requires 48–360 normalized frames; CUDA accepts at least five
frames and up to 360 total across videos. Soundtrack/audio duration limits still
apply independently. No reference is silently resized to satisfy patch limits.

The VAE selects a `17*n+5` prefix, encodes independent 17-frame chunks with
repeat-last padding, drops the three-token tail, samples the posterior with
seed 42, rounds through FP16 and normalizes in F32. This produces `5*n+2`
latents: 39/56/73/124 VAE frames produce 12/17/22/37 latents. Raw causal encoder
geometry remains an internal operator contract for chunks and keyframes.

Qwen uses the original normalized RGB stride, samples indices `0,12,24,...`,
pairs adjacent samples and repeats the last sample when necessary. Pair
timestamps are the average sample time. Soundtracks follow the normalized
video duration, not the shorter selected VAE prefix. Diagnostics record the
resolved recipe, normalized frames, VAE frames, latent count and posterior seed.

## Saved conditioning

Current conditioning schema 2 includes the released recipe identity. Sampler
schema 2 requires section 32 containing one U32: zero without reference video,
two for released video. No user-selectable pipeline field remains. Resume uses
stored conditioning without reopening source references and validates the
recipe against the recorded layout. See the
[state contract](../features/current-state-contract.md).

Current checks are `make test-refvideo test-sampler`, their sanitizer targets,
and the separate complete [204-output SGLang gate](../../CONTRIBUTING.md).

## Historical T001–T026 evidence (superseded)

The commands, counts and old-reader behavior below describe an earlier build.
Retired regression scripts are no longer shipped. Current acceptance uses the
recorded SGLang gate and current functional tests.

```sh
make -j8 all
make test-refvideo test-sampler
make test-refvideo-sanitize
make test-sampler-sanitize
make -j8 test
```

Focused validation passed 10,424 geometry, duration, Qwen and cache checks;
25 actual media cases in five groups; 1,277 sampler host checks; 122 adversarial
container cases in 14 groups; and 41 CLI parser cases. Refvideo and sampler
ASan/UBSan runs passed. Media tests generate a lossless 30-fps source, embedded
and supplied stereo audio, and duration-boundary clips. Independent FFmpeg RGB
and PCM output plus explicit Python index/packing calculations serve as oracles.
These validate the preprocessing boundary, not future released VAE numerics.

The full `make test` suite passed, including sampler GPU, continuation, bridge,
audio primitives and concurrent media muxing. Eleven optional fixture-dependent
checks were skipped because their expected fixture paths are not installed.
The separate real-model regression below used the available local model weights.

The real-model regression used the retained 72-frame audiovisual fixture at
`outputs/continuation-validation/reference.mp4`, a 256×256 / 90-frame target,
20-step schedule and seed 72. It compared against the frozen pre-change build
at commit `3efffc122f10570657fba529f98b98de3d879bb2`. All 12 checked checkpoint
sections matched byte for byte: augmented video/audio conditions, reference
descriptors, full layout, schedules, current target latents, RNG state,
continuation metadata, evaluation history and original noise. Conditions were
encoded afresh, without replaying captured rows. This comparison does not claim
complete cross-build render or cold text-encoder parity.

A fresh CLI process resumed the legacy checkpoint from boundary 0 to 1 with
original reference paths unavailable, restoring the selector, conditioning and
prepared BF16 tensors. Both directions of old/new reader rejection passed.
The default released front end reported `72 / 56 / 17` and rejected generation
at the pending encoder stage without writing a checkpoint.

Logs, baseline source/binary identity, checkpoints and regression section hashes
are retained under `outputs/refvideo-validation/`; the real-model result is
`legacy-regression-final/results.json`. M5 and complete released-pipeline
generation were not tested in that batch. Current encoder results are recorded
in [refvideo-encoder.md](refvideo-encoder.md); current integration results are recorded separately.
