# Single-still sampling contract, version 1

This is an optional H3 execution mode, not a newly trained image model.
The image codec passed its independent F32 equations gate at 256×256, 512×512
and 640×480 before generation tests. Reconstruction does not establish prompt
quality. The six-evaluation generation tests below cannot qualify 50-step quality.

The sampling geometry is independently corroborated by two pinned implementations:

- [Combined Image and Reference node, `40c74969`](https://github.com/entropicnoise/MiniMax-H3-Combined-Image-And-Reference-to-Video/blob/40c74969fd805174bf3e1aed1f83f66ef7de6838/minimax_h3_combined_image_and_reference.py): `_empty_av_latent_allow_single_frame` explicitly sets frame count and video T to 1, with two audio ticks.
- [T8 still implementation, `6063fafb`](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/6063fafbd9c3b85c5ff40aef435ae11b2844e558/h3_t8/still_image.py): `direct_1_frame`, `resolve_still_target`, `empty_still_av_latent`, and `assert_still_layout_contract` agree on T=1, audio T=2 and ordinary packed image-reference/audio/video segments. Choose its `generate_and_discard` policy; do not lock audio to silence.

Retained source files, API commit records and file hashes are under
`outputs/single-still/source/` and `source-manifest.json`.

Frozen recipe:

- Target latent `[1,24,1,H/16,W/16]`, normalized H3 latent space. At 640×480,
  DiT 2×2 patching gives 300 visual target rows.
- Auxiliary audio `[1,32,2,2]`: four packed rows / 128 F32 scalar values. Sample
  with the ordinary audio head and audio sigma schedule, then discard. Never
  load the audio VAE or create an audio output stream.
- Ordinary FL2VA task/packed positions for prompt-only generation; ordinary
  Ref2VA image slots for ordered image references, with `match`/`high`/`max` sizing
  under the shared [reference-image policy](design-reference-image-size.md).
  No first/last anchors, video/audio references or continuation.
- Existing shifted flow schedule (video 12, audio 3), ordinary Euler updates,
  all 50 blocks and dense BF16 model execution. Keep the existing h3cli RNG:
  separate video/audio generators, each initialized with the requested seed.
  This preserves h3cli seed semantics; it does not claim PyTorch RNG identity.
- Image VAE uses the metadata-declared output slice and released 256/64 tiles.
  The sampled video target has **one** temporal latent; no video rounding,
  frame repetition, hidden five/22-frame render or audio decode.
- No change to `.h3av`/`.h3sample`; save a normalized `.safetensors` still
  artifact instead. Generation/cache keys identify `still-v1:audio2` separately.

The gate is a **go for integration** based on the independent
sampling recipe and existing DiT equations. Local layout/noise/schedule fixtures
and bounded real DiT transitions must pass. Retained velocity/trajectory records
are regression fixtures for this implementation, not a claim of independent
full-model numerical parity with ComfyUI. Production-quality promotion requires
separate prompt/reference evidence; failed screens remain failures.

Generation test contract: fixed 640×480, original BF16 model, seed 42, six
Euler evaluations; prompt-only and `inputs/2.jpg` image-reference cases; repeat
prompt-only with the same seed. Require finite states, exact target counts,
exact repeated latent checksum on the same execution policy, one final image
callback, no AV state/audio stream, and successful still-latent re-decode.
Record prompt/reference semantic screens independently from deterministic
execution and codec numerical gates. No automatic human-review gate.
