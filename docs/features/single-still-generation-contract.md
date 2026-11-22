# Single-still sampling and decoding contract, version 2

Updated 2026-11-21. The [version 1 contract](single-still-generation-contract-v1.md)
and dedicated-image-VAE results are historical. Version 2 retains its denoising
trajectory and replaces the final decoder with the stock H3 video VAE.

The frozen [source and model identities](../../tests/empty_still_contract.json)
pin ComfyUI revision `5df2ce62f1408572c18a12a3aefee326cd2305a5`.
Its empty-latent adaptation, sampling preparation and stock VAE establish this
contract. Native h3cli constructs the H3 target directly; it does not allocate
ComfyUI's intermediate four-channel empty image latent.

- Batch one, normalized video `[1,24,1,H/16,W/16]`, auxiliary audio `[1,32,2,2]`.
  Audio contributes four packed rows, is sampled, and is discarded without an
  audio decoder. Video contributes `(H/32)*(W/32)` target rows, including 8,228
  at 3872×2176. Text and ordered reference rows are additional.
- All 50 DiT blocks, dense BF16, reference RoPE, separate seeded video/audio
  noise generators, Euler updates and shifted flow schedules (12 video, 3 audio).
  This differs from the supplied ComfyUI workflow's `er_sde`/`simple` trajectory;
  output images and PyTorch noise sequences are not claimed identical.
- FL2VA for prompt-only generation; Ref2VA for ordered image references.
  Shared `match`, `high`, and `max` reference preprocessing stays independent of
  target size. Existing task IDs, slot order and packed positions are retained.
- Positive multiples of 32, at most 16,777,216 pixels and 8192 per side. The
  geometry check is shared by generation, artifact I/O and codec-only decode.
  Ordinary video/upscale/reference limits are unchanged. Admission does not
  guarantee sufficient device memory.
- After denoising, save the original T=1 normalized tensor. Only the decoder
  repeats it five times, separately per tile; unwhiten exactly once using the
  selected stock VAE configuration. All five temporal positions and the four
  register tokens plus zero suffix participate in all 36 F32 decoder blocks.
  Select zero-based raw frame 3, before video trimming.
- Tiles use the pinned 256-pixel profile and minimum 64-pixel overlap, including
  adjusted edge overlaps. Blend the previous blended bottom strip, then the
  previous blended right tail. Assemble one image; normalize RGB with
  mean `[.485,.456,.406]`, std `[.229,.224,.225]`, then clip to `[0,1]`.
  No full-resolution multi-frame movie is retained.
- The decoder has a separate lifetime from video caches. Release the completed
  DiT before loading it. Deliver exactly one final RGB24 callback (index 0,
  count 1, step -1), then atomically publish PNG. Cancellation before publication
  preserves an existing PNG. Results own their normalized still latent and
  have no AV/sampler state, frame rate or audio fields.

Saved artifacts require schema `2`, recipe `stock-video-repeat5-frame3-f32-v1`,
latent space `h3-normalized-v1`, model variant `FL2VA` or `Ref2VA`, and a SHA-256
compatibility identity over the recipe and 24 F32 normalization means/stds.
The digest binds normalization and decode semantics, not every model weight.
Schema 1 is rejected with guidance to regenerate. Changing its metadata is not
migration: its encoder/whitening identity describes the retired dedicated VAE.
Existing `.h3av`, `.h3sample` and video presentation formats are unchanged.

The independent runner is [empty_still_reference.py](../../tests/empty_still_reference.py).
It implements the F32 transformer equations in PyTorch and executes the
hash-verified upstream tile methods. Gates were frozen before comparison:
RGB MAE ≤0.01 and PSNR ≥35 dB. Layout/RNG/schedule tests independently check
packing. Native trajectory repeats establish same-policy determinism, not
independent full-model sampling equivalence. See the
[replacement qualification record](empty-still-results.md) for tested geometry,
quality observations, resources, backend limits and reproducible commands.
