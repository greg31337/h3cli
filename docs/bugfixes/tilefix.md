# VideoVAE decoder tile policy

An unset `H3_VAE_TILE_PIXELS` selects **256 pixels** at every output resolution.
This is the recommended quality and compatibility setting: both released
MiniMax-H3 model configs specify `vae_tile_size=256` and
`vae_tile_overlap_min=64`. The decoder's existing overlap distribution, blend
order, temporal chunks, positional encoding, architecture, and weights are
unchanged. Overlap can exceed 64 to fit the canvas; dimensions below the selected
tile size use their actual extent.

| Environment | Policy |
| --- | --- |
| unset | Fixed 256, production default |
| `256` | Explicit fixed 256, same reconstruction |
| numeric 256–320, multiple of 16 | Explicit tile |
| `auto` | Legacy optimizer: minimum tile count × tile area among 256, 272, 288, 304, 320; smaller candidate wins ties |
| empty, malformed, below 256, above 320, or not a multiple of 16 | Configuration error |

Invalid explicit values now fail before decoder initialization. The error reports
the requested value, the supported range, and the released default. **Unset** the
variable to select 256; an empty string does not unset it. Values above 320 are
unsupported because larger decoder tiles have demonstrated severe grid/quilt
reconstruction artifacts. The historical report below documents the earlier
256–512 policy; the [runtime robustness report](bugfix1.md) supersedes that range.
The existing `strtol` syntax (leading whitespace, plus sign, leading
zeroes) remains accepted. Trailing whitespace remains invalid.

With `H3_PROFILE=1`, both standalone decoding and resident preview/final decoding
report the effective tile and its origin. Explicit values above 256 also print a
nonfatal reconstruction diagnostic. The resident decoder selects
its policy at load time; use a fresh decoder/process to change policy. The
two-token diagnostic path still supports only one spatial tile up to 256 pixels.

Larger tiles and `auto` remain exposed as **performance options**.
Tile dimensions affect the decoder's spatial positional representation as well
as stitching, so fewer tiles do not imply equivalent reconstruction. Use these
modes for controlled comparisons or when accepting different pixels in exchange
for speed. Do not interpret their speedup as an improvement in DiT generation.

## Upgrade notes

Previous releases chose geometry-dependent tiles by default. Upgrading changes
final video pixels wherever the old heuristic selected 272, 288, 304, or 320.
For example, 320×320 used one 320 tile; the new default uses four overlapping
256 tiles. `H3_VAE_TILE_PIXELS=auto` reproduces the previous selection without
reverting the code. No encoder or reference-video path reads this decoder option.

Text and Qwen vision conditioning, reference encoding, RNG, DiT-generated video
and audio latents, AudioVAE output, and continuation conditioning are unaffected.
Existing `.h3av` and `.h3sample` payloads retain their formats and their video
latents can be decoded under any supported policy. Full sampler continuation
still enforces its existing build/environment compatibility checks; a decoder
policy change does not waive those checks or imply cross-build exact resume.
Decoded previews of intermediate sampler latents may contain noise.

## Development validation

`make test-video-vae-tiles` checks policy, the captured legacy resolution matrix,
logging, and unchanged 256/64 geometry. `make test-video-vae-tiles-sanitize` runs
ASan/UBSan on the configuration/geometry code.

`make bin/tilefix_decode` builds a development utility that bypasses DiT. Its
arguments are `MODE T H W INPUT OUTPUT WEIGHTS`: H/W are output pixels; T is latent
time for decoding or source frames for encoding. Inputs and outputs are raw
little-endian F32, except the `av` and `sample` modes read and validate existing
state files directly. Decode RGB is `[frames,H,W,3]`; raw latent input is
`[24,T,H/16,W/16]`. `resident` measures weight loading separately from decode;
`decode`, `av`, and `sample` include loading in their decode timing. `plan` needs
no weights. `encode` uses the unchanged released Ref2VA encoder to create test
latents. JSON includes the actual tile starts/overlaps, timings, process maximum
RSS, and tracked Metal tensor peak. RSS and Metal memory share physical RAM and
must not be added together. Metal tensor accounting excludes driver scratch.

Example (run from the repository root):

```sh
H3_VAE_TILE_PIXELS=auto ./bin/tilefix_decode resident 7 576 1024 \
  latent.f32 auto.f32 models/MiniMax-H3/FL2VA/video_vae/source
```

With the packages in `tests/requirements-tilefix.txt`:

```sh
python tests/tilefix_validation.py --oracle
python tests/tilefix_metrics.py outputs/tilefix-validation/matrix
python tests/test_tilefix_metrics.py
python tests/tilefix_generation.py
python tests/tilefix_saved.py
python tests/tilefix_compatibility.py
python tests/tilefix_gallery.py
```

The official comparison uses the installed upstream Diffusers
`AutoencoderKLMiniMaxH3` with released weights, its 256/64 tiling, F32 on MPS, and
the unmodified upstream converter at the path accepted by
`tests/tilefix_oracle.py --converter`. Artifacts retain decoder/converter/weight
hashes and package versions. No network is needed when these are installed.

The matrix builds deterministic 22-frame pans from `inputs/face1.jpg`,
`inputs/body1.jpg`, and `inputs/2.jpg`, plus a gradient and low-contrast straight
lines. Each canvas is encoded once and the **identical saved latent** goes to all
four decoders. All reconstructed F32 frames, MP4s, selected plans, checksums,
contact sheets, and error maps stay in `outputs/tilefix-validation/`.

PSNR and per-pixel errors use unclipped-difference F32 RGB in [0,1]. SSIM uses
11×11 valid uniform windows on Rec.709 luminance, population covariance,
K1=.01, K2=.03, and L=1. Seam diagnostics sample both edges of each overlap and
compare gradients and four-pixel luminance steps with nearby non-boundary
positions. Content edges can raise the ratios; these are not pass/fail thresholds.
