## VideoVAE Tile-Size Compatibility and Quality Default

### Objective

Align `h3cli` VideoVAE decoding with the released MiniMax-H3 inference configuration by making a fixed 256-pixel decoder tile the production default.

The existing geometry-dependent tile-size optimizer should remain available only as an explicit performance mode.

The implementation should:

* use 256-pixel VideoVAE tiles by default;
* preserve the existing numeric `H3_VAE_TILE_PIXELS` override;
* preserve the current 256–320 geometry optimizer behind `H3_VAE_TILE_PIXELS=auto`;
* preserve the existing overlap behavior unless separately configured;
* avoid changing DiT generation, latents, audio, Ref2VA conditioning, or continuation behavior;
* provide validation that 256-pixel decoding has equal or better compatibility with the official MiniMax-H3 decoder;
* document larger tile sizes as performance-oriented options that may alter reconstruction quality.

---

## Background

The current VideoVAE decoder automatically selects a tile size from approximately:

```text
256
272
288
304
320
```

based on output geometry and an estimated decoding-cost function.

This can cause the effective VideoVAE tile geometry to change when output resolution changes.

For example:

```text
576x1024 -> 320-pixel tiles
768x1344 -> 320-pixel tiles
768x768  -> 304-pixel tiles
```

The released MiniMax-H3 configurations instead specify:

```text
vae_tile_size = 256
vae_tile_overlap_min = 64
```

for both FL2VA and Ref2VA.

The VideoVAE also uses spatial positional encoding whose coordinates depend on the dimensions of the processed latent tile. Changing tile dimensions therefore does more than change execution granularity: it can change the positional representation presented to the decoder transformer.

This means the automatic selection of larger tiles should be regarded as an optimization that potentially changes reconstruction behavior rather than as an output-neutral implementation detail.

---

## Default Tile Policy

The decoder should implement the following policy:

```text
H3_VAE_TILE_PIXELS unset
    -> 256

H3_VAE_TILE_PIXELS=<valid numeric value>
    -> explicitly requested tile size

H3_VAE_TILE_PIXELS=auto
    -> existing geometry-dependent 256-320 heuristic
```

The default production behavior should therefore match the released H3 configuration.

Conceptually:

```text
                         +--------------------+
unset -----------------> | fixed 256 default  |
                         +--------------------+

numeric value ---------> explicit tile size

"auto" ----------------> legacy performance heuristic
```

### Recommended Implementation

Refactor the current tile-selection code so that the existing optimizer remains available as a separate function.

For example:

```c
static int auto_tile_pixels(
        int pixel_height,
        int pixel_width)
{
    /*
     * Existing geometry-dependent 256-320 selection logic.
     */
}
```

The public configuration logic becomes approximately:

```c
static int configured_tile_pixels(
        int pixel_height,
        int pixel_width)
{
    const char *value = getenv("H3_VAE_TILE_PIXELS");

    if (!value || !*value)
        return 256;

    if (!strcmp(value, "auto"))
        return auto_tile_pixels(
            pixel_height,
            pixel_width);

    char *end = NULL;
    long pixels = strtol(value, &end, 10);

    if (end &&
        !*end &&
        pixels >= 256 &&
        pixels <= 512 &&
        pixels % 16 == 0)
        return (int)pixels;

    /*
     * Use the existing error/configuration handling policy.
     */
    return 256;
}
```

If invalid environment values currently produce an explicit error rather than silently falling back, preserve that behavior instead of using the illustrative fallback above.

---

## Tile Overlap

This phase should not change the overlap algorithm.

The default decoder configuration should continue to correspond to:

```text
tile size:    256 pixels
min overlap:   64 pixels
```

where supported by the current implementation.

Any future investigation into overlap should be independent from this change because tile size and overlap both affect seam behavior and changing both simultaneously would make quality regressions harder to isolate.

---

## Numerical and Functional Scope

The change affects only VideoVAE reconstruction.

The following should remain unchanged:

```text
text tokenization
Qwen text conditioning
Qwen vision conditioning
Ref2VA reference processing
DiT denoising
random-number generation
generated latent tensor
audio generation
audio VAE
continuation conditioning
bridge conditioning
saved/resumed latent state
```

For identical inputs and seed:

```text
latent before VideoVAE decode
```

should remain identical before and after this change.

Only the decoded video pixels may differ when the old automatic policy would have selected a tile size other than 256.

---

## Compatibility Modes

### Default / Production

```bash
unset H3_VAE_TILE_PIXELS
```

or:

```bash
H3_VAE_TILE_PIXELS=256
```

Use fixed 256-pixel tiles.

This is the recommended quality and MiniMax-compatibility mode.

### Explicit Tile

Example:

```bash
H3_VAE_TILE_PIXELS=320
```

Use the requested tile size.

This provides a controlled way to trade potential reconstruction differences for higher decoder performance.

### Legacy Automatic Mode

```bash
H3_VAE_TILE_PIXELS=auto
```

Run the current geometry-dependent optimizer.

This preserves the existing optimization for benchmarking and users who intentionally prefer decoder speed.

The legacy automatic mode should not be described as equivalent-quality behavior unless testing demonstrates that equivalence for the requested geometry.

---

## Logging

At normal or verbose initialization output, expose the selected VideoVAE tile policy.

Examples:

```text
VideoVAE tile: 256 px (default)
```

```text
VideoVAE tile: 320 px (explicit override)
```

```text
VideoVAE tile: 320 px (auto)
```

For an explicit tile greater than 256, optionally emit a verbose diagnostic:

```text
VideoVAE: using non-default 320 px decoder tiles;
output may differ from the released MiniMax-H3 256 px configuration.
```

This should not be treated as a fatal warning.

---

## Quality Validation

The main validation should operate on saved latents rather than complete generations.

This removes stochastic generation differences and isolates VideoVAE reconstruction.

For each test case:

```text
generate or load one fixed latent
               |
               +-> decode at 256
               |
               +-> decode using auto
               |
               +-> decode at 320 where supported
               |
               +-> decode using official MiniMax implementation
```

Compare the resulting video frames.

### Required Resolutions

At minimum test:

```text
320x320
512x512
480x864
576x1024
768x768
768x1024
768x1344
```

These geometries exercise both cases where the old heuristic already selected 256 and cases where it selected larger tiles.

### Primary Reference Comparison

The most important comparison is:

```text
h3cli / 256 px
        versus
official MiniMax-H3 / 256 px
```

This establishes whether the proposed default improves implementation parity.

### Quality Metrics

Measure, where practical:

```text
PSNR
SSIM
per-pixel absolute difference
tile-boundary gradient discontinuity
tile-boundary luminance discontinuity
```

Automated metrics should supplement, not replace, visual inspection.

Inspect especially:

* uniform walls and backgrounds;
* skies and gradients;
* skin;
* faces;
* hair;
* fine fabric patterns;
* straight edges;
* low-contrast surfaces;
* camera pans where stationary tile artifacts become more visible.

---

## Tile-Boundary Analysis

Add or retain a development-only seam metric that compares edge differences around reconstructed tile boundaries.

For each decoded configuration:

1. determine the output-space boundaries corresponding to tile transitions;
2. measure gradient magnitude immediately across each boundary;
3. compare with equivalent nearby non-boundary positions;
4. aggregate horizontal and vertical seam ratios.

A useful diagnostic is:

```text
boundary edge magnitude /
nearby baseline edge magnitude
```

Values significantly above the surrounding baseline can identify grid artifacts that may be difficult to notice in individual frames but become visible during playback.

The seam test should not become a strict correctness assertion until acceptable thresholds have been established empirically.

---

## Performance Validation

For each representative resolution record:

```text
selected tile size
number of tiles
decoder wall-clock time
peak process memory
peak Metal/unified-memory use where measurable
```

Compare:

```text
256
auto
320
```

where applicable.

Expected outcome:

* 256 may require more overlapping tiles;
* 256 may decode more slowly;
* 256 may slightly reduce per-tile peak memory;
* DiT runtime is unaffected;
* total generation impact should be smaller than the VideoVAE-only percentage for normal full-quality renders.

---

## Regression Testing

### Configuration Tests

Verify:

```text
unset                    -> 256
256                      -> 256
272                      -> 272
288                      -> 288
304                      -> 304
320                      -> 320
auto                     -> legacy heuristic
```

Test valid upper-bound numeric values if the existing implementation permits values above 320.

Also test:

```text
empty value
non-numeric value
invalid mixed value
value below minimum
value above maximum
non-multiple-of-16 value
```

and preserve the chosen configuration-error policy consistently.

### Automatic Mode Parity

For a representative resolution matrix, verify that:

```text
new H3_VAE_TILE_PIXELS=auto
```

selects exactly the same tile values that the current implementation selected before this change.

This establishes that the legacy heuristic has been preserved rather than rewritten.

### Default-Behavior Tests

Update tests that previously expected automatic geometry selection when the environment variable was absent.

They should now expect:

```text
256
```

for every supported output geometry.

---

## Small-Image Validation

Pay particular attention to geometries where the old optimizer could decode the entire image as one tile.

For example:

```text
320x320
```

may change from:

```text
one 320-pixel tile
```

to multiple overlapping 256-pixel tiles.

Test this explicitly for:

* visual quality;
* seam behavior;
* runtime;
* parity with the official implementation.

Do not assume that one larger tile is superior merely because it eliminates stitching; the VAE positional representation also changes with tile dimensions.

---

## Saved-Latent Compatibility

The change should remain fully compatible with existing saved latent files.

A latent produced before this change should be decodable using:

```text
default 256 mode
explicit numeric mode
auto mode
```

without format conversion.

This provides a useful mechanism for direct before/after comparisons.

No saved-state or resume format version bump should be necessary.

---

## Documentation

Update documentation that currently describes geometry-dependent 256–320 tile selection as the default.

The documentation should instead state:

```text
The VideoVAE decoder defaults to 256-pixel spatial tiles,
matching the released MiniMax-H3 configuration.

H3_VAE_TILE_PIXELS=<N> selects an explicit
tile size.

H3_VAE_TILE_PIXELS=auto enables the previous
geometry-dependent performance heuristic.
```

Explain that larger tiles can improve VideoVAE decode performance but may change reconstructed pixels and potentially introduce geometry-dependent artifacts.

Do not imply that larger tiles affect DiT generation quality; they affect latent-to-video reconstruction.

---

## Out of Scope

This change does not attempt to:

* modify the VideoVAE architecture;
* modify VideoVAE model weights;
* modify spatial RoPE implementation;
* change temporal tiling;
* change tile overlap;
* change the VideoVAE encoder;
* change DiT;
* address long-reference memory consumption;
* address issue #47 causal-GQA sequence limits;
* change Ref2VA reference handling;
* change continuation logic;
* guarantee arbitrary tile sizes are equivalent to 256;
* optimize the 256-pixel implementation to recover lost performance.

Performance optimization of the official 256-pixel path can be investigated independently after correctness and quality parity are established.
