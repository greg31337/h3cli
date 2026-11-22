# Stock-video-VAE still replacement qualification

Qualification date: 2026-11-21. **Complete: all 20 replacement tasks passed.**
The [machine-readable acceptance record](empty-still-results.json) retains
commands, hashes, numerical results, timings, resource samples and build identities.

The replacement preserves h3cli's dense BF16, 50-block Euler/shifted-flow
sampling and changes delivery to the stock F32 video VAE, five temporal copies
and raw frame 3. It is not a comparison of sampling trajectories with ComfyUI's
`er_sde`/`simple` workflow. See the [contract](single-still-generation-contract.md),
[user guide](single-still.md), and [frozen identities](../../tests/empty_still_contract.json).

## Completed generation coverage

Hardware: Apple M4 Max with 128 GiB unified memory, macOS 27.2.1; NVIDIA RTX
PRO 5000 72GB Blackwell, Ubuntu 24.04.5, driver 595.99.02 and CUDA 13.0.
Each backend passed nine small-canvas cases and five high-resolution cases.
The small matrix covers prompt-only 640×480 and 1344×768, two ordered image
references with `match/high/max`, reversed references, and exact seed repeats.
Repeated trajectories, normalized latents and PNGs match within each backend;
reversing the references changes the latent. Every successful case produces one
callback, no AV/sampler state or audio output, and a byte-identical codec-only
PNG re-decode.

All rows below use seed 42 and two evaluations. Reference rows use one portrait
image with `--ref-image-size high`. Seconds include the generation process,
conditioning, weight loading, sampling and PNG delivery; the subsequent
verification re-decode is excluded. These are functional timings on the tested
machines, not a controlled backend performance comparison.

| Canvas and conditioning | Target rows | Metal seconds | CUDA seconds |
| --- | ---: | ---: | ---: |
| 2144×1216, text | 2,546 | 78.17 | 96.99 |
| 2144×1216, reference | 2,546 | 182.10 | 173.86 |
| 3872×2176, text | 8,228 | 222.08 | 258.55 |
| 3872×2176, reference | 8,228 | 329.87 | 316.86 |
| 2176×3872, reference | 8,228 | 326.84 | 306.63 |

Both backends therefore qualify direct 8,425,472-pixel generation in landscape
and portrait. The 16,777,216-pixel admission boundary has host coverage; it is
not a measured generation limit. Two-step fixtures do not establish quality
or identity fidelity.

## Fifty-step quality observations

CUDA completed both 3872×2176 runs with seed 42, all 50 DiT blocks, 8,228 target
rows, 51 finite latent callbacks and one final image callback. Saved-latent
re-decodes reproduce both PNGs byte for byte. Text generation took 712.66 seconds;
one-reference generation with `high` preprocessing took 1553.07 seconds.

The text prompt requests a polished red ceramic teapot on a pale wooden table
under soft studio lighting. The image shows that subject, coherent lid/handle/
spout and reflections, with substantial empty background and soft tabletop
detail. The reference prompt requests a person in a blue jacket standing in a
sunlit garden. The image shows a bright blue padded jacket and sunlit plants;
the raised hand is partly obscured by leaves. No regular tile grid is visible
in the reviewed previews.

These are two prompt/seed observations, separate from numerical decoder
accuracy and determinism. They establish neither general image quality nor
identity fidelity. Metal's 50-step quality run is unverified; its two-step
functional and direct-8-MP coverage is complete.

## Measured resources

| CUDA phase, 8 MP generation | Text device peak, GiB | One reference device peak, GiB |
| --- | ---: | ---: |
| Reference video-VAE encoder | — | 1.050 |
| Qwen vision | — | 0.742 |
| Qwen text | 2.728 | 3.336 |
| DiT | 37.497 | 38.710 |
| Stock still decoder | 9.269 | 9.269 |

These are instrumented CUDA allocation peaks per context, not additive phases
or a guarantee of the minimum card size. The completed DiT is freed before the
stock decoder loads. Two `max` references at 640×480 raised the DiT peak to
40.252 GiB, illustrating the additional reference cost independently of target
size. The 8 MP stock decoder took about 167 seconds on CUDA and 132 seconds on
Metal, with fixed-size tile scratch.
The 50-step 8 MP runs reached DiT peaks of 38.367 GiB for text and 39.581 GiB
for one `high` reference; stock decoding remained at 9.269 GiB.

Metal's sampled process footprint reached 38.80 GiB for 8 MP text and 40.96 GiB
for portrait-reference generation. Corresponding sampled Metal allocations
reached 39.97 and 42.21 GiB. These overlap in unified memory and must not be
added together. A separate 8 MP codec-only process had a kernel-reported peak
footprint of 10,302,480,160 bytes (9.59 GiB), measured with macOS `footprint`.
Two `max` references at 640×480 reached a 44.07 GiB sampled process footprint.

## Reproduce the independent codec check

Download the pinned `vae.py` and license from the URLs in the identity manifest
into `outputs/empty-still/source/`. The runner rejects a different source hash.
Its tile methods retain the upstream ComfyUI GPL-3.0 license; the source is kept
outside the checkout. The transformer equations are independently implemented
in PyTorch. Install PyTorch and NumPy in a test environment, then run:

```sh
make -j8 bin/still_probe bin/still_tests bin/still_generate bin/still_context
bin/still_probe fixture models/MiniMax-H3 320 288 outputs/fixture.safetensors
bin/still_probe decode models/MiniMax-H3 outputs/fixture.safetensors outputs/native.f32 2
python3 tests/empty_still_reference.py --model models/MiniMax-H3 \
  --source outputs/empty-still/source/vae.py --latent outputs/fixture.safetensors \
  --candidate outputs/native.f32 --out outputs/oracle.json --device mps
# Use --device cuda on CUDA hardware.
```

The fixture uses seed 123 normalized F32 noise. The oracle unwhitens once,
repeats T=1 to five, runs all 36 blocks with five-token positions and suffixes,
selects frame 3, applies the pinned tile assembly, then finalizes RGB. Gates
were selected before native comparisons: MAE ≤0.01 and PSNR ≥35 dB. Small,
seam-crossing and portrait fixtures exercise both overlap axes and edge tiles.
A deliberately double-unwhitened input is retained as a negative control.

Completed comparisons are below. The extra Metal 480×480 case checks three-way
overlaps near adjusted tile edges; that extra size was not run on CUDA. Every
measured row passes the frozen gates.

| Fixture | Metal RGB MAE | Metal PSNR, dB | CUDA RGB MAE | CUDA PSNR, dB |
| --- | ---: | ---: | ---: | ---: |
| 256×256 | 3.504e-7 | 126.08 | 3.926e-7 | 125.15 |
| 320×288 | 2.601e-7 | 128.35 | 2.975e-7 | 127.16 |
| 288×320 | 2.624e-7 | 128.38 | 3.014e-7 | 127.09 |
| 480×480 | 2.686e-7 | 128.19 | — | — |

The double-unwhitening negative control has MAE 0.152786 and PSNR 13.69 dB,
correctly failing both gates. This separates normalization validation from
same-implementation PNG repeatability.
The CUDA guard run passes 7,994 checks under `compute-sanitizer` with zero errors;
Metal passes the same GPU checks, with the supported ASan/UBSan test executable.
Both final native builds pass `make test-current-host test-still-sanitize`,
including 87 still host assertions per build. Linux launcher testing records
15 passes and one skipped `/dev/shm` noexec-mount case: the test machine lacks
that mount configuration, so that case is unverified and is not counted as a
pass. The macOS standalone is an ad-hoc signed development package.

## Reproduce generation and recovery

```sh
python3 tests/empty_still_campaign.py --model models/MiniMax-H3 --out outputs/still-functional
python3 tests/empty_still_campaign.py --model models/MiniMax-H3 --out outputs/still-high --high
python3 tests/empty_still_campaign.py --model models/MiniMax-H3 --out outputs/still-quality --quality
bin/still_context models/MiniMax-H3
python3 tests/empty_still_http.py --model models/MiniMax-H3 --out outputs/still-http
python3 tests/empty_still_models.py --model models/MiniMax-H3 --out outputs/still-minimal
make test-still-host test-current-host test-still-sanitize
```

The campaigns record commands, seed 42, every finite sampler boundary, hashes,
exact PNG and latent shapes, wall time, footprint and phase profiles. They
compare the entire repeated trajectory, normalized latent and PNG for identical
seeds/policies; reverse ordered references; exercise `match/high/max`; and
require byte-identical PNGs from codec-only re-decode. Functional probes use two
evaluations. The quality campaign explicitly uses 50, beyond the default six-
evaluation developer-test ceiling. No output resize or upscaler is involved.

The minimal-model runner creates separate FL2VA and Ref2VA views containing
exactly the 53 planned generation files, with no audio or image VAE. It checks
offline generation, explicit/default model discovery, and byte-identical decode
using only three VAE/configuration files. It also checks missing dependencies
and an unrelated incomplete audio download. Startup audio inventory is optional;
video requests still validate their audio dependencies before conditioning.
Pass `--binary /path/to/h3cli` to repeat these checks against a standalone package.

Both native backends pass [context reuse](../../tests/still_context.c),
[delivery cancellation](../../tests/still_lifecycle.c), and
[allocation/encoder controls](../../tests/still_controls.c). Video→still→video→still
repeats produce identical pixels within each mode. Cancelling transformer load,
sampling or a decoder tile permits a clean retry. Memory denial is recoverable;
three decoder runs retain the same live GPU allocation count. Load/final-frame
cancellation and an encoder failure preserve an existing PNG. These are bounded
recovery probes, not a claim that every possible allocator failure was injected.

## Resource and arithmetic audit

The checked still maximum is 16,777,216 pixels and 8192 per side. Latent spatial
area is at most 65,536; video payload is at most 1,572,864 F32 values. The 8 MP
canvas has 8,228 target visual rows. `h3_latent_canvas` divides by 16 without
adapting the output. Shared video and reference preprocessing caps are intact.
The layout checks row counts before conversion; DiT allocation products use
`size_t`/`uint64_t`. Weight planners include actual text/reference rows and live
activation/scratch estimates on Metal and CUDA. Existing memory admission and
per-block safety checks remain active, including the 110 GB process ceiling.

VAE attention is bounded to five temporal copies of a spatial tile, at most
1,280 patch tokens plus five suffix tokens. Per-tile projection includes the
stock four-pixel temporal factor, but readback selects only frame 3. CPU storage
holds one RGB canvas, a blended bottom strip, a right tail and one tile, plus
RGB24 delivery after the decoder is freed. No full-canvas five-frame RGB
allocation exists. Width/height bounds keep pixel strides and backend indices
within their integer types. Schema validation rejects overflow, nonfinite and
truncated inputs before GPU work. Decoder admission reserves 11 GB plus the
canvas allowance and reports actionable memory errors.

Compatibility binds normalization and recipe, not every weight byte. The frozen
manifest records official model revisions and catalog weight hashes; this
qualification does not reread hundreds of GB merely to hash model weights.

## Builds and integration

Final macOS and Linux source/package snapshots have the same implementation and
test fingerprint, `7d8bc444805c…`, using the immutable regression runner's source
inventory. The production-file fingerprint is `c875242e0177…`. Campaign binaries
were staged during implementation; the final gates qualify the integrated
change. Documentation completion notes are written after execution.

| Final regression | Golden outputs | Test seconds |
| --- | ---: | ---: |
| Native CUDA source build | 204/204 | 97.90 |
| Linux standalone package | 204/204 | 122.25 |

The packaged render suite passes 19 cases and seven byte-identity comparisons:
still, AV, minimal AV, sampler and upscale round trips, plus online/offline
server and server/CLI equality. It covers first/last anchors, mixed image/video/
audio references, both continuation modes, preview decoding and conditioning
replay. The separate current CUDA suite passes all 18 checks: its reference,
four-segment continuation and sampler-resume batch; attention/precision policy
matrix; SOL/SubBlock and conditioning contexts; weight placement and cache
lifetime; GPU guards and memory; and decoder cancellation/failure recovery.
Injected allocation failures recover with bit-identical results. LoRA filesystem
coverage runs on XFS without skipped clone tests.

Native HTTP jobs on both backends report one still PNG, one frame, zero frame
rate and zero audio fields, with no AV/audio artifact. The render-script host
suite also passes. Additional dry runs verify that global `--ref-image-size
match|high|max` reaches both segment commands unchanged; the CLI parser assigns
that value to the pipeline's shared reference-size parameter. `--render` remains
a video-script operation.

Both final standalone packages pass offline FL2VA and Ref2VA generation from
53-file model trees without audio or image VAEs. Their codec-only trees contain
just three files per variant and reproduce generated PNGs exactly. The macOS
and Linux standalone 8 MP re-decodes also match their respective native PNGs
byte for byte. Missing VAE files and video requests without audio dependencies
fail; unrelated partial audio downloads do not break still startup.

| Standalone | Executable SHA-256 prefix | Accepted operations |
| --- | --- | --- |
| macOS arm64 | `61fc6a51b16f…` | Offline base/reference generation, minimal codec decode, 8 MP decode |
| Linux x86_64 | `135407cc74a5…` | Same still operations, 204-output regression, render/server/current-feature suites |

Full identities and individual checks are in the JSON record. The tested
packages are retained in `outputs/empty-still/macos-package-accepted/` and
`outputs/empty-still/linux-package-accepted/` on their respective test machines.

## Task reconciliation

| Tasks | Acceptance evidence |
| --- | --- |
| ELS001–002 | [Frozen sources and thresholds](../../tests/empty_still_contract.json), [version 2 contract](single-still-generation-contract.md), explicit API/schema migration |
| ELS003–005 | Host layout/geometry checks, direct 8 MP target-row assertions, resource and arithmetic audit above |
| ELS006–008 | Independent five-token/frame-3 comparisons, edge/overlap fixtures, single-frame delivery and decoder lifetime checks |
| ELS009–012 | [Minimal-model runner](../../tests/empty_still_models.py), schema/CLI/model-resolution tests, HTTP still results, retired dedicated decoder and preserved historical records |
| ELS013–014 | Final host suites, Metal sanitizer/GPU guards, CUDA compute-sanitizer, seven independent decoder comparisons and normalization negative control |
| ELS015–016 | Fourteen functional cases per backend, both 8 MP orientations, two 50-step CUDA quality cases, exact repeated trajectories and re-decodes |
| ELS017 | Reused contexts, load/sampling/tile/frame cancellation, memory denial, allocation plateau, encoder failure and clean retry |
| ELS018 | Unchanged 204-output source/standalone gates, 19 packaged render cases, seven round trips, 18 current CUDA checks |
| ELS019 | Updated README, usage, model/hardware guides, [still guide](single-still.md), contract and this acceptance record; changed links and privacy checked |
| ELS020 | Matching source/test fingerprints, both standalone builds, offline minimal-model generation and 8 MP codec acceptance |

## Retained evidence

Raw commands, profiles, PNGs, latents, independent reference arrays and private
machine records are retained under ignored `outputs/empty-still/` on the two
test machines. The CUDA run uses an isolated checkout; existing work in the
shared checkout is preserved. Published records omit private hostnames and paths.
Historical dedicated-VAE oracle/campaign scripts remain with their archived
reports; they require their original implementation. Current qualification uses
the `empty_still_*` runners and updated native probes listed above.

The immutable CUDA manifest remains unchanged. The first implementation gate
passed all 204 outputs in 140.9 seconds, the PNG-limit correction passed in
128.34 seconds, and the final source passed in 97.90 seconds. The final Linux
standalone also passes all 204 outputs in 122.25 seconds, and its separate
current-feature suite passes all 18 checks. Historical dedicated-image-VAE
results remain in [the original report](single-still-results.md) and
[original CUDA report](single-still-cuda-results.md).
