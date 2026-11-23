# Extended CUDA canvas contract

Baseline frozen on 2026-11-22 at source revision
`2b78e5944d7d7903983710d981c29d7401da81a6`. Implementation and acceptance are
tracked in [the archived checklist](../large-canvas-todo.md).
The learned network retains
[its pinned artifact, source and numerical tolerances](../../tests/upscale/contract.json).
No model weight, normalization constant, attention reduction or golden output
is replaced by this extension.

| Serialized profile | Meaning | Geometry | Backends |
| --- | --- | --- | --- |
| 0 | Legacy ordinary video | Existing 1,032,192-pixel ceiling and legacy axes | CUDA, Metal |
| 1 | Existing explicit upscale policy | 2,088,960 pixels, each axis ≤1920 | CUDA, Metal |
| 2 | Extended ordinary video/source | 2,088,960 pixels, each axis ≤1920 | CUDA |
| 3 | Extended learned 2× output | 8,355,840 pixels, each axis ≤3840 | CUDA |

Generation dimensions remain multiples of 32. Learned output doubles both
source axes, so its dimensions are multiples of 64. Portrait transposes have
the same area and axis checks. Existing even-sized codec presentation canvases
remain valid. `H3_MAX_PIXELS`, still-image policy and reference-media adaptation
retain their previous meanings.

The ordinary CLI/API selects profile 2 when a requested canvas exceeds the
legacy area. Explicit width/height HTTP requests and `--render` use that policy.
The pinned SGLang compatibility transport retains its established `short_edge`
adaptation; use explicit native dimensions to request the extended canvas.
Fresh ordinary generation cannot select profile 3. The upscale planner selects
0, 1 or 3 according to the doubled output; native profile 2 cannot claim an
upscale refinement recipe.

New profiles require dense BF16, all 50 blocks and default attention. Quantized
denoising, approximate attention, reuse/token-reduction/adaptive modes and
runtime LoRA are unqualified and rejected. Native text, first/last anchors,
ordered supported references, preview decoding, full decoding and checkpoint
resume retain their existing contracts. Upscaling retains K=0/2/3/4, full VAE
only, unchanged time and audio, and the existing source-capture restrictions.
`--ref-image-size match|high|max` controls reference preparation independently
of the output canvas. High/max keep their stored intrinsic size during transfer.

AV schema 3 stores the explicit profile. Sampler/source schema 2 uses required
section 46, version 1, for nonzero profiles. Existing readers reject unknown
profiles; no old profile is reinterpreted. Profile 2 source bundles are portable
without the original reference files. Profile 3 sampler states require a
refinement record, and delivered profile 3 AV requires upscale presentation
provenance. Metal rejects profiles 2/3 before geometry-dependent allocation;
its existing profile 1 upscaler remains available.

`src/geometry.h` provides checked geometry, row and phase estimates. The CUDA
dense sequence limit is 1,000,000 rows, matching the flash-attention interface
and remaining below the 65,535-block Y grid of 16-row elementwise launches.
QKV uses the actual width 3×7168; its CUDA dense
kernel uses `size_t` addresses. cuBLAS dimensions are checked against signed
32-bit limits. An MLP that would cross U32 FC1 elements uses independent
131,072-row slabs, with complete dot-product K dimensions. Smaller requests
retain their previous GEMM shapes. Global attention and full-context learned
network normalization remain intact. Metal/diagnostic indexing bounds are
checked separately. The text refiner separately checks its U32 residual-add
element count. Continuation must retain the matching native geometry profile;
a mismatch is rejected during request validation.

Admission includes actual text/reference/audio rows. Device weight planning
uses the shared conservative DiT estimate and current free memory. Host
estimates separately cover sampler copies, the optional 40 GiB weight cache,
retained decode tiles, a chunk plus temporal overlap, and encoder buffers.
Decode host storage is bounded by chunk geometry, not total movie duration.
For a 20-tile axis the CUDA stitcher selects the bounded host stitcher before
accessing its 16-entry device arrays. The original tile blend order is retained.

Reproduce the [memory table](large-canvas-memory-estimates.md) without models:

```sh
make test-large-canvas
bin/large_canvas_tests --report
```

Development runners use an exclusive GPU lock, 22-frame defaults, explicit
streaming, a 60 GiB capacity ceiling and separate current-free GPU/host checks.
They sample whole-device usage and process-tree RSS (with possible shared-page
double counting). See the [acceptance record](large-canvas-results.md). A 39-frame case requires its
own admission. Continuation is restricted to a 39-frame source/56-frame target
at at most 1344×768 or transpose. Saved-state replay uses the same inspection
guards. Independent script segments must deliver at most 72 combined frames.
The 124/243/362-frame cases are host arithmetic and metadata checks only.

The immutable CUDA golden manifest remains
`fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3`.
Its 204-output gate was initially deferred under the duration limit, then
explicitly requested by the user and passed **204/204** on 2026-11-22 using the
unchanged 124-frame case. See the [follow-up evidence](large-canvas-results.md#full-immutable-regression-follow-up).
The still milestone
archive remains byte-identical, SHA-256
`5131acf0e6c3a847374487ec77453d8c4850715682750497704c3c1c8a362166`.

Use `make test-large-canvas-host` for this campaign. Do not invoke
`test-current-host`, `test-render-media`, `test-refvideo`, `test-current-cuda`,
the old 90-frame upscale campaign, or the original
multi-segment render integration without auditing their durations. The first
host-suite invocation in this work inadvertently ran the existing long
synthetic FFmpeg fixtures (including 361/362-frame clips and long merges).
That violated the campaign duration cap; those runs are excluded from bounded
acceptance. The dedicated host target excludes both synthetic-media runners.
Real-model extended CUDA acceptance uses only the guarded short cases. The
separate, explicitly authorized immutable gate retains its original duration.
