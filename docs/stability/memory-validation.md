# Memory implementation validation

Validated on **Apple M4 Max, 128 GiB**, using baseline commit
`b162b9fd631b1bd57ddeee248411d249c268747e`. The implementation covers T001–T048
from `todo.md` using `design-memory.md`. No M5 validation was requested.

## Guards, cancellation and sequence limits

| Area | Validation |
|---|---|
| Physical-memory API | 25 checks: exact floor/reserve boundaries, overflow, overridden floor, query failure, diagnostic contents, and user/memory cancellation. Also compiled with the macOS backend disabled to verify the portable fail-open path. |
| Qwen presentation prediction | Existing tokenizer conformance plus 236 presentation checks, including exact predicted counts and metadata-only vision descriptors. ASan/UBSan passed. |
| Runtime GQA limit | 15,907 checks. M4 device budget 32,768 bytes, static usage 1,024 bytes, 16-byte dynamic alignment, maximum 7,936 tokens. Actual GQA dispatches at 7,935 and 7,936 succeed; 7,937 fails without dispatching. A missing-weight-store test confirms preflight precedes opening Qwen weights. |
| Model cancellation | Instrumented entries in real model execution verify that neither user nor memory cancellation enters/enqueues the next Qwen layer, vision block, audio/video VAE block, video-encoder block, or DiT step. Covers CPU Euler, GPU Euler and RES, resident decoder reuse, and populated generation-cache cleanup. Normal and ASan/UBSan runs pass. |
| Issue #47 configuration | A 360-frame, 1280×720 body-reference video requests 13,337 tokens after normalization. It is rejected at 7,936 before video VAE, vision, text transformer or DiT execution. No output video is created. |
| Safe long-reference stress | The same long input is decoded with live macOS headroom measurements and an increased test floor. Cancellation occurs during decoding, with all tracked allocations freed and the child reaped. Physical headroom remains above 100 GiB. This tests the guard without approaching actual OOM or physically exhausting the normal 10-GiB reserve. |

Execution counters are injected only into isolated test source copies. They count
entries into the actual model functions/step bodies, including GPU command
encoding. Production has no counters. Cancellation waits for already committed
work and discards uncommitted commands; it cannot undo a running kernel.

The cleanup audit covers Qwen's prefetched-layer ring and worker joins, vision
mergers and partial outputs, DiT's step-local arrays and command chains, video
VAE resident/nonresident paths, audio stages, encoder tiles/chunks, and generation
cache ownership. The cache test verifies a populated conditioning/DiT cache is
empty after cancellation and reuses the same generation context. The resident
video-decoder test completes a decode after two cancelled decodes.

## Decoder parity and allocation accounting

**86 decoder runs pass**, both normally and with ASan/UBSan. Paired decodes use
the actual pre-change implementation from the pinned commit and compare frame
counts, logical byte counts, and every F32 byte. Inputs cover one frame, short
clips, a 15-second clip, a 30→24-fps conversion, and caps below/equal/above source
length, for normalized and legacy cadence paths.

Error cases include malformed/missing input, partial-frame EOF, empty output,
FFmpeg spawn/exit failures, short reads, injected read errors, both allocation
failures, overflowing products/reserves, excess output, cancellation during
reading and child completion, and a child that ignores SIGTERM. The probe checks
that no decoder allocations or child processes remain after failure.

Measured allocation for 360 frames at 48×32 RGB:

| Decoder | Peak decoder-owned allocation |
|---|---:|
| Before: full RGB plus final F32 | 8,294,400 bytes |
| After: one RGB frame plus final F32 | 6,640,128 bytes |

This is a **19.94% reduction** at the same cap and output length. The new staging
allocation is 4,608 bytes regardless of clip length. Instrumentation counts
requested allocations inside the decoder, not FFmpeg's memory or whole-process
RSS. Short clips retain the final allocation's cap-sized capacity; their logical
output remains tightly packed. See [the memory guide](memory.md) for this limit.

## End-to-end regressions

All runs use seed 72 and all 50 DiT blocks. References are `inputs/face1.jpg`,
`inputs/2.jpg`, and two videos made from `inputs/body1.jpg`. The steps below are
reduced for numerical regression; these videos are not quality benchmarks.

| Case | Output | Steps | Text handling | Result |
|---|---|---:|---|---|
| Small FL2VA face anchor | 32×32, 22 frames | 4 | Fresh on both builds | Bit-identical pixels, conditioning, text, final AV latents, decoded RGB/PCM and MP4 |
| FL2VA face anchor | 256×256, 56 frames | 4 | Fresh text captured; baseline text used for downstream comparison | Same exact comparison passes |
| Ref2VA image plus 2-second video | 256×256, 56 frames | 4 | Fresh text captured; baseline text used for downstream comparison | Same exact comparison passes |
| Ref2VA 15-second video | 256×256, 362 frames | 2 | Fresh text captured; baseline text used for downstream comparison | Same exact comparison passes |

The initial fresh 256×256 FL2VA comparison matched pixels, visual VAE conditions,
IDs, positions and spans, but differed in Qwen text and consequently the rendered
output. Repeating the **unchanged baseline binary** also changed Qwen text and
output while preserving visual conditions. This reproduces the cold multimodal
variability already documented in [continuation acceptance](continuation-acceptance.md#ordinary-generation-regression-investigation).

Normal-resolution regression builds therefore encode and save fresh text, then
replace **only the text embedding** with the captured baseline embedding before
DiT. All reference pixels and visual conditions are freshly computed and checked
exactly. This establishes downstream invariance and unchanged video conditioning;
it does **not** claim arbitrary cold Qwen runs are bitwise deterministic. The
small FL2VA pair requires no replay and passes the complete fresh comparison.
The original differing renders and unchanged-baseline repeat are retained.

## Reproduction and evidence

Commands are listed in [memory.md](memory.md#tests-and-reproducibility).
The complete `make test` suite passes. Ten pre-existing optional MLX fixture
groups are absent and are explicitly skipped; the new model cancellation and
generation tests run independently against the installed released weights.

[Measured manifest](memory-validation.json) records commands, matching hashes,
binary hashes, timings, stress measurements and the unchanged-baseline repeat.
Detailed local artifacts are in:

- `outputs/memory-validation/decoder/` and `decoder-sanitize/`
- `outputs/memory-validation/cancellation/` and `cancellation-sanitize/`
- `outputs/memory-validation/long-reference/`
- `outputs/memory-validation/generation/`

`make test-memory`, `make test-memory-gpu`, `make test-memory-models`, and
`make test-memory-sanitize` are the new focused targets. `make test` includes the
host memory and actual-device GQA tests. The callback API and user-facing failure
behavior are documented in [memory.md](memory.md).
