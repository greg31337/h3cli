# Memory guards and long references

H3 checks process footprint, Metal reservations, and reclaimable physical-memory
headroom before allocations, Metal encoding/submission, and model boundaries.
The default process/Metal limit is **110 GB decimal** (110,000,000,000 bytes,
102.45 GiB), reduced on smaller machines to leave at least one eighth of RAM or
10 GiB free. `H3_MEMORY_LIMIT_BYTES` can lower this cap, never raise or disable it.
The separate physical-headroom floor remains **10 GiB**, centralized in
`src/memory.h`. On macOS the query uses `host_statistics64(HOST_VM_INFO64)` and
`host_page_size`: `(free_count + inactive_count) * page_size`. Speculative pages
are not added again. This is a conservative unified-memory estimate, not a
separate GPU-memory budget.

Known reference-video allocations require **floor + F32 destination capacity +
one RGB24 frame** before allocation. Metal buffers and major MPSGraph operations
also check their known allocation or conservative workspace estimate. Footprint
includes compressed process memory. Footprint and Metal reservations overlap,
so the guard compares their maximum rather than adding them together.
Comparisons avoid overflowing when a requested reserve is very large. Errors
name the phase and include measured available bytes, configured floor, and
requested allocation reserve. Failed OS headroom or footprint queries stop the
generation through normal error cleanup. The previous headroom-only, fail-open
policy did not enforce a 110 GB process limit.

Checks run during reference-video reads, after decoding, before vision/text/DiT,
and at Qwen, vision, DiT, audio/video VAE, and reference-encoder boundaries.
They are boundary checks, not a memory reservation: another process or a single
large block can change headroom between checks. This is not an OS-enforced quota.
Long Metal sequences use bounded command batches; completed command objects are
retired, and a failed workspace reservation first attempts to drain safe queued
work and recheck. Work that is already over budget is not submitted.

## Cancellation

All seven internal long-running progress callback types now return `int`:
Qwen text, Qwen vision, DiT, AdaLN preparation, video VAE, audio VAE, and video
encoder. **0 continues; nonzero cancels.** Existing callers defining `void`
callbacks must return 0 after their notification work.

Generation bridges check memory first, retain the first cancellation diagnostic,
notify the public callback, and propagate cancellation to the active subsystem.
The next model block or denoising step does not begin after a cancellation gate.
Video encoder callbacks also poll between CNN blocks while reporting the same
completed-tile count; callers must permit repeated counts and an initial 0/0
notification while loading. Video decoder block counts restart for each tile.

Cleanup joins Qwen prefetch workers, discards uncommitted Metal commands, waits
for already committed commands, and frees temporary tensors, weights, and host
buffers. Cancelled generations and memory-guard failures release their retained caches.
Completed Metal production renders also release the prepared DiT before final
decoding. Explicit live previews can load a VAE earlier while DiT is still
needed; the same budget checks apply to that overlap. Resident video
VAE decode/preview callbacks belong to each invocation, so cached decoders do
not retain callback pointers into a previous generation's stack. The original
callback-free resident decode/preview entry points remain available.

## Causal-GQA limit: a different constraint

The current causal-GQA kernel needs one F32 score per text/presentation token in
threadgroup memory. Having ample physical memory does **not** increase that
limit or resolve issue #47. The runtime queries the active device and compiled
pipeline, and both preflight and direct dispatch use:

```
score_bytes = align_up(sequence_length * sizeof(float), 16)
score_bytes + pipeline.staticThreadgroupMemoryLength
    <= device.maxThreadgroupMemoryLength
```

The alignment follows Apple's [`setThreadgroupMemoryLength` requirements](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setthreadgroupmemorylength(_:index:)).
No global token limit is hard-coded. The tested M4 Max reports 32,768 device
bytes and 1,024 static bytes, supporting **7,936 tokens**. Other devices/pipelines
are queried independently.

An exact presentation-count pass reuses the real tokenizer/presentation builders
with reference dimensions, decoded frame counts, and timestamps before visual
VAE or Qwen vision execution. The final count is checked again at entry to Qwen
text, before opening its weight store or allocating activations. Errors report
the requested length, computed maximum and device budget, and recommend reducing
reference length/resolution/count or prompt length until tiled GQA is available.
The text preflight conservatively requires the direct-kernel limit even when the
MPS GQA alternative is enabled, since its fallback uses that kernel.

## Reference-video decoder allocation

The decoder allocates the bounded F32 destination up front and reuses **one RGB24
frame**. Short pipe reads accumulate into that frame, then conversion writes
immediately into channel-major `[3, T, H, W]` storage using the original
`float(byte) * (1.0f / 255.0f)` expression. Short clips and legacy cadence trimming
compact channel strides in place with `memmove`, preserving every returned
sample. No second full F32 buffer is created.

For a clip filling its frame cap, decoder-owned allocation changes from
`RGB_frame_bytes * T + F32_video_bytes` to
`RGB_frame_bytes + F32_video_bytes`—nearly **20% lower peak** for long clips.
FFmpeg's separate process and codec buffers are outside this accounting.
The returned logical frame count and byte count still describe only decoded,
retained frames. The destination keeps its original capacity for shorter clips;
therefore this formula is not a claim that every short clip uses less memory
than before when requested with an oversized cap.

Clean EOF, partial-frame EOF, read failures, spawn/exit failures, and memory
cancellation follow normal cleanup. One sentinel byte checks for output beyond
`max_frames`; no additional frame is converted or retained. Pipe and child waits
continue checking memory, including when FFmpeg stalls. Cancellation terminates
and reaps the child, with a bounded grace period before forced termination.

The F32 reference remains fully resident after decoding. A future chunked
reference representation could further reduce memory; this change only removes
the full-video RGB staging buffer.

## Tests and reproducibility

```
make test-memory           # host guard + old/new decoder parity and allocation tests
make test-memory-gpu       # real device limit and below/at/above dispatch tests
make test-memory-models    # real installed weights, instrumented execution counters
make test-memory-sanitize # ASan/UBSan decoder tests
python3 tests/memory_cancellation.py --sanitize
python3 tests/memory_long_reference.py
python3 tests/memory_generation.py
make test
```

Metal/model tests need access to the Apple GPU. Generation comparisons use all
50 DiT blocks and reduced denoising-step counts; they test numerical regression,
not visual quality. Normal-resolution comparisons record fresh Qwen text and
replay only baseline text in isolated builds to control documented cold Qwen
variability. Reference pixels and visual VAE conditions are always fresh.
Production binaries contain no replay or execution-tracing instrumentation.

`H3_TEST_MIN_AVAILABLE_MEMORY_BYTES` overrides the floor for deterministic tests;
unset it for normal use. `h3_memory_set_test_query()` installs a thread-local test
query provider; NULL restores the platform query. The safe stress test uses live
macOS measurements and raises the test floor above current headroom. It does not
exhaust physical memory to reach the real 10-GiB floor.

Historical results are in [memory-validation.md](memory-validation.md). The
new cap, scheduling and 362-frame measurements are in
[the M5 report](../metal/m5-memory-results.md).
