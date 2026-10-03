# CUDA weight placement

The SGLang cookbook and layerwise manager were consulted at upstream revision
[`b2800461e5bd5967f4aa00cbc8ab7337757f1867`](https://github.com/sgl-project/sglang/tree/b2800461e5bd5967f4aa00cbc8ab7337757f1867).
Their placement ideas inform this design; their resident-layer counts and timing
claims are not portable to this implementation or comparison matrix.

The placement plan is context-owned execution state. It changes weight storage
and transfers, never the numerical recipe, block execution order, sampler state,
or decoder arithmetic. Metal retains its existing policy.

## Admission

`resident` requires every eligible matrix to fit; `stream` retains no core
matrices; `auto` first tries full residency, then the largest safe leading set
of active BF16 blocks. Explicit SSD streaming implies zero residency and
conflicts with explicit `resident`. Options are validated and captured per
request, including test-only capacity reductions. Later environment changes
cannot change an existing GPU context.

BF16 admission uses free device bytes after text refinement, AdaLN precomputation,
layout maps and adaptive buffers. Those live allocations are not charged twice.
The remaining budget subtracts future activations, small core weights and
library scratch, plus `max(1 GiB, total VRAM / 10)` safety reserve. Exact equality
fits. All additions/multiplications must reject overflow. Full residency needs
no streaming slots; every partial or zero-resident BF16 plan reserves two whole
matrix slots before admitting persistent blocks. Each block contains four BF16
matrices totaling 770,703,360 bytes (735 MiB); all 50 use 35.889 GiB.

The packed paths remain all-resident or streamed. Their earlier admission uses the
actual padded descriptor sizes, includes a BF16 block-0 probe for adaptive
caching, and reserves preparation and conversion/GEMM scratch. There is no partial packed cache.

An automatic allocation failure tears down the complete failed attempt and
retries with a strictly smaller resident-count ceiling, halving until zero.
The final streamed attempt is made only once. Explicit resident requests and
I/O, cancellation, policy or arithmetic failures never trigger this fallback.

## Scheduling and lifetime

The leading active blocks own persistent matrices. Only other blocks own stream
descriptors and pinned host cache entries. Small norms remain independently
owned. The existing two slots retain upload-ready and last-consumer events.
Prefetch finds the next nonresident active block, skipping resident gaps. Slot
reuse waits for its last consumer on the copy stream; computation waits for its
upload on the compute stream. No device-wide fence is added to steady execution.

At evaluation boundaries, prime the next streamed block only when execution
requires it. Never upload the next evaluation merely because the previous one
finished: completion, pause, core reuse and adaptive hits can make that upload
unnecessary. Adaptive suffix prefetch starts after the block-0 decision.

Resident matrices, slots, mappings and prefetch jobs have one DiT owner. Join
jobs before returning an execution error. Teardown drains the GPU streams before
releasing registered memory. Retained contexts are invalidated by placement
options or insufficient current headroom as well as existing geometry,
precision, model and LoRA keys. Reload and saved-state resume rebuild placement;
no placement fields are serialized. Constrained automatic contexts are replanned
on each new request so admission can respond to both rising and falling available
capacity. Existing decode headroom checks may evict
the DiT. Upscaling loads a new target-geometry plan.

## Accounting and qualification

The planner logs requested/effective mode, resident count, exact matrix and slot
bytes, free/live bytes, future estimate, reserve, test caps and fallback reason.
Existing CUDA events provide upload and exposed-wait measurements when profiling
is enabled. Host-cache logical reads are reported separately from process
physical I/O counters. Pinned memory retains the existing 40 GiB cap and host
reserve; resident blocks do not acquire duplicate host cache entries.

The [comparison manifest](weight-residency-matrix.json) fixes the primary runs.
Capture preparation, velocities and latent trajectories and compare bitwise;
compare decoded RGB/PCM independently of file headers. Supplement these with
planner boundaries, real-weight slot/failure probes, composition and retained
context tests, the unchanged 204-output golden gate and local Metal tests.
Performance measurements follow a passing gate and use three interleaved pairs.
Natural admission and deliberately capped partial plans are reported separately.
