## Memory Safety and Long-Reference Stability

### Objective

Improve stability when processing long image/video references by addressing four independent but related failure modes:

1. abort generation before macOS reaches critically low unified-memory headroom;
2. make cancellation propagate promptly through long-running model execution loops;
3. detect the current Metal GQA sequence-length limitation before expensive reference processing proceeds;
4. reduce avoidable peak memory during reference-video decoding by eliminating the simultaneous full-video RGB8 and F32 buffers.

This phase intentionally does **not** redesign the Metal GQA kernel. The sequence-length limitation described in issue #47 remains until a later tiled/online-attention implementation is added. The objective here is to fail safely and early while reducing avoidable memory pressure.

---

## 1. Dynamic 10-GiB Memory Guard

### Goal

Abort generation cleanly whenever reclaimable physical-memory headroom drops below 10 GiB.

The guard should primarily protect against unified-memory exhaustion, macOS memory-pressure escalation, process termination, system instability, and GPU allocation failures.

It is not intended to detect the Metal GQA threadgroup-memory limit; that is handled separately by the sequence preflight.

### macOS Memory Measurement

On macOS, query VM statistics through Mach APIs using:

```text
host_statistics64()
HOST_VM_INFO64
```

Use a conservative approximation of immediately reclaimable physical memory:

```text
available =
    (free_count + inactive_count) * page_size
```

Do not independently add `speculative_count` if it is already represented in `free_count`.

Define:

```text
H3_MIN_AVAILABLE_MEMORY = 10 GiB
```

The check should be centralized in a reusable helper rather than duplicating platform-specific code throughout the execution pipeline.

Suggested interface:

```c
int h3_memory_available_bytes(uint64_t *bytes);

int h3_memory_check(
    uint64_t reserve_bytes,
    const char *phase,
    char *error,
    size_t error_size);
```

`reserve_bytes` represents an optional known upcoming allocation.

The effective requirement should be:

```text
available >= 10 GiB + reserve_bytes
```

when the next operation has a predictable large allocation.

For generic periodic checks:

```text
reserve_bytes = 0
```

### Failure Behavior

If available memory drops below the threshold:

* mark the current generation as cancelled;
* return a normal H3 error through existing error propagation;
* allow all currently allocated resources to follow the normal cleanup path;
* do not call `abort()`, `_exit()`, or otherwise bypass cleanup;
* do not continue into another model layer or denoising iteration.

Suggested diagnostic:

```text
Generation aborted during Qwen text encoder:
8.7 GiB reclaimable physical memory remains;
minimum safety reserve is 10.0 GiB.
```

M5 supersedes the original fail-open rule: failed VM/footprint queries stop the
generation through normal cleanup. Enforce a default 110 GB decimal cap on the
maximum of process footprint (including compression) and Metal reservations,
in addition to the headroom floor. Reduce the cap on smaller machines and allow
only downward overrides through `H3_MEMORY_LIMIT_BYTES`. Reserve known upcoming
allocations before encoding and recheck at command submission. Bounded command
lifetimes reduce exposure between checks; the guard is not an OS quota.

### Check Locations

Perform memory checks at expensive phase boundaries and inside long-running loops.

At minimum:

```text
before reference video decoding
after reference video decoding
before each Qwen vision block
after each Qwen vision block
before multimodal text encoding
between Qwen text layers
before DiT initialization
between DiT denoising steps
before video VAE execution
between video VAE blocks where practical
before audio VAE execution
between audio VAE blocks where practical
```

Checks should be inexpensive relative to the computation being performed.

---

## 2. Cancellation Propagation Through Inner Loops

### Problem

Some current internal progress callbacks return `void`, which means a cancellation request can be recorded by the outer generation context but ignored until an entire subsystem completes.

This is unacceptable for the low-memory guard because the process may continue allocating memory for seconds or minutes after the guard decides generation should stop.

### Design

Change internal progress/cancellation callbacks from notification-only semantics to cancellation-aware semantics.

Preferred convention:

```c
typedef int (*h3_*_progress)(
    int completed,
    int total,
    void *opaque);
```

Return:

```text
0 = continue
non-zero = cancel
```

Each long-running inner loop must check the callback result before beginning the next expensive iteration.

This applies to at least:

```text
Qwen vision
Qwen text encoder
DiT denoising
video VAE
audio VAE
video encoder, where used
```

The bridge from internal progress callbacks to the public H3 progress mechanism should:

1. check available memory;
2. set the generation cancellation state if the threshold is violated;
3. invoke the existing external progress callback;
4. propagate either memory-triggered or user-triggered cancellation back into the inner loop.

Conceptually:

```c
if (!h3_memory_check(...))
    progress->cancelled = 1;

h3_progress_emit(...);

return progress->cancelled;
```

Every subsystem must leave through its normal cleanup path after cancellation.

### Granularity

Cancellation checks should happen at natural computational boundaries rather than inside individual Metal kernels.

Recommended granularity:

```text
Qwen text       once per transformer layer
Qwen vision     once per vision block/layer
DiT             once per denoising step
video VAE       once per major block
audio VAE       once per major block
```

This keeps overhead negligible while guaranteeing reasonably prompt termination.

---

## 3. GQA Sequence-Length Preflight

### Problem

The current Metal causal GQA implementation requires a score array proportional to sequence length in threadgroup memory.

The maximum sequence length is therefore constrained by:

```text
maxThreadgroupMemoryLength
```

and by any static threadgroup memory already required by the pipeline.

Issue #47 demonstrates that sufficiently long multimodal sequences can exceed this device limit.

The current failure occurs too late: reference processing, vision encoding, and other expensive operations may already have consumed substantial time and memory before the first incompatible GQA dispatch is attempted.

### Design

Expose a GPU helper that computes the maximum sequence length supported by the current causal GQA kernel on the active Metal device.

Suggested interface:

```c
uint32_t h3_gpu_gqa_causal_max_sequence(h3_gpu *gpu);
```

The calculation must derive the value from the actual pipeline/device properties rather than hard-coding a limit such as 8192.

Conceptually:

```text
available_dynamic_threadgroup_memory =
    device.maxThreadgroupMemoryLength
    - pipeline.staticThreadgroupMemoryLength

max_sequence =
    available_dynamic_threadgroup_memory / sizeof(float)
```

Include any alignment or additional dynamic-threadgroup requirements actually used by the kernel.

### Preflight Location

Run the check as soon as the final multimodal token count is known and **before expensive Qwen text execution begins**.

Preferably, perform it even earlier if the multimodal sequence length can be predicted reliably before creating all vision embeddings.

The preflight should compare:

```text
presentation token count
```

against:

```text
maximum supported causal-GQA sequence
```

and return a controlled error if unsupported.

Suggested error:

```text
Qwen multimodal sequence contains 10,231 tokens,
but the current Metal causal-GQA kernel supports at most
8,0xx tokens on this GPU.

Reduce the reference length/resolution or use a future
tiled-GQA implementation.
```

The exact maximum should be reported.

### Scope

This work does not alter generated output and does not remove the sequence limitation.

Its purpose is to convert a late GPU/runtime failure into an early, deterministic, understandable validation error.

---

## 4. One-Frame RGB Staging for Reference Video Decode

### Problem

The current `h3_ffmpeg_read_video_f32()` implementation temporarily holds:

```text
entire video in RGB8
+
entire video in F32 channel-major form
```

during conversion.

For long references this creates a significant and unnecessary transient memory spike.

### Target Architecture

Replace:

```text
FFmpeg
  ↓
full RGB8 video buffer
  ↓
full F32 video buffer
  ↓
free RGB8 buffer
```

with:

```text
FFmpeg
  ↓
one RGB8 frame staging buffer
  ↓
immediate conversion into final F32 destination
  ↓
reuse staging buffer for next frame
```

The final F32 representation remains unchanged.

### Allocation Strategy

After dimensions and maximum frame count are known:

1. allocate the final F32 buffer;
2. allocate one RGB24 frame buffer;
3. read one complete frame from FFmpeg;
4. convert that frame immediately into its destination position in the F32 buffer;
5. reuse the RGB buffer for the next frame;
6. stop on EOF or `max_frames`;
7. shrink or logically report the final F32 buffer based on actual frame count.

Peak decode memory becomes approximately:

```text
full F32 video + one RGB frame
```

instead of:

```text
full F32 video + full RGB video
```

### Layout Compatibility

The resulting F32 buffer must remain byte-for-byte compatible with the existing downstream layout.

If the current output layout is:

```text
frame-major / channel-major / spatial
```

or another specific ordering, the streaming conversion must populate exactly the same indices that the current post-decode conversion loop produces.

No downstream API should need to change.

### Error Handling

Handle partial reads correctly.

A single frame must not be considered valid until exactly:

```text
width * height * 3
```

RGB bytes have been read.

On:

```text
EOF before first byte of a new frame
```

finish successfully.

On:

```text
EOF after only part of a frame
```

return a decode error or explicitly discard the incomplete final frame according to existing FFmpeg behavior.

All failure paths must free both:

```text
F32 destination
RGB staging frame
```

and close the pipe/process normally.

### Memory Preflight

Before allocating the final F32 video buffer, calculate:

```text
estimated_f32_bytes =
    max_frames *
    width *
    height *
    3 *
    sizeof(float)
```

and call the memory guard with that value as `reserve_bytes`.

Where frame count is not known precisely in advance, use the configured maximum as the conservative reservation estimate.

---

## Compatibility Requirements

The implementation must preserve:

```text
existing output for successful generations
existing reference-video numerical values
existing public API behavior
existing progress reporting
FL2VA behavior
Ref2VA behavior
continuation behavior
```

except that previously catastrophic or late failures may now terminate earlier with explicit errors.

The video-decoder rewrite must produce numerically identical F32 pixel values for the same FFmpeg input.

The GQA preflight must not reject sequences that the existing kernel can successfully execute.

---

## Validation

### Memory Guard

Test using configurable thresholds so the abort behavior can be triggered without actually exhausting system memory.

For example, allow tests to override:

```text
10 GiB
```

with an artificially high value.

Verify:

```text
generation exits normally
error message identifies the phase
no subsequent layer/step executes
allocated resources are freed
process remains alive
```

### Cancellation

Add synthetic cancellation tests for every affected subsystem.

Request cancellation after a known iteration and verify that execution stops before the next expensive iteration.

### GQA Preflight

Compare the calculated maximum sequence against the actual existing Metal dispatch requirement.

Test:

```text
max_sequence - 1  -> accepted
max_sequence      -> accepted if legal
max_sequence + 1  -> rejected before dispatch
```

### Video Decoder

Decode a fixed reference video using both old and new implementations during development and compare:

```text
frame count
dimensions
buffer length
all resulting F32 samples
```

They should match exactly.

Also test:

```text
1-frame video
short video
long video
max_frames truncation
EOF
partial/corrupt frame
FFmpeg error
allocation failure
```
