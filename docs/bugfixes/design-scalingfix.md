## Qwen Causal-GQA Attention Scaling Precision

### Objective

Correct the precision loss in the Qwen causal-GQA Metal kernel caused by re-quantizing scaled query values to BF16 before the QK contraction.

The implementation should initially support three arithmetic modes for validation and backward compatibility:

```text
legacy
scaled-q
reference
```

After parity and regression testing is complete, the `reference` implementation should become the production default.

The final default should calculate:

```text
QK dot product with FP32 accumulation
             ↓
multiply resulting score by scale in FP32
             ↓
softmax
```

This most closely follows the explicit Qwen3-VL/Hugging Face formulation and the corrected high-precision attention behavior in current MLX.

---

## Background

The existing causal-GQA kernel effectively performs:

```text
Q BF16
  ↓
convert to FP32
  ↓
multiply by attention scale
  ↓
round back to BF16
  ↓
convert to FP32
  ↓
QK dot product with FP32 accumulation
```

Conceptually:

```c
shared_query[d] =
    bf16_to_f32(
        f32_to_bf16(
            bf16_to_f32(query[d]) * scale));
```

The resulting `shared_query` storage is already FP32.

The intermediate BF16 conversion therefore does not reduce shared-memory consumption and does not reduce Q/K input bandwidth. It only introduces an additional quantization point.

Issue #3 demonstrated substantially larger numerical error from this operation compared with a high-precision attention reference.

The proposed implementation should eliminate that unnecessary quantization while allowing controlled comparisons between:

1. current behavior;
2. PR #4's scale-Q-in-FP32 behavior;
3. the preferred reference formulation that applies scale to the FP32-accumulated QK score.

---

# Arithmetic Modes

## 1. Legacy

The legacy mode must reproduce the current kernel behavior exactly:

```text
scaled_q =
    BF16_round(FP32(q) * scale)

score =
    FP32_dot(FP32(scaled_q), FP32(k))
```

Equivalent conceptual code:

```c
float q =
    h3_bf16_to_f32(query[q_base + d]);

shared_query[d] =
    h3_bf16_to_f32(
        h3_f32_to_bf16(q * args.scale));
```

Then:

```c
score += shared_query[d] *
         h3_bf16_to_f32(key[k_base + d]);
```

This mode exists only for:

* reproducibility;
* comparison with historical `h3cli` output;
* diagnosing regressions;
* reproducing previously generated videos.

It should not remain the recommended quality mode.

---

## 2. Scaled-Q FP32

This mode implements the core PR #4 approach:

```text
scaled_q =
    FP32(q) * scale

score =
    FP32_dot(scaled_q, FP32(k))
```

Conceptually:

```c
shared_query[d] =
    h3_bf16_to_f32(query[q_base + d]) *
    args.scale;
```

The dot product then proceeds unchanged.

This eliminates the BF16 re-round while retaining the existing operation order.

The mode is useful because it isolates the effect of removing the quantization step from the separate issue of changing arithmetic order.

---

## 3. Reference

The reference mode should become the final default.

It should perform:

```text
q =
    FP32(q_bf16)

score =
    FP32_dot(q, FP32(k))

score =
    score * scale
```

Conceptually:

```c
shared_query[d] =
    h3_bf16_to_f32(query[q_base + d]);
```

then:

```c
float score = 0.0f;

for (uint d = ...; d < head_dim; ...) {
    score = fma(
        shared_query[d],
        h3_bf16_to_f32(key[k_base + d]),
        score);
}

score *= args.scale;
```

The scaling operation must occur before:

```text
maximum reduction
softmax exponentiation
attention weighting
```

so the mathematical operation remains:

```text
softmax((QK^T) / sqrt(d))
```

---

# Configuration

During the validation period, expose the mode through a diagnostic/runtime configuration.

Recommended environment variable:

```text
H3_QWEN_GQA_SCALE_MODE
```

Accepted values:

```text
reference
scaled-q
legacy
```

After implementation is validated:

```text
unset H3_QWEN_GQA_SCALE_MODE
    -> reference
```

Explicit behavior:

```text
H3_QWEN_GQA_SCALE_MODE=reference
    -> reference FP32 score scaling

H3_QWEN_GQA_SCALE_MODE=scaled-q
    -> FP32 scaling of Q before QK

H3_QWEN_GQA_SCALE_MODE=legacy
    -> historical BF16-rounded scaled-Q behavior
```

Unknown values should fail configuration validation rather than silently selecting another arithmetic mode.

---

# Implementation Strategy

Prefer compiling dedicated Metal pipeline variants rather than branching on the arithmetic mode for every attention score operation.

For example:

```text
h3_gqa_causal_bf16_legacy
h3_gqa_causal_bf16_scaled_q
h3_gqa_causal_bf16_reference
```

or equivalent function-constant specialization.

The objective is for the production `reference` kernel to have no per-element runtime branch.

If maintaining three shader entry points would create excessive duplication, use a shared implementation parameterized with compile-time constants.

Avoid:

```c
if (mode == ...)
```

inside the innermost QK loop.

---

# Reference Kernel

The final production kernel should conceptually follow:

```c
/* Load Q without scaling or re-quantization. */
for (uint d = tid; d < args.head_dim; d += tg_size) {
    shared_query[d] =
        h3_bf16_to_f32(query[q_base + d]);
}

threadgroup_barrier(
    mem_flags::mem_threadgroup);

/* FP32 accumulated QK dot product. */
float local_score = 0.0f;

for (uint d = tid;
     d < args.head_dim;
     d += tg_size) {
    local_score = fma(
        shared_query[d],
        h3_bf16_to_f32(key[k_base + d]),
        local_score);
}

/* Existing threadgroup reduction. */

float score =
    reduced_dot * args.scale;
```

The exact reduction structure should remain unchanged unless a separate numerical-validation task demonstrates a need to modify it.

---

# Numerical Semantics

Input Q and K tensors remain BF16.

The change does not attempt to reconstruct precision that was already lost when those activations were written as BF16.

It changes only the location of attention-scale multiplication.

Final intended path:

```text
BF16 Q ─┐
        ├─> FP32 multiplication/FMA accumulation
BF16 K ─┘
             ↓
         FP32 score
             ↓
       FP32 × scale
             ↓
       FP32 softmax path
```

No additional BF16 conversion should occur between loading Q/K and producing the pre-softmax attention score.

---

# Scope

The affected code is the Qwen causal-GQA implementation used by the H3 text/multimodal encoder.

The change can therefore affect:

```text
FL2VA text conditioning
Ref2VA multimodal conditioning
image-reference conditioning
video-reference conditioning
dialogue conditioning
continuation prompts that run through Qwen
```

It does not directly modify:

```text
DiT attention implementation
VideoVAE
AudioVAE
tokenizer
reference-video encoder
random-number generation
noise initialization
```

However, because Qwen output conditions the DiT, changing Qwen numerical output can result in different final video latents and therefore different renders for the same seed.

---

# Compatibility

## Existing Render Reproducibility

Changing the default from legacy to reference is expected to break bit-for-bit generation reproducibility for at least some prompts.

The dependency chain is:

```text
attention score change
        ↓
Qwen hidden-state change
        ↓
conditioning tensor change
        ↓
DiT trajectory change
        ↓
potentially different latent/video
```

Therefore retain:

```text
H3_QWEN_GQA_SCALE_MODE=legacy
```

for reproducing historical behavior.

No saved-file format change should be necessary because the difference is computational rather than structural.

---

# CPU Reference Implementation

Create a simple high-precision CPU GQA implementation that defines the intended arithmetic independently from the Metal kernel.

The reference must perform:

```text
for each query/head/key:

    score = 0

    for d:
        score += float(Q[d]) * float(K[d])

    score *= scale
```

followed by the same:

```text
causal masking
stable softmax
V accumulation
```

as the production implementation.

Use FP32 throughout, with BF16 inputs converted to FP32 once.

This CPU path should define correctness for low-level kernel tests.

---

# Synthetic Kernel Tests

Test all three modes against the CPU reference.

Include:

```text
random Q/K/V
small sequence
medium sequence
long sequence
head_dim = 128
64 query heads
8 KV heads
causal masks
sharp attention distributions
large positive logits
large negative logits
nearly tied logits
```

Collect:

```text
max absolute error
mean absolute error
RMSE
relative L2 error
BF16 output mismatch count
```

Expected ordering should generally be:

```text
reference <= scaled-q << legacy
```

although `reference` and `scaled-q` may sometimes be very close.

Do not require exact equality between `reference` and `scaled-q`.

---

# Full Qwen Encoder Parity

Kernel-only accuracy is insufficient.

Run end-to-end Qwen encoder comparisons using identical:

```text
model weights
input token IDs
vision embeddings
presentation
deepstack inputs
BF16 activation storage
```

Compare:

```text
legacy
scaled-q
reference
official reference implementation
```

at selected layers.

Recommended checkpoints:

```text
layer 1
layer 10
layer 25
layer 40
final H3 conditioning layer
```

For each checkpoint calculate:

```text
max absolute error
mean absolute error
RMSE
relative L2
cosine similarity
BF16 mismatch percentage
```

---

# Validation Presentations

Use several distinct classes of Qwen input.

## Plain Text

Example:

```text
A woman walks through a quiet room and looks toward the camera.
```

## Dialogue

Example:

```text
A woman says:
<d>[English] Where are you going?</d>
```

## Ref2VA Image

Use an image reference plus prompt.

## Ref2VA Video

Use a sufficiently long video reference to create a substantial multimodal sequence while remaining below the active GQA sequence limit.

## Complex Prompt

Use several subjects, spatial relationships, dialogue, and scene instructions so attention contains competing alternatives.

---

# Generation Validation

After encoder-level parity is established, perform deterministic H3 generations using identical:

```text
seed
noise
scheduler
step count
resolution
prompt
references
model weights
```

Compare:

```text
legacy
scaled-q
reference
```

Do not expect identical resulting videos.

The objective is to detect:

* obvious regressions;
* instability;
* NaNs/Infs;
* subject loss;
* prompt-adherence degradation;
* dialogue regressions;
* Ref2VA identity/reference regressions;
* temporal instability.

Quality testing should focus especially on prompts where attention choices are sharp or ambiguous because small numerical differences are more likely to matter there.

---

# Performance

Benchmark GQA kernel execution separately from complete Qwen runtime.

Measure:

```text
kernel time
complete Qwen layer time
complete Qwen encoder time
total H3 render time
```

Expected impact from removing BF16 re-rounding should be negligible.

Reference score scaling introduces approximately one FP32 multiply per attention score.

This operation should normally be tiny relative to the QK contraction and softmax.

A measurable large slowdown should be investigated as an implementation problem rather than accepted as an inherent cost of the change.

---

# Memory

No material memory change is expected.

The query threadgroup buffer is already FP32.

The reference implementation therefore does not enlarge:

```text
Q
K
V
shared query
score arrays
output tensors
```

The change should not materially affect the issue #47 sequence-length limit or long-reference memory consumption.

---

# Interaction with GQA Sequence Preflight

The GQA scaling change is independent from the previously designed threadgroup-memory sequence preflight.

The sequence limit calculation should remain based on:

```text
dynamic score storage
+
static threadgroup storage
```

The new scaling mode must not change or bypass that validation.

If later tiled/online GQA replaces the current sequence-sized score buffer, the same reference scaling rule should be carried into that implementation:

```text
QK accumulation
→ scale FP32 score
→ online softmax
```

---

# Logging

At verbose startup or Qwen initialization, report:

```text
Qwen causal GQA scaling: reference
```

or:

```text
Qwen causal GQA scaling: scaled-q
```

or:

```text
Qwen causal GQA scaling: legacy
```

Normal production output does not need to print the setting unless verbose diagnostics are enabled.

---

# Rollout

Use a staged rollout.

### Stage 1 — Implement Three Modes

Default may temporarily remain explicit in development builds while all three implementations are tested.

Required modes:

```text
legacy
scaled-q
reference
```

### Stage 2 — Kernel Validation

Compare all three against the independent CPU FP32 reference.

Do not proceed if the reference GPU mode unexpectedly has materially worse numerical agreement than `scaled-q`.

### Stage 3 — Qwen Encoder Validation

Compare complete Qwen output against the official reference implementation.

The primary decision criterion is full-encoder parity, not merely synthetic kernel error.

### Stage 4 — H3 Generation Validation

Perform deterministic FL2VA and Ref2VA generation comparisons and confirm absence of quality regressions.

### Stage 5 — Finalize Default

Once validation succeeds:

```text
unset H3_QWEN_GQA_SCALE_MODE
    -> reference
```

Retain:

```text
legacy
scaled-q
```

as explicit diagnostic/compatibility modes.

### Stage 6 — Long-Term Cleanup

After sufficient compatibility history, `scaled-q` may eventually be removed if it provides no diagnostic value.

`legacy` should be retained longer because it is useful for reproducing historical renders and regression bisects.

---

# Final Production Behavior

After rollout is complete, the canonical implementation is:

```text
BF16 Q
   +
BF16 K
   ↓
FP32 QK accumulation
   ↓
FP32 attention scaling
   ↓
FP32 softmax computation
   ↓
attention × V
```

Configuration defaults:

```text
H3_QWEN_GQA_SCALE_MODE unset
    -> reference

H3_QWEN_GQA_SCALE_MODE=reference
    -> reference

H3_QWEN_GQA_SCALE_MODE=scaled-q
    -> corrected alternative used for diagnostics

H3_QWEN_GQA_SCALE_MODE=legacy
    -> historical compatibility
```

The legacy BF16 re-round must not remain on the default production path.
