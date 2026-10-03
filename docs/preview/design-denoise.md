## Denoised-Estimate Preview Semantics

### Objective

Change H3 preview generation so that preview frames represent the model's current estimate of the clean video latent rather than the current noisy sampler state.

For Euler sampling, the preview latent should be:

```text
x0_estimate = x_t + sigma_t * velocity
```

or, after completing an Euler step:

```text
x0_estimate = x_next + sigma_next * velocity
```

These formulations are algebraically equivalent for the Euler update used by `h3cli`.

The implementation should:

* preserve the actual sampler trajectory exactly;
* preserve deterministic final generation output;
* make early and intermediate previews substantially more informative;
* preserve raw sampler-state access through the existing latent-step/checkpoint callbacks;
* support normal generation, hard continuation, bridge continuation, and velocity-reuse schedules;
* avoid introducing an unnecessary full-size preview allocation on the CPU path;
* eventually provide equivalent behavior for the GPU-state sampler;
* make the denoised estimate the default preview behavior after validation.

---

## Background

The current CPU Euler path approximately performs:

```text
velocity = DiT(x_t, sigma_t)

x_next =
    x_t +
    (sigma_t - sigma_next) * velocity

preview(x_next)
```

The latent passed to the preview decoder is therefore still the noisy state corresponding to `sigma_next`.

At high sigma, this can remain largely noise even when the model already has a useful prediction of the eventual scene.

The model's corresponding clean estimate is:

```text
x0 =
    x_t +
    sigma_t * velocity
```

Using the Euler update:

```text
x_next =
    x_t +
    (sigma_t - sigma_next) * velocity
```

gives:

```text
x_next +
sigma_next * velocity

=
x_t +
(sigma_t - sigma_next) * velocity +
sigma_next * velocity

=
x_t +
sigma_t * velocity
```

Therefore:

```text
x0_estimate =
    x_next +
    sigma_next * velocity
```

can be constructed immediately after the Euler update without changing sampling.

---

# Preview Semantics

## Raw Sampler State

The actual sampler state remains:

```text
x_next
```

This state must continue to be used for:

```text
next denoising iteration
checkpointing
stop/resume
continuation-state persistence
debugging
latent-step callbacks
```

It must not be replaced by the denoised estimate.

## User-Facing Preview

The preview callback should receive:

```text
x0_estimate =
    x_next +
    sigma_next * effective_velocity
```

The distinction should be explicit:

```text
sampler/checkpoint callbacks
    -> exact noisy sampler state

preview callback
    -> estimated clean latent for display
```

This avoids conflating visualization with sampler state.

---

# CPU Euler Implementation

For the CPU sampler, no additional full-size preview allocation is required.

The existing velocity buffer can be reused as scratch storage after the Euler update if no subsequent operation in the current step requires its original values.

Conceptually:

```c
for (size_t i = 0; i < video_count; i++) {
    video_latent[i] +=
        (sigma - sigma_next) *
        video_velocity[i];
}
```

After all state-saving and reuse-history operations that require the velocity have completed:

```c
for (size_t i = 0; i < video_count; i++) {
    video_velocity[i] =
        video_latent[i] +
        sigma_next *
        video_velocity[i];
}
```

Then:

```c
preview(
    step + 1,
    total_steps,
    video_velocity,
    video_count,
    preview_opaque);
```

This changes the semantic role of `video_velocity` only after it is no longer needed as velocity for that step.

The exact location of this transformation must be chosen carefully relative to continuation masks, bridge bookkeeping, and reuse-history updates.

---

# Effective Velocity

The preview must use the same effective velocity that was applied to the sampler.

Do not reconstruct the preview from an unmasked or pre-bridge DiT prediction if the Euler step used a modified velocity.

Conceptually:

```text
raw DiT velocity
      ↓
continuation / bridge masking
      ↓
reuse/extrapolation adjustments
      ↓
effective velocity
      ↓
Euler step
      ↓
denoised preview
```

The preview should therefore satisfy:

```text
x0_preview =
    x_after_step +
    sigma_next * velocity_used_for_step
```

---

# Standard Generation

For ordinary generation:

```text
effective_velocity =
    DiT velocity
```

and:

```text
preview =
    updated_latent +
    sigma_next * effective_velocity
```

No other behavior changes.

---

# Hard Continuation

For preserved hard-continuation prefix rows, the existing continuation mask sets effective velocity to zero.

Therefore:

```text
prefix preview =
    preserved_prefix +
    sigma_next * 0
```

which gives:

```text
prefix preview =
    preserved clean prefix
```

This is desirable.

The generated suffix continues to use:

```text
updated_suffix +
sigma_next * effective_suffix_velocity
```

The preview therefore displays a clean preserved prefix joined to the current denoised estimate of the newly generated suffix.

---

# Bridge Continuation

Bridge continuation modifies velocity according to its transition/masking policy before the Euler update.

The preview should use the resulting bridge-scaled effective velocity:

```text
bridge_preview =
    bridge_updated_state +
    sigma_next *
    bridge_effective_velocity
```

Do not use the raw pre-bridge DiT velocity.

This ensures the preview represents the clean estimate corresponding to the actual bridge trajectory being sampled.

If bridge logic stores the raw velocity for later reuse or diagnostics, perform those copies before overwriting the velocity buffer with the preview latent.

---

# Velocity Reuse / Extrapolation

When a denoising step reuses or extrapolates velocity instead of invoking DiT, the preview must use the effective extrapolated velocity actually used by the Euler update.

For example:

```text
effective_velocity =
    extrapolate(last_velocity,
                previous_velocity,
                ratio)
```

Then:

```text
x_next =
    x_current +
    delta_sigma * effective_velocity

preview =
    x_next +
    sigma_next * effective_velocity
```

The preview implementation must not assume that every step's velocity originates directly from a fresh DiT evaluation.

---

# Ordering Requirements

The CPU path should preserve the following logical ordering:

```text
1. obtain effective velocity
2. preserve any velocity/history required for reuse
3. apply continuation/bridge modifications
4. perform Euler update
5. emit/save actual sampler state through latent/checkpoint path
6. construct denoised preview
7. invoke preview callback
8. proceed to next step
```

If current reuse-history logic requires the post-mask velocity rather than the pre-mask velocity, preserve exactly the existing semantics.

The preview conversion must occur only after all consumers needing the velocity representation are finished.

---

# Final Step

At the final step:

```text
sigma_next = 0
```

therefore:

```text
preview =
    x_final +
    0 * velocity
```

so:

```text
preview = x_final
```

The final preview is consequently identical to the actual final latent.

No special-case implementation should be necessary.

---

# Preview Callback Contract

Update the preview callback documentation.

Current semantics such as:

```text
current video latent
```

should be replaced with wording such as:

```text
current estimated clean video latent intended for visualization
```

Explicitly document that the preview latent:

* is not necessarily the current sampler state;
* must not be persisted as a resume/checkpoint state;
* may change substantially between early denoising steps;
* represents the model's current clean estimate.

The raw state remains available through the sampler-state/latent-step mechanism.

---

# Configuration

During rollout, expose preview semantics through:

```text
H3_PREVIEW_MODE
```

Recommended accepted values:

```text
denoised
noisy
```

After final rollout:

```text
H3_PREVIEW_MODE unset
    -> denoised
```

Explicit behavior:

```text
H3_PREVIEW_MODE=denoised
    -> x_next + sigma_next * effective_velocity

H3_PREVIEW_MODE=noisy
    -> raw x_next sampler state
```

`noisy` should be retained mainly for:

* debugging;
* comparison with historical preview behavior;
* diagnosing preview-specific regressions.

It should not remain the recommended user-facing mode.

Invalid non-empty values should produce a clear configuration error.

---

# Memory Behavior

The CPU implementation should avoid the extra full-latent F32 buffer introduced by a literal implementation of PR #34.

Instead reuse the existing velocity workspace after all velocity consumers have completed.

Target incremental memory cost:

```text
approximately zero
```

apart from trivial configuration/state metadata.

This is preferable given ongoing work to reduce long-reference and unified-memory pressure.

---

# GPU-State Sampler

The CPU change should not block deployment if equivalent GPU behavior is not ready immediately.

However, the end state should provide the same preview semantics for GPU-state sampling.

Avoid implementing the GPU path by reading back the full velocity tensor separately every preview step.

Preferred design:

```text
GPU sampler state
      +
effective GPU velocity
      +
sigma_next
      ↓
small Metal preview kernel
      ↓
GPU denoised-preview tensor
      ↓
existing preview readback
```

Conceptually:

```metal
preview[i] =
    sample[i] +
    sigma_next *
    velocity[i];
```

If velocity reuse/extrapolation is performed on GPU, the preview kernel must use the effective velocity corresponding to the actual Euler step.

Where possible, the existing sampler output/readback buffer should be reused rather than allocating an additional long-lived GPU tensor.

---

# GPU/CPU Parity

After GPU denoised preview support exists, CPU and GPU samplers should produce numerically equivalent preview latents within the precision expected from their storage formats.

Test:

```text
same starting state
same velocity
same sigma
same continuation masks
```

and compare:

```text
CPU x0 estimate
GPU x0 estimate
```

using:

```text
max absolute error
RMSE
relative L2
```

Exact equality is not required where the GPU path stores velocity or sample state in BF16.

---

# Preview Quality Validation

Use fixed sampler states and velocities to isolate preview behavior.

For selected steps, save:

```text
x_t
effective velocity
sigma_t
sigma_next
```

Then calculate independently:

```text
reference_x0 =
    x_t +
    sigma_t * velocity
```

and compare with:

```text
post_step_x0 =
    x_next +
    sigma_next * velocity
```

These should agree within expected floating-point error.

---

# End-to-End Validation

For fixed-seed generation:

```text
preview=noisy
preview=denoised
preview disabled
```

must all produce the same:

```text
final latent
final video
final audio
saved continuation state
saved sampler/checkpoint state
```

provided preview callbacks themselves do not request cancellation.

Hash or byte-compare deterministic internal states where practical.

The preview mode must have no influence on sampling.

---

# Continuation Validation

Test:

```text
ordinary generation
hard continuation
bridge continuation
multi-segment continuation
velocity reuse enabled
velocity reuse disabled
```

At preserved hard-prefix locations assert:

```text
denoised preview prefix ==
preserved continuation prefix
```

within expected numerical precision.

For bridge rows, verify that the preview calculation uses the bridge-modified velocity rather than the raw DiT velocity.

---

# Cancellation

Preview callbacks may request cancellation.

Changing the latent passed to the callback must not alter cancellation semantics.

The execution order should remain:

```text
construct preview
invoke callback
if callback requests cancellation:
    exit through normal sampler cleanup
```

The previously designed inner-loop cancellation propagation should remain unaffected.

---

# Performance

CPU denoised-preview construction adds one simple elementwise operation:

```text
preview[i] =
    latent[i] +
    sigma *
    velocity[i]
```

per preview step.

This is expected to be negligible relative to:

```text
DiT
VideoVAE preview decode
GPU synchronization/readback
```

Benchmark nevertheless:

```text
preview construction time
preview VAE decode time
complete denoising-step time
total render time
```

The optimization should not introduce a material runtime regression.

---

# User Experience

Early denoised estimates can differ substantially from the final composition.

This is expected.

The preview represents:

```text
the model's current prediction of the clean sample
```

not:

```text
a guarantee of the eventual final frame
```

Do not apply clipping, smoothing, temporal blending, or artificial enhancement solely to make early previews look more finished unless such behavior is implemented as a separate display-only feature.

The default preview should remain a faithful representation of the model's current clean estimate.

---

# Compatibility

Changing preview semantics does not intentionally change:

```text
random seeds
DiT evaluation
sampler equations
noise
scheduler
latent state
continuation state
checkpoint files
final VAE decoding
audio generation
```

It changes only the latent delivered to the preview visualization path.

External integrations that consume preview tensors as though they were exact noisy sampler states may observe a behavioral change.

Such integrations should use the existing raw sampler-state callback instead.

---

# Rollout

## Stage 1 — CPU Implementation

Implement both:

```text
denoised
noisy
```

preview modes for the CPU Euler sampler.

Keep tests explicitly selecting modes during development.

## Stage 2 — Sampling-Invariance Validation

Verify that enabling either preview mode or disabling previews entirely yields identical sampler/final state.

## Stage 3 — Continuation and Reuse Validation

Verify hard continuation, bridge continuation, and reused/extrapolated velocity paths.

## Stage 4 — GPU Implementation

Implement denoised estimate construction on GPU without an additional full velocity readback.

## Stage 5 — CPU/GPU Parity

Verify equivalent preview semantics across sampler implementations.

## Stage 6 — Finalize Default

Once validation succeeds:

```text
H3_PREVIEW_MODE unset
    -> denoised
```

Retain:

```text
H3_PREVIEW_MODE=noisy
```

as a compatibility/debugging mode.

---

# Final Production Behavior

The canonical preview operation should be:

```text
effective velocity
      ↓
Euler update
      ↓
actual x_next sampler state
      ├──────────────> checkpoint / resume / next step
      │
      └─ + sigma_next * effective_velocity
                     ↓
              x0 estimate
                     ↓
             preview decoder
```

Default configuration:

```text
H3_PREVIEW_MODE unset
    -> denoised

H3_PREVIEW_MODE=denoised
    -> denoised estimate

H3_PREVIEW_MODE=noisy
    -> historical raw sampler-state preview
```

The denoised estimate is the production/default preview representation.
