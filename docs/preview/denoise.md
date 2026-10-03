# Denoised display previews

Live previews default to the current estimated clean video latent on both CPU
and GPU samplers. Unset or empty `H3_PREVIEW_MODE` selects `denoised`.
`H3_PREVIEW_MODE=noisy` displays the historical noisy state for compatibility
and debugging; `denoised` is the supported production display mode.
Unknown non-empty values fail configuration validation, including on resume.
Set the mode before generation; changing environment variables during a run is
unsupported. `H3_PROFILE=1` or `H3_PREVIEW_DIAGNOSTICS=1` reports the selected mode
when previews are active.

The serving Euler step uses positive delta:

```
x_next = x_t + (sigma_t - sigma_next) * effective_velocity
x0_preview = x_next + sigma_next * effective_velocity
           = x_t + sigma_t * effective_velocity
```

The two clean-estimate expressions agree within FP32 rounding. A final-step
preview with zero remaining sigma is an exact copy of the final latent.
Early clean estimates can change substantially as denoising progresses.

## State and callback ordering

`on_latent_step` receives the exact raw video and audio sampler state before
sampling and after every Euler transition. It runs before the display callback
and remains the authoritative interface for exact latent-state access. Display
estimates must never be saved as checkpoint or continuation state.

The CPU sampler preserves velocity history, applies continuation/bridge masks,
integrates, and invokes the raw-state callback before reusing the video-velocity
workspace for the display estimate. Bridge history keeps raw predictions;
display uses the already scaled/masked working velocity. Reuse steps use the
extrapolated velocity actually applied by Euler. Zero-velocity prefix elements
are copied directly, preserving even signed zero.

The GPU sampler keeps raw BF16 prediction histories. A small Metal operation
repeats the same extrapolation and mask arithmetic as its Euler kernel and
writes the clean estimate directly in video latent layout. It reuses the dead
QKV activation buffer after the final head and the existing host packing buffer
for readback. It allocates no additional persistent full-latent buffer and does
not read back velocity. Core residuals, prediction histories and sample buffers
are separate and remain untouched.

Hard and bridge continuation use CPU state by default. Explicit
`H3_GPU_SAMPLER=1` selects GPU state for either; `H3_CPU_SAMPLER=1` overrides it.
Preserved hard-prefix rows have zero effective velocity. Bridge previews use
the same fractional strengths as the actual transition.

## Checkpoints and compatibility

Preview mode changes neither DiT evaluations nor Euler integration, seeds,
noise, schedules, conditioning, audio state, or completed generation output.
Preview mode and its diagnostics are excluded from mathematical checkpoint
identity, so a saved run may resume with a different display mode. Existing
engine-build and numerical-environment compatibility checks remain enforced.
The on-stop full-clip decode continues to show the actual stored pause state;
it is separate from per-step denoised display previews.

Cancellation through a preview callback retains the same ordering and cleanup.
The memory guard and inner-loop cancellation checks remain active independently
of the selected display mode.

## Implementation audit

Before this change, the CPU `h3_dit_denoise_euler_range` and GPU
`denoise_euler_gpu_range` both passed raw post-Euler `x_next` to their display
callback. The public `h3_deliver_denoise_preview` decoded that argument without
changing its semantics. The RES sampler has no live preview callback.

CPU raw bridge diagnostics and reuse-history copies finish before display
construction. GPU histories remain in `video_output_bf16` and
`previous_video_velocity`; QKV scratch is not retained across Euler steps.
The preview kernel uses the same sample offset and packed-row class map as the
Euler update, then unpatchifies its output for the decoder. At zero sigma or
zero effective velocity the implementation explicitly preserves sample bits.

Released-model invariance testing also exposed loss of optional presentation
diagnostics on conditioning-cache hits. Cache copies now retain those IDs,
positions and spans, so enabling previews on a cached rerun does not change
serialized checkpoint metadata.

## Release note

Live preview images now become recognizable earlier and can look substantially
different from historical noisy previews. They remain predictions and may
change composition during the early steps. Identical seeds and numerical
settings produce unchanged final generations across preview modes. No clipping,
smoothing, temporal blending or enhancement is added to the preview latent.
Integrations needing exact noisy latents should use `on_latent_step`.

See [validation methodology and results](denoise-validation.md) for numerical,
generation, cancellation, memory, performance and visual evidence.

Implementation and M4 validation: [denoised previews](denoise.md), [test results and visual evidence](denoise-validation.md).
