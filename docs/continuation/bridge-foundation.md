# Bridge continuation: T001–T020

This page records the first twenty tasks in [todo.md](todo.md), guided by
[design-bridge.md](design-bridge.md). Hard continuation remains the default and
keeps its existing initialization, modulation, velocity masking, Euler sampler,
state format, and trimming behavior.

**Bridge rendering is now implemented through T042.** See the
[bridge continuation guide](bridge-continuation.md) for current usage and
integration validation. The original T001–T020 preparation stage at `110def6`
rejected bridge renders until audio initialization, modulation materialization,
and CPU Euler integration were available. The foundation mathematics and
historical validation below remain useful; motion-quality acceptance is pending.

## Configuration

`h3_params` appends these fields, preserving the previous field order:

| Field | Default | Meaning |
|---|---|---|
| `continuation_mode` | `H3_CONTINUE_HARD` (zero) | `H3_CONTINUE_BRIDGE` opts into bridge continuation |
| `bridge_video_steps` | `8` | Number of inherited video temporal rows inside the bridge interval |
| `bridge_max_strength` | `0.50f` | Maximum denoise strength, finite and in `[0,1]` |
| `bridge_profile` | `H3_BRIDGE_STEPPED` | Also supports `H3_BRIDGE_LINEAR` and `H3_BRIDGE_EASE_OUT` |

The CLI accepts `--continue-mode hard|bridge`, `--continue-bridge-steps N`,
`--continue-bridge-max-strength X`, and `--continue-bridge-profile
stepped|linear|ease-out`. Hard mode ignores the bridge tuning fields. Existing
commands that omit these options follow the original hard path. As with any
public C structure extension, rebuild applications against the updated header.

Bridge steps must be positive and strictly shorter than the video context.
One configured exact row is valid and emits a warning; two or more are
recommended. Zero maximum strength is allowed as a hard-equivalence diagnostic.
The complete continuation context still follows `39 + 51*k` frames and must
leave a generated suffix in the target.

## Temporal profiles and quantization

`h3_bridge_profile` in `src/sampling/bridge.h` owns small temporal class arrays, independent
of spatial dimensions. Each position is a `uint8_t` class identifier. There are
at most 24 logical classes: the four original generated/exact video/audio
classes and ten nonzero bridge levels per stream. The levels subdivide the
configured maximum into tenths; nearest-level rounding is deterministic, with
ties going upward. Zero maps to the existing exact class, and one maps to the
existing generated class. Only active classes need modulation vectors.

For normalized time `u` inside the bridge interval:

* `linear`: `maximum * (1 - u)`.
* `ease-out`: `maximum * (1 - u*u)`, keeping greater strength in the older context.
* `stepped`: eight equal time bins with relative strengths
  `1, 1, .8, .8, .6, .4, .2, .1`.

At or beyond the end of the interval, strength is zero. Quantization can make
the last part of a linear/ease-out interval exact earlier. This adds preserved
rows; it never reduces the configured exact endpoint.

The authoritative video time grid uses repeating frame spans `1,4,4,4,4`.
`h3_video_time_boundary_frames(t)` exposes that mapping without changing the
existing packed layout. Video strengths are sampled at row start times; audio
strengths are sampled at 40 Hz from the same time-domain function. The audio
bridge endpoint rounds the rational boundary to the nearest tick, with ties
going later. Both stereo planes use the same tick classification.

For the default context and bridge:

| Interval | Video temporal rows | Frame duration | Audio ticks |
|---|---:|---:|---:|
| Bridge | 8 | 26 | 43 |
| Configured exact endpoint | 4 | 13 | 22 |
| Complete inherited context | 12 | 39 | 65 |

The bridge is 26/24 seconds; its audio boundary is 43/40 seconds, a difference
of 1/120 second. The complete inherited context ends at exactly 1.625 seconds
in both streams.

The resulting default video strengths are:

```text
0.50 0.50 0.50 0.40 0.30 0.20 0.20 0.10 | 0 0 0 0 | 1 1 ...
```

These differ from the design's illustrative equal-step examples because the
actual video temporal rows have unequal durations. Video and audio sample one
shared time function instead of separately authored strength arrays.

## Timesteps, velocities, and initialization

For a fractional class, `effective_sigma = strength * stream_sigma` and
`timestep = 1 - effective_sigma`, evaluated in F32. The four original classes
delegate to the existing timestep helper. In particular, exact video remains
`max(1 - sigma_video, .999)`, and exact audio remains `1`.

The design's visual floor applies when the row is an exact visual condition.
Clamping all fractional bridge timesteps upward to `.999` would label noisy
bridge rows almost clean and contradict the fractional effective-sigma rule.
Fractional rows therefore retain their own computed timestep.

`h3_dit_schedule_plan_bridge` appends host timestep rows to the original
schedule. All original text, reference, and condition row indices remain
unchanged. Target rows select their temporal class through the existing video
packing and `[left audio ticks][right audio ticks]` ordering. Equal timesteps
within a step share one row, including equality with existing condition or
generated rows. The host plan contains no GPU block/final tensors;
`h3_dit_schedule_precompute_bridge` now materializes those vectors for use by the
Metal transformer with the CPU sampler (T027).

`h3_bridge_mask_velocity` scales unpacked video/audio velocities by their class
strength, clears exact predictions even if they contain NaNs, and leaves
generated rows untouched, including signed-zero bits. **Apply this helper once
to raw predicted or reused velocities.** Fractional multiplication is not
idempotent. The current hard path masks after forward and again before Euler;
replacing both calls with fractional multiplication would incorrectly square
bridge strengths. The integrated bridge forward pass returns raw predictions;
the CPU Euler step applies the mask once. Bridge velocity reuse was disabled
for this foundation stage; the [current guide](bridge-continuation.md) describes
the subsequent GPU and reuse implementation.

The flow audit used the existing normal target initialization (standard normal
noise, initial stream sigma one), condition augmentation in `src/engine.c`, the shifted
stream schedules, and Euler velocity updates. The compatible arbitrary-sigma
interpolation is:

```text
(1 - sigma) * clean + sigma * already_generated_target_noise
```

`h3_flow_mix` uses separate F32 products and preserves the input bits at sigma
zero and one. It performs no RNG operations. `h3_av_state_insert_bridge_video`
uses this helper with `strength * initial_video_sigma` for fractional rows.
Exact video follows a separate branch with the original literal `.999f` and
`.001f` products. The source is borrowed, the generated suffix retains its
ordinary noise, and audio is untouched by this video-only preparation helper.

## Validation

Host tests cover all seven supported context sizes, every valid bridge length,
all three profiles, eight maximum-strength cases including zero, one and
subnormal F32 values, invalid configuration, AV boundary rounding, common-time
AV class agreement, packed stereo mapping, modulation deduplication, preserved
reference classes, and initialization at multiple effective sigmas.

The five requested images (`face1.jpg`, `body1.jpg`, `face2.jpg`, `body2.jpg`, and
`2.jpg`) are decoded by FFmpeg and encoded with the released Ref2VA video VAE.
For numerical testing only, each clean image latent is repeated through a
90-frame-shaped state with zero audio. These are explicitly labeled test
fixtures, not generated clips. Each fixture exercises all 27 combinations of
three profiles, three maximum strengths, and three initial sigmas, checking
every video value, exact-row parity, suffix preservation, source immutability,
and unchanged audio in the original video-only suite. The current suite also
checks joint audio initialization. No image or video is decoded from these test
states.

Reproduce the checks from the repository root:

```sh
make -j8 all bin/bridge_tests
./bin/bridge_tests
make test-bridge-sanitize test-continuation-sanitize
mkdir -p outputs/bridge-validation/encoded-inputs
./bin/bridge_tests --inputs models/MiniMax-H3 outputs/bridge-validation/encoded-inputs
python3 tests/bridge_cli.py
make test
python3 tests/bridge_regression.py
```

Metal tests require access to the local GPU. The existing `make test` target
still skips optional parity checks whose external fixtures are not installed.

The regression harness builds commit `6c09804` and the working source in
isolated directories. It runs five sequential renders with 20 Euler transitions,
all 50 DiT layers, and the face/body references: original/current unmasked
Ref2VA, original/current default hard continuation, and explicit hard with
nondefault bridge tuning. Captured text/reference conditioning is replayed to
isolate the previously documented multimodal encoder variability. It requires
identical final AV latents, `.h3av` bytes, and MP4 bytes, and checks synchronized
AV duration and zero start timestamps. It also tests bridge video preparation
using the generated source state.

Results and exact commands are recorded under `outputs/bridge-validation/`.
The regression script rebuilds and reruns current cases. Its optional
`--reference` flag now supports verified baseline reuse, described in the
[current guide](bridge-continuation.md).

Historical validation results at `110def6` (T001–T020):

| Check | Result |
|---|---|
| Bridge host suite | PASS: 13,535,516 assertions |
| Bridge ASan/UBSan suite | PASS: same host suite |
| Existing continuation host and sanitizer suites | PASS: 103,190 assertions each |
| Existing `make test` suite | PASS; optional missing-fixture parity checks skipped |
| Requested image fixtures | PASS: all five images, 135 initialization combinations |
| Bridge CLI parsing/validation | PASS: 31 cases |
| Real generated state bridge preparation | PASS: 27 initialization combinations |
| Pre-bridge versus default/explicit hard regression | PASS: all five 20-step renders |
| Final compiler and whitespace checks | PASS: no warnings or whitespace errors |

The real-model comparison used baseline commit
`6c09804` and produced identical video/audio latents, `.h3av` states, and MP4
bytes. The source clips have synchronized 3.75-second AV streams; the trimmed
continuations have synchronized 2.125-second AV streams, all starting at zero.
See the [recorded regression results](../outputs/bridge-validation/regression/results.json),
[CLI cases](../outputs/bridge-validation/cli.json), and
[image fixture log](../outputs/bridge-validation/inputs.log).
