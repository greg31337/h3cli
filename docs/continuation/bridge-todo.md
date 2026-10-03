# Implementation Tasks

T001–T042 are implemented and tested. The [bridge foundation](bridge-foundation.md)
is integrated with end-to-end CPU bridge generation; see the
[acceptance report](bridge-integration-acceptance.md) and
[usage guide](bridge-continuation.md). T043–T065 implementation and evaluation
are recorded in [quality/sampler validation](bridge-quality-acceptance.md).
T051 and T065 remain open because material motion-quality improvement was not
established. T052 has an implemented, numerically validated GPU opt-in, but its
CPU-quality prerequisite remains unmet. T060 is conditional on enabling token
reduction, which remains explicitly disabled. T061 passes a three-segment stationary-pose chain; the locomotion stress-case limitation is recorded in the report.

* [DONE] T001: Add an `H3_CONTINUE_BRIDGE` continuation mode while retaining the existing hard-prefix implementation as `H3_CONTINUE_HARD` and making hard mode the unchanged default.

* [DONE] T002: Extend `h3_params` and CLI parsing with `--continue-mode hard|bridge`, ensuring continuation requests that do not specify the option execute the existing path exactly.

* [DONE] T003: Define bridge configuration parameters including bridge video-step count, maximum denoising strength, and bridge profile type, with defaults that have no effect unless bridge mode is explicitly enabled.

* [DONE] T004: Implement a bridge-profile representation that describes a spatially uniform temporal denoise strength in the range `[0,1]`, with `0` meaning fully preserved and `1` meaning normally generated.

* [DONE] T005: Implement initial `stepped`, `linear`, and `ease-out` bridge-profile generators, with `stepped` as the initial default profile.

* [DONE] T006: Add validation requiring the bridge region to be shorter than the continuation context so that at least one fully preserved video latent temporal row remains immediately before the generated suffix.

* [DONE] T007: Add a warning when the configured exact preservation region contains fewer than two video latent temporal rows, while allowing configurations when otherwise valid.

* [DONE] T008: Generate video bridge classes from the selected bridge profile and map each continuation-context video temporal row to either a fractional bridge class or the fully preserved class.

* [DONE] T009: Define the bridge interval in time and derive corresponding audio latent tick boundaries from that interval rather than reusing the number of video latent steps directly.

* [DONE] T010: Generate audio bridge classes from the same time-domain bridge profile used for video so that video and audio adaptation behavior remains temporally aligned.

* [DONE] T011: Ensure both packed stereo audio timelines receive identical bridge-strength classifications for every corresponding audio latent tick.

* [DONE] T012: Extend the existing target-row modulation-class map to represent exact video, fractional bridge-video strengths, generated video, exact audio, fractional bridge-audio strengths, and generated audio without changing existing reference or condition classes.

* [DONE] T013: Quantize bridge strengths into a small finite set of modulation classes so timestep-dependent AdaLN/modulation vectors can continue to be computed once per distinct class rather than independently for every target token.

* [DONE] T014: Implement per-denoising-step effective video sigma and timestep calculation for bridge classes using the existing H3 masked-continuation sigma/timestep convention and the existing visual condition-strength floor.

* [DONE] T015: Implement per-denoising-step effective audio sigma and timestep calculation for bridge classes using the existing H3 audio timestep convention.

* [DONE] T016: Generalize velocity masking so video bridge rows apply `velocity *= bridge_strength`, exact rows continue applying zero velocity, and generated rows retain full predicted velocity.

* [DONE] T017: Generalize velocity masking so audio bridge rows apply the corresponding fractional velocity scale, exact audio rows remain unchanged, and generated audio rows retain normal predicted velocity.

* [DONE] T018: Audit the current H3 flow/noise initialization formula and define a reusable helper that constructs a latent at an arbitrary effective initial sigma from a clean latent and the corresponding already-generated target noise.

* [DONE] T019: Initialize each bridge video row from the previous clean video tail and the corresponding normal target noise using the effective initial sigma implied by that row's bridge strength.

* [DONE] T020: Preserve the existing near-clean `0.999` initialization semantics for fully exact video rows rather than routing exact rows through the new bridge initialization path.

* [DONE] T021: Initialize each bridge audio tick from the previous clean audio tail and the corresponding normal target noise using the effective initial audio sigma implied by that bridge strength.

* [DONE] T022: Preserve the existing direct clean-latent copy behavior for fully exact audio rows.

* [DONE] T023: Guarantee that bridge initialization consumes no additional random numbers and uses only the noise tensor already created by the ordinary target initialization path so RNG ordering remains unchanged.

* [DONE] T024: Refactor target-prefix construction only as necessary to share code between hard and bridge continuation while maintaining a separate fast/exact hard path when required for byte-identical regression behavior.

* [DONE] T025: Implement bridge target-row classification in the packed video sequence using the authoritative existing video temporal-index-to-packed-row mapping rather than assumptions about packed memory contiguity.

* [DONE] T026: Implement bridge target-row classification in the packed audio sequence using the authoritative existing `[32,2,T]` audio packing order and explicit recovery of the audio time index for each target row.

* [DONE] T027: Extend timestep-dependent AdaLN/modulation preparation on the CPU path so every active bridge class receives its own correctly calculated modulation vector at each denoising step.

* [DONE] T028: Integrate bridge semantics into the simplest exact CPU Euler sampler using `--reuse 1`, `--core-reuse 1`, full layers, and token reduction disabled as the correctness oracle.

* [DONE] T029: Add assertions or debug checks proving that fully exact video rows do not change after their initial near-clean initialization during bridge sampling.

* [DONE] T030: Add assertions or debug checks proving that fully exact audio rows remain bit-identical throughout bridge sampling.

* [DONE] T031: Add diagnostics verifying that fractional bridge rows change during sampling by amounts consistent with their configured mask strengths while generated rows remain unaffected by bridge scaling.

* [DONE] T032: Extend `--keep-continuation-prefix` debug output to support bridge mode so developers can inspect the adapted bridge region, exact endpoint, and generated suffix in a single decoded clip.

* [DONE] T033: Add continuation logging that reports continuation mode, total context size, bridge duration, exact duration, selected bridge profile, maximum strength, video mask classes, audio bridge interval, and number of packed rows assigned to each modulation class.

* [DONE] T034: Add profile diagnostics for bridge-profile construction, bridge latent initialization, bridge class preparation, and any incremental memory required by the modulation-class map.

* [DONE] T035: Confirm that bridge mode requires no incompatible change to the `.h3av` binary format and optionally record bridge configuration only as nonessential generation metadata for reproducibility.

* [DONE] T036: Verify that an `.h3av` state produced by an existing hard-continuation run can be used directly as the source of a bridge-continuation run without conversion.

* [DONE] T037: Verify that an `.h3av` state produced by bridge continuation can subsequently be used by either hard or bridge continuation because the saved state continues to contain the complete final clean AV target latent.

* [DONE] T038: Verify that normal T2VA/FL2VA generation without continuation remains unchanged after bridge code is introduced.

* [DONE] T039: Verify that hard continuation remains unchanged after bridge code is introduced, including latent-level regression comparison against the implementation immediately preceding bridge development.

* [DONE] T040: Verify Ref2VA image references with bridge continuation while retaining the existing rule that continuation state is target history and never becomes an implicit Ref2VA reference.

* [DONE] T041: Verify combinations of bridge continuation with existing Ref2VA image, video, and audio references using the current reference packing and checkpoint-selection paths.

* [DONE] T042: Add explicit tests in which the Ref2VA references remain constant while the segment prompt changes from one action to another, because this is the primary bridge-continuation use case.

* [DONE] T043: Add an unchanged-prompt baseline test comparing hard-39 and bridge-39 and require that bridge mode does not materially degrade scene, background, character, color, or camera stability.

* [DONE] T044: Add a small-action-change test such as walking to walking-while-looking-upward and compare hard-39 against bridge-39.

* [DONE] T045: Add a meaningful-action-change test such as walking to completing the current step, turning toward the camera, and beginning to run, and use it as the primary bridge-quality acceptance case.

* [DONE] T046: Add a pose-transition test such as arms-down to gradually raising both arms and compare the first 1–2 seconds of new output between hard and bridge modes.

* [DONE] T047: Add a motion-direction-change test such as walking forward to slowing, turning left, and walking left, evaluating whether bridge mode reduces abrupt body and background trajectory changes.

* [DONE] T048: Create a reproducible parameter-sweep test harness covering bridge lengths such as 4, 6, 8, 9, and 10 video latent steps; maximum strengths such as 0.25, 0.40, 0.50, and 0.65; and all initial bridge profiles.

* [DONE] T049: Add automated extraction of frame sequences covering at least the final second of the source segment and first two seconds of the delivered continuation so hard and bridge outputs can be compared consistently.

* [DONE] T050: Add optional diagnostic measurements for optical-flow magnitude/direction, temporal motion acceleration, luminance, contrast, and background feature displacement across the continuation boundary.

* [OPEN] T051: Establish initial recommended bridge-39 defaults from the parameter sweep, prioritizing reduced action/pose discontinuity while preserving background geometry, character identity, lighting, and camera continuity. The 17-setting pilot and held-out confirmation did not support a new recommended default; existing defaults remain opt-in.

* [OPEN] T052: Integrate bridge semantics into the GPU-state sampler after the CPU implementation passes correctness and quality tests, including bridge initialization, fractional timestep classes, velocity scaling, and exact-row preservation. Implementation and seven CPU/GPU comparisons pass, but the prerequisite quality gate does not; CPU remains the default and GPU requires explicit opt-in.

* [DONE] T053: Add CPU-versus-GPU bridge regression tests using equivalent exact settings and require final latent agreement within the tolerances already accepted by h3cli.

* [DONE] T054: Keep bridge mode on the validated CPU sampler or produce an explicit unsupported-mode error until the GPU-state bridge implementation is complete rather than silently executing incorrect GPU semantics.

* [DONE] T055: Audit whole-denoiser velocity reuse so reused or extrapolated velocities are always multiplied by the current row's bridge mask before Euler application and exact rows remain zero.

* [DONE] T056: Validate bridge mode incrementally with existing `--reuse 2` and `--reuse 3` settings and keep unvalidated reuse combinations disabled with clear errors.

* [DONE] T057: Audit core reuse so timestep-dependent heads and modulation operations continue distinguishing exact, fractional bridge, and generated target rows on every denoising step.

* [DONE] T058: Validate bridge mode incrementally with existing `--core-reuse 4` and `--core-reuse 6` settings after exact CPU/GPU bridge behavior is established.

* [DONE] T059: Continue disabling token reduction in bridge mode initially and add an explicit error or forced-safe configuration rather than allowing an unvalidated token-reduction path.

* [DEFERRED] T060: If token reduction support is later enabled, verify that reduced token groups never mix temporal rows with different bridge classes and that restoration preserves exact rows and their original class assignments. Not activated: token reduction is rejected in bridge mode; no pooling/restoration support is claimed.

* [DONE] T061: Add a three-or-more-segment bridge-chain test using the same scene and character but progressively changing actions to verify that the mechanism remains stable beyond a single continuation boundary. Three stationary bridge pose segments pass state/AV and primary-pose review; the four-segment locomotion stress case fails when the character leaves the camera frame.

* [DONE] T062: Add documentation explaining that bridge mode is intended primarily for same-scene meaningful action changes and that hard mode remains preferable when the prompt and motion trajectory change only minimally.

* [DONE] T063: Add prompting guidance recommending transition language such as completing the current motion, gradually changing pose, then beginning the new action instead of describing an immediately replaced target pose or scene state.

* [DONE] T064: Document that bridge continuation does not guarantee continuity for intentionally incompatible scene, identity, environment, or reference changes and that such cases require a different continuation strategy.

* [OPEN] T065: Mark bridge continuation complete when hard-mode regressions pass, AV alignment and exact-prefix invariants are verified, multi-segment chaining succeeds, and bridge-39 demonstrates materially fewer same-scene action/pose discontinuities than hard-39 over the first 1–2 seconds of newly delivered output.
