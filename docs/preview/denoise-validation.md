# Denoised preview validation

Validation uses the M4 Max with 128 GiB unified memory and the installed
MiniMax-H3 weights. CPU state and explicit GPU state are tested on this machine;
these results do not claim an M5 hardware run.

## Reproduce

Run GPU jobs sequentially, with Metal device access:

```sh
make test
make test-preview-sanitize
make bin/preview_generate bin/preview_decode
python3 tests/denoise_generation.py
python3 tests/denoise_generation.py --only cpu-chain,gpu-chain
python3 tests/denoise_generation.py --quality --only cpu-fl2va,cpu-image,cpu-video,cpu-hard,cpu-bridge
python3 tests/denoise_analyze.py --decode
python3 tests/denoise_generation.py --default --only cpu-fl2va,gpu-fl2va
```

The analysis script requires NumPy and Pillow. The local validation uses
`outputs/refvideo-encoder-validation/venv/bin/python` for that script. Generation
records completed cases in `outputs/denoise-validation/*/results.json` and only
reuses records with matching binary/shader hashes and command. Use `--force`
to repeat a matching case or after changing fixture contents. Each record
contains the binary/shader hashes and complete command.

## Coverage and interpretation

`test_preview.c` checks the Euler identity over 100,000 randomized inputs,
zero-sigma final equality, zero-velocity prefix equality, parser errors, and 36
Metal parity cases. The latter cover ordinary, hard, and fractional bridge
masks with fresh and extrapolated BF16 velocities. They check max absolute
error, RMSE and relative L2, untouched sample/history buffers, packing offsets,
and unchanged live/cumulative GPU allocation counters.

`denoise_sampler.py` substitutes deterministic BF16 DiT predictions into an
isolated copy of the production sampler. It retains the actual Euler loops,
Metal preview operation, continuation masks, reuse histories, raw/display
callbacks, pause/resume and cancellation. CPU/GPU, ordinary/hard/bridge and
reuse 1/2/3 form 18 combinations. This test isolates sampling semantics; it
does not replace the released-weight generation tests.

Released-weight invariance runs use all 50 DiT blocks, seed 72, 128-square
output, 56 frames and six Euler steps. They compare every video/audio latent
boundary, full MP4 files, completed `.h3av` files, completed and paused
`.h3sample` files, and final output after cross-mode resume. FL2VA also tests
preview cancellation and low-memory cancellation. The chained cases continue a
hard-continuation output using bridge continuation for a third segment.

Quality runs use all 50 blocks, 256-square output, 56 frames and 20 steps. Inputs
are `face1.jpg`, `2.jpg`, and a deterministic pan video derived from
`body1.jpg`. Prompts cover dialogue, a multi-subject scene, walking/turning,
hard continuation and bridge continuation. Test-only instrumentation records
raw states, effective velocities and delivered clean estimates. Independent
Float64 arithmetic checks the formula and the pre-/post-Euler identity.
Offline VideoVAE decodes compare noisy and denoised previews from that same
trajectory; their denoised pixels must match the public callback exactly.

Performance compares cached noisy/denoised runs, excluding initial model load.
Raw-boundary timing excludes preview decode; preview timing includes the
display construction/readback and decoder. Process peak RSS is cumulative and
includes allocator/cache effects, so it cannot isolate a small preview buffer.
The direct allocation checks and workspace audit establish that construction
adds no persistent full-latent allocation. The existing preview VideoVAE's
memory cost remains.

The on-stop full-clip decode intentionally uses the raw stored pause state;
these tests concern live per-step previews.

## Released-weight invariance results

All ten primary cases passed exact video/audio trajectory, MP4, AV-state,
checkpoint and resume comparisons. The times below sum the six denoising
steps, excluding preview decoding and initial loading:

The primary matrix used explicit modes before default promotion. The final
default-mode runs repeat FL2VA on both samplers with the final build; binary
hashes are recorded separately so these validation stages remain distinguishable.

| Case | CPU noisy / denoised (s) | GPU noisy / denoised (s) |
| --- | ---: | ---: |
| FL2VA, reuse 1 | 9.758 / 9.740 | 9.846 / 9.836 |
| Image reference, reuse 2 | 6.678 / 6.609 | 6.681 / 6.744 |
| Video reference, reuse 3 | 13.157 / 13.240 | 13.181 / 13.136 |
| Hard continuation, reuse 2 | 6.549 / 6.603 | 6.540 / 6.559 |
| Bridge continuation, reuse 3 | 5.056 / 5.141 | 4.913 / 4.905 |

Changes range from -1.04% to +1.68%, consistent with run-to-run timing noise.
There is no material step-time regression in these measurements.

The initial randomized test measured max Euler-identity error `1.22070312e-4`,
RMSE `6.9463e-6`, and relative L2 `4.07817e-8` over inputs spanning magnitudes
`2^-10` to `2^10`. All 36 CPU/Metal preview parity cases had zero error in all
three metrics, including the bit-preservation assertions. Building a preview
of 1,032,192 elements took approximately 0.15 ms on CPU; the 128-square live
VideoVAE preview decode took roughly 230 ms. The elementwise operation is
negligible at these sizes.

The first checkpoint comparison exposed missing optional token/presentation
diagnostics on conditioning-cache hits; deep-copying that metadata fixed full
file equality. A development run also saw a transient trajectory mismatch in
a post-resume cancellation check. It did not reproduce in the repeated full
CPU run, GPU run, or final default-mode runs; cancellation checks still require
exact trajectory bytes and were not relaxed.

## Visual and independent-oracle results

All 100 captured clean estimates (five cases, 20 steps each) are byte-identical
to the independent post-Euler formula. Maximum difference from the equivalent
pre-Euler formula is `9.53674316e-7`. At all 35 selected decode boundaries,
offline clean-estimate pixels match the actual public preview callback exactly.
At step 20, noisy and denoised preview pixels are identical in every case.
The 256-square offline preview decode averages approximately 0.90 s.

Visual inspection of the matched galleries shows:

* Dialogue/FL2VA: the face, clothing and background are recognizable at step 1;
  early facial distortions settle progressively. Noisy previews remain
  noise-like through the displayed step 15.
* Multi-subject image reference: step 1 is still abstract. The room and people
  become recognizable around step 5, while later estimates refine faces and
  composition. The corresponding noisy images reveal little scene structure.
* Motion/video reference: early estimates remain distorted; people and the
  room become recognizable by step 10, materially before the noisy previews.
* Hard continuation: the prefix remains visually fixed, while the new suffix
  develops from an abstract first estimate into a recognizable person and
  window by step 10. Raw suffix previews remain noise-like.
* Bridge continuation: the partially re-noised leading prefix shows the face
  clearly in the clean estimate at step 1. The generated suffix evolves toward
  the later scene. The numerical checks require the exact bridge-scaled
  effective velocity, so this display follows the actual bridge trajectory.

The hard-prefix latent is exact at every step, and decoded prefix pixels at
steps 1, 10 and 20 match exactly. For these short continuation fixtures, the
existing representative-frame selector lands within the prefix. The additional
full-clip comparisons therefore include the first generated frame and the final
suffix frame, as well as the prefix. Preview frame selection itself is unchanged.

The local [comparison gallery](../outputs/denoise-validation/quality/index.html)
contains all five matched cases and links to the final videos.
The [arithmetic and pixel results](../outputs/denoise-validation/quality/analysis.json)
record each step's metrics. Early estimates are more useful, but are not a
promise of final composition or artifact-free intermediate frames.

## Final regression suite

`make test test-sampler test-refvideo` passed, including the legacy/reference
GQA race regressions (zero repeat mismatches), 653,390 preview numerical/parity
checks and 30,335,755 sampler preview checks with explicit, unset and empty
modes. The sampler CLI, adversarial checkpoint container and normalized/legacy
reference-video media tests also passed. Ten optional external-fixture groups
were skipped because their oracle fixtures are not installed; the released
model runs above executed with installed weights.

`make test-preview-sanitize` passed the numerical/Metal host, cache ownership
and complete sampler-loop tests under AddressSanitizer and
UndefinedBehaviorSanitizer. Both CPU and GPU three-segment continuation
regressions passed exact trajectory, output, checkpoint and resume comparisons.

The final released-weight default runs passed on CPU and GPU. Unset and empty
settings deliver the exact same preview pixels as explicit `denoised`; all five
off/noisy/denoised/unset/empty modes preserve output and checkpoint bytes.
Cross-mode resume, callback cancellation and low-memory cancellation injected
inside the second preview's real VideoVAE loop also passed in both modes.

`python3 tests/memory_cancellation.py --sanitize` passed against installed
weights: text/vision, audio encode/decode, video encode/decode, CPU/GPU Euler
and RES loops stop before the next execution after user or memory cancellation.
The resident decoder and public context remain usable after cancellation, and
populated conditioning/DiT caches are released correctly.

Logs are under `outputs/denoise-validation/`: `full-test.log`,
`final-sanitize.log`, `chain-run.log`, `default-run.log` and
`memory-cancellation.log`. Generated assets remain in that ignored output
directory; no model weights or generated videos are added to source control.
