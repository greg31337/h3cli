# Bridge motion-quality and sampler validation

The acceptance harness compares hard-39 with bridge-39 using the same clean
source state, seed, prompt, face/body references and cached conditioning. Every
quality render uses all 50 blocks and 20 Euler steps at 256 square. Each target
saves the complete 90-frame AV state and delivers 51 new frames (2.125 seconds).

The five cases cover an unchanged walking prompt, looking upward while walking,
finishing a step and turning into a run, gradually raising both arms, and turning
left to walk across the path. The running transition is the primary case. The
source and action prompts are recorded verbatim in
[`run_bridge_quality.py`](../tests/run_bridge_quality.py).

## Reproducible experiments

Install the optional analysis dependencies in a workspace environment. They are
not runtime dependencies of h3cli:

```sh
python3 -m venv outputs/bridge-quality/venv
outputs/bridge-quality/venv/bin/pip install -r tests/requirements-bridge-quality.txt
make -j8 all bin/continuation_generate bin/bridge_tests bin/bridge_gpu_tests
make test-bridge-quality BRIDGE_PYTHON=outputs/bridge-quality/venv/bin/python
```

Run full-model jobs **sequentially**. They need local MiniMax-H3 weights, Metal,
FFmpeg/FFprobe, `inputs/face1.jpg` and `inputs/body1.jpg`. Create a full-body
walking source and run the declared pilot:

```sh
python3 tests/run_bridge_quality.py --phase source --seed 71
python3 tests/run_bridge_quality.py --phase sweep --seed 72
outputs/bridge-quality/venv/bin/python tests/bridge_quality_metrics.py --require-complete --select
```

The pilot contains 17 bridge settings plus hard-39. It covers lengths
`4, 6, 8, 9, 10`, maximum strengths `0.25, 0.40, 0.50, 0.65`, and all three
profiles. `--full` expands the sweep to the complete 60-cell Cartesian grid;
`--list` prints the planned cases without rendering. Use separate output
directories for separate experiments. A running batch uses a frozen executable
and shader copy, so later repository edits cannot change its implementation.
Manifests record configuration, source/reference hashes, executable/shader
hashes, timings and output hashes. `--resume` only reuses matching, verified
results. The source is never modified.
Separate cold multimodal-encoder calls are not guaranteed byte-deterministic.
Paired comparisons share cached conditioning in one process; frozen code and
configuration alone do not promise identical renders across independent runs.

`--select` freezes the lowest-scoring candidate that passes the stability
guards in `selection.json`. This is a choice for confirmation testing, **not a
quality endorsement**. Finish the other action pairs, then test the primary and
unchanged cases on the held-out seed:

```sh
python3 tests/run_bridge_quality.py --phase actions --only unchanged,look-up,arms,turn-left \
  --selection outputs/bridge-quality/acceptance/selection.json
python3 tests/run_bridge_quality.py --phase actions --only run,unchanged --seed 73 \
  --selection outputs/bridge-quality/acceptance/selection.json
outputs/bridge-quality/venv/bin/python tests/bridge_quality_metrics.py --require-complete
```

## Measurements and review

The fixed [quality protocol](bridge-quality-protocol.json) specifies the subject
and background regions, flow parameters and acceptance thresholds. The source
clip was inspected to place those regions; thresholds were recorded before
examining target quality measurements. Seed 72 selects a candidate; seed 73
tests confirmation without retuning.

Each comparison extracts **24 final source frames and 48 first delivered
frames**. `review/` contains all 72 PNGs, a labeled contact sheet, a three-second
comparison window and an HTML gallery. Frame `+0` is the first newly delivered
frame; the duplicated context is not counted as new output.
Add `--plots` to export paired diagnostic curves alongside the review gallery.

The optional diagnostics use OpenCV's
[Farneback dense flow and pyramidal Lucas–Kanade feature tracking](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html).
Background feature tracks estimate camera motion. Subject-region flow after
camera compensation measures motion magnitude/direction and motion-compensated
acceleration/jerk. Reports also include background deformation, luminance,
encoded-Lab color change, contrast, sharpness and tracking coverage. Color
distance is measured in OpenCV's encoded Lab coordinates, not calibrated ΔE.

The continuity score combines acceleration and jerk, with the components
reported separately in pixels/frame² and pixels/frame³. It is a diagnostic
score, not a physical measurement or an action/identity classifier. Lower flow
can result from frozen or blurred output, so acceptance also requires retained
motion/sharpness, stable background/camera/color/contrast, and visual review.
Fixed regions have an additional limitation: in the running fixture the
character enters the left background region near the end of the window.
Its late residual-flow spike therefore includes foreground motion and must
not be presented as proof of background warping. The declared regions and
thresholds are retained for the paired experiment; inspect the frames and
camera feature tracks when interpreting those curves.

The primary case must improve by at least 10% on both seeds. The changed-action
geometric mean must also improve by at least 10%; unchanged-prompt output must
pass its non-degradation limits. Every selected comparison needs a
`visual-review.json` entry with the hard/bridge RGB hashes, reviewer, method,
notes, and `pass` for each field listed in the protocol. Missing, stale, failed
or uncertain reviews prevent acceptance. `--require-quality` returns failure
unless all those gates pass. A failed gate leaves bridge unqualified; numerical
correctness alone does not complete T065.

## Samplers and chaining

After validating the CPU implementation, compare equivalent CPU and GPU
settings in a separate output directory:

```sh
./bin/bridge_gpu_tests outputs/bridge-validation/encoded-inputs
make test-bridge-sanitize test-continuation-sanitize test-bridge-gpu-sanitize
python3 tests/run_bridge_quality.py --phase samplers --output outputs/bridge-quality/samplers \
  --selection outputs/bridge-quality/acceptance/selection.json
outputs/bridge-quality/venv/bin/python tests/bridge_sampler_metrics.py outputs/bridge-quality/samplers
python3 tests/run_bridge_quality.py --phase chain --chain-style pose --output outputs/bridge-quality/chain-pose \
  --selection outputs/bridge-quality/acceptance/selection.json
outputs/bridge-quality/venv/bin/python tests/bridge_quality_metrics.py outputs/bridge-quality/chain-pose --require-complete
```

The sampler matrix includes reuse 1/2/3, core reuse 4/6, and GPU windows with
callbacks disabled. It compares equivalent settings bit-for-bit, including all
21 captured AV states where callbacks are enabled. This is stricter than the
existing DiT parity tolerances. Reuse configurations are approximate relative
to reuse 1; matching CPU/GPU results do not imply identical action quality
between different reuse settings.
The paired outputs must also have identical decoded RGB/stereo PCM, finite
audio, and aligned 51-frame AV delivery.

The small GPU suite tests actual video patch packing and both stereo timelines,
nonzero condition offsets, all profiles, exact signed zeros/NaN payloads,
fresh/reused/extrapolated raw velocities and deliberate exact-row corruption.
Its optional input directory reuses the stored VAE-encoded face, body and
`2.jpg` numerical fixtures without decoding new media.

The default chain consists of four segments with three bridge boundaries:
walking, stopping and raising arms, lowering arms and looking left, then looking
back and bringing hands together. These stationary pose changes keep the
character visible to the locked camera. `--chain-style locomotion` retains the
walking → running → walking → raising-arms stress case. That case can leave
the camera's view before later actions, which is a failed visible-action test,
not proof of successful chaining. Preserve separate experiment directories.
Each new
segment loads its predecessor's complete state, retains the face/body references
and records parent/child hashes. Initial noise, exact-prefix invariants, saved
state checksums and AV output lengths are checked on every segment.
The completion gate also requires a `visual-review.json` entry for each child
segment, with source/target RGB hashes and the same eight visual checks used
above. A valid saved-state chain alone does not establish visual stability.

Token reduction remains disabled. T060's conditional grouping/restoration work
is not activated: no temporal pooling path is claimed to support bridge classes.

`python3 tests/bridge_completion.py outputs/bridge-quality --chain-directory chain-pose --require-complete`
combines quality, CPU/GPU/reuse, chaining, and pre-bridge hard/normal regression
reports. It verifies artifact hashes and fails if any required gate is absent
or unsuccessful. Its directory expects `acceptance/`, `samplers/`, the explicitly
selected chain directory (default `chain/`),
`hard-regression/` and `unmasked-regression/` from the commands above and the
existing regression harnesses.
