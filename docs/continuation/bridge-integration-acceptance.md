# Bridge integration acceptance: T021–T042

The implementation uses the CPU F32 Euler sampler with the Metal transformer
and VAEs, guided by [design-bridge.md](design-bridge.md). The
[usage guide](bridge-continuation.md) describes configuration and reproduction.
This report covers numerical integration and compatibility, not the motion
quality acceptance scheduled for T043 onward.

Hardware: Apple M4 Max, 128 GiB unified memory. Full-model tests use the local
released MiniMax-H3 weights, 256×256 targets, 20 Euler transitions, all 50 DiT
blocks, `--reuse 1 --core-reuse 1`, and no token reduction. Bridge tests set
`H3_GPU_SAMPLER=1` to verify the explicit CPU continuation fallback.

## Task coverage

| Tasks | Implementation and check |
|---|---|
| T021–T024 | Joint audio/video bridge initialization, direct exact-audio copies, existing target RNG tensors, separate hard path; independent F32 value oracle and zero-strength parity |
| T025–T027 | Authoritative video/stereo packing, class row maps, materialized effective-timestep AdaLN; host mapping tests and released-weight vector comparison |
| T028–T031 | CPU Euler applies fractional scaling once; exact AV bytes checked every step; per-class velocity/update measurements and signed-zero host tests |
| T032 | Paired normal/debug renders with cached conditioning; identical full state, decoded RGB suffix, and stereo PCM suffix |
| T033–T034 | Mode/context/interval/mask/packed-row logs; construction, initialization and modulation timings; map, vector, class-plan and audit-snapshot memory |
| T035–T037 | Unchanged v1 state checksum/geometry; pre-bridge hard state → bridge, bridge → bridge, and bridge → hard; full final AV state saved/reloaded and borrowed source unchanged |
| T038–T039 | Pre-bridge `6c09804` versus current normal T2VA/FL2VA/Ref2VA and default/explicit hard continuation; final AV latent, state, and MP4 byte comparisons |
| T040–T041 | Two face/body pairs, image/video, image/audio, video with embedded audio, replacement audio, and zero-reference T2VA using the same history state |
| T042 | Constant face/body references, source and seed; unchanged walking prompt versus turning/raising-arms prompt; subsequent changed-action state handoffs |

## Validation results

| Check | Result |
|---|---|
| Bridge host tests | PASS: 16,544,087 assertions |
| Bridge ASan/UBSan | PASS: same expanded host suite |
| Existing continuation host and ASan/UBSan | PASS: 103,190 assertions each |
| Existing `make test` | PASS: host tests, binary-mask oracle, AudioVAE Metal primitives and FFmpeg mux; optional external parity fixtures absent and skipped |
| Real AdaLN vectors | PASS: 2,295 vectors across all 50 blocks and final heads; zero observed relative maximum and L2 error |
| Five encoded image fixtures | PASS: 11,351,317 assertions; 135 profile/strength/sigma combinations |
| CLI validation | PASS: 36 invalid/unsupported configuration cases, no partial outputs |
| Full render matrix and paired checks | PASS: 14 renders, each with 20 transitions and all 50 blocks |
| Normal T2VA/FL2VA regression | PASS: both original/current pairs have identical final AV latents and MP4 bytes |
| Unmasked Ref2VA regression | PASS: final AV latents, full state, and MP4 bytes match the pre-bridge baseline |
| Default/explicit hard regressions | PASS: final AV latents, full states, and MP4 bytes match the pre-bridge baseline, including nondefault bridge tuning in hard mode |
| Generated source-state initialization | PASS: all 27 profile/strength/sigma combinations |
| Positive CLI render | PASS: ease-out profile, maximum strength 0.4, 20 transitions, valid full state and synchronized 51-frame AV output |
| Final build/source/whitespace checks | PASS: binaries current, no compiler warnings or whitespace errors; isolated regression sources verified |

The real AdaLN comparison evaluates each active class at three sigma pairs,
including near-terminal sigmas. Its reference uses the original materializer
at the same effective timestep. It compares all materialized BF16 values with
the existing 2.5% parity bounds; observed error was zero.

The input fixture suite encodes `face1.jpg`, `body1.jpg`, `face2.jpg`, `body2.jpg`,
and `2.jpg` with the actual Ref2VA VAE. Repeated image latents with zero audio
are explicitly labeled synthetic numerical fixtures; no media is decoded from
them. Each exercises all three profiles, three strengths, and three starting
sigmas. Real video/audio renders use the clothed face/body references and
existing video/audio fixtures.

## Render artifacts and measurements

The integration matrix writes each full `.h3av` state, MP4, delivered RGB,
planar stereo F32 PCM, metadata, and log under
[`outputs/bridge-integration/acceptance`](../outputs/bridge-integration/acceptance/).
The [run manifest](../outputs/bridge-integration/acceptance/runs.json) records
commands, binary/source/input/output hashes, return codes, and durations.
The [metrics report](../outputs/bridge-integration/acceptance/metrics.json) records
state checksums and geometry, callback counts, exact-row audits, class scaling,
packed row counts, AV timing, and paired comparisons. The
[review page](../outputs/bridge-integration/acceptance/review/index.html) contains
clips and sampled frames, including the debug context and its endpoint.

The source state used in this run is
`outputs/bridge-validation/regression/hard-before.h3av`. Its prior passing
manifest, original binary, and recorded conditioning/latent/state/MP4 hashes
were verified against baseline `6c098047584cfc5c9a1853965f08dbd8a9ad63b3`.

With 90 raw frames and 39 context frames, normal output has 51 frames and
68,000 samples per channel (2.125 seconds). Debug output has 90 frames and
120,000 samples per channel (3.75 seconds). Both retain the complete 90-frame
state; their decoded RGB and PCM suffixes must match exactly.

The default smoke run used 255 time rows (214 beyond hard), 2,075,437,056 extra
AdaLN bytes, a 3,672-byte class plan, and a 701,952-byte exact-row audit snapshot.
The packed modulation map has zero incremental allocation over hard mode.
Profile-specific timings and memory are recorded in the metrics report.

All 14 matrix renders passed state-format/checksum, finite-latent, callback,
source-immutability, packed-class-count and AV duration/start-time checks. Exact
video/audio rows remained byte-identical after every bridge step. The largest
observed velocity-scale ratio error was `2.8082e-8`; generated classes retained
their raw velocity RMS exactly. Zero-strength bridge and hard mode produced
identical full `.h3av` bytes, decoded RGB, and decoded PCM.

Normal/debug-prefix pairs produced identical complete state bytes, RGB suffix,
and planar stereo PCM suffix. The changed-action prompt used the same source,
seed, and face/body references as the unchanged-prompt case. Their exact rows
matched, while generated video/audio latent RMSE was respectively `0.588827`
and `0.127284`. This confirms changed conditioning reaches the generated output;
it is not a motion-quality score.

## Regression method and limits

The original and current binaries are built in isolated directories. Normal
T2VA and FL2VA are compared with `tests/continuation_regression.py`; unmasked
Ref2VA, default hard, and explicit hard with otherwise unused bridge tuning
are compared with `tests/bridge_regression.py`. The latter compares five real
20-step render artifacts and runs a generated source-state initialization test.
This invocation uses `--reference outputs/bridge-validation/regression` to reuse
the two verified, unchanged original renders and rebuilds and reruns all three
current cases. Reference reuse checks commit, input hashes, steps, exact command
arguments, binary hashes, and every conditioning/latent/state/MP4 artifact hash.

Captured text/reference conditioning is replayed to isolate pre-existing cold
multimodal encoder variability. This tests the changed initialization,
modulation, sampler, decode, and output paths under identical conditioning.
It does not claim that separate cold multimodal encoder runs are deterministic.
Replay instrumentation exists only in the isolated regression copies.

The [normal-generation regression results](../outputs/bridge-integration/unmasked-regression/results.json)
and [Ref2VA/hard regression results](../outputs/bridge-integration/hard-regression/results.json)
include exact commands, elapsed times, binary hashes, and artifact hashes.
The [source audit](../outputs/bridge-integration/source-audit.json) verifies that
the isolated current sources match the workspace, with only the stated replay
hooks added, and that the original sources match `6c09804`.

The [CLI results](../outputs/bridge-integration/cli.json) record the 36 validation
cases and successful direct CLI render. The latter has no latent-step callback,
so it also verifies that continuation itself selects CPU Euler when
`H3_GPU_SAMPLER=1`. Its 51 video frames and audio both last 2.125 seconds and
start at timestamp zero; the saved v1 state retains all 90 raw frames.

At completion of this stage, T043 onward remained open. These checks do not establish improved motion quality,
recommended tuning from a sweep, arbitrary multi-segment stability, GPU-state
bridge parity, or support for bridge reuse greater than one.
Subsequent work is recorded in [T043–T065 validation](bridge-quality-acceptance.md).
