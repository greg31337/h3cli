# M0/M1 implementation and measurements

Validation date: 2026-09-20. **M0 is complete**, including B1/B2/B5/B6 and the
long-lived-context cache regression. M1's standalone native attention is numerically
accepted, but its production performance gate failed; DiT integration remains
blocked. `--backend metal` reports that gate explicitly. MPSGraph remains the
default, with its existing kernels and normal generation settings.

## Reproduction and retained evidence

The machine is an Apple M4 Max with 128 GiB RAM, GPU family 9 / applegpu_g16s.
Xcode is installed at `/Applications/Xcode.app`; commands use a scoped
`DEVELOPER_DIR`. The runtime Metal compiler builds the native prototype, so the
separate offline Metal Toolchain component is unnecessary.

The production baseline is FL2VA, 640×480, 362 frames, 50 blocks, BF16
QKV/output/MLP weights, dense attention, CPU Euler, seed 42 and reuse intervals 1.
The prompt is “A person in a blue jacket walks through a sunny park. Birds sing
softly.” The main attention shape is S=33,322, H=56, D=128. B1/B2/B5 use the
preview VAE; final B6 requests the production VAE. Every process is limited to
at most six denoising evaluations; preparation-only checkpoints execute zero.
Desktop GPU activity is visible in the trace, and GPU clocks were not locked.
These are measurements of this validation system, not claims of peak hardware
performance. The M1 decision means a production gain was not demonstrated.

The [benchmark runner](../../scripts/metal_native_bench.py) and
[save-once/load-many suite](../../scripts/metal_native_suite.py) retain binaries,
commands, relevant environment, source hashes, device/OS/compiler/Xcode metadata,
cache/model identity, result hashes, logs and memory samples. Reusing a suite
record requires matching command, binary, runner, environment and cache.
Evidence is in the ignored [outputs/metal-native](../../outputs/metal-native/)
directory; it is not bundled into source control.
The command-split baseline uses a frozen executable with a
[verified source snapshot](../../outputs/metal-native/m0-command5-bin/build-provenance.json).
Per-run working-tree hashes describe files at launch and may include subsequent
host-cache fixes; the frozen binary hash and that snapshot identify its actual
implementation. The later live-context identity fix receives separate targeted
validation and does not change the denoising kernels.

## Conditioning correctness

The [format specification](conditioning-format.md) describes invariant raw
conditioning, refined text, optional complete-schedule AdaLN, identity matching,
atomic writes and strict parsing. Loading a matching cache bypasses tokenizer,
Qwen text/vision, reference encoders and text refinement. A schedule miss retains
invariants and rebuilds AdaLN unless exact scheduled records were required.

* Full-size FL2VA B1 cached and uncached AV states are byte-identical. Both video
  and audio relative L2 error are zero. The AV SHA-256 is
  `9655bab36c6bbb6d0b7199eadf64dcc91f0ff72f37f8a1b4c3c4c84cdb71bd2e`.
* Final cache-schema validation at the production DiT input boundary produced
  identical checkpoint sections 5–13: conditioning, references, positions/layout,
  full sigmas and initial video/audio latents.
* Ref2VA with an image reference, 256×256 / 22 frames / all 50 blocks also produced
  byte-identical cached and uncached B1 AV states.
* The early full-size AV pair predates the added device-identity fields; its exact
  executable and cache are retained. The final boundary and Ref2VA checks use
  the stricter device-aware identity.
* A model-backed long-lived-context regression replaced an image while retaining
  the same path, byte length and mtime. The changed reference invalidated both
  live conditioning and prepared DiT state. Warm and fresh-context raw F32 video,
  raw BF16 text and refined BF16 text were exact. Three preparation-only requests
  executed zero denoising evaluations.

Results: [FL2VA comparison](../../outputs/metal-native/conditioning-b1-comparison.json),
[final DiT boundary](../../outputs/metal-native/conditioning-boundary-comparison.json),
[Ref2VA comparison](../../outputs/metal-native/ref2va-conditioning-comparison.json).
[Live-context regression](../../outputs/metal-native/context-regression-final-record/record.json)
retains its binary, command, source hashes, logs and artifact checksums.
The cache removes conditioning startup work; variation between denoising timings
is not attributed to that cache.

## Production B1 profile

The ordinary cached B1 denoising step took **226.526 s**; uncached B1 took
241.187 s. A separate component-fenced B1 with a bounded Metal System Trace took
242.779 s. Component fences and tracing make this a diagnostic run, not the
ordinary steady-state performance baseline.

| Component | Completed wall time | Share of diagnostic step |
| --- | ---: | ---: |
| Dense attention | 147.390 s | 60.7% |
| MLP | 56.801 s | 23.4% |
| QKV projection, normalization and RoPE | 27.991 s | 11.5% |
| Attention output projection | 9.850 s | 4.1% |
| Gate / next-attention AdaLN | 0.344 s | 0.14% |
| Gate / MLP AdaLN | 0.292 s | 0.12% |

The diagnostic step executed 50 MPS attention calls and 200 MPS linears. At the
step boundary Metal reported 67.37 GiB allocated, tracked tensors 39.60 GiB,
resident memory 42.75 GiB, physical footprint 43.30 GiB, process compression zero,
system compressor 1.04 GiB and swap 2.44 GiB. These are distinct measurements,
not quantities to add together. The unfenced cached baseline had substantial
process compression, so memory pressure is a material performance variable.

The successful [Metal trace](../../outputs/metal-native/m0-trace-window/metal.trace)
attached for 15 seconds while a complete production-shape B1 continued. Its
[GPU interval summary](../../outputs/metal-native/m0-trace-window/gpu-summary.json)
contains 2,724 top-level H3 compute intervals, totaling 9.58 seconds of GPU busy
time within a 14.11-second observed H3 span. This measures busy time, not SIMD
occupancy or achieved memory bandwidth. The standard trace template captures
Metal shader/driver activity; per-component completed timings come from the
explicit profiling fences. Custom signposts are emitted under `org.h3.native-metal`.

A whole-process Instruments launch was killed by macOS under memory pressure.
The failed run is retained separately and is not accepted evidence. Two later
ordinary B2 attempts also hit memory pressure with only about 17 GiB disk free;
swap pressure drove free space below 400 MiB before macOS killed H3. The second
attempt used a separately prepared complete-schedule cache, which did not solve
the retained-temporary-memory problem. The existing
`H3_DIT_COMMAND_BLOCKS=5` override completed production B1 in 239.565 seconds with
**byte-identical complete AV state**, zero process-compressed memory at its final
step boundary and 53.71 GiB current Metal allocation. See the
[exact comparison](../../outputs/metal-native/command5-exact-comparison.json).
It uses ten command-buffer submissions per 50-block step rather than the default
two. This is an explicitly selected benchmark memory policy, not a change to
normal user defaults or a claimed speed optimization. The subsequent B2/B5/B6
reference measurements use that same override and separately prepared schedule
caches. Future candidate comparisons must use the same policy.

The completed B2 warm-up case took 240.703 s for step 1 and 241.415 s for step 2
(482.118 s total denoising). Both steps reported zero process compression and
50 MPS attention / 200 MPS linear calls. There is no substantial second-step
speedup in this run. [B2 record](../../outputs/metal-native/baseline-b2-command5/record.json).

### B5 steady-state baseline

| Step | Completed denoising time |
| --- | ---: |
| 1, reported separately | 241.713 s |
| 2 | 244.634 s |
| 3 | 243.400 s |
| 4 | 242.633 s |
| 5 | 241.676 s |

For steps 2–5: **median 243.017 s**, mean 243.086 s, minimum 241.676 s and maximum
244.634 s. Total denoising was 1,214.055 s; process wall time including cached
startup, preview video and audio delivery was 1,225.385 s. All five steps passed
the evaluation contract, used 50 MPS attention / 200 MPS linear operations and
reported zero process-compressed bytes. Current Metal allocation stayed at
53.78 GiB across the step boundaries; physical footprint stayed near 43.48 GiB.
[Complete B5 record](../../outputs/metal-native/baseline-b5-command5/record.json).

### B6 final baseline

| Step | Completed denoising time |
| --- | ---: |
| 1, reported separately | 235.963 s |
| 2 | 226.632 s |
| 3 | 236.139 s |
| 4 | 239.909 s |
| 5 | 239.962 s |
| 6 | 240.269 s |

For steps 2–6: **median 239.909 s**, mean 236.582 s, minimum 226.632 s and maximum
240.269 s. Total denoising was 1,418.873 s; complete process wall time was
1,601.614 s. Production video VAE decoding took 171.895 s, separate from its
1.534-second load; audio VAE took 0.609 s and FFmpeg delivery took 0.860 s.
Every evaluation reported 50 MPS attention / 200 MPS linear operations, zero
native attention operations and zero process-compressed bytes.

The result contains 362 H.264 video frames at 640×480 / 24 fps and stereo AAC
audio at 32 kHz, lasting 15.083 seconds. All saved video/audio latent values are
finite and result checksums match the run record. A sampled decoded frame shows
the prompted blue-jacketed person in a park. This is MPSGraph pipeline validation;
no native DiT generation or human native-model quality qualification is claimed.

Evidence: [B6 record](../../outputs/metal-native/baseline-b6-command5/record.json),
[delivery checks](../../outputs/metal-native/baseline-b6-command5/delivery-check.json),
[production clip](../../outputs/metal-native/baseline-b6-command5/result.mp4),
[sampled frame](../../outputs/metal-native/baseline-b6-command5/frame-7s.png).

## Block and sequence scaling

| Active blocks, 362 frames | B1 denoising time |
| --- | ---: |
| 1, clean run | 4.950 s |
| 5 | 24.214 s |
| 50 | 226.526 s |

These are first-N-block diagnostics, not useful quality samples. The one-to-five
block slope is 4.816 seconds per additional block with a 0.134-second intercept.
It is an approximate local decomposition: kernel compilation, command grouping,
memory pressure and thermal state prevent extrapolating it as an exact 50-block
model. The earlier one-block QKV-capture run is excluded from this fit.

| Frames at 640×480, all 50 blocks | B1 denoising time |
| --- | ---: |
| 22 | 6.807 s |
| 124 | 51.398 s |
| 362 | 226.526 s |

Sequence-length growth increases dense attention work quadratically while linear
layers grow approximately linearly. The measured mixture is consistent with
attention becoming the main production-shape target. Each case executes exactly
one evaluation; frame count changes the workload, not the evaluation budget.

## Native attention qualification and M1 gate

The prototype vendors the MIT-licensed MLX attention template at commit
`59d600b5e64c238427d0f8d897ab7c682ef4d3d2`, with checksums and license retained in
[third_party/mlx-attention](../../third_party/mlx-attention/README.md).
It uses tiled BF16 storage, FP32 QK/PV accumulation and FP32 online softmax without
materializing S×S scores. Both sequence-major and head-major inputs and outputs
are supported directly through strides. MPSGraph remains the independent GPU
reference; short cases also check distributed positions with a double-precision
CPU oracle. Tests cover tails, zero/constant/outlier inputs, guards and input
immutability.

| Production S=33,322 workload | MPSGraph median | Best retained native tile | Relative L2 |
| --- | ---: | ---: | ---: |
| Bounded random QKV, sequence-major | 2.957733 s | 3.143960 s | 0.001053 |
| Real H3 block-0 QKV, head-major input | 2.954249 s | 3.033273 s | 0.000518 |

The microbenchmark's MPS reference always starts with sequence-major inputs and
returns sequence-major output. In the real-QKV case only the native input is
prearranged head-major; that rearrangement is outside native timing. Thus this
case already gives the native path the benefit of direct head-major preparation,
but it still does not establish a speed gain. The random production case uses
matching sequence-major input/output layouts for both implementations.

The best retained tile is BQ=64, BK=32, 256 threads. It is **6.3% slower** on
random production inputs and **2.7% slower** on real H3 inputs. Larger tiles,
register-resident Q, alternative BF16/FP16 matrix fragments, fast math and layout
experiments also failed to establish a production gain. The retained code uses
the simpler unchanged FP32-accumulator template.

All seven final numerical cases passed their fixed component gates. Synthetic
inputs require relative L2 ≤0.01 and max absolute error ≤0.05. Real inputs require
relative L2 ≤0.01 and max absolute error ≤0.02×max(1, reference peak), established
before testing that case. Real-QKV max absolute error was 0.0625. These are
component results, not a six-step native-model quality claim.

Evidence: [production random](../../outputs/metal-native/native-final-production.json),
[real QKV](../../outputs/metal-native/native-final-real.json),
[commands and checksums](../../outputs/metal-native/native-final-manifest.json).

T035 requires a meaningful production-shape gain before integration. Consequently
T032 and T036–T039 remain blocked: no hybrid/native DiT B1/B5/B6 speedup or native
clip quality is claimed. `bin/metal_attention` remains available for further kernel
work; `--backend metal` cannot silently fall back to MPSGraph.

The standalone production-shape test can be reproduced without model weights:

```sh
make bin/metal_attention
./bin/metal_attention 33322 3 random 0 0 > attention-production.json
```

Its arguments are sequence length, measured repetitions, input pattern, input
layout and output layout (0 sequence-major, 1 head-major). The benchmark includes
one separately reported warm-up per implementation and performs attention operations only,
not complete denoising evaluations. Real-QKV commands and checksums are retained
in the final manifest linked above.

## Host validation and external reference

The final host checks passed: 32 conditioning/backend assertions, 36 malformed
conditioning containers, 13 CLI preflight cases, 1,561 sampler assertions,
17 sampler-container tests, 165 adversarial sampler files and 41 sampler CLI
cases. Conditioning/file tests also passed ASan and UBSan. Build and whitespace
checks passed. The full historical GPU test suite was not rerun.

[VPIPE reference measurements](vpipe-reference.md) document upstream M4 reports
and the hardware, geometry, precision and decoder differences. They are external
context, not a matched local speedup denominator.
