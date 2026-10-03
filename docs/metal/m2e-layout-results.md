# M2E: norm/RoPE ranges and owned-buffer packing

M2E implements an optional layout path around the accepted mixed attention
recipe. `--metal-attention-layout fused` collects rounded BF16 range partials
inside cooperative norm/RoPE and packs dense Q/K/V in place. The retained
`adapter` path stays the default. MPSGraph remains the default backend.

The user explicitly selected this work after accepting the measured dense and
SOL results. ANE and the held-out conditioning/continuation renders stay deferred.
The earlier 1.10× dense speed gate and B6 perceptual screen failures remain in
[m2-reference-qualification.md](m2-reference-qualification.md); accepting them
is separate from passing those thresholds.

## Implementation

BF16 projection and sensitive model state are unchanged. Both paths use mixed
recipe 4: scaled half attention operands, FP32 accumulation, reported BF16/FP32
recovery, and a checked BF16 output commit. Fused preparation records extrema,
minimum positive magnitudes and invalid flags from the same rounded norm/RoPE
outputs. A reduction over seven 32-bit words per head/row applies the original
range policy. Dense packing reuses the owned Q/K/V storage, retaining BF16 bits
for recovery heads. SOL keeps BF16 summary inputs and separate half workspaces.

Scratch is context-owned. A context that previously ran the adapter or routed
SOL may retain those half allocations; the reported dense memory saving uses
fresh matched contexts.

No per-block CPU synchronization is added. Preparation is bound to buffers,
shape and scale, consumed once, and invalidated on cancellation. The existing
projection-compatible output layout and FP32 staging/checked commit remain.
There was already no output transpose to remove. Checkpoint attention version 5
and layout recipe 1 reject incompatible resume/prepared identities.

At S=22,426, H=56, D=128, each block replaces 964,497,408 logical scan bytes with
a write and read of 35,163,968 bytes of partial records. These are algorithmic byte counts,
not measured hardware memory-controller traffic. Dense in-place conversion
removes three scratch allocations, not the conversion's reads/writes. Exact
counts are retained in each `h3_layout` record.

## Validation scope

All model tests use 640×480, 243 frames and 50 blocks. GPU runs are serial; no run
exceeds six evaluations. Dense B1 captures block boundaries and sampler tensors;
dense B5 measures unfenced steady steps; dense B6 uses exactly six evaluations
and production VAE. SOL B5 measures the incremental/cumulative fusion effect
with minimum exact fraction .75, one early dense evaluation, high-noise threshold
.99, one early dense layer, K/V block 64 and local radius 1. SOL B6 is not added
to this bounded M2E scope.

The immutable build is `outputs/metal-native-m2e-243/final-frozen/`.
The runtime shader is explicitly pinned to that source snapshot. An initial
attempt correctly rejected an older conditioning cache because the shader
identity differed. A fresh cache was generated and reused for all matched runs;
its non-identity tensor payloads, first inputs and velocities match the previously
accepted dense result.
The rejected attempt and initial harness correction are retained under
`outputs/metal-native-m2e-243/development/` and `validation/`.

Twelve Metal API-validation operator cases pass byte-identical preparation,
range records and outputs, covering sequence tails, both output layouts, 56
heads, protected/mixed/all-exact SOL, zeros, recovery, nonfinite/subnormal
rejection, guards, source immutability, overflow shape rejection and preparation
lifetime/reuse. Invalid attention leaves the destination unchanged. Host tests
include fusion option validation, checkpoint round-trip/cache identity and all
Ref2VA/anchor/39–192-frame continuation protection layouts at a 243-frame target.
ASan/UBSan conditioning/protection checks pass. These host protection checks do
not qualify deferred conditioned renders. Full shader instrumentation was not
repeated; the prior SOL centroid instrumentation resource limit remains recorded.

## Measured results

The matched B1 captures, final AV state, sampler tensors and MP4 are byte-identical.
Its diagnostic denoise time is 124.086 s adapter versus 127.551 s fused (2.8%
slower). Tracked peak memory is 39.887 versus 39.022 GiB, a 0.866 GiB reduction.
B1 includes explicit region fences/capture and is not the steady-step speed gate.

Dense B5 also produces byte-identical AV state, MP4 and per-block range/recovery
records. Its steady median is **124.853690 s adapter versus 125.105889 s fused**
(0.997984×; approximately 0.2% slower). Steady ranges overlap slightly; there is
no measured incremental speed win. Denoising peak tracked storage is
**39.960 versus 39.095 GiB**, saving **0.866 GiB**. Both paths use 507 direct
dispatches, 51 blits and 10 submissions per steady step. No submission or wait
boundary was removed. Encoder-call wall time includes MPSGraph scheduling and
must not be interpreted as pure CPU computation.

Dense B6 passes: all **36 captured input/velocity/latent tensors**, final AV state
and production-VAE MP4 are byte-identical between adapter, fused and the
previously accepted M2 render. This preserves the user's accepted result on the
measured case; the earlier perceptual screen failure is not reclassified. B6
capture timings vary between runs and are not used to override B5's speed result.

Conservative SOL B5 also preserves **byte-identical AV state, MP4, range/recovery
records, routing decisions and protected/local counters**. There are 196 routed
blocks across the four steady evaluations; **83.341%** of block pairs are exact.
Fusion is slower here: **113.595533 s adapter versus 120.518556 s fused**
(0.942556×; 6.09% slower), with **41.208 versus 41.241 GiB** tracked denoising peak.
Both have 948 direct dispatches, 51 blits and 10 submissions per steady step.
The extra partial buffer remains resident alongside SOL's required half buffers.

The observed SOL-adapter B5 ratio over dense-adapter is **1.099107×**; adding
fusion reduces it to **1.035971×**. These are measured ratios from this serial
suite, not multiplied component claims or a new Reference+ benchmark. They do
not qualify SOL B6, the held-out corpus or M5. No new perceptual judgment is
asserted for SOL B5's preview-VAE clip.

**Disposition:** M2E implementation and selected validation are complete. Dense
fusion is an optional memory-saving path with no demonstrated speed improvement;
retain the adapter for the tested SOL workload. MPSGraph Reference+ and the
native adapter remain defaults. No failed speed/perceptual screen is promoted.
The B1 attention-region medians are 1.741798/1.748876 s and complete-block medians
2.506550/2.515672 s (adapter/fused). Attention-region work remains approximately
70% of block time; layout fusion does not justify starting broad native BF16
linears or Q8 work. ANE and broader qualification remain deferred.

## Reproduction and retained evidence

```sh
python3 scripts/metal_native_freeze.py --output outputs/m2e-fresh/frozen
python3 scripts/metal_layout_validate.py --frozen outputs/m2e-fresh/frozen \
  --output outputs/m2e-fresh/runs --conditioning outputs/m2e-fresh/conditioning.h3cond
python3 scripts/metal_layout_report.py --runs outputs/m2e-fresh/runs \
  --output outputs/m2e-fresh/review \
  --accepted-b6 outputs/metal-native-m2-243/continued/qualification/calibration/fp16
```

Use fresh destinations; the harness creates conditioning once and runs all GPU
jobs serially. The final report verifies executable/source, result and captured
state hashes before rendering HTML. The tested snapshot remains immutable; the
subsequent host-file edit only clarifies its ownership comment, and the workspace
build succeeds.

* [Final playback page](../../outputs/metal-native-m2e-243/review/review.html)
* [Verified report](../../outputs/metal-native-m2e-243/review/report.json)
* [Suite index](../../outputs/metal-native-m2e-243/runs/index.json)
* [Frozen build provenance](../../outputs/metal-native-m2e-243/final-frozen/build-provenance.json)
* [Conditioning payload comparison](../../outputs/metal-native-m2e-243/development/conditioning-payload-comparison.json)

