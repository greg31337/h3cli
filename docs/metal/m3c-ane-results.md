# M3C GPU/ANE QKV cooperation

M3C is closed as a measured prototype: the incremental performance
and quality promotion gates failed. The GPU-only default is retained.
This is a bounded QKV experiment on the local 128 GiB M4 Max, not a new default
or an all-ANE renderer. Every generation gate uses 640×480, **243 frames**,
all 50 blocks, and at most six evaluations.

## Source and API audit (T157)

The VPIPE source is pinned to
[`f34e2cc3a3adae759eea254419f436f5b7800057`](https://github.com/tgo-app-dev/vpipe/tree/f34e2cc3a3adae759eea254419f436f5b7800057).
The retained audit covers its graph emitter, CoreML loader, worker, module,
FFN/tier scheduler, H3 integration and tests. Original-file hashes are in
`m3c-ane-audit.json`; the working evidence is under
`outputs/metal-native-m3c-243/audit/vpipe/`.

Only `ane-emitter.h/.cc` is adapted into `third_party/vpipe-ane/`. It emits a
fixed-shape, tiled, runtime-weight matrix multiplication. Its dependency on the
VPIPE loader and its loader-dependent self-test were removed; h3cli supplies its
own numerical startup check. Apache-2.0 LICENSE, NOTICE and modification notices
are retained. No ANE private framework wrapper, all-ANE model, int8 path, VPIPE
worker pool or unrelated model infrastructure is imported.

Loading and prediction use public CoreML: `MLModel`, `MLMultiArray`, IOSurface
backings, `MLComputeUnitsCPUAndNeuralEngine` and `MLComputePlan`. However, the
emitter writes the **undocumented compiled CoreML MIL/container format**. This
remains an isolated research dependency, not a claim of a fully supported model
authoring API. Loading requires macOS 14.4 or later, all planned matmuls assigned
to the Neural Engine, and an exactly representable numerical layout self-test.
Public output access uses scoped `getBytesWithHandler`; permanently locking a
prediction output's IOSurface prevents its reuse on subsequent predictions.

VPIPE provides useful scheduling ideas, but h3cli's selected implementation is
smaller: a persistent serial ANE worker and independent row shards in QKV only.
MLP/output-projection offload is not implemented or claimed by this experiment.

## Arithmetic and bounds (T158)

The actual H3 QKV projection is `[rows,5376] × [21504,5376]ᵀ`; this 243-frame
workload has **22,426 packed rows**. Its suffix goes
to ANE; the prefix remains on MPSGraph. Compiled row chunks are 256/512/1024;
512 is the default. The default suffix budget is 4096 rows, rounded to chunks
and capped at half the sequence. The public maximum budget is 16384 rows.

Original/folded BF16 weights are never modified. Every invocation reads the
current weights and supplies a scaled FP16 execution copy; no quantized weights
are created. A power-of-two weight scale and per-input-row scales bound absolute
dot-product sums below 8192. Conversions must have relative squared error at
most `1e-8`, including tiny-value underflow. Nonfinite inputs are rejected.
The emitted matmuls and partial additions have FP16 tensor types; **FP32 ANE
accumulation is not claimed**. Output scaling uses a wider host intermediate,
then float32 and BF16 round-to-nearest-even. All outputs must be finite before
the suffix is committed. Residuals, protected state and all other projections
retain the existing BF16/FP32 path.

The graph cache contains **no checkpoint data or execution copies**. Its key
includes operation, recipe, emitter revision, dtype/scaling policy, shape/tile,
OS build, compiler and hardware model. A weight/LoRA change therefore rebinds
fresh runtime weights rather than recompiling identical weightless graphs.
Per-block source SHA256 and actual scaling exponents are recorded in the run
log; model/fold identity remains in the existing render/checkpoint provenance.
This is the selected alternative to baking weights/scales into a compiled key.

The default application cache is `outputs/h3-ane-cache`; `H3_ANE_CACHE_DIR`
selects another private, user-owned directory. Content/recipe hashes are checked
before loading; corrupt entries fail closed and recover to GPU. New entries are
staged and atomically published. Cache admission stops at 64 MiB/512 entries;
there is no automatic eviction of user-owned files. Only one graph is loaded per
DiT. Prediction buffers and staging are bounded to 2 GiB. A measured 4 GiB
incremental system-wired/process-footprint guard rejects loading or releases a
growing prediction backend. System-wired measurements include other processes;
this guard is not a hard quota on OS-managed CoreML allocations. CoreML's system
cache is OS-managed and is not removed by h3cli.

Unsupported devices, failed compilation, invalid cache, allocation failures,
range failures and prediction failures retain/recompute the entire original GPU
projection. No partially written ANE output is published. Restoring a checkpoint
that requires ANE rows refuses an unavailable ANE rather than silently changing
the recorded execution plan.

## Scheduling and persistence (T159)

Modes are `--metal-ane off|serial|static|dynamic`. Serial offload uses the same
shards and arithmetic as the static split but waits for ANE before submitting
the GPU projection. Static submits independent GPU prefix and ANE suffix work
concurrently after materializing the ready AdaLN input. Both shards complete
before norm/RoPE/attention begins. There is no illegal overlap between QKV,
attention, output projection and MLP on dependent inputs.

Dynamic mode measures a GPU-only QKV probe every eighth block on its first
encounter, and uses observed per-row rates including packing/prediction/unpacking
to select a bounded suffix. It changes the budget by at most one chunk between
first encounters, and disables subsequent offload if the complete region loses
by more than 3%. Each block's first decision is frozen for subsequent steps;
it is not a continually changing timing-dependent sampler. The 50 decisions and
decision mask are serialized in Metal checkpoint recipe 6. Restored decisions
are validated for range, alignment, actual sequence size and availability.
Completed-step checkpoints must contain all 50 decisions. Fresh processes can make
different dynamic decisions; exact replay requires the recorded plan.

Telemetry includes packing, casts, prediction, unpacking, command draining,
GPU time, waits, join cost, whole-region time, hashes, scales, memory, recovery
and plan decisions. `host_predict_overlap_seconds` intersects the span from the
first prediction's start to the last prediction's end, including gaps between
chunks; it is not integrated API or hardware activity. Hardware overlap requires
the retained Metal/Neural Engine trace.
Root-GPU counters do not cover nested MPSGraph command buffers; prefix wall
time and device traces supply that evidence. Cache-hit telemetry refers to the
application graph cache, not a forced purge of CoreML's OS-managed compiler
cache. Setup timing includes loading, plan verification and the format self-test.

The 30-second static B1 trace contains 96 ANE hardware intervals and 4885 native GPU
compute intervals. Eleven complete QKV worker windows contain 1.258 seconds of
ANE activity; 0.776 seconds overlaps native GPU activity (61.7%). ANE hardware rows
do not carry a PID, so attribution uses the native worker signposts and the public
ANE-only matmul plan. GPU intervals carry the CLI PID. This verifies overlap,
not an end-to-end speedup. Raw trace, exported tables, attribution method and
intersection intervals are retained in `runs/B1-static/`.

## Validation (T160)

The immutable executable/source snapshot and raw records are retained under
`outputs/metal-native-m3c-243/`. The suite compares off/serial/static/dynamic on
identical conservative SOL: FP16 `steel-routed`, adapter layout, minimum exact
fraction .75, one dense initial evaluation, sigma threshold .99, one dense
initial layer, Q/KV blocks 32/64, tau 1 and local radius 1. BF16 source weights,
seed 42, text conditioning, all 50 blocks and Euler sampling are held fixed.

B1 is diagnostic with complete-region fences and captures. Static B1 records
Metal System Trace plus Neural Engine, Core ML and signposts. B5 has no capture
or component fences and measures four steady evaluations after warmup. B6 uses
exactly six evaluations and the original production VAE. The incremental speed
gate is ≥1.10× B5, beyond measured variability, over the same GPU-only attention
recipe. Quality uses the frozen Reference screens; a measured loss does not
promote the backend. Held-out conditioning/continuation and 50-step qualification
are not inferred from this calibration workload.

The final build and Metal host regression target pass. The host sampler suite
passes 1642 checks, including under ASan/UBSan, with 17 container tests and 165
adversarial container cases. CLI rejection tests pass, and ANE split tests pass
serial/static equivalence, frozen-plan
replay, output guards, invalid restore/shape rejection and GPU range recovery.
Small operator relative L2 versus GPU is 0.0010183. A separate full-QKV-width
4096-row synthetic probe has sampled relative L2 0.0019332; its packing,
prediction and unpacking measured 0.1664/0.1167/0.0368 seconds. These are operator
diagnostics, not an end-to-end speed claim. Cache tests cover cold/warm loading,
shape identity, recipe/content tampering, permission rejection, quota admission
and temporary-directory cleanup.

The unfenced B5 comparison is complete:

| QKV mode | Steady median, seconds | Steady range, seconds | Gain over GPU-only | Maximum end-of-step footprint, GiB |
| --- | ---: | ---: | ---: | ---: |
| GPU-only | 113.538 | 113.459–113.593 | 1.0000× | 43.323 |
| Serial ANE | 128.533 | 128.428–128.633 | 0.8833× | 43.911 |
| Static split | 113.508 | 113.445–113.535 | 1.0003× | 43.911 |
| Dynamic split | 113.598 | 113.550–113.612 | 0.9995× | 43.951 |

All three ANE modes fail the **1.10× incremental speed gate**. The static and
GPU-only ranges overlap. Static recovers the serial offload penalty through
concurrency but provides no meaningful gain over GPU-only. Across all five
static evaluations, packing/prediction/unpacking consume 46.853/29.616/9.757
seconds; the complete offloaded QKV regions consume 90.847 seconds, including
waits and joins. Serial consumes 161.915 seconds in those regions. Static and
serial final AV states are byte-identical for B1, B5 and B6. All captured B6
step tensors are also byte-identical between serial and static. Dynamic chooses one
4096-row offloaded block per B5 evaluation and keeps the other blocks on GPU.
All B5 ANE executions complete without recovery. The first full-width graph
application-cache miss takes 3.783 seconds to load and validate. Warm setup
takes 0.161–0.166 seconds, including the format self-test; startup cannot amortize
into a demonstrated whole-step gain.

All 12 B1/B5/B6 runs completed with no ANE recovery. B6 static and dynamic
steady medians are 113.516 and 113.546 seconds, versus 113.526 GPU-only and
128.641 serial. B6 uses independent six-evaluation trajectories and the original
VAE; it does not establish 50-step fidelity.

All three ANE modes fail the frozen **incremental** quality screen against the
identical GPU-only SOL recipe:

| Mode | Mean LPIPS (≤0.06) | Mean SSIM (≥0.90) | Audio envelope relative L2 (≤0.15) | Audio envelope correlation (≥0.98) |
| --- | ---: | ---: | ---: | ---: |
| Serial | 0.2166 | 0.7035 | 0.5195 | 0.6081 |
| Static | 0.2166 | 0.7035 | 0.5195 | 0.6081 |
| Dynamic | 0.1986 | 0.7239 | 0.5098 | 0.6372 |

Spatial percentile, temporal-delta and optical-flow screens also fail. Against
the original MPSGraph Reference+, all modes fail the visual screen; GPU-only
SOL retains its earlier accepted-subset disposition, while the ANE candidates
also fail audio-envelope checks. No prior acceptance is extended to this new
precision change. Captured tensors remain finite but fail the older strict dense
numerical thresholds; those measurements are retained as diagnostics.

The selected QKV result does not establish the performance of a future
MLP/projection split or prepacked-weight pipeline. **Do not promote this backend.**
Keep `--metal-ane off`; no further M3C tuning or broader offload is justified by
this measured prototype. MLP/output offload, held-out conditioning/continuation,
full-model checkpoint replay, 50-step and M5 qualification remain unclaimed.

Final artifacts:

- [Paired playback gallery](../../outputs/metal-native-m3c-243/review/review.html)
- [Verified timing, quality gates and provenance](../../outputs/metal-native-m3c-243/review/report.json)
- [Hardware overlap evidence](../../outputs/metal-native-m3c-243/runs/B1-static/trace-overlap.json)
- [Final build and verification provenance](../../outputs/metal-native-m3c-243/closing/provenance.json)

The renders use the immutable `frozen/` binary/source snapshot. Final changes
after that snapshot only separate probe drain/wait diagnostics, hide raw ANE
JSON unless profiling is enabled, reject malformed/oversized checkpoint plans,
and track vendored header dependencies in the build. A brace-formatting change
in `src/metal/ane.m` produces the identical object hash. Valid render arithmetic and
scheduler decisions are unchanged. The final binary passes profiled and quiet
operator/replay/recovery tests, actual CoreML cache tests and the host/sanitizer
checks above. Both source snapshots and their differences are retained.

## Reproduction

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  python3 scripts/metal_native_freeze.py --output outputs/m3c-fresh/frozen
python3 scripts/metal_ane_validate.py --frozen outputs/m3c-fresh/frozen \
  --output outputs/m3c-fresh/runs --conditioning outputs/m3c-fresh/conditioning.h3cond
python3 tests/test_ane_cache.py --binary outputs/m3c-fresh/frozen/bin/ane_probe
```

After exporting the static B1 trace tables, run `scripts/metal_ane_trace.py`
on its directory. The report uses the pinned packages in
`tests/requirements-metal-quality.txt`, the retained model checksums and passing
reference-only controls. This workspace's existing analysis environment is:

```sh
PYTHONPATH=outputs/metal-native-m2-243/quality-deps \
TORCH_HOME=outputs/metal-native-m2-243/quality-models \
outputs/continuation-validation/venv/bin/python3 scripts/metal_ane_report.py \
  --runs outputs/metal-native-m3c-243/runs \
  --reference outputs/metal-native-m2-243/continued/calibration-reference-B6 \
  --controls outputs/metal-native-m2-243/continued/quality-controls-final/metrics.json \
  --output outputs/metal-native-m3c-243/review
```

Use a fresh output directory when reproducing the report. Its seven paired
quality pages cover each ANE mode against GPU-only SOL and all four modes
against the original MPSGraph Reference+. The original and current conditioning
caches have different arithmetic-environment metadata; the report requires
their actual conditioning tensor payloads to be byte-identical.
