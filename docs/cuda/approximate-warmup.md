# Adaptive-cache and SubBlock warmup controls

`--adaptive-cache-warmup N` controls the initial exact cache evaluations;
`--subblock-warmup N` controls the initial dense attention evaluations.
Defaults remain **4** and **10**. Explicit values are integers **2–16**, with
**N ≤ steps − 2**. For an 18-step schedule, 16 is valid; for a 10-step schedule,
the largest explicit value is 8. Values are rejected rather than clamped.
Each flag requires its corresponding feature, independent of argument order.

Omitting a flag preserves the original fixed default even if a short schedule
finishes during warmup. At the API level, zero means that default; explicit
CLI zero is rejected. The presets' thresholds, maximum hit streaks, final
adaptive refresh, dense block 0 and protected-query/key behavior do not change.
Existing supported FP8/NVFP4 combinations use the same controls; this does not
enable aggressive-cache quantization or triple combinations.
Adaptive warmup evaluates all blocks freshly at the selected precision; it
does not disable quantization. SubBlock warmup changes attention selection only.

The resource ceiling is independent of warmup: `--adaptive-cache-max-mib N`
now defaults to 4096 MiB on fresh requests and restores the saved ceiling on
resume (512 MiB for legacy checkpoints). SubBlock alone also accepts BF16 image, video and audio
references, including mixed sets under the existing Ref2VA input limits. A six-step reference run needs `--subblock-warmup 2` to exercise
sparse steps 2–5; omitting it remains entirely dense. See the
[budget/reference qualification](adaptive-budget-reference-results.md).

Counts refer to absolute evaluation indices of the original full schedule:
with warmup 4, evaluations 0–3 remain exact/dense and evaluation 4 becomes
eligible for approximation. A checkpoint at step 3 resumes with one warmup
evaluation remaining. A cache hit is still subject to score, streak and other
refresh conditions. When both features are enabled, the first SubBlock
evaluation forces a cache refresh at its configured transition.

```sh
./bin/h3cli -d /path/to/MiniMax-H3 -p 'A pianist plays in a sunlit concert hall.' \
  --width 640 --height 480 --frames 90 --steps 50 \
  --adaptive-cache conservative --adaptive-cache-warmup 6 \
  -o outputs/adaptive-warmup6.mp4

./bin/h3cli -d /path/to/MiniMax-H3 -p 'A pianist plays in a sunlit concert hall.' \
  --width 640 --height 480 --frames 90 --steps 50 \
  --cuda-attention subblock --subblock-sparsity 0.75 --subblock-warmup 4 \
  -o outputs/subblock-warmup4.mp4
```

Checkpoint resume restores saved counts. Omit the flags on resume, or supply
matching values; conflicting counts or a flag for a disabled feature fail
before model loading in the CLI. The range check uses the original schedule,
not the number of remaining transitions. Decode-only commands reject these
denoising flags.

Custom selections use required sampler section 44 version 2, preserving device
identity and adding both requested counts. Section 43 stores the effective
SubBlock count and must agree. Older readers reject the required extension;
new readers retain the original 4/10 interpretation of existing version-1
records. Preparation keys include the new controls. Version-6 completed AV
sidecars record effective counts for enabled features and zero for disabled
ones; existing version-1 through version-5 sidecars remain readable. Default
execution keeps its previous checkpoint layout and sidecar version.

The accepted six-video [SubBlock comparison](subblock-quant-experiment.md) and
[adaptive comparison](adaptive-quant-experiment.md) used the original 4/10
defaults. Their videos, numeric metrics and acceptance records remain fixed.
Custom counts are configuration changes, separate from those visual approvals.

## Validation

The [BF16 qualification record](../../outputs/warmup/2026-09-27-sm120/summary.json)
passes. Metal and CUDA builds and ordinary tests passed, together with the
complete **204/204** CUDA golden regression. Optional Metal tests requiring
uninstalled released-model fixtures reported skips and are not counted as
passes. Existing goldens and six-evaluation test limits remain unchanged.

CPU policy tests cover every count 2–16, first eligible evaluations, final
refreshes, phase changes, malformed inputs, short schedules and explicit resume
mismatches. Sampler/presentation tests cover both boundary counts and required
persistence. Fifty checks on real BF16 checkpoints cover required-section
versions, corruption, invalid counts and conflicting CLI overrides.

Native BF16 checks confirm that explicit 4/10 produce byte-identical latents to
omitted defaults. Warmup 2 restores exactly across stop/resume; adaptive cache
also restores exactly after cancellation. Combined adaptive warmup 2 and
SubBlock warmup 3 force the expected refresh at evaluation 3 and resume exactly.
GPU counters show dense attention at indices 0–1 for SubBlock warmup 2, followed
by one dense probe block and 49 sparse blocks per evaluation.

These bounded checks retain the original 50-step schedule and execute at most
six evaluations per invocation. Adaptive fixtures use 256×256/22 frames;
SubBlock and combined fixtures use 640×480/56 frames to exercise sparse dispatch.
They establish execution and persistence correctness, not visual qualification
of custom counts. FP8/NVFP4 were removed from the requested coverage; the
qualification record explains the already completed background probes retained
in the raw log. No additional comparison videos were generated.
