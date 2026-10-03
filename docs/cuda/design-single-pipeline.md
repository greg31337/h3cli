# Single CUDA pipeline with explicit attention and projection precision

M3B/M4 replace the [historical two-mode design](design-two-modes.md).
M3B/M4/M6/M7 are complete; see [initial measurements](single-pipeline-results.md),
[final qualification and playback](single-pipeline-final.md), and [completed checklist](single-pipeline-tasks.md).
The renderer has one SGLang-based preparation, sampler and DiT implementation.
Default dense BF16 retains arithmetic identity 4 and the original full decoder.
`--fast-cuda` and `--resume-default-cuda` are removed and return unknown-option errors.

## Request and dispatch

The immutable request policy records base arithmetic, attention, projection
precision and presentation. Constructors capture it; nested scopes restore it
on success/error/cancellation. CUDA video derives its recipe internally; there
is no public execution selector or old tuning override.

| Option | Scope | Default |
| --- | --- | --- |
| `--cuda-attention default\|sage2++\|sage3\|sol\|subblock` | Main DiT attention only | `default` (dense) |
| `--cuda-denoise-quant off\|fp8\|nvfp4` | Repeated QKV/output/MLP projections | `off` (BF16) |
| `--cuda-denoise-quant-cache DIR` | Persistent packed weights | User cache directory |
| `--preview-vae` | Independent approximate delivery option | SGLang full decoder |

Select attention and projection precision independently. Sage3's FP4 attention
operands do not imply NVFP4 projection weights. The original dense/Sage/SOL
matrix covers twelve configurations; SubBlock adds its separately guarded
compositions. Dense BF16 is the recorded-reference configuration.

Both dense and approximate choices retain reference tokenization, conditioning,
noise, schedules, normalization/RoPE, sensitive FP32 heads, residual conventions,
Euler updates, and delivery. Only explicitly selected main-DiT kernels change.
SOL retains protected conditioning/continuation ranges and early/local dense
regions. `--sol-min-exact` defaults to 0 on both CUDA and Metal; this removes
the exact-block quota while preserving routing and protections. The earlier
M3B/M4/M7 measurements used their recorded previous defaults. Shape fallback uses shared dense primitives, with reason/counters.
Unsupported explicit build/device requests fail before model loading.

M3A's mapped DiT weight staging, copied/bounce fallback, 40 GiB pinned-cache
budget, admission reserve and one Qwen prefetch lane remain shared. Separate
full-VAE casts remain default; fusion was not independently performance-qualified.
No old fast TF32/math/workspace policy can be activated. M6 removes unreachable
legacy helpers, device-weight caching and tuning. Shared still/decoder/operator
primitives remain; see the [consumer audit](single-pipeline-final.md).

## Quantization and compatibility

Quantized weight descriptors dispatch before BF16 GEMM. FP8 uses tensorwide
E4M3 scales; NVFP4 uses VEC16 UE4M3 scales. Both use native SM120 cuBLASLt,
BF16 outputs and FP32 accumulation. QKV/output/MLP weights in the 50 main blocks
are eligible; refiners, encoders, norms, patch/velocity heads and VAEs retain
original precision. Tail padding, finite checks and cancellation recovery are
required. Compressed streaming follows existing memory admission.

Quantization recipe 2 identifies the shared arithmetic-4 integration and its
packing/kernel contract. Persistent metadata-cache version 2 retains canonical
source path, inode/size/timestamps, offset, shape, precision and recipe identity,
atomic publication, malformed-header/length rejection and scale/finite checks.
It never hashes checkpoint weights on the normal path. It does not detect every
possible finite payload bit flip; the separately requested strict verification
option remains diagnostic. Models/cache files must remain immutable during use.

Packed **weights** are reusable across attention options because packing depends
only on weights/precision, not attention; duplicating them by Sage/SOL would waste
space. Execution plans and sampler states retain explicit attention/SOL/precision
and existing model/fold/layout/device/build/library identities. LoRA folding
precedes packing. Default BF16 does not open/create quantized caches.

The conservative adaptive-cache extension uses quantized execution recipe 3
and adaptive recipe 2: block 0 stays BF16, while blocks 1–49 use FP8 or NVFP4.
Packed weights retain recipe 2 and are shared with quantization-only execution.
Both checkpoint policy sections are required and cross-check their recipes.
This adaptive extension requires dense attention; aggressive cache with
quantization remains unsupported. See the
[matched six-video experiment](adaptive-quant-experiment.md).

SubBlock with quantization and adaptive cache off is a separate composition:
all 50 blocks retain quantized projections, while attention uses the unchanged
SubBlock recipe-1 kernel/plan. Quantization execution recipe 4 and SubBlock
execution recipe 2 cross-check both checkpoint policies. The
[SubBlock 0.75 comparison](subblock-quant-experiment.md) records qualification
and the matched FP8/NVFP4 triplets. Triple combinations remain unsupported.

Adaptive and SubBlock [warmup counts](approximate-warmup.md) are independent
request parameters. Defaults remain 4/10; explicit counts 2–16 must leave two
post-warmup evaluations. Required checkpoint section 44 version 2 stores custom
selections and cross-checks the SubBlock plan. Presentation schema 8 stores the
effective counts explicitly. All saved containers use the
[current-only state contract](../features/current-state-contract.md).

Arithmetic identity 4 is retained. Current states record explicit attention and
precision recipes without a fast-mode extension. The no-op probe bit and VAE
shims are removed; the frozen AV fixture uses a test-only loader bound to its
recorded checksum. All measured numerical operations still execute production
code, and the 204 recorded hashes are unchanged.

CUDA still/image-VAE retains its existing geometry/decoder restrictions and
rejects video-only approximation options. Metal behavior and independent preview
VAE are preserved. There is no claim of SGLang still parity.

## Commands

```sh
# Default SGLang-parity video
./bin/h3cli -d /path/to/models/MiniMax-H3 -p "$PROMPT" \
  --width 640 --height 480 --frames 243 --steps 6 -o reference.mp4

# Explicit approximate attention and projections; no mode flag
./bin/h3cli -d /path/to/models/MiniMax-H3 -p "$PROMPT" \
  --width 640 --height 480 --frames 243 --steps 6 \
  --cuda-attention sage2++ --cuda-denoise-quant fp8 \
  --cuda-denoise-quant-cache /path/to/quant-cache -o sage-fp8.mp4

./bin/h3cli -d /path/to/models/MiniMax-H3 -p "$PROMPT" \
  --width 640 --height 480 --frames 243 --steps 6 \
  --cuda-attention sol --cuda-denoise-quant nvfp4 \
  --cuda-denoise-quant-cache /path/to/quant-cache -o sol-nvfp4.mp4
```

Current builds use CUDA 13.0.3, shared cuDNN 9.20 and the
SGLang math/media libraries described in the [reference guide](cuda-sglang-reference.md). Build with
`CUDA_SGLANG=1 CUDA_CUDNN=1`; optional attention requires `CUDA_SAGE=1` and/or
`CUDA_SOL=1` with their pinned dependencies. FP8/NVFP4 currently require SM120.

## Validation and reporting

Run the complete `make test-cuda-reference-regression` after each code
patch and again on final source. Keep all 204 recorded golden outputs and the
bundled input fixtures unchanged. Configure library and model paths in the calling environment.
A missing/skipped/stale/failed case is not a pass. Mutable policy, cache, state,
option-matrix and recovery tests are separate.

Explicit approximate options may degrade visual quality. Require finite stable
execution, actual selected dispatch, valid full video/audio, bounded resources,
and side-by-side HTML against matched dense BF16. Report every-frame quality,
worst frames, temporal/audio differences and historical threshold failures as
measurements; never relabel them as reference parity or imply human acceptance.
Human review is optional and does not block delivering the report.

Visual clips use 640×480 / 124 frames / six evaluations with C0 and ordered R1
references. Match/max and continuation get supplemental coverage. Performance
uses three interleaved matched reference/candidate pairs at 640×480 / 243 frames /
six evaluations on the qualification RTX PRO 5000, binding both policies to NUMA node 0.
Report median wall and denoise speedups separately, conditioning/loading/decoding,
peak NVML VRAM and process RSS, cache preparation/hits and fallback counts.
Cold packing is reported separately; repeated comparisons use warm packed caches.
No speedup is promised for an option that measured slower. Preview VAE is not used
in these matched full-output comparisons.

## Final M7 test scope

The user reduced final optional testing to three representative combinations:
Sage2++/BF16, Sage3/NVFP4 and SOL/FP8. Use one matched 243-frame/six-evaluation
pair per combination and use those same clips for visual comparisons. Report
single observations and untested combinations explicitly. Keep the normal BF16
reference gates, retained C1/C2 every-update replays and conditioning/continuation
checks. The earlier twelve-cell campaign above remains historical evidence.
See [final qualification](single-pipeline-final.md).
