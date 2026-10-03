## Technical Design — Native-Metal H3 Backend for M4→M5

### Objective

Develop the fastest maintainable H3 execution path on M4 whose production output is **visually extremely close** to the original renderer. Preserve original BF16 dense as Reference+, while Reference may use range-safe mixed precision, conservative SOL and selected GPU/ANE cooperation. Full-BF16 native attention is optional on M4, not a production prerequisite. Keep portable primitives for M5 Max and M5 Ultra. Native Metal is an architectural means; removing MPSGraph is not an acceptance criterion for M4 production.

The M4 production candidate may use:

```text
MPSGraph QKV / projections / MLP
              +
native FP16 FlashAttention / conservative SOL with FP32 accumulation
              +
BF16/FP32 state, native layout/fusion and selected GPU/ANE work sharing
```

Choose hybrid or fully native execution from complete-block and whole-step measurements. MPSGraph remains a production option, numerical oracle, compatibility fallback, and A/B backend while it provides value. On actual M5 Ultra hardware, re-evaluate whether a complete native Metal DiT clearly wins before committing to full conversion.

Removal of MPSGraph from non-DiT components such as text encoding, vision encoding, audio encoding, or VideoVAE is explicitly **not required** unless profiling later demonstrates a material performance, memory, or maintainability benefit.

---

# Current evidence and perceptual-quality revision

M0 is complete. M1 remains open: **M1A dense component qualification passes, but integrated dense B5 and SOL B1 fail acceptance**. The retained implementation already uses the MIT-licensed MLX Steel attention template, with BF16 storage, FP32 accumulation, and direct input/output strides. See [vendor provenance](../../third_party/mlx-attention/README.md) and the historical [M0/M1 measurements](m0-m1-results.md).

| Historical 362-frame QKV case, S=33,322 | Native median | MPSGraph median | Native overhead |
| --- | ---: | ---: | ---: |
| Real H3 block-0 QKV | 3.033273 s | 2.954249 s | +2.7% |
| Bounded random QKV, matching sequence-major layouts | 3.143960 s | 2.957733 s | +6.3% |

The representative workload is now 243 frames. The table above retains historical 362-frame evidence. New matched baselines, real QKV fixtures, integration results and the original linear-scope disposition are recorded in [the 243-frame report](m1-243-results.md); they do not retroactively change the older result. That report's former M2 linear scope is now M3A.

The historical real fixture's native input was rearranged head-major outside the timed region; MPSGraph consumed and returned sequence-major data. This result motivated direct-layout integration, but did not establish a complete-pipeline gain. Short, medium, production-size, real-QKV and integrated dense block numerical tests now pass. Dense A fails the 243-frame B5 accumulated quality gate, so six-step production qualification is not opened.

The previous revision replaced the T035 integration blocker with a within-5% real-QKV admission exception for M1's one-block experiment. The real fixture met that threshold; the synthetic production fixture did not. **T035's historical standalone result remains FAILED.** That exception no longer admits new M2 work: M2A requires a meaningful adapter-inclusive win before integration.

The M1 experiments and full-DiT profile are recorded. At 243 frames, dense A passes B1 but fails B5 quality and is 1.9% slower by the steady median; SOL's 1.314× B1 speedup fails quality. Keep these failed gates and both existing Steel candidates as regression baselines. Neither changing precision nor importing another Steel header is evidence that the accumulated error is fixed.

The new priority is **safe FP16 dense attention → perceptual Reference qualification → conservative SOL → M4 GPU/ANE cooperation → QKV/norm/RoPE/layout fusion → Q8 evaluation**. M2C/M2D now establish stability and perceptual quality, rather than mandating a native BF16 port. The previous BF16 implementation/quality/performance tasks T144–T146 remain optional and deferred; they do not block M4 Reference. M3A/M3B full-native work remains conditional, M4 SOL is pulled forward, and M3C adds the bounded ANE branch. These are execution changes within the original model architecture, not a new checkpoint or sampler.

The hybrid requires `--backend metal`; MPSGraph remains the default. M2 now implements explicit FP16 dtype and candidate-tier selection, range adaptation/recovery, and an optional all-block hybrid. The final standalone suite passes at 1.101–1.131×, but the matched 243-frame B5 whole-step result is **1.049×** (118.933 s versus 124.738 s), below the 1.10× gate. **M2 is closed on the user-accepted measured subset.** The independent 243-frame production-VAE B6 also failed its frozen screen (LPIPS mean 0.176; SSIM mean 0.725); the user accepted that quality and selected conservative SOL next, deferring the remaining held-out corpus. Both original failures remain recorded, and no untested-case qualification is claimed. Conservative SOL is active under its own gates; ANE and broad fusion remain deferred. See [the M2 implementation](m2-fp16-implementation.md) and [results](m2-243-results.md).

The M1 failures retain their original strict numerical meaning: the new objective does not retroactively accept their clips or erase their errors. A separate versioned perceptual contract is now frozen before Reference-quality tuning; see [the qualification protocol](m2-reference-qualification.md). M2's operator/range and performance contracts remain distinct. See [the M1 source audit](m1-attention-sources.md) for the retained BF16/SOL implementation.

## External evidence informing the experiment

[VPIPE documents](https://github.com/tgo-app-dev/vpipe#acknowledgements) a native Metal forward path with vendored MLX Steel GEMM/attention headers and no MLX runtime in that path. Its [v0.1.46 release](https://github.com/tgo-app-dev/vpipe/releases/tag/v0.1.46) reports up to 2.6× H3 SOL acceleration. These are reasons to evaluate the combined layout, routing, kernel, and scheduling strategy locally, not a predicted speedup for our workload.

[Upstream h3cli](https://github.com/antirez/h3.c/blob/main/README.md#metal-4-and-tensorops-paths) retains MPSGraph for some larger-sequence operations and describes native microbenchmark wins that do not carry through to complete DiT scheduling. Our acceptance unit must therefore be the complete attention region, block, and denoising step.

VPIPE's [published H3 comparison](https://github.com/tgo-app-dev/vpipe) and M5 NAX discussion motivate portable native work. They do not establish performance on our M4 Max workload or on M5 Ultra; those require matched local measurements.

For mixed precision, [FP16Safe's development measurements](https://github.com/aaalll12322/ComfyUI-MiniMaxH3-FP16Safe/blob/main/DEVELOPMENT_EN.md) report residual/MLP-output values around 5×10⁵ and attention output around 7×10² in their tested configuration. This motivates selective precision and local range checks, not wholesale FP16 conversion or an assumption that our Q/K/V fit FP16. The [M4 FlashAttention project](https://github.com/apstenku123/metal_fast_attention#benchmark) reports 1.3–2.3× on a much smaller eight-head transformer; it supplies another kernel reference, not an H3 speed prediction.

[VPIPE's release notes](https://github.com/tgo-app-dev/vpipe/releases/tag/v0.1.51) describe dynamic GPU/ANE work sharing for pre-M5 hardware. The [h3cli-ane proof of concept](https://github.com/maderix/h3.c-ane#disclaimer) reports 26.3 versus 31.9 seconds/step on a base M4, but uses private APIs and an int8 execution path. Neither result qualifies our BF16-weight mixed-precision path or proves useful overlap on this M4 Max.

[PipeNetwork's paired evaluation](https://github.com/PipeNetwork/minimax-h3-mlx#published-quants) separates teacher-forced velocity error from independently diverging trajectories; use that distinction in our diagnostics. Its 8-bit numbers are not guarantees for final clips. [VPIPE's H3 documentation](https://github.com/tgo-app-dev/vpipe/blob/main/docs/MINIMAX-H3.md) says its accelerated INT8 GEMM option is inactive on M4. On this 128 GiB system, prioritize attention/SOL/ANE/fusion before measuring Q8 bandwidth/residency benefits.

## Shape and source corrections for the new experiment

Our gate stays **640×480, 243 frames**, approximately 10.1 seconds at 24 fps:

```text
20 × 15 spatial patches × 72 video latent positions = 21,600 video rows
2 audio streams × 405 positions                     =    810 audio rows
fixed benchmark prompt                             =     16 text rows
packed sequence                                    = 22,426 rows
H3 attention                                       = 56 heads × 128 dimensions
```

Use the captured layout metadata for each run; references and other prompts change the packed sequence. Three image references in M1 produced 24,250 rows. The old 32,100-video-row / 33,322-packed-row calculation belongs to 362 frames and is historical only. **No 362-frame, 32K-row or 37K-row performance gate is added.** Optional isolated scaling probes cannot replace or delay the 243-frame gate.

The local [`src/denoise/dit.c`](../../src/denoise/dit.c) defines 56 heads, not eight. VPIPE's [published FP16 timing](https://github.com/tgo-app-dev/vpipe/blob/main/docs/MINIMAX-H3.md#cheaper-attention--sageattentions-int8-qk) is an M5 microbenchmark at **8 heads × 20,036 rows × 128**, reporting 156.6 ms. It is a source-reproduction probe, not our complete H3 attention timing. Its ability to load BF16 checkpoints also does not establish BF16 attention arithmetic.

The [upstream 3,072-row discussion](https://github.com/antirez/h3.c/blob/main/README.md#metal-4-and-tensorops-paths) describes native **QKV/output projection dispatch**, not a universal native-attention cutoff. The local M1 native attention already runs at 22,426 and 33,322 rows. Prioritize this work because of measured attention cost and failed end-to-end qualification, not because a sequence-length cutoff prevents native attention.

MLX's [NAX source/compiler discussion](https://github.com/ml-explore/mlx/issues/4533) identifies `steel/attn/nax.h`; audit its pinned implementation and actual operand/accumulator support before adopting it. An issue mentioning BF16 elsewhere is not a BF16 attention correctness or performance result.

## Source reuse and licensing

Extract a minimal dense-attention implementation and dispatch layer; do not port the VPIPE runtime. Prefer the applicable MLX Steel templates under their retained MIT notices, with local `h3cli` shape/layout dispatch. Compare against the already vendored MLX revision `59d600b5e64c238427d0f8d897ab7c682ef4d3d2` and both M1 candidates before adding code; the new candidate must have a documented arithmetic, scheduling, tile or hardware-path difference.

Use VPIPE as the H3 integration/tuning reference. Pin the exact source revisions and record file hashes, local changes and SDK/compiler requirements. If VPIPE-specific source is copied, retain its Apache-2.0 license, attribution, modification notices and applicable [NOTICE](https://github.com/tgo-app-dev/vpipe/blob/main/NOTICE); distinguish those files from MIT components described in its [third-party inventory](https://github.com/tgo-app-dev/vpipe/blob/main/THIRD_PARTY_LICENSES.md). No new third-party source is imported by this plan revision.

---

# Quality tiers and execution policy

These are conceptual quality tiers, not existing CLI flags. A tier records its
actual weights, attention dtype/routing, residual precision, device dispatch,
sampler and decoder; it must never silently select a different recipe.

| Tier | Purpose | Intended computation |
| --- | --- | --- |
| **Reference+** | Immutable oracle, exact original baseline and regression diagnosis | Original BF16 checkpoint, original dense MPSGraph execution, unchanged sampler/conditioning/VAE |
| **Reference** | Final rendering visually extremely close to Reference+ | Original BF16 weights; range-safe FP16 dense or conservative SOL attention; FP32 softmax/output accumulation; BF16/FP32 state and sensitive operations; selected ANE/FP16 linears only after separate qualification |
| **Preview** | Faster iteration with a declared larger quality tradeoff | Separately qualified Turbo/reduced-step settings, more aggressive SOL and later precision/quantization choices |

Reference initially retains the original **50-step production sampling regime**,
noise schedule, all 50 layers, original model architecture, VAE, text/reference
conditioning and seed semantics. Qualification tests stay at **243 frames**:
B1/B2/B5 and exactly-six-step B6, never more than six evaluations per test. No
50-step or longer-duration render is added to the test gate. B6 is limited
trajectory evidence, not proof that all 50-step outputs are visually equivalent.

Native BF16 dense with FP32 state remains an optional research/reference
candidate, especially for M5. It uses the original strict numerical gates when
claiming BF16 parity. It is not the only way to qualify Reference on M4.

Reference+ remains selectable and unchanged. A Reference candidate becomes a
production/default candidate only after its own quality, stability, memory and
whole-step performance gates pass. This documentation revision changes no runtime
default. Preview never supplies the oracle or substitutes its reduced step count
for a Reference speed comparison.

Low-bit weights are not part of the initial Reference recipe on the 128 GiB M4.
Evaluate Q8 only after attention, SOL, ANE and fusion have measured dispositions.
Only Q8 is in scope for weight quantization. Sage remains a later device-specific option.

---

# Strategic scope

```text
freeze the new Reference perceptual contract and range checks
    ↓
M2A / M2B: dense FP16 standalone and full-step proof with explicit adapters
    ↓
M2C / M2D: stable mixed precision and visually-close Reference qualification
    ↓
M4: conservative SOL on the same FP16 kernel infrastructure
    ↓
M3C: selected M4 GPU + ANE work sharing, with measured dependencies/overlap
    ↓
M2E: fuse QKV → norm → RoPE → attention layouts; requalify the composition
    ↓
profile, then M6 Q8 evaluation; deeper native linears only if justified
```

Reuse M0/M1 caches, fixtures, kernels, protected ranges, scratch and profiling.
Keep MPSGraph QKV/projections/MLP during the first attention substitution. Add
one execution change at a time and retain ablations. A losing SOL or ANE stage
may be deferred while the qualified dense Reference path remains usable.

M4 simdgroup attention and M5 NAX/TensorOps attention share an interface but
qualify independently. Native BF16 and full-native DiT are optional branches;
no hardware or precision purity requirement can displace a faster qualified
hybrid. Milestone numbers retain traceability rather than execution order.

---

# Explicit M4 policy

Prioritize reducing attention work and using suitable compute engines over
broad BF16 GEMM tuning. Safe FP16 dense attention can qualify Reference directly;
conservative SOL is the next experiment, followed by selected GPU/ANE cooperation.
Full-native BF16 attention is not a prerequisite for any of these stages.

Keep BF16 weights and BF16/FP32 state. FP16 may be used only at measured-safe
boundaries or with independently validated rescaling/recovery. Later native
FP16 GEMMs require the same range and perceptual checks; do not narrow the whole
DiT, residual stream or output/MLP projections by assumption.

Profile after each stage. Select only linears or ANE work partitions that improve
the complete block/step after transfers, casts, graph compilation and scheduling
are counted. Broad M3A/M3B conversion remains deferred; complete native Metal is
an architectural option, not the M4 objective.

---

# Representative workload

Use:

```text
640×480
243 frames (~10.1 seconds at 24 fps)
50 transformer layers
original BF16 weights
MPSGraph BF16 oracle; attention precision recorded per candidate
fixed prompt
fixed seed
fixed references
```

Routine benchmarks may use no more than five denoising evaluations.

No test introduced by this project may use more than six denoising evaluations.

Final validation must use exactly:

```text
6 denoising steps
```

Primary performance metric:

```text
steady-state seconds / denoising step
```

The 243-frame gate is sufficient for this experiment; do not raise it to 362 frames to seek a larger speedup. The fixed prompt has S=22,426, H=56, D=128. Larger-row standalone probes are optional diagnostics. Do not extrapolate performance milestones primarily from complete long renders.

---

# Phase 0 — Conditioning-state cache

Introduce:

```text
*.h3cond
```

with:

```bash
--save-conditioning PATH
--load-conditioning PATH
```

Persist safely reusable step-independent state including, where applicable:

```text
tokenized prompt state
final text embeddings
attention masks
position information

Qwen vision embeddings

reference-image conditioning
reference-video encoded conditioning
reference-audio conditioning

refined text conditioning

FL2VA / Ref2VA conditioning tensors
```

Do not blindly persist timestep/sigma-dependent tensors.

Where possible separate:

```text
persistent conditioning
+
schedule-dependent state
```

For schedule-specific data, key the cache by the complete denoising schedule.

The `.h3cond` header must record:

```text
format version
h3cli compatibility version

checkpoint/model signature
VAE signature
text-encoder signature
tokenizer signature

conditioning mode

prompt hash
negative-prompt hash if applicable

reference-file hashes
reference ordering

width
height
frame count where relevant

audio configuration

T2VA / Ref2VA / FL2VA settings

tensor shapes
tensor dtypes

conditioning implementation version
```

Cache mismatches must fail explicitly.

---

# Benchmark harness

## B1 — single-step development

```text
640×480
243 frames
1 denoising step
50 layers
BF16
cached conditioning
--preview-vae
```

Purpose:

```text
kernel iteration
Metal traces
numerical comparisons
memory analysis
layer scaling
backend comparison
```

## B2 — warm-up

```text
640×480
243 frames
2 denoising steps
50 layers
BF16
cached conditioning
--preview-vae
```

Purpose:

```text
step-1 vs step-2 comparison
pipeline compilation
graph compilation
allocator warm-up
Metal pipeline warm-up
```

## B5 — routine performance

```text
640×480
243 frames
5 denoising steps
50 layers
BF16
cached conditioning
--preview-vae
```

Report:

```text
step 1

step 2
step 3
step 4
step 5

median steps 2–5
mean steps 2–5
minimum steps 2–5

total denoise wall time
```

Primary speed metric:

```text
reference steady-state median
--------------------------------
candidate steady-state median
```

## B6 — milestone/final validation

```text
640×480
243 frames
6 denoising steps
50 layers
cached conditioning
```

Use only for:

```text
major milestone acceptance
six-step error accumulation
hybrid/native DiT production qualification
SOL validation
quantization validation
continuation validation
Ref2VA validation
CUDA comparison
final integration
```

Report:

```text
step 1 separately

steps 2–6:
    median
    mean
    minimum
    maximum

total denoise time

peak memory

backend configuration

attention configuration

weight precision

latent error metrics
```

---

# Preview-VAE policy

Use:

```bash
--preview-vae
```

by default for:

```text
native attention
Metal linear kernels
Metal block development
SOL
tensor-layout changes
RoPE / normalization changes
memory-layout changes
command scheduling
sampler work
quantization kernel development
```

Use production VideoVAE only for:

```text
VAE-specific work

final visual parity checks

SOL visual validation

Q8 visual validation

Ref2VA identity validation

continuation-boundary validation

transformer → VAE memory cleanup

late-memory-spike investigation

final B6 milestone validation
```

Keep existing production VideoVAE tiling unchanged during the initial DiT project.

---

# Backend architecture

Introduce an explicit execution backend rather than growing independent environment-variable paths.

Conceptually:

```c
typedef enum {
    H3_BACKEND_MPSGRAPH_REFERENCE,
    H3_BACKEND_METAL
} h3_backend;
```

User-facing selection:

```bash
--backend mpsgraph
--backend metal
```

Currently `mpsgraph` is production plus reference. `metal` requires the explicit `--backend metal` opt-in and reports its hybrid composition and unqualified status. Existing invocations keep their defaults; MPSGraph linears are identified in every run.

Report backend selection, candidate/qualified status, execution composition (`mpsgraph`, `hybrid`, or fully `native`), dense kernel candidate, attention policy, and weight precision in each manifest. Unsupported combinations must fail with an actionable error rather than silently switch implementations.

M1 experiments and new M2 candidates remain opt-in. Production promotion is separate: Reference needs M2C/D stability and perceptual quality plus measured B5/B6 evidence; conservative SOL adds M4 gates and ANE adds M3C gates. Optional strict native BF16 uses T144–T146. Preview has a separate looser contract. A qualified hybrid can be an M4 production backend without completing M3A/M3B. Any later default change must follow matched end-to-end acceptance, and MPSGraph remains explicitly selectable.

---

# Orthogonal Metal policies

Within hybrid and native execution, keep algorithm choices independent.

Conceptually:

```c
typedef enum {
    H3_ATTN_DENSE,
    H3_ATTN_SOL
} h3_attention_mode;

typedef enum {
    H3_WEIGHT_BF16,
    H3_WEIGHT_Q8
} h3_weight_format;
```

Planned attention precision must also be explicit (`BF16|FP16`), independently of weight storage. A proposed `--attention-precision bf16|fp16` control is not an existing CLI flag. Persist the numerical recipe in checkpoints and prepared-state keys. Device capability selects portable simdgroup or NAX kernels, not an implicit dtype change.

Future M5 capability selection may independently choose:

```text
BF16 vs INT8 matrix arithmetic

dense vs SOL routing

BF16 QK vs Sage QK
```

Avoid combining these into monolithic backends such as:

```text
SOL_Q8_M5_FAST
```

The policies should compose.

---

# Native Metal DiT architecture

Current path:

```text
hidden
  ↓
MPSGraph
  ↓
MPSGraph
  ↓
custom kernels
  ↓
MPSGraph
  ↓
custom kernels
  ↓
MPSGraph
```

Retained M1B/M1C path:

```text
MPSGraph QKV
  ↓
direct attention-native layout
  ↓
native Q/K normalization + RoPE where applicable
  ↓
native dense attention (M1B) or SOL (M1C)
  ↓
direct projection-compatible output
  ↓
MPSGraph output projection / rest of block
```

M1 already integrated this boundary. M2 inserts an explicit FP16 conversion/repacking adapter around a proven kernel without changing QKV generation and qualifies its range/perceptual behavior. Native BF16 adaptation is optional. M2E later removes avoidable adapters and fuses preparation using retained direct-layout support, after SOL/ANE dispositions. Include all remaining copies, graph boundaries, temporary buffers and waits in measured costs.

Conditional M3A/M3B path:

```text
hidden
  ↓
Metal AdaLN
  ↓
Metal QKV
  ↓
Metal Q/K norm + RoPE
  ↓
Metal attention
  ↓
Metal output projection
  ↓
Metal residual
  ↓
Metal MLP preparation
  ↓
Metal FC1
  ↓
Metal SwiGLU
  ↓
Metal FC2
  ↓
Metal gate / residual
```

If the post-M2 profile justifies complete native conversion, the milestones are:

```text
one complete DiT block
executes without MPSGraph
```

followed by:

```text
all 50 blocks
execute without MPSGraph
```

followed by:

```text
complete denoising evaluation
executes without MPSGraph
```

Those are proofs of full-native coverage, not mandatory M4 production gates. Keep the measured hybrid if it is faster or simpler to maintain at equivalent quality.

---

# Existing native BF16 attention — M1 evidence

Implement exact dense attention that:

```text
tiles Q

streams K/V

does not materialize N×N scores

uses online softmax

keeps BF16 Q/K/V storage

uses FP32 running max / normalization state where required

accumulates output with adequate precision
```

Conceptually:

```text
for each Q tile:

    initialize online-softmax state

    for each K/V tile:

        compute Q × K

        update running max

        update normalization

        accumulate weighted V

    write final output
```

Retain this implementation as:

```text
an optional strict-BF16 native candidate and diagnostic reference
+
the primitive used by SOL
```

## M1A — Numerical dense qualification: PASS

The current Steel-based implementation passes short, medium, production-size, and real H3 QKV component tests and supports direct sequence-major/head-major input/output layouts. Keep its source, pinned provenance, fixtures, and fixed numerical gates. This is component qualification, not integrated dense or SOL qualification.

## M1B — Integrated dense attention subpipeline: implemented; B5 failed

The following records the M1 policy, not the new M2 admission rule. Preserve its failed outcomes; M2A now requires a meaningful adapter-inclusive standalone win. M1 used three independent decisions:

1. **Integration admission:** numerical qualification plus a representative real-QKV native/MPSGraph median ratio ≤1.05 permits T036–T039 and integration of T040's completed output-layout primitive. The present real fixture qualifies for this narrow experiment; keep the synthetic +6.3% result visible and expand representative real fixtures before broader claims. T032 tuning is not a prerequisite.
2. **Integrated result:** compare from the same hidden input through QKV, normalization/RoPE, layout, attention, and output projection, and also compare the complete block. The preferred result is native/hybrid region time ≤MPSGraph region time without a material full-block regression. A slower region may remain opt-in if traces demonstrate meaningful reductions in traffic, temporary allocation, or synchronization with a concrete next hypothesis. That evidence alone is not production acceptance.
3. **Production dense qualification (revised T035 policy):** the native kernel must either win standalone or remain within the approximately 5% representative real-QKV tolerance, and the complete attention region/block plus B5 must demonstrate reproducible end-to-end savings at accepted memory use. Require exact-BF16 B6 accumulation/quality validation before promotion. A standalone win alone never suffices. The current standalone T035 result stays FAILED; no production gain has been measured.

SOL performance cannot qualify the dense backend: record dense integration gains and the incremental SOL gain separately. If dense integration remains meaningfully slower, retain the kernel as a SOL foundation/reference and leave MPSGraph as the exact production path.

Instrument both the attention subpipeline and complete block, including CPU encode/wait, graph and command-buffer boundaries, dispatches, copies/bytes moved, live scratch, and GPU time where reliable. Use diagnostic fences only for attribution; collect acceptance timings without those fences. Use repeated, balanced A/B ordering, matched warm-up and thermal conditions, and retain raw timings and run manifests. Keep the validated `H3_DIT_COMMAND_BLOCKS=5` policy matched on both sides; report any scheduling experiment separately from kernel substitution.

## M1 candidate comparison retained as evidence

Candidate A uses the pinned MLX Steel template at commit `59d600b5e64c238427d0f8d897ab7c682ef4d3d2`, BQ=64/BK=32. Candidate B (`steel-routed`) is already implemented: a local adaptation holding Q fragments in registers, with a 32×16 dense dispatch and partial-softmax support. It has no MLX runtime dependency. Preserve both candidates and their source audit; vendoring the same headers again is not a new implementation.

T128 compared A, B and MPSGraph on identical saved QKV at blocks 0/24/49, reference-heavy and continuation layouts, and the synthetic 243-frame fixture. Balanced standalone wins did not establish model quality or B5 savings. Retain the raw timing, conversion, numerical and provenance records for M2 comparisons.

The former immediate SOL priority is superseded by the dense-first M2 experiment below. M1's failures remain visible and do not block starting M2A.

---

# M2 — Safe FP16 attention and perceptual Reference qualification

**Measured disposition:** the scaled FP16/FP32 simdgroup candidate is implemented
and remains opt-in after failing M2B's whole-step speed gate. The first
block preserves QKV byte-for-byte and has output relative L2 0.000850. B5 executes
all 250 blocks with no invalid attention heads; 82 of 14,000 head evaluations use
declared BF16/FP32 recovery. Tracked Metal peak increases from 38.46 to 39.96 GiB.
These checks retain a useful kernel/adapter foundation, but neither the isolated
win nor finite execution establishes perceptual Reference quality. The initial performance failure did not open T152/B6. The user acceptance below
now permits that qualification; broad fusion still awaits SOL/ANE dispositions.
The requirements below remain the acceptance contract for subsequent work.

**Accepted continuation:** the user accepted the measured 1.049× result. The
original failed 1.10× result remains historical evidence, while an explicit
exception admits this candidate to the remaining M2C/D and six-evaluation B6
work. It does not waive range/operator checks or claim perceptual qualification.
The accepted binary and evidence hashes are retained in
`tests/metal_fp16_performance_acceptance.json`.

The [implemented qualification protocol](m2-reference-qualification.md) adds
identical-input six-noise-level diagnosis, GPU range/independent sampled-score
probes, and a frozen reference-only calibrated perceptual screen. The serial
driver retains production-VAE paired HTML and stops on a failed frozen screen;
it does not require human approval or infer 50-step quality. The completed B6
failed that screen; the remaining held-out corpus and conditioned-case range
qualification were not run. The user then accepted the current quality and chose
“Start conservative SOL next.” M2 closes on that explicitly accepted measured
subset, with the remaining corpus deferred. The original automated failures
remain visible; no unrun-case or 50-step claim follows. The quality evidence and
scope are pinned in `tests/metal_fp16_quality_acceptance.json`. Conservative SOL
starts with the same dense arithmetic and must establish its own incremental
quality/performance result.

Original BF16 weights and state feed existing QKV generation. Only the attention
boundary changes first:

```text
BF16 QKV / existing norm and RoPE
    ↓
range-checked conversion or validated power-of-two rescaling
    ↓
FP16 native dense FlashAttention
FP32 score accumulation, row maxima, exponentials, sums and output accumulation
    ↓
BF16 output (or proven-safe FP16 output converted before projection)
    ↓
existing BF16/FP32 projection, residual and sampler state
```

Keep attention dtype, dense/SOL routing, weight format, numerical tier and device
capability independent. Record them, the actual operand/math policy, range
recovery, kernel revision and layout in manifests/checkpoint recipes/prepared
keys. Incompatible resumes fail explicitly. Original BF16 native candidates
remain available for diagnosis; an FP16 path is labeled mixed precision even
when it qualifies visually as Reference.

## M2A — Isolated FP16 production-shape proof

Audit/pin the minimal VPIPE/MLX dense strategy and reuse M1 infrastructure. Test
**S=22,426, H=56, D=128**, captured 243-frame blocks 0/24/49, reference-heavy and
continuation layouts. Short, aligned and tail fixtures exercise kernel edges;
8-head or larger-row source probes are optional, not acceptance gates.

Freeze component tolerances before tuning. Compare the kernel with an independent
oracle on the same converted operands, then separately measure conversion error
against Reference+. Exercise outliers, nonfinite inputs, underflow, saturation,
strides and guards. FP16 implementation correctness is still a hard requirement;
visual tolerance does not excuse a broken softmax or an out-of-bounds kernel.

Retain kernel-only and complete **BF16 input → projection-compatible BF16 output**
times, including conversions, rescaling, range handling and repacking. Warm and
balance all candidates, record thermal/memory state and retain raw samples/hashes.
Gate integration on **≥1.10× adapter-inclusive median speedup** on the production
fixture set above variability, with per-fixture results and no material
conditioning regression. A preconverted kernel timer cannot substitute.

## M2B — Integrated dense FP16 performance proof

Insert the adapter at the existing hybrid boundary without altering QKV,
normalization/RoPE, projection, MLP or sampler arithmetic. Validate a complete
block before substituting attention across 50 blocks. Measure the preparation
through projection region and complete block, then unfenced matched B1/B5.
Use cached conditioning and `H3_DIT_COMMAND_BLOCKS=5`; report cold and steady
costs separately. No SOL, ANE, quantization or sampler changes enter this ablation.

Require correct implementation, finite outputs, accepted memory, no material
block regression and **≥1.10× B5 steady-step speedup** above variability. Numeric
drift from BF16 is reported, not automatically judged by the old strict model
limit. This performance proof alone does not qualify Reference.

## M2C — Range-safe mixed precision

Profile Q, K, V separately before/after norm/RoPE, score tiles, softmax state,
attention output, projection/MLP output and residuals over representative layers,
noise levels, prompts and reference/continuation cases. A measured small attention
output says nothing sufficient about its Q/K/V inputs or score range. Retain
BF16/FP32 for residuals and sensitive operations; record extrema and nonfinite,
overflow and underflow counts without adding unmeasured CPU stalls per layer.

Where needed, test algebraically valid power-of-two scaling with the corresponding
QK scale and V/output inverse compensation; do not copy another implementation's
fixed scale without proving our norm/rounding semantics. Never silently clamp.
Provide a reported BF16/FP32 recomputation path from preserved inputs before a
bad result reaches residual/sampler state. Record recovery frequency and include
its time/memory in acceptance. A requested unsupported mode must still fail
clearly; runtime recovery is an explicit numerical recipe, not hidden dispatch.

Keep online-softmax statistics and accumulation FP32 for dense and SOL. Diagnose
M1 drift enough to distinguish kernel/layout defects from expected precision or
trajectory divergence. The old strict BF16 port and its T144–T146 gates are
optional; their failure cannot by itself reject a perceptually qualified Reference.

## M2D — Visually-close Reference acceptance

Freeze the versioned perceptual contract below before tuning; preserve the old
strict thresholds/results separately. Run matched teacher-forced operation/velocity
diagnostics and independent B1/B5/B6 trajectories at **243 frames**. The decisive
quality evidence includes decoded production-VAE video, motion, identity,
conditioning/continuation and audio/A/V checks, not one velocity-L2 threshold.

Safe FP16 dense can qualify **Reference**, not only Preview. Require the frozen
perceptual/stability/protection checks, accepted memory and the M2B whole-step
speedup. Retain paired playback HTML and metrics. Reference+ remains the oracle;
no previous M1 failure is reclassified without fresh evaluation under the new
contract. A native BF16 implementation is not required to complete this gate.

## M2E — Accepted-path layout fusion

The user accepted the measured dense and conservative-SOL results, then explicitly
selected M2E. Bring this work ahead of ANE; ANE and the held-out conditioning/
continuation render corpus remain deferred. Acceptance does not turn earlier
failed screens into passes or qualify unrun cases.

Keep MPSGraph BF16 QKV/projection and the accepted mixed recipe 4. The existing
cooperative norm/RoPE kernel already writes `[head,row,128]`; add optional range
partials computed from the **rounded BF16 outputs** in that same pass. Reduce
seven words per head/row into the unchanged per-head range policy. Preserve
normalization order, RoPE rounding, power-of-two scaling, bounded/countable
underflow and BF16/FP32 recovery.

For dense attention, pack the owned Q/K/V buffers in place after their last BF16
consumer. Recovery heads keep their original BF16 bits in these buffers; admitted
heads contain scaled half operands. Bind preparation to context, buffers, shape
and scale, consume it once, and invalidate it on cancellation. Diagnostic capture
may submit before consumption. Never reuse stale range records or expose packed
buffers as model state. For routed SOL, retain the original BF16 Q/K/V for centroid
summaries and use separate half workspaces. Both modes retain FP32 accumulation
and the checked BF16 commit directly in projection-compatible output layout.

Expose `--metal-attention-layout adapter|fused`, default `adapter`, alongside the
existing FP16/Steel selection. Checkpoint attention version 5 and
layout recipe 1 bind this choice to resume/prepared identities. Require compatible
cached cooperative BF16 QKV preparation; unsupported paths fail explicitly.
There is no residual, sampler, checkpoint-weight or default-backend conversion.

Require byte-identical Q/K/V, range records and attention outputs against the
retained adapter for tails, both output layouts, recovery, invalid inputs and
protected SOL. Run matched **640×480, 243-frame, all-50-block B1/B5/B6**; B6 has
exactly six evaluations and production VAE. Use the same frozen executable and
shader, fresh matching conditioning, and serial GPU runs. Preserve per-step
sampler hashes and block-boundary captures to establish whether fusion changes
accepted arithmetic. If byte-identical, the existing user quality acceptance
carries forward for those measured cases; held-out cases remain unqualified.

Measure unfenced B5, plus dense/SOL fusion ablations. Report measured peak memory,
dispatches/submissions/waits and logical traffic accounting separately: replacing
one three-buffer range scan costs a write and read of the small partial records;
in-place packing removes extra dense storage, not its read/write conversion.
No output transpose is claimed removed where the accepted path already writes
the requested layout. Keep a losing result optional, record variability, and
re-profile before Q8 or native linears. M4 does not qualify M5.

M2E measured disposition: implementation and selected 243-frame validation are
complete. Dense B5 retains identical output and saves 0.866 GiB, but takes
125.106 s versus 124.854 s adapter. SOL fusion takes 120.519 s versus 113.596 s
adapter and adds 0.033 GiB. Both retain identical arithmetic/routing outputs;
dense B6 also matches the accepted production-VAE MP4 exactly. Keep fusion
optional for dense memory savings, preserve adapter defaults, and retain all
unrun conditioning/ANE/M5 deferrals. See [M2E results](m2e-layout-results.md).

## Optional strict BF16 / M5 branch

T144–T146 may implement BF16 Q/K/V/output with FP32 state and retain the original
strict component/model/B6 gates when claiming BF16 close-reference parity. This
is useful research or a device-specific option, not a prerequisite for M4
Reference. M5 NAX work is capability/SDK gated and must report actual precision;
M4 results or a successful compile cannot establish M5 quality/performance.

---

# M3C — M4 GPU/ANE cooperation

Start after safe FP16 dense and the conservative-SOL experiment have measured
dispositions; do not wait for native BF16 attention or a full-native block.
Audit the pinned VPIPE dynamic-split implementation, actual API dependencies,
licenses, supported shapes/dtypes, compile costs and cache behavior. Prefer a
maintainable supported integration; isolate any private-API research dependency
and keep a GPU-only path. Do not import an all-ANE/int8 architecture as part of
this original-BF16-weight experiment.

Prototype selected QKV/MLP/projection partitions, retaining BF16 source weights
and the qualified range policy. Compilation may use measured-safe execution
copies; record their dtype and size, and do not silently quantize. Cache compiled
graphs by weights/LoRA identity, shape, operation, precision/scales, OS/toolchain
and device. Bound wired memory, graph residency and cache growth, and exercise
invalidation, cleanup and GPU recovery on unsupported/failed execution.

A block is dependent: **QKV → attention → output projection/residual → MLP**.
Those operations cannot simply run concurrently on the same inputs. Begin with
independent row/output-channel shards within eligible linears, GPU preprocessing
or transfers, and overlap only work proven ready by the dependency graph. A
pipelined tile can start only after all its input/head/reduction dependencies are
complete. Preserve concatenation/reduction order and protected state. Keep a
static split as the oracle before implementing a bounded adaptive scheduler;
record its choices, determinism and resume behavior.

Use traces to show real GPU/ANE work and overlap rather than inferring it from
API selection. Compare GPU-only, serial offload, static split and dynamic split
on identical 243-frame B1/B5/B6 work. Count packing, copies, casts, waits, shared
bandwidth contention, scheduling, cold compilation and warm cache hits. Require
Reference perceptual/stability acceptance and a **≥1.10× incremental B5 gain**
above variability over the same attention/precision recipe without ANE. Report
startup amortization separately. Retain GPU-only execution if the split loses;
ANE availability is not a reason to force its use.

M3C implementation: the selected prototype offloads independent **QKV row
shards** through one CoreML graph and one persistent worker. MLP/output-projection
offload remains outside this bounded experiment. `--metal-ane` selects `off`
(default), `serial`, `static`, or `dynamic`; `--metal-ane-rows` bounds the suffix
(default 4096), and `--metal-ane-chunk` selects 256/512/1024 compiled rows
(default 512). Native FP16 attention is required; all modes retain original
BF16 source weights and BF16/FP32 residual state.

Measured disposition: M3C's 12-run 243-frame B1/B5/B6 comparison is complete.
Static B5 is 113.508 s versus GPU-only 113.538 s, with overlapping ranges;
serial and dynamic also fail the 1.10× incremental gate. All ANE modes fail the
frozen incremental B6 quality screen. Keep ANE disabled and defer broader
offload/tuning. Actual device overlap and serial/static equality through B6
qualify the scheduling experiment, not production quality or acceleration.
See [M3C results and playback](m3c-ane-results.md).

The adapted VPIPE emitter is pinned and isolated in `third_party/vpipe-ane/`.
Prediction uses public CoreML APIs, but graph creation depends on undocumented
compiled MIL/container formatting. Require all matmuls assigned to ANE plus a
numerical startup check, and retain GPU recovery. ANE matmuls and partial sums
use FP16 tensor types; the implementation does not claim FP32 accumulation.
Power-of-two operand scaling, conversion-error checks and finite BF16 commit
checks bound this separate mixed-precision path.

Compiled graphs are weightless: key them by emitter/recipe, operation, shape,
precision/scaling policy, OS/toolchain and device. Rebind current BF16 weights
on every call; log weight/LoRA content hashes and actual scales rather than
caching an execution copy under an incomplete identity. Keep one graph resident,
bound prediction buffers/staging to 2 GiB, cap application-cache admission, and
monitor incremental wired/physical-footprint growth. The OS owns CoreML driver
allocations and its system cache; a measured guard is not a hard system quota.

Static/serial modes share arithmetic. Dynamic mode adapts within the row budget
on first encounters, then freezes each block's decision and saves it in the
Metal checkpoint recipe. A replay restores the plan rather than retiming it.
The four-way 243-frame B1/B5/B6 comparison and actual hardware overlap evidence
are recorded in [M3C results](m3c-ane-results.md). Promotion still requires the
original incremental speed and Reference quality gates above.

---

# Native BF16 linear layer

Broad native BF16 linears under M3A remain deferred. After attention/SOL/ANE/fusion, select only measured bottlenecks. Range-safe native FP16 GEMMs with BF16 source weights are a separate bounded option (T162); keep residuals and sensitive outputs BF16/FP32. MPSGraph linears remain valid, and neither M4 ANE nor M5 attention needs a full-native block.

Required operations include:

```text
QKV projection
attention output projection
FC1
FC2
```

Initial goal:

```text
reduce the measured projection / MLP bottleneck
with a portable interface and a complete-block benefit
```

not:

```text
maximum possible M4 GEMM benchmark score
```

Use appropriate native Metal matrix kernels/primitives or portable source-level kernels with orchestration under `h3cli` control. Full-native conversion is a conditional branch, not a reason to replace a faster MPSGraph operation.

Benchmark complete blocks, not merely isolated GEMMs.

A faster microkernel that causes slower complete-block execution must not be accepted.

---

# Tensor-layout architecture

Use one native layout strategy across:

```text
QKV
normalization
RoPE
attention
projection
```

Avoid:

```text
write layout A
transpose
repack
layout B
attention
transpose again
```

Prefer:

```text
QKV
 ↓
write directly in attention-native layout
 ↓
norm / RoPE in-place or fused
 ↓
attention
 ↓
write directly in projection-compatible layout
```

Layout is a portable architecture decision and should be reusable by:

```text
M4 Metal
M5 Metal
CUDA
```

even if physical tile arrangements differ.

---

# SOL architecture

M1C built SOL but failed its strict numerical model gates. Preserve the implementation, errors and protected ranges. Now pull conservative SOL forward immediately after safe dense FP16 Reference qualification, without waiting for native BF16 attention or full-native linears. Apply the new perceptual contract to fresh results; compare with the same dense FP16 composition and also Reference+ so precision, routing and integration gains remain distinct.

For each query block:

```text
calculate cheap K-block importance estimates
              ↓
classify K/V block
              │
              ├── protected → exact
              ├── important → exact
              └── low importance → approximate
              ↓
single online-softmax result
```

Initial configurable parameters:

```text
Q block size
K/V block size

tau

local exact radius

initial dense-layer count

dense early-evaluation / noise-level policy

minimum exact-block fraction

protected ranges
```

Use VPIPE-like parameters only as initial defaults:

```text
K/V block       64
tau             1.0
dense layers    1
local radius    1
```

These are historical seeds, not a qualified conservative preset. Start the new sweep near dense coverage and vary tau, dense early evaluations/layers, radius and minimum exact fraction independently. Report actual exact/approximate/protected fractions. Define absolute evaluation index and noise-level semantics and persist them in checkpoints so resumed runs do not restart their dense prefix. A B1 or B5 run with only dense dispatches cannot qualify SOL: exercise routed evaluations within the six-step cap and record their counts.

## Mixed SOL implementation and evaluation policy

M4 SOL recipe 1 reuses the accepted dense FP16/FP32 kernel for exact blocks.
Summaries retain BF16 storage; routing, centroid attention and the stable merge
use FP32 arithmetic. The merged FP32 result passes the existing all-head range
and finiteness check before BF16 output commits. Recovery heads force every K/V
block exact and use original BF16 operands with FP32 accumulation. Protected
query or key blocks always override approximation, including overlapping tails.
The dense baseline's arithmetic remains mixed recipe 4.

`--sol-dense-steps N` keeps absolute zero-based schedule indices `[0,N)` dense.
It uses the saved sampler index, not a counter restarted by resume.
`--sol-dense-sigma F` keeps an evaluation dense when the maximum of its video
and audio sigmas is **greater than or equal to** F; `-1` disables this check.
Both protections apply in addition to initial dense layers and `min-exact=1`.
Legacy defaults are `0` and `-1`; conservative experiments explicitly request
one dense evaluation and a high-noise threshold. No runtime default is promoted.

Attention checkpoint version 4 persists both fields, binds SOL recipe 1 and
rejects older native recipes. Prepared keys include the complete policy.
Per-block `h3_sol_policy` records distinguish early evaluation, high noise,
early layer, all-exact and routed dispatches; `h3_sol` counters report effective
routes after recovery protection. Six-evaluation tests must include actual
approximate pairs to count as SOL evidence. No per-layer CPU readback is added.

The first mixed-SOL experiment is recorded in
[m4-sol-243-results.md](m4-sol-243-results.md). Operator, policy and range checks
pass, and matched B2 confirms byte-identical dense protection plus actual
routing. The 75% minimum-exact candidate executes 83.40% of pairs exactly and
improves its routed evaluation by 1.08655× diagnostically. No tested preset
passes the complete preliminary component screen, so production B5/B6 promotion
remains unopened. Dense stays the accepted baseline; M4 qualification is open.

## M1C evidence and the new conservative SOL gate

Implement K/V summaries, importance routing, and a single numerically stable online-softmax merge of exact and approximate contributions. First validate the all-exact limit against native dense and test mixed routes, tails, and extreme logits. Then implement protected conditioning/local ranges before accepting model-level SOL measurements. The protection mask must override routing for every overlapping block; a block straddling a protected boundary is conservatively exact. Test protected query rows against all K/V and protected K/V ranges for every query, so conditioning protection cannot become a one-sided routing convention.

M1 froze numerical SOL limits and failed them; retain that conclusion. The new
Reference contract allows selective approximation and prioritizes decoded fidelity,
but requires independent operator correctness and all protections. Full dense
coverage uses the chosen qualified arithmetic, not an assertion of BF16 bit
identity. Stored continuation prefix state must still remain byte-identical.

Run B1 diagnostics then B5 with actual routed work, including summaries/routing,
FP32 stable merge, range handling and synchronization. Require the same frozen
Reference perceptual/stability checks and **≥1.10× incremental B5 speedup** above
variability versus dense FP16, with total gain versus Reference+ reported. Use
production-VAE B6, Ref2VA identity, continuation boundaries and audio/A-V metrics
for qualification. If protection removes sparsity or quality fails, retain dense;
moderate/aggressive settings belong to Preview under separate limits.

---

# Protected Ref2VA / continuation ranges

SOL must support explicit query/key ranges with full dense coverage in the chosen validated dtype. Dense coverage prevents routing approximation; it does not imply BF16 arithmetic identity. Stored continuation-prefix state is separately required to remain byte-identical.

Protect:

```text
text-conditioning rows

audio-conditioning rows

reference-image rows

reference-video rows

reference-audio rows

Ref2VA conditioning rows

continuation-prefix rows

first/last-frame anchor rows where applicable

local temporal neighborhood
```

For latent continuation:

```text
entire preserved AV continuation prefix
→ exact attention
```

This policy must work with all legal continuation contexts, including:

```text
39
90
141
192
```

frames.

No SOL kernel should contain hard-coded assumptions about one context size. These protections were implemented in M1C and remain required in any later SOL retry. M4 adds full six-step quality coverage and tuning.

---

# Quality and numerical validation

## Two versioned contracts

**Reference+ / optional strict BF16:** retain the original frozen numerical
limits, including dense model relative L2 ≤0.02 and cosine ≥0.999 when claiming
that parity. Keep M1 numerical failures and artifacts unchanged.

**Reference / visually extremely close:** create a separate versioned contract
before new tuning, with fixed prompts/seeds/reference assets, metric versions,
thresholds, range/fallback policy and timing rules. Calibrate thresholds using
repeatability and predeclared acceptable/degraded controls, then freeze them
before evaluating held-out candidates. Do not derive a passing threshold from
the candidate being qualified or replace the old limits file in place. Preview
uses a separately labeled looser contract and cannot qualify Reference.

The new contract allows finite, measured numerical divergence from BF16. Hard
requirements remain: correct implementation of the declared operator, finite
values, no silent clipping/corruption, all requested evaluations/layers, preserved
conditioning and exact saved continuation prefixes, and valid state/cache/resume
semantics. Large unexplained drift must be investigated, but exceeding an old
strict velocity/latent limit alone is no longer a Reference rejection criterion.

## Paired diagnostics and free trajectories

Record operation/block outputs, video/audio velocities and latents separately.
Teacher-forced comparisons use identical saved inputs and timestep/sigma to
isolate arithmetic/routing error; they do not predict final perceptual quality.
Also compare independently evolved B1/B5/B6 trajectories and the actual decoded
B6 clips. Every test remains **243 frames** and executes at most six evaluations;
exactly six for B6. Diagnostic state/step captures and range probes are excluded
from unfenced acceptance timers. No 50-step test is introduced or implied.

Freeze quantitative checks for spatial perceptual similarity (for example
version-pinned LPIPS plus SSIM/PSNR diagnostics), temporal consistency/flicker
and motion, face/reference identity where applicable, audio spectral/envelope
similarity and clipping/dropouts, and A/V timing/lip motion. Specify metric
preprocessing and per-case failure bounds before tuning; a global average must
not hide a failed face, continuation boundary or audio case. These are screening
tools, not a mathematical proof of visual indistinguishability.

Cover faces/hands/fine texture, camera and subject motion, water/fire/smoke,
multiple subjects, speech/singing, T2VA, each Ref2VA modality, anchors and all
39/90/141/192-frame continuation contexts with a 243-frame target. Check protected
prefix bytes separately from newly generated suffix quality, including a boundary
window. SOL tests distinguish full dense coverage from BF16 arithmetic identity.

Retain synchronized Reference+/candidate playback HTML, representative stills,
audio, metrics, settings, hashes and per-case dispositions. Provide the final
review pages; do not introduce a mandatory human-review/approval step. Optional
review notes can supplement the automated contract, but reports must distinguish
measured metric acceptance from any actual perceptual observations. B6 acceptance
does not establish full 50-step production equivalence.

## SOL and ANE attribution

For SOL, check the independent approximate-operator oracle, all-dense limit,
protected/local coverage, routing statistics and early-step/layer policy before
model quality. For ANE, verify the split/recombination against the same operator
and precision without offload. Then apply the **same frozen Reference perceptual
contract** to each incremental stage and the combined pipeline. Do not multiply
independent speed claims or mask one stage's quality loss with another's change.

---

# Profiling architecture

Each denoising evaluation reports:

```text
wall time

GPU execution time where available

CPU encode / wait

Metal allocations

peak live tensors

physical memory

compressed memory

swap

command-buffer submissions

kernel-dispatch counts
```

Add signposts around:

```text
AdaLN

QKV

Q/K normalization

RoPE

attention routing

attention

attention output

FC1

SwiGLU

FC2

gate/residual

Euler update
```

For SOL additionally report:

```text
query blocks

K/V blocks

exact blocks

approximate blocks

protected blocks

effective exact %

routing time

exact-attention time

approximation time
```

---

# Scheduling architecture

A native Metal backend does not automatically outperform MPSGraph.

The project must optimize complete execution scheduling:

```text
pipeline reuse

command-buffer boundaries

encoder reuse

argument-buffer reuse

persistent scratch

minimal CPU synchronization

minimal inter-kernel barriers
```

Benchmark:

```text
single kernel
single block
50 blocks
complete denoising evaluation
```

A kernel-level win must not be accepted if complete DiT execution regresses.

Pull the layout, scratch reuse and synchronization work needed for honest M2 adapter-inclusive measurements forward from milestone M5. Broad memory/VAE cleanup remains separate; no full-native M3 dependency may block a hybrid measurement. Re-profile attention, QKV, projection, MLP, CPU/wait and memory after M2E before opening M3A linears. Keep the M1 profile as historical evidence.

---

# Memory-lifetime architecture

Define explicit lifetime classes:

```text
persistent model weights

persistent conditioning

per-run state

per-step state

per-layer activations

attention scratch

SOL summaries

ANE compiled graphs, execution copies and transfer buffers

range telemetry and recovery buffers

quantized-weight scratch

VAE-only state
```

Allocate reusable arenas where appropriate.

The native Metal path should make lifetime boundaries explicit rather than relying on graph-object destruction.

M5 implementation assigns these lifetimes explicitly:

| Lifetime | Ownership and reuse |
| --- | --- |
| Model / prepared context | Mapped or copied BF16 weights, compiled pipelines, cached graph definitions, prepared conditioning; retain only when the caller enables caching. |
| Run | Geometry-specific DiT buffers, row maps, AV state and ANE execution state. Saved AV/checkpoint state owns its continuation data independently of GPU execution objects. |
| Step | Scheduler index, latents/velocity and diagnostic counters; no new full attention arena per step. |
| Layer | Existing DiT QKV, normalized inputs and projection outputs are reused in ordered execution. |
| Attention / SOL | `H3ScratchArena` owns named, typed, geometry-checked Metal buffers. Q/K/V adapter, range data, summaries, route flags and online-softmax partials persist across layers/steps. Separate buffers retain Metal hazard tracking. Head-major FP16 SOL merges may reuse the exact partial's output slot; row-major transpose keeps distinct storage. |
| Encoding | Local autorelease pools release encoders and temporary Objective-C objects. Immutable pipeline states are cached; inline small argument bytes belong to the command, not a growing per-layer cache. |
| GPU commands | At most two committed batches, with completed command objects retired promptly. Long sequences default to five blocks per batch, or one above 65,536 rows. Pressure checks may drain safe queued work and retry a reservation. |
| VAE | Completed production renders drop the DiT execution context and weights before final decoding even with prepared caching enabled; without live preview, this precedes VAE loading. Explicit production-VAE live previews necessarily overlap DiT and remain subject to the same budget. A paused run retains its context for immediate resume; decode-pressure checks can evict that cache when needed. |

The M5 reproduction uses **640×480, 362 frames, `inputs/2.jpg`,
`--ref-image-size max`**, matching the user's clarified workload. This is a
memory/lifetime regression; prior 243-frame quality records remain historical.
1344×768 memory tests execute **at most two denoising evaluations**. Keep
production VideoVAE's existing 256-pixel tiling and arithmetic unchanged.

The default guard now requires both at least 10 GiB reclaimable headroom and
`max(process footprint, Metal allocated bytes) + reserve <= cap`, where the cap
is at most **110 GB decimal**, with more headroom on smaller machines. It counts
compressed process memory, fails closed when monitoring is unavailable, and
accepts only lower `H3_MEMORY_LIMIT_BYTES` overrides. This replaces the old
headroom-only test and the unconditional Metal cache-admission success. Checks
before allocations, encoding, blocks and submission are conservative boundaries,
not an OS-enforced memory quota.

Scope copied tensor allocation/loading with local autorelease pools as well as
native and MPS dispatch. Plain C callers may have no outer pool: a temporary
buffer reference must not keep a freed weight's Metal reservation alive across
model loads. Clear cached graph-data wrappers when freeing a tensor. The M5
regression verifies copied-weight release without an outer pool and exercises
real repeated cancellation and cache reuse in one process.

With `H3_PROFILE=1`, `h3_memory` records distinguish footprint, resident,
process/system compression, wired memory, headroom, logical tensor bytes and
Metal reservations. `h3_lifetime` records scratch reuse, pipeline counts,
command retirement, in-flight peaks and pressure drains. Measure transformer
teardown and VAE load separately from denoising. Do not add overlapping footprint
and Metal figures or compare summed allocation traffic with live memory.


---

# Transformer → VideoVAE cleanup

Before production VideoVAE initialization, explicitly release everything no longer required:

```text
transformer weights where permissible

layer scratch

QKV scratch

attention arenas

SOL summaries

temporary quantization buffers

DiT pipeline objects not needed later

MPSGraph objects no longer needed by the selected hybrid/reference execution
```

Measure:

```text
physical memory

compressed memory

Metal allocations

reclaimable state
```

before and after teardown.

A specific objective is to understand and reduce the observed late-render increase:

```text
~93 GB
   ↓
~130 GB
```

physical memory.

Do not initially optimize VideoVAE compute itself.

---

# Quantization phase

Begin after:

```text
qualified Reference execution with original BF16 source weights
attention → conservative SOL → ANE → fusion dispositions recorded
```

is stable. Q8 is a later independent bandwidth/residency experiment on M4; do not assume M5 INT8 compute gains. Qualify SOL separately before claiming a quantized SOL composition. Full-native M3A/M3B completion is not a prerequisite. Quantized native linears may be justified independently by memory/bandwidth benefits; keep their quality and speed contribution separate from dense/SOL routing.

## Q8

Implement:

```text
Q8 weight storage
+
BF16/F16 activation path
```

and measure separately:

```text
memory saving

load-time saving

memory-bandwidth saving

denoise speed

quality deviation
```

On M4, Q8 may primarily be a memory/bandwidth optimization.

On M5, allow a separate native INT8 GEMM backend where hardware support justifies it.

M6 recipe 1 is explicitly selected with
`--backend metal --metal-weight-format q8`. It quantizes only the 200 repeated DiT QKV, output,
FC1 and FC2 matrices, from the original BF16 checkpoint (including a resolved
runtime LoRA fold). All conditioning, modulation, normalization, patch/final
projection and VAE weights retain their original paths. Attention dtype and
dense/SOL routing remain independent. No defaults change.

Each output row has symmetric signed int8 groups of 64 along K, FP32 scales
and round-to-even packing. Zero groups use scale 1; scales are bounded below by
FLT_MIN; nonfinite source weights are rejected. Loading uses bounded BF16
staging and keeps only packed weights resident. Full groups occupy 53.125% of
the corresponding BF16 matrix bytes. There is no persistent quantized cache.

The portable M4 kernel uses 32x32x32 simdgroup tiles. It reconstructs BF16
weights in threadgroup storage, consumes BF16 activations, accumulates in FP32
and writes BF16. No FP16 activation conversion or W8A8 arithmetic is implied.
Select the direct kernel with `--metal-q8-kernel simdgroup`. The default Q8
execution policy, `--metal-q8-kernel mpsgraph`, instead expands weights into
four context-owned BF16 scratch slots (one per projection shape), reused in
command order across all layers. Only about 0.771 GB of BF16 matrix scratch
is resident for the complete core; no full-model BF16 copy is retained.
Dequantization is a native Metal dispatch and is included in linear timing;
MPSGraph owns GEMM arithmetic. The 243-frame QKV microbenchmark showed this
composition close to BF16 throughput, while direct simdgroup was slower.
The selection is explicit, never a silent fallback.

Both paths retain explicit BF16 SwiGLU. A native kernel is not automatically a
performance win; record complete-step cost and extra activation/scratch storage.
ANE and BF16 SSD streaming are rejected with Q8. M5 uses the portable path
until independently qualified, and CUDA Q8 is outside this Metal milestone.

Required checkpoint section 37 records Q8 recipe, group size and kernel policy; old readers
reject it, old BF16 checkpoints retain their section-36 format, and resume
rejects an explicit change in weight format. Prepared DiT keys include Q8.
Conditioning tensors can still be reused without changing their identity.

Validation retains the 640x480/243-frame B1/B5/B6 gates, 50 blocks and at most
six evaluations. Compare each Q8 run against the identical BF16 attention
recipe and separately against Reference+; dense/SOL deltas cannot be counted
as Q8 gains. B6 uses the production VAE and the existing perceptual screens,
with a final playback gallery. Weight storage, startup conversion, native
linear timing, complete-step timing and total memory are separate metrics.
The user has skipped M6's CUDA comparison. Record local M4 results without a
hardware ratio. Any future hardware comparison must use a live RTX 5090 with
matched inputs/evaluation count; other GPUs and mismatched historical renders
cannot substitute for it.

M6's bounded local implementation/evaluation is complete; see
[the Q8 results](m6-q8-results.md). Observed B5 footprint falls about 40%,
while small timing differences do not establish a repeatable speedup. Both
linear policies pass correctness; direct simdgroup is slower and only the
default bounded-dequant policy has full-render coverage. All four incremental
and total Q8 B6 comparisons fail the frozen visual screens; SOL additionally
fails audio-envelope screens. Q8 remains opt-in, BF16 remains the
default, and no earlier acceptance is inherited by this composition.



---

# M5 transition

The M4 implementation must make M5 migration a backend-capability exercise rather than an engine rewrite.

Reuse unchanged where possible:

```text
.h3cond

B1/B2/B5/B6

profiling

Metal DiT orchestration

tensor lifetimes

tensor layouts

dense tiled attention architecture

SOL routing

protected ranges

VAE cleanup

Q8 representation
```

Then add M5-specific execution:

```text
M5 TensorOps

GPU NAX matrix acceleration (distinct from the separate ANE engine)

INT8 GEMM

optional SageAttention

M5-specific tile selection
```

Benchmark the retained MPSGraph reference, best M4 hybrid, and available portable native candidates on actual M5 Ultra hardware before extensive M5 micro-tuning. Qualify the M2 NAX attention branch independently and re-evaluate M3A/M3B if deferred on M4. Select TensorOps/NAX or other matrix paths only from capability checks and measured benefit; published M5-family results are not an M5 Ultra acceptance result.

---

# MPSGraph retention policy

MPSGraph retention follows measured value, not a mandatory retirement schedule.

## Stage 1 — current

```text
MPSGraph
=
production + reference
```

## Stage 2 — hybrid/native work

```text
hybrid/native Metal
=
explicit opt-in experiments

MPSGraph
=
production + reference
```

## Stage 3 — measured production selection

```text
best qualified hybrid or fully native path
=
eligible production DiT

MPSGraph
=
production operations where beneficial + selectable reference
```

Keep original MPSGraph dense selectable as Reference+ on M4 and M5. Remove
redundant execution code only when the oracle and useful hybrid operations remain
available; zero MPSGraph dependencies are not an acceptance objective.

---

# Architectural milestones

## M0 — Infrastructure: COMPLETE

Retain conditioning caches, B1/B2/B5/B6, profiling and the matched
`H3_DIT_COMMAND_BLOCKS=5` policy. M0's 362-frame results remain historical;
all new required render gates use 243 frames.

## M1 — Strict numerical evidence: retained, unqualified

Keep dense component passes, dense B5 failures, SOL B1 numerical failures and
T133's profile/GEMM deferral. The new Reference objective is a separate contract,
not a retroactive pass. Existing native BF16 kernels remain diagnostic assets.

## M2 — FP16 Reference and later fusion

M2A isolated FP16 + adapters → M2B integrated performance → M2C range-safe mixed
precision → M2D perceptual Reference qualification. The user accepted the measured
dense/SOL subset and selected M2E fusion ahead of deferred ANE. Native BF16 T144–T146 is optional and deferred.
FP16 can qualify final-render Reference, not only Preview.

## M4 — Conservative SOL, pulled forward

After M2C/D, reuse the safe FP16 kernel and FP32 statistics, M1 protections and
new dense early-evaluation controls. Qualify conservative settings against the
new Reference contract and the same dense FP16 execution. Do not wait for M3
or native BF16. Keep moderate/aggressive Preview separate.

## M3C — M4 GPU/ANE cooperation

After the dense/SOL decisions, audit and implement selected independent work
partitions, bounded compile caches and a traced static/dynamic split. Require
range/perceptual acceptance and incremental whole-step gain after all overhead.
Keep GPU-only when offload loses. M3A/M3B are not prerequisites.

## M3A / M3B — Conditional native linears and complete DiT

T041–T063 remain deferred. Only a new profile can justify broader conversion.
T162 covers selected range-safe FP16 linears as a bounded option. Full-native
coverage never substitutes for quality or complete execution benefit; strict
BF16 gates apply only to a path claiming that parity.

## M5 — Memory lifetime and VAE transition

Pull required scratch/layout/scheduling work into the experiments, including
ANE caches and recovery buffers. Keep broader VAE cleanup separate, production
VAE tiling at 256 pixels, and report memory/teardown costs independently.

## M6 — Weight-only Q8 execution

Evaluate Q8 after attention/SOL/ANE/fusion dispositions. Keep Reference+ intact
and measure bandwidth/residency, conversions and whole-step time separately.
No four-bit weight option is planned.

## M7 — Actual M5 Ultra validation

Qualify M5 NAX and the chosen numerical tier on real hardware, separately from
M4. Re-evaluate optional BF16, native linears, INT8, Sage and ANE contribution;
M4 or another M5-family result does not qualify M5 Ultra.

---

# Performance milestones and attribution

M2A requires ≥1.10× adapter-inclusive standalone speedup; M2B requires ≥1.10×
whole-step B5 improvement above observed variability. Conservative SOL and ANE
each require ≥1.10× incremental B5 gain against the **same immediately preceding
recipe without that stage**. Fusion must show repeatable incremental benefit
above variability without quality/memory regression. A losing stage is optional,
not a reason to discard the qualified dense Reference path.

Always report the combined Reference result against Reference+ as well. Retain
longer-term 1.25×/1.5×/2×/2.5× dense and 1.5×/2×/2.5×/3× SOL reporting targets,
but qualify every result by tier, dtype, protection, device and complete-step
measurement. These are aspirations, not predicted speedups.

Use the **measured local** attention fraction for Amdahl estimates:
`speedup = 1 / ((1 - p) + p / attention_speedup)`. Do not assume an external
85% attention share applies to our 243-frame M4 run. Independent kernel, SOL,
ANE and bandwidth gains overlap and cannot be multiplied. Separate cold load,
compilation and VAE wall time from warmed denoising. Matched 243-frame B5/B6 on
the same RTX 5090 can evaluate the approximate 3× M4 / 1.5× M5 slowdown targets;
old 15-second or 50-step wall times are not the new acceptance baseline.

# Revised implementation priority

```text
Reference+ unchanged; retain M1 failures and freeze new perceptual contract
    ↓
FP16 dense FlashAttention + explicit adapters, actual 243-frame H3 shape
    ↓
FP32 softmax/accumulation, range guards, BF16/FP32 sensitive state
    ↓
Reference perceptual/whole-step acceptance
    ↓
conservative SOL, dense early evaluations/layers and protected regions
    ↓
fused QKV → norm → RoPE → attention layouts and measured requalification
    ↓
selected GPU + ANE cooperation when resumed, with dependency/overlap evidence
    ↓
profile, then Q8; selected native linears only if still justified
```

Native BF16 attention is an optional branch. No 362-frame or 32K/37K gate and
no 50-step test is introduced. M4 simdgroup and M5 NAX implementations are
separate device candidates; full-native conversion and low-bit Preview do not
block the mixed-precision Reference path.

# Decision points

* **After FP16 proof:** distinguish operator correctness, range safety and visual
  similarity. A standalone win is insufficient; include adapters and B5/B6.
* **After perceptual qualification:** accept the declared Reference recipe only
  under the frozen new contract. Keep old BF16 drift reports and Reference+.
* **After conservative SOL:** check routed work and all protections, then quality
  and incremental gain. Keep dense fallback when protection or quality removes
  useful savings; do not force a sparse fraction.
* **After ANE:** require traced useful overlap, bounded compile/residency costs,
  correct dependencies and a complete-step gain. Keep GPU-only when it wins.
* **After fusion:** rerun the combined contract and ablations; count range
  recovery and residual/sampler semantics as part of the implementation.
* **Before Q8 or broader linears:** use the new profile and original BF16 weights
  as baseline. Do not assume low-bit arithmetic acceleration on M4 or make it
  part of the earlier attention claim.
* **On M5 Ultra:** qualify the actual device/precision combination before
  asserting NAX, INT8, full-native or ANE benefits. Reference quality is not
  synonymous with one storage type or compute engine.

---

# Final M4 acceptance

Use matched 640×480, 243-frame, 50-layer runs with exactly six denoising evaluations, fixed prompt/seed/references, and production VideoVAE for final quality comparisons.

Required configurations are **Reference+** and each proposed Reference recipe:
safe dense FP16, conservative SOL, selected ANE and fused combinations. Qualify
each increment and the final composition under the versioned perceptual contract,
with hard stability/protection checks and matched whole-step performance. Strict
native BF16 remains optional; Preview/Turbo/Q8 have separately labeled evidence.
Produce final paired playback HTML and metric records without a new mandatory
human-review approval step. B6 evidence does not claim validated 50-step quality.

The final report separates:

* conditioning-cache savings;
* dense mixed-precision gains, adapters, range recovery and optional BF16 results;
* incremental conservative SOL and ANE gains, plus total gain over Reference+;
* ANE cold compilation, warm-cache residency, real overlap and contention;
* Q8 storage/arithmetic contributions;
* layout, scheduling, and memory-lifetime contributions;
* transformer teardown and VideoVAE cleanup results;
* peak RAM, compressed memory, and swap;
* numerical errors, decoded quality, Ref2VA, continuation, and A/V findings;
* matched RTX 5090 per-step ratios.

Record unqualified, qualified, failed, and deferred modes distinctly. Neither a documentation gate revision nor a component PASS changes a runtime default.

---

# Final architectural target

The objective is fast, maintainable H3 execution with original BF16 dense as Reference+, a visually extremely close mixed-precision Reference renderer, and a separate aggressive Preview tier. Original architecture, weights, production sampler/schedule, conditioning, VAE and seed semantics remain fixed initially. Qualified FP16 attention, conservative SOL and selected GPU/ANE execution are primary M4 tools; full-native BF16 and later quantization are optional.

On M4, use the best measured hybrid/native architecture. On M5 Ultra, re-evaluate complete native Metal DiT with actual hardware evidence. Keep orchestration and explicit lifetimes under `h3cli` control, original MPSGraph dense selectable as Reference+, and useful MPSGraph production operations. Full-native coverage is optional.
