> Historical renderer/qualification record. Commands that refer to removed
> fast/legacy modes require the archived source. Use the
> [current single-pipeline design](design-single-pipeline.md) and
> [M6/M7 qualification](single-pipeline-final.md) for supported execution.
> The separate encoder process described in this history has since been retired.
> Current encoding and decoding share cuDNN 9.20 in the main process; see the
> [runtime guide](cuda-sglang-reference.md#build-and-runtime).

# Pinned CUDA reference audit

Status: content qualification passes for all three primary cases, all three
held-out prompts and all five conditioning cases. Every free-running AV
trajectory matches the pinned oracle bit-for-bit, including all 50 C2 updates;
final video/audio passes every frozen content gate. Joint reference speed/VRAM
measurement gates do not all pass. The user subsequently accepted all results,
closed all 40 tasks and requested SGLang-equivalent rendering as the default.
Full fast outputs differ; that behavior change is also explicitly accepted.
`--fast-cuda` remains version 1. Historical open/failed entries below describe
their original observations and are retained unchanged.

The final arithmetic identity is 4. See the final soundtrack
section below for the R1 fix and the
[selected full-length comparison gallery](../../outputs/cuda-sglang/review/final-review.html).
Historical failed attempts remain in the complete report and this audit.

The oracle is the installed SGLang 0.5.20 / PyTorch 2.13.0+cu130 environment on
the RTX PRO 5000 at the qualification host. Its packages and source are retained under
`outputs/cuda-sglang/`; they are not updated during this campaign. The native
baseline is commit `e69a3a42c7dac1910493aab2fd65bf89b7b76ad9`, built separately
under `/path/to/qualification/baseline`. Candidate builds use a separate directory.
Model inventory records paths, sizes and modification times; weights are not
hashed or copied.

## Execution differences found in installed source

| Boundary | Existing native behavior | Installed oracle |
| --- | --- | --- |
| Qwen RMSNorm | Single BF16 output cast after the weight | Cast normalized values to BF16 before multiplying BF16 weights |
| BF16 linears | Linked cuBLASLt 12.8, FP32 partial reductions on the first Qwen projection | cuBLAS 13, BF16 partial reductions permitted |
| Initial noise | PCG/Box–Muller | CPU MT19937, AVX normal conversion, low 24-bit uniforms |
| Audio RNG traversal | Channel/stream/time | Stream/time/channel rows |
| Sigma grid | Native serving schedule | CPU FP32 `torch.linspace`, then shift 12/3 |
| Euler update | Fused velocity update | Separate FP32 clean estimate, scale and sum |
| AdaLN | Combined arithmetic followed by BF16 output | BF16 norm output, rounded `1+scale`, rounded product, then shift |
| Gated residual | Combined multiply/add | BF16 product before residual addition |
| SwiGLU | Final product rounded | Activation rounded before product |
| Q/K norm and RoPE | Combined norm/rotation | BF16 norm, BF16 rotary products, then sum |
| Noncausal scale | BF16 scale matching the Metal oracle | Original FP32 scale |
| Final velocities | BF16 output, including the F32-head diagnostic | FP32 final projection output |

The original checkpoint has hidden width 5,376, **56 attention heads of width
128**, attention inner width 7,168 and MLP width 14,336. Derive the workload from
the checkpoint; an eight-head attention microbenchmark is not the complete H3
attention workload.

`runtime/platforms/cuda.py` disables cuDNN SDPA. The installed `SDPAImpl`
variable-length shortcut additionally requires compute capability 9; it does
not apply to this SM120 GPU. Its fallback runs ordinary PyTorch SDPA separately
over the live packed segment and any trailing padding. Native layout has no
padding segment. Actual captured tensor strides and live bounds must agree
before arithmetic comparisons. cuDNN is a candidate native backend, not an
assumed copy of the oracle algorithm.

The original checkpoint's Qwen encoder consumes the selected intermediate
residual stream without the language model's final norm. The full video VAE
uses FP16 decoder weights; the audio decoder uses FP32 with TF32 disabled and
deterministic convolution selection. The existing native FP32 decoder and
balanced/preview modes must retain their current meanings.

## Implemented policy

`--cuda-reference sglang` selects a separately versioned, non-fast CUDA recipe.
It currently provides native matching CPU noise, schedule and sampler arithmetic,
reference-specific BF16 kernels, a private cuBLAS 13 handle and FP32 velocity readback.
The recipe requires `libcublas.so.13`;
`H3_SGLANG_CUBLAS_LIBRARY` can name its absolute path. This library is loaded
only for reference contexts with local symbol binding. Legacy/fast contexts
continue using their original linked cuBLAS 12.8 library. No Python runtime is
added to the executable. It is an optional
candidate; primary final-media content passes, but the complete qualification
matrix remains open.

The request scope is captured by GPU/DiT constructors. Conditioning identities,
prepared-object keys and a required sampler extension distinguish the recipe
from legacy/fast execution. The old fast extension and version are preserved.
Existing fast requests cannot select the reference recipe, including when an
environment override disables fast acceleration.

## Retained tests and evidence

- `tests/cuda_sglang_host.py`: exact comparisons against the installed CPU
  oracle. Seeds 0, 42, 987654321 and 2³²+42; aligned/tail and production-sized
  tensors; one, two, six and 50-evaluation schedules and Euler updates.
  `records/host-parity-v2.json` records all comparisons passing bit-for-bit.
- C0, C1 and C2 each have two complete, byte-identical SGLang MP4s. The
  machine-readable repeatability record and all six videos are retained in
  [the local playback report](../../outputs/cuda-sglang/review/review.html).
  `tests/cuda_sglang_contract.json` freezes the numerical/media limits and held-out
  inputs before the first candidate CUDA render. Native comparisons are serial.
- Existing host tests pass on Linux and macOS. New tests cover distinct
  reference checkpoint/cache identity, legacy identity restoration and the
  explicit 50-evaluation test budget. Full fast GPU regression remains open.
- `tests/cuda_sglang.py` retains exact commands, binary/setup checksums, logs,
  completed evaluations, FFprobe metadata and full media-decode validation.
  NVML and host memory are sampled every 50 ms; the maximum observed sampling
  gap is reported. Driver calls blocked for up to 9.4 seconds during loading
  and teardown in the initial runs; their memory sampling gate is therefore
  incomplete. The collector now runs in a separate thread so those stalls
  cannot inflate renderer process time. Initial wall timings are conservative
  polling measurements and will be replaced by fresh paired measurements.
  Valid media does not imply content parity.
- `tests/cuda_sglang_capture.py` installs read-only hooks through a test-only
  import path. It never edits the installed oracle. Captures preserve tensor
  dtypes, include checksums and have an 8 GiB process budget. Native diagnostic
  capture has the same budget. Instrumented runs do not qualify performance.
- `tests/cuda_sglang_compare.py` reports exact input checks and numerical
  differences at captured boundaries. Free-running comparisons are identified
  separately from teacher-forced replay and final decoded-media gates.

Failed preflight/output-path attempts remain under `/path/to/qualification`.
In particular, configuration-only SGLang output paths were ignored by its
request layer; the runner now passes `--output-file-path` explicitly. Their
elapsed time and failed statuses are not rewritten as successful runs.

## First-divergence evidence

- Initial AV noise, sigma grids, token IDs and packed positions match exactly.
  The sampler comparison also passes bitwise against CUDA Torch operations.
- Qwen layer zero input RMSNorm differed at 60,478 of 230,400 elements.
  Adding its intermediate BF16 cast makes this output bit-identical.
- On those identical normalized inputs, native cuBLAS 12.8 disagreed with the
  oracle at 106,034 of 368,640 first-projection values (relative L2 0.002419).
  Changing only flags, legacy/new API entry points or the 32 MiB workspace did
  not resolve it. The same operands through cuBLAS 13 reproduce every oracle
  output bit. Disabling BF16 partial reductions in Torch instead reproduces
  the native 12.8 result. This is a library/reduction difference, not a tokenizer
  or checkpoint mismatch. The private reference-only loader passes that replay.
  The [pinned Torch cuBLAS implementation](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/cuda/CUDABlas.cpp)
  confirms the allowed-reduction setting; retained replays establish the
  behavior on this GPU.
- The Qwen-rounding-only candidate still fails full content gates. Its step-zero
  video velocity relative L2 is 0.0310; final video-latent relative L2 is 0.3613.
  Every-frame PSNR, SSIM, LPIPS, temporal, worst-region and audio gates all fail.
  No limit was adjusted. The cuBLAS-13 candidate is under evaluation.
- The instrumented SGLang C0 MP4 has the same SHA-256 as both uninstrumented
  oracle repeats, validating those capture hooks on C0.
- Old native attention/MLP captures included unused tails of aliased QKV storage.
  New captures export explicit logical sizes. The comparator validates and
  selects the documented live prefix when reading those older artifacts.
- Initial native baseline runs also saved AV states, which include existing
  compatibility-signature work. New uninstrumented campaigns omit that diagnostic
  work; final-state captures are retained separately and cannot qualify timing.

## Further isolated corrections and rejected candidates

- Pinned PyTorch FlashAttention source is
  `Dao-AILab/flash-attention@6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15`.
  The runtime profiler resolves a BF16 128×128×64, four-warp forward kernel.
  The native cuDNN candidate failed the frozen absolute operation bound on
  both real Qwen and production DiT inputs and was rejected for this recipe.
- PyTorch defines `UNFUSE_FMA` for the FlashAttention softmax and does not use
  the prototype's `--use_fast_math`. The latter left three differing attention
  values in Qwen layer 2 and 162 in the first production DiT tensor even after
  `UNFUSE_FMA`. With the pinned PyTorch CUTLASS headers and ordinary division,
  both retained same-input tensors match **bit-for-bit**, including all
  82,854,912 elements of the 11,559-row DiT attention output. This establishes
  those operation replays, not a full-trajectory pass.
- Qwen's FP32 mean reduction uses four independent accumulators per lane,
  followed by descending warp shuffles. Matching this order plus the strict FlashAttention build makes all 50
  Qwen blocks and the final selected text embedding bit-identical on C0.
  The earlier three-value attention discrepancy was enough to cause later
  encoder drift; its failed candidate is retained.
- The installed oracle gives cuBLASLt **1 MiB**, separately from its ordinary
  cuBLAS **32 MiB** workspace. API logs show that 32 MiB selects a different
  K-reduction algorithm for the identical text-conditioning projection,
  producing relative L2 0.004356 and maximum error 512. The reference-only
  Lt handle now uses the observed 1 MiB budget; legacy/fast workspaces are
  unchanged. The corrected condition-projection replay and the complete
  C0 conditioning projection now match bit-for-bit.
- Original native timestep preparation rounded the time embedding to BF16
  before SiLU. SGLang evaluates SiLU in FP32 and then casts. The reference-only
  branch now follows that order. A same-input modulation replay confirms the
  corrected order; it does not imply all native timestep inputs are equal.
- Captures now include every Qwen block and the text refiner when explicitly
  requested, still within the 8 GiB capture budget. The extended C0 oracle
  capture produces the same MP4 SHA-256 as the two uninstrumented repeats.
- The FlashAttention/FP32-patch candidate remains numerically unqualified:
  teacher-forced step-zero video-velocity relative L2 is 0.02151 (limit 0.01).
  Its complete C0 output is retained. This diagnostic measured a 10.38 GiB
  sampled total GPU peak with a passing sampling interval; instrumented timing
  is excluded from qualification. Projection and text-refiner differences
  remain under investigation.

## Subsequent exact boundaries and remaining trajectory drift

- Q/K normalization used `x * (inverse * weight)`. The installed fused kernel
  evaluates `(x * inverse) * weight`; this changed two BF16 query values and
  two key values in the first text refiner. Matching the association makes
  those real-tensor replays exact.
- PyTorch fused DiT RMSNorm uses 128 threads, groups of four contiguous
  values, descending warp shuffles and a four-warp reduction. The old scalar
  reduction changed one value in the second refiner and many production rows.
  The new reference-only reduction matches all 62,275,584 values in the padded
  first-block norm replay. Both complete text refiners now match exactly.
- Batching timestep projections changes rounding even when the input rows
  are identical. Native and Torch agree exactly at each tested batch size
  (1, 2 and 11), but the one-row result differs at 40 BF16 values from the
  larger batches. The T2VA schedule now retains the oracle's per-evaluation
  batch sizes while precomputing. Legacy and fast schedules are unchanged.
- The resulting `candidate-batches-C0` matches every first-block output and
  the **complete step-zero video velocity and updated video latent bitwise**.
  Step-zero audio velocity relative L2 is 1.21e-6, with maximum error 1.24e-5.
  Subsequent steps still diverge: final video-latent relative L2 is 0.2522
  (limit 0.01), so the candidate remains unqualified. Later timestep features
  and final-head GEMM geometry are being localized; the oracle applies both
  final heads to every padded row before selecting media rows.
- Retained operation replays include `qknorm-replay`, `rms-replay`,
  `time-replay-v1` and `time-replay-v2` under `/path/to/qualification`.
  These are numerical diagnostics, not whole-render timing qualifications.

## Remaining qualification

The subsequent `candidate-head-shape-C0` closes the six-evaluation T2VA
denoising divergence: **all six video and audio sampler updates are bitwise
identical** to the pinned oracle from ordinary native prompt/seed input.
CUDA timestep features removed tiny CPU-libm differences. The audio head also
needed the oracle's GEMM algorithm selected for the full padded sequence;
executing that algorithm on the compact audio rows preserved exact output
without allocating a second full-sequence projection. The first-block and
text/refiner captures remain exact. Full decoder, C1/C2 trajectories and
performance qualification are still separate, open gates.

The saved C0 MP4 also exposed a delivery mismatch unrelated to denoising:
the installed oracle truncates RGB float samples to bytes, uses x264 `fast`
at CRF 25 and the AAC default, and resolves ImageIO's FFmpeg 4.2.2. Legacy h3cli
rounds RGB values and uses CRF 6 / AAC 192 kb/s through system FFmpeg 6.1.1.
The reference-only streaming delivery now follows the oracle's conversion
and codec settings, recorded as presentation codec 2. Legacy/fast codec 1
is unchanged. Matching tests explicitly select the oracle FFmpeg binary;
pre-codec RGB/PCM comparisons remain required, so codec agreement cannot
conceal decoder errors. Decode-only calls require explicit reference selection
for codec-2 states; ordinary states are not silently reinterpreted.

The complete **C1 362-frame/six-evaluation** free-running trajectory now also
matches every video/audio velocity and sampler update bit-for-bit. Its
instrumented oracle MP4 retains the earlier repeatability SHA-256
`e70c97fe8700e4f850347808a3383ca5445619ee706d13911a396615d85c2d5a`.
The complete **C2 50-evaluation** comparison subsequently passed bit-for-bit
for every video/audio velocity and update as well. The ordinary native seed
and prompt produce those states without imported oracle inputs. All 200 C2
velocity/state comparisons are retained in the trajectory report.

The full original video decoder now has a reference-only FP16 path matching
the installed autocast boundaries. Same-input first-tile testing gives exact
agreement at all 15 captured first-block boundaries, including QKV, norm/RoPE,
dense attention, both residuals and SwiGLU. Its final RGB maximum error was
3.58e-7. The fixes include separately rounded post-quant convolution bias,
FP16 coordinate construction for RoPE, FP16 embedding output before FP32
register concatenation, FP32 norms/residuals and the SM120 separate W2 bias.
Full-clip testing then exposed a different error: native delivery clipped each
tile before blending, whereas the installed decoder blends normalized values
before inverse normalization and clipping. The new policy now defers that
conversion to the final blended samples. The corrected complete C0 RGB has
maximum error 4.17e-7 and RMS error 3.01e-8 against the identical clean oracle
latent. Raw PCM has relative L2 2.67e-6, cosine 0.999999999996 and maximum
error 2.54e-6. Every-frame perceptual and encoded-media checks are running;
C1/C2 decoder checks remain separate gates.

The source-independent fast fixture in `tests/cuda_sglang_isolation.c` produces
identical BF16 output when linked against the untouched baseline and candidate.
It also passes fast→reference→fast→reference→fast contexts, repeated dispatches,
an allocation plateau and cancellation after invalid arguments. This focused
check uses fixed GEMM tuning. Full fast-render, quantization, checkpoint and
performance regression gates remain open.

Complete the checklist now archived in [the parity task list](cuda-sglang-parity-tasks.md), including identical conditioning,
teacher-forced early/middle/final comparisons, all 50 free-running C2 updates,
crossed full-VAE decoding, held-out/conditioning cases, interleaved timing/VRAM
pairs and unchanged fast-mode evidence. Every required failed or unmeasured
gate remains open. The final report must distinguish numerical fixes from
unqualified candidates.


## Current primary content result and performance candidate

`reference-cache-C0-v2`, `reference-cache-C1-v2` and `reference-cache-C2-v2`
render normally from the frozen prompt/seed, with no imported oracle tensors.
All decoded video frames are bit-identical to their matching SGLang outputs;
all six frozen final-media gates pass, including audio. C1 and C2 also pass
all captured velocity and sampler-update comparisons bitwise, including every
one of C2's 50 evaluations. These instrumented runs establish content; their
wall times are not the required interleaved performance qualification.

The final decoder corrections match Torch's 128-thread Welford LayerNorm,
its explicit FMA order, and overlap blending with a pre-rounded reciprocal.
The complete C0 pre-codec RGB tensor is now bit-identical too. Raw C0 PCM has
maximum absolute error 2.54e-6 and relative L2 2.67e-6. Reference-only GPU unpack
retains unclamped values until after overlap blending and performs pinned D2H
readback. A full C0 decode including raw-file writing measured 7.507 seconds;
additional graph reuse remains a candidate pending numerical validation.

A reference-only host weight cache removes repeated pageable-file staging from
DiT streaming. The cache has a process-wide 40 GiB pinned-memory cap, admits
allocations against the existing memory guard with a 16 GiB reserve, and falls
back to the existing two 16 MiB transfer buffers when admission fails. The
200 DiT entries occupy 35.889 GiB and are released before VAE decoding. Cache
identity uses file device/inode, size, mtime/ctime, offset and dtype, without
weight hashes. Profiling measured a C0 denoising reduction from 71.47 to 31.32
seconds; this is diagnostic evidence, not a final timing gate. Isolation v3
covers cache hits, metadata invalidation, pressure fallback, release, context
switching and unchanged baseline fast BF16 fixture bytes.

The explicit reference recipe also uses versioned local metadata identities
for AV-state/model/component compatibility, avoiding legacy full-weight hash
reads. Legacy/fast identities retain their previous meaning. Reference states
from the earlier prototype that used a legacy content signature are incompatible
with the new metadata signature and must be regenerated; they are never silently
reinterpreted. Metadata-bound states refer to that local file inventory, not a
portable guarantee of byte-identical models across machines. Model files are
neither hashed nor copied by the new reference startup path.

Condition augmentation now reproduces the pinned CPU generator's full visual
noise shape before slicing and uses the rounded `1 - float32(0.999)` coefficient.
CPU tests match installed SGLang bitwise for image/video layouts, multiple
conditions, both target durations and audio. GPU encoder/posterior conditioning
and complete conditioned renders remain separate pending gates. The frozen
conditioning corpus is `tests/cuda_sglang_conditioning_cases.json`.

## Held-out and conditioning follow-up

All three frozen held-out cases (H0–H2) now have bit-identical full video/audio
denoising trajectories and decoded video frames. Their final AAC audio fails
the frozen gate: initial relative-L2 errors are 0.09565, 0.10428 and 0.01496.
H0 raw decoder PCM is much closer (relative L2 4.04e-5); independently encoding
the oracle PCM reproduces the oracle AAC exactly, isolating the remaining issue
to decoder arithmetic. The native scalar weight-norm reduction differs from
Torch's 256-thread reduction and `(g*v)/sqrt(sum)` rounding order. The new
reference-only kernel matches all 14,680,064 first-convolution weight elements
bit-for-bit and reduces raw PCM error to 9.46e-6. AAC still fails; this is not
a final decoder pass. The original fast normalization launch is unchanged.

Reference-image diagnostics identified division by 255 versus reciprocal
multiplication, separate BF16 patch-convolution bias rounding, full-precision
position interpolation/RoPE, vision LayerNorm/GELU arithmetic, and the missing
72-dimensional reference attention route. Identical-input first-layer
LayerNorm, QKV, MLP projection and GELU now match Torch exactly. The remaining
patch-convolution reduction differs from GEMM. A cuDNN 9.20 tensor-op
implicit-precomputed-GEMM replay is exact; the optional linked cuDNN 9.10
variant differs. Loading a second cuDNN family in an isolated ELF namespace
failed CUDA initialization and was rejected. A subsequent cuBLAS replay
matched exactly after rearranging the operands from NCDHW to NDHWC, padding
three channels to sixteen, and disabling reduced-precision partial reductions
for this projection alone. That adaptation is now the native candidate;
no second cuDNN runtime is introduced and fast cuDNN selection is unchanged.
The subsequent rotary audit found that the oracle initializes its 18 fixed
frequency constants on CPU; recomputing them with GPU `pow` changes rounding.
Using the pinned CPU FP32 values makes the full vision tower bit-identical:
patch embedding, first-block Q/K/V/attention, merged output and all three
deep-stack outputs match in `vision-native-F0-v8` against
`vision-oracle-F0-v3`. This is a real 640×480 image replay, not imported
hidden states; full conditioning and encoder latents remain separate gates.
First-frame encoder posterior handling is also under validation; the complete
conditioned corpus has not passed.

The first uninstrumented C0 timing attempt measured native process wall
122.45 s, complete playable output 122.12 s, denoising 28.46 s, audio decode
1.94 s, video-decoder load 3.29 s and video decode 8.09 s. Peak total GPU
usage was 6,447,104,000 bytes. The fresh SGLang attempt measured process wall
244.40 s, playable output 220.60 s, generation 41.75 s, denoising 31.15 s,
joint VAE decode 6.33 s and peak total GPU 17,645,895,680 bytes. Native cold
startup is faster, but its video decode alone exceeds the oracle's joint
decode time. Native generation after model readiness is not yet separately
qualified; subtracting an arbitrary startup amount would not establish it.
Three matched interleaved pairs per primary case remain required.

SGLang's NVML calls block during startup and teardown, with gaps up to 9.13 s
even using `nvmlDeviceGetMemoryInfo_v2`. Separating host `/proc` collection
confirmed that NVML itself is responsible (9.08 s maximum versus 2.8 ms for
host reads). The native collector stays below 61 ms. Keep these failed
sampling attempts visible; allocator peaks and a lower native observed peak
do not by themselves satisfy the frozen 100 ms sampling requirement.

The local playback report includes H0–H2 and all timing attempts, with the
failed audio gates and unqualified performance clearly labeled. Latest local
shared-code checks pass 1,772 sampler, 34 conditioning and 675 progress checks.

## Subsequent decoder and conditioning diagnostics

`audio-campaign-v1` now passes the frozen decoded-audio gate on **C0, C1, C2,
H0, H1 and H2**, using retained clean states, fresh decoder processes and the
pinned AAC encoder. All 124-frame raw PCM and decoded AAC samples match
bit-for-bit, including the three previously failing quiet held-out clips.
C1 raw PCM relative L2 is 7.63e-7; its decoded AAC relative L2 is 0.0024831
and cosine similarity 0.9999969. This crossed decoder campaign does not
replace the remaining whole-render, paired-performance and memory checks.
The old failed media and their scores remain retained.

The reference-only fixes are separate FP32 cuDNN convolutions with explicit
algorithm selection, bias-after-convolution rounding, sum-then-average
residual blocks, and the pinned JIT's SnakeBeta arithmetic. That JIT profiles
two static shapes and one dynamic shape before using its dynamic fused
kernel; its first calls use separate multiply/add, later calls use FMA.
Real first-stage captures now match at all 36 activation/convolution
boundaries. One transposed convolution requires cuDNN frontend engine 1;
the legacy API's engine 2 differs. All these plans and scratch buffers are
owned by the reference component context, bounded, and absent from fast
CUDA dispatch. Audio replay including weight loading takes about 0.6–0.9 s
at 124 frames; those individual diagnostic timings are not a matched speed
qualification. The tested reference decoder now requires the optional
`CUDA_CUDNN=1` build; an unavailable runtime gives an explicit error.

The complete F0 input capture now has identical tokens, Qwen text, positions
and sigma arrays (`native-inputs-F0-v3`). Its encoder latents still fail the
operation maximum-error gate. The video encoder uses reciprocal pixel
scaling, whereas Qwen uses division; a scoped adapter on the owned FFmpeg
byte-derived pixels preserves both recipes without changing arbitrary float
image APIs or fast CUDA. The pinned video encoder also allows TF32 cuDNN
convolutions, unlike its FP32 audio decoder. Matching that policy gives an
exact image input and first convolution in `encoder-native-F0-v2`.
GroupNorm is the next identified boundary and is being qualified against
the pinned PyTorch Welford/affine implementation. Initial convolution probe
fixtures incorrectly modeled the spatial padding as zero/replicate; the
correct reflect-padding fixtures are retained as `encoder-convs-F0-v3`.


## Qualified first-frame conditioning and bounded GPU stitching

`native-worker-F0-v1` now passes every decoded-video and audio gate against
`oracle-condition-F0-v1`, and all six video/audio velocities and target sampler
states match bit-for-bit. A diagnostic with imported condition rows first
proved that the transformer and sampler already agreed. The earlier
`native-condition-F0-v1` failure remains retained; passing an encoder operation
bound alone was insufficient to prevent later diffusion drift.

The causal video encoder now matches pinned PyTorch's GroupNorm Welford,
affine FMA and SiLU arithmetic. With linked cuDNN 9.10, the first six residual
blocks and first two downsampling convolutions match exactly; the third
strided convolution is the first divergence. cuDNN 9.20 makes the entire
image encoder, all 48 stitched moments and the sampled condition exact.
The bounded legacy/frontend algorithm probes could not reproduce that
convolution with 9.10. This is a library-version arithmetic difference.

`bin/sglang_encoder_worker` runs the native C/CUDA encoder with the pinned
9.20 library family in a separate process. The parent, audio decoder and
fast CUDA continue using their original linked cuDNN. No Python inference
is introduced. `H3_SGLANG_CUDNN_LIBRARY` identifies the absolute 9.20 facade;
the worker is built beside `h3cli` with `CUDA_SGLANG=1 CUDA_CUDNN=1`. Anonymous
memfd staging is bounded to 512 MiB of input and 128 MiB of result. Fixed
progress records preserve cancellation; the parent kills/reaps a cancelled
worker and checks exact result size/shape/finiteness. The real-image recovery
test passes, including identical fast output before/after and an unchanged
parent cuDNN version of 91002. All model identities remain metadata-based.

Reference full-video decode now retains at most 256 MiB of spatial tiles
and 128 MiB of stitched output on the GPU for one temporal chunk, plus at
most 128 MiB of pinned readback. Larger geometries use the existing bounded
CPU stitch path. The blend retains separate multiply/add and above-then-left
ordering. `vae-stitch-C0-v1` matches all 114,278,400 raw RGB floats exactly.
Repeated decode measurements are 6.279/6.287 s with GPU stitching versus
7.198/7.206 s with CPU stitching; the first runs also include graph setup.
Device tensor allocation plateaus at 5,569,141,492 bytes. This is a local
optimization comparison, not the remaining three-pair joint speed gate.

The trajectory comparator now removes only the oracle's explicitly recorded
conditioning prefix when comparing target sampler states. Velocities already
contain only target rows. The Ref2VA oracle harness explicitly selects the
Ref2VA checkpoint partition instead of inheriting the T2VA sample's FL2VA
partition. Remaining conditioning cases are being tested before full renders.

### Exact multi-image conditioning inputs and crossed full-video replay

`vae-campaign-v2` verifies the independently generated native/oracle clean AV
payloads are bit-identical for C0, C1 and C2, then executes both full video VAEs
on their common payload. All raw RGB floats match exactly, including 362 frames
and the 124-frame state after all 50 evaluations. Equal payloads make duplicate
crossed decodes redundant; the equivalence and state checksums are retained.
The earlier `audio-campaign-v1` supplies the corresponding crossed PCM/AAC
checks. This qualifies decoder content, not matched performance. Native fresh
raw decode measures 6.79/19.78/6.84 seconds for C0/C1/C2 in this diagnostic;
model loading, raw file-write costs and oracle delivery timing are recorded
separately, and none is represented as complete generation timing.

Input diagnostics `native-inputs-F2-v7` and `native-inputs-R0-v7` now match every
captured input bit-for-bit. F2's first divergence was the incoming Qwen vision
embeddings: processing images separately selected different projection batch
shapes from SGLang. The reference policy now packs all image projections into
one tower invocation while preserving each image's independent attention range.
Video pairs form a separate packed batch, matching Qwen's modality calls.
Legacy/fast calls retain their single-input execution.

Ref2VA R0 also required always scaling the image to a 2048-pixel short edge,
with nearest-even 32-pixel rounding, including upscaling. Its original JPEG
pixels differed under FFmpeg before any resizing. The native reference path
uses TurboJPEG's accurate DCT/default chroma upsampling and Pillow 11.3's
fixed-point Lanczos recipe. The portrait fixture matches every decoded/resized
byte. `H3_SGLANG_JPEG_LIBRARY` can select a private runtime; the server's
user-owned Ubuntu libturbojpeg 2.1.5 extraction leaves system libraries intact.
Pillow is used only by validation, and its MIT-CMU attribution is retained.

Full F1/F2/R0 renders and held-out requalification are in progress. R1 exposed
additional legacy assumptions: a two-second reference minimum, truncation of
the soundtrack to visual duration, and direct FFmpeg resampling to 32 kHz.
The pinned oracle accepts the one-second fixture, retains its AAC tail up to
the requested output duration, then uses 44.1 kHz PCM followed by torchaudio's
32 kHz sinc resampler. The native reference correction is being tested; R1
remains unqualified until its complete conditioning and output gates pass.

The subsequent F1 full run (`native-full-F1-v8`) passes its complete six-step
trajectory bit-for-bit and produces a byte-identical MP4 to
`oracle-full-F1-v1`. F2's full run **fails** despite exact conditioned inputs:
first video/audio velocities have relative L2 0.01254/0.01238; by evaluation six
its state errors exceed the frozen trajectory gate. This is a separate DiT
boundary investigation; neither the passing input diagnostic nor F1 qualifies
F2. Its failed attempt and wall time are retained.

`preprocessing-v2` passes exact CPU comparisons for five JPEG/PNG resize cases
and the complete reference soundtrack. All resampled PCM samples are
bit-identical to the pinned torchaudio result (including the 1.02403125-second
AAC tail). System package installation was unavailable without a password;
only a user-owned runtime extraction was needed and used.

`native-full-H0-v8`, `native-full-H1-v8` and `native-full-H2-v8` pass the
complete six-evaluation trajectories and every video/audio media gate against
the frozen held-out oracle runs. Their local players and records are retained.
R0's full result fails beginning with the first video velocity, despite its
exact input diagnostic; it remains unqualified alongside F2.

The pinned soundtrack encoder explicitly disables cuDNN and TF32 and selects
math SDPA inside `_AudioVAEDeterminismContext`. Its convolution fallback is
PyTorch's channel-first im2col plus per-channel NN SGEMM with prefilled bias
and beta=1. A reference-only implementation follows that ordering; the full
audio decoder retains its independently qualified cuDNN path. The initial
cuDNN encoder probe and a failed oracle capture-hook attempt remain in the
records. GPU encoder qualification is still pending.

### Rotary tie rounding, reference identity 2 and temporal encoder follow-up

The F2/R0 DiT boundary investigation found identical QKV projections and
normalization followed by a handful of different BF16 rotary values. Host
libm and CUDA sine/cosine round differently at these particular angles (for
example temporal position 746). The pinned oracle transfers FP32 angles to
CUDA before computing trigonometry. The native reference path now does the
same, retaining the original fast/legacy host calculation. `native-rope-F2-v9`
and `native-rope-R0-v9` pass every media gate and all six video/audio
velocities and sampler states bit-for-bit. Earlier failed runs remain visible.

The reference arithmetic identity is now `H3_SGLANG_VERSION=2`.
New conditioning/AdaLN keys include this identity, and sampler replay rejects
the earlier reference identity 1. Legacy and fast serialization remains
unchanged, including `H3_FAST_CUDA_VERSION=1`. The host sampler suite passes
1,773 checks, including old/unknown reference identity rejection.

The soundtrack fallback now passes the real R1 audio encoder replay:
`audio-encode-native-R1-v2` versus `audio-encode-oracle-R1-v3` has a
bit-identical initial convolution and final normalized latent relative L2
3.64e-7, cosine 0.9999999999999343 and maximum error 2.04e-6. The full mixed
image/video render remains unqualified. Its 17-frame video CNN tile exposed
the 512 MiB convolution-workspace cap inherited from image-only qualification;
the bounded temporal allocation is being tested separately before full replay.

Native generation timing explicitly includes the complete `h3_generate` call:
lazy model loading, conditioning, denoising, decoding and media delivery.
SGLang loads its offloaded weights before entering its reported generation
region. Native loading is not subtracted to manufacture a latency pass. Both
fresh-process time to playable media and total command wall time remain
separate measurements; this pipeline difference remains a performance issue.

`teacher-full-C0-v10`, `teacher-full-C1-v10` and `teacher-full-C2-v10` pass
full teacher-forced replay at every evaluation (6/6/50). Each evaluation
receives its independent oracle state, and video/audio velocity and next
state are checked separately with finite/near-zero guards. The retained
free-running C2 trajectory separately proves all 50 accumulated updates agree.
`resume-C0-v10` also passes interrupted schedule replay and serialized
conditioning/AdaLN reuse against the independently completed clean AV payload.

`resume-C1-v10` and `resume-C2-v10` pass as well, including C2 paused after
evaluation 25 and resumed through evaluation 50. All clean video/audio
payloads equal the independent uninterrupted baselines. The optional build
without cuDNN compiles, passes the host sampler suite, and rejects reference
rendering before any denoising with its explicit dependency error.

The isolated temporal replay `video-encoder-native-v11` completes with the
2 GiB workspace cap, 6,735,811,472 bytes of peak device tensors and 32 MiB
of pinned staging. Both channel-layout scratch buffers remain capped at
1 GiB; no process safety limit was raised. Full R1 input/media comparisons
are still required. The worker IPC now validates its own protocol and
reference arithmetic identity so an older executable cannot silently supply
conditioning to a newer parent.

### Complete per-evaluation CUDA profiles

`profile-full-C0-v12`, `profile-full-C1-v12` and `profile-full-C2-v12` complete
all 6/6/50 evaluations and preserve the teacher-run MP4 bytes exactly.
`profile-analysis-v12.json` retains every step's event times, transfers,
allocation counters, process residency and swap. These instrumented times
are diagnostic; the fresh-process paired campaign supplies qualification.

| Case | Profiled denoising total | Live device tensors, every step | Pinned bytes, every step | Swap |
| --- | ---: | ---: | ---: | ---: |
| C0 | 26.437 s | 3,648,624,864 | 38,568,722,432 | 0 |
| C1 | 113.195 s | 7,411,127,136 | 38,568,722,432 | 0 |
| C2 | 225.942 s | 4,506,136,464 | 38,568,722,432 | 0 |

C2 process residency remains exactly 39,549,861,888 bytes across all 50
step boundaries. Increasing cumulative tensor-allocation counts include
released views and do not indicate live-memory growth. CUDA driver/library
overhead is outside these native tensor counters and remains covered by the
separate total-GPU collector.

The short-case profile spends about 2.8 s/step in GEMMs, 1.18 s in attention
and 0.20 s in elementwise operations. Roughly 2.8–3.1 s of H2D transfers
overlap compute, leaving only 0.003–0.068 s of exposed upload wait in the
shown steady steps. Adding those category times as if serial would be wrong.
The remaining complete-generation issue is model/component readiness and
lazy staging; fresh C0 generation includes 104.576 s in this profile.

The final R1 input diagnostic exposed two policy differences beyond CNN
arithmetic. SGLang's reference-video `adapt_shape_v1` always starts from a
768-pixel short edge, so the one-second 640×480 source becomes 1024×768.
The ordinary h3cli reference-video helper deliberately avoids upscaling. Only
the new reference policy now uses the oracle's adaptive geometry, including
its aspect-ratio bounds and soft area cap. FFmpeg's existing single-pass
cadence/Lanczos conversion already matches the pinned video recipe.

The pinned audio-reference timestep is **1.0**, while visual conditioning
uses **0.999**. Applying the latter to audio was an incorrect assumption.
Native reference audio now retains the clean encoded rows without consuming
a random draw. The CPU comparison test reads the pinned default rather than
testing an assumed augmentation strength. Ordinary fast/legacy augmentation
is unchanged.

`native-inputs-R1-v14` passes: tokens, Qwen embeddings, sigma arrays, positions
and all video condition rows are bit-identical. Audio condition relative L2
is 5.45e-6, cosine 0.9999999999851628 and maximum error 3.64e-5, within the
frozen operation gates. The complete R1 trajectory/media comparison is next;
input agreement alone does not establish its full-render qualification.

The first complete R1 oracle attempt (`oracle-full-R1-v1`) stopped before
denoising when broad Qwen/refiner snapshots reached the explicit 8 GiB
capture limit. That failure is retained. The retry selects trajectory-only
instrumentation, preserving the cap and the already-qualified input fixtures
instead of collecting redundant large intermediate tensors.

The local Metal build and bounded compatibility fixtures pass: 129 sampler
GPU checks (including all 65,536 BF16 bit patterns and 51 AdaLN tensors),
129 linked GPU API symbols, and full-VAE policy/unpack checks. The first
sandboxed GPU invocation could not initialize Metal; the same fixtures pass
with normal GPU access. This is compatibility evidence, not Metal reference
rendering qualification.

`fast-source-audit-v14.json` records unchanged pre/post identities for the
fast execution policy, shared scalar CUDA kernels, attention selection,
quantization/cache implementation, Sage kernels, memory guard and preview
VAE. Guarded shared-file changes and complete fast-render regression remain
separate checks. The report's per-evaluation tables include all 50 C2 memory
and timing records. Local artifact audit v14 passes 2,831 checks before the
final paired campaign assets are added.

### Mixed-reference localization and input codec isolation

`native-full-R1-v14` fails the full media/trajectory gates although its input
errors pass the operation thresholds. First video/audio velocity relative L2
is 0.0300/0.0258. The first-block diagnostic finds bit-identical text,
refined text, video conditioning, generated video/audio inputs, token tags
and actual timesteps. Only projected reference audio differs. Five encoded
reference-audio values cross BF16 rounding boundaries. Importing the oracle
conditioning in `replay-condition-R1-v17` restores bit-identical video/audio
velocities and states at all six evaluations. This is localization evidence,
not a native-conditioning pass.

The cause is the input/output FFmpeg split. SGLang's reference soundtrack
loader invokes system FFmpeg, while final delivery uses imageio's bundled
FFmpeg 4.2.2. Our delivery override also selected the older binary for input
AAC decoding. The resulting PCM difference is small (relative L2 1.11e-7,
maximum 5.96e-8), but propagates through the audio encoder and BF16 rounding.
The system-decoded/resampled PCM is bit-identical to the pinned oracle.

Reference soundtrack decoding now defaults to PATH FFmpeg and optionally
uses `H3_SGLANG_INPUT_FFMPEG`; legacy/fast `H3_FFMPEG` semantics are unchanged.
`preprocessing-v18` checks five image resizes and the complete soundtrack,
including a second audio decode with the delivery override set. All checks
pass exactly. The corrected native encoder replay returns to relative L2
3.64e-7 and maximum error 2.04e-6. A fresh, ordinary native R1 render is
required before calling this final-conditioning qualified.

Reference identity 3 rejects earlier identity 1/2 conditioning and sampler
reuse. The new input-runtime override participates in arithmetic environment
identity; no model weight hashing is introduced. The current host sampler
suite also explicitly rejects both old identities. The frozen, trace-free
primary campaign uses the retained identity-2 binary: its unconditioned
arithmetic is unchanged by the soundtrack correction. Its source/binary
identities must remain visible rather than being replaced during measurement.

### Lifetime, cancellation and repeated sessions

`reference-lifetime-v15.log` passes full original-VAE materialized versus
streamed equality, repeated allocation plateaus, mid-tile cancellation,
nonfinite input rejection, memory-admission and sink failures, and clean
retry. An earlier invocation used an incorrect model directory; that failed
attempt is retained and does not count as a test pass.

`reference-session-v15.log` passes five cached requests alternating
64×64/22-frame and 96×64/39-frame geometries, then cancellation and recovery.
Clean AV payloads repeat exactly for each geometry; conditioning, prepared
DiT and decoder caches are bounded and cleared after failure. Interleaved
fast GPU fixtures remain unchanged. DiT live device allocations plateau at
1,673,474,640 and 1,689,838,624 bytes for those geometries; pinned staging is
38,568,722,432 bytes, process residency is approximately 40.0 GB and process
swap is zero. These small-geometry lifetime checks supplement, rather than
replace, the full C2 50-evaluation memory history.

`native-full-R1-v18` still fails: first video/audio velocity relative L2 is
0.03193/0.02778 and all final media gates fail. This corrects the earlier
hypothesis that the output FFmpeg override fully explained R1. The corrected
standalone encoder's BF16 values are identical to the oracle, so a narrow
capture now compares PCM and encoded rows inside the complete native request.
The primary campaign continues on its frozen binary; diagnostic GPU work is
inserted only between complete measured processes.

The narrow `audio-scope-R1-v19` capture confirms that full-request PCM equals
the oracle and that encoded FP32 rows equal the standalone native probe.
Thus there is no separate in-request encoder drift. Merely comparing a BF16
cast of those rows was insufficient: H3's patch projection consumes FP32
operands and only casts its output. The first encoder-stage discrepancy is
Snake1d: the pinned unfused oracle computes a reciprocal followed by square,
multiply and add, while the old helper divides the squared sine. A separate
reference-encoder kernel now preserves those operation boundaries; the shared
fast kernel remains unchanged. Operation replay is pending before another
full mixed-reference render.

The first tracing-free matched C0 and C1 pairs both pass all decoded-video
and audio gates. Their denoising times are within the frozen 5% band. These
single pairs are not the required medians:

| Case | Engine | Process wall | Generation | Denoising | Full AV decode | Sampled total GPU peak | Largest sampling gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | SGLang | 233.446 s | 41.197 s | 30.191 s | 6.459 s | 16.434 GiB | 9.130 s |
| C0 | native | 111.621 s | 110.989 s | 28.661 s | 7.695 s | 6.217 GiB | 0.059 s |
| C1 | SGLang | 389.979 s | 133.868 s | 112.166 s | 15.996 s | 19.387 GiB | 9.248 s |
| C1 | native | 228.706 s | 228.047 s | 114.867 s | 22.433 s | 10.748 GiB | 0.057 s |

Native lazy-weight staging remains inside its generation value. Full decoding
is slower and the oracle's sampling gaps exceed 100 ms, so these measurements
do not pass the joint speed/VRAM contract despite lower native process wall
and sampled memory. No missed memory samples are interpolated or counted as
fresh observations. Local artifact audit v20 passes 4,349 checks with the
first matched pairs and all retained failed R1 attempts included.

`audio-encode-native-R1-v20` proves that the reciprocal-order correction makes
the initial convolution, all five downsampling stages and final convolution
bit-identical to the oracle. The remaining first divergence is LayerNorm in
the pre-attention projection. This encoder was still using the old generic
FP32 norm rather than the existing pinned vectorized Welford implementation.
`audio-encode-native-R1-v21` then proves that both normalization and its linear
projection are exact after selecting that implementation for the reference
encoder. Attention/output branches remain under operation-level comparison.
The vectorized reduction follows the
[pinned PyTorch LayerNorm source](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cuda/layer_norm_kernel.cu).

The next reference-encoder candidate preserves the oracle's biased QKV
projection, separately scaled FP32 math-SDPA, causal softmax, head mean followed
by adaptive channel pooling, and tanh-GeGLU rounding. Attention scratch is
bounded to 64 MiB per buffer, with the checked stereo/eight-head/256-dimension,
at-most-600-audio-row geometry. This materialized small-audio attention is
never selected for video DiT or fast CUDA. Extended hooks capture its real
queries/keys/values, attended output and pooled projection input for direct
comparison; neither the installed SGLang source nor its weights are edited.

### Final reference soundtrack arithmetic (identity 4)

`audio-encode-native-R1-v25/operations.json` passes all 22 retained checks
bit-for-bit, from the initial convolution through normalized FP32 output rows.
The extended oracle hooks preserve the earlier oracle rows exactly. This is
operation-level evidence; the complete R1 trajectory/media result is recorded
separately. The reusable comparison is
`tests/cuda_sglang_audio_encode_compare.py`, which validates oracle checksums,
shape/dtype/size bounds, native checksums and the frozen operation gates.

The reference-only corrections are reciprocal-order Snake, pinned Welford
LayerNorm, biased QKV projection, separately scaled FP32 math SDPA, direct
softmax division, four-accumulator head averaging, separate adaptive channel
pooling and tanh-GeGLU rounding. FP32 audio attention is restricted to the
stereo/eight-head/256-channel encoder and bounded to 600 rows and 64 MiB per
scratch buffer. It is not selected for DiT or fast attention. The math follows
the pinned PyTorch
[persistent softmax](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cuda/PersistentSoftmax.cuh)
and [four-accumulator reduction](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cuda/Reduce.cuh).
No model weight hashing or oracle source modification was introduced.

Identity 4 rejects exploratory identity 1–3 sampler/conditioning reuse and
requires a matching encoder worker. The local final host suite passes 1,775
sampler checks, 19 container tests, 208 adversarial cases and 41 CLI cases.
The protected fast identity remains 1. Timed unconditioned comparisons retain
their frozen identity-2 source/binary; soundtrack-only corrections do not alter
their arithmetic and the report preserves that distinction.

`native-full-R1-v25` now passes the complete native-conditioning render:
all six video/audio velocities and next states are bit-identical to the oracle,
and every-frame PSNR/SSIM/LPIPS, temporal, worst-region and audio gates pass.
No oracle inputs are imported. This closes the mixed-conditioning numerical
failure; earlier failed clips remain available for inspection. The frozen
primary timing campaign resumed after the GPU render completed; CPU media
comparison ran independently.

The reference scalar generator now reproduces both generated CUDA headers
byte-for-byte (`generator-final-v25.json`). Its shared kernel output is unchanged
from the protected baseline. `fast-source-audit-final-v25.json` verifies the
original execution policy, attention selection, quantization/cache, Sage,
memory guard and preview sources remain byte-identical. The measured-to-final
source delta is retained separately: it contains guarded soundtrack corrections,
arithmetic/cache identity changes and inactive test-only capture hooks. These
source audits supplement runtime fast regression; they do not replace it.

The final native snapshot is `/path/to/qualification/final-reference-v4`.
`build-identity-final-v25.json` verifies all production C/C++/CUDA sources against
the compiled candidate before copying the executable and encoder worker. The
compressed source snapshot contains only repository files, not model weights,
downloaded oracle assets or generated outputs. The normal server installation
and the frozen measured candidate remain intact.

`initial-rng-final-v25` replays the original renderer's production PCG helper
from the retained pre-change source and from the final source. The initial
video/audio FP32 arrays and resulting RNG states are byte-identical at
640×480/124 and /362 frames, seed 42. C0/C2 share the initial 124-frame geometry;
their schedules differ afterward. This bounded CPU-only fixture supplements
the full before/after fast renders; it is not presented as a captured DiT call.
The original arrays remain on the server with retained checksums and JSON RNG
states; no model weights or GPU operations are involved.

T019's scoped copy/fusion work is implemented and numerically qualified:
FlashAttention consumes the native packed strides directly, reference norm/RoPE
preserves the oracle rounding boundaries, bounded pinned weight staging removes
repeated pageable reads, and reference GPU unpack/stitch avoids the old CPU tile
round trips. Boundary, full-trajectory, cancellation and lifetime checks pass.
This does not close the separate complete-generation/full-decoder speed or
startup-memory sampling gates; those tasks remain unchecked.

## Final primary measurements and accepted fast differences

All nine tracing-free matched pairs completed, including three complete
50-evaluation C2 pairs, and every pair passes the frozen decoded-media gates.
The final medians and ranges are in `matched-campaign-v13/result.json` and
[the results document](cuda-sglang-results.md). Denoising ratios are
0.87845 / 1.02151 / 1.02229 for C0/C1/C2: all pass the 5% tolerance.
Native process wall time is lower, but reported generation (with native lazy
loading retained) and AV decoding fail. Native sampled peaks are lower;
SG memory queries leave gaps up to 9.284 seconds and native C1 has one
462 ms gap. Thus the required peak-memory evidence is incomplete. The report
leaves gaps blank in its memory plots instead of connecting missing samples.

The before/after fast BF16 full-decoder outputs differ consistently across
repeats; decoded RGB and PCM differ too (`fast-stream-diagnostic-v26.json`).
The NVFP4 preview outputs also differ. No unchanged-full-render claim follows
from the unchanged source files or fixed-tuning fixture. The user subsequently
instructed, “ignore if the behavior of fast mode has changed.” Investigation of
that discrepancy stopped; the exception is explicit in
`review/qualification-waivers.json`, separate from the unchanged frozen contract.
Reference content, speed and memory gates retain their original limits.

The first fast NVFP4 preview pair failed immediately because both isolated
source directories lacked the default tiny decoder. The existing local
22,709,752-byte `taeh3.safetensors` was copied to one shared test-fixture directory
and linked from both builds. No weight hash was added. The original failed
records remain, and only this setup-failed pair is repeated after the rest of
the campaign. Reused successful results retain their original commands and paths.

## Closing verification

All queued jobs have finished and the RTX PRO 5000 is idle (0% utilization,
22 MiB total device use, no compute processes). The retained inventory contains
133 render/input-diagnostic records, including failed attempts; none remains
`running`. All complete playback assets and test records are local.
`artifact-audit-final-v26.json` passes **9,885** checks for report/source checksums,
local links, exact frame counts and complete media decoding.

The final mixed-context GPU fixture passes with the original baseline checksum.
The clean optional-dependency build without cuDNN passes and rejects reference
rendering before loading models or creating output. Source snapshot v26 verifies
748 files and changes only test/report helpers plus README since the measured
arithmetic-4 build; executable and worker bytes are unchanged. Host compatibility,
local Metal GPU compatibility, generated-kernel consistency, lifetime/cancellation,
all-50-step allocation plateau and initial-PCG replay records remain retained.

The repaired fast comparison contains all required three timing pairs and one
captured pair per variant. Both timing and sampled-memory medians pass their
5% bands, but the complete BF16 sampler and both final-media comparisons differ.
NVFP4's captured CPU-sampler arrays match. These differences remain visible under
the explicit user waiver; they are not relabeled as unchanged behavior.

**32 of 40 checklist tasks are closed.** T020, T024, T025, T026, T028, T033,
T034 and T040 remain open because the required reference generation/full-decoder
speed and complete memory observation have not qualified. T033 additionally
retains the unmatched model-readiness boundaries. T040's prescribed partial
report is delivered without changing the original limits. No full joint
content/performance/memory qualification is claimed.

## User acceptance and default selection — 2026-09-25

The user instructed: “accept the results, close all tasks, and make the
sglang-equivalent mode the default.” This supersedes the preceding open-task
status: **40 of 40 tasks are closed**. The frozen contract, measured failures,
unequal readiness boundaries, incomplete peak-memory observations and fast
behavior differences remain retained. The separate acceptance record covers
speed, memory and fast gates; content and corpus coverage still pass directly.

Ordinary BF16 CUDA video now resolves to SGLang arithmetic identity 4 before
model/cache construction. `--cuda-reference legacy` explicitly opts out;
`auto` restores the default, and explicit `sglang` rejects incompatible options.
Existing accelerated selections and Metal/still paths keep their dispatch.
Resume preserves the recorded policy and existing build/environment checks;
decode-only restores reference delivery from its presentation metadata.
CUDA builds include the reference FlashAttention/cuDNN dependencies by default.

Verification of this selection change is separate from the frozen campaign:

- Local Metal build and library pass, with 33 host policy checks, 1,775 sampler
  checks, 19 container tests, 208 adversarial cases and 52 CLI checks. The CUDA
  build passes the same 33 policy, 1,775 sampler and 52 CLI checks.
- One plain CUDA command (no reference flag), 640×480, 124 frames, six evaluations,
  reproduces the accepted complete AV latents exactly. All decoded RGB frames
  and decoded audio are byte-identical to the accepted C0 media.
- Decode-only without a reference flag reproduces that same media. Explicit
  legacy delivery on a reference state and incompatible explicit fast/preview
  combinations are rejected. Initial negative checks hit the test-evaluation
  guard or missing tiny-model fixture; only those short checks were retried,
  with the original records retained. No rendering was repeated.
- All three acceptance reports pass the 9,885 retained artifact checks. The
  final server source matches all 749 manifest entries. Rebuilding with
  `CUDA_SGLANG` and `CUDA_CUDNN` unset confirms the new Makefile defaults and
  preserves the tested executable/worker bytes. The GPU returned to idle.

See [the validation closeout](../../outputs/cuda-sglang/records/default-validation-v4.json),
[commands and exact media comparisons](../../outputs/cuda-sglang/records/default-verification-v4/result.json),
[default-render playback](../../outputs/cuda-sglang/records/default-verification-v4/default.mp4)
and [build identities](../../outputs/cuda-sglang/records/default-build-identity-v4.json).
The default build is `/path/to/qualification/default-reference-v4`; source its
`reference-env.sh` for the pinned runtime dependencies. Historical qualification
builds and the normal server installation remain separate.

## Large reference-layout patch projection

A 1344×768 / 362-frame request with a max-size reference image exposed a hard
limit in `sglang_patch`: the entire FP32 result had to fit within 1 GiB.
The generated-video rows alone are 42 × 24 × 107 = 107,856; multiplying by
5,376 channels and four bytes requires 2,319,335,424 bytes (2.16 GiB), before
adding any reference rows. The combined dtype/scratch error misidentified this
as potentially invalid operands. This was a per-operation temporary-buffer
limit, not the global process memory guard.

The projection now batches independent rows into the same bounded FP32 buffer,
casts/scatters each batch on the same CUDA stream, and reuses the allocation.
Batch boundaries preserve pointer alignment. The existing pinned cuBLAS
heuristic is queried with the full row geometry; the K reduction and FP32 bias
epilogue are unchanged. Mapped destinations use the corresponding slice of the
original row map; contiguous destinations retain input/output offsets. Calls
already below the cap keep their original geometry and algorithm selection.
Arithmetic identity 4 is unchanged; the newly supported large calls previously
returned an error.

`make test-cuda-sglang-patch` runs model-free projection checks. On the RTX PRO
5000, all six cases passed: short video/audio, the last row count below the old
limit (49,932), the first above it (49,933), 107,856 generated rows, and 113,296
rows including a representative 2048×2720 reference image. Every BF16 element
matches the unbatched FP32 reference addmm followed by BF16 rounding. Tests also
check reversed mappings, untouched gaps, nonzero offsets, invalid bias and
stable allocation across repeated calls. Large-case scratch is 1,073,479,680
bytes, below 1 GiB. No denoising or full video render was run.

The fixed executable/worker were built in
`/path/to/qualification/patch-batching-v1`, retaining its `reference-env.sh`.
See [the test records](../../outputs/cuda-sglang/patch-batching/result.json).
