# H100 qualification

Historical record: renderer modes and legacy test launchers described below have been
retired. Retained artifacts preserve the original results. Use the
[single-pipeline design](design-single-pipeline.md) and
[current qualification](single-pipeline-final.md) for supported commands.

The supplied server has an NVIDIA H100 80GB HBM3, SM90, 132 multiprocessors,
81,559 MiB reported by `nvidia-smi` and 79.2 GiB visible to the CUDA runtime.
It has a 700 W power limit and MIG is disabled. The host runs Ubuntu 24.04.3,
driver 580.126.09, CUDA toolkit/runtime 12.8, cuDNN 9.10.2 and
cudnn-frontend 1.11.0. The container has a 250,999,996,416-byte memory limit
and no swap. Tests use the original BF16 FL2VA and Ref2VA models.

Source, dependencies, temporary files, CUDA caches and results are under
`/workspace`, which is local storage on this instance. Models remain in
`/path/to/models/MiniMax-H3`. The source checkout is
`/path/to/h3.c-h100`; the environment and reproducible build wrapper are
`/path/to/h3-h100/env.sh` and `/path/to/h3-h100/make.sh`.
Local evidence is under `outputs/cuda-validation/h100`.
An under-load snapshot records PCIe 5.0 ×16, 74% GPU utilization and 468 W;
it is a link check, not an average utilization or power measurement.

## Generic SM90 baseline

The generic CUDA build was tested before changing architecture dispatch.
The initial matched `inputs/1.jpg` comparison used 288×384, 56 frames,
20 steps, seed 1001, all 50 layers, reuse/core-reuse one and token reduction
off. Default CUDA took **97.78 seconds**, and existing fast CUDA took
**41.89 seconds**, including loading, inference, MP4 generation and media
validation. This is about 2.33× for this short case, not a prediction for
production-length renders. The full transformer core fits resident on this
H100; the partial streaming cache used on the 3090/5090 is unnecessary at
this shape.

The baseline binary is preserved on the server as
`outputs/cuda-validation/h100/h3-baseline`, with its SHA-256 in
`baseline-binary.sha256`. Each render record also identifies its executable,
command and settings.

## Hopper attention tuning

The existing SM120 specialization was evaluated on H100 before promotion.
It preserved exact BF16 output but regressed the 18,225-token, 56-head case:
about 1,067 ms versus 982 ms for portable CUDA. Copying Blackwell's dispatch
choice to Hopper would therefore have made this workload slower.

Thread-count experiments retained the same QK, softmax and F32 PV arithmetic
order. The 64-thread candidate regressed; 256 threads improved the large case
to about 819 ms; 512 threads improved it to 782–787 ms. The resulting SM90
dispatch uses the fixed-128-head kernel with 512 threads. Its smaller
per-thread PV accumulator contains four values instead of SM120's sixteen.
SM120 continues to use 128 threads. Both specializations share one template;
a CUDA 12.8 SM120 code-generation comparison verifies that the refactor emits
identical machine instructions for the existing specialization.

The final direct-kernel comparison uses one warmup and three alternating timed
iterations, reporting the median. Both layouts pass byte-for-byte comparison
with the unchanged portable tiled kernel at seven shapes, including tail
tiles, multiple batches and full DiT head counts. Ordinary-layout examples:

| Tokens / heads / batches | Portable | SM90 tuned | Speedup |
| --- | ---: | ---: | ---: |
| 513 / 3 / 2 | 0.334 ms | 0.122 ms | 2.74× |
| 2,048 / 8 / 1 | 2.208 ms | 1.739 ms | 1.27× |
| 2,281 / 56 / 1 | 16.697 ms | 15.317 ms | 1.09× |
| 18,225 / 56 / 1 | 981.637 ms | 785.263 ms | 1.25× |

Production dispatch remains limited to SM90, noncausal BF16 attention,
128-wide heads and at least 512 tokens. Reference mode and other unsupported
shapes retain their prior paths. The short direct test also covers 19 tokens
to check masking; production does not dispatch that shape to this kernel.
The matched 20-step default `image05.h3av` and final MP4 are byte-identical
before and after the tuning, including saved video/audio latents and encoded
audio/video.

The final inference binary SHA-256 is
`4dd6d7523454b1d09f1cc63c86ab9bb23e2e77b6e305b817ca77bf468286f2bd`.
The original baseline SHA-256 is
`eef7b6a5d08e9d20c75f109a9446f6cbc5ca78767b392ea6472c20aa242b1d41`.

## Fast CUDA on H100

The existing cuDNN-first policy remains the best measured fast attention
choice. With BF16, 56 heads, dimension 128 and ordinary layout:

| Tokens | Generic default | Native fast | cuDNN fast |
| ---: | ---: | ---: | ---: |
| 2,281 | 16.807 ms | 1.435 ms | 0.277 ms |
| 18,225 | 982.505 ms | 78.524 ms | 14.347 ms |

These isolated measurements precede the default SM90 tuning; the tuned
default measurements are in the preceding table. Fast attention does not
use the default specialization when cuDNN or native flash attention succeeds.
Separate production-scale kernel checks pass at 32,768 and 110,592 tokens:
native/cuDNN timings are 250.90/44.85 ms and 2,848.61/544.79 ms respectively.
They do not establish long-render wall time or long-form video quality.

Native shared-memory padding 128/136/144, pipelining on/off and four tile
shapes were measured on SM90. Padding 136 with pipelining remains appropriate.
A 64×32 tile experiment reached about 70 ms at 18,225 tokens, still far behind
cuDNN's 14.35 ms. Generated tile experiments also change Q staging, so their
timings are not a clean tile-size-only comparison. No fast tile
or CUDA graph replay path was promoted. The tuning scripts now detect the
local architecture, accept `--arch 90` explicitly and record compiler flags;
they previously hard-coded SM120.

## Matched renders

All six pairs use the final binary, 20 steps, all 50 layers, reuse/core-reuse
one, token reduction off and original BF16 weights. Times are the shared
ledger's process-plus-media-validation wall times (validation adds roughly
0.2 seconds); each JSON also records process-only wall time.

| Case | Resolution / target frames | Tuned default | Fast CUDA | Speedup |
| --- | --- | ---: | ---: | ---: |
| `1.jpg` | 288×384 / 56 | 96.22 s | 38.62 s | 2.49× |
| `2.jpg`, second seed | 288×384 / 56 | 95.93 s | 39.25 s | 2.44× |
| `2.jpg` + `body1.jpg` | 288×384 / 56 | 93.36 s | 37.32 s | 2.50× |
| Detail | 480×640 / 22 | 96.18 s | 35.85 s | 2.68× |
| Continuation segment 1 | 288×384 / 90 | 120.24 s | 40.52 s | 2.97× |
| Continuation segment 2 | 288×384 / 90 | 118.59 s | 40.33 s | 2.94× |

Segment two consumes a 39-frame overlap and delivers 51 new frames. All
outputs pass full FFmpeg decoding and checks of dimensions, frame counts,
frame rate and 32-kHz stereo audio. The gallery is
`outputs/cuda-validation/h100/quality/review.html`.

For the final `1.jpg` pair, the cumulative DiT load/denoise profile falls
from 32.00 to 14.25 seconds; VideoVAE decoding falls from 13.06 to 4.87 seconds.
Peak DiT device allocation is 36.69 GiB in both modes, with zero streamed
DiT weight bytes. Qwen's profile remains about 8–9 seconds, mostly uploads.
The earlier fast pilot took 41.89 seconds; the final one took 38.62 seconds.
These are separate short invocations with ordinary cache/timing variation,
not a statistical benchmark or an additional fast-kernel speedup claim.

A final matched upload-policy check takes 41.31 seconds with normal uploads
and 41.46 seconds with `H3_CUDA_REGISTER_WEIGHTS=1`. Registration offers no
useful gain in this pair, so normal bounded uploads remain the recommendation.
These additional timing clips are outside the six-pair quality-review scope.

The numerical default optimization improves the short complete render only
slightly (97.78 → 96.22 seconds). Its meaningful 1.25× gain is demonstrated
at the larger isolated attention shape; loading, compatibility work and
decoding limit this small render's benefit. The larger full-render gains in
the table come from the existing opt-in fast workflow.

## Validation bounds

Renders share a 2,700-second cumulative ledger and a 300-second cap per
invocation. Component and isolated kernel cases have separate time limits.
The completed ledger contains **41 invocations totaling 2,512.06 seconds
(41.87 minutes)**, including pilots, checkpoint/caching tests, functional
smokes and upload-policy comparisons. All recorded render invocations succeed.
No `test2.sh` or 362-frame/50-step production render is used. Twenty-step
clips are quality review candidates; two-step clips are functional evidence
only. Successful tensor comparisons and media decoding do not constitute
human playback/listening acceptance. Prior acceptance of other GPUs' videos
does not cover these H100 outputs.

The six-pair human review is currently pending. Its exact MP4 and metadata
hashes are captured in `outputs/cuda-validation/h100/quality-acceptance.json`.
Sampled frame sheets show coherent subjects/scenes without gross spatial
corruption in the inspected images; the observations are recorded separately
in `sampled-frame-review.json` and do not establish motion/audio acceptance.

Both the generic and tuned builds pass the host/CLI/state/tokenizer suites,
all 41 default primitive comparisons and all 23 released-model comparisons
across vision, text, video encoding, audio encoding/decoding, video decoding,
one DiT block and a full small DiT evaluation. The final build also passes
fast attention, decoder, QKV, reference-override, zero-scratch fallback and
asynchronous streaming/cache regressions. The SM120-specific execution test
correctly skips this H100; the separate SM120 instruction comparison is a
compiler check, not a new Blackwell hardware run.

At both 25 and 50 layers, memory tests pass explicit resident execution,
forced streaming, automatic residency and injected allocation failure with
streaming fallback. Default results are byte-identical across those memory
modes; fast results are checked for finite values. Unlike 24/32-GB cards,
this H100 actually executes the full 50-layer resident case successfully.
Those small component shapes exercise residency policy, not the maximum
possible render geometry.

The generic and final builds pass default/fast fresh-process checkpoint
restart and explicit fast-to-default handoff. Final default restart also
uses 288×384, 56 frames and 20 steps: checkpoint capture stops at step three,
and every resumed latent boundary matches its uninterrupted control exactly.
The record invocation takes 104.29 seconds and includes both the uninterrupted
control and separate checkpoint capture; restart takes 81.06 seconds. These
are functional test timings, not standalone render benchmarks.

Default and fast continuation audits use changed face/body references,
rectangular 288×384 states, two steps, a 90-frame target and 39-frame overlap.
They verify initial noise/prefix construction, protected-prefix preservation
at every boundary, source-state immutability, 51 new frames, audio trimming
and AV-state roundtrip. A separate fast continuation checkpoint also passes
fresh-process restart. Fast tests check functional invariants and finite
outputs without imposing the default numerical-equivalence gate.

Fast/default/fast mode switching, prepared/decoder reuse, reuse/core-reuse,
token reduction pass. The pressure fixture leaves
only 4 GiB free, evicts prepared conditioning, retains the decoder and
successfully renders and saves a checkpoint. It forces streaming and disables
the optional weight cache to bound its allocation; the runtime suite tests
the optional cache's own pressure eviction.

Compute Sanitizer reports zero shared-memory race hazards for the new SM90
kernel and the native fast kernel. Runtime/stream-cache memcheck reports zero
errors and zero leaked allocations. The build without cuDNN passes fast
dispatch and runtime/cache tests. Fat compilation covers SM86/89/90/120, and
the fat binary passes all 41 primitive comparisons on H100.

Global `CUDA_FORCE_PTX_JIT=1` cannot initialize cuBLASLt in this installed
toolkit/library configuration (`fat-ptx.log`); it is not counted as a pass.
The replacement check compiles the project's CUDA object with **compute_86
PTX only**, verifies that it contains no device SASS, and links normally to
the vendor libraries. All 41 primitive comparisons pass on H100 using that
object. `ptx-build-commands.json` and `ptx-inventory.txt` preserve the exact
build and object inventory. This establishes the project's forward PTX path
without claiming that every installed vendor library supports forced PTX.

Additional two-step end-to-end smokes pass for the build without cuDNN,
FL2VA first/last-frame conditioning, and Ref2VA video with independently
replaced audio. These clips are functional evidence only. LoRA/Turbo renders
were not repeated because this server has no supplied folded LoRA model.
All 38 saved AV states pass independent checksum, geometry and finite-value
inspection. The three default restart pairs (generic compact, tuned compact
and tuned twenty-step) have byte-identical saved states.

The local Mac build and `make test` pass. Optional released-weight Metal
fixtures and the ignored legacy `misc/fixtures` parity suite are absent;
their skips are not counted as successful comparisons. No shared model,
sampler, file-format or Metal implementation changed in this round.

## Running on this server

```sh
source /path/to/h3-h100/env.sh
cd /path/to/h3.c-h100
/path/to/h3-h100/make.sh -j8 all
./bin/h3cli -d /path/to/models/MiniMax-H3 --fast-cuda \
  --ref-image inputs/1.jpg --width 288 --height 384 --frames 56 \
  --steps 20 --layers 50 --reuse 1 --core-reuse 1 --seed 1001 \
  -p 'The woman in <Picture 1> walks through a sunlit forest.' \
  --profile -o outputs/preview.mp4
```

Leave weight mode on `auto`. The full transformer is resident for these
shapes; larger allocations can still select streaming when required. This
build includes optional cuDNN and OpenSSL acceleration. Model weights are
neither quantized nor modified.

To reproduce the numerical attention check and bounded experiments:

```sh
H3_TEST_FIXED128_BENCH=1 /path/to/h3-h100/make.sh cuda-sm90-test
python3 tests/cuda_fast_bench.py --out outputs/h100-kernels
python3 tests/cuda_fast_tune.py --arch 90 --out outputs/h100-staging
python3 tests/cuda_fast_tiles.py --arch 90 --out outputs/h100-tiles
```
