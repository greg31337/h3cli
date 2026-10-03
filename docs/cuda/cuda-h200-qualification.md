> Historical renderer/qualification record. Commands that refer to removed
> fast/legacy modes require the archived source. Use the
> [current single-pipeline design](design-single-pipeline.md) and
> [M6/M7 qualification](single-pipeline-final.md) for supported execution.

# H200 qualification

The supplied x86-64 node runs the existing SM90 implementation successfully.
The six matched 20-step comparisons take **35–43 seconds with `--fast-cuda`**
and **99–127 seconds with default CUDA**, a **2.6–3.2×** improvement for these
bounded cases. This is the benefit of the existing fast workflow, not a new
H200-specific optimization or a prediction for a long production render.
The user accepted H200 video and audio quality on 2026-09-18: "H200 videos
look and sound good". This closes the quality gate for the six matched pairs;
see [the acceptance record](cuda-fast-acceptance.json). Individual criterion
scores and playback speed were not separately supplied.

Evidence is under `outputs/cuda-validation/h200`; the six-pair gallery is
`quality/review.html`. Commands, source/input/binary hashes, effective routes,
process timing and media checks are retained with each render. The acceptance
manifest freezes the exact MP4 and metadata hashes. Two-step functional clips
are separate from quality evidence.

## Node and dependencies

The GPU is an NVIDIA H200, SM90, with 132 SMs, 143,771 MiB reported by
`nvidia-smi` and 139.8 GiB visible to CUDA. MIG is disabled; the power limit is
700 W. Under-load samples report 1,980 MHz SM and 3,201 MHz memory clocks.
These are snapshots, not average utilization or power measurements.

The host runs Ubuntu 24.04.3, driver 570.124.06, CUDA 12.8.93, GCC 13.3 and
FFmpeg 6.1.1. Its Xeon Platinum 8568Y+ exposes 192 logical CPUs; the container
memory limit is 250,999,996,416 bytes with no swap. `/workspace` is on the local
overlay filesystem, not a network mount. Source, dependencies, temporary files
and caches live in `/path/to/h3.c-h200`. Original BF16 FL2VA/Ref2VA models are
in `/path/to/models/MiniMax-H3`.

The qualification uses cuDNN **9.10.2.21** and cudnn-frontend **1.11.0**. This
provider holds its installed cuBLAS and cuDNN packages. The driver and holds
were preserved. Pinned cuDNN debs were downloaded and extracted under
`outputs/setup/cudnn-9.10.2/usr`, and that prefix was used for headers, linking
and runtime libraries. `outputs/cuda-validation/h200/prepare-cudnn.sh` records
the commands; `env.sh` records the build environment.

The bootstrap initially exposed two provider-image problems: installing a
second CUDA apt source conflicted with the provider's signing configuration,
and installing the toolkit meta-package tried to replace held cuBLAS packages.
`setup_linux.sh` now reuses an existing working NVIDIA repository and a complete
CUDA 12.8 toolkit. The duplicate source added by the first attempt was archived;
the original provider source remains. Requesting a system cuDNN installation
still respects package holds; this qualification used the isolated prefix.

The baseline checkout is `d567806546ada0946b06fea1dd8ce27991357a3f` plus the
documented build/helper changes. Inference sources are unchanged, with the
workflow source digest
`2fc8a48f75a1453e88de06489519ff17b67a175973adb3c51394e510fadb43b9`.
The cuDNN-enabled render binary SHA-256 is
`ab21844d25793f367ad47d4501b71b2342179cc275d7a76318762ce935a976db`.

## Matched complete renders

All pairs use 20 steps, all 50 layers, reuse/core-reuse one, token reduction
off, original weights and matched prompts/references/seeds. Times below are
**process wall time through MP4 generation**. Separate full-media validation
adds about 0.16–0.17 seconds per invocation and is also charged to the ledger.

| Case | Resolution / target frames | Default | Fast CUDA | Speedup |
| --- | --- | ---: | ---: | ---: |
| `1.jpg` | 288×384 / 56 | 106.24 s | 39.46 s | 2.69× |
| `2.jpg`, second seed | 288×384 / 56 | 98.63 s | 37.76 s | 2.61× |
| `2.jpg` + `body1.jpg` | 288×384 / 56 | 100.83 s | 35.15 s | 2.87× |
| Detail | 480×640 / 22 | 100.73 s | 38.06 s | 2.65× |
| Continuation segment 1 | 288×384 / 90 | 126.86 s | 42.91 s | 2.96× |
| Continuation segment 2 | 288×384 / 90 | 125.15 s | 39.01 s | 3.21× |

The second segment consumes a 39-frame overlap and delivers 51 new frames.
All twelve files pass full FFmpeg decoding, dimensions/frame-count checks and
32-kHz stereo audio checks. Their source AV states remain available for
continuation and decoder testing. Sampled frame sheets show coherent subjects
and scenes without gross spatial corruption in the inspected images; this
does not establish motion, continuation-join or audio quality. Empty cells in
short-clip contact sheets are sheet padding, not black video frames.

An additional matched video-reference pair uses the same quality geometry,
20 steps and an existing coherent default clip as conditioning. It takes
158.30 seconds in default mode and 48.22 seconds in fast mode (3.28×). Both
files pass complete media validation. Its separate gallery is
`video-quality/review.html`; the frozen human-acceptance request covers the
six pairs above. This additional pair is outside that frozen acceptance scope.

The first pair is also the budget pilot. It includes colder host/model caches;
different corpus cases are not repeat measurements. Do not infer H200-versus-
H100 complete-render speed from these runs: the hosts, dependency details and
cache state differ. The previous H100's isolated cuDNN timings are close to
this H200's, consistent with reusing the same SM90 computation path.

An unchanged fast repeat of `1.jpg` completes in 36.31 seconds versus the
pilot's 39.46 seconds, an 8% difference consistent with initialization/cache
and short-run variation. These two observations do not establish a statistical
performance distribution or an optimization gain.

The subsequent registered-upload candidate (`H3_CUDA_REGISTER_WEIGHTS=1`)
exceeds its 60-second cap while still loading transformer layer 26/50. The
process group is terminated, its partial log and failure record are retained,
and the full 60.17-second attempt is charged to the render ledger. It produces
no complete review video and is rejected. Keep that override **unset** on this
node. All selected-path renders use ordinary uploads.

For `1.jpg`, cumulative DiT load/denoise profile time falls from 29.69 to
14.77 seconds; loading is included in those figures (9.07/10.83 seconds), not
an additional phase to add again. Cumulative attention kernel time falls from
15.42 to 1.17 seconds. VideoVAE decoding falls from 12.29 to 4.95 seconds.
Peak DiT device allocation is 36.69 GiB in either mode, with no streamed weights.
Qwen text processing takes 15.35/7.87 seconds, dominated by uploads and host
cache variation; it is not evidence of a newly accelerated H200 text kernel.

## Attention and memory behavior

The existing cuDNN-first policy remains preferable to native fast attention.
Unprofiled one-warmup/three-iteration ordinary-layout measurements use BF16,
56 heads and dimension 128:

| Tokens | Default SM90 | Native fast | cuDNN fast |
| ---: | ---: | ---: | ---: |
| 2,281 | 15.29 ms | 1.42 ms | 0.277 ms |
| 18,225 | 818.26 ms | 82.20 ms | 14.37 ms |
| 18,945 | 890.29 ms | 89.67 ms | 16.17 ms |

Cooperative QKV normalization/RoPE takes 0.117 ms at 2,281 tokens and 0.843 ms
at 18,225 tokens, versus 2.136/16.966 ms for the default operation. Decoder
attention at 2,048 tokens, 32 heads and dimension 64 takes 0.976 ms with the
existing TF32 route, versus 1.953 ms for F32 GEMM, 8.772 ms for the tiled
candidate and 24.255 ms for default attention. These component results support
retaining the existing selections; they are not additional complete-render
speedup factors to multiply together.

The available VRAM permits full transformer residency. At 50 layers the
resident default fixture peaks at 38,574,523,160 tensor bytes; forced streaming
uses 1,597,895,680 bytes and transfers 39,305,871,360 weight bytes. Fast forced
streaming can retain 38,535,168,000 bytes in its bounded weight cache on this
card; total peak tensor bytes reach 40,115,929,880. These fixtures test memory
policy at small shapes, not the largest possible video. Both 25- and 50-layer
resident/stream/auto/injected-allocation-failure cases pass. Default output
retains its exact comparisons; fast output uses validity and safety checks.

Production-length isolated checks pass in both layouts at 33,757 and 110,212
tokens. These derive from 480×640 and 1344×768 at 362 frames, respectively,
using the quality prompt's 151 text rows and the adapted `1.jpg` reference.
`production-shapes.json` records video/audio/reference/text row counts. The
ordinary-layout native/cuDNN times are 279.32/52.75 ms and 2,961.35/587.55 ms.
No production-length video was generated. The earlier H100's rounded control
shapes also pass: 32,768 tokens takes 264.60/45.49 ms, and 110,592 takes
2,979.71/538.47 ms. Tail shapes need not have exactly proportional throughput.

At 18,225 tokens, a zero attention-scratch limit correctly falls back to
native execution (82.23 ms). cuDNN takes 14.37/14.91/14.77 ms with limits of
32/128/512 MiB, offering no useful gain over the existing 64-MiB policy. Each
selected cuDNN plan here requests only 256 bytes of workspace. The generic
benchmark's
`scratch_bytes` field does not include cuDNN-owned scratch.

The decoder projection sweep covers 1,797 rows, K=2,048, N=6,144/16,384,
workspace 0/32/64/128 MiB and algorithm timing on/off. Larger workspace gives
no useful warmed improvement. At N=16,384, timing candidates improves roughly
2.73 to 2.61 ms but increases first-call time from about 69–79 to 111–115 ms.
At N=6,144, warmed time stays about 1.05 ms. The existing bounded tuning/cache
policy remains appropriate; no device-wide algorithm or workspace override
was promoted from these component measurements.

Six decoder candidates reuse the same saved 56-frame fast AV state, without
rerunning denoising. Component wall times are 12.88 seconds for default,
5.57 seconds for F32 GEMM, 8.07 seconds for tiled attention, and
5.42/5.77/5.42 seconds for TF32 with 32/64/128-MiB GEMM workspace. Every result
is finite and produces a fully decodable MP4 with the expected geometry/audio.
These times exclude each candidate's subsequent conversion/mux; the complete
46.22-second experiment, including those steps, is charged to the shared
render ledger. Candidate videos are in `decoder/review.html`. No alternate
decoder is promoted; the unchanged selected TF32 path is also exercised by the
six main quality pairs.

Nsight Compute cannot access hardware performance counters on this provider
(`ERR_NVGPUCTRPERM`). Its attempted capture is retained but is not counted as
a successful counter profile. Kernel resource reporting, unprofiled timings,
built-in cumulative profiles and the independent sanitizer runs remain usable.

## Regression evidence and scope

The cuDNN-enabled native SM90 build passes `make test`, all 41 default primitive
comparisons, 70 exact tokenizer fixtures and all 23 released-model comparisons
covering vision, text, video encoding, audio, video decoding, one DiT block and
a full small DiT evaluation. The SM90 specialization preserves the portable
kernel's default numerical contract. Fast dispatch, QKV, tails/layouts,
causal/GQA fallback, reference precedence and zero-scratch fallback pass.

Compute Sanitizer reports zero shared-memory race hazards for SM90 default and
native fast kernels. Runtime/stream-cache memcheck reports zero errors and no
leaked allocations. These are bounded checks of exercised paths, not an
exhaustive proof of race freedom.

Fresh-process fast restart and continuation restart pass, as does an explicit
fast-to-default checkpoint handoff. The default restart additionally uses
288×384, 56 frames and 20 steps: it stops at step three and checks every resumed
latent boundary against its uninterrupted control. The resulting saved AV
states are byte-identical. Recording takes 107.44 seconds, including a complete
control run and separate checkpoint capture; restarting takes 86.12 seconds.
These are test-operation timings, not independent render benchmarks.

Default and fast continuation audits check protected prefixes, source-state
immutability, changed face/body references, a 39-frame overlap and delivery of
51 new frames. Fast/default/fast switching, conditioning/prepared-DiT/decoder
reuse, reuse/core-reuse and token reduction pass. A
separate pressure test leaves only 4 GiB free, evicts the prepared cache while
retaining the decoder, renders successfully and captures another checkpoint.
The runtime suite covers 64 asynchronous slot handoffs, 16 cancellations,
in-flight teardown, file replacement/modification and cache-pressure eviction.
Profiles record a 32-MiB peak pinned bounce allocation per GPU context;
per-context tensor peaks do not include the deliberately reserved pressure
allocation or represent whole-process host RSS.

All reference families execute successfully in fast mode: T2VA, FL2VA
first/last/both frames, Ref2VA single/multiple images, video, silent video,
replacement audio and separate image-plus-audio. Selected default smokes cover
T2VA, both-frame conditioning and replacement audio, supplementing the seven
coherent default/fast pairs. Smoke outputs use two steps and are functional
evidence only. Additional fast bridge continuation and a default-state-to-fast
continuation handoff deliver 51 valid new frames with correct audio. Existing
default/fast continuation audits supply changed-reference coverage.

The no-cuDNN native build passes fast dispatch, runtime/cache checks, all 41
default primitive comparisons and a complete two-step smoke (30.10 seconds).
Native and cuDNN-enabled fat builds each pass all 41 primitive comparisons on
H200. Both fat objects contain SM86/89/90/100/120 SASS and compute_86 PTX;
SM100 and the other unavailable GPUs are cross-compilation evidence only.
Final dispatch checks retain exact SM90 behavior and skip SM120 execution on
this SM90 device. The cuDNN runtime still selects and executes 9.10.2 plans
with `LD_LIBRARY_PATH` removed, using the isolated prefix's embedded runtime
paths. The fixed base bootstrap also completes successfully on the provider
image without replacing its held packages.

Independent inspection validates checksums, geometry and finite latents for
all **38 AV states**, plus whole-file/section checksums for all **five sampler
states**. The Ref2VA content fingerprint stored by those checkpoints is
`2eed3a9f8c38f909ca8919cb6b744ceebb1bfe070bc4937244bfece87970f4d1`.
Default mode's absent optional fast-provenance section retains its defined
default semantics; fast checkpoints record implementation version one.

The local Mac build and `make test` pass. Optional released-weight Metal
fixtures and the ignored legacy `misc/fixtures` suite are absent; their skips
are not counted as successful comparisons. The setup/architecture suite has
21 passing mocked tests on macOS and Linux; four additional tests verify failed-attempt charging,
timeout logs, cumulative budget enforcement and extended-test opt-in.
The added fat-object test exercises the real Make dependency graph with a fake
compiler: unchanged settings avoid rebuilding, while cuDNN flags, included
headers and the architecture helper invalidate the object. Fat compilation now
tracks these dependencies, preventing stale SM coverage or cuDNN selection.

The Metal-to-H200 checkpoint check is **skipped at the user's request**. No
Metal checkpoint was transferred. No folded LoRA/Turbo model is supplied on
this node. Soft continuation is not implemented; supported hard/bridge modes
are the qualification targets. B200, multi-GPU and ARM/Grace execution remain
outside this node's evidence.

The final ledger contains **42 entries totaling 2,655.12 seconds
(44.25 minutes)**, including pilots, failed attempts, state/cache operations,
the six-candidate decoder experiment and media validation. It remains below
the 2,700-second limit. Individual renders are capped at 300 seconds; smoke
and final timing experiments use tighter 120/60-second caps. Isolated kernel
and component cases are capped at 120 seconds. Forty-one ledger entries
complete successfully; the registered-upload timeout is the rejected candidate
described above. No original `test2.sh` or long production render was run.

No H200-specific inference change is justified by these measurements. The
existing cuDNN-first attention, cooperative QKV, bounded TF32 decoder,
32-MiB GEMM workspace and automatic residency remain selected behind the
non-default `--fast-cuda` flag. H200's functional qualification and user
video/audio acceptance are complete within this bounded scope. B200 hardware
execution and its own quality review remain pending.

## Reproduction

```sh
cd /path/to/h3.c-h200
source outputs/cuda-validation/h200/env.sh
make -j8 all
./bin/h3cli -d /path/to/models/MiniMax-H3 --fast-cuda \
  --ref-image inputs/1.jpg --width 288 --height 384 --frames 56 \
  --steps 20 --layers 50 --reuse 1 --core-reuse 1 --seed 1001 \
  -p 'The woman in <Picture 1> walks through a sunlit forest.' \
  --profile -o outputs/preview.mp4
```

Keep weight mode on `auto` and the default cuDNN/native fallback policy. Do not
force a larger workspace just because this GPU has more VRAM. The staged
qualification commands are saved as `stage1.sh` through `stage4.sh` in the
evidence directory. `stage4-finish.sh` records the targeted continuation after
the rejected upload experiment; it does not repeat earlier renders.
Existing successful helper records are reused; an explicit
rerun remains charged to the shared rendering budget.
