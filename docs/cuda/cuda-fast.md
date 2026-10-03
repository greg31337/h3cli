> Historical kernel/qualification guide. M3B/M4 remove `--fast-cuda` and use

Historical record: renderer modes and legacy test launchers described below have been
retired. Retained artifacts preserve the original results. Use the
[single-pipeline design](design-single-pipeline.md) and
[current qualification](single-pipeline-final.md) for supported commands.
> [one CUDA pipeline with explicit attention/precision options](design-single-pipeline.md).
> Commands and old acceptance statements below describe their recorded recipes.
> Use the current design for supported commands and the
> [single-pipeline report](single-pipeline-results.md) for new comparisons.
> Explicit approximate options may degrade quality; default reference parity remains frozen.


`--fast-cuda` is an optional execution policy. Functional qualification and
human quality approval are recorded separately for each GPU and tested corpus.
The user accepted the PRO 6000 corpus quality on 2026-09-17
and the 3090/5090 outputs' visual quality and H200/B200 video/audio quality on
2026-09-18; see
[the acceptance record](cuda-fast-acceptance.json) for each decision's scope.
Original model files, all 50
transformer layers, the requested schedule, references, continuation geometry,
reuse settings and output format remain authoritative. The existing CUDA path
is the default and comparison baseline. Metal rejects explicit `--fast-cuda`.

Fast-mode acceptance concerns speed and perceived video/audio quality: identity,
reference and action fidelity, anatomy, detail, color, flicker, tile seams,
continuity, audible artifacts and AV timing. BF16 probabilities, parallel
reductions, TF32 attention products and different denoising trajectories are
permitted. Numerical equivalence is not its acceptance gate. Memory safety,
synchronization, state/file integrity and valid output remain requirements.
The default backend's numerical tests are unchanged.

Usage:

```sh
./bin/h3cli -d ~/models/MiniMax-H3 --fast-cuda \
  -p 'The woman in <Picture 1> walks through a sunlit forest.' \
  --ref-image inputs/1.jpg --width 288 --height 384 \
  --frames 56 --steps 20 --reuse 1 --core-reuse 1 --seed 1001 \
  --profile -o fast.mp4
```

The implementation is tested on RTX PRO 6000 Server Edition and RTX 5090
(SM120), RTX 3090 (SM86), H100/H200 (SM90), and B200 (SM100), with CUDA 12.8. See the
[5090 qualification](cuda-5090-qualification.md) for the streaming optimization
and the [3090 qualification](cuda-3090-qualification.md) for 24-GB behavior,
attention/upload comparisons, measured results and review limits. The
[H100 qualification](cuda-h100-qualification.md) records resident execution,
Hopper attention measurements and its separate quality-review status.
The [H200 qualification](cuda-h200-qualification.md) records the existing
cuDNN-first path's measurements and accepted video/audio quality for the six
matched pairs. H200 reuses the
SM90 implementation; the larger memory capacity does not imply an additional
kernel speedup. [B200 technical qualification](cuda-b200-qualification.md)
passes native/no-cuDNN/fat/project-PTX, default/model, fast reference/state and
memory/sanitizer checks. Its six short matched pairs take 47–51 seconds fast
versus 119–158 seconds default (2.4–3.1×); the user accepted video/audio quality
for these six pairs on 2026-09-18. Keep the qualified cuDNN 9.10.2 / frontend 1.11.0 combination,
64-MiB scratch and cached/timed 32-MiB GEMM workspace. Larger workspaces and
alternate decoder choices supplied no useful gain; no SM100-specific inference
change was promoted. These are existing-fast versus default measurements,
not an added B200 optimization or a long-render prediction.
On the tested H200 host, retain automatic residency and ordinary uploads;
leave `H3_CUDA_REGISTER_WEIGHTS` unset. Its registered-upload candidate exceeded
60 seconds during loading, while ordinary uploads completed the matched
20-step preset in 36.3 seconds. Larger workspace limits also supplied no
useful improvement in the bounded H200 comparisons.
The native candidate requires SM80 or later and sufficient per-block shared
memory. Other CUDA devices retain safe shape/capability fallbacks; this is not
hardware qualification of untested cards. `H3_CUDA_REFERENCE=1` disables fast
execution, including accelerated compatibility hashing, and prints a notice.

Builds need no cuDNN or Python inference package. Optional native cuDNN SDPA
was tested with cuDNN 9.8.0/9.10.2 and NVIDIA cudnn-frontend 1.11.0:

```sh
git clone --depth 1 --branch v1.11.0 \
  https://github.com/NVIDIA/cudnn-frontend.git /path/to/cudnn-frontend
make CUDA_ARCH=120 CUDA_CUDNN=1 \
  CUDNN_FRONTEND_PATH=/path/to/cudnn-frontend/include
```

Use `CUDA_ARCH=86` for an RTX 3090, or `CUDA_ARCH=auto` on the target machine.
cuDNN outperformed native attention in the measured 3090 cases; a build without
cuDNN still supports native fast attention. Keep automatic weight residency and
caching on this 24-GB card. Registered-file uploads remain opt-in and were much
slower on the tested 3090 host; leave `H3_CUDA_REGISTER_WEIGHTS` unset there.

The ordinary build uses native fused BF16 attention. An optional cuDNN build
tries cached SDPA plans first, then native attention, then the existing backend
for unsupported shapes. DiT attention keeps linear workspace in sequence
length. Causal/GQA and unsupported head dimensions retain the existing route.
VideoVAE's F32 attention uses TF32 GEMM products, F32 softmax and bounded head
groups; the default score workspace limit is 64 MiB per context. The fast
workflow also uses cooperative Q/K RMS normalization and RoPE, context-owned
cuBLASLt descriptors and bounded candidate timing. Algorithms are cached by
shape/type/bias/alignment within the immutable device/library/workspace context.
Bias addresses are refreshed for each call.

Resident fast execution omits redundant upload waits and release events.
The first resident overwrite drains prior reads; streamed slots keep their
ready/release fences. Decoder input and suffix buffers are reused with their
owning decoder. CUDA Graphs, batched tile decoding and GPU frame stitching are
not enabled: the bounded experiments did not establish a useful complete-run
gain for their added resource and scheduling costs.

When the full DiT cannot fit, fast mode retains a bounded subset of streamed
BF16 weights in spare VRAM. Cache hits copy unchanged bytes into the existing
streaming slots, with their original release/ready fences. Admission reserves
estimated activations, two GiB for streaming/other allocations, and the larger
of one GiB or 10% of VRAM. A tensor allocation that needs that space drops the
cache and retries; the rest of that context continues without caching. Entries
are retained rather than evicted on every miss, avoiding churn during repeated
scans of a model larger than the cache. Context teardown frees them.
Before decoding, an optional-cache release can preserve the prepared DiT while
making room for the VAE. This phase release permits later cache admission;
the retained decoder's memory then limits how much can be cached next time.

The cache changes transfers, not weight precision or model computation. Its
keys include file identity, size, nanosecond modification/change times, offset,
length and dtype. Model files must remain immutable during inference; editing
checkpoint bytes while preserving metadata is unsupported. Profiling includes
cached bytes in the device/live/peak totals and reports cache hits separately
from actual streamed H2D bytes. Resident execution and default CUDA do not use
this cache.

Linux detects the optional OpenSSL development package using `pkg-config
libcrypto`. Fast execution then uses accelerated SHA-256 for compatibility
checks. The digest and the files checked are unchanged. `CUDA_OPENSSL=0`
preserves a build without this dependency, with portable hashing in both modes.
The default path always uses its existing portable implementation; Darwin keeps
CommonCrypto. Checkpoint hashes and signatures are not skipped or weakened.

Requested/effective execution, candidate fallback and selected routes appear
in stderr. `--profile` adds dispatch counts, actual attention/GEMM shapes,
workspace, tensor peaks and stage timings. CUDA event waits overlap device
execution and are not additive phase costs. GEMM and cuDNN plan construction
costs appear in cold measurements; cached reruns reuse the plans.

Diagnostic candidate controls, only active with `--fast-cuda`:

| Variable | Values / default |
| --- | --- |
| `H3_FAST_CUDA_ATTENTION` | `auto` (cuDNN if built, native fallback), `native`, `cudnn`, `default` |
| `H3_FAST_CUDA_VAE` | `tf32` (default), `f32`, `tiled` (comparison candidate), `default` |
| `H3_FAST_CUDA_SCRATCH_MB` | 0–512, default 64; zero exercises decoder fallback |
| `H3_FAST_CUDA_WORKSPACE_MB` | 0–128, default 32, cuBLASLt workspace |
| `H3_FAST_CUDA_GEMM_TUNE` | `0` disables candidate timing; descriptors remain cached |
| `H3_FAST_CUDA_WEIGHT_CACHE_MB` | Automatic VRAM-based budget by default; a nonnegative MiB cap, or `0` to disable streamed-weight caching |

Candidate controls belong to conditioning and decoder cache keys. Prepared-DiT
keys include the conditioning key; checkpoint prepared keys also include fast
mode/version. Context teardown owns and frees descriptors, plans, scratch and
handles. Failed capability/plan/workspace preflight can fall back before output
is modified. A failed executed CUDA/cuBLAS/cuDNN operation aborts; it never
restarts a different kernel on partially written output.

Sampler checkpoints retain the requested mode and implementation version in a
required extension. Files without that extension mean default execution. The
checkpoint restores execution policy; do not add `--fast-cuda` to a resume
command. Existing build/model/environment compatibility validation still
applies. A mode change is explicit:

```sh
./bin/h3cli -d ~/models/MiniMax-H3 --resume-sampler-state saved.h3sample -o resumed.mp4
./bin/h3cli -d ~/models/MiniMax-H3 --resume-sampler-state saved.h3sample \
  --resume-default-cuda -o default-handoff.mp4
```

`--resume-default-cuda` invalidates fast prepared caches and also permits a
fast-to-Metal handoff with compatible source/model files. A fast checkpoint on
Metal without that option is rejected. H3 environment choices must still match
the checkpoint; changing candidate variables does not silently restore a
different execution policy. AV states retain their backend-neutral format.

The required bounded test presets are frozen for both modes:

| Preset | Resolution | Target frames | Steps | Evidence |
| --- | --- | --- | --- | --- |
| Smoke | 128×128 | 22; 56 for video references | 2 | Functional only |
| Short | 288×384 | 56 | 20 | Performance and quality candidate |
| Continuation | 288×384 | 90 × two segments | 20 | 39-frame overlap, 51 new frames in segment two |
| Detail | 480×640 | 22 | 20 | Native-resolution face/detail/seam check |
| Compact (`--compact`) | 192×256 | 22; 56 for video references; 90 for continuation | 10 | Matched preview comparison on slower hosts |

If the standard pilot reaches its timeout, keep that result and use the compact
preset in a separate output directory, with the **same shared `--ledger`**.
Both modes must use the smaller settings. Compact detail cases retain 480×640
and use ten steps; compact continuation keeps the 39-frame overlap and delivers
51 new frames. These previews do not replace twenty-step quality acceptance.
The helper rejects reuse of an existing result with different command settings.
`cuda_fast_features.py --compact` reduces ordinary resume checks to 128×128,
22 frames and two steps; continuation resume keeps its source dimensions and
90-frame target. `--cases default-resume` audits the unchanged backend's exact
completed-boundary restart contract using the same bounded helper.
`--cases continuation-audit` checks exact initial noise/prefix construction and
protected prefixes at every Euler boundary in default and fast modes, including
rectangular source states. It uses two steps and a 90-frame target; its clips
are functional evidence only. Build `bin/continuation_generate` for this case.

Use the same seed, inputs, original weights, 50 layers, reuse=1, core-reuse=1
and token reduction disabled for each pair. The frozen corpus includes 1.jpg,
2.jpg, face1/face2 and body1, generated moving reference video and its audio.
There is one seed per reference mode and a selected second-seed image case.
Inspect all nine reference modes at full speed and listen to the audio; contact
sheets alone are insufficient. Tests create review records with explicit
pending playback/listening fields. The completed corpus has a separate,
user-supplied workflow acceptance; per-clip playback/listening annotations are
not inferred from that overall confirmation. New results require their own
review. No script marks perceptual acceptance from finite tensors, a successful
mux, tiny smoke clips or a scalar metric.

Each CUDA render is capped at 300 seconds and the required rendering round
at 2,700 seconds. M4 renders are capped at 600 seconds and the representative
Metal round at 1,200 seconds. The helpers reserve the timeout against the
remaining budget before starting a case, preserve partial logs, and support
targeted reruns. Isolated shapes use one warmup and three timed iterations,
with 120 seconds per case. Production-length shapes are kernel benchmarks;
they are not fresh long-form quality evidence. The original `test2.sh` is
unchanged. Long 362-frame/50-step rendering requires explicit `--extended`.

```sh
make cuda-fast-test
python3 tests/cuda_fast_bench.py
python3 tests/cuda_fast_workflow.py --model ~/models/MiniMax-H3 \
  --cases image05,t2va,first,last,first-last,images,video,silent,video-audio,audio,image12-seed2,detail
python3 tests/cuda_fast_workflow.py --model ~/models/MiniMax-H3 \
  --cases chain1,chain2,bridge,changed-reference,handoff
python3 tests/cuda_fast_workflow.py --review-only
```

See [fast CUDA validation](cuda-fast-validation.md) for measured results,
review status, rejected experiments and remaining coverage.

Additional bounded targets (run them on an otherwise idle GPU):

```sh
make cuda-fast-features # resume, cache pressure and memory modes
make cuda-fast-decode   # requires baseline/image05.h3av; same-state candidates
make cuda-fast-tune     # detect local architecture; 120 seconds per case
python3 tests/cuda_fast_tune.py --arch 90  # explicit Hopper layout/staging sweep
python3 tests/cuda_fast_tiles.py --arch 90 # explicit Hopper tile sweep
make h3_cuda_fast_projection_test
timeout 120 ./h3_cuda_fast_projection_test --dit 2281 # four BF16 DiT projections
python3 tests/cuda_fast_features.py --cases lora --lora-model /path/to/Realism
python3 tests/cuda_fast_features.py --cases turbo --lora-model /path/to/Turbo
```

The short corpus plus continuation variants took about 30 minutes in the first
paired round; functional additions, targeted reruns and pilots remain subject
to the aggregate 45-minute rendering limit. Isolated sweeps do not render
videos. Decoder artifact conversion uses optional NumPy; inference remains C/CUDA.

The attention benchmark records the first operation separately as
`cold_seconds`, including plan setup, then retains three warmed samples.
The projection benchmark's `--dit ROWS` option covers QKV, attention output and
both MLP projections with BF16 inputs/weights. It reports first-call and warmed
times on the same context, preserving algorithm-cache reuse. `TILES COLUMNS`
continues to exercise the F32 decoder shapes. These are isolated measurements;
process startup, model loading and complete rendering must be measured separately.

`bin/cuda_references_test MODEL OUTPUT_DIR VIDEO AUDIO [CASE]` covers T2VA,
first/last/both-frame conditioning and video, silent-video, replacement-audio
and image-plus-audio references in one reusable API session. Build it with
`make bin/cuda_references_test` and supply an existing output directory and
coherent reference media. It retains all 50 layers, uses two steps, validates
finite latent boundaries and saves an MP4/AV state per case. The B200 driver
caps the whole session at 300 seconds, charges it to the shared rendering
ledger, and independently validates every output's complete media and AV state.
These outputs test functionality and reference changes; they cannot establish
visual/audio quality. Supply a case name to run a targeted remainder when a
combined session would exceed the available budget; preserve earlier attempts
in the same ledger.

Folded LoRA directories may contain FL2VA, Ref2VA or both. Loading no longer
requires unrelated FL2VA assets for an ordered-reference request. Prompt-only
and first/last-frame requests require FL2VA; ordered image/video/audio references
require Ref2VA. Missing selected modes fail before generation. Keep LoRA settings
outside the original-weight performance comparison; Turbo's six-step check is
feature evidence, not acceptance of ordinary twenty-step quality at six steps.
