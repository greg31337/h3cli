# Sampler checkpoints

Only current schema 2 is accepted. See the [current state contract](current-state-contract.md)
for all saved formats. Older files are intentionally unsupported.

A `.h3sample` stores an unfinished generation, including exact conditioning,
current AV latents, original sigma grids, continuation masks, and native CPU/GPU
velocity and core-reuse history. A `.h3av` still stores a completed clean AV latent for starting the
next segment. Neither format embeds model weights.

## Stop and resume

Use the same model, engine build, GPU family, and numerical environment for both
commands. Checkpoints support CPU-state and GPU-state Metal sampling, native BF16
GPU velocity history, core reuse, and token reduction. Whole-denoiser reuse 1, 2,
and 3 is supported, including explicit `H3_REUSE_STEPS` evaluation schedules.
Existing generation restrictions still apply: core reuse and whole-denoiser reuse
cannot be combined; continuation requires all 50 blocks and token reduction off.
Hard and bridge continuation use CPU state by default and support explicit GPU
state selection. Bridge continuation permits core reuse 1/4/6 and
rejects custom whole-denoiser schedules.

M4 selects CPU state by default. Set `H3_GPU_SAMPLER=1` with `H3_CPU_SAMPLER`
unset or zero to select GPU state explicitly. Keep those settings identical for
save and resume. M5 execution requires a separate hardware validation run; M4 GPU
results do not certify M5. Ordinary generation retains its existing selection rules.

```sh
export H3_CPU_SAMPLER=1
./bin/h3cli -d models/MiniMax-H3 -p 'A person walks through a sunlit garden.' \
  --width 256 --height 256 --frames 90 --steps 20 --layers 50 \
  --ref-image inputs/face1.jpg --ref-image inputs/body1.jpg \
  --stop-after-step 4 --save-sampler-state outputs/shot-step4.h3sample \
  --preview-on-stop -o outputs/shot-preview.mp4

./bin/h3cli -d models/MiniMax-H3 --resume-sampler-state outputs/shot-step4.h3sample \
  -o outputs/shot-final.mp4 --save-av-state outputs/shot-final.h3av
```

`--stop-after-step 4` executes transitions 0, 1, 2, and 3 of the **20-step**
schedule. The stored `next_step` is 4 and the next transition uses `sigma[4]`.
It does not construct a four-step schedule. A resumed `--stop-after-step 8`
executes four more transitions. Zero is a valid initial checkpoint boundary;
a boundary equal to the full schedule length produces a complete result.

Stopping requires a checkpoint output path on the CLI. Without
`--preview-on-stop`, a pause writes no media. The preview contains silent video
only. Saving happens first; the decoder receives a separate latent copy, and the
checkpoint retains all current audio latent values. A complete pause-preview clip
uses ordinary `on_frame` delivery (including `--frames-dir`); result metadata carries
its completed-step boundary. A paused result is successful and the CLI exits zero.
`--save-av-state` requires final completion.

For long generations, pass `--save-sampler-state shot.h3sample` without
`--stop-after-step` to save the completed denoising state before audio/video
decoding. If decoding fails, resume that checkpoint with the same compatible
build, model and numerical settings to retry decoding without repeating
denoising. `--save-av-state` alone does not provide this protection: the CLI
writes it only after generation and decoding succeed. `test2.sh` saves a
separate completed sampler checkpoint for each segment.

The checkpoint is authoritative during CLI resume. Prompt, seed, reference,
geometry, step count, layer, reuse, precision, RoPE, streaming, and continuation
arguments are rejected even when their values happen to match the checkpoint.

Model directory, output path, frame delivery, profiling, preview settings, a later
absolute stop, and a new checkpoint destination are permitted. The original
reference media and preceding continuation `.h3av` are not required after saving.

## API and ownership

`h3_params` adds `stop_after_step` (default -1), `save_sampler_state`,
`resume_sampler_state`, and `preview_on_stop`. Resume uses only the caller's output,
callback, preview, stop, and checkpoint-destination fields; generation fields
come from the checkpoint. Current video checkpoints restore the sole released recipe. Use `h3_generate(ctx, NULL, &params)` with
`resume_sampler_state` set.

`h3_result` adds `status`, `completed_steps`, `total_steps`, and an owned
`sampler_state`. `H3_RESULT_PAUSED` owns its state and has no clean `av_state`;
`H3_RESULT_COMPLETE` retains the existing completed AV result. Release either with
`h3_result_free()`. The API may pause without a file destination and the caller
can subsequently call `h3_sampler_state_save()` on the result's state.

The internal `h3_dit_denoise_euler_range()` runs from `next_step` through an
absolute exclusive stop boundary using the original complete sigma arrays. It
updates `next_step` after each successful Euler transition, before callbacks. A
callback abort remains an error. GPU range execution synchronizes the Metal command
chain before exporting packed F32 samples into canonical channel-major layout and
copies velocity history as raw BF16 words. Ordinary Euler and preview entry points create
step-zero state and run the whole schedule.

Live denoising previews can display an estimated clean latent while
`on_latent_step` and checkpoint files always contain the exact noisy sampler
state. `H3_PREVIEW_MODE` and `H3_PREVIEW_DIAGNOSTICS` are presentation settings
and may change across resume. See [denoised previews](../preview/denoise.md).

`h3_result` also exposes optional `resume_count`, `resume_step`, `resume_format`,
and `resume_hash` diagnostic fields. The hash is SHA-256 of the complete checkpoint
file. Each actual resume increments the count and records its input boundary;
loading a file for inspection does not increment it. This provenance is logged
and propagated through subsequent `.h3sample` files. Completed `.h3av` bytes remain
canonical so uninterrupted and resumed continuation sources compare exactly.

## Format version 2

All integers are little-endian. Signed scalars use 32-bit two's complement. F32,
BF16, and F64 payloads retain their exact IEEE/native BF16 bits; there is no
quantization, compression, JSON float conversion, or ABI/padding serialization.
Current reader limits: 16 GiB file, 256 sections, 1 MiB strings/provenance, one
million conditioning tokens, ten million layout rows, and 1,000 segments/steps.
Tensor lengths must agree with released geometry and the resolved layout.

The 96-byte header is:

| Offset | Bytes | Meaning |
| --- | --- | --- |
| 0 | 8 | ASCII `H3SAMPLE` |
| 8 | 4 | Container version: 2 |
| 12 | 4 | Header bytes: 96 |
| 16 | 4 | Flags: 0 |
| 20 | 4 | Section count |
| 24 | 8 | Exact total file length |
| 32 | 4 | Endian marker `0x01020304` |
| 36 | 28 | Reserved, zero |
| 64 | 32 | SHA-256 of the entire file with these 32 bytes zeroed |

The section table immediately follows the header. Each 72-byte entry contains:

| Offset | Bytes | Meaning |
| --- | --- | --- |
| 0 | 4 | Section type |
| 4 | 4 | Section version: 1 |
| 8 | 4 | Flags: bit 0 means required; other bits reserved |
| 12 | 4 | Dtype: 1 structured bytes, 2 F32, 3 BF16 |
| 16 | 8 | Element count (byte count for structured sections) |
| 24 | 8 | Absolute payload offset |
| 32 | 8 | Payload bytes |
| 40 | 32 | SHA-256 of payload |

Payloads are contiguous in table order, without overlaps, gaps, or trailing
bytes. Section order need not match type order; the reader restores identity
before dependent tensors. Duplicate types, unsupported required versions,
missing required payloads, bad bounds, and inconsistent shapes are errors.
Unknown optional types/versions are integrity-checked and skipped. Optional
noise/diagnostic sections may be removed without preventing resume.

| Type | Section | Required |
| --- | --- | --- |
| 1 | Versioned generation identity and explicit sampler mode | Yes |
| 2 | Build/compiler/git identity, backend version, device capabilities, environment | Yes |
| 3 | Selected model metadata fingerprint and AV metadata identity | Yes |
| 4 | Verbatim prompt and ordered source provenance | Yes |
| 5 | Text embedding raw BF16 `[tokens,5120]` | Yes |
| 6 | U64 modality-tag count followed by exact U8 tags; zero preserves NULL | Yes |
| 7 | Final augmented patchified visual condition rows, F32 | Yes (may be empty) |
| 8 | Final augmented patchified audio condition rows, F32 | Yes (may be empty) |
| 9 | Ordered layout reference descriptors | Yes (may be empty) |
| 10 | Complete resolved layout, segments, F64 positions, counts, signature, prefix | Yes |
| 11 | I32 total steps and complete video then audio F32 sigma arrays | Yes |
| 12 | Current video latent F32 `[24,T,H,W]` | Yes |
| 13 | Current audio latent F32 `[32,2,T]` | Yes |
| 14 | RNG version, both RNG states, consumed normal-value counts | Yes |
| 15 | Continuation provenance, initialization, trim context and effective bridge mask | Yes |
| 16 | I32 last/previous evaluation indices and U8 evaluation bitmap | Yes |
| 17–20 | Last/previous video, then last/previous audio F32 velocities | CPU state with reuse > 1 |
| 21–22 | Original video/audio F32 noise before prefix insertion | Optional |
| 23 | Tokenizer IDs, presentation position IDs and vision-span boundaries | Optional |
| 24 | Execution version, core counters/shape, resolved reduction configuration/topology | Yes |
| 25–28 | Last/previous video, then last/previous audio raw BF16 velocities in packed row order | GPU state with reuse > 1 |
| 29 | Core residual raw BF16 `[full_sequence,5376]` | Core reuse > 1 |
| 30 | Versioned prepared DiT cache with compatibility key and native BF16 tensors | Optional |
| 31 | Resume count, input boundary, format version and complete checkpoint-file SHA-256 | Optional |
| 32 | Effective released Ref2VA video recipe | Yes |
| 34 | CUDA denoiser quantization policy and recipe | CUDA quantization |
| 35 | CUDA attention policy, recipe and plan | Sage attention |
| 36 | Metal attention/layout/ANE options and recipe | Native Metal backend |
| 37 | Metal Q8 weight format, recipe, group size and kernel policy | Q8 weights |

The [Metal Q8 extension](../metal/q8.md) adds required section 37 without changing
BF16 section 36. Its 16-byte payload is I32 format (`1` = Q8), U32 recipe
(`1`), U32 group size (`64`), and I32 kernel (`0` = bounded Metal dequantization
plus MPSGraph, `1` = native simdgroup). Unknown recipes, group sizes, kernels,
optional markings and truncated payloads are rejected. BF16 checkpoints omit
this section; older readers reject Q8 checkpoints as an unknown required type.
Prepared-cache identity includes this policy, and resume cannot silently
change weight format or kernel.

The current container accepts only schema 2. Section 32 is mandatory; retired
section 33 is unsupported. Sections 40 (cache budget) and 44 (warmups) require
version 2 with explicit values. Section 45 uses stage-specific source version 1
or refinement version 2; these are active record types, not migration paths.
All other supported sections use version 1.

Section 1 starts with U32 schema, I32 `total_steps`, `next_step`, and reuse
interval. Its ordered I32 parameter fields are `width`, `height`, `frames`,
`steps`, `reference_image_size`, `denoise_reuse`, `dit_layers`, `core_reuse`,
`token_reduction`, `use_int8_row_fc2`, `use_reference_rope`, `ssd_streaming`,
`render_width`, `render_height`, the ten `use_slower_*` flags in `h3_params`
declaration order, `continuation_context_frames`, `keep_continuation_prefix`,
`continuation_mode`, `bridge_video_steps`, and `bridge_profile`. These are followed
by U64 seed/reference count, F32 maximum bridge strength, I32 Ref2VA/conditioned
flags, I32 resolved render width/height/aligned frames/video T/H/W/audio T, F32
spatial RoPE scale, U64 current video/audio counts, condition video/audio counts,
text token count/width/reference count, and U32 sampler mode (0 = CPU-state Metal;
1 = GPU-state Metal; other modes reserved).

Strings use a U64 byte length followed by unmodified UTF-8 bytes without a NUL.
Section 4 contains the prompt string, U64 provenance byte count, and provenance
records. Provenance begins with U64 record count. Each record has U64 reference
kind and embedded-audio flag, followed by two path records (media and optional
separate audio). A path record has the length-delimited original path, U64 file
size, and 32-byte content SHA-256. Kinds 1–4 match `h3_reference_kind`; 5 and 6
identify first/last-frame anchors. Missing optional paths have zero length, size,
and digest. Paths are provenance only; no resume operation opens them.

Section 9 encodes each reference as five I32 values: kind, latent T/H/W, audio T.
Section 10 begins with six U64 values: sequence length, segment count, visual
condition/target rows, audio condition/target rows. Five I32 signature dimensions
and two I32 prefix lengths follow. Each segment has U64 start/stop and I32 kind;
each position has F64 t/h/w. No layout reconstruction runs during resume.

RNG version 1 means the existing PCG32 generator with Box–Muller normals. Each
stream has U64 state/increment, raw F32 spare, and I32 spare-present flag. Counters
measure consumed **normal values**, not underlying U32 draws. Original noise is
captured before inherited continuation values are mixed into the target.

Section 15 begins with I32 continuation/context frames, the SHA-256 of the
canonical source `.h3av` file, and separate SHA-256 values for the channel-major
raw copied video/audio tails. F32 0.999 and 0.001 coefficients, I32 exact-audio
preservation mode (1), and I32 bridge-present flag follow. When present, the
bridge subsection stores all profile scalar fields, active-class bitmap, F32
class strengths, and complete temporal video/audio class arrays. The output trim
policy is stored explicitly in section 1. The source latent itself is omitted.

Section 23 has U64 token count and raw U32 tokenizer IDs, I32 positions-present,
optional U32 `[3,tokens]` positions, U64 vision-span count and U64
`[start,token_count]` pairs. Multimodal presentation diagnostics are captured when
available; an existing conditioning-cache hit may omit them. They do not affect
inference.

Section 24 contains U32 execution version (1), core forward count and residual-ready
flag; U64 core rows, columns and element count; U32 reduction enabled, active, begin
block, end block, early-step count and early end block; F32 reduction scale; U64
full and reduced sequence lengths. Every valid Euler boundary has full topology:
reduction-active must be zero. Core residuals always have the full sequence shape,
because the block loop restores full topology before computing the residual.
The reader rejects absent or inconsistent core history instead of resetting it.
GPU history indices use section 16; sections 25–28 preserve every BF16 bit directly.

Section 30 has U32 cache version, a 32-byte compatibility key, U64 tensor count,
then records of U32 tensor ID, U64 element count and raw BF16 words. IDs 1 and 2
mean refined text and final AdaLN; IDs 100–149 mean block AdaLN. The key hashes
versioned canonical generation/implementation identity, model identity, exact
conditioning, layout, masks and sigmas. It excludes the mutable completed-step
index, latents, histories and resume provenance. Unsupported cache versions,
missing tensors, wrong tensor shapes or a different cache key trigger deterministic
rebuilding; corrupt container structure/checksums remain errors.

Same-process resume first checks the key of the live `ctx->dit`. A matching model
is reused, with mutable state overwritten from the checkpoint. Otherwise the loader
imports compatible refined text and AdaLN tensors before loading the transformer
core, or recomputes them from serialized conditioning. With layer thinning, pruned
block tensors may force AdaLN rebuilding; refined text can still be imported.

RoPE and row/modulation maps are rebuilt from the exact saved layout, text tags,
bridge classes and sigmas. They require no model projections, and retaining them
would duplicate inexpensive derived data. The expensive text/AdaLN cache is about
361 MiB in the 256×256, 90-frame, 20-step T2VA fixture. Object identities, model
weights, command queues/buffers, MPSGraph objects, compiled pipelines, temporary
attention/QKV/MLP activations, VAE decoders and FFmpeg state are excluded.

Section 31 has U32 resume count, input completed-step index and source format
version, then 32 bytes containing the source checkpoint SHA-256.

Section 32 contains U32 effective recipe: 0 = no Ref2VA video, 2 = released-v1.
It must agree with section 9 reference kinds and participates in prepared-cache
identity. Missing, unknown or inconsistent recipe metadata is rejected.

## Model and implementation compatibility

The current model fingerprint uses sorted model metadata and the effective LoRA
transformer identity, without rereading entire model shards at startup. Decoder
identity and container payload checksums are separately verified. Explicit
strict quantization-cache verification may hash weights; this is an active
verification feature, not a saved-file compatibility fallback.

Build identity includes a SHA-256 over library source, headers, shaders and
compiler options, plus git commit when available and compiler version. Exact
resume rejects another identity, backend state version, GPU family, Metal4 or
unified-memory capability, model fingerprint, or numerical `H3_*` environment.
Environment entries are sorted and length-prefixed, preserving even values with
embedded newlines without ambiguity. Profiling, diagnostic logging and FFmpeg
executable overrides are excluded from the numerical environment comparison.
Cross-build and incompatible-device resume are
not certified. Test guarantees apply to the tested same-device execution path. A future compatible
Metal mode could allow equivalent results across devices/builds with tolerance-based
validation; the current loader deliberately exposes only strict same-path resume.
Canonical F32 latents, BF16 histories with explicit dtype, implementation/backend
versions and reserved sampler modes leave room for a future CUDA importer without
claiming that Metal and CUDA produce identical results.

Checkpoint writes use a unique same-directory temporary file, an initially
invalid header, flushed/fsynced section payloads, then the complete checksummed
header and another flush/fsync. On-disk envelope validation precedes rename. A
failed write removes the temporary and leaves an existing destination untouched.
The containing directory is synced after rename when available.

## Tests

```sh
make test-sampler
make test-sampler-sanitize
make test-sampler-gpu
# Optional existing VAE-encoded face/body/12 image fixtures, with no decoding:
./bin/sampler_tests --encoded outputs/bridge-validation/encoded-inputs
# Full-model jobs run sequentially; retain sufficient memory for one model.
make bin/sampler_generate
python3 tests/resume_baseline.py
python3 tests/resume_baseline.py --current
python3 tests/run_sampler.py --output outputs/resume-validation/new-suite
python3 tests/sampler_noise.py outputs/resume-validation/new-suite/*/oracle.h3sample
python3 tests/sampler_cli.py --render
# T033–T070 matrix; choose a fresh output directory. Large jobs are sequential.
python3 tests/run_sampler_advanced.py --output outputs/resume-validation-advanced/new-suite
# Actual GPU CLI without per-step callbacks (use the matching oracle/build):
python3 tests/sampler_gpu_cli.py \
  --oracle outputs/resume-validation-advanced/new-suite/gpu_reuse3/oracle \
  --output outputs/resume-validation-advanced/new-gpu-cli
python3 tests/sampler_noise.py outputs/resume-validation-advanced/new-suite/*/oracle.h3sample
python3 tests/sampler_provenance.py \
  outputs/resume-validation-advanced/new-suite/t2va/oracle.h3sample \
  outputs/resume-validation-advanced/new-suite/t2va-resume4.log
# On actual M5 hardware, require its presence explicitly:
# python3 tests/run_sampler_advanced.py --require-m5 --only gpu,gpu_reuse3,gpu_core4,gpu_reduction --output outputs/m5-resume
```

Recorded results and tested scope are in the [initial acceptance report](resume-acceptance.md)
and [T033–T070 acceptance report](resume-advanced-acceptance.md).
Choose a fresh matrix output directory for each run.

The pinned pre-change build is `7f4aa56`; hashes and commands are recorded in
[resume-baseline.json](resume-baseline.json). Its cold multimodal text encoder can
vary across processes. Baseline regression replays captured conditioning only in
isolated test copies, using the existing continuation regression instrumentation.
Production resume restores its own serialized conditioning directly.

The real-model harness writes an uninterrupted 21-boundary trajectory, checks a
four-step pause against it in the same prepared context, verifies preview leaves
checkpoint bytes intact, then checks every remaining boundary after process
restart. It compares final `.h3av` and MP4 bytes and verifies silent previews.
References and the prior continuation `.h3av` disappear from the isolated working
directory before resume. Host tests cover exact F32/BF16/position round-trips,
provenance, RNG, model-cache invalidation, configuration incompatibility, optional
sections, structural corruption, and failed-write preservation.

## Scouting without restarting the shot

Keep final resolution, frame count, model, references and the full 20- or 50-step
schedule fixed. Save after the true first 3–5 transitions, inspect the silent
preview for composition, then resume that checkpoint when approved. Changing to
`--steps 4` creates a different schedule; generating again from the seed repeats
conditioning and sampling rather than continuing the saved shot.

A second scout can advance the same shot from step 4 to absolute step 8:

```sh
./bin/h3cli -d models/MiniMax-H3 --resume-sampler-state outputs/shot-step4.h3sample \
  --stop-after-step 8 --save-sampler-state outputs/shot-step8.h3sample \
  --preview-on-stop -o outputs/shot-step8-preview.mp4
./bin/h3cli -d models/MiniMax-H3 --resume-sampler-state outputs/shot-step8.h3sample \
  -o outputs/shot-final.mp4 --save-av-state outputs/shot-final.h3av
```

For a 50-step shot, specify `--steps 50` only in the initial generation. Resume
reads all 51 sigma entries directly from its checkpoint. Early previews do not
predict final detail or audio quality; the saved mathematical state is unchanged
by inspecting them.
