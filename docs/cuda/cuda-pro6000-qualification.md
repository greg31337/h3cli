# RTX PRO 6000 qualification

This extends the [RTX 4090 qualification](cuda-qualification.md) using the same
original BF16 models, shared H3 implementation, fixtures, and numerical gates.
Development and results are under `/path/to/h3.c` on the supplied server; local
evidence is under `outputs/cuda-validation/pro6000`.

## Long-audio launch fix (2026-09-17)

The 362-frame `test2.sh` run exposed an AudioVAE launch limit missed by the
shorter qualification clips: SnakeBeta used `grid.y = ceil(length / 4)` and
exceeded CUDA's 65,535-block y limit at more than 262,140 samples. The final
stage of that run needs 482,400 samples (603 latents × 800), or 120,600 blocks.
Free VRAM does not affect this limit. The generated CUDA wrapper now splits
time into bounded launches with a global time offset, preserving filter access
across every split. The generator owns both the wrapper and kernel changes.

Validation evidence is under `outputs/cuda-validation/alias-free-fix`:

- The new regression fails against the old CUDA object at 262,141 samples.
- Patched default and reference CUDA modes pass 262,139, 262,140, 262,141,
  482,400 and 524,281 samples, with mono/stereo batches, partial channel tiles,
  complete output writes and full-input filtering checks around each split.
- The released AudioVAE completes all seven stages for 603 synthetic latent
  positions, producing 482,400 finite stereo samples at 32 kHz. This test runs
  with a 120-second timeout and requires no denoising or video rendering.
- Linux `make test` and the released-model audio encoder/decoder comparisons
  pass. The long primitive regression also passes on the M4 Metal backend;
  local evidence is in `outputs/cuda-validation/alias-free-investigation`.
- CUDA generation is reproducible; shell validation and a stubbed `test2.sh`
  run verify three distinct checkpoint paths, preserved continuation inputs,
  paths containing spaces, and the corrected 1,008-frame duration summary.
- A 128×128, 22-frame, two-step run using `inputs/1.jpg` saves a completed
  checkpoint before decoding. A deliberate MP4 output failure retains that
  state; restarting it preserves the final AV latents and writes a valid
  22-frame stereo MP4. The failure/recovery pair takes 73.43 seconds.
  An initial FL2VA-only setup hit its 180-second cap; recovery qualification
  uses the Ref2VA setup matching `test2.sh`.

`test2.sh` now saves a completed `.h3sample` for each segment before decoding,
in addition to its existing `.h3av` outputs. The long three-segment render was
not repeated. Synthetic-latent and two-step functional checks are not visual
or audio quality evaluations.

## Environment

SSH access and hardware were verified before development. The device is an
NVIDIA RTX PRO 6000 Blackwell **Server Edition**, SM120, reporting 97,887 MiB
VRAM. The node runs Ubuntu 24.04.3, driver 595.91.07, CUDA 12.8.93, and GCC
13.3.0, with an AMD EPYC 9335 CPU. The cgroup v1 memory limit is
282,999,996,416 bytes; the host's reported 2.2 TiB is not the container limit.
The runtime snapshot reports PCIe 5.0 ×16 and a 600 W power limit.

`/path/to/models` was empty at first inspection. The downloaded model was found at
`/path/to/models/MiniMax-H3` on the same root filesystem and linked as
`/path/to/models/MiniMax-H3`. Source, build outputs, temporary files, compiler
caches, and generated artifacts remain under `/path/to/h3.c`.
The remote helpers now derive temporary/cache locations from `H3_CUDA_DIR`,
with an optional `H3_CUDA_STORAGE_DIR` override, and record cgroup v1/v2 limits.

## Interpreting the generated videos

The directory mixes visual samples with small functional test artifacts.
`feature-*.mp4` uses the original model at **128×128 and only two denoising
steps**, without a Turbo LoRA. Image/audio cases contain 22 frames; the three
video-reference cases contain 56. These settings are for inexpensive execution
coverage and are not suitable for judging generation quality. The inspected
frames are severely distorted. All nine files nevertheless decode completely
without FFmpeg errors and match their generic-CUDA counterparts byte for byte.
The inspection record is `feature-review/audit.json` in the local evidence tree.

A passing feature test means generation completed and frame geometry and audio
format passed assertions. Paired exact-output comparisons show the SM120 tuning
preserves the generic output; they do not establish that either output has good
visual quality. Full-quality visual acceptance of every reference mode has not
been performed. The tiny workloads alone do not rule out a shared defect.

`full.mp4` is the 864×480, 141-frame, 20-step Ref2VA visual sample discussed below.
`chain-*.mp4` uses 256×256 and 20 steps. `resume-full*.mp4` still uses 128×128
despite its name; `resume-continuation*.mp4`, `cache*.mp4`, and the reuse/core/
reduction artifacts are also functional tests with deliberately small workloads.

## Generic BF16 validation

The generic numerical source SHA-256 is
`827e91148b9545dd71bc3659dc96412159fb80e14c63187b7ed4e8287d0e8584`.
No architecture-specific kernel was enabled for this baseline.

Completed checks:

- Linux `make test`: host/container/CLI, all 126 GPU API symbols, storage and
  asynchronous streaming fences, BF16 rounding, sampler, protected prefixes,
  audio, causal GQA, preview, convolution, and attention.
- All 41 checked-in Metal primitive comparisons and 70 tokenizer cases pass.
- All seven released-model component groups pass (23 tensor comparisons).
  The real-text full-DiT video relative L2 is 0.100438299 and audio relative L2
  is 0.022551152, matching the previous CUDA measurements. Video-decoder L2
  is 7.1164e-7; video-encoder L2 is 6.6430e-6.
- The local Mac `make test` passes with Metal access. Its optional external
  real-component fixtures remain absent and are reported as skips.

All 50 transformer blocks now pass a real-weight resident/stream/fallback
comparison. Every output F32 bit matches across these modes:

| Mode | Peak tracked tensor bytes | Streamed bytes |
| --- | ---: | ---: |
| Resident | 38,574,523,160 | 0 |
| Stream | 1,597,895,680 | 39,305,871,360 |
| Automatic resident | 38,574,523,160 | 0 |
| Automatic with injected 3 GiB allocation limit | 1,597,895,680 | 39,305,871,360 |

The final row forces a partial resident allocation failure and verifies cleanup
and streaming recovery. Tensor peaks exclude CUDA context/library overhead.
Reproduce with `H3_TEST_MEMORY_LAYERS=50 make cuda-memory-test`.

The full 864×480, 141-frame, 20-step Ref2VA workload with `2.jpg` and `body1.jpg`
passes in 1,087.00 seconds, including decoding and muxing. It uses 50 blocks,
seed 72, reuse/core-reuse one, and no token reduction. Video and stereo 32-kHz
audio both have the expected geometry and duration; canonical AV checksums and
finite values pass. Sampled first/middle/last frames are coherent without
obvious tile corruption. The DiT profile records 940.988 seconds including
loading, 886.498 seconds of attention, 34.916 seconds of GEMMs, a 39.909-GiB
tracked tensor peak, and zero streamed block bytes. This makes attention the
measured tuning target.

The earlier RTX 4090 run of this workload took 3,313.85 seconds. This is a
comparison of complete nodes, including different storage and memory limits,
and does not isolate the GPU's contribution. Independently generated outputs
also differ: PRO-versus-4090 final video-latent relative L2 is 0.435684, audio
L2 is 0.318589, downscaled RGB mean error is 24.419/255, and decoded PCM L2
is 0.569653. This confirms functional rendering, not numerical or perceptual
equivalence across GPU families. See `generic/full-vs-4090.json` and
`generic/full-review.png` in the local evidence directory.

Generic SM120 qualification is complete, followed by the separately validated
architecture-specific tuning below.

The generic four-segment chain passes all 21 boundary callbacks per segment,
with seeds 72–75 and changed face/body references on the fourth segment.
Segment times are 90.64, 91.67, 91.80, and 90.35 seconds. The first delivers
141 frames/188,000 stereo sample frames; subsequent segments deliver
102 frames/136,000 after the protected 39-frame overlap is trimmed.
The 20-step same-device checkpoint record/restart passes every F32 comparison
(470.99/41.03 seconds, including first-use model hashing in the record run).
The masked continuation checkpoint also restarts exactly (56.67/45.93 seconds).

The generic PRO 6000 → Metal 20-step handoff restores boundary three exactly
and passes all 17 subsequent comparisons. Final combined latent relative L2
is 0.055876361, maximum absolute error 0.497619122, and cosine 0.998440162.
This uses the preserved Metal binary with the matching generic source identity.
The masked PRO → Metal handoff also passes: final combined L2 0.012333397,
maximum error 0.180288553, and every protected video/audio F32 bit preserved.

All nine reference-mode smoke tests, reuse, core-reuse, token reduction, and
cache-pressure restart pass. Reference cases take 40.82–45.19 seconds. Cached runs compare
every latent and RGB byte. The pressure case forces streamed weights before
reserving VRAM, because freeing a large resident cache would otherwise defeat
the low-memory scenario on this card; eviction assertions remain unchanged.
Native fat-binary execution, forced compute_86 PTX JIT, and permanent reference
mode each pass all 41 primitive comparisons.

## SM120 attention specialization

An isolated seven-variant benchmark found that the DiT's fixed 128-wide heads
allow smaller shared tiles and statically indexed register accumulators.
The selected 128-thread variant uses 13,504 bytes of shared memory and no local
accumulator stack, versus 25,792 bytes and a 128-byte stack in the portable
kernel. At 18,225 tokens and 56 heads, head-major output takes 328.484 ms
versus 826.052 ms (2.51×), with every output BF16 byte identical. Rotated
measurement order and warmed medians are recorded in `tuning/results.jsonl`;
the earlier preliminary run is separate.

The implementation dispatches this specialization only on SM120, for
noncausal BF16 attention with 128-wide heads and at least 512 tokens.
`H3_CUDA_REFERENCE=1` disables it. The unchanged portable tiled kernel remains
the exact-output control. `cuda-sm120-test` exercises both layouts, a short
sequence, the dispatch threshold, tail tiles, and batches; the 56-head timing
case is enabled with `H3_TEST_SM120_BENCH=1`.
The qualified numerical source SHA-256 is
`6893c2aaf0ade88b1eb0aae1b1e2c7108004cb74b771a9562bfb1eac69bd33b7`.
The optimized standard suite, all 23 component comparisons, and 50-block
resident/stream/automatic/fallback checks pass. The full production render also
passes with byte-identical AV state **and MP4** relative to the generic run:

| Measurement | Generic SM120 | Tuned SM120 |
| --- | ---: | ---: |
| Attention GPU events | 886.498 s | 357.967 s |
| Complete DiT phase | 940.988 s | 412.867 s |
| End-to-end render including mux | 1,087.00 s | 554.61 s |

The end-to-end gain is 1.96× on the same node and workload. AV SHA-256 is
`1ecfe3be7070df102caa584c735a69e01190ba897fac622c5a5728be26acb8c8`;
MP4 SHA-256 is
`ab0b608fa5e2ae3721aaca3114d8c135c96f11648b882276df67b864a89ee9c4`.
The tuned continuation segments pass in 74.19, 72.71, 72.96, and 72.90 seconds,
including the changed-reference segment and every protected-prefix callback.
Twenty-step checkpoint record/restart passes in 47.14/41.31 seconds; masked
continuation record/restart passes in 52.46/45.63 seconds. All same-device
boundary comparisons remain exact. All nine reference-mode smoke tests, reuse/core-reuse,
token reduction, and cache pressure also pass. Native fat-binary execution,
forced PTX JIT, and permanent reference mode each pass 41 primitive comparisons.

The final artifact audit verifies **26 AV states and their 26 MP4s** against the
generic run: all 52 file pairs are byte-identical, as is the additional
cache-pressure MP4. Every AV state passes checksum and finite-value validation;
every video has the expected geometry and 32-kHz stereo audio. Available
checkpoint traces also match byte for byte. Evidence is in
`exact-output-audit.json` under the local evidence directory.

The final-source ordinary and masked PRO → Metal handoffs pass, with the same
drift measurements as the generic handoffs above and exact protected prefixes.
Mac `make test` passes, and ordinary/masked Metal restart produces identical AV
states. The reverse Metal/4090 → PRO checkpoint transfers were not rerun in this
qualification; the initial 4090 record retains the earlier two-way coverage.

The primary `/path/to/h3.c` checkout now contains the qualified optimized source
and binaries. Its final native CUDA suite and fat-binary parity pass, and a
fresh masked-checkpoint restart reproduces the qualified AV state and MP4
byte for byte. The fat-object dependency check also confirms that changing
`src/cuda/cuda_dispatch.h` rebuilds the fat CUDA object. Frozen generic source and
binaries are under `outputs/cuda-validation/generic`; the final source and
binary snapshot is under `outputs/cuda-validation/qualified-sm120` on the server.
Final build, device, parity, and restart logs are retained in
`outputs/cuda-validation/pro6000/final-primary` locally.

## Remaining limits

T110 and T115 completed the original PRO 6000 round at 116/120 tasks.
The subsequent [RTX 5090 qualification](cuda-5090-qualification.md) closes T111,
and [RTX 3090 qualification](cuda-3090-qualification.md) closes T107, bringing
the original list to 118/120 at that point. Subsequent T109 and conditional
T114 work is recorded in the [H100 qualification](cuda-h100-qualification.md).
Fat compilation and PTX execution on SM120 do not replace SM90 hardware tests.

The external legacy `misc/fixtures` suite remains unavailable and is not counted
as passing. The separate artificial-input `dit-stress` diagnostic still fails
its unchanged 5% relative-L2 gate: video 0.428498822 and audio 0.866796588.
The generic and tuned diagnostic records are identical. It is reported
separately from the passing released-model acceptance tests.
Independent trajectories on different GPU families still exhibit the drift
documented above; the tuning's exact-output claim is for the paired generic
and optimized runs on this PRO 6000.
