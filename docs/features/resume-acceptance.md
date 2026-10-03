# Sampler resume acceptance: T001–T032

T001–T032 are implemented and tested against [design-resume.md](design-resume.md).
The [checkpoint guide](sampler-state.md) describes the CLI, API ownership, binary
format, and compatibility rules. Tasks T033 onward remain open.

The implementation provides absolute completed-step boundaries, a range-capable
CPU Euler sampler, successful paused results, and atomic, checksummed `.h3sample`
files. Checkpoints preserve generation identity, model/build compatibility,
conditioning and layout, full sigma grids, current AV latents, original noise/RNG
state, continuation metadata, and CPU reuse velocity history. Resume restores
conditioning without reopening the original reference media or continuation source.

## Acceptance evidence

Validation ran on an Apple M4 Max with 128 GiB memory, using the local BF16 model,
all 50 DiT blocks, 256×256 geometry, 90 requested frames, a 20-step schedule, and
`H3_CPU_SAMPLER=1`. Full-model jobs ran sequentially.

| Check | Result |
| --- | --- |
| Pinned pre-change T2VA, Ref2VA, and 39-frame masked continuation | All three post-refactor runs matched captured conditioning, final AV, `.h3av`, and MP4 bytes |
| T2VA, image Ref2VA, hard continuation, reuse 2, reuse 3, bridge continuation, mixed image/video/audio Ref2VA | All seven uninterrupted → pause at 4 → process restart → completion comparisons passed |
| Per-transition oracle | Every paused boundary 0–4 and resumed boundary 4–20 matched the uninterrupted 21-boundary AV trace |
| Self-contained conditioning | Resume succeeded after reference paths and the preceding `.h3av` became unavailable; mixed-reference resume also had no access to its audio/video files |
| Preview and completion | Preview left checkpoint bytes unchanged and contained video only; completed `.h3av` and MP4 bytes matched uninterrupted output |
| Final-build CLI | 37 cases passed, including a real 4-of-20 pause, exact sigma/latent checks, silent preview, 90 delivered frames, and subsequent fresh-process completion |
| Host state/codec tests | 904 checks passed across 18 mode/continuation/reuse combinations |
| Adversarial files | 88 cases in 10 test groups passed: bounds, truncation, overflow, duplicate/missing sections, checksums, versions, shapes, compatibility, and optional sections |
| Sanitizers | 904 host checks and all 88 adversarial cases passed under ASAN/UBSAN |
| Existing encoded image fixtures | Face1, body1, face2, body2, and 12 passed exact F32 round-trips: 877,440 values total; 989 checks including the host suite |
| Independent RNG oracle | Python PCG32/Box–Muller reconstruction matched original noise and exact saved RNG state/counts for all seven checkpoints: 1,228,416 normal values |
| Existing `make -j8 test` suite | Host, continuation, bridge CPU/GPU, binary-mask oracle, AudioVAE GPU primitives, and FFmpeg mux checks passed; absent optional legacy fixtures were skipped |

The reuse-3 test resumes at skipped step 4 with evaluated steps 3 and 0 in its
history, exercising both restored velocity tensors. Continuation checks independently
verify the full source-file SHA-256 and hashes of the raw copied video/audio tails
before removing the source. Host tests also verify same-size model edits with
restored mtime invalidate the fingerprint cache, and interrupted writes preserve an
existing checkpoint destination.

Sanitizer coverage applies to sampler state/file, AV state, and host code; the
full-model GPU kernels were not sanitizer-instrumented. The image fixture check
uses existing encoded latents and does not decode or generate new media from 12.

## Baseline and build provenance

The baseline is commit `7f4aa56e7d4048116e6f1ffcb39d27ee7c629cd3`.
[resume-baseline.json](resume-baseline.json) records original commands, input hashes,
and output hashes. The pinned implementation's cold multimodal encoder can vary
between processes, so paired baseline tests replay captured conditioning in isolated
instrumented source copies. Production resume uses its serialized conditioning.
These tests certify sampler/finalization parity given identical conditioning, not
independent cold-encoder determinism.

The seven-case matrix retained its executable, shader, and complete source snapshot.
Its recomputed source/compiler-options hash matches the identity embedded in its
checkpoints:
`6f79b821fe4e9d89d56b765b1c30f71d71670abddc63d40da3ddabcbb91450ad`.
Subsequent changes tightened configuration validation and environment encoding,
fixed pause-preview frame delivery, restored profiling counts, and adjusted build/test
targets. They did not change sampler arithmetic. The paired baseline build precedes
only the final profiling adjustment.

The final CLI pause and fresh-process resume, host/adversarial/sanitizer checks, and
existing test suite exercised the final implementation. The CLI checkpoint's build
identity was independently checked against the current source/compiler-options hash:
`73cd2395c48eb890148e0d1cd65beb91d367a841abef07a28134896bfa5a8c43`.
Older matrix checkpoints require their retained matching executable.

Detailed local artifacts are under `outputs/resume-validation/` (git-ignored):

- `validation-summary.json`, `followup-results.json`: final results and commands.
- `baseline/original.json`, `baseline/current.json`: baseline comparisons.
- `suite/results.json`, `suite/build/source-audit.json`: seven-case hashes and build audit.
- `suite/*/oracle.trace`, `suite/*/resumed.h3av`, `suite/*/resumed.mp4`: exact comparison artifacts.
- `cli/step4.h3sample`, `cli/resumed.h3av`, `cli/resumed.mp4`: final-build CLI artifacts.
- `unit-tests.log`, `sanitizers.log`, `encoded.log`, `noise.log`, `make-test.log`: remaining checks.

## Reproduction and scope

Run the commands in the [checkpoint guide](sampler-state.md#tests); use a fresh
`--output` directory for another real-model matrix so its frozen artifacts are
retained. The original baseline command requires the pinned commit and local models.
The encoded-fixture and mixed-reference tests also require the existing bridge
validation assets under `outputs/`.

Exact resume currently requires matching model contents, engine build, GPU family,
backend capabilities, and numerical environment. CPU whole-denoiser reuse 1, 2, and
3 is tested. GPU-state sampling, custom `H3_REUSE_STEPS`, core reuse greater than 1,
and token reduction are rejected for checkpoints. Their state support, prepared
DiT caches, broader pause boundaries, multi-resume chains, and cross-backend
certification belong to later tasks. No M5 or CUDA parity is claimed here.
