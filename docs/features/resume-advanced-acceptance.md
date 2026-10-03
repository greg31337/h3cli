# Resume acceptance: T033–T070

T033–T070 are implemented and validated on the Apple M4 Max as of 2026-09-15.
All required checks in the requested M4-only scope passed.

The implementation follows [design-resume.md](design-resume.md). Usage, ownership,
format details, and the scouting workflow are in the [checkpoint guide](sampler-state.md).
The user requested M4-only validation. GPU-state tests therefore run explicitly on
the M4 Max; no M5 or CUDA certification is claimed. The same harness has an actual
M5 hardware gate for a later M5 run.

## Implemented behavior

| Tasks | Implementation and verification |
| --- | --- |
| T033–T034 | Capture the resolved evaluation bitmap, including custom schedules; compare restarts on both sides of skipped evaluations |
| T035–T038 | Synchronize Metal work, export canonical F32 AV samples, and preserve/import packed native BF16 GPU velocity history with explicit evaluation indices |
| T039 | Retain the M4 CPU quality oracle: BF16 model, all 50 blocks, reuse/core reuse 1, token reduction off |
| T040–T043 | Store core forward count, ready flag and full-sequence BF16 residual; preserve resolved token-reduction settings and topology; restore state before sampling |
| T044–T050 | Import optional native BF16 refined text and AdaLN caches, reuse a matching live DiT, or rebuild from serialized conditioning without running reference/text encoders |
| T051–T055 | Enforce checkpoint authority, allow output/display overrides, keep pause previews silent and immutable, and retain normal finalization and prefix trimming |
| T056–T057 | Expose/log checkpoint SHA-256, boundary and resume count; report sampler mode, geometry, sigmas, prefix sizes, payload sizes and persistence timing |
| T058–T067 | Test five restart boundaries, repeated resumes, unavailable references, mixed media, 141-frame continuation, downstream segment 3, corruption, and optional-cache fallback |
| T068–T070 | Provide a gated M5 test command, distinguish strict same-path guarantees from future compatible/cross-backend modes, and document fixed-geometry 20/50-step scouting |

Core-history tests stop at steps 3 and 5 as well as 4. At 3 and 5, the next
transition must consume the saved residual, making these tests sensitive to
missing or reset history. Core residuals retain native BF16 words with shape
`[full_sequence,5376]`; every completed Euler boundary has full topology even when
token reduction was used inside the block stack.

Prepared caches hold refined text, 50 block AdaLN tensors, and final AdaLN.
RoPE and row/modulation maps are rebuilt from exact layout, tags, masks and sigmas:
they require no model projections. The T2VA fixture's prepared cache is about
361 MiB. Model weights, Metal/MPSGraph objects, pipelines, scratch activations,
VAE decoders and FFmpeg state are excluded. Cache version/key/shape mismatches
fall back to rebuilding; structural corruption remains an error.

Resume provenance lives in the result API, checkpoint diagnostics and logs.
Completed `.h3av` files retain their canonical bytes, preserving exact comparison
and downstream continuation. The checkpoint hash identifies the complete input
file, independently of the container's checksum field.

## Validation protocol

Full-model jobs run sequentially on the Apple M4 Max with 128 GiB memory. Tests use
256×256 geometry, all 50 blocks, 20 steps, and 90 requested frames except for the
141-frame continuation/segment-3 cases with 39-frame overlap.

The matrix covers T2VA, face/body Ref2VA, mixed image/video/audio Ref2VA, hard and
bridge continuation, whole-denoiser reuse 2/3, a custom evaluation bitmap, core
reuse 4, token reduction, and CPU core reuse with token reduction. GPU-state
cases cover ordinary sampling, reuse 3, core reuse 4, and token reduction.
Each comparison checks AV bytes at every observed Euler boundary and final
`.h3av`/MP4 bytes. Same-process tests rewind an already prepared context; separate
processes validate complete restoration. Input references disappear before restart.

The CPU oracle restarts at 1, 2, 4, 10, and 19, then follows `4→8→12→20` and tests
removed, unknown-version, and mismatched-key prepared caches. Segment 3 consumes
both uninterrupted and resumed segment-2 `.h3av` files in the same conditioning
context, checking that downstream generation also matches.

The GPU CLI test has no latent callbacks. It pauses at skipped step 5 with reuse 3,
checks current F32 latents and all four BF16 velocity tensors against the oracle,
then completes through the actual CLI. This also exercises export synchronization
when callbacks are not forcing a wait after every transition.

## Results

| Check | Result |
| --- | --- |
| Model matrix | 15 configurations; 46 fresh-process resume checks and 8 same-process rewind cases passed |
| Exact output | Every compared Euler boundary matched; all completed comparisons had byte-identical final `.h3av` and MP4 files |
| CPU restart boundaries | 1, 2, 4, 10, 19; repeated `4→8→12→20`; all three optional-cache fallback variants passed |
| Reuse history | Reuse 2/3 and custom schedules passed around evaluated/skipped steps; CPU and GPU core reuse passed at 3/4/5 |
| Continuation | 141-frame target, 39-frame overlap, unavailable source state, and downstream segment 3 passed |
| Actual CLI | 41 CPU cases including a silent 90-frame preview and exact fresh resume; GPU skipped-step pause/resume passed without latent callbacks |
| Host and container tests | 1,249 host checks; 114 adversarial container cases in 13 groups; ASAN/UBSAN passed both sets |
| Native GPU tensors | 129 checks; all 65,536 BF16 bit patterns, packed F32 samples, and 51 AdaLN tensors preserved exactly |
| Existing encoded inputs | Face1/body1/face2/body2/12 fixtures passed; 1,334 total host checks including 877,440 fixture F32 values |
| Independent audits | Original noise/RNG matched for 15 checkpoints; 47 checkpoint SHA-256, step and resume-count checks passed |
| Ordinary generation | T2VA, Ref2VA and continuation matched the pinned pre-resume baseline, including conditioning, final AV and MP4 hashes |
| Existing regression suite | `make -j8 test` passed; 11 optional legacy fixture checks skipped explicitly |

The optional skips cover legacy MLX, tokenizer, Qwen, audio and visual fixtures
absent at those tests' configured paths. The full-model matrix above ran against
the installed `models/MiniMax-H3` model and its real media references.

The tested source identifier is
`894edc53bb349443f06fc22db313723dd068e8af0ac6b3c80716f76b45f03dbf`,
with git identity `8fc4e532b2f6f1caf358e350707c770e57f59536` and Clang
`21.0.0 (clang-2100.3.34.2)`. All 50 retained source/driver files match the current
workspace, and the checkpoint's embedded source identifier matches the build.

The standalone CPU token-reduction result is retained from the initial snapshot.
Its only subsequent library changes were two GPU previous-velocity pointer guards;
the CPU path is unchanged. The initial failed GPU trial is excluded from these
results, and all corrected GPU cases passed. The later core-history suite adds
residual-dependent boundaries 3 and 5 to the preliminary step-4 coverage.

## Retained evidence

Local artifacts are retained under `outputs/resume-validation-advanced/`:

- `final-suite/results.json`: broad model matrix, commands and output hashes.
- `core-history/results.json`: residual-dependent CPU/GPU boundaries 3/4/5.
- `gpu-cli/results.json`, `cpu-cli/`, `followup-results.json`: actual CLI checks.
- `baseline/`: current-versus-pinned ordinary-generation comparisons.
- `unit-tests.log`, `corruption.log`, `corruption-sanitized.log`, `sanitizers.log`:
  host, malformed-file and ASAN/UBSAN checks.
- `gpu-tensors.log`: all 65,536 BF16 patterns, packed F32 and 51 AdaLN tensors.
- `encoded.log`, `rng-oracle.log`, `provenance-all.log`: existing face/body/12
  encoded fixtures, independent PCG32/Box–Muller reconstruction, and provenance.
- `make-test.log`: existing regression suite, including optional fixture skips.
- `summary.json`: consolidated passing results and artifact hashes.
- `build-audit.json`, `earlier-cpu-source-audit.json`: source/checkpoint identity
  checks and the precise difference between the initial and final snapshots.

The original ordinary-generation baseline and input hashes are pinned in
[resume-baseline.json](resume-baseline.json). Cold multimodal encoding can vary;
the baseline comparison replays captured conditioning in isolated test builds.
Production resume restores its own serialized conditioning directly.

## Reproduction and limits

Run the commands in the [checkpoint guide](sampler-state.md#tests). Each model
matrix requires a fresh output directory and local model/reference fixtures.
Retained binaries, shaders and source snapshots identify the exact tested builds.
Checkpoints require their matching build and numerical environment.

ASAN/UBSAN covers host state/codec/AV code, not full-model GPU kernels. Encoded
face/body/12 checks operate on existing F32 latents without creating media from 12.
Absent optional legacy fixtures are skipped explicitly. Current certification is
same-path M4 behavior; a forced GPU-state M4 run does not establish M5 parity.
