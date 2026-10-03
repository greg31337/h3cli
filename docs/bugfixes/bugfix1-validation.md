# Runtime robustness validation on M4

All 56 tasks in `todo.md` are complete. Hardware: Apple M4 Max, 128 GiB.
Baseline: commit `1959b14`, built and captured before editing. No M5 run was
performed, as requested. Raw commands, logs, fixtures, models, and media are
under `outputs/bugfix1-validation/`; structured results are retained in
[bugfix1-validation.json](bugfix1-validation.json).

## Test results

- `make test`: passed, including tokenizer, Qwen scaling, preview, sampler,
  continuation, bridge, reference-video geometry, memory, and Metal tests.
  Ten optional external fixture groups reported their existing skips (MLX,
  released component oracles). Separate real-model regressions below did run.
- `make test-bugfix1-sanitize`: passed under ASan/UBSan, including Metal mapping,
  copied parity, malformed files, shader path precedence, long paths, spaces,
  relative invocation, symlinks, and explicit override errors.
- Tile policy/geometry: **33,336 checks**, normal and ASan/UBSan. All accepted
  canvases keep default 256 and auto within 256..320. Every valid explicit tile
  is honored. Invalid values fail both resident and standalone decoder entry
  points before nonexistent weights/shaders can be opened.
- Mixed ownership: 32 repeated cycles hold mapped F32 and copied BF16 tensors
  from one shard alongside copied tensors from its unaligned equivalent.
  Metal blits/casts return exact values; logical live bytes return to zero and
  all 32 mapping deallocators run. Host reads/writes now drain temporary
  Objective-C buffer references so they cannot retain mappings past tensor free.
- Malformed cases cover truncated headers, overflowing sizes/offsets, out-of-file
  ranges, invalid dtype/shape/length, overlap/gaps/trailing payload, duplicate
  keys, metadata types, UTF-8 and JSON syntax. Reads after file truncation fail.

## Real-model numerical parity

Each generation used seed 72, 128x128, 56 frames, six denoising steps,
`H3_CPU_SAMPLER=1`, and default 256 tiles. The installed binary and shader were
in a path containing spaces; CWD was a separate directory with no shader.

| Case | Final MP4 | Video/audio latent bytes | Complete AV-state file |
| --- | --- | --- | --- |
| Official FL2VA, `inputs/face1.jpg` | Bit-identical to baseline | Bit-identical | Bit-identical |
| Official Ref2VA, `inputs/2.jpg` | Bit-identical to baseline | Bit-identical | Bit-identical |
| Unaligned FL2VA, face, zero-copy requested | Bit-identical to baseline | Bit-identical | Metadata fingerprint differs as expected |

The copied checkpoint contains 29 shards / 144,016,376,465 bytes. Only header
padding changed; the repacker preserved all tensor descriptors and payloads.
The final copied-model AV state has a different model representation signature;
its actual video and audio arrays match byte-for-byte. No comparison waives
existing checkpoint identity validation.

Qwen (80 MiB), DiT (220.5 MiB), VideoVAE (128 MiB), and AudioVAE (80 MiB) weights
were loaded through the shared model weight store from official, unaligned full
checkpoint, page-aligned extracted, and unaligned extracted representations.
All readback SHA-256 values match original payload bytes. All three eligible
released Qwen tensors additionally map and release successfully. Other official
components have no page-aligned tensor starts under this implementation.

## Storage timing and memory

Median of three fresh processes per extracted tensor, warm filesystem cache;
values are **mapped / copied**. Loading timing includes header/store indexing
but excludes shader startup. Ready timing also touches every tensor byte through
readback, avoiding a misleading comparison of lazy mmap setup against eager I/O.
Peak RSS includes the verification readback buffer; persistent physical-footprint
deltas are sampled with only the loaded weight, before allocating readback.

| Component | MiB | Load ms | Ready ms | Peak RSS MiB | Loaded footprint delta MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| text_encoder | 80 | 0.228 / 5.743 | 7.158 / 9.180 | 176.0 / 176.0 | 0.0 / 80.0 |
| transformer | 220.5 | 0.457 / 15.240 | 19.839 / 24.888 | 457.0 / 457.0 | 0.0 / 220.6 |
| video_vae/source | 128 | 0.301 / 9.394 | 11.285 / 14.941 | 271.9 / 272.0 | 0.0 / 128.1 |
| audio_vae | 80 | 0.230 / 5.565 | 7.070 / 9.055 | 176.0 / 176.0 | 0.0 / 80.0 |

Mapped pages are file-backed and reclaimable; copied buffers immediately consume
resident storage. RSS counts mapped aliases and is not equivalent to physical
memory pressure. Readback allocation/driver caches can remain resident after
free; the JSON records all before/load/touch/release snapshots separately. The
mapping deallocator diagnostics confirm that actual file mappings are released.
These are isolated tensor measurements, not a claim about cold-start whole-model
speed. M4 defaults already copy; stricter eligibility can increase M5 memory and
loading cost, and no M5 performance claim is made.

## 960x544 tile quality evidence

A 22-frame translating composite uses `face1.jpg`, `body1.jpg`, and `2.jpg`,
plus flat/striped regions. One released encoder output supplies every decoder.
The baseline-only 512 mode is development evidence, with no production bypass.
Default and auto outputs match their respective pre-change F32 files exactly.

| Mode | Tile | Spatial tiles | Decode seconds | MAE vs 256 | Luminance SSIM vs 256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| default | 256 | 15 | 13.268 | 0 | 1 |
| auto | 304 | 8 | 9.915 | 0.012545 | 0.965953 |
| 320 | 320 | 8 | 11.214 | 0.016255 | 0.942843 |
| legacy-512 | 512 | 4 | 19.242 | 0.054280 | 0.602866 |

The [comparison image](../outputs/bugfix1-validation/quality-960x544/comparison.png)
was visually inspected: 512 produces conspicuous dense grid/quilt distortion
across faces, fabric, and background. It is also slower here. The supported
320 mode differs from the released 256 reconstruction but avoids the severe
512 artifact. These reference-relative metrics document this case; they are
not a universal perceptual score or a claim that 320 equals 256 quality.

## Scope and task coverage

The structured record contains baseline/current source hashes proving no edits
to DiT/schedules, Metal shaders, tokenizer, Qwen math, sampler serialization,
continuation state, bridge, VideoVAE encoder, or AudioVAE. Changes in `src/engine.c` only
validate tile configuration; GPU host access scopes change ownership, not bytes.
Full-suite sampling, tokenizer, continuation, attention, and bridge checks pass.

| Tasks | Evidence |
| --- | --- |
| T001–T014 | Shared helper, shader precedence/path tests, real arbitrary-CWD CLI runs |
| T015–T022 | Parser/range audit, malformed fixtures, captured stage-1 guard and padded parity |
| T023–T031 | Central mapping plan, per-tensor copied fallback, mixed ownership/release; T028 is inapplicable because shard-level fallback was not selected |
| T032–T039 | Byte/Metal parity, actual model weights, full unaligned generation, timings/memory, eligibility audit and documentation |
| T040–T049 | 33,336 policy/geometry checks, direct early failures, exact default/auto decode regressions |
| T050–T052 | 960x544 legacy-512 comparison, visual inspection, updated supported-policy documentation |
| T053–T056 | Combined CLI/checkpoint regressions, release notes, output and source identity evidence |
