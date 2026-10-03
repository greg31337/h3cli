# Reference-image sizing validation

Status: **complete: 24/24 tasks, 12/12 functional outputs and 204/204 existing
golden outputs pass**. Tasks are tracked in
[the archived checklist](reference-image-size-tasks.md); behavior is specified in [the design](design-reference-image-size.md).
Only local Apple M4 Max (128 GB) and RTX PRO 5000 Blackwell (72 GB) are used.
CUDA qualification uses CUDA 13.0.3 and cuDNN 9.20. No base-model weight hashes are
computed by this campaign; video source export selects metadata identities.

| Environment | Recorded configuration |
| --- | --- |
| Metal | M4 Max, 40 GPU cores, 128 GB unified memory; macOS 26.6.2; Apple Clang 21.0.0 |
| CUDA | RTX PRO 5000 72 GB Blackwell; Linux 6.8; driver 595.91.07; CUDA 13.0.3 toolkit (`nvcc` 13.0.88); cuDNN 9.20 |

## Scope and implementation audit

The shared host resolver implements `MATCH=0`, `MAX=1`, `HIGH=2`, integer
nearest-even fixed-edge sizing, inclusive 1:4–4:1 bounds and per-image capacity.
Both old sizing symbols delegate to this policy. The engine supplies the same
prepared canvas to vision and visual-VAE encoding on still/video and both
backends. Image order and `<Picture N>` bindings remain unchanged.

The three old aggregate guards (vision admission, CUDA attention-group setup,
CUDA patch projection) have been replaced by per-image and resource checks.
The checked vision plan admits nine 65,536-patch images (589,824 rows) without
promising that every device can execute that request. Two-frame video blocks
use their own geometry and do not inherit the single-image area rule. The
largest vision element-index consumer is the 4,304-wide MLP; cumulative rows
must fit its `uint32_t` element range. Patch, position/RoPE, hidden/QKV, merger
and deepstack offsets then fit that bound. Separate text MLP, presentation span,
VAE and packed DiT validations remain in force.

CUDA packs the padded filter once and reuses input scratch for at most 32,768
projection rows (512 MiB plus 18 MiB filter). Complete image attention groups
are retained; they are not split at projection boundaries. Group replacement,
invalid replacement and cancellation clear stale group state. Tensor release
waits for its GPU consumers through the existing backend lifetime rules.

The 65,536-patch test exposed a stall in MPSGraph D=72 vision attention. Above
32,768 patches, zero-pad heads to 128, use the existing bounded dense BF16
kernel with the original scale, then discard padded output lanes. Every image
retains all its keys. GPU consumers complete before scratch is made purgeable.
Smaller vision requests retain MPSGraph. Unified-memory admission includes
MPSGraph transient copies and padded scratch.

The consumer audit also found the Metal Qwen causal-attention scratch limit.
Its original direct path remains for fitting sequences; larger sequences use
512-key tiles and stable FP32 online softmax across all causal keys. The total
query-index limit is separate from the exposed direct scratch boundary. This
change adds no numerical-parity qualification.

Fresh conditioning identifies `high` and the changed Metal/still `max` geometry
with `reference-image-geometry=2`. CUDA-video `max`, `match`, and text-only
identities avoid unrelated invalidation. Sampler records already serialize the
mode integer; validation now accepts 2, rejects unknown modes, and preserves
prepared geometry without reopening media. Upscale planning retargets only
`match`; `high`, current `max` and legacy stored `max` remain intrinsic canvases.

Unrelated `32768` kernel/linear heuristics, image/reference counts, anchor and
video/audio preprocessing, quantization and approximate attention are outside
this change.

## Reproduction and evidence

All generated inputs and evidence are under ignored
`outputs/reference-image-size/` (with corresponding isolated CUDA campaign
folders). The independent [functional manifest](../../tests/reference_image_matrix.json)
fixes four owned deterministic PNG fixtures, six rows per backend, prompt,
seed 42, two evaluations, 50 blocks, dense BF16, 640×480, and 90 video frames
at 24 FPS. Each row has a 10,800-second deadline; the full frozen golden suite
retains its existing 12-minute post-build deadline.

```sh
make -j8 all test-reference-image test-sampler test-upscale-state \
  bin/conditioning_tests bin/reference_image_gpu bin/reference_image_vision
bin/conditioning_tests
# Metal only: direct/tiled attention dispatch boundary.
make test-memory-gpu
# CUDA only: bounded packing, complete writes, guards, failure/recovery.
bin/reference_image_gpu
# Either backend: real 65,536-patch encoding, finite outputs, cleanup.
bin/reference_image_vision models/MiniMax-H3/Ref2VA/text_encoder 8192 2048
python3 tests/reference_image_matrix.py --prepare \
  --fixtures outputs/reference-image-size/fixtures
python3 tests/reference_image_matrix.py --backend metal --source . \
  --model models/MiniMax-H3 \
  --image-vae models/image-vae/minimax_h3_t1_image_vae_step1597.safetensors \
  --fixtures outputs/reference-image-size/fixtures \
  --out outputs/reference-image-size/metal-matrix
# Use --backend cuda with the qualified CUDA build/environment on the PRO 5000.
python3 tests/cuda_reference_regression.py --source . \
  --model models/MiniMax-H3 --out outputs/reference-image-size/final-gate
```

The matrix records exact commands, binary/fixture/manifest identities, runtime,
RSS/VRAM samples, geometry logs, finite saved values and complete media decoding.
Each process is reaped, including failure/timeout cleanup. Logs and failed
attempts are retained. Projection tests use NaN sentinels and tail guards, not
an expected numerical output. No new SGLang case, oracle capture, tolerance,
cross-backend tensor comparison or quality/performance threshold is introduced.

## Results

Starting revision: `dabbdc05752ad06c5dbe12881074d23d67016085`.
Golden manifest SHA-256:
`fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3`.
Baseline records retain all twelve fixture hashes for final immutability checks.

| Check | Result | Evidence |
| --- | --- | --- |
| Baseline frozen CUDA gate | 204/204, 113.12 s after build | `baseline-gate/result.json` |
| Shared sizing/state gate | 204/204, 114.27 s | `m1-gate/result.json` |
| Capacity gate | 204/204, 113.10 s | `m2-gate/result.json` |
| Extended Metal/code gate | 204/204, 112.86 s | `m3-gate/result.json` |
| Memory/test wiring gate | 204/204, 112.91 s | `m4-gate/result.json` |
| Large Metal vision and host tooling gate | 204/204, 113.15 s | `m5-gate/result.json` |
| Final coherent code/tool gate | 204/204, 112.22 s; source `2e828c3f295a20ef214aa50ddb2903b78e82bbc69fa14151b956884f4826fa68` | `m6-gate/result.json` |
| Host sizing/capacity/cache | 342 checks; CLI mode tests pass | `m3-host.log` |
| Sampler/state | 3,014 native checks; 19 sampler-file and 10 upscale-file tests; 24 upscale CLI, 210 adversarial container, 52 sampler CLI cases pass | `m2-metal.log` |
| Conditioning/backend | 34 checks pass | native test output |
| Metal GQA boundary | 23,841 checks pass | `m2-metal-gqa-retry.log` |
| Metal large vision | 65,536 patches, 16,384 merged tokens; four finite output tensors; small/large/small, cancellation and memory rejection pass; peak explicit tensors 4,255,191,712 bytes | `m5b-metal-vision.log` |
| Metal GPU guards/lifetime | 7,937/16,385-token GQA, 65,536-key vision attention; finite complete writes, guards, repeated context and cleanup pass | `m5b-metal-gpu.log` |
| Host presentation | Geometry and 176 tokenizer/ordinal checks pass | `m5-multi-host.log` |
| Synthetic tokenizer fixtures | 240 multimodal and 209 tokenizer conformance checks pass | `tokenizer-fixture.log` |
| General host/device | 1,777 checks pass | `m3-core-retry.log` |
| CUDA host/state | CLI/library and all affected tools build; sizing, conditioning, sampler/upscale and ordinal tests pass | `m5-build-host.log`, `m5-conditioning.log`, `m5-multi-host.log` |
| CUDA projection/lifecycle | 32,764/32,768/32,772/65,540 rows; complete finite writes, guards, bounded scratch, allocation failure and same-context recovery pass | `m5-gpu.log` |
| CUDA large vision | 65,536-patch real encoding, small/large/small, memory rejection and cancellation pass | `m5-vision.log` |

The final source/tool gate passes with the unchanged manifest and all twelve
fixture identities. Its CUDA CLI SHA-256 is
`475b877c568251a4681629229fe25fd56b064f14863a60afe4c52199d69ef51d`;
all six CUDA functional rows use that binary.
The Metal CLI SHA-256 is
`5b9ca8864271db9b30086b71675ed8bfced6780376f7b4be589a6d0892c73894`.
The closing `final-audit.json` verifies all twelve functional records, media
hashes, fixtures, geometry, finite-state/media checks, deadlines, resource
samples and child-process cleanup. Every input source file matches the final
gate. The sole extra file in the Metal build inventory is the existing ignored
generated `src/metal/native_attention.inc`; regenerating it from the unchanged inputs
produces the same bytes. No source, golden, fixture or parity case was changed
after the final passing gate. CUDA gate/component evidence is under `cuda/`;
Metal evidence is at the local evidence root.

### CUDA functional matrix

All six rows pass: expected ordered geometry, finite saved values, complete
media decoding and reaped child processes. Video outputs contain 90 frames at
24 FPS and audio; still outputs contain one image. Every output is 640×480.
Video states contain 787,200 finite F32 values; still latents contain 28,800.
Evidence is in `cuda/cuda-matrix-2/<row>/` beneath the local evidence root.

| Row | Reference canvases | Total patches | Wall seconds | Peak RSS GiB | Peak VRAM GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| `video-match` | 640×480 | 1,200 | 93.83 | 36.76 | 5.60 |
| `video-high-two` | 2 × 2048×1536 | 24,576 | 111.65 | 36.98 | 6.46 |
| `video-max-two` | 2 × 2720×2048 | 43,520 | 115.78 | 37.16 | 8.40 |
| `video-max-wide` | 8192×2048 | 65,536 | 140.39 | 37.45 | 10.67 |
| `still-high` | 1536×2048 | 12,288 | 174.95 | 0.85 | 9.46 |
| `still-max` | 2048×2720 | 21,760 | 241.68 | 0.95 | 9.39 |

RSS and per-process CUDA VRAM are sampled once per second. Wall time includes
rendering and output validation. These are descriptive measurements from the
functional campaign, with streamed BF16 weights; they are not performance
claims or a comparison with another implementation. Metal's unified-memory
measurements are recorded separately below.

### Metal functional matrix

All six rows pass the same geometry, finite-state and complete media
checks. Evidence is in `metal-matrix-2/<row>/` beneath the local evidence root.
Their reference canvases and output requirements match the CUDA table.

| Row | Total patches | Wall seconds | Peak RSS GiB | Peak sampled footprint GiB | Peak sampled Metal allocations GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| `video-match` | 1,200 | 134.99 | 36.03 | 38.99 | 38.63 |
| `video-high-two` | 24,576 | 814.99 | 36.73 | 42.96 | 42.17 |
| `video-max-two` | 43,520 | 1,833.53 | 36.81 | 46.38 | 45.03 |
| `video-max-wide` | 65,536 | 1,518.28 | 36.86 | 77.12 | 75.36 |
| `still-high` | 12,288 | 307.44 | 36.20 | 38.40 | 37.90 |
| `still-max` | 21,760 | 850.57 | 36.36 | 40.00 | 39.33 |

Metal RSS is sampled once per second. Physical footprint and Metal allocation
values come from the existing profile checkpoints, so they may miss peaks
between checkpoints and are not directly comparable to CUDA VRAM samples.

## Preserved failures and corrections

- The first Metal GQA boundary test exposed missing registration of the new
  pipeline names; registration was added and the boundary test passed.
- An ordinary device-query test initially ran inside the restricted sandbox;
  the same test passed with authorized GPU access.
- The old multi-reference host executable had no Make target. Wiring it exposed
  a mock/real text-encoder symbol collision; the target now links only its host
  dependencies. Its byte-only tokenizer fixture also masked independently
  tokenized prefix boundaries; the expectation now works with the released
  tokenizer as well as the fixture.
- The first 65,536-patch Metal vision attempt was interrupted after over thirteen
  minutes in a GPU submission without completing a layer. A diagnostic rerun
  isolated MPSGraph attention. The bounded dense BF16 path resolved it. An early
  bounded-path run was superseded after review found that scratch needed an
  explicit completion fence before Metal marked its storage purgeable. The
  corrected path passed the full encoder and guard/lifetime checks.
- First Metal/CUDA matrix attempts failed CLI preflight: the existing test
  safety override accepts only 6 or 50, while the runner supplied 2. The runner
  now uses the six-evaluation safety ceiling and still requests exactly two
  evaluations for every row. No failed attempt counts as a matrix pass.

Existing strict sampler build/model compatibility checks remain in force;
geometry preservation does not bypass them. This campaign performs no new
numerical-parity or perceptual-quality qualification.
