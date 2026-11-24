# CUDA duration contract

Native CUDA video accepts up to **736 aligned frames at 24 fps (30.667 s)**.
Metal retains **362 frames (15.083 s)**. Alignment remains upward to `5+17*n`;
`--frames 720` and `--seconds 30` therefore produce 736 frames. A request whose
aligned result exceeds the backend ceiling fails; it is never clamped.
`--seconds 20` produces 481 frames, slightly more than 20 seconds. Scripts
retain their 22-frame minimum; the lower-level native API retains its existing
five-frame shape. The duration extension does not alter these minimums.

The same limits govern the CLI, API, HTTP native options, `--render`, continuation,
checkpoint resume, clean source capture and full/preview saved-state delivery.
The SGLang target adapter accepts 4–736/24 seconds on CUDA and its original
4–15 seconds on Metal. Its geometry policy and explicit-native precedence remain
unchanged. Input references still obey the independent 2–15-second media rules.

Above 362 frames, the qualified policy is dense BF16, all 50 blocks, default
attention, reuse/core-reuse 1, without LoRA, token reduction, quantization or
adaptive caching. Unsupported optional modes fail explicitly. First/last anchors
and supported Ref2VA image/audio/video inputs remain available, with their
existing mutual exclusions. Global script references, local additions, comma
lists, image-size inheritance and seed offsets keep their existing semantics.

## Temporal and state boundaries

| Backend | Output ceiling | Video latent T | Audio latent T | Largest legal context |
| --- | ---: | ---: | ---: | ---: |
| Metal | 362 | 107 | 603 | 345 |
| CUDA | 736 | 217 | 1,227 | 702 |

Contexts are `39+51*k`, must fit the source, and must be shorter than the aligned
target. Hard and bridge continuation preserve their exact protected rows,
existing noise initialization, audio alignment and suffix trimming.

AV schema 3 and presentation schema 9 remain unchanged. Their existing frame
and dimension fields express the longer shapes. Old AV readers reject the new
shapes; current Metal readers reject them before tensor allocation. Sampler and
clean-source envelope schema 2 remain unchanged. Their **required section 15
uses version 2 above 362 frames**, even without continuation. Its bridge arrays
hold 217 video and 1,228 audio class bytes. Legacy section 15 version 1 still
writes exactly 107/604 bytes, so short-state wire representation is preserved.
Old readers reject the required section version before interpreting payloads.
Build, model, execution and provenance checks remain mandatory for resume.

Full VAE decoding keeps 22-frame chunks, five-frame overlap and ordered bounded
stitching; increasing duration does not retain the entire RGB movie. Preview
frame-count helpers use the same backend limit. Generated audio follows the
actual target audio T; reference audio encoding retains its 15-second ceiling.
CUDA decoder convolution planning and scratch allocation share a 3 GiB
workspace ceiling. The pinned FFT plan needs just over 2 GiB at longer
durations; scratch is allocated only as required, with normal memory admission.
`make test-cuda-audio-plan` queries every decoder convolution shape for all
aligned durations through 736 without allocating or decoding movie tensors.
See the [audio workspace correction](long-video-audio-workspace.md).

## Resource and connected-operation limits

Frame support does not waive canvas, sequence, memory or indexing checks.
Native CUDA retains 1920×1088 (including transpose); Metal's canvas limit is
unchanged. The one-million-row CUDA sequence ceiling remains in force.
Actual text/reference rows enter resource admission. Long legacy-size canvases
also receive sampler/delivery host-memory checks, and streamed weight placement
keeps the existing device headroom.

The learned upscaler accepts the longer CUDA time dimension for K=0/2/3/4 where
its existing shape and sequence limits allow it. K=0 still passes the shared
sequence planner. A 736-frame 1920×1088 native target has 442,680 video rows;
its 3840×2176 upscale has 1,770,720 video rows and is rejected, even with unlimited
VRAM. Metal rejects long source/upscale execution, including K=0.

## Audited limits

| Location | Output-duration handling or retained independent limit |
| --- | --- |
| `src/geometry.h`, `src/host.c` | Shared backend ceilings, aligned time conversion, checked memory/index plans; huge audio/encoder counts avoid signed overflow. |
| `src/engine.c`, `src/h3cli.c` | Effective request/backend checks, long dense policy, LoRA rejection and host admission. |
| `src/cli/render_plan.c`, `src/server/sglang.c` | Script and target-duration bounds before workers/model work. |
| `src/sampling/av_state.c`, `src/media/presentation.c` | Existing dimensions, byte counts, file lengths, trim samples and checksums express extended CUDA states. |
| `src/sampling/sampler_file.c`, `sampler_state.c` | Backend metadata preflight, required temporal section version and authoritative shape validation. |
| `src/sampling/bridge.h`, `bridge.c` | Expanded class arrays; backend-limited context construction; exact prefix/bridge indices. |
| `src/vae/video_vae.c`, `tiny_vae.c`, `src/media/decode.c` | Backend temporal guards for streamed full decode, preview/shared frame counts and owned/file decode. |
| `src/vae/audio_vae.c` | Generated decode follows target T; the 15-second audio **encoder** guard is unchanged. |
| `src/sglang/sglang_media.c`, `src/media/refvideo.c` | Soundtrack helpers accept a long target; input duration/crop caps remain independent. |
| `src/upscale/upscale_network.c`, `upscale_plan.c`, `upscale_state.c`, `upscale_refine.c` | Extended time guards, dense source policy, target sequence checks and refinement host admission. |
| `src/denoise/adaptive_cache.*`, `approximate.c` | 107/604 class storage and 362-frame approximate-context cap intentionally retained; extended-duration modes cannot reach these unqualified paths. |
| `src/host.c`, `src/cuda/gpu_cuda.cu` | Positional coordinates are generated from actual layout, with no new temporal window. Existing size_t tensor offsets, sequence ceiling and independent MLP slabs remain. |

## Validation scope

Use `make test-long-video-host` for bounded metadata, overflow, resource,
bridge-index, CLI/script and HTTP-contract checks without model work. The fixed
`tests/long_video_campaign.py` entry point checks each runtime plan, references,
saved-state dimensions, script merges, descendant process memory and final media.
Script codec/overlay preflight uses the first active segment’s canvas.
All development videos, including those probes, are 256×256 and at most 20 seconds: the representative
native case is 464 frames; continuation uses 447→464 with 447 context and a
17-frame delivered suffix. Real-model cases use two evaluations and all 50
blocks. The monitor retains the 60 GiB capacity ceiling and physical-device
reserve. Tests are serial on each GPU.

736-frame execution and all larger-canvas long execution are **host-only** in
this milestone. No learned 2× runtime test is performed: its output would
violate the 256×256 test envelope. Motion and scene consistency are excluded.
The user's separate execution request authorizes the unchanged 204-output CUDA
regression as the final gate, including its historical 640×480 fixture.
See [acceptance evidence](long-video-results.md) and [capacity estimates](long-video-memory-estimates.md).
