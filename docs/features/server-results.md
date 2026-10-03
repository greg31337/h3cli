# Native queued server qualification

Updated 2026-09-29. The native server is implemented. **34/34 tasks are complete**.
All required M4 cases, including M04 at the approved 256×256/56 frames/two steps,
have passed. The complete recorded SGLang parity regression passed **204/204
hashes** on the PRO 5000, with the original golden manifest unchanged.

[Usage/API guide](server.md), [design](design-server.md), [task checklist](server-tasks.md).
All server render/lifecycle qualification ran on the local M4. The user then
requested the existing 204-output recorded SGLang parity regression on the
PRO 5000. Its CLI arithmetic coverage is separate from CUDA server qualification.
No live SGLang oracle is run and the golden state is unchanged.

## Provenance and reproducibility

Baseline commit: `ea96baaffff5110216e175adaf21347dd074fa3f`.
The [baseline manifest](../../outputs/server-validation/run01/baseline.json)
records 719 source-file hashes, 494 model/component metadata entries, the M4
Mac Studio, macOS 26.6.2 (25G83), Darwin 25.6 arm64, and Apple clang 21.0.0.
Configured models were the existing `models/MiniMax-H3`, image VAE, preview VAE
and latent-upscale components. No model was downloaded for this campaign.

Compatibility is pinned to SGLang
[`7ee7bef79decaf78f7669994f74e6f75c7784c14`](https://github.com/sgl-project/sglang/tree/7ee7bef79decaf78f7669994f74e6f75c7784c14).
Small authored fixtures and their upstream paths are in
[sglang.json](../../tests/fixtures/server/sglang.json); fetched source snapshots
are in the ignored `outputs/server-validation/run01/upstream/` directory.
CivetWeb is the unmodified v1.16 subset: 11 files verified against
[the hash manifest](../../third_party/civetweb/manifest.json), with its upstream
MIT license. SQLite/libcurl are supplied by the macOS SDK. Linux setup dependency
changes were checked locally; a clean Linux build was not run.

Exact submitted bodies, job IDs, effective metadata, per-variant worker hashes,
artifact digests, prior attempts and client times are in
[results.json](../../outputs/server-validation/m4/results.json).
[Normalized requests](../../outputs/server-validation/m4/normalized-requests.json)
export the typed requests actually sent to workers. Models/inputs use local
managed paths in this private evidence; API responses use asset/artifact URLs.
The campaign spans several implementation builds, recorded per result. M01 was
rerun on the final production binary, together with its same-build CLI control.
[Final inventory](../../outputs/server-validation/run01/final-inventory.json)
and [final source manifest](../../outputs/server-validation/run01/final-source.json)
identify the delivered files and binary.

Reproduction commands, from the repository root:

```sh
make -j8 all bin/server_artifacts
make test
make test-server-sanitize
make test-server-http-sanitize
python3 tests/server_dependency.py
# Serial model work, including the approved 56-frame M04 exception.
python3 tests/server_m4.py
python3 tests/server_artifacts.py --http
```

The M4 driver sets `H3_TEST_MAX_EVALUATIONS=6` only in its child environment,
uses loopback on an ephemeral port, and shuts down its server on exit. Its
current per-variant deadline is 900 seconds. Earlier workflow runs used an
1800-second timeout, but every invocation finished within 900 seconds; the
longest was the 514-second cold LoRA job. Production defaults remain 3600
seconds and the native quality presets retain their normal step counts.
The evidence stays under gitignored `outputs/server-validation/`.

## Host and service validation

| Check | Result / evidence |
| --- | --- |
| CLI + library + service build | Pass; [build log](../../outputs/server-validation/run01/build-final.log) |
| Existing complete local `make test` | Pass; [log](../../outputs/server-validation/run01/local-suite-final.log). Includes current host/CLI, state, conditioning, geometry, Metal/operator, media and oracle checks; no CUDA regression execution |
| Request/registry contract | 497 cases, all 111 original options accounted for; [coverage](../../outputs/server-validation/contract/coverage.json) |
| Request ASan/UBSan | Same 497 cases pass; [log](../../outputs/server-validation/run01/contract-qualification-sanitize.log) |
| HTTP/queue/worker ASan/UBSan | 21 lifecycle tests pass in 91.33 s; [log](../../outputs/server-validation/run01/http-qualification-sanitize.log) |
| Vendored dependency | 11 CivetWeb files match pinned hashes |
| Recorded CUDA SGLang parity | **204/204 hashes pass** in 117.59 s after build (216.10 s including isolated build); [result](../../outputs/server-validation/cuda-parity/result.json), [verification](../../outputs/server-validation/cuda-parity/verification.json) |
| CUDA regression runner host checks | 8 tests pass; [log](../../outputs/server-validation/run01/cuda-gate-host.log) |
| Real artifacts | 14 current native states and 43 MP4/PNG/PPM files pass native loaders / full media decode; [audit](../../outputs/server-validation/m4/artifact-validation.json), [log](../../outputs/server-validation/run01/artifacts-approved-56.log) |
| Cleanup | Download-audit service reaped, port closed, zero remaining M4 leases; final process audit found no h3cli servers/workers or FFmpeg children |

The HTTP suite uses a separately compiled **fake worker**. Its outputs are
explicitly labeled fake and never count as model evidence. Tests cover FIFO
and a single inference worker, concurrent admission, fan-out, exact seeds,
JSON/multipart/SDK normalization, idempotency, auth and path constraints,
upload/lease/deletion behavior, private/metadata URL rejection, SSE disconnects,
log/body/storage/queue caps, deadlines, cancellation, crashes, failed database
writes, corrupt/unsupported schemas, model-metadata changes, symlinks, bundle
corruption and recovery before/after the publication rename. The production
binary contains no test-worker environment switch. Sanitizer instrumentation
covers the new service/request/CLI and vendored HTTP code; the existing engine
library is validated by its normal local tests and the real matrix.

## Recorded CUDA parity follow-up

At the user's request, the existing complete gate ran on the authorized RTX
PRO 5000 72 GB / SM120, driver 595.91.07, using CUDA 13.0.3 (`nvcc V13.0.88`)
and the server's qualified runtime/media environment. It includes all recorded
component probes and the complete 640×480/124-frame/six-evaluation C0 render.
The gate compared every generated artifact directly with the unchanged recorded
golden hashes; no live SGLang process, case selection or tolerance was used.

The isolated source snapshot exactly matches the current local code, tests,
scripts and vendored files. SQLite 3.45.1-1ubuntu2.8 and curl 8.5.0-2ubuntu10.15
development files were extracted into a private test prefix, matching the
existing system runtimes. No system package or existing checkout was replaced.
The Linux CLI, including its linked server modules, compiled successfully;
this is not a CUDA HTTP-server render qualification or a clean-system setup test.

| Identity | SHA-256 |
| --- | --- |
| Tested source | `4302af383ac5b1c0b5b81580007a2ec3a6f8b5b01026f178dbba80e085ec7de8` |
| CUDA CLI binary | `0306f26ead52130330b6f853b3986ebdd7bd18018c644f87e675ca975c151a68` |
| Original golden manifest | `fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3` |

Exact setup/launch commands are in the
[driver script](../../outputs/server-validation/run01/run-cuda-parity.sh),
individual probe/render commands in
[commands.json](../../outputs/server-validation/cuda-parity/commands.json), and
the complete build output in [build.log](../../outputs/server-validation/cuda-parity/build.log).
After the run there were no remaining CUDA compute processes or inference/codec
workers; GPU usage returned to 22 MiB. See the
[cleanup record](../../outputs/server-validation/cuda-parity/cleanup.log).

## Real local M4 matrix

Frozen prompt: “A red wooden toy boat floating on a quiet pond. Gentle ripples
and soft birdsong.” Seed 42. Default geometry is 256×256/22 frames/two native
evaluations. The canonical, continuation, upscale and six-step quality exceptions
are listed below. Videos are 24 fps; full videos contain stereo 32 kHz audio.
The native paused preview deliberately omits audio. All files were fully decoded
with FFmpeg, independently probed, and matched against advertised SHA-256 hashes.

Times are seconds. Queue/run/total come from the durable service timestamps;
total starts after admission/preflight. Client total also includes submission
and polling latency (one-second polling). These are one-shot functional checks,
not performance benchmarks. Downloads are local evidence links.

| Case | Outcome | Queue | Run | Total | Client total | Artifact | Coverage |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| M01a | completed | 0.00 | 24.21 | 24.21 | 24.24 | [video.mp4](../../outputs/server-validation/m4/M01a/video.mp4) | Conflicting SGLang settings overridden; CLI MP4/AV bytes match |
| M01b | completed | 24.19 | 24.61 | 48.80 | 49.51 | [video.mp4](../../outputs/server-validation/m4/M01b/video.mp4) | FIFO second video; waited for M01a |
| M02 | completed | 0.00 | 39.63 | 39.63 | 40.34 | [video.mp4](../../outputs/server-validation/m4/M02/video.mp4) | SGLang only; 107 frames, extra-high, 3 sigma points → 2 evaluations |
| M03 | completed | 0.00 | 27.12 | 27.12 | 27.19 | [video.mp4](../../outputs/server-validation/m4/M03/video.mp4) | First/last uploads; replaced forbidden first anchor |
| M04 | completed | 0.00 | 39.64 | 39.64 | 40.21 | [video.mp4](../../outputs/server-validation/m4/M04/video.mp4) | Approved 56-frame mixed image/silent-video/audio references; native list replaces SGLang list |
| M05 | completed | 0.00 | 11.19 | 11.19 | 12.14 | [video.mp4](../../outputs/server-validation/m4/M05/video.mp4) | Conditioning bundle reimport; preview VAE, 22 frames + live preview artifacts |
| M06-pause | completed | 0.00 | 21.30 | 21.30 | 22.19 | [video.mp4](../../outputs/server-validation/m4/M06-pause/video.mp4) | 4-step schedule stopped after 2; silent preview + checkpoint |
| M06-resume | completed | 0.00 | 7.34 | 7.34 | 8.17 | [video.mp4](../../outputs/server-validation/m4/M06-resume/video.mp4) | Uploaded checkpoint bundle; remaining 2 evaluations |
| M06-state-only | completed | 0.00 | 20.52 | 20.52 | 21.18 | [only.h3av.h3bundle](../../outputs/server-validation/m4/M06-state-only/only.h3av.h3bundle) | Clean AV and upscale source without decode |
| M06-decode | completed | 0.00 | 3.04 | 3.04 | 4.05 | [video.mp4](../../outputs/server-validation/m4/M06-decode/video.mp4) | Uploaded AV bundle; decoder-only video |
| M07-source | completed | 0.00 | 26.89 | 26.89 | 27.21 | [video.mp4](../../outputs/server-validation/m4/M07-source/video.mp4) | 39-frame continuation source |
| M07-hard | completed | 0.00 | 30.20 | 30.20 | 30.22 | [video.mp4](../../outputs/server-validation/m4/M07-hard/video.mp4) | 56 generated/context frames; 17 delivered frames |
| M07-bridge | completed | 0.00 | 29.98 | 29.98 | 30.24 | [video.mp4](../../outputs/server-validation/m4/M07-bridge/video.mp4) | Same trim; one bridge evaluation |
| M08-inspect | completed | 0.00 | 0.01 | 0.01 | 1.02 | [log](../../outputs/server-validation/m4/M08-inspect/worker.log) | Native source inspection |
| M08-zero | completed | 0.00 | 1.55 | 1.55 | 2.04 | [zero.h3av.h3bundle](../../outputs/server-validation/m4/M08-zero/zero.h3av.h3bundle) | 512×512 AV state; zero refinement |
| M08-refine | completed | 0.00 | 29.48 | 29.48 | 30.28 | [video.mp4](../../outputs/server-validation/m4/M08-refine/video.mp4) | 512×512/22, 2 refinement evaluations; audio latent unchanged |
| M09-still | completed | 0.00 | 28.64 | 28.64 | 29.22 | [image.png](../../outputs/server-validation/m4/M09-still/image.png) | 256×256 PNG and saved still latent |
| M09-decode | completed | 0.00 | 8.66 | 8.66 | 9.10 | [image.png](../../outputs/server-validation/m4/M09-decode/image.png) | Uploaded still bundle; PNG bytes match |
| M10-lora | completed | 0.00 | 514.16 | 514.16 | 514.27 | [video.mp4](../../outputs/server-validation/m4/M10-lora/video.mp4) | Turbo v4 step600 EMA, scale 0.5; includes cold CoW fold |
| M10-high | completed | 0.00 | 29.55 | 29.55 | 30.27 | [video.mp4](../../outputs/server-validation/m4/M10-high/video.mp4) | Standard quality high; Metal reuse 2, 6 evaluations |
| M10-fast | completed | 0.00 | 22.87 | 22.87 | 23.19 | [video.mp4](../../outputs/server-validation/m4/M10-fast/video.mp4) | Native fast-preview; reuse 3 + preview VAE, 6 evaluations |
| M11 | completed | 0.00 | 23.80 | 23.80 | 24.20 | [video.mp4](../../outputs/server-validation/m4/M11/video.mp4) | Metal, steel-routed FP16, reference tier |
| M12-cancel | cancelled | 0.00 | 2.36 | 2.36 | 3.03 | — | Expected cancellation after the worker started |
| M12-next | completed | 0.00 | 23.06 | 23.06 | 23.21 | [video.mp4](../../outputs/server-validation/m4/M12-next/video.mp4) | Successful generation following cancellation |

M01's MP4 and `.h3av` are **byte-identical** to its ordinary CLI control; the
exact CLI argv is recorded in `results.json`. M03's first/last frames match
their corresponding center-cropped anchors: mean absolute pixel differences
were 7.26 versus 101.71 for the first frame, and 8.61 versus 108.31 for the last.
This checks anchor order/placement, not upstream numerical parity.

M05 reimported a conditioning bundle and published all 22 frame files plus a
live preview. M06 reimported checkpoint/AV bundles and restored omitted settings
through the native loaders. M07 generated 56-frame states with 39 context frames
and delivered 17 new frames; the presentation sidecars passed fingerprint
validation. M08 preserved the source audio latent **byte-for-byte** in both
zero- and two-refinement outputs. M09's latent decode reproduced the original
PNG exactly. LoRA output includes its provenance sidecar and transfer bundle.

Worker peak RSS reached approximately 36.32 GiB. The cold LoRA cache occupies
about 61.73 GiB of logical managed storage, counted against the default 64 GiB
quota even when CoW sharing is available. The cache and media have been retained
as evidence/reusable artifacts. There is no automatic eviction of live inputs.

## Earlier failures and their resolution

M04 initially failed with the planned 22 generated frames, using one- and
two-second reference clips. Metal caps reference decoding to the generated
length and rejects fewer than 48 decoded frames. The user approved a 56-frame
exception; the same image/silent-video/audio request then completed in 39.64 s.
The output is 256×256, 56 frames at 24 fps, with stereo 32 kHz audio. The native
worker reports two evaluations and 74,400 delivered PCM samples. The reference
video supplies 48 normalized frames (39 VAE frames). Its effective request
contains exactly three ordered references; the replaced forbidden URL is absent.
[Earlier failure](../../outputs/server-validation/m4/M04/prior-attempt-2.log)
and [successful worker log](../../outputs/server-validation/m4/M04/worker.log)
retain both outcomes. SRV027 is now complete.

An initial M11 submission paired the reference tier with an
unsupported BF16 kernel configuration. Native preflight rejected it before
admission; the corrected `steel-routed` / `fp16` pair passed. Historical attempts
and logs remain in the evidence directory. The paused preview's silent-audio
expectation was also made explicit in the artifact validator.

Host fixtures cover optimization mappings that are unavailable or deliberately
not qualified on M4. Native combination guards remain authoritative. CUDA
attention, quantization, adaptive caching and SubBlock were not executed here.
The service is a documented SGLang API subset with native quality policies and
cold model loading per variant; it does not implement Cache-DiT or claim to be
a drop-in replacement for every upstream diffusion-server capability.
