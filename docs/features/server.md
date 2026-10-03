# Queued native server

`bin/h3cli --server` accepts asynchronous SGLang-shaped H3 video requests and
native h3cli jobs. It runs one fresh inference process at a time; the HTTP
process handles the durable SQLite queue, uploads, status, events and downloads.
There is no Python inference service and no persistent GPU model residency.
The ordinary CLI and `bin/libh3.a` remain available.

This is a compatible **subset** of SGLang's diffusion video API, pinned to
[`7ee7bef79decaf78f7669994f74e6f75c7784c14`](https://github.com/sgl-project/sglang/tree/7ee7bef79decaf78f7669994f74e6f75c7784c14).
It implements H3 tasks and the queued video lifecycle, not chat, arbitrary
models or upstream Cache-DiT. Server validation uses the local M4. The existing
recorded SGLang parity gate checks CUDA CLI arithmetic separately; CUDA server
execution and a live upstream oracle are not qualified by this work. See the
[design](design-server.md), [results](server-results.md), and
[implementation checklist](server-tasks.md).

## Start and discover

Build with the normal `make -j8`. CivetWeb 1.16 is vendored with its license and
hash manifest; SQLite and libcurl come from the macOS SDK. Linux setup installs
`libsqlite3-dev` and `libcurl4-openssl-dev`. FFmpeg/ffprobe remain required for
media. Startup `--models-path ROOT` selects the shared main/auxiliary root.
It may be empty; accepted jobs prepare missing models before GPU work.
See [automatic model preparation](model-downloads.md#queued-server-jobs).

```sh
./bin/h3cli --server -d models/MiniMax-H3 \
  --server-host 127.0.0.1 --server-port 30000 \
  --server-state-dir outputs/server
curl -sS http://127.0.0.1:30000/health
curl -sS http://127.0.0.1:30000/v1/models
curl -sS http://127.0.0.1:30000/v1/h3/capabilities
```

`--server --help` lists startup flags. Defaults: 32 waiting variants, one active
worker, 21600 seconds for model preparation, 3600 seconds per executing variant,
2048 MiB per uploaded asset and
65536 MiB total managed storage. Override with `--server-queue-limit`,
`--server-job-timeout`, `--server-max-upload-mib` and
`--server-max-storage-mib`. Waiting time does not consume the execution timeout.
`--server-model-download-timeout` controls preparation independently. Model
preparation stays queued and reports byte progress under `h3`; inference workers
are offline. Startup `--offline` also disables URL imports.
The storage limit counts logical bytes, including LoRA/quantization caches;
allow enough space for a folded model even when the filesystem supports CoW.

State must reside on a local filesystem, owned by the server user. A directory
lock rejects a second owner. The default bind address is loopback. Non-loopback
binding requires `--server-api-key-file PATH` (at least 16 printable characters).
With a key configured, send `Authorization: Bearer KEY` on all `/v1/` requests.
Health/liveness/readiness remain public. TLS belongs at a reverse proxy; CORS
and generic filesystem serving are disabled.

## One request envelope

Use SGLang-defined fields and, optionally, **one added field**, the string
`h3cli`. It contains the same long/short options as the native CLI. There is
no `h3.options`, `references`, `loras` or `operation` request object.

A small native-shaped request:

```json
{
  "prompt": "A red wooden toy boat floating on a quiet pond.",
  "h3cli": "--width 256 --height 256 --frames 22 --steps 2 --seed 42"
}
```

An upstream-shaped request:

```json
{
  "model": "MiniMaxAI/MiniMax-H3",
  "task": "t2va",
  "prompt": "A red wooden toy boat floating on a quiet pond.",
  "conditions": [],
  "target": {"short_edge": 256, "aspect_ratio": "1:1", "duration_seconds": 4},
  "num_inference_steps": 3,
  "quality": "extra-high",
  "seed": 42
}
```

The second request resolves to 256×256, **107 frames**, 24 fps and **two native
sampler evaluations**. SGLang's `num_inference_steps=N` counts sigma points;
its mapping is `--steps N-1`. Explicit `--steps` retains native evaluation
semantics. Target geometry follows the pinned short-edge/area-cap/nearest-32
rounding policy, including ties to even. Target duration is 4–15 seconds,
aligned to `17n+5` frames. Smaller native durations use `--frames` or `--seconds`.
`auto` uses 16:9, or effective first/last anchor geometry for FL2VA. Native
geometry flags bypass only the target components they replace.

Omitted `quality` retains CLI defaults, including 20 evaluations. Standard
quality values are `lossless`, `extra-high` and `high`, mapped directly to
native `--quality`. Native `high` uses conservative adaptive caching on CUDA
and reuse 2 on Metal; this is not Cache-DiT. `preview` and `fast-preview` are
available only through `h3cli`. Explicitly cap `--steps` in small test requests:
regular quality presets otherwise select 50 evaluations.

The independent SGLang `output_quality` field accepts `default`, `maximum`,
`high`, `medium`, and `low`, mapped to H.264 CRF 25, 0, 5, 22, and 33.
Omitting it uses the production default, CRF 18. Native `--output-quality`,
`--ffmpeg-crf`, or `--lossless-video` overrides that field. See
[output encoding](output-encoding.md) for exact precedence and lossless RGB.

Priority, from lowest to highest:

1. Native defaults.
2. The SGLang `quality` preset.
3. Explicit SGLang fields, such as `num_inference_steps`.
4. A native `--quality` preset in `h3cli`.
5. Individual options in `h3cli`.

Individual options beat presets regardless of order inside the string. Repeated
scalars follow the CLI's last-value behavior; references and LoRAs retain order.
For example, `"quality":"high", "num_inference_steps":51` together with
`"h3cli":"--quality fast-preview --steps 2 --no-preview-vae"` selects two
steps, reuse 3 and the full decoder. A native preset without `--steps` overrides
the lower-priority SGLang step count.

The lexer supports single/double quotes, backslash escaping, Unicode, empty
values, short aliases, `--name=value`, and the two paths following
`--ref-video-audio`. It does **not** execute a shell or expand `$variables`,
`$(commands)`, backticks or globs. Do not include the executable name. Limits
are 65536 string bytes, 1024 tokens and 512 options. Unknown/removed names,
startup flags and private worker flags are errors. Native combination and
backend guards still apply.

## Submit, poll, variants and download

Save the first example as `request.json`:

```sh
curl --fail-with-body -sS http://127.0.0.1:30000/v1/videos \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: boat-001' \
  --data-binary @request.json
# Replace JOB_ID with the returned id.
curl -sS http://127.0.0.1:30000/v1/videos/JOB_ID
curl --fail -o boat.mp4 http://127.0.0.1:30000/v1/videos/JOB_ID/content
```

Submission returns HTTP 200 after durable admission, without waiting for model
execution. Queue exhaustion returns 429 with `Retry-After`. Reusing an
`Idempotency-Key` with the same normalized request returns the existing job;
a changed request is 409. Equivalent JSON key order, multipart text fields and
`extra_body`/`extra_json`/`extra_params` wrappers normalize before admission.
Conflicting wrapper aliases are errors. Multipart's text `h3cli` is parsed
once as flags, without an additional JSON-string decode.

`n` or `num_outputs_per_prompt` requests 1–10 sequential variants. Both aliases
must agree. A scalar seed `s` becomes `s, s+1, ...`; a seed list must match the
variant count. `--seed` replaces either form before fan-out. Exact uint64
values are preserved, and overflow is rejected. Download variant 1 with
`/v1/videos/JOB_ID/content?variant=1`. A later failure leaves earlier committed
variants downloadable but marks the parent unsuccessful.

Video states are `queued`, `in_progress`, `completed`, `failed`; native status
also distinguishes `cancelled` and `interrupted`. `h3` response metadata contains
effective settings, override provenance, variant results, delivered dimensions,
artifact URLs/digests, build identities, queue/run times and CPU peak RSS.
Checkpoint-inherited settings remain unknown until execution restores them.
Progress is a phase estimate capped below 100 until publication. GPU counters
alone never mark a job successful. There are no public absolute file paths;
use content/artifact URLs. Native state payloads retain their native provenance.

The standard-library client can submit, wait and download:

```sh
python3 scripts/server_client.py submit request.json --wait --output boat.mp4
python3 scripts/server_client.py get JOB_ID
python3 scripts/server_client.py cancel JOB_ID
python3 scripts/server_client.py delete JOB_ID
```

Use `--url` for another configured address and `H3_SERVER_API_KEY` for auth.

## References and immutable assets

Upload each input file, then use its returned `asset://...` URI:

```sh
curl --fail-with-body -sS -F 'file=@inputs/1.jpg' \
  http://127.0.0.1:30000/v1/h3/assets
```

SGLang conditions use `type`, `role`, `uri`, optional `frame_index`, and optional
zero `start_time_seconds`. Supported image keyframes are first `0`, last `-1`,
or both in that order. Reference images, videos and audio retain their order.
`video` maps to native `--ref-video`, preserving an available soundtrack;
`video_audio` additionally requires an embedded audio stream. Use
`--ref-silent-video` for a silent reference or `--ref-video-audio VIDEO AUDIO`
for a separate soundtrack. Native reference counts and combinations apply.
Nonzero seeks and native-unsupported hybrid reference/keyframe combinations
return explicit errors. Audio-derived target duration requires exactly one
effective audio-bearing reference.

Metal's existing reference-video encoder requires 48–360 decoded reference
frames and caps decoding to the requested generated length. Consequently a
22-frame job with a video reference fails; use at least 56 aligned generated
frames and a reference clip of at least two seconds. Reference images and
audio do not have that video-specific minimum.

```json
{
  "task": "fl2va",
  "prompt": "A gentle camera move through the scene.",
  "conditions": [
    {"type":"image", "role":"keyframe", "frame_index":0, "uri":"asset://FIRST"},
    {"type":"image", "role":"keyframe", "frame_index":-1, "uri":"asset://LAST"}
  ],
  "h3cli": "--width 256 --height 256 --frames 22 --steps 2"
}
```

A native `--first-frame` or `--last-frame` replaces its matching anchor slot,
retaining the other compatible SGLang anchor. Any native ordered reference flag
replaces the **whole** SGLang reference list and clears incompatible keyframes.
Native anchors clear incompatible reference lists. `--ref-image-size` alone
keeps references. Replaced missing files, forbidden paths and URLs are never
imported or probed. Geometry derives from the effective replacement anchors.
A native model path overrides `model`; an operation flag overrides lower-priority
SGLang task/generation settings. Incompatible explicit native flags remain errors.

A multipart video submission may include one uploaded `input_reference` file
alongside text fields. It becomes a first anchor when canonical conditions do
not conflict. All request fields are also available as multipart text; encode
arrays/objects/numbers as JSON, but send `h3cli` as plain text.

Local input imports require repeatable `--server-read-root PATH`. Relative
paths resolve against startup cwd; output and cache names must be safe relative
names inside managed storage. Model components may use the configured model
root or the project's `models/` tree. Media are copied before acknowledgement;
large configured models/LoRAs use checked metadata identities and must remain
read-only. Queued/active jobs lease their source assets, preventing deletion.

URLs are off by default. `--server-allow-url-inputs` enables HTTP(S) with three
redirects, a 30-second deadline, byte limits and connection-address validation.
Private, loopback, link-local/metadata and non-global destinations are rejected,
including redirects and DNS rebinding attempts. Proxy environment variables
are not used. Uploads support all workflows without enabling network imports.

## Saved states and native operations

Use `POST /v1/h3/jobs` (HTTP 202) for stills, inspection, paused checkpoints and
state-only jobs. The body remains the same SGLang-plus-string envelope:

```json
{"prompt":"A red wooden toy boat floating on a quiet pond.",
 "h3cli":"--width 256 --height 256 --frames 22 --steps 4 --stop-after-step 2 --save-sampler-state pause.h3sample"}
```

A successful checkpoint has `status=completed` and `result.kind=paused`.
Its artifact list includes the native file, informational sidecars and a
`.h3bundle` transfer file. Download and upload the **bundle** when moving a state
between servers or back through the asset API; this preserves presentation and
LoRA sidecars. Bundles use a bounded manifest with fixed suffixes and per-file
SHA-256 checks, with no archive paths or traversal. Current native loaders
still validate state versions, recipe/model identities and checkpoint overrides.

Reuse an artifact directly without a download/reupload:

```json
{"h3cli":"--resume-sampler-state artifact://JOB_ID/ARTIFACT_ID --preview-vae --save-av-state resumed.h3av"}
```

The artifact ID can identify the native state or its bundle. Other operations:

| Workflow | `h3cli` flags |
| --- | --- |
| Save/reload conditioning | `--save-conditioning cache.h3cond` / `--load-conditioning asset://ID`, with the same prompt/recipe |
| AV decode | `--decode-av-state asset://ID` |
| Continuation | `-p 'Next scene' --continue-from asset://ID --continue-context 39 --frames 56 --width 256 --height 256 --steps 2` |
| Bridge | Add `--continue-mode bridge --continue-bridge-steps 1` to continuation |
| Upscale | `--upscale-state asset://ID --upscale-refine-steps 2 --save-av-state upscale.h3av` |
| Upscale inspection | `--inspect-upscale-state asset://ID` |
| Still/latent save | `--still -p 'A toy boat' --width 256 --height 256 --steps 2 --save-still-latent image.safetensors` |
| Still decode | `--decode-still-latent asset://ID` |
| LoRA | `--lora asset://ID:0.5`, repeatable; managed CoW cache and native identity checks |
| Frames/previews | `--frames-dir frames --show --zoom 2` |

`--show` becomes bounded preview events and a preview artifact. `--zoom` is
client display metadata. CUDA/Metal options pass through the shared registry
and native guards; discovery marks unavailable backend features. All 111
current native options are inventoried by `/v1/h3/capabilities`.

## Routes, cancellation and retention

| Method and route | Behavior |
| --- | --- |
| `GET /health`, `/readiness`, `/liveness` | Service/storage readiness; liveness stays healthy for recoverable storage faults |
| `GET /v1/models` | Configured model name; `h3cli` and `MiniMaxAI/MiniMax-H3` are accepted aliases |
| `GET /v1/h3/capabilities`, `/v1/h3/schema` | Versioned request schema, flag inventory, platform and limits |
| `POST /v1/videos` | Queue a video-producing request; HTTP 200 |
| `GET /v1/videos` | List; `limit=1..100`, `order=asc|desc`, `after=JOB_ID` |
| `GET/DELETE /v1/videos/ID` | Retrieve / cancel and hide a job |
| `GET/HEAD /v1/videos/ID/content?variant=N` | Committed MP4 with byte ranges |
| `POST/GET /v1/h3/jobs` | Submit native job (202) / list |
| `GET/DELETE /v1/h3/jobs/ID` | Full job metadata / cancel and hide |
| `POST /v1/h3/jobs/ID/cancel` | Cancel queued or active work |
| `GET /v1/h3/jobs/ID/events` | SSE; reconnect with `Last-Event-ID` |
| `GET/HEAD /v1/h3/jobs/ID/preview?variant=N` | Latest running preview, if requested |
| `GET/HEAD /v1/h3/jobs/ID/artifacts/ART_ID` | Committed artifact; byte ranges and SHA-256 ETag |
| `POST /v1/h3/assets` | One streamed multipart `file`; HTTP 201 |
| `GET/DELETE /v1/h3/assets/ID` | Metadata / delete an unleased upload |

SSE retains the latest 512 events per job, limits concurrent streams to four,
and closes each stream after ten seconds for bounded reconnects. Polling is
sufficient; disconnecting a stream does not cancel inference. Logs are capped
at 2 MiB. Failed jobs expose their diagnostic log, never unfinished media.

Cancellation asks the worker cooperatively, then terminates its process group
within a bounded grace period, including FFmpeg. Shutdown preserves queued
jobs and reaps active work. On restart, interrupted work is not regenerated;
a durable completed manifest is reconciled, and other queued jobs retain their
order. Explicitly resume a valid saved checkpoint to continue interrupted work.
Replacing the executable while a service runs requires a restart.

Completed artifacts are retained until explicit job deletion. Deletion hides
them immediately; garbage collection waits for input/download leases. Uploads
require explicit deletion when no longer needed. Orphan staging files are
cleaned on restart. Tombstones preserve idempotency/history. Shared native
LoRA/quantization caches remain for reuse and count toward the disk limit;
stop the service before administrator cache maintenance. Quota exhaustion
rejects new work instead of evicting queued inputs or active downloads.
