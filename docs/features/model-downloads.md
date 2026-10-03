# Automatic model downloads

A normal CLI or server job downloads missing supported weights before loading
models or allocating GPU memory. Downloads use native libcurl, verified pinned
sources and full SHA-256 checks. Python, the Hugging Face CLI and external curl
are unnecessary at runtime. Complete local installations make no model-network
requests; build/setup commands do not download models.

```sh
# Download only what this render needs.
./bin/h3cli --models-path /fast/models -p 'A sailboat on a calm lake.' \
  --width 256 --height 256 --frames 22 --steps 2 -o sailboat.mp4

# Optional advance provisioning and local inspection.
./bin/h3cli --models-path /fast/models --download-models base,preview
./bin/h3cli --models-path /fast/models --download-models all
./bin/h3cli --models-path /fast/models --list-models
```

Groups are `base`, `references`, `preview`, `image-vae`, `upscale`, and `all`.
`base` supplies FL2VA; `references` supplies Ref2VA, including all supported
image/audio/video reference types. Auxiliary groups provision only that
component. Prefetch and list cannot be combined with generation flags.
`--info`, help, saved-state inspection and validation-only requests never fetch.
List reports presence without reading hundreds of gigabytes to verify them;
explicit prefetch verifies the selected pinned files.

## Locations and overrides

`--models-path ROOT` defaults to cwd `models`. Relative paths remain relative
to the directory where h3cli was launched. There is no search in another root.

| Component | Default beneath ROOT | Explicit override |
| --- | --- | --- |
| Main H3 | `MiniMaxH3/` | `-d/--model-dir` |
| Tiny preview decoder | `preview-vae/taeh3.safetensors` | `--preview-vae-model` |
| Still-image VAE | `image-vae/minimax_h3_t1_image_vae_step1597.safetensors` | `--image-vae` |
| BF16 latent upscaler | `latent-upscale/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors` | `--upscale-model` |

When `MiniMaxH3` is absent, an existing `MiniMax-H3` entry in the selected root
is reused in place. The canonical entry wins when both exist. Explicit main or
auxiliary overrides always win, regardless of flag order. Changing `-d` changes
only the main directory. An explicit missing auxiliary filename is the download
destination for that pinned component; it is not silently redirected elsewhere.
Existing symlink roots are resolved before publication; dangling/inaccessible
roots fail. Complete readable custom installations remain local inputs.

```sh
# Main checkpoint on one disk, auxiliaries on another.
./bin/h3cli --models-path /fast/aux -d /bulk/custom-H3 \
  --preview-vae --preview-vae-model ./my-preview.safetensors \
  -p 'A sailboat.' --width 256 --height 256 --frames 22 --steps 2 -o preview.mp4
```

## Offline, storage and recovery

`--offline` or `H3_OFFLINE=1` disables model networking, including authentication
and metadata probes. Missing dependencies produce an error naming the component
and destination. Existing models can run without DNS or a CA store.

The [file dependency table](model-download-dependencies.md) records the exact
closure for every operation. All groups occupy about **294 GB logically**, with
about **216 GB of unique download content**. Space checks allow copy fallback;
CoW can reduce physical storage. Keep additional room for LoRA/quantization
caches, states and media. No weight files belong in the executable or Git.

Private `.h3cli-downloads/` directories beside the main model root and auxiliary
files hold locks, receipts and resumable parts. They are outside native model
identity scans. Four transfers run at most; identical verified files are cloned
with CoW or copied into independent inodes. Interrupted transfers resume from
pinned origins, check saved prefixes and verify the complete file before an
atomic publication. Retry delays are bounded; Ctrl-C leaves reusable partials.
Re-run the same command to resume. Keep the model path and pinned catalog fixed.

A changed/corrupt installed file is never overwritten automatically. Partial
custom or older-revision trees that conflict with the catalog are rejected;
use a separate empty destination. This includes adding a missing main-model mode
beside an existing custom mode: both modes are checked before adopting the root. Do not delete an existing checkpoint to repair
it automatically. To discard a corrupt **unpublished** partial, stop all jobs
using that destination and remove its `.part` and matching `.part.meta` files
inside the private directory; the next run starts that file again. Receipts
avoid repeated full hashing on warm runs and detect file-identity changes.

Downloading does not migrate saved-state identities. A checkpoint tied to the
metadata of one model installation still requires that installation; fetching
identical weights into a different tree does not bypass those existing checks.
Changing only `--offline` or `H3_OFFLINE` does not change numerical identity.

HTTPS certificate verification stays enabled. Standard curl proxy settings and
`CURL_CA_BUNDLE` / `SSL_CERT_FILE` are supported. Optional `HF_TOKEN` is read from
the process environment and sent only to the exact Hugging Face origin; redirect
hosts and changed ports receive no token. Logs/events omit URLs and credentials.
No token is written into receipts, plans or the catalog.
Minimal Linux images must provide a system CA certificate bundle for HTTPS;
the standalone executable does not embed an aging trust store. Offline use
requires no certificate bundle.

## Queued server jobs

```sh
./bin/h3cli --server --models-path /fast/models \
  --server-model-download-timeout 21600
```

The root may initially be empty. Startup `-d` overrides only the main model.
Workers receive absolute resolved paths even if their working directory changes.
Clients cannot set `--models-path`, `--download-models` or `--list-models` in a
job; `h3cli: "--offline ..."` is permitted and cannot override server offline
policy. Startup/environment offline policy also blocks URL media imports.
Existing custom paths under configured read roots remain readable, but a client
cannot authorize downloads into missing paths outside administrator-managed
main/auxiliary destinations.

Preparation runs in a supervised CPU helper before the inference worker.
Standard job status remains `queued`; `h3.phase`, `h3.model_preparation` and
bounded events expose component/byte progress. The preparation deadline defaults
to six hours; `--server-job-timeout` starts only when inference starts. Cancellation,
shutdown and restart retain shared completed models and resumable parts. A
preparation interrupted by server restart is reported as interrupted; resubmit
with a new idempotency key to resume the installation. Inference workers are
always offline. Model disk space is separate from the result-storage quota.

## Provenance

The [embedded catalog](../../src/models/catalog.json) pins the main model to
[`42ed227ee7df40d41602854ae760620d6eb651fe`](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe).
All 115 previously installed main metadata/weight files match this revision
byte for byte; the additional main LICENSE is included by prefetch. Model
licensing is separate from h3cli code: see the pinned
[MiniMax model license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE),
[TAEH3 MIT notice](../../third_party/taeh3/LICENSE),
[image-VAE provenance](single-still-sources.md), and
[upscaler model card](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/blob/3f941d5d182014dd5c0a5e16330420ee2d4aa0c6/README.md).
The image-VAE publisher supplies no license field in the pinned repository
metadata; this implementation makes no additional licensing claim for it.
The upscaler's model card declares Apache-2.0. No upstream executable model code
is downloaded or executed.
