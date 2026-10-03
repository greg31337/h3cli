# h3cli

**Create videos with sound on your own Mac or NVIDIA GPU.**

h3cli runs [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) locally from
the command line. Describe a scene, add reference images or clips if you want,
and get an MP4 with generated audio. Inference is native: no Python, PyTorch,
or SGLang installation is needed to run a release package.

h3cli's CUDA backend can generate **videos with frames bit-equal to the official
SGLang implementation** in the validated reference configuration. See
[SGLang parity](docs/cuda/cuda-sglang-reference.md) for the supported hardware,
matching settings, and validation scope.

[Get started](#get-started) · [First video](#make-your-first-video) ·
[Quality](#choose-your-quality) · [Documentation](docs/README.md)

## What you can do

- Generate videos with dialogue, music, and sound effects from a text prompt.
- Guide the result with images, video, or audio, or choose the first and last frames.
- Continue a video across multiple segments and save progress for later.
- Make quick previews, then use higher quality settings for the final render.
- Upscale saved video latents to twice their width and height.
- Apply LoRA adapters to customize generation.
- Submit jobs through a queued video API compatible with a subset of SGLang's API.

Apple Silicon uses Metal; Linux with NVIDIA GPUs uses CUDA. Some acceleration
options are specific to particular hardware. The [documentation](docs/README.md)
covers those options and their limits.

## Get started

### Choose your installation

For published builds, open this repository's **Releases** page and choose the
package for your platform. If no release is available yet, follow the
[source-build guide](docs/build/source-build.md).

| Platform | Release executable | Requirements |
| --- | --- | --- |
| Apple Silicon Mac | `h3cli-macos-arm64` | macOS 26 or later |
| Linux with NVIDIA GPU | `h3cli-linux-x86_64` | Linux x86-64 and a supported NVIDIA GPU and driver |

For small videos, a practical starting point is **64 GB RAM and 24 GB VRAM on
Linux**, or **64 GB unified memory on Mac**. These are planning estimates;
larger or longer videos need more memory. Weight streaming can support smaller
machines, but lower-memory configurations still need hardware validation. See
[hardware and memory requirements](docs/hardware-requirements.md) for measured
results, GPU/CPU requirements, and SSD recommendations.

Packages include the media tools they need. Linux packages also include the CUDA
runtime, so you do not need to install the CUDA toolkit. See the
[Mac](docs/build/macos-distribution.md) and
[Linux](docs/build/linux-distribution.md) guides for installation details and
hardware coverage. Check the release notes for signing and qualification status.

Copy the executable into a working folder and name it `h3cli`. On macOS, copy it
out of the DMG first. On Linux, make it executable with `chmod +x h3cli`.
The examples below assume you run commands from that folder:

```sh
./h3cli --help
```

If you built from source, use `./bin/h3cli` in place of `./h3cli` and load the
setup environment described in the build guide.

### Choose where to store models

Missing models download automatically the first time a command needs them.
By default, they are stored in a `models/` folder in your current directory.
Use `--models-path /path/to/models` on your commands to choose another location.

The complete model collection takes about **300 GB of disk space**; individual
jobs download only the components they need. Allow extra space for generated
videos and saved states. H3 also needs substantial GPU or unified memory; start
with a small preview before increasing resolution and duration.

If you already have the main model, point to it with `-d /path/to/MiniMaxH3`.
This changes only the main model location. Use `--offline` to require existing
local files. See [model downloads and storage](docs/features/model-downloads.md)
for auxiliary models, advance downloads, and recovery after interruptions.

## Make your first video

Start with a short preview:

```sh
./h3cli \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, soft footsteps and wind." \
  --quality preview \
  --width 640 --height 480 --seconds 4 \
  --seed 42 -o fox-preview.mp4
```

The first run may spend time downloading and loading models. The completed
video is saved as `fox-preview.mp4` in your current directory.

For a final render, change `--quality preview` to `--quality extra-high` and
choose a new output filename. Describe the subject, motion, camera, lighting,
and sounds in your prompt. Reuse the same `--seed` when comparing settings.

Videos use 24 frames per second. Duration is rounded up to a supported frame
count, so a four-second request produces about 4.46 seconds. You can use
`--frames` instead of `--seconds` for direct frame control. See
[resolution and duration](docs/usage.md#5-pick-resolution-and-duration) for
landscape, portrait, and longer clips.

## Choose your quality

| `--quality` | Intended use | Steps |
| --- | --- | ---: |
| `fast-preview` | Quick checks of a prompt or composition | 6 |
| `preview` | A more developed draft | 12 |
| `high` | Faster final renders on supported hardware | 50 |
| `extra-high` | Final renders with the full decoder | 50 |
| `lossless` | Currently the same rendering settings as `extra-high` | 50 |

Preview levels use a faster, approximate video decoder. The names describe
generation settings; `lossless` does not mean lossless MP4 compression.
Without `--quality`, h3cli uses its normal 20-step defaults.

MP4 compression defaults to CRF 18 on both platforms. Use `--ffmpeg-crf N` or
the SGLang-compatible `--output-quality` levels to change it. `--lossless-video`
preserves the delivered RGB pixels exactly, with larger files and more limited
player support. See [video compression and lossless RGB](docs/features/output-encoding.md).

You can override individual settings, such as `--quality preview --steps 20`.
The `high` preset has hardware and feature restrictions; use `extra-high` for
first/last-frame renders. See [quality presets and overrides](docs/usage.md#4-choose-a-speedquality-preset)
for the full behavior.

## Use your own images and clips

A reference image guides the subject or appearance of the generated video:

```sh
./h3cli \
  -p "The person in Picture 1 walks along a sunny beach. Gentle waves and seabirds." \
  --ref-image reference.jpg \
  --quality preview --width 640 --height 480 --seconds 4 \
  -o beach.mp4
```

To set the opening and closing frames instead:

```sh
./h3cli \
  -p "The camera slowly moves closer as the person turns toward it." \
  --first-frame opening.jpg --last-frame ending.jpg \
  --quality extra-high --width 640 --height 480 --seconds 4 \
  -o transition.mp4
```

Both anchors are scaled to cover the video canvas and center-cropped, preserving
their aspect ratio. Use reference media or first/last-frame anchors as separate
modes. Multiple image references, video clips, soundtracks, and image sizing are
covered in the [reference guide](docs/usage.md#8-add-image-video-and-audio-references).

## Go further

| Goal | Guide |
| --- | --- |
| Continue a video across segments | [Continuation](docs/continuation/continuation.md) |
| Pause and resume a render | [Saved sampler checkpoints](docs/features/sampler-state.md) |
| Preview quickly and decode the saved result later | [Preview decoder](docs/preview/preview-vae.md) |
| Upscale a saved generation | [Latent upscaling](docs/features/latent-upscaling.md) |
| Apply an adapter | [LoRA](lora/README.md) |
| Queue jobs and fetch results over HTTP | [Server API](docs/features/server.md) |
| Generate a still image | [Still generation](docs/features/single-still.md) |
| Explore advanced controls | [Usage guide](docs/usage.md) |

To start the local job server, run `./h3cli --server`. The server guide explains
how to submit jobs, check progress, cancel work, and download the finished video.

## Documentation and contributing

The [documentation index](docs/README.md) brings together installation guides,
features, troubleshooting, and developer notes. Build dependencies, backend
internals, benchmarks, and validation reports live under `docs/`.

For bug reports, include your operating system, GPU and memory, the command you
ran, and the relevant error output. For code changes, read
[CONTRIBUTING.md](CONTRIBUTING.md). Maintainers can follow the
[release guide](docs/release/README.md) to build, sign, and publish packages.

## License

h3cli's code is [MIT licensed](LICENSE). Model weights and bundled dependencies
have their own terms; see [model licensing](docs/features/model-downloads.md#provenance)
and [third-party notices](THIRD_PARTY_NOTICES.md).

## Acknowledgements

**h3cli started from [h3.c](https://github.com/antirez/h3.c) by Salvatore
Sanfilippo (antirez).** Its native MiniMax-H3 implementation is the foundation
of this project. Thank you to Salvatore and the upstream contributors.

Thanks also to:

- [MiniMax](https://github.com/MiniMax-AI/MiniMax-H3) for H3 and its released
  models, and the [Qwen team](https://github.com/QwenLM/Qwen3-VL) for Qwen3-VL.
- [SGLang](https://github.com/sgl-project/sglang) for the H3 reference
  implementation that informs the CUDA pipeline and compatibility work.
- [Ollin Boer Bohan / TAEHV](https://github.com/madebyollin/taehv) for the tiny
  preview decoder, and [LBH-123-AI](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)
  for the latent-upscaler implementation and models.
- The [MLX](https://github.com/ml-explore/mlx), [ccv](https://github.com/liuliu/ccv),
  [FlashAttention](https://github.com/Dao-AILab/flash-attention), and
  [SageAttention](https://github.com/thu-ml/SageAttention) contributors for work
  used in the native GPU backends.
- [FFmpeg](https://ffmpeg.org/) and the other open-source projects that support
  h3cli. Their attributions and license notices are retained in
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [licenses/](licenses/), and
  [third_party/](third_party/).
