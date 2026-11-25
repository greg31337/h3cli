# h3cli documentation

Start with the [project README](../README.md) for installation and your first
video. This directory contains the detailed guides and engineering records.
Most command examples here assume a source checkout and `./bin/h3cli`; replace
that path with your downloaded executable when using a release package.

## Installation and models

- [Hardware and memory requirements](hardware-requirements.md): RAM, VRAM,
  supported hardware, storage, and measured versus estimated requirements.
- [Source builds and dependencies](build/source-build.md): macOS and Ubuntu
  setup, CUDA versions, build outputs, and platform requirements.
- [Linux packages](build/linux-distribution.md) and
  [macOS packages](build/macos-distribution.md): building and running standalone
  executables, runtime requirements, and package caches.
- [Automatic model downloads](features/model-downloads.md): model locations,
  offline use, disk space, and interrupted downloads.
- [Memory and long references](stability/memory.md): memory admission,
  cancellation, and reference-processing limits.

## Making videos

| Topic | Guide |
| --- | --- |
| Rendering options and examples | [Advanced usage](usage.md) |
| Text scripts, merged sequences and inherited prompt sections | [Text-script rendering](features/sequence-rendering.md) |
| Quality and speed | [Quality presets](usage.md#4-choose-a-speedquality-preset) |
| MP4 compression and lossless RGB | [Output encoding](features/output-encoding.md) |
| Long CUDA video and NVFP4 | [1,450-frame contract](cuda/long-video-1450-contract.md), [capacity estimates](cuda/long-video-1450-memory-estimates.md) |
| Resolution and duration | [Canvas sizes and frame counts](usage.md#5-pick-resolution-and-duration) |
| Image, video, and audio references | [Reference media](usage.md#8-add-image-video-and-audio-references) |
| Preview decoding and later full-quality delivery | [Preview VAE](preview/preview-vae.md) |
| Continuing a completed segment | [Continuation](continuation/continuation.md) |
| Adapting inherited context | [Bridge mode](continuation/bridge-continuation.md) |
| Pausing and resuming generation | [Sampler checkpoints](features/sampler-state.md) |
| Doubling spatial resolution from saved latents | [Latent upscaling](features/latent-upscaling.md) |
| Applying adapters | [LoRA workflows](../lora/README.md) |
| Still-image generation | [Stills](features/single-still.md) |
| Queued HTTP jobs | [Server API](features/server.md) |
| Dialogue and other prompt markers | [Tokenizer and prompt markers](minimax/tokenizer.md) |

## Performance and internals

- [Implementation and development notes](development.md): sampler behavior,
  weight streaming, Metal optimizations, diagnostics, and validation links.
- [CUDA caching and sparse attention](cuda/acceleration.md): adaptive caching,
  SubBlock, SOL, warmup controls, and supported combinations.
- [SageAttention](cuda/attention.md) and
  [FP8/NVFP4 denoising](cuda/denoiser-quantization.md).
- [CUDA weight placement](cuda/weight-residency-design.md) and
  [hardware measurements](cuda/weight-residency-results.md).
- [Native Metal design](metal/design-native.md) and
  [Q8 weights](metal/q8.md).
- [Current saved-state formats](features/current-state-contract.md) and
  [source layout](../src/README.md).

## Development and releases

- [Current implementation tasks](todo.md): plan CUDA rendering through 1,450
  frames with native NVFP4 support; Metal retains its 362-frame limit. Feature
  runtime tests remain at 256×256 and under 20 seconds, with host/descriptor-only
  maximum-duration checks and the unchanged 204-output regression gate.
- [Completed 736-frame CUDA milestone](long-video-736-todo.md): the archived
  duration-extension checklist, with its [contract](cuda/long-video-contract.md),
  [acceptance evidence](cuda/long-video-results.md), and
  [audio workspace correction](cuda/long-video-audio-workspace.md).
- [Completed CUDA large-canvas milestone](large-canvas-todo.md): native rendering
  to 1920×1088 and learned 2× upscaling to 3840×2176, with
  [memory estimates](cuda/large-canvas-memory-estimates.md),
  [backend/profile contract](cuda/large-canvas-contract.md), and
  [acceptance evidence](cuda/large-canvas-results.md).
- [Completed empty-latent still milestone](empty-still-todo.md): stock video
  VAE decoding and native 8 MP output, with
  [acceptance evidence](features/empty-still-results.md).
- [Completed rendering milestone](render-integration-todo.md): built-in text-script rendering,
  script-wide merging and structured prompt inheritance, with
  [acceptance evidence](features/render-integration-results.md).
- [Completed release-tooling milestone](release-tooling-todo.md) and its
  [acceptance evidence](release/results.md).
- [Contributing](../CONTRIBUTING.md) and
  [test commands](development.md#tests-and-runtime-requirements).
- [Recorded SGLang regression](cuda/cuda-sglang-reference.md).
- [Release process](release/README.md): build, sign, and upload GitHub Releases.
- [Linux package qualification](build/linux-distribution-pro5000.md) and
  [macOS package qualification](build/macos-distribution-results.md).
- [Publication privacy](publishing.md),
  [third-party notices](../THIRD_PARTY_NOTICES.md), and
  [release source materials](../scripts/linux/SOURCE-MATERIALS.md).

Design documents, task lists, and results retain their original scope. A result
for one GPU or combination of options does not qualify every other combination;
follow the current user guide and the limits stated in each report.
