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
| Quality and speed | [Quality presets](usage.md#4-choose-a-speedquality-preset) |
| MP4 compression and lossless RGB | [Output encoding](features/output-encoding.md) |
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
