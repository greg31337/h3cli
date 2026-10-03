# Video compression and lossless RGB

h3cli writes MP4 with H.264 video, using **CRF 18** by default on both CUDA and
Metal. Lower CRF values keep more detail and usually produce larger files.
These controls affect delivery only: they do not change the seed, denoising,
VAE, conditioning, or saved latents. The separate `--quality` flag chooses
generation settings; even `--quality lossless` still uses CRF 18 unless you
select different output settings.

## Compression controls

`--output-quality` follows SGLang's video quality names and CRF mapping:

| Selection | H.264 CRF |
| --- | ---: |
| Flag omitted: h3cli production default | 18 |
| `--output-quality default` | 25 |
| `--output-quality maximum` | 0 |
| `--output-quality high` | 5 |
| `--output-quality medium` | 22 |
| `--output-quality low` | 33 |

The mapping follows SGLang's
[sampling parameters](https://github.com/sgl-project/sglang/blob/482b062d282b2ef4e449bd108df5228d81da2112/python/sglang/multimodal_gen/configs/sample/sampling_params.py)
and [video writer](https://github.com/sgl-project/sglang/blob/482b062d282b2ef4e449bd108df5228d81da2112/python/sglang/multimodal_gen/runtime/entrypoints/utils.py).
Matching the CRF alone does not guarantee identical encoded files.

Use `--ffmpeg-crf N` for an explicit integer from 0 through 51. It overrides
`--output-quality` regardless of argument order. Repeating either scalar flag
uses its last value. For example:

```sh
./bin/h3cli -p "A small boat on a calm lake." --quality extra-high \
  --output-quality medium --ffmpeg-crf 18 -o boat.mp4
```

Ordinary output uses `libx264`, the `fast` encoder preset, 8-bit `yuv420p`,
explicit full-range RGB to limited-range BT.709 conversion, and BT.709 color
metadata. Audio uses AAC at 192 kbit/s, retaining the model's 32 kHz stereo
sample format. MP4 metadata is placed before the media data (`+faststart`) so
playback can begin before a complete download.

CRF 0 with `yuv420p` preserves the converted YUV samples, but conversion and
chroma subsampling still discard RGB information. Use the following option
when the delivered RGB pixels must survive encoding exactly.

## Lossless RGB video

```sh
./bin/h3cli -p "A small boat on a calm lake." --lossless-video -o boat-rgb.mp4
```

`--lossless-video` selects `libx264rgb`, CRF 0, and full-range RGB without chroma
subsampling. Decoding to RGB24 reproduces the encoder's input bytes exactly.
This preserves the **delivered 8-bit RGB frames**, not the VAE's original
floating-point values. Audio remains AAC and is not lossless. Save AV state
separately when you need the latents for future decoding.

The flag takes precedence over `--output-quality`. An explicit `--ffmpeg-crf 0`
is allowed; other CRF values with `--lossless-video` are rejected. Files are
usually much larger. H.264 RGB/High 4:4:4 playback support is more limited than
ordinary H.264/YUV420 MP4, especially in browsers and hardware players.

All three controls work with fresh video generation, continuation, bridge,
sampler resume, saved AV-state decoding, and upscaling. They are rejected for
still images, inspection, and state-only jobs. Encoding settings are not saved
in sampler checkpoints: select them on each delivery command.

## Server requests

The SGLang `output_quality` field maps to `--output-quality`. Omitting it keeps
CRF 18; explicitly sending `"output_quality":"default"` selects CRF 25.
Any encoding flag inside the native `h3cli` string overrides this field:

```json
{
  "prompt": "A small boat on a calm lake.",
  "output_quality": "medium",
  "h3cli": "--width 256 --height 256 --frames 56 --steps 2 --ffmpeg-crf 18"
}
```

Use `"h3cli":"--lossless-video"` for lossless RGB. SGLang's numeric
`output_compression` field remains unsupported; use `--ffmpeg-crf` instead.

## Recorded SGLang regression

The immutable 204-output gate retains its historical FFmpeg 4.2.2 encoder,
CRF 25, automatic color conversion, original AAC defaults, thread count, and
MP4 layout. Its private harness selects this recipe independently of the CUDA
arithmetic path. Production rendering uses the new settings, so its decoded
MP4 pixels and audio are not expected to match the historical delivery hashes. No golden output is
updated. See the [codec isolation policy](../build/ffmpeg-upgrade.md).
