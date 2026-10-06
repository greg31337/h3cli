# Swiss Alps flight

A 20-segment aerial route through one connected Alpine valley, adapted from
[`samples/panda`](../panda). The camera follows meltwater upstream from scattered
pines through meadow, a rock shelf, a small lake, moraine, and a glacier toe.
Each prompt repeats the camera and lighting constraints and opens with the
preceding prompt's exact ending view. The same twin summit and right-hand cliff
provide persistent landmarks. No reference image is needed to start.

From the repository root, with Python 3.10+, FFmpeg, FFprobe, and h3cli available:

```sh
sh samples/flight/run.sh
```

The runner uses `bin/h3cli` if present, otherwise `h3cli` on PATH. Set
`H3CLI_BIN` or pass `--h3cli /path/to/h3cli` for another executable. For example:

```sh
sh samples/flight/run.sh --h3cli /path/to/h3cli \
  --models-path /path/to/models --offline
```

Model lookup follows the CLI's usual rules. `-d /path/to/MiniMaxH3` changes the
main model location; `--models-path` also locates the auxiliary upscaler.
`--upscale-model /path/to/weights.safetensors` overrides that upscaler alone.
Without `--offline`, h3cli can download missing models.

| Setting | Value |
| --- | --- |
| Source render | 960×544, 243 frames per segment, 50 denoising steps |
| Upscale | Native learned 2× latent upscale, 4 refinement steps, noise 0.25 |
| Delivered resolution | 1920×1088 at 24 fps |
| Handoff | Previous upscaled clip's final frame becomes the next `--first-frame` |
| Combined length | 4,841 frames, about 3 minutes 21.71 seconds |
| Audio | Generated natural airflow, retained during assembly |
| Output | `flight.mp4` in the launch directory, without number overlays |

The full VAE and dense generation settings are used. Each source is saved as
`.h3up` with `--state-only`; a separate `--upscale-state` command produces the
high-resolution clip and `.h3av` state. There is no conventional resizing of the
final video. The final frame is extracted from the unlabelled upscaled clip,
before assembly, so optional number badges never enter the next prompt's image.
H3 fits this image to the next 960×544 source canvas.

Native [latent upscaling](../../docs/features/latent-upscaling.md) currently
rejects AV continuation and bridge continuation, including upscale-source
capture from those modes. This sample therefore uses first-frame anchors.
They carry visual composition, but not the motion or audio history provided
by panda's `--continue-from`. Prompt continuity helps guide motion and sound;
seamless boundaries still need visual and listening review after rendering.
The inherited opening frame is removed from each later clip, leaving 243 frames
from the first clip and 242 from each of the other nineteen.

Review all forty generation/upscale commands, the nineteen image handoffs,
and the assembly commands without models or a render:

```sh
sh samples/flight/run.sh --dry-run > /tmp/flight-plan.json
```

Use `--upscale-refine-steps 2` for fewer refinement steps, or `0` to demonstrate
the learned transfer alone. Zero refinement omits the explicit noise setting,
as required by the CLI. `--steps N` controls source generation independently.
For numbered segment overlays, invoke `python3 samples/flight/test.py` directly
without `--no-overlay`.

The runner checks FFmpeg handoff extraction, trimming, audio, badges, and joining
before generation. It verifies resolution, frame count, frame rate, and audio
presence for every clip and the combined video. A fresh directory under
`outputs/flight-upscale-*` retains source bundles, high-resolution states and
presentation sidecars, clips, handoff PNGs, prepared clips, logs, and `run.json`.
Use `--work-dir PATH` to select a fresh directory and `-o PATH.mp4` to choose
the combined output. Existing combined outputs are never overwritten.

Run the model-free checks from the repository root:

```sh
python3 tests/test_flight_sample.py
```

These check the complete prompt/command plan and exercise a two-segment assembly
at 1920×1088 using synthetic clips, real FFmpeg/FFprobe, exact final-frame
extraction, and output overwrite protection. They do not assess generated
landscape quality or prove seamless model-generated motion.
