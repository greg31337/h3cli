# Cabinet of miniature worlds

Forty independent macro scenes showcase small structures and surface detail:
brass island machines, paper railways, shell villages, glass reefs, miniature
libraries, fabric farms and mechanical creatures. Each ten-second scene has a
small action that settles before the cut. There is no shared character, state,
reference image or first-frame handoff between scenes.

The runner uses h3cli's
[native latent upscaler](../../docs/features/latent-upscaling.md).
Fine materials, controlled lighting, restrained motion and moderate depth of
field keep detail visible throughout the collection.

| Setting | Value |
| --- | --- |
| Segments | 40 independent text-to-video scenes |
| Source | 960×544, 243 native frames at 24 fps, full VAE, 50 steps |
| Upscale | Learned 2× latent transfer to 1920×1088; 4 refinement steps, noise 0.25 |
| Delivery | First 240 frames of each clip: exactly 10 seconds |
| Final crop | 1920×1080; remove 4 pixels from both top and bottom |
| Combined video | 9,600 frames, exactly 6 minutes 40 seconds |
| Audio | Quiet isolated sounds, half gain, 0.5-second fade-in and fade-out per clip |
| Default output | `miniatures.mp4` in the launch directory, without number overlays |

H3's native frame alignment makes 243 frames the source length for this target.
The runner trims the three extra frames after native upscaling. It then crops
vertically without resizing, prepares uniform H.264/PCM intermediates, and joins
them with copied video and a single final AAC audio encode. Cuts do not overlap
or shorten the 400-second timeline.

Prompts avoid sustained music, voices, wind, water and room-tone beds. Each allows
at most a very quiet foreground sound in the middle of the shot and asks for
near-silent edges. Gain reduction and fades soften the actual generated audio
at every join even if the model produces more sound than requested. Final sound
and image quality still need review after rendering.

From the repository root, with Python 3.10+, FFmpeg, FFprobe and h3cli available:

```sh
sh samples/miniatures/run.sh --models-path models --offline
```

For a Linux checkout configured with `scripts/setup_linux.sh`, first source
`outputs/setup/linux-env.sh`. Select a different installed model root with
`--models-path /path/to/models`. The upscaler defaults to
`MODEL_ROOT/latent-upscale/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`;
`--upscale-model /path/to/upscaler.safetensors` overrides it. `-d` can select the
main H3 model directory separately. Without `--offline`, h3cli can download
missing weights. The full native resolution requires sufficient memory.

The runner uses the repository's `bin/h3cli` if present, then `h3cli` on PATH.
Override it with `H3CLI_BIN` or `--h3cli /path/to/h3cli`.

Inspect all 80 source/upscale commands and the crop, audio and assembly commands
without loading models, running tools or creating outputs:

```sh
sh samples/miniatures/run.sh --dry-run > /tmp/miniatures-plan.json
```

`--steps N` changes source denoising. `--upscale-refine-steps 2` reduces refinement;
`0` demonstrates the learned transfer alone and omits the explicit noise setting.
The default is four steps. For a smaller selection of scenes, create a prompt
file with the same `### NUMBER ###` section format and invoke
`python3 samples/miniatures/test.py /path/to/prompts.txt --no-overlay`.
Omit `--no-overlay` for numbered clip badges.

Before inference, the runner checks the actual crop, codecs, audio and optional
badge with real media tools. Each native upscaled clip, prepared clip and final
movie is checked for dimensions, frame count, frame rate and audio. A fresh
`outputs/miniatures-upscale-*` directory retains all `.h3up` sources, upscaled
`.h3av` states and presentation sidecars, uncropped clips, prepared clips, logs
and `run.json`. Those saved states are artifacts, never inputs to another scene.
Use `--work-dir PATH` for a fresh directory and `-o PATH.mp4` for another output.
Existing final movies are never overwritten.

Run the model-free tests:

```sh
python3 tests/test_miniatures_sample.py
```

They check the full independent command plan and use synthetic upscale outputs
with real FFmpeg/FFprobe to verify exact ten-second clips, the centered crop,
audio gain and edge fades, joining, optional badges and overwrite protection.
