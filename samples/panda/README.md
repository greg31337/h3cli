# An Afternoon Walk of a Cat Called Panda

Follow Panda, the black-and-white cat in [panda.png](panda.png), on a leisurely
late-afternoon walk through a suburban neighborhood and its parks and gardens.
The route begins beside a rotating lawn sprinkler, passes a pond and footbridge,
and eventually leads through a hedge maze and gazebo to a sun-warmed hilltop
stone. Panda settles there to survey the neighborhood after the adventure.

The [40 prompts](panda.txt) describe one continuous journey. They preserve the
cat's appearance, lighting, camera movement and surroundings at each handoff.
The soundtrack follows the action with pawsteps, birds, leaves, water and distant
neighborhood activity, including occasional background dialogue. There is no
music or narration.

From the repository root, with Python 3.10+, FFmpeg, FFprobe and h3cli available:

```sh
sh samples/panda/run.sh --models-path models
```

For a Linux checkout configured with `scripts/setup_linux.sh`, first source
`outputs/setup/linux-env.sh`, then select the installed model root:

```sh
. outputs/setup/linux-env.sh
sh samples/panda/run.sh --models-path /path/to/models
```

The launcher resolves the script, prompts and reference image relative to its
own location, so it can be invoked from any working directory. It uses the
repository's executable `bin/h3cli` when available, otherwise `h3cli` on PATH.
Set `H3CLI_BIN` or pass `--h3cli /path/to/h3cli` to select another executable.
`-d /path/to/MiniMaxH3` selects the main model directory separately.
Set `H3_OFFLINE=1` to require local weights and disable model downloads.

| Setting | Value |
| --- | --- |
| Theme | An Afternoon Walk of a Cat Called Panda |
| Render and delivery | 960×544 at 24 fps, full VAE, 50 denoising steps |
| Segments | 40 prompts, 243 internal frames each |
| Reference | The same `panda.png` in every segment, high reference size |
| Continuation | Previous `.h3av` via `--continue-from`, 39 context frames |
| Delivered frames | 243 first, then 204 per continuation |
| Combined length | 8,199 frames, 5 minutes 41.625 seconds |
| Output | `panda.mp4` in the launch directory, without number overlays |

Every segment saves its video/audio state and a `.presentation` sidecar for the
next segment. H3 removes the inherited 39 context frames from each continuation's
delivered video and audio; the runner joins those clips without trimming them
again. With overlays disabled, preparation and assembly copy the video, while
PCM audio intermediates feed one final AAC encode. There is no upscaling.

Additional launcher arguments are passed through to [test.py](test.py). Later
arguments override the default step count and output path:

```sh
sh samples/panda/run.sh --models-path models --steps 30 -o panda-walk.mp4
sh samples/panda/run.sh --models-path models --preview --steps 10 -o panda-preview.mp4
```

The preview example explicitly sets ten steps because the launcher otherwise
requests fifty. To use a different prompt file or reference image, or to include
numbered segment badges, invoke the Python runner directly:

```sh
python3 samples/panda/test.py samples/panda/panda.txt samples/panda/panda.png \
  --h3cli bin/h3cli --models-path models -o panda-numbered.mp4
```

Before generation, the runner checks the media tools. It then verifies the
dimensions, frame count, frame rate and presence of audio in every clip and the
combined movie. Actual visual and sound continuity still need review after
rendering. A fresh `outputs/panda-continuation-*` directory in the launch
directory retains states, sidecars, original and prepared clips, logs and
`run.json`. Use `--work-dir PATH` for another fresh directory and `-o PATH.mp4`
for another final output. Existing final movies are never overwritten.
