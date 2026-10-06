# Swiss Alps paraglider flight

A 50-segment first-person flight through one connected Swiss Alpine valley,
using the AV continuation workflow from [`samples/panda`](../panda). The pilot
starts on a timber launch platform on a high limestone outcrop, takes off, and
descends past forested spurs, pasture, cascades and a turquoise lake. The flight
ends with a gentle landing on an open lakeside meadow beside a small stone
chapel, with a waterfall on the cliff beyond. This is an imagined Alpine route.

Red-and-cream, ochre-yellow, and blue-and-white paragliders appear during three
parts of the flight. Each enters, travels through and leaves the view gradually.
Two waiting pilots with a folded wing stay at the landing field's edge.
The soundtrack follows the flight: launch footsteps, wind, subtle harness and
canopy rustle, distant cowbells, birds and water, then grass footsteps and quiet
valley ambience after landing. There is no music or narration.

Each prompt starts with the preceding segment's exact ending view. The prompts
preserve the camera, lighting, terrain and motion, and treat the carried frames
as authoritative. Takeoff and landing occur once; intermediate segments advance
the route rather than replaying completed actions. Actual continuity still needs
visual and listening review after rendering.

From the repository root, with Python 3.10+, FFmpeg, FFprobe and h3cli available:

```sh
sh samples/flight/run.sh --models-path models --offline
```

For a Linux checkout configured with `scripts/setup_linux.sh`, source its
environment before rendering and select the installed model root:

```sh
. outputs/setup/linux-env.sh
sh samples/flight/run.sh --models-path /path/to/models --offline
```

The runner uses the repository's `bin/h3cli` if present, otherwise `h3cli` on
PATH. Set `H3CLI_BIN` or pass `--h3cli /path/to/h3cli` for another executable.
Model lookup follows the CLI's usual rules; `-d /path/to/MiniMaxH3` selects a
main model directory. Without `--offline`, h3cli can download missing models.

| Setting | Value |
| --- | --- |
| Render and delivery | 960×544 at 24 fps, full VAE, 50 denoising steps |
| Segments | 50 prompts, 243 internal frames each |
| Continuation | Previous `.h3av` via `--continue-from`, 39 context frames |
| Delivered frames | 243 first, then 204 per continuation |
| Combined length | 10,239 frames, 7 minutes 6.625 seconds |
| Conditioning | Text-only start; subsequent segments inherit video and audio state |
| Upscaling | None |
| Output | `flight.mp4` in the launch directory, without number overlays |

Every render saves a `.h3av` state and its `.presentation` sidecar for the next
segment. No reference image or extracted first-frame image is used. H3 already
removes the 39 context frames from each continuation's delivered video and audio;
the runner joins those delivered clips without trimming them again. It copies
the video during preparation when overlays are disabled and uses PCM audio
intermediates followed by one final AAC encode, as in panda.

Review all 50 render commands and the assembly plan without loading models or
creating outputs:

```sh
sh samples/flight/run.sh --dry-run > /tmp/flight-plan.json
```

Use `--steps N` to change the denoising step count. To render a shorter route,
pass a separate prompt file to `python3 samples/flight/test.py`; sections use
the same `### NUMBER ###` format. For numbered segment overlays, invoke that
script without `--no-overlay`.

The runner checks media tools before generation, then verifies dimensions, frame
count, frame rate and audio for each clip and the combined video. A fresh
`outputs/flight-continuation-*` directory retains states, sidecars, original and
prepared clips, logs and `run.json`. Use `--work-dir PATH` to choose a fresh
directory and `-o PATH.mp4` to choose the output. Existing outputs are never
overwritten.

Run the model-free checks from the repository root:

```sh
python3 tests/test_flight_sample.py
```

These check the complete prompt and continuation plan and exercise a two-segment
assembly using synthetic 960×544 clips and real FFmpeg/FFprobe. They verify state
handoffs, frame counts, preservation of the first new continuation frame, audio,
and output overwrite protection. They do not assess generated scenery or prove
seamless motion across the full flight.
