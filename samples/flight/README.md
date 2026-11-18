# Swiss Alps paraglider flight

A 50-segment first-person flight through one connected Swiss Alpine valley,
using the AV continuation workflow.
The pilot starts on a timber launch platform on a high limestone outcrop,
takes off, and descends past forested spurs, pasture, cascades and a turquoise
lake. The flight ends with a gentle landing on an open lakeside meadow beside
a small stone chapel, with a waterfall on the cliff beyond. This is an imagined
Alpine route.

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

The [flight.txt](flight.txt) file is a native `h3cli --render` script. The launcher
calls h3cli directly; no Python sample runner is required. Source builds need
FFmpeg/FFprobe and their configured environment. Standalone packages include
the media tools.

From the repository root:

```sh
sh samples/flight/run.sh --models-path models --offline
sh samples/flight/run.sh --dry-run > /tmp/flight-plan.json
```

For a source build, follow the [build guide](../../docs/build/source-build.md)
and source `outputs/setup/macos-env.sh` on the Mac or
`outputs/setup/linux-env.sh` on Linux. Omit `--offline` to allow missing models
to download. `-d /path/to/MiniMaxH3` selects the main model directory separately.

The launcher finds the script relative to itself and works from any directory.
It selects `H3CLI_BIN` when set, otherwise the repository's executable
`bin/h3cli`, then `h3cli` on PATH. All arguments go directly to h3cli:

```sh
H3CLI_BIN=/path/to/h3cli sh samples/flight/run.sh \
  --models-path /path/to/models --output-dir outputs/flight-another-run
```

The equivalent direct invocation is:

```sh
./bin/h3cli --render samples/flight/flight.txt \
  --models-path models --output-dir outputs/flight-render
```

| Setting | Value |
| --- | --- |
| Render and delivery | 960×544 at 24 fps, full VAE, 50 denoising steps |
| Segments | 50 prompts, 243 internal frames each |
| Seed | 42 for every segment, inherited from the first segment's `#seed 42` |
| Conditioning | Text-only start, followed by video/audio continuation; no external references |
| Continuation | `#continue` after the first segment, with 39 context frames |
| Delivered frames | 243 first, then 204 per continuation |
| Combined length | 10,239 frames, 7 minutes 6.625 seconds |
| Default movie | `outputs/flight-render/flight.mp4`, relative to the launch directory |
| Segment overlays | Disabled by default |

Edit the directives at the first segment to change size, frame count, steps or
decoder. Script directives override CLI defaults, so `--steps` cannot replace
the script's `#steps 50`; change that directive instead. For preview decoding,
replace `#no-preview-vae` with `#preview-vae` and choose the desired `#steps`.
The old Python-only `--h3cli`, `--preview`, `--no-overlay`, `--work-dir` and
`-o` sample options are replaced by `H3CLI_BIN`, script directives and
`--output-dir`.

To add numbered badges or render separate seed variants, uncomment the optional
file-wide directives before the first segment:

```text
#overlay-segment-name
#seed-set 100,200,300
```

Each seed-set value is added to the base seed of 42, keeping the seed constant
throughout each variant (142, 242 or 342 in this example). Each variant gets its
own continuation states and `seed-N/flight.mp4` under the output directory.
Remove the seed-set directive to render a single sequence. See the
[script-format guide](../../docs/features/sequence-rendering.md) for aliases,
prompt inheritance, skipped segments and `#nomerge`.

Native rendering retains the audio and removes inherited context once. It
validates delivered clips and merges them in script order. The output directory
also contains `1.mp4` through `50.mp4`, the continuation-chain movie
`1-merged.mp4`, `run.json`, and private states, sidecars and logs in `.render/`.
Media delivery uses the native renderer's merging and audio alignment.
Existing output directories are never reused;
choose a fresh `--output-dir` for another run. Without the launcher's explicit
output option, direct `--render` defaults to `flight-render` beside the script.

Run the model-free sample checks with:

```sh
make bin/render_driver
python3 tests/test_flight_sample.py
```

They validate both samples' native continuation plans and Panda's reference
alias, plus a short flight assembly using synthetic workers and real
FFmpeg/FFprobe. They do not establish
visual or sound continuity for the full generated flight.
