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

The [panda.txt](panda.txt) file is a native `h3cli --render` script. The launcher
calls h3cli directly; no Python sample runner is required. Source builds need
FFmpeg/FFprobe and their configured environment. Standalone packages include
the media tools.

From the repository root:

```sh
sh samples/panda/run.sh --models-path models --offline
sh samples/panda/run.sh --dry-run > /tmp/panda-plan.json
```

For a source build, follow the [build guide](../../docs/build/source-build.md)
and source `outputs/setup/macos-env.sh` on the Mac or
`outputs/setup/linux-env.sh` on Linux. Omit `--offline` to allow missing models
to download. `-d /path/to/MiniMaxH3` selects the main model directory separately.

The launcher finds the script relative to itself and works from any directory.
It selects `H3CLI_BIN` when set, otherwise the repository's executable
`bin/h3cli`, then `h3cli` on PATH. All arguments go directly to h3cli:

```sh
H3CLI_BIN=/path/to/h3cli sh samples/panda/run.sh \
  --models-path /path/to/models --output-dir outputs/panda-another-run
```

The equivalent direct invocation is:

```sh
./bin/h3cli --render samples/panda/panda.txt \
  --models-path models --output-dir outputs/panda-render
```

| Setting | Value |
| --- | --- |
| Render and delivery | 960×544 at 24 fps, full VAE, 50 denoising steps |
| Segments | 40 prompts, 243 internal frames each |
| Seed | 42 for every segment, inherited from the first segment's `#seed 42` |
| Conditioning | `#ref panda panda.png`, reused with `#ref-image panda` in every segment; high reference size |
| Continuation | `#continue` after the first segment, with 39 context frames |
| Delivered frames | 243 first, then 204 per continuation |
| Combined length | 8,199 frames, 5 minutes 41.625 seconds |
| Default movie | `outputs/panda-render/panda.mp4`, relative to the launch directory |
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
own continuation states and `seed-N/panda.mp4` under the output directory.
Remove the seed-set directive to render a single sequence. See the
[script-format guide](../../docs/features/sequence-rendering.md) for aliases,
prompt inheritance, skipped segments and `#nomerge`.

Native rendering retains the audio and removes inherited context once. It
validates delivered clips and merges them in script order. The output directory
also contains `1.mp4` through `40.mp4`, the continuation-chain movie
`1-merged.mp4`, `run.json`, and private states, sidecars and logs in `.render/`.
Media delivery uses the native renderer's merging and audio alignment.
Existing output directories are never reused;
choose a fresh `--output-dir` for another run. Without the launcher's explicit
output option, direct `--render` defaults to `panda-render` beside the script.
