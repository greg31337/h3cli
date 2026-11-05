# Text-script rendering

Use `h3cli --render <filename>` to render a sequence of shots and continuations.
The native CLI handles the script, saves each segment, and merges the rendered
segments in script order. Standalone macOS and Linux packages include the media
tools; no Python interpreter is needed.

```sh
./bin/h3cli --render scenes.txt -d /path/to/MiniMaxH3 --dry-run
./bin/h3cli --render scenes.txt -d /path/to/MiniMaxH3
./bin/h3cli --render scenes.txt --output-dir outputs/my-sequence --steps 20
```

For a complete example, see [samples/render-example.txt](../../samples/render-example.txt).
From the checkout root, inspect it with
`./bin/h3cli --render samples/render-example.txt --dry-run`.

Replace `./bin/h3cli` with your downloaded executable when using a release.
Source builds need their configured FFmpeg/FFprobe environment. The existing
model discovery, downloading and `--offline` behavior also apply to render mode.
See the [integration acceptance record](render-integration-results.md) for
tested platforms, package coverage and validation limits.

`--dry-run` validates the whole script and prints a JSON plan containing the
expanded prompts, effective commands, skipped segments and merge destinations.
It does not load models, start render workers or create output files. Native
CLI generation options provide initial defaults; script directives override
those defaults. Direct `--prompt`, `--output`, continuation-state overrides and
other operation modes cannot be combined with `--render`.

## Script format

```text
### first
#frames 56
#steps 2
#size 256x256
#offline
A red toy boat on a quiet pond.
### next
#continue
#continue-mode bridge
#continue-bridge-steps 1
The boat continues past the reeds.
### independent
#frames 22
A blue toy boat beside a wooden dock.
```

A `### NAME` heading starts `NAME.mp4`; an existing `.mp4` suffix is retained.
Names must be unique, contain no directories or control characters, and must
not collide with an enabled merge output. Collisions are rejected without
rendering, including names differing only by case.
The real run also checks the destination filesystem's Unicode filename aliases
before starting workers. Publishing a clip or merge never replaces an existing file.

Ordinary settings persist until changed. References and first/last-frame
anchors reset in each segment; repeat references on a continued Ref2VA segment
so its model mode matches the previous segment. `#continue` applies only to its
own segment and continues the immediately preceding segment. Multiple
independent shots and continuation chains can share one script. Media, model
and other directive paths are resolved beside the script; CLI paths are resolved
from the current working directory. Quoted paths can contain spaces.

`#refimage PATH`, `#refvideo PATH` and `#refaudio PATH` alias their hyphenated
native names. Other directives use native long option names without `--`.
`#size WIDTHxHEIGHT` sets both dimensions. `#unset OPTION` restores the native
default; `#unset size` restores both dimensions. Repeated reference options keep
their order. A new segment's first `#lora` replaces the inherited adapter list;
additional `#lora` directives append to that list.

Use `##` for comments and `\#` for a literal leading hash in a prompt. Prompts
can span multiple lines. Directives belong before prompt text, except for the
global `#nomerge` directive. Scripts must be UTF-8; a BOM and CRLF line endings
are accepted. Unknown directives, malformed values and invalid native option
combinations produce diagnostics before generation starts.

## Outputs and merging

By default, `scenes.txt` creates a fresh `scenes-render` directory beside the
script. `--output-dir DIR` selects a different fresh directory. Existing output
directories are never reused or overwritten.

The example above produces:

- `first.mp4`, `next.mp4` and `independent.mp4` for the individual segments.
- `first-merged.mp4` for the continuation chain.
- `scenes.mp4` containing **all rendered segments**, once each, in script order.
- `run.json` with effective prompts, commands, statuses, tool paths and merge
  results, plus logs, the original script and continuation states in `.render/`.

The combined filename is the input basename with only its final extension
removed, followed by `.mp4`. A script with one rendered segment still produces
this combined output. Individual clips and chain merges are retained.

Continuation clips exclude their inherited context. In this example, `next`
delivers 17 new frames after its 39-frame context: the chain contains 73 frames,
and `scenes.mp4` contains 95 frames at 24 fps. Audio is aligned to those delivered
frames. Compatible video streams are copied; differing encodings are normalized
to lossless RGB for merging. Different independent-shot dimensions are fitted
and padded to the first rendered segment's canvas without cropping. Original
clips retain their dimensions and encoding.

Add this directive **anywhere** in the script to disable every merge:

```text
#nomerge
```

It works before the first heading, after prompt text, at the end of the file,
or in a skipped segment. It applies to the whole file, including earlier
segments, and disables both chain merges and the script-named MP4. Individual
clips, continuation states and prompt inheritance still work. Repeating it is
harmless; it takes no arguments. `## #nomerge`, `\#nomerge` and inline mentions
in ordinary prose do not activate it. The directive itself is not prompt text.

Only verified complete merges receive their final filename. A render or merge
failure returns nonzero and records the error without presenting a partial
script as a completed combined video. Ctrl+C returns 130; SIGTERM returns 143.
Cancellation stops the active worker and its child processes, retaining completed
clips, states and diagnostic logs.

### Media validation

Validation reads the MP4/MOV container's stored frame count, dimensions, frame
rate and video/audio durations first. This avoids decoding every frame of each
segment, prepared merge input and final merge. A reported frame count that
conflicts with the planned count fails validation.

If the container has no usable frame count, h3cli scans only the video stream
to count decoded frames. Each scan has a **600-second** default deadline.
Set `H3_RENDER_SCAN_TIMEOUT` to a positive finite number of seconds to change
that limit. `H3_RENDER_VERIFY_FRAMES=1` requests a full video-frame scan even
when the stored count is available; the default is `0`.

```sh
H3_RENDER_SCAN_TIMEOUT=1200 H3_RENDER_VERIFY_FRAMES=1 \
  ./bin/h3cli --render scenes.txt
```

The initial metadata query retains a 120-second deadline for stalled probes.
Scan failures identify the file and probing phase, and cancellation stops the
active probe. `run.json` records the selected scan timeout and verification mode
under `validation`. Metadata checks validate container information; enable full
frame verification when decoded-video integrity needs checking too.

## Continued prompt sections

Recognized headings appear on their own lines, with optional surrounding
whitespace. Their bodies extend to the next recognized heading or the end of
that segment's prompt. Effective prompts emit populated sections in this order:

```text
subject_definitions:
...
summary:
...
retention_analysis:
...
detailed_description:
...
overall_soundscape:
...
non_diegetic_music:
...
```

A `#continue` segment automatically inherits each omitted section from the
previous segment's effective prompt. Writing a heading again replaces its entire
body; other sections still inherit. These values propagate through subsequent
continuations in hard and bridge modes. An explicitly empty section clears it
until a later segment supplies it again. A segment without `#continue` resets
the prompt-section context.

```text
### opening
#frames 56
#size 256x256
subject_definitions:
A red toy boat with a white sail.
summary:
The boat rests beside the reeds.
retention_analysis:
Keep the boat shape and sail color consistent.
detailed_description:
A fixed camera watches the boat on a quiet pond.
overall_soundscape:
Soft water and a light breeze.
non_diegetic_music:
A quiet piano melody.
### drifting
#continue
summary:
The boat drifts away from the reeds.
detailed_description:
The camera slowly follows the same boat into open water.
### farther
#continue
summary:
The boat reaches the center of the pond.
non_diegetic_music:
```

`drifting` replaces two sections and inherits the other four. `farther` keeps the
updated description, replaces the summary and clears the inherited music section.
The expanded prompts are visible in `--dry-run` and `run.json`.

Unstructured text before the first recognized heading remains the current
segment's preamble and is never inherited. Unrecognized headings remain ordinary
body text. Duplicate recognized headings in one prompt are errors. An active
segment must have a nonempty effective prompt after inheritance; a continuation
can omit its prompt entirely if inherited sections supply one. Multiline bodies
and their internal formatting are preserved.

## Skipping segments

Start a segment with `!### NAME` to skip it. Its directives still update inherited
settings, including `#unset`; references still reset. Skipped segments need no
prompt or existing reference files and produce no clip or state.

A continuation chain includes its first segment and all following `#continue`
segments. Every heading in that chain must be skipped, or every heading must be
rendered. Mixing them anywhere in the script is a preprocessing error. Skipped
chains never seed the prompt sections of a subsequent independent segment.

```text
!### omitted-first
#frames 56
#size 256x256
#steps 2
An omitted shot.
!### omitted-next
#continue
Its omitted continuation.
### included
A new independent shot using the inherited generation settings.
```

An entirely skipped script succeeds without invoking render or media workers
and without creating an output directory.
