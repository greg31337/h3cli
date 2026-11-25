# Text-script rendering

Use `h3cli --render <filename>` to render a sequence of shots and continuations.
The native CLI handles the script, saves each segment, and merges the rendered
segments in script order. Standalone macOS and Linux packages include the media
tools; no Python interpreter is needed.

On CUDA, `#size 1920x1088` and its portrait transpose use the extended native
canvas policy. Metal retains its previous ceiling. `--ref-image-size` is
inherited by segments and can be overridden with `#ref-image-size`; it sizes
reference images independently of `#size`. The
[native CUDA contract](../cuda/long-video-1450-contract.md) supports BF16 or NVFP4
and memory admission. Independent segments are qualified at the new canvas;
continuation/bridge above 1344×768 or transpose has host checks only.

`#frames` and `#seconds` follow the backend duration ceiling: **1,450 aligned
frames on CUDA, 362 on Metal**. Preserved continuation context is limited to
1,416 frames on CUDA or 345 on Metal and must be shorter than the target. The existing
upward alignment remains; `#seconds 60` resolves to 1,450 frames on CUDA.
Long native segments accept BF16 or NVFP4 with default attention; `#upscale`
requires BF16; see the
[duration contract](../cuda/long-video-1450-contract.md). Merge lengths remain the
sum of delivered segments and can exceed one segment’s duration.

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

## Upscaling before merging

Put `#upscale 0`, `#upscale 2`, `#upscale 3` or `#upscale 4` **before the first
segment definition**. The number selects refinement steps; the scale is always
2× on both axes. The setting applies to every rendered segment and may appear
only once. For example:

```text
#upscale 0
### first
#size 1344x768
#frames 39
#steps 30
A cat walks along a sunny sidewalk.
### next
#continue
#frames 56
The cat continues toward a garden gate.
```

On CUDA this delivers 2688×1536 clips and merges. With any `#continue` present,
including in skipped definitions, refinement must be **0**; values 2/3/4 fail
before output creation or generation. Independent segments support all four
values. Source and doubled target geometry must fit the selected backend;
Metal sources are limited to 960 pixels per axis for learned upscaling.

Generation stays at `#size`. For zero refinement, the worker upscales each
complete `.h3av` volume, including continuation context, preserves its audio,
and applies the saved prefix trim only when delivering the larger video.
The next continuation always reads the original-resolution state. For nonzero
refinement, generation captures a private `.h3up` source for the existing
refinement pipeline. Both paths require the full VAEs, dense BF16/default
attention, all 50 layers, no reuse or LoRA, and an exact internal render canvas.
Select `#quality lossless`, or use explicit steps and the ordinary dense defaults.

Overlays are applied after upscaling. Individual clips, chain merges and the
script-named movie all use the doubled canvas; `#nomerge` still upscales each
individual clip. `#upscale-model` or CLI `--upscale-model` selects the weights.
With refinement, `#upscale-noise` and `#upscale-seed` are forwarded only to the
upscale worker; zero refinement rejects both. `#upscale` manages source capture
and refinement steps, so separate `#save-upscale-state`, `#save-sampler-state`,
`#upscale-refine-steps` and sampler-import controls are rejected.

The dry-run JSON includes `command`, `upscale_command`, `source_width` /
`source_height`, and final `width` / `height`. During execution, `run.json`
records the upscale stage and its status. Private source clips/states and
upscaled states remain under `.render/`; an upscale or validation failure
preserves those files and already completed clips, and prevents merging.
See [latent upscaling](latent-upscaling.md) for memory limits and model setup.

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

Ordinary settings persist until changed. Reference directives before the first
segment apply to every segment. References added inside a segment apply only
to that segment, after the global references; first/last-frame anchors also
reset in each segment. A continued Ref2VA segment needs global or local
references so its model mode matches the previous segment. `#continue` applies
only to its own segment and continues the immediately preceding segment. Multiple
independent shots and continuation chains can share one script. Media, model
and other directive paths are resolved beside the script; CLI paths are resolved
from the current working directory. Quoted paths can contain spaces.

`#refimage PATH`, `#refvideo PATH` and `#refaudio PATH` alias their hyphenated
native names. Other directives use native long option names without `--`.
`#size WIDTHxHEIGHT` sets both dimensions. `#unset OPTION` restores the native
default; `#unset size` restores both dimensions. `#unset ref-image` (and the other
reference options) removes that type's global and local references for the
current segment; the global list returns in the next segment. Repeated reference
options and comma-separated entries keep their order. A new segment's first
`#lora` replaces the inherited adapter list; additional `#lora` directives append
to that list.

Use `##` for comments and `\#` for a literal leading hash in a prompt. Prompts
can span multiple lines. Directives belong before prompt text, except for the
global `#nomerge` directive. Scripts must be UTF-8; a BOM and CRLF line endings
are accepted. Unknown directives, malformed values and invalid native option
combinations produce diagnostics before generation starts.

## File-wide directives

These directives must appear **before the first `###` or `!###` heading**.
They can be combined, and `--dry-run` shows the expanded seeds, alias paths,
segment labels and output destinations.

```text
#overlay-segment-name
#seed-set 100,200,300
#ref fox "references/fox portrait.png"
#ref ambience references/forest.wav
#ref-image fox
#ref-audio ambience

### opening
#frames 56
#size 256x256
#seed 42
A fox stands in a snowy forest.
### walking
#continue
#seed 43
The fox walks toward the camera.
```

### Segment-name overlays

`#overlay-segment-name` adds the heading's name in white bitmap lettering on a
dark badge at the bottom left of each delivered clip, like the numbered badges
in the sample videos. The name excludes the `###` delimiters and does
not include an automatically added `.mp4` suffix or seed directory. An explicit
`.mp4` in the heading remains part of the label.

The badge appears in individual clips, continuation-chain merges and the
script-named movie, including when `#nomerge` disables merging. It is applied
after generation; saved continuation states stay free of overlays. Original
unlabeled clips remain in `.render/` and are listed as `original_video` in
`run.json`. Overlay encoding uses lossless H.264 RGB and copies the original
audio, preserving decoded pixels outside the badge without another audio encode.

The built-in alphabet needs no external fonts or FFmpeg `drawtext` support.
Names shrink and wrap to fit; names too long for the smallest lettering end
with an ellipsis. Non-ASCII characters appear as `[U+XXXX]` codepoint labels.
The overlay pipeline is checked before any generation worker starts.

### Seed sets

`#seed-set 100,200,300` renders the **whole sequence once per offset**, in the
listed order. Each effective seed is the segment's normal seed plus that
offset. Existing seed inheritance and CLI defaults still apply; absent an
override, the native seed is `42`. `#unset seed` restores that native default
before the offset is added. Offsets never accumulate between segments or runs.

In the example, the three sequences use seeds `142/143`, `242/243` and
`342/343`. Each sequence has independent continuation states and outputs in
`seed-100/`, `seed-200/` or `seed-300/` beneath the normal output directory.
Each directory contains its individual clips, chain merges and script-named
movie. A sequence's merges finish before the next seed starts; different seed
runs are never merged together. The top-level `run.json` records all runs with
their `seed_offset` and effective `seed` values.

Use one seed-set directive with one to 1,024 distinct unsigned 64-bit integers;
whitespace around commas is accepted, and `0` is a valid offset. Empty entries,
duplicates, negative values and additions that would overflow the seed range
are rejected before creating outputs. With no seed set, output layout and seed
behavior are unchanged. `#nomerge` applies to every seed run.

### Reference aliases

`#ref ALIAS FILE_PATH` defines a case-sensitive alias. Paths resolve beside
the script and may be quoted to contain spaces. Alias names must be nonempty
and contain no whitespace, directory separators or `#`; duplicate definitions
are errors. Definitions do not activate a reference or change the model mode.

Use an alias wherever a script directive accepts a media path: `#ref-image`,
`#ref-video`, `#ref-silent-video`, `#ref-audio`, both paths of `#ref-video-audio`,
or `#first-frame` and `#last-frame`. The `#refimage`, `#refvideo` and `#refaudio`
spellings work too. Define an alias before using it in a reference directive.

An exact alias match takes precedence over a literal filename. Use `./ALIAS`
to refer to a same-named file directly. Unmatched values remain ordinary paths;
aliases do not expand inside prompts, model paths or other settings. An alias
definition's path is literal, not another alias. Unused aliases and aliases used
only by skipped segments do not require existing files; active references are
validated before rendering.

### Global references and file lists

Place `#ref-image`, `#ref-audio`, `#ref-video`, `#ref-silent-video` or
`#ref-video-audio` before the first heading to apply those references to every
segment, including continuations and each seed-set run. The shorthand spellings
work here too. A segment starts with the global references in declaration order,
then appends its own references. Local additions never carry into the next
segment, including additions in skipped segments. `#first-frame`, `#last-frame`
and ordinary settings such as `#ref-image-size` remain segment directives.

Each reference directive accepts a comma-separated list of aliases or paths:

```text
#ref body1 references/body.png
#ref face1 "references/face portrait.png"
#ref-image body1, face1, path/file.png

### opening
#ref-image "references/prop, closeup.png", references/background.png
A character holds a prop.
### next
The character walks away.
```

Both segments use the three global images; only `opening` uses the two extra
images. Quote filenames containing spaces, commas or `#`; an unquoted escaped
comma (`\,`) also denotes a literal comma. Empty entries and trailing commas
are errors. Lists can be combined with repeated directives, and each item
resolves independently beside the script or through its alias.

For paired references, each entry contains **VIDEO AUDIO**, for example:

```text
#ref-video-audio video1.mp4 audio1.wav, "video two.mp4" "audio two.wav"
```

Global and local references together must satisfy the native limits of nine
images, three videos and three audio inputs. Standalone reference audio still
requires an image or video reference. CLI media defaults retain their existing
first-segment scope; a script reference directive replaces CLI defaults of the
same type, while local script references append to the global script list.

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
