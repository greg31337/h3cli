# Integrated text-script rendering

Status: **complete — 8/8 tasks complete**.
Updated 2026-11-03.

Text-script rendering is built into h3cli as `--render <filename>`.
The native command renders the text script's segments, merges all rendered
segments into `<text_script_name>.mp4` by default, and inherits structured
prompt sections through continued segments unless the new prompt overrides them.

The completed **8/8** release-tooling milestone is
[archived intact](release-tooling-todo.md), with its
[acceptance evidence](release/results.md). The requirements below describe
implemented behavior; the [sequence-rendering guide](features/sequence-rendering.md)
documents the native command. The
[rendering acceptance record](features/render-integration-results.md) records
source and standalone checks on the Mac and RTX PRO 5000, including the
unchanged 204-output CUDA reference gate.

## Built-in rendering

```sh
./bin/h3cli --render scenes.txt -d /path/to/MiniMaxH3
./bin/h3cli --render scenes.txt --dry-run
```

- Make `--render` a first-class CLI mode in source builds and standalone macOS
  and Linux packages. It must work without a helper script, a source
  checkout, or a user-installed Python interpreter. Reuse the native option
  registry, generation/continuation paths and packaged FFmpeg/FFprobe support.
- Preserve the script format and behavior: `### NAME`, `!### NAME`, native
  option directives and aliases, inherited settings, `#unset`, per-segment
  reference/anchor resets, relative paths beside the script, comments, escaped
  hashes, multiline prompts, hard/bridge continuation and context trimming.
  Retain the rule that a continuation chain is entirely rendered or skipped.
- Use native CLI generation options as initial defaults, with script directives
  overriding them. Validate incompatible operation modes and options managed by
  the script, including direct prompt/output and continuation-state overrides,
  before rendering. Keep ordinary single-video CLI behavior unchanged.
- Retain the default fresh output directory `<text_script_name>-render` beside
  the input and expose `--output-dir` in render mode for an explicit directory.
  Preserve individual clips, continuation states, logs and `run.json`.
  `<text_script_name>` means the input basename with its final extension removed:
  `scenes.txt` produces `scenes-render/scenes.mp4` by default.
- Validate the entire script before starting generation. A dry run shows the
  effective prompts, settings, skip decisions and merge destinations without
  loading models or creating output files. Failures and cancellation return
  nonzero, stop owned workers and retain completed clips and diagnostics.

## Merge all rendered segments

- After all active segments succeed, concatenate their delivered clips once
  each, in script order, into `<text_script_name>.mp4`. Include independent shots
  and every continuation chain. Omit skipped segments and avoid repeating
  continuation context or including both a chain merge and its source clips.
- Preserve the existing `<first_segment>-merged.mp4` chain outputs by default;
  the script-wide file is an additional output. Keep the original clips.
- Recognize a standalone `#nomerge` directive anywhere in the text script:
  before the first segment, inside any segment (including a skipped one), after
  prompt text, or at the end. Scan the complete file before execution. Its
  presence disables all merged outputs for the entire run, including chain
  merges; it does not affect generation, continuation, or prompt inheritance.
  Repeated directives are harmless. It takes no arguments and is removed from
  prompt text. Comments (`## ...`), escaped literal text (`\#nomerge`) and inline
  mentions in prose do not activate it.
- With one rendered segment, still produce the script-named MP4. With every
  segment skipped, succeed without rendering or creating outputs. Reject
  collisions among segment names, chain outputs and the script-wide filename
  before rendering; disabled merge destinations do not reserve filenames.
- Preserve audio/video order and synchronization. The merged frame count must
  equal the sum of delivered segment frames at the existing 24 fps. Normalize
  incompatible encodings when needed. For independent shots with different
  dimensions, use the first rendered segment's canvas and fit/pad other shots
  without cropping or changing their duration; individual clips stay intact.
- Publish the final filename only after the complete merge passes media
  validation. Report render or merge failures accurately; a partial script
  cannot leave a file presented as the successful full result. Record the merge
  inputs, normalization and final output, or the `#nomerge` decision, in `run.json`.

## Structured prompt inheritance

Recognize these prompt section headings and emit populated sections in exactly
this order, regardless of their order in the source prompt:

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

- A heading occupies its own line, with optional surrounding whitespace. Its
  body is all following text up to the next recognized heading or the end of
  that segment's prompt. Preserve multiline bodies and internal formatting.
  These are prompt sections, not `#` option directives.
- A `#continue` segment inherits each omitted section from the immediately
  preceding segment's **effective** prompt. Carry inherited or replaced values
  forward through any number of continued segments, in hard and bridge modes.
  A segment without `#continue` starts a fresh prompt-section context.
- A heading present in the new prompt replaces that entire inherited section;
  other sections still inherit. An explicitly empty section clears that section
  for this and subsequent continuations until it is supplied again. Do not
  manufacture missing sections or retain both old and new copies.
- Keep the current segment's unstructured preamble before the ordered sections;
  never inherit that preamble. Unrecognized headings are ordinary body text.
  Prompts without recognized headings retain their existing behavior when
  there are no structured sections to inherit.
- Reject duplicate recognized headings within one prompt with a script line
  diagnostic. Resolve inheritance before checking whether an active segment's
  effective prompt is empty. Fully skipped chains cannot seed later independent
  prompts, and mixed skipped/rendered chains remain invalid.
- Pass the expanded prompt to generation and include it in the dry run and
  execution report so users can inspect what each segment actually receives.

For example, in `scenes.txt`:

```text
### opening
#frames 56
#size 256x256
subject_definitions:
A red toy boat with a white sail.
summary:
The boat rests beside the reeds.
retention_analysis:
Keep the boat's shape and sail color consistent.
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
```

`drifting` inherits four sections and replaces `summary` and
`detailed_description`. `farther` inherits the resulting five omitted sections,
including `drifting`'s new `detailed_description`, and replaces only `summary`.
All effective prompts use the six-section order above. The default outputs
include `opening.mp4`, `drifting.mp4`, `farther.mp4`, `opening-merged.mp4` and
`scenes.mp4`. Adding `#nomerge` anywhere as a directive leaves only the three
individual MP4s, plus the normal states, logs and report.

## Implementation milestone

- [x] RND001 Add `--render <filename>`, render-mode `--output-dir` and
  `--dry-run` to the native CLI inventory, help and dispatch. Define option
  precedence and reject incompatible modes before model initialization.
- [x] RND002 Integrate script parsing and planning with the native option
  registry. Preserve existing directives, path handling, skip/continuation
  rules and diagnostics; parse whole-file `#nomerge` before prompt processing.
- [x] RND003 Implement structured-section parsing, transitive inheritance,
  section replacement/clearing, independent-segment resets and canonical order.
  Expose the expanded prompt in dry runs and reports.
- [x] RND004 Execute plans through the existing generation and continuation
  paths, preserving saved states, trimmed clips, logs, progress and cancellation
  behavior. Use `--render` as the single entry point for text-script rendering.
- [x] RND005 Implement the script-wide merge, preserve chain merges, and apply
  `#nomerge` globally. Handle audio, mixed encoding/canvas normalization, skipped
  segments, filename collisions, one-clip inputs and atomic final publication.
- [x] RND006 Add host/parser and synthetic-media coverage. Test every inherited
  heading, canonical order, partial/full/empty overrides, multiline bodies,
  transitive propagation, independent resets, preambles and duplicate headings.
  Cover `#nomerge` at every allowed location, repeated/literal/comment forms,
  scripts with multiple independent shots and chains, zero/one active clip,
  paths with spaces, merge frame/audio correctness and failure/cancellation.
  Port relevant coverage from [script tests](../tests/test_render_script.py) and
  [media tests](../tests/render_script_media.py).
- [x] RND007 Validate source and standalone macOS/Linux entry points, packaged
  media discovery and bounded real-model rendering with continuation. Run the
  applicable host/current-feature checks and the unchanged full **204-output
  CUDA reference regression** required by [CONTRIBUTING.md](../CONTRIBUTING.md).
  Record actual evidence and limitations; missing prerequisites are not passes.
- [x] RND008 Update CLI help, the project README, usage and sequence-rendering
  guides, and [samples/render-example.txt](../samples/render-example.txt) with
  the built-in command, output locations, global `#nomerge` behavior and prompt
  inheritance/override examples. Record acceptance against this checklist.

All eight tasks are complete. The [acceptance evidence](features/render-integration-results.md)
confirms that the built-in command renders scripts, creates the complete
script-named MP4 by default, honors global `#nomerge`, and supplies expanded
prompts to continued segments on both supported platforms. The record includes
test limits and unrelated platform/filesystem skips.
