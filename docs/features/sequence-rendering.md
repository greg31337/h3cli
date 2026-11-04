# Standalone sequence rendering

`render.py` works beside an existing standalone `h3cli` executable. It contains
its option metadata and uses only the Python standard library. It does not need
`options.def`, the h3cli source tree, a compiler, a CUDA toolkit or pip packages.
It never installs software.

## Prerequisites

- Python 3.10 or newer.
- A working h3cli executable and the GPU driver required by that executable.
- The model weights required for the requested rendering modes.
- FFmpeg and FFprobe. The standalone h3cli packages already include both; the
  script discovers them through the existing launcher's runtime metadata.

On Ubuntu/Debian, if Python is missing, these are manual installation commands:

```sh
sudo apt update
sudo apt install python3
```

Only if the executable has no bundled media tools and system tools are missing:

```sh
sudo apt install ffmpeg
```

The `ffmpeg` package supplies FFprobe too. No installation command is executed
by the script. The existing h3cli launcher may populate its normal runtime cache
when first run; it uses the runtime already contained in the executable.

## Run

```sh
./render.py --check
./render.py scenes.txt -d /path/to/MiniMaxH3 --dry-run
./render.py scenes.txt -d /path/to/MiniMaxH3 -o sequence-output
```

`--check` verifies the binary and media tools without rendering or loading model
weights. `--dry-run` prints and validates the plan without running the binary or
creating output files. Rendering requires a fresh output directory.

The script looks for `h3cli`, `h3cli-linux-x86_64`, or `h3cli-macos-arm64` beside
it, then a development `bin/h3cli`, then `h3cli` on PATH. Use `--h3cli PATH` or
`H3CLI_BIN` to override this. Explicit `--ffmpeg`/`--ffprobe` options or
`H3_FFMPEG`/`H3_FFPROBE` take precedence over bundled tools and PATH tools.
Shell profiles load only with `--env-file PATH`; old `--no-env` invocations remain
valid. No checkout environment is loaded automatically.

## Input format

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

This produces three clips and `first-merged.mp4` for the continuation chain.
Ordinary settings persist until changed; references and first/last-frame
anchors reset in each section. Repeat references on a continued Ref2VA segment
so its model mode matches the previous segment. `#continue` applies only to its
own section. Multiple independent clips and continuation chains can share one
input file. Relative media paths are resolved beside that file.

`#refimage PATH` aliases `#ref-image PATH`, and other directives use native long
option names without `--`. `#unset OPTION` removes an inherited setting. Use
`##` for comments and `\#` for a literal leading hash in a prompt.

Start a section with `!### NAME` to skip its rendering. Its directives still
parse and update inherited settings normally, including `#unset`; references
still reset in each section. Skipped sections do not need a prompt or existing
reference files and produce no clips or saved states.

A continuation chain includes its first segment and every following segment
marked `#continue`. Either every header in that chain must use `!###`, or every
header must use `###`. Mixing them is a preprocessing error, even if the mixed
chain appears after an otherwise valid chain; no rendering starts first.

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
A new, independent shot using the inherited settings.
```

An entirely skipped input finishes successfully without invoking h3cli or
creating an output directory.

Continuation output excludes its context automatically: the earlier three-clip
example's second clip has 17 new frames after its 39-frame context, for 73 merged
frames at 24 fps.
The output retains clips, continuation states, logs and `run.json`, including
the exact binary and media-tool paths used. Ctrl+C stops the owned render
process group and preserves completed work.
