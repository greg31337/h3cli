# Integrated rendering acceptance

All eight tasks in the [integrated rendering milestone](../todo.md) are
complete. Validation ran on 2026-11-03–04. The
[sequence-rendering guide](sequence-rendering.md) describes the resulting
command and script format.

## Implementation

`h3cli --render scenes.txt` parses and validates the complete script before
generation, runs segments through the existing native generation/continuation
path, and produces `scenes-render/scenes.mp4`. `--dry-run` exposes effective
prompts and settings without loading models or creating output files.

The native option registry supplies directive names, aliases and validation.
Standalone executables contain the implementation and use their bundled media
tools; running a script requires no Python installation.

Continued prompts inherit omitted structured sections, replace explicitly
supplied sections, clear explicitly empty sections, and emit the six recognized
sections in canonical order. Independent segments reset this context. A
standalone `#nomerge` anywhere in the file disables both chain and script-wide
merges, including when it appears after prompt text or inside a skipped segment.

Merges include each delivered clip once in script order. Compatible video
streams are copied; incompatible encodings or canvases are normalized with
lossless RGB encoding and fit/pad to the first active canvas. Audio is aligned
to each clip's delivered duration before concatenation. Media validation checks
frame counts, 24 fps, dimensions and audio duration before atomic publication.
Completed clips, states, logs and `run.json` survive failures and cancellation.

Output-name checks include the destination filesystem's Unicode and case
equivalence. Publication refuses to replace an existing destination even if
another process creates it after preflight. This prevents differently encoded
Unicode filenames on APFS from silently overwriting an earlier clip.

## Platforms and coverage

The Mac was an Apple M4 Max with 128 GiB unified memory, running macOS 26.6.2
(25G83). The CUDA host ran Ubuntu 24.04.5 LTS with an NVIDIA RTX PRO 5000 72GB
Blackwell and driver 595.99.02. Both used the installed MiniMax-H3 model.
Current rendering used FFmpeg/FFprobe 9.0.2. The historical CUDA gate selected
its separately verified reference codecs as required by
[CONTRIBUTING.md](../../CONTRIBUTING.md).

The table records the original milestone runs. Subsequent entry-point cleanup
and its validation are recorded below.

| Check | Mac source | Mac standalone | CUDA source | Linux standalone |
|---|---|---|---|---|
| Native parser/CLI suite | 24 passed | 24 passed | 24 passed | 24 passed |
| Synthetic media/lifecycle suite | 12 passed | — | 12 passed | — |
| Current host and CLI suites | Passed | — | Passed; two skips below | — |
| Bounded real-model render | Passed | Passed | Passed | Passed |
| Unchanged CUDA reference outputs | — | — | 204/204 exact | 204/204 exact |

The Linux host suite skipped one Metal-only rejection case and one LoRA
clone-quota cancellation case because its filesystem did not provide cloned
files. Both ran on the Mac. No render parser, media or real-render cases were
skipped, and the CUDA reference gate has no skip mechanism.
The Linux standalone launcher suite passed 15 cases; its `noexec`-cache case
was skipped both inside the builder and on the host because neither provided
the required `/dev/shm` mount. That existing launcher case remains unqualified
by this run.

The parser suite covers all six sections, transitive hard/bridge inheritance,
partial/full/empty overrides, preambles, multiline bodies, canonical order,
independent resets, duplicate headings, UTF-8 validation, option precedence,
reference resets, LoRA lists, skipped chains and path handling.
It exercises every allowed `#nomerge` location plus repeated, escaped, comment
and inline forms. Invalid scripts and incompatible native modes fail before
any worker starts.

The 12 synthetic-media tests use the production runner and a test-only worker.
They compare every decoded RGB frame for compatible streams and mixed
encodings, verify audio tone order, check fit/pad behavior for mixed dimensions,
and cover one-clip output, `#nomerge`, render/merge failures, missing media tools,
existing output directories, filesystem aliases, concurrent destination
creation and cancellation of workers and descendants. The final Mac media run
also passed under AddressSanitizer and UndefinedBehaviorSanitizer. A separate
parser sanitizer run passed 104 valid/malformed cases (seed 42); it preceded the
filesystem publication fix, which did not change the parser.

## Real-model exercise

[`tests/render_integration.py`](../../tests/render_integration.py) uses two
denoising evaluations per segment. It renders a 256×256, 56-frame opening,
a hard continuation and a bridge continuation, followed by an independent
256×128, 22-frame shot. Each continuation delivers 17 new frames after trimming
its inherited context. The chain merge contains 90 frames and the complete
script contains 112, both at 24 fps with synchronized audio. The mixed canvas
exercises normalization. The report's expanded prompts must match the dry-run
plan, including inherited sections and the independent shot's reset.

A second script renders 22 frames with `#nomerge` after its prompt and must
produce only its individual clip. Package tests use an empty PATH, remove
media-tool overrides and shader-path overrides, and extract into a fresh
runtime cache. Each test verifies that its executable remains unchanged.

Mac runs used the existing `--ssd-streaming` option. An initial resident-weight
attempt was rejected by the existing memory headroom check under the machine's
current workload; resident mode is not claimed as passing this exercise.
The two-evaluation renders validate behavior and media structure, not final
visual quality or production-length throughput.

| Artifact | Real-model elapsed time | Executable SHA-256 |
|---|---:|---|
| Mac source | 147.03 s | `f5db8794bd7b46187af652cccc26b98127b85efe06e0a446e60964eba473f114` |
| Mac standalone | 149.26 s | `7f78a9eded61287be8e3df9f7c5b522408fcc0b8cb5b1f5104f5809d57276f07` |
| CUDA source | 182.14 s | `1274e3212e5886bcc19e7de588962e43ff4f7446a777330ca69f8f984249da89` |
| Linux standalone | 198.18 s | `54179e3916544c002cffabc92ef448d472d69f1281c1e1cd68a203f9ea3ec138` |

The script SHA-256 is
`be4951b55cb4690bf18a10f2e4c1b3e59f16e90488d45dc2000b4b6f9b9e7c02`.
The Mac source and standalone script-wide MP4s are byte-identical:
`140651f446262b265e1bb129cc5c1fc25795acf3300a0fe9356066eb172fa5f8`.
The CUDA source MP4 SHA-256 is
`4ee56aed34fad9bd799c8d46d5cad31da6edbe34484cfdb621a89efb15e04d7c`.
The Linux standalone MP4 SHA-256 is
`639cc61a5a43dea3452fe32f9af8d9bb21db80410d38fa29db9455d1f7fb4012`.
Cross-platform pixel equality is not part of this test.

The final source CUDA gate passed all 204 outputs in 109.00 seconds after its
isolated build (204.02 seconds including build). Its source fingerprint is
`0233ce1430b34a43efc9188a4419590e7b6c4d1e61d42d9a672ee7dd78514e75`;
the unchanged golden manifest is
`fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3`.
The final standalone executable and its matching probes passed the same 204
outputs in 128.17 seconds (148.69 seconds including artifact verification).
Both gates recorded the same source and golden-manifest fingerprints. No
golden outputs, fixtures or tolerances were changed.

## Reproduction and retained evidence

From a configured source build, with provisioned models and explicit media
paths when FFmpeg/FFprobe are not on PATH:

```sh
make -j8 test-current-host test-cli
python3 tests/render_integration.py --binary bin/h3cli \
  --model /path/to/MiniMax-H3 --out outputs/render-validation-source
python3 tests/cuda_reference_regression.py --source . \
  --model /path/to/MiniMax-H3 --out outputs/render-reference
```

Enable the qualified CUDA build's Sage, SOL and SubBlock options for its host
suite. Add `--ssd-streaming` to the integration command on a Mac requiring
weight streaming. To exercise a standalone package:

```sh
H3_RENDER_BIN=/path/to/h3cli-platform python3 tests/test_render_script.py
python3 tests/render_integration.py --binary /path/to/h3cli-platform \
  --model /path/to/MiniMax-H3 --packaged \
  --out outputs/render-validation-package
```

The release builders were run offline with pinned source dependencies. Final
development packages reused those newly built dependencies after verifying
that their dependency recipes were unchanged. These are test packages; this
record does not claim a new dependency qualification or a published release.
The Mac executable is ad-hoc signed, without notarization.

Raw logs, plans, reports, hashes and build metadata are retained under ignored
`outputs/render-integration/`. They include `current-host-macos-final.log`,
`media-macos-final-02/result.json`, `native-macos-final/result.json`,
`package-macos-final/result.json` and `final-code.json`. The `linux-evidence/`
subdirectory contains the final host/parser/launcher logs, source and standalone
render results, build metadata, and both complete CUDA gate reports. Private
host paths and connection details are excluded from this report.

## Cleanup after integration

On 2026-11-03, the compatibility entry point and its two dedicated tests were
removed. `--render` is the single command for text-script rendering. The
[example script](../../samples/render-example.txt) moved to `samples/`, with
updated commands and relative reference-image paths. The original 24-test
totals above include the two retired compatibility checks.

All **22 remaining parser tests** passed on the Mac and RTX PRO 5000 source
builds and on the previously built standalone executables. The relocated
example passed dry runs on both hosts: four segments, three merge destinations,
both reference images resolved, and no output directory created.

A fresh complete CUDA reference regression passed **204/204** unchanged golden
outputs in 98.03 seconds (195.37 seconds including its isolated build). Its
source fingerprint is
`61a8c9af54d5918c5a473a1e6519f148b427910756c866269208460458b99c79`.
The native rendering implementation and golden manifest were unchanged.
Cleanup logs, sample plans and the full gate report are retained under ignored
`outputs/render-cleanup-20261103/`.
