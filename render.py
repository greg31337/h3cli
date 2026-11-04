#!/usr/bin/env python3
"""Render a sectioned text file with h3cli (standalone, Python 3.10+).

Usage: ./render.py scenes.txt [-o OUTPUT_DIRECTORY] [--dry-run]
       ./render.py --check

### NAME starts NAME.mp4. Directives before the prompt use native long option
names without '--', e.g. #frames 243, #steps 50, #lora adapter.safetensors:0.8.
!### NAME skips rendering that segment while still processing its directives.
Each continuation chain must be entirely skipped or entirely rendered; mixed
chains fail during preprocessing, before any rendering or output creation.
#size WIDTHxHEIGHT and #refimage PATH are shortcuts. Quote paths with spaces.
Ordinary settings persist; references and first/last-frame anchors reset in
every section. Repeat a reference directive to supply multiple references.
#unset OPTION removes an inherited setting. Repeating #lora in a new section
replaces the inherited adapter list, then appends further adapters in order.

#continue links ONLY this section to the immediately previous section. Use
#continue-mode bridge to select bridge instead of the native hard default.
Continuation controls persist but are passed only to continued sections.
Each chain gets FIRST_NAME-merged.mp4; independent sections remain separate.
References must be listed again on reference-based continuations: h3cli cannot
switch between Ref2VA and FL2VA within a chain. Geometry must also match.

Relative directive paths are relative to the text file. Output defaults to
TEXT_STEM-render beside that file; choose a fresh directory for each run.
Individual clips, states, logs, source text and run.json are retained. Native
frame alignment applies; continuation clips exclude their context automatically
(243 frames with context 39 delivers 204 new frames). Seeds persist unchanged.
Use ## for comments; escape a leading prompt hash as \\#.

Copy this script beside the existing h3cli executable; no source checkout,
options.def, Python packages, compiler or CUDA toolkit is required. Packaged
FFmpeg/FFprobe are used automatically, with PATH tools as a fallback. Explicit
--ffmpeg/--ffprobe or H3_FFMPEG/H3_FFPROBE settings take precedence.
--check verifies the executables without rendering. Nothing is installed.
Shell environment files are loaded only when --env-file is supplied; --no-env
remains accepted for compatibility. --dry-run validates the plan and prints
commands without invoking h3cli, extracting its runtime, or creating outputs.
"""

import argparse
import copy
import csv
from dataclasses import dataclass, field
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
FPS = 24
ALIASES = {"refimage": "ref-image", "refvideo": "ref-video", "refaudio": "ref-audio",
           "d": "model-dir", "p": "prompt", "o": "output"}
EXCLUSIVE = (("frames", "seconds"), ("preview-vae", "no-preview-vae"),
             ("ffmpeg-crf", "output-quality", "lossless-video"))
MANAGED = {"prompt", "output", "continue-from", "save-av-state", "keep-continuation-prefix"}
NON_VIDEO = {"help", "info", "server", "still", "state-only", "decode-av-state",
             "decode-still-latent", "save-still-latent", "resume-sampler-state",
             "stop-after-step", "preview-on-stop", "upscale-state", "inspect-upscale-state",
             "download-models", "list-models"}


# Snapshot of src/cli/options.def; tests check that it remains in sync.
# Embedded so this one script runs without any source or metadata sidecars.
OPTION_CSV = """output-quality,1,ENUM,NONE,0,0,0,default|maximum|high|medium|low,NULL
ffmpeg-crf,1,INT,NONE,0,0,51,NULL,NULL
lossless-video,0,BOOL,NONE,0,0,0,NULL,false
quality,1,ENUM,NONE,0,0,1.7976931348623157e308,lossless|extra-high|high|preview|fast-preview,NULL
no-preview-vae,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
model-dir,1,STRING,MODEL,0,0,1.7976931348623157e308,NULL,models/MiniMaxH3
models-path,1,STRING,NONE,0,0,0,NULL,models
download-models,1,STRING,NONE,0,0,0,NULL,NULL
list-models,0,BOOL,NONE,0,0,0,NULL,false
offline,0,BOOL,NONE,0,0,0,NULL,false
backend,1,ENUM,NONE,0,0,1.7976931348623157e308,mpsgraph|metal,mpsgraph
metal-attention,1,ENUM,NONE,0,0,1.7976931348623157e308,dense|sol,dense
metal-ane,1,ENUM,NONE,0,0,1.7976931348623157e308,off|serial|static|dynamic,off
metal-ane-rows,1,INT,NONE,0,0,2147483647,NULL,4096
metal-ane-chunk,1,INT,NONE,0,0,2147483647,NULL,512
metal-weight-format,1,ENUM,NONE,0,0,1.7976931348623157e308,bf16|q8,bf16
metal-q8-kernel,1,ENUM,NONE,0,0,1.7976931348623157e308,mpsgraph|simdgroup,mpsgraph
metal-attention-kernel,1,ENUM,NONE,0,0,1.7976931348623157e308,steel|steel-routed,steel
metal-attention-dtype,1,ENUM,NONE,0,0,1.7976931348623157e308,bf16|fp16,bf16
metal-tier,1,ENUM,NONE,0,0,1.7976931348623157e308,diagnostic|reference|preview,diagnostic
metal-attention-layout,1,ENUM,NONE,0,0,1.7976931348623157e308,adapter|fused,adapter
sol-q-block,1,INT,NONE,0,0,2147483647,NULL,32
sol-kv-block,1,INT,NONE,0,0,2147483647,NULL,64
sol-tau,1,FLOAT,NONE,0,0,1.7976931348623157e308,NULL,1
sol-dense-layers,1,INT,NONE,0,0,2147483647,NULL,1
sol-dense-steps,1,INT,NONE,0,0,2147483647,NULL,NULL
sol-dense-sigma,1,FLOAT,NONE,0,-1,1,NULL,-1
sol-local-radius,1,INT,NONE,0,0,2147483647,NULL,1
sol-min-exact,1,FLOAT,NONE,0,0,1,NULL,0
save-conditioning,1,STRING,OUTPUT,0,0,1.7976931348623157e308,NULL,NULL
load-conditioning,1,STRING,STATE,0,0,1.7976931348623157e308,NULL,NULL
conditioning-schedule,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
lora,1,STRING,MODEL,1,0,1.7976931348623157e308,NULL,NULL
lora-cache,1,STRING,CACHE,0,0,1.7976931348623157e308,NULL,NULL
lora-memory-mib,1,U64,NONE,0,1,1.7976931348623157e308,NULL,512
preview-vae,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
preview-vae-model,1,STRING,MODEL,0,0,1.7976931348623157e308,NULL,models/preview-vae/taeh3.safetensors
decode-still-latent,1,STRING,STATE,0,0,1.7976931348623157e308,NULL,NULL
image-vae,1,STRING,MODEL,0,0,1.7976931348623157e308,NULL,models/image-vae/minimax_h3_t1_image_vae_step1597.safetensors
still,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
save-still-latent,1,STRING,OUTPUT,0,0,1.7976931348623157e308,NULL,NULL
decode-av-state,1,STRING,STATE,0,0,1.7976931348623157e308,NULL,NULL
prompt,1,STRING,NONE,0,0,1.7976931348623157e308,NULL,NULL
output,1,STRING,OUTPUT,0,0,1.7976931348623157e308,NULL,NULL
continue-from,1,STRING,STATE,0,0,1.7976931348623157e308,NULL,NULL
continue-context,1,INT,NONE,0,0,2147483647,NULL,39
continue-mode,1,ENUM,NONE,0,0,1.7976931348623157e308,hard|bridge,hard
continue-bridge-steps,1,INT,NONE,0,0,2147483647,NULL,NULL
continue-bridge-max-strength,1,FLOAT,NONE,0,0,1,NULL,NULL
continue-bridge-profile,1,ENUM,NONE,0,0,1.7976931348623157e308,stepped|linear|ease-out,NULL
save-upscale-state,1,STRING,OUTPUT,0,0,1.7976931348623157e308,NULL,NULL
state-only,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
upscale-state,1,STRING,STATE,0,0,1.7976931348623157e308,NULL,NULL
upscale-model,1,STRING,MODEL,0,0,1.7976931348623157e308,NULL,models/latent-upscale/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors
upscale-refine-steps,1,INT,NONE,0,0,2147483647,NULL,4
upscale-noise,1,FLOAT,NONE,0,0,1,NULL,0.25
upscale-seed,1,U64,NONE,0,0,1.7976931348623157e308,NULL,NULL
upscale-import-sampler,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
inspect-upscale-state,1,STRING,STATE,0,0,1.7976931348623157e308,NULL,NULL
save-av-state,1,STRING,OUTPUT,0,0,1.7976931348623157e308,NULL,NULL
stop-after-step,1,INT,NONE,0,0,2147483647,NULL,NULL
save-sampler-state,1,STRING,OUTPUT,0,0,1.7976931348623157e308,NULL,NULL
resume-sampler-state,1,STRING,STATE,0,0,1.7976931348623157e308,NULL,NULL
preview-on-stop,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
keep-continuation-prefix,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
width,1,INT,NONE,0,1,2147483647,NULL,864
height,1,INT,NONE,0,1,2147483647,NULL,480
render-width,1,INT,NONE,0,0,2147483647,NULL,0
render-height,1,INT,NONE,0,0,2147483647,NULL,0
frames,1,INT,NONE,0,1,2147483647,NULL,56
seconds,1,FLOAT,NONE,0,0,1.7976931348623157e308,NULL,NULL
steps,1,INT,NONE,0,1,2147483647,NULL,20
reuse,1,INT,NONE,0,0,2147483647,NULL,1
layers,1,INT,NONE,0,0,2147483647,NULL,50
core-reuse,1,INT,NONE,0,0,2147483647,NULL,1
adaptive-cache,1,ENUM,NONE,0,0,1.7976931348623157e308,off|conservative|aggressive,off
adaptive-cache-threshold,1,FLOAT,NONE,0,0,1,NULL,NULL
adaptive-cache-max-hits,1,INT,NONE,0,1,16,NULL,NULL
adaptive-cache-max-mib,1,U64,NONE,0,1,1.7976931348623157e308,NULL,4096
adaptive-cache-warmup,1,INT,NONE,0,2,16,NULL,4
subblock-warmup,1,INT,NONE,0,2,16,NULL,10
subblock-sparsity,1,FLOAT,NONE,0,0,1,NULL,0.75
token-reduction,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
ssd-streaming,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
use-int8-row-fc2,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
use-reference-rope,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
use-slower-bf16-mlp,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
use-slower-bf16-qkv,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
use-slower-bf16-attention-output,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
use-slower-row-major-attention-output,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
use-slower-unfused-int8-inputs,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
use-slower-unfused-qkv-rope,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
use-slower-scalar-qkv-rms,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
use-slower-uncached-int8-scales,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
use-slower-dynamic-fc1-k,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
use-slower-grouped-quantizer,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
seed,1,U64,NONE,0,0,1.7976931348623157e308,NULL,42
first-frame,1,STRING,MEDIA,0,0,1.7976931348623157e308,NULL,NULL
last-frame,1,STRING,MEDIA,0,0,1.7976931348623157e308,NULL,NULL
ref-image,1,STRING,MEDIA,1,0,1.7976931348623157e308,NULL,NULL
ref-image-size,1,ENUM,NONE,0,0,1.7976931348623157e308,match|high|max,match
ref-video,1,STRING,MEDIA,1,0,1.7976931348623157e308,NULL,NULL
ref-silent-video,1,STRING,MEDIA,1,0,1.7976931348623157e308,NULL,NULL
ref-video-audio,2,STRING,MEDIA,1,0,1.7976931348623157e308,NULL,NULL
ref-audio,1,STRING,MEDIA,1,0,1.7976931348623157e308,NULL,NULL
frames-dir,1,STRING,OUTPUT,0,0,1.7976931348623157e308,NULL,NULL
show,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
zoom,1,INT,NONE,0,1,2147483647,NULL,2
profile,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
info,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
cuda-device,1,INT,NONE,0,0,2147483647,NULL,0
cuda-attention,1,ENUM,NONE,0,0,1.7976931348623157e308,default|sage2++|sage3|sol|subblock,default
cuda-denoise-quant,1,ENUM,NONE,0,0,1.7976931348623157e308,off|fp8|nvfp4,off
cuda-denoise-quant-cache,1,STRING,CACHE,0,0,1.7976931348623157e308,NULL,NULL
cuda-denoise-quant-verify,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
cuda-weight-mode,1,ENUM,NONE,0,0,1.7976931348623157e308,auto|resident|stream,auto
help,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
verbose,0,BOOL,NONE,0,0,1.7976931348623157e308,NULL,false
"""

@dataclass
class Option:
    arity: int
    kind: str
    path: str
    repeat: bool
    minimum: float
    maximum: float
    choices: str
    default: str


@dataclass
class Segment:
    name: str
    line: int
    settings: dict
    references: list = field(default_factory=list)
    continuing: bool = False
    prompt: str = ""
    frames: int = 0
    delivered: int = 0
    width: int = 0
    height: int = 0
    skipped: bool = False

    def value(self, name, options):
        values = self.settings.get(name)
        return values[-1][0] if values else options[name].default


def option_inventory(path=None):
    """Use embedded metadata; an explicit native inventory is for development."""
    options = {}
    if path is None:
        rows = list(csv.reader(OPTION_CSV.splitlines()))
    else:
        rows = []
        for line in Path(path).read_text().splitlines():
            if line.startswith("H3_OPTION("):
                row = next(csv.reader([line[len("H3_OPTION("):-1]], skipinitialspace=True))
                rows.append([row[i] for i in (0, 2, 3, 4, 5, 7, 8, 9, 10)])
    for name, arity, kind, path_kind, repeat, low, high, choices, default in rows:
        options[name] = Option(int(arity), kind, path_kind, bool(int(repeat)),
                               float(low), float(high), choices, default)
    if "frames" not in options or "continue-from" not in options:
        raise ValueError(f"Cannot read the native CLI option inventory: {path}")
    return options


def canonical(name):
    name = name.lstrip("-")
    return ALIASES.get(name, name)


def filename(label):
    label = re.sub(r"\s+###$", "", label).strip()
    if (not label or label in (".", "..") or any(c in label for c in "/\\") or
            any(ord(c) < 32 or ord(c) == 127 for c in label)):
        raise ValueError("Use a filename after ###, without directories or control characters")
    name = label if label.lower().endswith(".mp4") else label + ".mp4"
    if len(name.encode()) > 240:
        raise ValueError("Video filename is too long")
    return name


def path_value(value, base):
    path = Path(value).expanduser()
    return str((base / path).resolve())


def directive_values(name, values, option, base):
    if len(values) != option.arity:
        raise ValueError(f"#{name} expects {option.arity} value(s), got {len(values)}")
    if not option.arity:
        return values
    value = values[0]
    if option.kind in ("INT", "U64", "FLOAT"):
        number = float(value)
        if option.kind in ("INT", "U64"):
            if not re.fullmatch(r"[0-9]+", value):
                raise ValueError(f"#{name} requires a nonnegative integer")
            number = int(value)
        maximum = (1 << 64) - 1 if option.kind == "U64" else option.maximum
        if not math.isfinite(number) or not option.minimum <= number <= maximum:
            raise ValueError(f"Invalid #{name} value: {value}")
    if option.kind == "ENUM" and value not in option.choices.split("|"):
        raise ValueError(f"#{name} expects {option.choices}")
    if option.path != "NONE" or name == "models-path":
        if any(not v for v in values):
            raise ValueError(f"#{name} requires a nonempty path")
        if name == "lora":
            adapter, separator, scale = value.rpartition(":")
            if separator:
                try:
                    scaled = math.isfinite(float(scale))
                except ValueError:
                    scaled = False
                if scaled and adapter:
                    return [path_value(adapter, base) + ":" + scale]
        return [path_value(v, base) for v in values]
    return values


def read_sections(path, options, defaults=None):
    sections, segment, lines, touched = [], None, [], set()
    inherited = copy.deepcopy(defaults or {})

    def finish():
        segment.prompt = "\n".join(lines).strip()
        if not segment.prompt and not segment.skipped:
            raise ValueError(f"{path}:{segment.line}: {segment.name} has an empty prompt")
        sections.append(segment)

    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        stripped = line.strip()
        try:
            if stripped.startswith(("###", "!###")):
                if segment is not None:
                    finish()
                    inherited = copy.deepcopy(segment.settings)
                skipped = stripped.startswith("!###")
                segment = Segment(filename(stripped[4 if skipped else 3:]), number,
                                  copy.deepcopy(inherited), skipped=skipped)
                lines, touched = [], set()
            elif stripped.startswith("##") or not stripped and not lines:
                continue
            elif segment is None:
                raise ValueError("Start the first video with ### NAME or !### NAME")
            elif stripped.startswith("#"):
                if any(text.strip() for text in lines):
                    raise ValueError("Put directives before the prompt; use \\# for a prompt hash")
                tokens = shlex.split(stripped[1:], comments=True)
                if not tokens:
                    continue
                key, equals, value = tokens[0].partition("=")
                name = canonical(key)
                values = ([value] if equals else []) + tokens[1:]
                if name == "continue":
                    if values or segment.continuing:
                        raise ValueError("Use #continue once, with no value; #continue-mode selects the mode")
                    segment.continuing = True
                    continue
                if name == "unset":
                    if len(values) != 1:
                        raise ValueError("Use #unset OPTION")
                    target = canonical(values[0])
                    names = ("width", "height") if target == "size" else (target,)
                    for target in names:
                        if target not in options or target in MANAGED:
                            raise ValueError(f"Cannot unset #{target}")
                        segment.settings.pop(target, None)
                        segment.references = [(k, v) for k, v in segment.references if k != target]
                        touched.discard(target)
                    continue
                if name == "size":
                    if len(values) != 1 or not re.fullmatch(r"[0-9]+[xX][0-9]+", values[0]):
                        raise ValueError("Use #size WIDTHxHEIGHT")
                    width, height = re.split("[xX]", values[0])
                    segment.settings.update(width=[[width]], height=[[height]])
                    continue
                if name in MANAGED:
                    raise ValueError(f"#{name} is managed automatically by render.py")
                if name in NON_VIDEO:
                    raise ValueError(f"#{name} does not render a complete video segment")
                if name not in options:
                    raise ValueError(f"Unknown directive #{name}; use a native h3cli long option")
                option = options[name]
                values = directive_values(name, values, option, path.parent)
                if option.path == "MEDIA":
                    if not option.repeat:
                        segment.references = [(k, v) for k, v in segment.references if k != name]
                    segment.references.append((name, values))
                else:
                    for group in EXCLUSIVE:
                        if name in group:
                            for other in group:
                                if other != name:
                                    segment.settings.pop(other, None)
                    if not option.repeat or name not in touched:
                        segment.settings[name] = []
                    segment.settings[name].append(values)
                    touched.add(name)
            else:
                if line.lstrip().startswith("\\#"):
                    index = line.index("\\#")
                    line = line[:index] + line[index + 1:]
                lines.append(line)
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    if segment is None:
        raise ValueError(f"{path}: no video sections found")
    finish()
    return sections


def chains(sections):
    groups = []
    for index, section in enumerate(sections):
        if not section.continuing:
            groups.append([])
        groups[-1].append(index)
    return groups


def merged_name(section):
    return Path(section.name).stem + "-merged.mp4"


def validate_plan(sections, options):
    if sections and sections[0].continuing:
        section = sections[0]
        raise ValueError(f"Section {section.name} (line {section.line}): #continue needs a preceding segment")
    groups = chains(sections)
    # Check the complete document before validating or executing active renders.
    # Skipping a prefix, suffix, or middle segment would break its saved states.
    for group in groups:
        if len({sections[i].skipped for i in group}) > 1:
            names = ", ".join(f"{sections[i].name} (line {sections[i].line})" for i in group)
            raise ValueError("Continuation chain mixes skipped (!###) and rendered (###) segments: "
                             + names + ". Skip every segment in this chain, or render every segment.")
    seen = set()
    for index, section in enumerate(sections):
        if section.skipped:
            continue
        def fail(message):
            raise ValueError(f"Section {section.name} (line {section.line}): {message}")

        if section.name in seen:
            fail("duplicate output filename")
        seen.add(section.name)
        section.width, section.height = (int(section.value(k, options)) for k in ("width", "height"))
        if (min(section.width, section.height) < 32 or section.width % 32 or
                section.height % 32 or section.width * section.height > 768 * 1344):
            fail("size must use multiples of 32, within the native 768*1344 pixel limit")
        internal = tuple(int(section.value(k, options)) for k in ("render-width", "render-height"))
        if internal != (0, 0) and (min(internal) < 32 or any(n % 32 for n in internal) or
                internal[0] > section.width or internal[1] > section.height or
                internal[0] * section.height != internal[1] * section.width):
            fail("internal render size must match the output aspect ratio and fit within it")
        requested = int(section.value("frames", options))
        if "seconds" in section.settings:
            seconds = float(section.value("seconds", options))
            if not 0 < seconds <= 362 / FPS:
                fail("seconds must be positive and fit within 362 frames")
            requested = math.floor(seconds * FPS + 0.5)
        section.frames = max(5, requested) + (5 - max(5, requested)) % 17
        if not 22 <= section.frames <= 362:
            fail("frame count must align within the native 22..362 generation range")
        section.delivered = section.frames
        for name, values in section.references:
            for value in values:
                if not Path(value).is_file():
                    fail(f"missing #{name} file: {value}")
        if not section.continuing:
            continue
        if not index:
            fail("#continue needs a preceding segment")
        previous = sections[index - 1]
        context = int(section.value("continue-context", options))
        if context < 39 or (context - 39) % 51 or context >= section.frames or context > previous.frames:
            fail("context must be 39 + 51*k, fit the previous state, and leave new frames")
        section.delivered -= context
        for key in ("width", "height", "render-width", "render-height", "model-dir"):
            if section.value(key, options) != previous.value(key, options):
                fail(f"#{key} changes within a continuation chain; start a new chain instead")
        ref_mode = lambda item: any(name.startswith("ref-") for name, _ in item.references)
        if ref_mode(section) != ref_mode(previous):
            fail("continuation cannot switch between Ref2VA and FL2VA; references reset, "
                 "so list #refimage/#ref-video/#ref-audio again for a reference-based continuation")
        if any(name in ("first-frame", "last-frame") for name, _ in section.references):
            fail("native continuation does not support first/last-frame anchors")
        if int(section.value("layers", options)) != 50 or "token-reduction" in section.settings:
            fail("native continuation requires 50 layers and token reduction off")
        core = int(section.value("core-reuse", options))
        mode = section.value("continue-mode", options)
        if core not in ((1, 4, 6) if mode == "bridge" else (1,)):
            fail(f"unsupported core-reuse for {mode} continuation")
    for group in groups:
        if len(group) > 1 and not sections[group[0]].skipped:
            name = merged_name(sections[group[0]])
            if name in seen:
                raise ValueError(f"Merged filename collides with another output: {name}")
            seen.add(name)


def render_command(binary, section, index, work):
    command = [binary]
    for name, occurrences in section.settings.items():
        if name.startswith("continue-") and not section.continuing:
            continue
        for values in occurrences:
            command += ["--" + name, *values]
    for name, values in section.references:
        command += ["--" + name, *values]
    if section.continuing:
        command += ["--continue-from", str(work / f"{index - 1:04d}.h3av")]
    return command + ["--prompt", section.prompt, "--save-av-state", str(work / f"{index:04d}.h3av"),
                      "--output", str(work / f"{index:04d}.mp4")]


def environment(no_env=False, env_file=None):
    env = dict(os.environ)
    if not no_env and env_file:
        result = subprocess.run(["bash", "-c", 'set -e; set -a; source "$1" >&2; env -0',
                                 "render.py", str(env_file)], stdout=subprocess.PIPE, check=True, env=env)
        env = dict(item.split("=", 1) for item in os.fsdecode(result.stdout).split("\0") if "=" in item)
    env.pop("H3CLI_BUNDLE_INFO", None)
    return env


def select_binary(value, env):
    if value or env.get("H3CLI_BIN"):
        return str(Path(value or env["H3CLI_BIN"]).expanduser())
    for name in ("h3cli", "h3cli-linux-x86_64", "h3cli-macos-arm64", "bin/h3cli"):
        path = ROOT / name
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return "h3cli"


def executable(value, env):
    found = shutil.which(str(Path(value).expanduser()), path=env.get("PATH"))
    if not found:
        raise ValueError(f"Executable not found: {value}")
    return str(Path(found).resolve())


def bundled_media(binary, env):
    # --help keeps an ordinary, non-bundled binary from starting a render when
    # it ignores H3CLI_BUNDLE_INFO. Bundle metadata itself does not extract files.
    result = subprocess.run([binary, "--help"], env=env | {"H3CLI_BUNDLE_INFO": "1"},
                            capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise ValueError(f"Cannot run {binary}: {result.stderr.strip() or result.stdout.strip()}")
    try:
        info = json.loads(result.stdout)
    except ValueError:
        return {}
    if not isinstance(info, dict) or not isinstance(info.get("cache"), str):
        return {}
    runtime = Path(info["cache"])
    if info.get("platform") == "macos-arm64":
        runtime /= info["runtime_id"]
    # Let the existing launcher extract and verify its own runtime normally.
    # No downloads, package manager, compiler, or source files are involved.
    result = subprocess.run([binary, "--help"], env=env, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise ValueError(f"Cannot start {binary}: {result.stderr.strip() or result.stdout.strip()}")
    return {name: str(runtime / "tools" / name) for name in ("ffmpeg", "ffprobe")
            if (runtime / "tools" / name).is_file() and os.access(runtime / "tools" / name, os.X_OK)}


def media_tools(binary, env, ffmpeg=None, ffprobe=None):
    chosen = {"ffmpeg": ffmpeg or env.get("H3_FFMPEG"), "ffprobe": ffprobe or env.get("H3_FFPROBE")}
    # Invalid explicit overrides are errors, not requests for a silent fallback.
    result = {name: executable(value, env) for name, value in chosen.items() if value}
    bundled = bundled_media(binary, env) if len(result) < 2 else {}
    missing = []
    for name in chosen:
        if name in result:
            continue
        value = bundled.get(name) or shutil.which(name, path=env.get("PATH"))
        if value:
            result[name] = executable(value, env)
        else:
            missing.append(name)
    if missing:
        raise ValueError("Missing " + ", ".join(missing) + ". Install FFmpeg (includes ffprobe), "
                         "or use --ffmpeg/--ffprobe with existing tools. On Ubuntu/Debian: "
                         "sudo apt install ffmpeg. This script installs nothing.")
    return result["ffmpeg"], result["ffprobe"]


def check_dependencies(binary, ffmpeg, ffprobe, env):
    subprocess.run([binary, "--help"], env=env, stdout=subprocess.DEVNULL, check=True, timeout=180)
    print(f"Python: {sys.version.split()[0]}\nh3cli: {binary}")
    for name, path in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)):
        result = subprocess.run([path, "-version"], env=env, capture_output=True, text=True, check=True, timeout=30)
        lines = (result.stdout or result.stderr).splitlines()
        if not lines or not lines[0].startswith(name + " version "):
            raise ValueError(f"Not a working {name} executable: {path}")
        print(f"{name}: {path}\n  {lines[0]}")
    print("PASS: executables ready; no software installed. Supply existing model weights with -d.")


def stop_child(child):
    # Signal the entire process group, including media workers spawned by h3cli.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def run(command, log, env, live=True):
    with log.open("xb") as stream:
        stream.write((shlex.join(list(map(str, command))) + "\n").encode())
        stream.flush()
        child = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while chunk := os.read(child.stdout.fileno(), 8192):
                stream.write(chunk)
                stream.flush()
                if live:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
            code = child.wait()
        except BaseException:
            stop_child(child)
            raise
        finally:
            child.stdout.close()
    if code:
        raise RuntimeError(f"Command exited {code}; see {log}")


def inspect_video(ffprobe, path, frames, width, height, env):
    result = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)],
                            env=env, capture_output=True, text=True, check=True)
    streams = json.loads(result.stdout)["streams"]
    video = next((s for s in streams if s["codec_type"] == "video"), None)
    audio = next((s for s in streams if s["codec_type"] == "audio"), None)
    if (not video or not audio or (video["width"], video["height"]) != (width, height) or
            int(video.get("nb_read_frames", 0)) != frames or Fraction(video["avg_frame_rate"]) != FPS):
        raise ValueError(f"{path}: expected {width}x{height}, {frames} frames at {FPS} fps and audio")
    if abs(float(audio["duration"]) - frames / FPS) > 0.1:
        raise ValueError(f"{path}: audio duration does not match video")
    return video


def prepare_command(ffmpeg, source, target, frames, reencode=False):
    # Decode/pad/trim audio to video length, then encode AAC only once per chain.
    samples = frames * 32000 // FPS
    video = (["-c:v", "libx264rgb", "-crf", "0", "-preset", "fast", "-pix_fmt", "rgb24"]
             if reencode else ["-c:v", "copy"])
    return [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "warning", "-n", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a:0", "-map_metadata", "-1", *video, "-af",
            f"aresample=32000,apad=whole_len={samples},atrim=end_sample={samples},asetpts=PTS-STARTPTS",
            "-c:a", "pcm_s16le", "-ar", "32000", "-ac", "2", "-video_track_timescale", "24000", str(target)]


def merge_command(ffmpeg, listing, target):
    return [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "warning", "-n", "-f", "concat",
            "-safe", "1", "-i", str(listing), "-map", "0:v:0", "-map", "0:a:0", "-map_metadata", "-1",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(target)]


def merge_chain(group, sections, videos, out, work, ffmpeg, ffprobe, env):
    keys = ("codec_name", "profile", "pix_fmt", "width", "height", "color_range",
            "color_space", "color_transfer", "color_primaries", "sample_aspect_ratio")
    signatures = {tuple(videos[index].get(k) for k in keys) for index in group}
    reencode = len(signatures) > 1
    if reencode:
        print("Encoding settings changed; normalizing this chain to lossless RGB for merging.", flush=True)
    lines = ["ffconcat version 1.0"]
    for index in group:
        section = sections[index]
        prepared = work / f"{index:04d}-prepared.mov"
        run(prepare_command(ffmpeg, out / section.name, prepared, section.delivered, reencode),
            work / f"{index:04d}-prepare.log", env)
        inspect_video(ffprobe, prepared, section.delivered, section.width, section.height, env)
        lines += [f"file '{prepared.name}'", f"duration {section.delivered / FPS:.12f}"]
    first = sections[group[0]]
    listing = work / f"{group[0]:04d}-concat.txt"
    listing.write_text("\n".join(lines) + "\n")
    partial = work / f"{group[0]:04d}-merged.mp4"
    run(merge_command(ffmpeg, listing, partial), work / f"{group[0]:04d}-merge.log", env)
    frames = sum(sections[i].delivered for i in group)
    inspect_video(ffprobe, partial, frames, first.width, first.height, env)
    target = out / merged_name(first)
    partial.rename(target)
    return {"video": str(target), "frames": frames, "seconds": frames / FPS,
            "segments": [sections[i].name for i in group], "lossless_normalization": reencode}


def preflight(ffmpeg, ffprobe, work, env):
    source, prepared = work / "probe.mp4", work / "probe-prepared.mov"
    run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-f", "lavfi", "-i",
         "color=c=gray:s=96x96:r=24:d=0.25", "-f", "lavfi", "-i", "anullsrc=r=32000:cl=stereo",
         "-t", "0.25", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source)],
        work / "probe-source.log", env)
    run(prepare_command(ffmpeg, source, prepared, 6), work / "probe-prepare.log", env)
    listing = work / "probe-concat.txt"
    listing.write_text("ffconcat version 1.0\nfile 'probe-prepared.mov'\nduration 0.25\n")
    run(merge_command(ffmpeg, listing, work / "probe-merged.mp4"), work / "probe-merge.log", env)
    inspect_video(ffprobe, work / "probe-merged.mp4", 6, 96, 96, env)


def write_report(out, report):
    temporary = out / "run.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(out / "run.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("-o", "--output-dir", type=Path, help="Fresh output directory (default: INPUT_STEM-render)")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--check", action="store_true", help="Check existing executables without rendering")
    parser.add_argument("--h3cli", help="Existing executable (default: beside this script, then bin/h3cli or PATH; also H3CLI_BIN)")
    parser.add_argument("--ffmpeg", help="Override H3_FFMPEG, bundled media, or PATH ffmpeg")
    parser.add_argument("--ffprobe", help="Override H3_FFPROBE, bundled media, or PATH ffprobe")
    parser.add_argument("-d", "--model-dir", type=Path, help="Initial model directory; #model-dir can override it")
    parser.add_argument("--no-env", action="store_true", help="Compatibility option; no environment file loads by default")
    parser.add_argument("--env-file", type=Path, help="Explicitly load this shell environment file")
    args = parser.parse_args(argv)
    if args.input is None and not args.check:
        parser.error("INPUT is required unless --check is used")
    started, created, report = time.monotonic(), False, {"status": "running", "segments": [], "merged": []}
    out = None
    try:
        if args.no_env and args.env_file:
            raise ValueError("Choose --no-env or --env-file")
        env = environment(args.no_env, args.env_file.expanduser().resolve() if args.env_file else None)
        binary = select_binary(args.h3cli, env)
        if args.check:
            binary = executable(binary, env)
            ffmpeg, ffprobe = media_tools(binary, env, args.ffmpeg, args.ffprobe)
            check_dependencies(binary, ffmpeg, ffprobe, env)
            return 0
        source = args.input.expanduser().resolve(strict=True)
        out = (args.output_dir or source.with_name(source.stem + "-render")).expanduser().resolve()
        work = out / ".render"
        options = option_inventory()
        model = args.model_dir or env.get("H3_MODEL_DIR")
        if not model:
            model = next((p for p in (ROOT / "models/MiniMaxH3", ROOT / "models/MiniMax-H3") if p.is_dir()), None)
        defaults = {"model-dir": [[str(Path(model).expanduser().resolve())]]} if model else {}
        sections = read_sections(source, options, defaults)
        validate_plan(sections, options)
        groups = [group for group in chains(sections) if not sections[group[0]].skipped]
        if groups and not args.dry_run:
            binary = executable(binary, env)
        commands = [None if s.skipped else render_command(binary, s, i, work) for i, s in enumerate(sections)]
        skipped = [s.name for s in sections if s.skipped]
        print(f"{len(sections)} segments, {len(groups)} active video chains, {len(skipped)} skipped; output: {out}", flush=True)
        for index, section in enumerate(sections):
            if section.skipped:
                print(f"  {section.name}: skipped", flush=True)
                continue
            kind = f"continue {section.value('continue-mode', options)}" if section.continuing else "new video"
            print(f"  {section.name}: {kind}, {section.delivered} delivered frames", flush=True)
            if args.dry_run:
                print(shlex.join(commands[index]))
        for group in groups:
            if len(group) > 1:
                print(f"  merge: {' + '.join(sections[i].name for i in group)} -> {merged_name(sections[group[0]])}")
        if args.dry_run:
            return 0
        if not groups:
            print("All segments skipped; nothing to render.", flush=True)
            return 0
        if out.exists():
            raise ValueError(f"Output directory already exists: {out}; choose a fresh -o directory")
        ffmpeg, ffprobe = media_tools(binary, env, args.ffmpeg, args.ffprobe)
        env.update(H3_FFMPEG=ffmpeg, H3_FFPROBE=ffprobe, H3_SGLANG_INPUT_FFMPEG=ffmpeg)
        out.mkdir(parents=True)
        created = True
        work.mkdir()
        shutil.copyfile(source, work / "source.txt")
        report.update(input=str(source), output=str(out), commands=commands,
                      skipped=skipped,
                      tools={"h3cli": binary, "ffmpeg": ffmpeg, "ffprobe": ffprobe})
        write_report(out, report)
        run([binary, "--help"], work / "h3cli-help.log", env, live=False)
        if any(len(g) > 1 for g in groups):
            preflight(ffmpeg, ffprobe, work, env)
        videos = {}
        for group in groups:
            for index in group:
                section = sections[index]
                print(f"\nRendering {section.name} ({index + 1}/{len(sections)})", flush=True)
                row = {"video": str(out / section.name), "prompt": section.prompt, "command": commands[index],
                       "frames": section.delivered, "status": "running"}
                report["segments"].append(row)
                write_report(out, report)
                began = time.monotonic()
                run(commands[index], work / f"{index:04d}-render.log", env)
                state = work / f"{index:04d}.h3av"
                for required in (state, Path(str(state) + ".presentation")):
                    if not required.is_file() or not required.stat().st_size:
                        raise ValueError(f"Missing continuation state or presentation sidecar: {required}")
                partial = work / f"{index:04d}.mp4"
                videos[index] = inspect_video(ffprobe, partial, section.delivered, section.width, section.height, env)
                partial.rename(out / section.name)
                row.update(status="complete", state=str(state), seconds=time.monotonic() - began)
                write_report(out, report)
            if len(group) > 1:
                merged = merge_chain(group, sections, videos, out, work, ffmpeg, ffprobe, env)
                report["merged"].append(merged)
                write_report(out, report)
                print(f"Merged video: {merged['video']}", flush=True)
        report["status"] = "complete"
        print(f"\nFinished: {out}", flush=True)
        return 0
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        print("\nInterrupted; completed clips and intermediate files were preserved.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        report["status"] = "failed"
        report["error"] = str(error)
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        if created:
            for row in report["segments"]:
                if row["status"] == "running":
                    row["status"] = report["status"]
            report["seconds"] = time.monotonic() - started
            write_report(out, report)


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
