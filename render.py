#!/usr/bin/env python3
"""Compatibility entry point for h3cli --render (Python 3.10+).

Use h3cli --render FILE directly for native, Python-free rendering. This wrapper
retains binary discovery, --check, -o, and explicit environment-file loading.
Script parsing, prompt inheritance and merging are implemented by h3cli.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("-o", "--output-dir", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--check", action="store_true")
    parser.add_argument("--h3cli", help="Executable supporting --render (also H3CLI_BIN)")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    parser.add_argument("-d", "--model-dir", type=Path)
    parser.add_argument("--no-env", action="store_true")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args(argv)
    if args.input is None and not args.check:
        parser.error("INPUT is required unless --check is used")
    if args.no_env and args.env_file:
        parser.error("Choose --no-env or --env-file")
    try:
        env = environment(env_file=args.env_file.expanduser().resolve() if args.env_file else None)
        binary = executable(select_binary(args.h3cli, env), env)
        if args.check:
            ffmpeg, ffprobe = media_tools(binary, env, args.ffmpeg, args.ffprobe)
            check_dependencies(binary, ffmpeg, ffprobe, env)
            return 0
        for key, val in (("H3_FFMPEG", args.ffmpeg), ("H3_FFPROBE", args.ffprobe)):
            if val:
                env[key] = executable(val, env)
        command = [binary, "--render", str(args.input)]
        if args.output_dir:
            command += ["--output-dir", str(args.output_dir)]
        model = args.model_dir or env.get("H3_MODEL_DIR")
        if model:
            command += ["--model-dir", str(model)]
        if args.dry_run:
            command += ["--dry-run"]
        os.execve(binary, command, env)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"render.py: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
