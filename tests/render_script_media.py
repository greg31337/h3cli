#!/usr/bin/env python3
"""Exercise render.py merging with real FFmpeg, including changed encodings.

Run from a configured checkout: python3 tests/render_script_media.py --out DIR
DIR must be fresh. No models or GPU are used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import render


def decoded_rgb(ffmpeg, path, env):
    result = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0",
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True)
    env = render.environment()
    ffmpeg = render.executable(env.get("H3_FFMPEG", "ffmpeg"), env)
    ffprobe = render.executable(env.get("H3_FFPROBE", "ffprobe"), env)
    records = []
    for mixed in (False, True):
        folder = out / ("different-encodings" if mixed else "stream-copy")
        folder.mkdir()
        work = folder / ".render"
        work.mkdir()
        sections, videos, original = [], {}, b""
        # Unequal lengths, fractional audio sample counts and quoted filenames.
        for index, (name, color, frames) in enumerate((("first 'clip.mp4", "red", 17),
                                                     ("second clip.mp4", "blue", 22))):
            section = render.Segment(name, 1, {}, continuing=index > 0,
                                     frames=frames, delivered=frames, width=96, height=96)
            sections.append(section)
            encoding = (["-c:v", "libx264rgb", "-crf", "0", "-pix_fmt", "rgb24"] if mixed and index else
                        ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"])
            command = [ffmpeg, "-nostdin", "-v", "error", "-n", "-f", "lavfi", "-i",
                       f"color=c={color}:s=96x96:r=24:d={frames / 24:.12f}",
                       "-f", "lavfi", "-i", f"sine=frequency={440 + index * 220}:sample_rate=32000",
                       "-t", f"{frames / 24:.12f}", "-frames:v", str(frames), *encoding,
                       "-c:a", "aac", "-ar", "32000", "-ac", "2", str(folder / name)]
            render.run(command, work / f"fixture-{index}.log", env)
            videos[index] = render.inspect_video(ffprobe, folder / name, frames, 96, 96, env)
            original += decoded_rgb(ffmpeg, folder / name, env)
        result = render.merge_chain([0, 1], sections, videos, folder, work, ffmpeg, ffprobe, env)
        actual = decoded_rgb(ffmpeg, Path(result["video"]), env)
        if result["lossless_normalization"] != mixed or original != actual:
            raise AssertionError("Merged frames differ from concatenated original RGB frames")
        result.update(decoded_rgb_sha256=hashlib.sha256(actual).hexdigest(), exact_decoded_rgb=True)
        records.append(result)
    (out / "result.json").write_text(json.dumps({"passed": True, "cases": records}, indent=2) + "\n")
    print("PASS: stream-copy and changed-encoding merges preserve every decoded RGB frame and audio timing")


if __name__ == "__main__":
    main()
