#!/usr/bin/env python3
"""Compare completed Metal/CUDA artifacts without an inference dependency.

These measurements expose drift and degenerate outputs; they do not replace
component tolerances, continuation invariants, or visual review.
"""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from continuation_metrics import state


def floats(raw):
    values = array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def statistics(values):
    assert values and all(math.isfinite(x) for x in values)
    mean = math.fsum(values) / len(values)
    square = math.fsum(x * x for x in values) / len(values)
    return {"elements": len(values), "minimum": min(values), "maximum": max(values),
            "mean": mean, "rms": math.sqrt(square),
            "stddev": math.sqrt(max(0, square - mean * mean))}


def compare(reference, actual):
    assert len(reference) == len(actual) and reference
    maximum = absolute = difference = norm = actual_norm = dot = 0.0
    for a, b in zip(reference, actual):
        assert math.isfinite(a) and math.isfinite(b)
        d = b - a
        maximum = max(maximum, abs(d))
        absolute += abs(d)
        difference += d * d
        norm += a * a
        actual_norm += b * b
        dot += a * b
    return {"max_abs": maximum, "mean_abs": absolute / len(reference),
            "rmse": math.sqrt(difference / len(reference)),
            "relative_l2": math.sqrt(difference / max(norm, 1e-30)),
            "cosine": dot / math.sqrt(max(norm * actual_norm, 1e-30))}


def decoded(base):
    media = str(base) + ".mp4"
    common = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", media]
    rgb = subprocess.check_output(common + ["-map", "0:v:0", "-vf", "scale=160:90",
                                            "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])
    pcm = floats(subprocess.check_output(common + ["-map", "0:a:0", "-ac", "2", "-ar", "32000",
                                                   "-f", "f32le", "pipe:1"]))
    frame_size = 160 * 90 * 3
    assert rgb and len(rgb) % frame_size == 0
    changes = [math.fsum(abs(a - b) for a, b in zip(rgb[start-frame_size:start], rgb[start:start+frame_size])) / frame_size
               for start in range(frame_size, len(rgb), frame_size)]
    return rgb, pcm, {"frames": len(rgb) // frame_size, "rgb": statistics(rgb),
                      "audio": statistics(pcm),
                      "mean_adjacent_frame_difference": math.fsum(changes) / max(1, len(changes))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Metal artifact basename, without extension")
    parser.add_argument("actual", type=Path, help="CUDA artifact basename, without extension")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifacts = []
    for base in (args.reference, args.actual):
        geometry, video, audio = state(Path(str(base) + ".h3av"))
        rgb, pcm, media = decoded(base)
        artifacts.append((geometry, floats(video), floats(audio), rgb, pcm, media))
    expected, actual = artifacts
    assert expected[0] == actual[0], "AV geometry mismatch"
    assert expected[5]["frames"] == actual[5]["frames"], "delivered frame count mismatch"
    report = {"reference": str(args.reference), "actual": str(args.actual),
              "geometry": expected[0], "comparisons": {}, "artifacts": []}
    for name, index in (("video_latent", 1), ("audio_latent", 2), ("decoded_rgb_160x90", 3), ("decoded_pcm", 4)):
        report["comparisons"][name] = compare(expected[index], actual[index])
    for base, item in zip((args.reference, args.actual), artifacts):
        report["artifacts"].append({"base": str(base), "video_latent": statistics(item[1]),
                                    "audio_latent": statistics(item[2]), "media": item[5],
                                    "sha256": {ext: hashlib.sha256(Path(str(base) + "." + ext).read_bytes()).hexdigest()
                                               for ext in ("h3av", "mp4")}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["comparisons"], indent=2))


if __name__ == "__main__":
    main()
