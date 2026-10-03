#!/usr/bin/env python3
"""Compare the frozen pre-port Metal executable with the current one."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("outputs/cuda-validation/baseline/bin/h3cli"))
    parser.add_argument("--model", default="models/MiniMax-H3")
    parser.add_argument("--output", type=Path, default=Path("outputs/cuda-validation/metal-baseline-regression"))
    args = parser.parse_args()
    assert platform.system() == "Darwin"
    assert args.baseline.is_file(), "record the unmodified Metal executable first"
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for label, binary in (("baseline", str(args.baseline.resolve())), ("current", str(Path("bin/h3cli").resolve()))):
        out = args.output / label
        out.mkdir(exist_ok=True)
        common = [binary, "-d", args.model, "--width", "128", "--height", "128", "--layers", "50",
                  "--reuse", "1", "--core-reuse", "1", "--seed", "72"]
        conditioning = ["-p", "The woman smiles and turns toward the camera. Steady camera.",
                        "--ref-image", "inputs/face1.jpg", "--ref-image", "inputs/body1.jpg",
                        "--ref-image", "inputs/2.jpg"]
        cases = [
            ("first", ["--frames", "141", "--steps", "2", *conditioning]),
            ("continued", ["--frames", "141", "--steps", "2", "--seed", "73", *conditioning,
                           "--continue-from", str(out / "first.h3av"), "--continue-context", "39"]),
            ("sampler", ["--frames", "22", "--steps", "4", *conditioning]),
            ("paused", ["--frames", "22", "--steps", "4", *conditioning, "--stop-after-step", "2",
                        "--save-sampler-state", str(out / "paused.h3sample")]),
            ("resumed", ["--resume-sampler-state", str(out / "paused.h3sample")]),
        ]
        for name, options in cases:
            command = ([binary, "-d", args.model] if name == "resumed" else common) + options + ["-o", str(out / (name + ".mp4"))]
            if name != "paused":
                command += ["--save-av-state", str(out / (name + ".h3av"))]
            started = time.monotonic()
            with (out / (name + ".log")).open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            records.append({"backend": label, "case": name, "command": command,
                            "returncode": result.returncode, "seconds": time.monotonic() - started})
            (args.output / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
            assert result.returncode == 0, (label, name)
            print("passed", label, name, flush=True)
    hashes = {}
    for name in ("first", "continued", "sampler", "resumed"):
        for extension in ("h3av", "mp4"):
            filename = name + "." + extension
            before = (args.output / "baseline" / filename).read_bytes()
            after = (args.output / "current" / filename).read_bytes()
            assert before == after, "Metal behavior changed: " + filename
            hashes[filename] = hashlib.sha256(after).hexdigest()
    for label in ("baseline", "current"):
        for extension in ("h3av", "mp4"):
            assert (args.output / label / ("sampler." + extension)).read_bytes() == (args.output / label / ("resumed." + extension)).read_bytes()
    (args.output / "exact-matches.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print("ok: frozen/current Metal Ref2VA, continuation, AV containers and fresh-process sampler restart are byte-exact")


if __name__ == "__main__":
    main()
