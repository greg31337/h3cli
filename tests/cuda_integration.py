#!/usr/bin/env python3
"""Provider-neutral released-model qualification. No Python inference packages.

Every run has an explicit workload, checked exit code, FFprobe assertions and a
JSON record containing device/toolchain information. Every tier stays within
the six-evaluation test ceiling.
"""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import json
import hashlib
from pathlib import Path
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUT = Path(os.environ.get("H3_CUDA_OUTPUT", "outputs/cuda-validation"))
OUT.mkdir(parents=True, exist_ok=True)
MODEL = os.environ.get("H3_MODEL_DIR", "models/MiniMax-H3")
PROMPT = "The woman smiles and turns toward the camera. Steady camera and quiet outdoor ambience."

def capture(args):
    try:
        p = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return {"returncode": p.returncode, "output": p.stdout}
    except OSError as e:
        return {"error": str(e)}

def metadata():
    source=hashlib.sha256()
    for file in sorted(p for p in ROOT.glob("src/**/*.*") if p.suffix in (".c",".m",".h",".cu",".cuh")):
        source.update(file.read_bytes())
    source.update((ROOT/"src/metal/shaders.metal").read_bytes())
    return {"platform": platform.platform(), "machine": platform.machine(), "portable_source_sha256": source.hexdigest(),
            "container_memory_limit": next((p.read_text().strip() for p in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")) if p.exists()), None),
            "device": capture(["./bin/h3cli", "--info", "-d", MODEL]),
            "nvidia": capture(["nvidia-smi", "--query-gpu=name,uuid,compute_cap,memory.total,driver_version", "--format=csv"]),
            "nvcc": capture([os.environ.get("CUDA_PATH", "/usr/local/cuda") + "/bin/nvcc", "--version"]),
            "compiler": capture(["cc", "--version"]), "ffmpeg": capture(["ffmpeg", "-version"]),
            "commit": capture(["git", "rev-parse", "HEAD"]),
            "environment": {k: v for k, v in os.environ.items() if k.startswith("H3_")},
            "model": MODEL}

META = metadata()

def run(name, args, extra_env=None):
    env = os.environ.copy()
    if extra_env: env.update(extra_env)
    print(f"running {name}", flush=True)
    started = time.monotonic()
    with (OUT / f"{name}.log").open("w") as log:
        p = subprocess.run([str(a) for a in args], stdout=log, stderr=subprocess.STDOUT, env=env)
    record = dict(META, name=name, command=[str(a) for a in args], seconds=time.monotonic()-started,
                  returncode=p.returncode, overrides=extra_env or {},
                  reference_workers=int(os.environ.get("H3_TEST_REFERENCE_WORKERS", "1")))
    if not p.returncode and Path(str(args[0])).name == "continuation_generate":
        # The shared driver writes its own geometry/callback audit at this
        # basename. Retain it before replacing the file with run metadata.
        record["generation"] = json.loads((OUT / f"{name}.json").read_text())
    (OUT / f"{name}.json").write_text(json.dumps(record, indent=2)+"\n")
    if p.returncode:
        print((OUT / f"{name}.log").read_text()[-5000:], file=sys.stderr)
        raise RuntimeError(f"{name} failed: {p.returncode}")
    print(f"passed {name}: {record['seconds']:.2f}s", flush=True)
    return record

def check_mp4(name, frames, width, height):
    video = OUT / f"{name}.mp4"
    data = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video)]))
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    a = next(s for s in data["streams"] if s["codec_type"] == "audio")
    assert (int(v["nb_frames"]), v["width"], v["height"]) == (frames, width, height), v
    assert (int(a["sample_rate"]), a["channels"]) == (32000, 2), a
    (OUT / f"{name}.ffprobe.json").write_text(json.dumps(data, indent=2)+"\n")

def generate(name, refs, width=128, height=128, frames=22, steps=2, extra=()):
    run(name, ["./bin/h3cli", "-d", MODEL, "-p", PROMPT, "--width", width, "--height", height,
               "--frames", frames, "--steps", steps, "--seed", 72, "--layers", 50,
               "--reuse", 1, "--core-reuse", 1, "--profile", "--save-av-state", OUT/f"{name}.h3av",
               "-o", OUT/f"{name}.mp4", *refs, *extra])
    check_mp4(name, frames, width, height)

def resume(name="resume", steps=2, overrides=None, continuation=None):
    base=OUT/name
    frames,width,height=22,128,128
    if continuation is not None:
        from continuation_metrics import state
        geometry,_,_=state(Path(continuation))
        width,height=geometry[:2];frames=102
        overrides=dict(overrides or {},H3_TEST_CONTINUATION=str(continuation))
    run(name+"-record", ["./bin/cuda_resume_test", "record", MODEL, base, steps], overrides)
    run(name+"-restart", ["./bin/cuda_resume_test", "resume", MODEL, str(base)+"-restart", steps,
                          str(base)+".h3sample", str(base)+".trace"], overrides)
    check_mp4(name,frames,width,height)
    check_mp4(name+"-restart",frames,width,height)

def chain(steps=2, size=128):
    previous="-"
    for segment,seed,mode in [(1,72,"image1"),(2,73,"image1"),(3,74,"image1"),(4,75,"image2")]:
        name=f"chain-{segment}"
        record=run(name,["./bin/continuation_generate",MODEL,OUT/name,mode,previous,seed,141,steps,1,0,
                         PROMPT,"-",39,size], {"H3_GPU_SAMPLER":"1"})
        assert record["generation"]["latent_callbacks"] == steps+1
        check_mp4(name,141 if segment==1 else 102,size,size)
        previous=str(OUT/name)+".h3av"

def references():
    # Real images supply visual content; generated reference clips make audio
    # inclusion/replacement repeatable without requiring an external dataset.
    run("reference-clip",["ffmpeg","-hide_banner","-loglevel","error","-y","-loop","1","-i","inputs/2.jpg",
        "-f","lavfi","-i","sine=frequency=330:sample_rate=32000","-t","2.5","-vf","scale=128:128,fps=24",
        "-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac","-ac","2",OUT/"reference.mp4"])
    run("replacement-audio",["ffmpeg","-hide_banner","-loglevel","error","-y","-f","lavfi","-i",
        "sine=frequency=660:sample_rate=32000","-t","2.5","-ac","2",OUT/"replacement.wav"])
    cases={"t2va":[],"first":["--first-frame","inputs/face1.jpg"],"last":["--last-frame","inputs/face2.jpg"],
        "first-last":["--first-frame","inputs/face1.jpg","--last-frame","inputs/face2.jpg"],
        "images":["--ref-image","inputs/2.jpg","--ref-image","inputs/body1.jpg"],
        "video":["--ref-video",OUT/"reference.mp4"],"silent":["--ref-silent-video",OUT/"reference.mp4"],
        "video-audio":["--ref-video-audio",OUT/"reference.mp4",OUT/"replacement.wav"],
        # Ref2VA requires visual conditioning for a separate audio reference,
        # just as on Metal; "standalone" means a separate file, not audio-only.
        "audio":["--ref-image","inputs/face1.jpg","--ref-audio",OUT/"replacement.wav"]}
    selected=os.environ.get("H3_CUDA_CASES")
    workers=int(os.environ.get("H3_TEST_REFERENCE_WORKERS", "2" if platform.system()=="Linux" else "1"))
    assert 1<=workers<=3, "reference workers must be between one and three"
    os.environ["H3_TEST_REFERENCE_WORKERS"]=str(workers)
    # Only these small, independent 128x128 cases overlap. Cache-pressure,
    # continuation, full-quality and benchmark tiers remain serial.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Video decoding is capped by the target frame count. Use 56 target
        # frames so the normalized reference still meets the two-second minimum.
        pending=[pool.submit(generate,"feature-"+name,refs,
                             frames=56 if name in ("video","silent","video-audio") else 22,
                             steps=2)
                 for name,refs in cases.items() if not selected or name in selected.split(",")]
        for future in pending: future.result()

def optimizations():
    resume("reuse",6,{"H3_TEST_REUSE":"2"})
    resume("core",4,{"H3_TEST_CORE":"2"})
    resume("reduction",4,{"H3_TEST_REDUCTION":"1"})

if __name__ == "__main__":
    tier=sys.argv[1]
    if tier=="smoke":generate("smoke",["--ref-image","inputs/face1.jpg"])
    elif tier=="features":
        references();chain();resume()
        resume("resume-continuation",continuation=OUT/"chain-1.h3av")
        optimizations()
    elif tier=="references":references()
    elif tier=="optimizations":optimizations()
    elif tier=="cache":
        resume("cache",2,{"H3_TEST_CACHE_PRESSURE":"1","H3_CUDA_WEIGHT_MODE":"stream"});check_mp4("cache.pressure",22,128,128)
    elif tier=="chain":chain()
    elif tier=="resume":resume()
    elif tier=="continuation-resume":
        resume("resume-continuation",continuation=os.environ.get("H3_TEST_CONTINUATION",OUT/"chain-1.h3av"))
    else:raise SystemExit("unknown test tier: "+tier)
