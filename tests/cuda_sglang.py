#!/usr/bin/env python3
"""Pinned CUDA/SGLang qualification runner; never hashes or copies model weights.

Run on the qualification server. Each invocation owns a new output directory;
failed/stale runs cannot silently become successful evidence. The NVML collector
runs in the parent, outside either renderer's library environment.
"""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import ctypes as C
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time

PROMPT = ("Cinematic medium shot of a woman passionately playing a grand piano in a "
          "sunlit concert hall. Her fingers move across the keys as the camera slowly "
          "glides sideways. Warm natural lighting, realistic details, flowing piano music.")
CASES = {"C0": (124, 6), "C1": (362, 6), "C2": (124, 50)}
CONTRACT_PATH=Path(__file__).with_name("cuda_sglang_contract.json")
if CONTRACT_PATH.exists():
    CONTRACT=json.loads(CONTRACT_PATH.read_text())
    HELD_OUT={row["id"]:row for row in CONTRACT["held_out"]}
    CASES.update({key:(row["frames"],row["evaluations"]) for key,row in HELD_OUT.items()})
else:HELD_OUT={}
CONDITION_PATH=Path(__file__).with_name('cuda_sglang_conditioning_cases.json')
CONDITIONED={row['id']:row for row in json.loads(CONDITION_PATH.read_text())['cases']} if CONDITION_PATH.exists() else {}
CASES.update({key:(row['frames'],row['evaluations']) for key,row in CONDITIONED.items()})


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class NVML:
    class Memory(C.Structure):
        _fields_ = [("version",C.c_uint),("total", C.c_ulonglong), ("reserved", C.c_ulonglong), ("free", C.c_ulonglong),
                    ("used", C.c_ulonglong)]

    def __init__(self):
        self.lib = C.CDLL("libnvidia-ml.so.1")
        self.device = C.c_void_p()
        self.check(self.lib.nvmlInit_v2())
        self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(0, C.byref(self.device)))

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"NVML error {code}")

    def sample(self,pid=None,host=True):
        started=time.monotonic()
        mem = self.Memory(version=0x02000028)
        self.check(self.lib.nvmlDeviceGetMemoryInfo_v2(self.device, C.byref(mem)))
        result={"gpu_used_bytes": mem.used+mem.reserved,"gpu_application_bytes":mem.used,
                "gpu_reserved_bytes":mem.reserved,"collector":"nvmlDeviceGetMemoryInfo_v2",
                "nvml_query_seconds":time.monotonic()-started}
        if host:result.update(self.host_sample(pid))
        return result

    @staticmethod
    def host_sample(pid=None):
        host = {k: int(v.split()[0]) * 1024 for k, v in
                (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
        result={"host_available_bytes": host["MemAvailable"],
                "host_swap_used_bytes": host["SwapTotal"] - host["SwapFree"]}
        if pid is not None:
            pending=[pid];seen=set();totals={k:0 for k in ('VmRSS','VmLck','VmPin','VmSwap')}
            while pending:
                current=pending.pop()
                if current in seen:continue
                if len(seen)>=128:raise RuntimeError('unexpectedly large renderer process tree')
                seen.add(current)
                try:
                    status=Path(f'/proc/{current}/status').read_text()
                    pending.extend(map(int,Path(f'/proc/{current}/task/{current}/children').read_text().split()))
                except (FileNotFoundError,ProcessLookupError):continue
                for line in status.splitlines():
                    k,_,v=line.partition(':')
                    if k in totals:totals[k]+=int(v.split()[0])*1024
            result.update(process_tree_count=len(seen),process_rss_bytes=totals['VmRSS'],
                process_locked_bytes=totals['VmLck'],process_kernel_pinned_bytes=totals['VmPin'],process_swap_bytes=totals['VmSwap'])
        return result


def validate(root, engine, frames, evaluations, width=640, height=480):
    log = (root / "render.log").read_text(errors="replace")
    pattern = (rf"minimax_h3 denoise:[^\r\n]*{evaluations}/{evaluations}" if engine == "sglang"
               else rf"denoise\s+{evaluations}/{evaluations}(?:\s|$)")
    if not re.search(pattern, log):
        raise ValueError("missing completed evaluation count")
    perf = None
    if engine == "sglang":
        perf = json.loads((root / "performance.json").read_text())
        steps = [s["step"] for s in perf["denoise_steps_ms"]]
        if steps != list(range(evaluations)):
            raise ValueError(f"wrong denoising evaluation sequence: {steps}")
    media = root / "video.mp4"
    if not media.is_file() or not media.stat().st_size:
        raise ValueError("missing output")
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(media)]))
    write(root / "ffprobe.json", probe)
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    if (video["width"], video["height"], int(video["nb_read_frames"]), video["avg_frame_rate"]) != (width, height, frames, "24/1"):
        raise ValueError(f"incorrect output geometry: {video}")
    if not any(s["codec_type"] == "audio" for s in probe["streams"]):
        raise ValueError("missing audio stream")
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(media), "-f", "null", "-"],
                             capture_output=True, text=True)
    (root / "decode.log").write_text(decoded.stderr)
    if decoded.returncode or decoded.stderr.strip():
        raise ValueError("full media decode failed")
    result = {"media_valid": True, "frames": frames, "evaluations": evaluations,
              "sha256": sha(media), "content_parity": "not_measured"}
    if perf:
        result["stage_seconds"] = {s["name"]: s["duration_ms"] / 1000 for s in perf["steps"]}
        result["generation_seconds"] = perf["total_duration_ms"] / 1000
        result["allocator_memory"] = perf.get("memory_checkpoints", {})
    else:
        result["stage_seconds"]={name:float(seconds) for name,seconds in
            re.findall(r'h3(?:cli)?: phase duration ([^\r\n:]+): ([0-9.]+) s',log)}
        times=re.findall(r'last step: ([0-9.]+) s',log)
        result['denoising_step_seconds']=[float(s) for s in times]
        total=re.findall(r'h3(?:cli)?: total wall time: ([0-9.]+) s',log)
        if total:result['cli_complete_wall_seconds']=float(total[-1])
        generation=re.findall(r'h3(?:cli)?: reference generation including lazy weights: ([0-9.]+) s',log)
        if generation:
            result['generation_seconds']=float(generation[-1])
            result['generation_boundary']='complete h3_generate call, including lazy weights, conditioning and media delivery'
    return result


def run(args):
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    frames, evaluations = CASES[args.case]
    case=CONDITIONED.get(args.case,HELD_OUT.get(args.case,{"prompt":PROMPT,"seed":42}))
    prompt,seed=case["prompt"],case["seed"]
    conditions=[dict(c,uri=str((args.fixtures/c['uri']).resolve(strict=True))) for c in case.get('conditions',[])]
    server = Path(args.sglang).resolve()
    source = Path(args.source).resolve()
    if args.engine == "sglang":
        config = json.loads((server / "sample.json").read_text())
        config.update(num_inference_steps=evaluations + 1, output_path=str(root),
                      output_file_name="video.mp4", prompt=prompt, seed=seed)
        config["target"]["duration_seconds"] = frames / 24 if frames == 124 else 15.0
        if conditions:config.update(task=case['task'],conditions=conditions,
            model_variant='ref2va' if case['task']=='ref2va' else 'fl2va')
        write(root / "config.json", config)
        command = [str(server / ".venv/bin/sglang"), "generate", "--model-path", args.model,
                   "--config", str(root / "config.json"), "--output-file-path", str(root / "video.mp4"),
                   "--perf-dump-path", str(root / "performance.json")]
        setup = server / "env.sh"
        cwd = server
    else:
        command = [str(source / "bin/h3cli"), "-d", args.model, "-p", prompt, "--seed", str(seed),
                   "--width", "640", "--height", "480", "--frames", str(frames), "--steps", str(evaluations),
                   "-o", str(root / "video.mp4")]
        if args.capture:
            command += ["--save-av-state", str(root / "final.h3av")]
        if args.reference:
            command += []
        for c in conditions:
            flag=('--first-frame' if c['frame_index']==0 else '--last-frame') if c['role']=='keyframe' else (
                '--ref-image' if c['type']=='image' else '--ref-video')
            command += [flag,c['uri']]
        if case.get('task')=='ref2va':command += ['--ref-image-size','max']
        setup = Path(args.h3_env).resolve() if args.h3_env else None
        cwd = source
    env = os.environ.copy()
    # Do not inherit diagnostic/reuse overrides into a qualification render.
    inherited = {k: v for k, v in env.items() if k.startswith(("H3_", "SGLANG_"))}
    for k in inherited:
        env.pop(k)
    env["H3_TEST_MAX_EVALUATIONS"] = str(evaluations)
    if args.inputs_only:
        if not args.capture:raise ValueError('input diagnostics require capture and a reference/oracle engine')
        env['H3_TEST_SGLANG_INPUTS_ONLY']='1'
    if args.reference_cublas:
        if not args.reference:raise ValueError("reference cuBLAS path requires the reference policy")
        library=Path(args.reference_cublas).resolve(strict=True)
        env["H3_SGLANG_CUBLAS_LIBRARY"]=str(library)
    if args.reference_ffmpeg:
        if not args.reference:raise ValueError("matched encoder requires the reference policy")
        env["H3_FFMPEG"]=str(Path(args.reference_ffmpeg).resolve(strict=True))
    if args.reference:
        input_ffmpeg=args.reference_input_ffmpeg or shutil.which('ffmpeg')
        if not input_ffmpeg:raise ValueError('reference soundtrack input requires FFmpeg on PATH')
        env['H3_SGLANG_INPUT_FFMPEG']=str(Path(input_ffmpeg).resolve(strict=True))
    elif args.reference_input_ffmpeg:
        raise ValueError('reference input FFmpeg requires the reference policy')
    if args.reference_cudnn:
        if not args.reference:raise ValueError('reference cuDNN path requires the reference policy')
        library=Path(args.reference_cudnn).resolve(strict=True)
        env['H3_SGLANG_CUDNN_LIBRARY']=str(library)
        env['LD_LIBRARY_PATH']=str(library.parent)+':'+env.get('LD_LIBRARY_PATH','')
    if args.reference_jpeg:
        if not args.reference:raise ValueError('reference JPEG library requires the reference policy')
        env['H3_SGLANG_JPEG_LIBRARY']=str(Path(args.reference_jpeg).resolve(strict=True))
    if args.capture and args.engine != "sglang":
        (root / "steps").mkdir()
        env["H3_TEST_NATIVE_STEP_DIR"] = str(root / "steps")
        (root / "capture").mkdir()
        env["H3_TEST_SGLANG_DIR"] = str(root / "capture")
    bootstrap = ""
    if args.capture and args.engine == "sglang":
        env["H3_SGLANG_CAPTURE"] = str(root / "capture")
        bootstrap = "export PYTHONPATH=" + shlex.quote(str(Path(__file__).resolve().parent / "sglang_capture")) + "; "
    for item in args.env:
        key, value = item.split("=", 1)
        env[key] = value
    audio_inputs_only=bool(env.get('H3_TEST_SGLANG_AUDIO_INPUT_ONLY'))
    if audio_inputs_only and not (args.reference and args.capture and env.get('H3_TEST_SGLANG_DIR')):
        raise ValueError('audio input diagnostics require native reference capture')
    shell = ["bash", "-c", "set -e; " + (("source " + shlex.quote(str(setup)) + "; ") if setup else "") + bootstrap + "exec " + shlex.join(command)]
    spec = {"engine": args.engine, "case": args.case, "frames": frames, "evaluations": evaluations,
            "command": command, "cwd": str(cwd), "setup": str(setup) if setup else None, "setup_sha256": sha(setup) if setup else None,
            "inherited_overrides_removed": inherited, "overrides": args.env, "capture": args.capture,
            "production_environment":{k:env[k] for k in ('H3_SGLANG_CUBLAS_LIBRARY','H3_SGLANG_CUDNN_LIBRARY','H3_SGLANG_JPEG_LIBRARY','H3_SGLANG_INPUT_FFMPEG','H3_FFMPEG') if k in env},
            "instrumented": args.capture or bool(args.env),
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "runner_sha256": sha(__file__), "binary_sha256": sha(command[0])}
    spec['inputs_only']=args.inputs_only
    spec['audio_inputs_only']=audio_inputs_only
    if conditions:
        spec['conditioning_manifest_sha256']=sha(CONDITION_PATH)
        spec['condition_inputs']=[dict(c,sha256=sha(c['uri'])) for c in conditions]
    if CONTRACT_PATH.exists():spec["contract_sha256"]=sha(CONTRACT_PATH)
    write(root / "command.json", spec)
    monitor = NVML()
    idle = monitor.sample()
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True).strip()
    # NVML includes a driver reservation on this device that nvidia-smi's
    # headline omits. Check active jobs instead of treating that as a renderer.
    if processes:
        raise RuntimeError(f"GPU has active compute processes: {processes}")
    started = time.monotonic()
    samples = 0
    peak = 0
    largest_gap = 0
    previous = started
    playable = None
    complete_playable = None
    playable_probe_seconds = None
    result = {"status": "running", "spec": spec}
    write(root / "result.json", result)
    with (root / "render.log").open("w") as log, (root / "telemetry.jsonl").open("w") as stream:
        child = subprocess.Popen(shell, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (root / "render.pid").write_text(str(child.pid) + "\n")
        stop=threading.Event();collector_errors=[];host_samples={};host_errors=[]
        def collect_host():
            nonlocal host_samples
            try:
                while not stop.is_set():
                    tick=time.monotonic();sample=monitor.host_sample(child.pid);finished=time.monotonic()
                    sample.update(host_sample_elapsed_s=finished-started,host_query_seconds=finished-tick)
                    host_samples=sample
                    stop.wait(.2)
            except BaseException as exc:host_errors.append(str(exc))
        host_collector=threading.Thread(target=collect_host);host_collector.start()
        def collect():
            nonlocal samples,peak,largest_gap,previous
            try:
                while not stop.is_set():
                    tick=time.monotonic();sample=monitor.sample(host=False);finished=time.monotonic()
                    sample.update(host_samples)
                    if 'host_sample_elapsed_s' in sample:sample['host_sample_age_seconds']=finished-started-sample['host_sample_elapsed_s']
                    sample.update(elapsed_s=finished-started,query_seconds=finished-tick)
                    largest_gap=max(largest_gap,finished-previous);previous=finished
                    peak=max(peak,sample["gpu_used_bytes"]);samples+=1
                    stream.write(json.dumps(sample)+"\n")
                    if samples%10==0:stream.flush()
                    stop.wait(max(0,.05-(finished-tick)))
            except BaseException as exc:collector_errors.append(str(exc))
        collector=threading.Thread(target=collect);collector.start()
        # Probe outside both the wall-clock loop and the NVML collector. A
        # finalized container with the complete frame count is stronger proof
        # than a file's first bytes; full decoding is still required below.
        def check_playable():
            nonlocal complete_playable,playable_probe_seconds
            media=root/'video.mp4'
            while not stop.is_set():
                if media.is_file() and media.stat().st_size>1024:
                    before=time.monotonic()
                    try:
                        probe=subprocess.run(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(media)],
                            capture_output=True,text=True,timeout=10,stdin=subprocess.DEVNULL)
                        if probe.returncode==0:
                            streams=json.loads(probe.stdout)['streams']
                            video=next(s for s in streams if s['codec_type']=='video')
                            if (video['width'],video['height'],int(video.get('nb_read_frames',0)),video['avg_frame_rate'])==(640,480,frames,'24/1') and any(s['codec_type']=='audio' for s in streams):
                                complete_playable=time.monotonic()-started
                                playable_probe_seconds=time.monotonic()-before
                                return
                    except (subprocess.TimeoutExpired,ValueError,KeyError,StopIteration):pass
                stop.wait(.1)
        playable_checker=threading.Thread(target=check_playable);playable_checker.start()
        exited=None
        try:
            while True:
                tick = time.monotonic()
                if child.poll() is not None:
                    exited=time.monotonic()
                    break
                media = root / "video.mp4"
                if playable is None and media.exists() and media.stat().st_size > 1024:
                    # File presence is not yet playable-output proof; retain only as a lower bound.
                    playable = tick - started
                if tick - started > args.timeout:
                    raise TimeoutError("render timeout")
                time.sleep(max(0, 0.05 - (time.monotonic() - tick)))
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
        finally:
            if exited is None:exited=time.monotonic()
            # NVML can block inside the driver during allocations and teardown.
            # Never include that delay in the renderer's process wall time.
            stop.set();collector.join();host_collector.join();playable_checker.join()
            result.update(exit_code=child.poll(), wall_seconds=exited - started,
                          sampled_peak_gpu_bytes=peak, sample_count=samples, largest_sample_gap_seconds=largest_gap,
                          first_output_bytes_seconds=playable, idle=idle, status="failed_or_unvalidated",
                          complete_playable_seconds=complete_playable,playable_probe_seconds=playable_probe_seconds,
                          wall_clock_collector_independent=True,collector_errors=collector_errors,
                          host_collector_errors=host_errors,
                          memory_sampling_gate=largest_gap<=.1 and not collector_errors)
            write(root / "result.json", result)
    try:
        if audio_inputs_only:
            marker='test-only reference audio capture complete; no denoising executed'
            if marker not in (root/'render.log').read_text(errors='replace'):
                raise ValueError('audio input diagnostics did not reach the stop boundary')
            records={}
            for name,limit in (('audio-input.f32',15*32000*2*4),('condition-audio.f32',15*40*2*32*4)):
                path=root/'capture'/name
                if not path.is_file() or not 0<path.stat().st_size<=limit or path.stat().st_size%8:
                    raise ValueError('invalid reference audio capture: '+name)
                records[name]=dict(bytes=path.stat().st_size,sha256=sha(path))
            result.update(status='audio_inputs_captured',qualification=False,expected_diagnostic_exit=child.returncode,
                          audio_capture=records)
            write(root/'result.json',result);print(json.dumps(result,indent=2));return 0
        if args.inputs_only:
            marker='test-only reference input capture complete; no denoising executed'
            if marker not in (root/'render.log').read_text(errors='replace'):raise ValueError('input diagnostics did not reach the stop boundary')
            # The positive static dictionary varies by task; the stop marker
            # and saved position/schedule arrays establish the oracle boundary.
            required=['input_capture_complete.json','sigmas_video.json','sigmas_audio.json'] if args.engine=='sglang' else ['tokens.u32','text.bf16','positions.f64','condition-video.f32','condition-audio.f32']
            if not all((root/'capture'/f).is_file() for f in required):raise ValueError('incomplete input capture')
            result.update(status='inputs_captured',qualification=False,expected_diagnostic_exit=child.returncode)
            write(root/'result.json',result);print(json.dumps(result,indent=2));return 0
        if child.returncode:
            raise ValueError(f"renderer exit {child.returncode}")
        result["validation"] = validate(root, args.engine, frames, evaluations)
        if complete_playable is None:
            # A fast native teardown can finish before the first asynchronous
            # probe. Keep this conservative verified upper bound explicit.
            result['complete_playable_seconds']=time.monotonic()-started
            result['playable_verified_after_process_exit']=True
        result["status"] = "media_valid"
    except Exception as exc:
        result["error"] = str(exc)
    write(root / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "media_valid" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("sglang", "native"), required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--h3-env", help="Optional shell environment file for native execution")
    parser.add_argument("--sglang", default="sglang")
    parser.add_argument("--model", default=os.environ.get("H3_MODEL_DIR", "models/MiniMax-H3"))
    parser.add_argument("--capture", action="store_true")
    parser.add_argument('--inputs-only',action='store_true',help='Capture conditioning/layout then deliberately abort before denoising; not a render qualification')
    parser.add_argument('--fixtures',type=Path,default=Path('tests/fixtures/cuda-reference'))
    parser.add_argument("--reference-cublas", help="Explicit production cuBLAS 13 library for the reference recipe")
    parser.add_argument("--reference-ffmpeg", help="Pinned oracle FFmpeg executable for matching delivery")
    parser.add_argument("--reference-input-ffmpeg", help="Pinned soundtrack input decoder; defaults to PATH FFmpeg independently of delivery")
    parser.add_argument('--reference-cudnn',help='Pinned shared cuDNN 9.20 library; adds its directory to LD_LIBRARY_PATH')
    parser.add_argument('--reference-jpeg',help='TurboJPEG runtime used only for reference JPEG preprocessing')
    parser.add_argument("--reference", action="store_true", help="Select the native SGLang policy")
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=14400)
    args = parser.parse_args()
    if args.reference and args.engine != "native":
        parser.error("--reference requires --engine native")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
