#!/usr/bin/env python3
"""Every-frame media gates, audio metrics and worst-region inspection assets.

Uses CPU metrics so it never competes for the qualification GPU. Run outside
timing campaigns. Raw pre-codec comparisons remain a separate decoder gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from cuda_sglang_compare import metric


def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def psnr(mse):return float(-10*np.log10(mse)) if mse else 999.


def probe(path):
    return json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-show_streams","-of","json",str(path)]))


def pcm(path):
    raw=subprocess.check_output(["ffmpeg","-v","error","-i",str(path),"-vn","-acodec","pcm_f32le","-f","f32le","-"])
    return np.frombuffer(raw,dtype="<f4")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference",type=Path);p.add_argument("candidate",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--raw-rgb",type=int,metavar="FRAMES",help="Compare THWC float32 640x480 decoder output before encoding")
    p.add_argument("--raw-pcm",nargs=2,type=Path,metavar=("REFERENCE","CANDIDATE"))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if bool(a.raw_rgb)!=bool(a.raw_pcm):p.error("raw comparisons require RGB frame count and both PCM files")
    contract_path=Path(__file__).with_name("cuda_sglang_contract.json");contract=json.loads(contract_path.read_text())
    if a.raw_rgb:
        if a.raw_rgb not in (124,362):p.error("raw frame count must match C0-C2")
        w,h,count=640,480,a.raw_rgb
        if any(path.stat().st_size!=count*w*h*3*4 for path in (a.reference,a.candidate)):raise ValueError("invalid raw RGB byte count")
        expected=(207 if count==124 else 603)*800*2*4
        if any(path.stat().st_size!=expected for path in a.raw_pcm):raise ValueError("invalid raw PCM sample count")
        decoders=[];streams=[path.open('rb') for path in (a.reference,a.candidate)]
    else:
        metadata=[probe(path) for path in (a.reference,a.candidate)]
        videos=[next(s for s in x["streams"] if s["codec_type"]=="video") for x in metadata]
        for key in ("width","height","nb_read_frames","avg_frame_rate"):
            if videos[0][key]!=videos[1][key]:raise ValueError(f"unaligned video {key}")
        audios=[next(s for s in x["streams"] if s["codec_type"]=="audio") for x in metadata]
        for key in ("channels","sample_rate"):
            if audios[0][key]!=audios[1][key]:raise ValueError(f"unaligned audio {key}")
        w,h=videos[0]["width"],videos[0]["height"];count=int(videos[0]["nb_read_frames"])
        decoders=[subprocess.Popen(["ffmpeg","-v","error","-i",str(path),"-an","-pix_fmt","rgb24","-f","rawvideo","-"],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE) for path in (a.reference,a.candidate)]
        streams=[d.stdout for d in decoders]
    net=None;frames=[];previous=None;worst=None;started=time.monotonic()
    try:
        for i in range(count):
            size=w*h*3*(4 if a.raw_rgb else 1)
            data=[s.read(size) for s in streams]
            if any(len(x)!=size for x in data):raise ValueError(f"incomplete decoded frame {i}")
            x,y=[np.frombuffer(raw,'<f4' if a.raw_rgb else np.uint8).reshape(h,w,3).astype(np.float32)/(1 if a.raw_rgb else 255) for raw in data]
            if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError("nonfinite RGB")
            if a.raw_rgb and (min(x.min(),y.min())<0 or max(x.max(),y.max())>1):raise ValueError("raw RGB outside normalized range")
            delta=y-x;mse=float(np.square(delta,dtype=np.float64).mean());perceptual=0.;ssim=1.
            if mse:
                if net is None:
                    import torch,lpips
                    torch.set_num_threads(4);net=lpips.LPIPS(net="alex",version="0.1").cpu().eval()
                with torch.inference_mode():
                    tx,ty=[torch.from_numpy(v).permute(2,0,1).unsqueeze(0)*2-1 for v in (x,y)]
                    perceptual=float(net(tx,ty).item())
                ssim=float(structural_similarity(x,y,data_range=1,channel_axis=-1,gaussian_weights=True,sigma=1.5,use_sample_covariance=False))
            temporal=float(np.sqrt(np.square(delta-previous,dtype=np.float64).mean())) if previous is not None else 0.
            previous=delta
            tiles=[]
            for top in range(0,h,64):
                for left in range(0,w,64):tiles.append((float(np.square(delta[top:top+64,left:left+64],dtype=np.float64).mean()),top,left))
            tile,top,left=max(tiles)
            row=dict(frame=i,psnr_db=psnr(mse),ssim=ssim,lpips=perceptual,temporal_error_rms=temporal,worst_region_psnr_db=psnr(tile),worst_region_yx=[top,left],mse=mse)
            frames.append(row)
            if worst is None or row["psnr_db"]<worst["psnr_db"]:
                worst=row
                Image.fromarray(np.rint(x*255).astype(np.uint8)).save(a.output/"reference-worst.png")
                Image.fromarray(np.rint(y*255).astype(np.uint8)).save(a.output/"candidate-worst.png")
                Image.fromarray(np.rint(np.clip(abs(delta)*20,0,1)*255).astype(np.uint8)).save(a.output/"difference-x20.png")
            if i%24==0:print(f"metrics {i+1}/{count}",flush=True)
        for d in decoders:
            if d.stdout.read(1):raise ValueError("unexpected extra decoded frames")
            err=d.stderr.read();code=d.wait()
            if code or err:raise ValueError(f"video decode error: {err.decode(errors='replace')}")
    finally:
        for d in decoders:
            if d.poll() is None:d.terminate();d.wait()
        if a.raw_rgb:
            for s in streams:s.close()
    audio=metric(np.fromfile(a.raw_pcm[1],dtype='<f4'),np.fromfile(a.raw_pcm[0],dtype='<f4')) if a.raw_rgb else metric(pcm(a.candidate),pcm(a.reference))
    g=contract["gates"]["decoded_video"];ag=contract["gates"]["decoded_audio"]
    gates=dict(every_frame_psnr=all(r["psnr_db"]>=g["every_frame_psnr_db_min"] for r in frames),
        every_frame_ssim=all(r["ssim"]>=g["every_frame_gaussian_ssim_min"] for r in frames),
        every_frame_lpips=all(r["lpips"]<=g["every_frame_lpips_alex_v01_max"] for r in frames),
        temporal=all(r["temporal_error_rms"]<=g["temporal_error_rms_max"] for r in frames),
        worst_region=all(r["worst_region_psnr_db"]>=g["worst_64x64_region_psnr_db_min"] for r in frames),
        audio=bool(audio.get("finite") and audio["relative_l2"]<=ag["relative_l2_max"] and audio["cosine"]>=ag["cosine_min"]))
    report=dict(domain="raw decoder RGB/PCM before encoding" if a.raw_rgb else "decoded MP4 RGB24/PCM; pre-codec parity is separate",reference=str(a.reference),candidate=str(a.candidate),
        reference_sha256=sha(a.reference),candidate_sha256=sha(a.candidate),contract_sha256=sha(contract_path),
        width=w,height=h,frame_count=count,frames=frames,audio=audio,gates=gates,passed=all(gates.values()),
        metric_seconds=time.monotonic()-started,worst_frame=worst["frame"],mean_mse=float(np.mean([r["mse"] for r in frames])))
    if a.raw_rgb:report['pcm_sources']=[dict(path=str(path),sha256=sha(path)) for path in a.raw_pcm]
    if net is not None:
        report["metric_weights"]={p.name:sha(p) for p in (Path(torch.hub.get_dir())/"checkpoints").glob("alexnet-*.pth")}
        weight=Path(lpips.__file__).parent/"weights/v0.1/alex.pth";report["metric_weights"]["lpips-alex-v0.1"]=sha(weight)
    else:report["lpips_note"]="All RGB samples identical; LPIPS is exactly zero without evaluating the network."
    (a.output/"metrics.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(gates));raise SystemExit(0 if report["passed"] else 1)


if __name__=="__main__":main()
