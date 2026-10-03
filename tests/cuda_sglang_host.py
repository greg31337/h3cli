#!/usr/bin/env python3
"""Historical oracle-generation diagnostic; not a current qualification gate.

No model assets or GPU allocations. Pass a shared library built from
src/sglang/sglang.c and src/sglang/sglang_rng.cpp.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import numpy as np
import torch
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import minimax_h3_time_shift_sigmas


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("library",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=p.parse_args();lib=C.CDLL(str(args.library.resolve()))
    fp=C.POINTER(C.c_float)
    lib.h3_sglang_normal.argtypes=[C.c_uint64,fp,C.c_size_t]
    lib.h3_sglang_euler.argtypes=[fp,fp,C.c_size_t,C.c_float,C.c_float]
    lib.h3_sglang_video_condition.argtypes=[fp,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.c_uint64]
    lib.h3_sglang_audio_condition.argtypes=[fp,C.c_size_t,C.c_uint64]
    # Read H3_MAX_STEPS from the matching source instead of guessing its ABI.
    import re
    header=Path(__file__).resolve().parents[1]/"src/host.h"
    maximum=int(re.search(r"#define H3_MAX_STEPS (\d+)",header.read_text())[1])
    class Schedule(C.Structure):
        _fields_=[("steps",C.c_int),("video",C.c_float*(maximum+1)),("audio",C.c_float*(maximum+1))]
    results=[]
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import minimax_h3_resolve_reference_image_shape
    lib.h3_sglang_reference_image_canvas.argtypes=[C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_int)]
    for w,h in ((1365,1821),(640,480),(4096,2160),(2048,2048),(128,513),(32,128),(32,129),(0,480),(2048,2064),(2048,2096)):
        rw,rh=C.c_int(),C.c_int();ok=bool(lib.h3_sglang_reference_image_canvas(w,h,C.byref(rw),C.byref(rh)))
        try:
            shape=minimax_h3_resolve_reference_image_shape(width=w,height=h)
            exact=ok and (rw.value,rh.value)==(shape['width'],shape['height'])
        except ValueError:exact=not ok
        results.append(dict(kind='reference_image_geometry',source=[w,h],native=[rw.value,rh.value],exact=exact))
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.condition_noise import (
        minimax_h3_imgvid_cond_noise_aug_rows,minimax_h3_audio_cond_noise_aug_rows)
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import MINIMAX_H3_AUDIO_REF_COND_TIMESTEP
    assert MINIMAX_H3_AUDIO_REF_COND_TIMESTEP==1.0
    for seed in (42,917):
        for shapes,target in [([(1,4,6)],37), ([(1,30,40),(1,30,40)],107), ([(1,8,12),(7,30,40)],37)]:
            sizes=[t*h*w*24 for t,h,w in shapes]
            clean=torch.linspace(-3,3,sum(sizes)).reshape(-1,96);got=clean.numpy().copy().reshape(-1)
            offset=0
            for (t,h,w),count in zip(shapes,sizes):
                assert lib.h3_sglang_video_condition(got[offset:].ctypes.data_as(fp),t,h,w,target,len(shapes),seed)
                offset+=count
            ref=minimax_h3_imgvid_cond_noise_aug_rows(clean.to(args.device),condition_shapes=shapes,
                target_latent_t=target,imgvid_cond_num_frames=len(shapes),seed=seed,noise_aug=.999).cpu().numpy().reshape(-1)
            results.append(dict(kind='visual_condition_noise',seed=seed,shapes=shapes,target_time=target,
                exact=bool(np.array_equal(got,ref)),max_abs=float(np.max(abs(got-ref)))))
        times=[40,80];clean=torch.linspace(-3,3,sum(times)*64).reshape(-1,32);got=clean.numpy().copy().reshape(-1);offset=0
        for t in times:
            assert lib.h3_sglang_audio_condition(got[offset:].ctypes.data_as(fp),t*64,seed);offset+=t*64
        ref=minimax_h3_audio_cond_noise_aug_rows(clean,condition_audio_t=times,seed=seed,noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP).numpy().reshape(-1)
        results.append(dict(kind='audio_condition_noise',seed=seed,times=times,exact=bool(np.array_equal(got,ref)),max_abs=float(np.max(abs(got-ref)))))
    for seed in (0,42,987654321,2**32+42):
        for n in (16,17,31,32,97,13248,1065600,3081600):
            target=np.empty(n,dtype=np.float32)
            assert lib.h3_sglang_normal(seed,target.ctypes.data_as(fp),n)
            ref=torch.randn(n,generator=torch.Generator().manual_seed(seed),dtype=torch.float32).numpy()
            results.append(dict(kind="rng",seed=seed,n=n,exact=bool(np.array_equal(target,ref)),
                                max_abs=float(np.max(np.abs(target-ref))),different=int(np.count_nonzero(target!=ref))))
    for evaluations in (1,2,6,50):
        schedule=Schedule();assert lib.h3_sglang_schedule(evaluations,C.byref(schedule))
        for modality,shift in (("video",12.0),("audio",3.0)):
            ref=np.array(minimax_h3_time_shift_sigmas(num_steps=evaluations+1,shift_scale=shift),dtype=np.float32)
            candidate=np.asarray(getattr(schedule,modality))[:evaluations+1]
            results.append(dict(kind="schedule",evaluations=evaluations,modality=modality,
                                exact=bool(np.array_equal(candidate,ref)),reference=ref.tolist(),candidate=candidate.tolist()))
            x=torch.randn(4096,generator=torch.Generator().manual_seed(71))
            v=torch.randn(4096,generator=torch.Generator().manual_seed(91))
            got=x.numpy().copy()
            native_velocity=v.numpy().copy();x=x.to(args.device);v=v.to(args.device)
            for i in range(evaluations):
                sigma,next_=ref[i:i+2]
                assert lib.h3_sglang_euler(got.ctypes.data_as(fp),native_velocity.ctypes.data_as(fp),got.size,float(sigma),float(next_))
                t=torch.tensor(1.0-float(sigma),dtype=torch.float32,device=args.device)
                sigma_t=1.0-t
                ratio=torch.tensor(float(next_),device=args.device)/torch.tensor(float(sigma),device=args.device)
                x=ratio*x+(1-ratio)*(x+sigma_t*v)
            results.append(dict(kind="euler",evaluations=evaluations,modality=modality,
                                exact=bool(np.array_equal(got,x.cpu().numpy())),max_abs=float(np.max(np.abs(got-x.cpu().numpy())))))
    report=dict(torch=torch.__version__,cpu_capability=torch.backends.cpu.get_cpu_capability(),
                device=args.device,passed=all(r["exact"] for r in results),cases=results)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="cases"}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__=="__main__":main()
