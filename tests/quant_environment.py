#!/usr/bin/env python3
"""Capture host/library/fixture identity without launching an inference job."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess

def command(args):
    r=subprocess.run(args,text=True,capture_output=True,timeout=20)
    return dict(command=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

out=Path('outputs/quant-5090');out.mkdir(parents=True,exist_ok=True)
lib=ctypes.CDLL('/usr/local/cuda/lib64/libcublasLt.so');lib.cublasLtGetVersion.restype=ctypes.c_size_t
data=dict(starting_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),work=str(Path.cwd()),
    model=str(Path(os.environ.get('H3_MODEL_DIR','models/MiniMax-H3')).expanduser().resolve()),
    cache=str(Path(os.environ.get('H3_TEST_QUANT_CACHE','outputs/quant-5090/packed')).expanduser().resolve()),cublasLt=lib.cublasLtGetVersion(),
    recipe=1,exceptions=[],rotations=[],correction=None,
    gpu=command(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,compute_cap','--format=csv']),
    nvcc=command(['/usr/local/cuda/bin/nvcc','--version']),storage=command(['df','-h',str(Path.cwd())]),
    sources={str(p):sha(p) for p in sorted(Path('.').glob('src/**/*')) if p.is_file() and p.suffix in ('.c','.h','.m','.cu','.cuh','.metal')},
    fixtures={str(p):sha(p) for p in sorted(Path('inputs').glob('*.jpg'))},
    baseline_binary_sha256=sha(out/'h3-baseline'),
    unavailable=['Turbo/LoRA folded models and adapter weights are absent on this node'],
    calibration=['inputs/1.jpg (pilot only; dynamic amax does not fit dataset statistics)'],
    held_out=['image12','detail','face','first-last','images','video-audio'],
    attention='fast CUDA cuDNN where eligible',sampler='existing CUDA Euler',layers=50,reuse=1,core_reuse=1,
    token_reduction=False,steps=20,primary_presets=[[288,384,56],[480,640,22]],
    quality_decisions=dict(fp8='pending',nvfp4='pending'))
(out/'environment.json').write_text(json.dumps(data,indent=2)+'\n')
print('wrote',out/'environment.json')
