#!/usr/bin/env python3
"""Read-only environment manifest; never collects credentials or process argv."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

def command(args):
    try:
        r=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
        return {'argv':args,'returncode':r.returncode,'output':r.stdout}
    except (OSError,subprocess.TimeoutExpired) as e:return {'argv':args,'error':str(e)}

result={'recorded_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'platform':platform.platform(),
    'environment':{k:v for k,v in os.environ.items() if k.startswith(('H3_','CUDA_','CUBLAS_'))},
    'commands':[command(c) for c in [
        ['nvidia-smi','-q'],['/usr/local/cuda/bin/nvcc','--version'],['gcc','--version'],
        ['df','-h','/root','/workspace'],['free','-b'],
        ['python3','-c','import torch,triton; print(torch.__version__,torch.version.cuda,triton.__version__)']]]}
for p in ('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.swap.max'):
    try:result[p]=Path(p).read_text().strip()
    except OSError:pass
for p in ('third_party/sageattention/upstream.json','bin/h3cli'):
    if Path(p).is_file():result[p+'_sha256']=hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=list(Path('.').glob('src/**/*.c'))+list(Path('.').glob('src/**/*.h'))+list(Path('.').glob('src/**/*.cu'))+list(Path('.').glob('src/**/*.cuh'))+[Path('Makefile')]
result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source) if p.is_file()}
drivers=list(Path('tests').glob('attention*'))+list(Path('tests').glob('test_attention*'))
result['driver_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(drivers) if p.is_file()}
result['native_dependencies']=command(['ldd','./bin/h3cli'])
Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
