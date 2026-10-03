#!/usr/bin/env python3
"""Compare Sage3 smoothing, correction, scales and bytes with pinned upstream."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import numpy as np
import torch
from attention_oracle import metrics

p=argparse.ArgumentParser()
p.add_argument('--binary',default='./bin/attention_native')
p.add_argument('--output',required=True,type=Path)
p.add_argument('--length',default=129,type=int)
p.add_argument('--seed',default=1001,type=int)
p.add_argument('--mode',default='sage3',choices=['sage2++','sage3'])
p.add_argument('--input',type=Path,help='Captured NHD BF16 Q/K/V; length must match')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=False
cpu=torch.from_numpy(np.fromfile(a.input,dtype=np.uint16).copy()).view(torch.bfloat16).reshape(3,1,a.length,56,128) if a.input else torch.randn(3,1,a.length,56,128).bfloat16()
inputs=a.output/'input.qkv';inputs.write_bytes(cpu.view(torch.uint16).numpy().tobytes())
r=subprocess.run([a.binary,a.mode,str(a.length),str(a.output/'output.bf16'),'0','1',str(inputs)],
    env={**os.environ,'H3_TEST_SAGE_PACK_DIR':str(a.output)},capture_output=True,text=True,timeout=120)
(a.output/'native.log').write_text(r.stdout+r.stderr)
if r.returncode:raise RuntimeError(r.stdout+r.stderr)
q,k,v=(t.permute(0,2,1,3).contiguous().cuda() for t in cpu)
if a.mode=='sage3':
    from sageattn3.api import preprocess_qkv,scale_and_quant_fp4,scale_and_quant_fp4_permute,scale_and_quant_fp4_transpose,triton_group_mean
    padded=torch.nn.functional.pad(q,(0,0,0,(-a.length)%128))
    _,qm=triton_group_mean(padded)
    q,k,v,delta=preprocess_qkv(q,k,v,True)
    qi,qs=scale_and_quant_fp4(q);ki,ks=scale_and_quant_fp4_permute(k);vi,vs=scale_and_quant_fp4_transpose(v)
    reference={'q-center.bf16':q,'k-center.bf16':k,'q-mean.bf16':qm,'delta.f32':delta,
               'q.fp4':qi,'k.fp4':ki,'v.fp4':vi,'q.scales':qs,'k.scales':ks,'v.scales':vs}
else:
    from sageattention.triton.quant_per_thread import per_thread_int8
    from sageattention.quant import per_channel_fp8
    km=k.mean(2,keepdim=True);qi,qs,ki,ks=per_thread_int8(q,k,km)
    vi,vs,_=per_channel_fp8(v,tensor_layout='HND',scale_max=2.25,smooth_v=False)
    qi=torch.nn.functional.pad(qi,(0,0,0,(-a.length)%128))
    ki=torch.nn.functional.pad(ki,(0,0,0,(-a.length)%64))
    reference={'k-mean.bf16':km,'q.int8':qi,'k.int8':ki,'v.fp8':vi,
        'q-scale.f32':qs,'k-scale.f32':ks,'v-scale.f32':vs}
rows={}
for name,expected in reference.items():
    actual=np.fromfile(a.output/name,dtype=np.uint8)
    want=expected.cpu().contiguous().view(torch.uint8).numpy().reshape(-1)
    (a.output/(name+'.reference')).write_bytes(want.tobytes())
    assert actual.shape==want.shape,(name,actual.shape,want.shape)
    row={'bytes':len(actual),'byte_equal_fraction':float(np.mean(actual==want))}
    if name.endswith(('.bf16','.f32')):
        dtype=torch.bfloat16 if name.endswith('.bf16') else torch.float32
        row.update(metrics(torch.from_numpy(actual.copy()).view(dtype),expected.cpu()))
    unequal=np.flatnonzero(actual!=want)
    row['first_different_bytes']=[{'offset':int(i),'actual':int(actual[i]),'expected':int(want[i])} for i in unequal[:8]]
    rows[name]=row
(a.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
