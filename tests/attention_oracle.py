#!/usr/bin/env python3
"""Pinned upstream oracle and independent FP32 SDPA checks for native attention.

Run in the separate Torch/Sage validation environment, never during inference.
Native inputs/outputs are BF16 files; the native process owns its CUDA context.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
import re
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from attention_run import sha

def metrics(candidate,reference):
    x=candidate.float().reshape(-1).double();y=reference.float().reshape(-1).double()
    delta=x-y;yn=torch.linalg.vector_norm(y);xn=torch.linalg.vector_norm(x)
    return {'finite':bool(torch.isfinite(x).all()),'rmse':float(delta.square().mean().sqrt()),
        'relative_l2':float(torch.linalg.vector_norm(delta)/yn.clamp_min(1e-30)),
        'cosine':float(torch.dot(x,y)/(xn*yn).clamp_min(1e-30)) if xn and yn else float(torch.equal(x,y)),
        'max_abs':float(delta.abs().max()),'exact_fraction':float((x==y).double().mean())}

def blocked_reference(q,k,v,scale):
    # B,H,Q,D queries against all keys, with bounded online FP32 softmax.
    result=torch.empty_like(q,dtype=torch.float32)
    for first in range(0,q.shape[2],128):
        query=q[:,:,first:first+128].float()
        maximum=torch.full((*query.shape[:-1],1),-float('inf'),device=q.device)
        denominator=torch.zeros_like(maximum);value=torch.zeros_like(query)
        for start in range(0,k.shape[2],4096):
            key=k[:,:,start:start+4096].float();val=v[:,:,start:start+4096].float()
            scores=(query@key.transpose(-1,-2))*scale
            next_max=torch.maximum(maximum,scores.max(-1,keepdim=True).values)
            factor=(maximum-next_max).exp();probability=(scores-next_max).exp()
            value=value*factor+probability@val
            denominator=denominator*factor+probability.sum(-1,keepdim=True);maximum=next_max
        result[:,:,first:first+128]=value/denominator
    return result

def upstream(mode,q,k,v,scale):
    if mode=='sage2++':
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
        return sageattn_qk_int8_pv_fp8_cuda(q,k,v,tensor_layout='HND',is_causal=False,
            qk_quant_gran='per_thread',sm_scale=scale,pv_accum_dtype='fp32+fp16',smooth_k=True,smooth_v=False)
    from sageattn3.api import preprocess_qkv,scale_and_quant_fp4,scale_and_quant_fp4_permute,scale_and_quant_fp4_transpose
    import fp4attn_cuda
    count=q.shape[2];kl=k.shape[2]
    # Public Sage3 wrapper ignores caller scale and modifies K. Clone inputs and
    # call its unchanged native kernel with the actual h3cli BF16-rounded scale.
    q,k,v,delta=preprocess_qkv(q.clone(),k.clone(),v.clone(),True)
    qi,qs=scale_and_quant_fp4(q);ki,ks=scale_and_quant_fp4_permute(k);vi,vs=scale_and_quant_fp4_transpose(v)
    # Upstream's packed-row tail mask is incorrect for general true lengths.
    # Apply an independent logical-key mask through the score correction and
    # pass the padded length, leaving the upstream binary unchanged.
    if kl%128:delta[...,kl:]=-float('inf')
    return fp4attn_cuda.fwd(qi,ki,vi,qs,ks,vs,delta,k.shape[2],None,scale,False,True,True)[0][:,:,:count,:].contiguous()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--binary',default='./bin/attention_native');p.add_argument('--output',required=True)
    p.add_argument('--modes',nargs='+',default=['sage2++','sage3'])
    p.add_argument('--lengths',nargs='+',type=int,default=[1,15,16,17,63,64,65,127,128,129,257,2281])
    p.add_argument('--patterns',nargs='+',default=['normal','outlier','tiny','zero','constant'])
    p.add_argument('--seed',type=int,default=1001);p.add_argument('--no-upstream',action='store_true')
    p.add_argument('--input',help='Captured NHD BF16 Q/K/V file; use one sequence length')
    p.add_argument('--segments-log',help='Model log containing captured joint-token segment ranges')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(a.seed)
    scale=float(torch.tensor(1/math.sqrt(128),dtype=torch.bfloat16))
    results=[];failed=False;segments={};binary_hash=sha(a.binary)
    if a.segments_log:
        for kind,start,stop in re.findall(r'attention validation segment kind=(\d+) start=(\d+) stop=(\d+)',Path(a.segments_log).read_text()):
            label=['text','condition','reference_image','reference_audio','audio','video'][int(kind)]
            segments.setdefault(label,[]).extend(range(int(start),int(stop)))
    for seq in a.lengths:
      for pattern in a.patterns:
        if a.input:
            raw=np.fromfile(a.input,dtype=np.uint16)
            cpu=torch.from_numpy(raw.copy()).view(torch.bfloat16).reshape(3,1,seq,56,128)
        else:
            cpu=torch.randn(3,1,seq,56,128,dtype=torch.float32)
            if pattern=='outlier':cpu[:,:,:,::7,0]*=12
            elif pattern=='tiny':cpu*=1e-4
            elif pattern=='zero':cpu.zero_()
            elif pattern=='constant':cpu.fill_(.5)
            cpu=cpu.bfloat16()
        # Real short captures otherwise write and reread four native results
        # over the network mount. Their QKV + one output need at most 224 MiB.
        # Keep all evidence under out; only regenerable interchange files use
        # private RAM scratch. Long cases retain the disk-backed path.
        scratch=out
        ram=Path('/dev/shm')
        if seq<=4096 and ram.is_dir() and shutil.disk_usage(ram).free>=512*1024*1024:
            scratch=ram
        with tempfile.TemporaryDirectory(prefix='h3-attention-oracle-',dir=scratch) as temp:
            inputs=Path(temp)/'qkv.bin';inputs.write_bytes(cpu.view(torch.uint16).numpy().tobytes())
            input_hash=sha(inputs)
            # Correctness checks retain only reference outputs on the GPU while
            # the native child runs. Performance/VRAM qualification uses the
            # separate native benchmark, with no Torch process present.
            q,k,v=(t.permute(0,2,1,3).contiguous().cuda() for t in cpu)
            indices=None
            if seq>4096:
                blocks=sorted({0,(seq//256)*128,((seq-1)//128)*128})
                indices=torch.tensor([r for start in blocks for r in range(start,min(seq,start+128))],device=q.device)
                query=q.index_select(2,indices)
                reference=blocked_reference(query,k,v,scale)
            else:
                query=q
                reference=blocked_reference(q,k,v,scale)
            oracle={}
            if not a.no_upstream:
                for mode in a.modes:oracle[mode]=upstream(mode,query.clone(),k.clone(),v.clone(),scale)
            selected=indices.cpu() if indices is not None else None
            del q,k,v,query,indices;torch.cuda.synchronize();torch.cuda.empty_cache()
            for mode in a.modes:
              for layout in (0,1):
                name=f'{mode}-{seq}-{pattern}-layout{layout}';dest=Path(temp)/'output.bin'
                command=[a.binary,mode,str(seq),str(dest),str(layout),'1',str(inputs)]
                start=time.monotonic()
                r=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
                (out/(name+'.log')).write_text(r.stdout)
                row={'name':name,'mode':mode,'sequence':seq,'pattern':pattern,'head_major':layout,
                    'seed':a.seed,'scale':scale,'returncode':r.returncode,'wall_seconds':time.monotonic()-start,
                    'input_sha256':input_hash,'binary_sha256':binary_hash,'command':command}
                if r.returncode==0:
                    got=torch.from_numpy(np.fromfile(dest,dtype=np.uint16).copy()).view(torch.bfloat16)
                    got=got.reshape(1,56,seq,128) if layout else got.reshape(1,seq,56,128).permute(0,2,1,3)
                    if selected is not None:got=got.index_select(2,selected)
                    got=got.cuda()
                    row['fp32']=metrics(got,reference)
                    if mode in oracle:row['upstream']=metrics(got,oracle[mode])
                    if segments:
                        positions=selected.tolist() if selected is not None else list(range(seq))
                        row['query_domains']={}
                        for label,indices in segments.items():
                            wanted=set(indices);at=torch.tensor([i for i,pos in enumerate(positions) if pos in wanted],dtype=torch.long,device=got.device)
                            if not len(at):continue
                            domain={'fp32':metrics(got.index_select(2,at),reference.index_select(2,at))}
                            if mode in oracle:domain['upstream']=metrics(got.index_select(2,at),oracle[mode].index_select(2,at))
                            row['query_domains'][label]=domain
                    # Port errors are judged separately from approximate SDPA error.
                    # Calibration gate; held-out/model thresholds are frozen separately.
                    oracle_finite=mode not in oracle or bool(torch.isfinite(oracle[mode]).all())
                    row['oracle_finite']=oracle_finite
                    row['oracle_adapter']='sage3-logical-delta-tail-mask-v1' if mode=='sage3' else 'explicit-h3-scale'
                    row['port_pass']=row['fp32']['finite'] and (mode not in oracle or row['upstream']['relative_l2']<=.01)
                    if mode in oracle and segments:
                        row['port_pass'] &= all(v['fp32']['finite'] and v['upstream']['relative_l2']<=.01
                            for v in row['query_domains'].values())
                    if mode=='sage2++' and pattern=='zero' and not oracle_finite:
                        row['oracle_exception']='Pinned Sage2++ divides zero V by zero; independent exact-zero contract applies.'
                        row['port_pass']=row['fp32']['finite'] and row['fp32']['max_abs']==0
                    if pattern in ('zero','constant'):row['port_pass'] &= row['fp32']['max_abs']<=.02
                else:row['port_pass']=False
                results.append(row);failed |= not row['port_pass'];print(json.dumps(row),flush=True)
                (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    return int(failed)

if __name__=='__main__':raise SystemExit(main())
