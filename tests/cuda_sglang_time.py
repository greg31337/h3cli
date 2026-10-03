#!/usr/bin/env python3
"""Compare batching of the real timestep/AdaLN projections to pinned Torch."""
import argparse,json,os,struct,subprocess
from pathlib import Path
import numpy as np
import torch
from cuda_sglang_compare import metric,native,oracle

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--native',type=Path,required=True);p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--cublas',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    headers=[]
    for f in a.model.glob('*.safetensors'):
        with f.open('rb') as stream:
            n=struct.unpack('<Q',stream.read(8))[0];headers.append((f,n,json.loads(stream.read(n))))
    def weight(name):
        for f,n,h in headers:
            if name in h:
                meta=h[name];begin,end=meta['data_offsets']
                if end-begin>600*1024*1024:raise ValueError('replay weight exceeds bound')
                with f.open('rb') as s:s.seek(8+n+begin);raw=bytearray(s.read(end-begin))
                return torch.frombuffer(raw,dtype={'F32':torch.float32,'BF16':torch.bfloat16}[meta['dtype']]).reshape(meta['shape']).cuda()
        raise KeyError(name)
    wi,bi,wo,bo=[weight('time_embedder.'+s) for s in ('proj_in.weight','proj_in.bias','proj_out.weight','proj_out.bias')]
    aw,ab=[weight('blocks.0.adaln_proj.linear.'+s) for s in ('weight','bias')]
    matrix_name='blocks.0.adaln_proj.linear.weight';bias_name='blocks.0.adaln_proj.linear.bias'
    f,header_bytes,header=next((f,n,h) for f,n,h in headers if matrix_name in h and bias_name in h)
    report={};env=os.environ.copy();env['H3_SGLANG_CUBLAS_LIBRARY']=a.cublas
    with torch.inference_mode():
        for rows in (1,2,11):
            x=np.tile(np.r_[np.ones(128),np.zeros(128)].astype('<f4'),(rows,1))
            source=a.out/f'features-{rows}.f32';dest=a.out/f'activation-{rows}.bf16';x.tofile(source)
            run=subprocess.run([str(a.source.resolve()/'bin/cuda_sglang_time'),str(a.model),str(source),str(rows),str(dest)],
                cwd=a.source,env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True)
            (a.out/f'native-{rows}.log').write_text(run.stdout+run.stderr);run.check_returncode()
            t=torch.from_numpy(x).cuda();hidden=torch.nn.functional.linear(t,wi,bi)
            tem=torch.nn.functional.linear(torch.nn.functional.silu(hidden),wo,bo)
            act=torch.nn.functional.silu(tem).bfloat16()
            got=native(a.out,dest.name,'<u2').reshape(rows,-1)
            report[f'activation-{rows}']=metric(got,act.float().cpu().numpy())
            projected=a.out/f'modulation-{rows}.bf16'
            run=subprocess.run([str(a.source.resolve()/'bin/cuda_sglang_replay'),'linear-bias',str(dest),str(f),
                str(8+header_bytes+header[matrix_name]['data_offsets'][0]),str(8+header_bytes+header[bias_name]['data_offsets'][0]),
                str(projected),str(rows),'2688','96768'],cwd=a.source,env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True)
            (a.out/f'projection-{rows}.log').write_text(run.stdout+run.stderr);run.check_returncode()
            expected=torch.nn.functional.linear(act,aw,ab).reshape(rows,96768)
            report[f'projection-{rows}']=metric(native(a.out,projected.name,'<u2').reshape(rows,96768),expected.float().cpu().numpy())
            if rows==1:
                reference=act.clone()
                params=torch.nn.functional.linear(act,aw,ab).reshape(-1,6,5376)
                params.float().cpu().numpy().tofile(a.out/'oracle-modulation.f32')
                recorded=native(a.native/'capture','step-000.block-0.modulation.bf16','<u2').reshape(-1,6,5376)[:3]
                report['recorded-modulation']=metric(recorded,params.float().cpu().numpy())
                rms=oracle(a.oracle/'capture','step-000.blocks.0.norm1.output')[:11559]
                indices=oracle(a.oracle/'capture','step-000.blocks.0.kwargs.combined_indices')[:11559].astype(np.int64)
                y=torch.from_numpy(rms.copy()).cuda().bfloat16();idx=torch.from_numpy(indices).cuda()
                y=y*(1+params[idx,1])+params[idx,0]
                report['reconstructed-modulation']=metric(y.float().cpu().numpy(),oracle(a.oracle/'capture','step-000.blocks.0.attn.qkv_proj.input.0')[:11559])
            report[f'activation-{rows}-versus-one']=metric(got[:1],reference.float().cpu().numpy())
    (a.out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
