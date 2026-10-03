#!/usr/bin/env python3
"""Independent FP32 tile oracle and range-audited FP16 matrix experiment."""
import argparse,json,pathlib,time
import numpy as np
import torch
from safetensors.torch import load_file
import torch.nn.functional as F

def main():
    p=argparse.ArgumentParser();p.add_argument('weights');p.add_argument('fixture');p.add_argument('output');p.add_argument('--precision',choices=['fp32','fp16'],default='fp32');a=p.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.set_num_threads(4);out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
    weights=load_file(a.weights,device='cuda');half=a.precision=='fp16';ranges=[]
    matrices={k:(v.half() if half and v.ndim==2 and min(v.shape)>=128 else v) for k,v in weights.items()}
    for name,value in matrices.items():assert torch.isfinite(value).all(),name
    def norm(x,w=None):
        result=x*torch.rsqrt((x*x).mean(-1,keepdim=True)+1e-5)
        return result if w is None else result*weights[w]
    def linear(x,name):
        w=matrices[name+'.weight'].reshape(weights[name+'.bias'].shape[0],-1)
        if w.dtype==torch.float16:
            ranges.append((name,float(x.abs().max())));assert torch.isfinite(x.half()).all(),name
        return F.linear(x.to(w.dtype),w,None).float()+weights[name+'.bias']
    cos=torch.from_numpy(np.fromfile(a.fixture+'.cos','<f4').reshape(1,1797,24)).cuda();sin=torch.from_numpy(np.fromfile(a.fixture+'.sin','<f4').reshape(1,1797,24)).cuda()
    def rope(x):return torch.cat([x[...,:24]*cos-x[...,24:48]*sin,x[...,24:48]*cos+x[...,:24]*sin,x[...,48:]],-1)
    x=torch.from_numpy(np.fromfile(a.fixture+'.input','<f4').reshape(1792,24)).cuda();torch.cuda.synchronize();start=time.monotonic()
    with torch.inference_mode():
        x=linear(linear(x,'post_quant_conv'),'decoder.x_embedder')
        x=torch.cat([x,weights['decoder.register_tokens'].reshape(4,2048),torch.zeros((1,2048),device='cuda')])
        for i in range(36):
            p=f'decoder.transformer_blocks.{i}.'
            qkv=linear(norm(x,p+'norm1.weight'),p+'attn.to_qkv').reshape(1797,32,3,64)
            q,k,v=[qkv[:,:,j,:].transpose(0,1) for j in range(3)];q,k=rope(norm(q)),rope(norm(k))
            if half:q,k,v=[z.half() for z in (q,k,v)]
            prob=torch.softmax((q@k.transpose(-2,-1)).float()*.125,-1)
            y=(prob.to(v.dtype)@v).float().transpose(0,1).reshape(1797,2048)
            x=x+linear(y,p+'attn.to_out')*weights[p+'scale1']
            y=linear(norm(x,p+'norm2.weight'),p+'ff.w1');gate,up=y.chunk(2,-1)
            y=gate/(1+torch.exp(-gate))*up
            x=x+linear(y,p+'ff.w2')*weights[p+'scale2']
        x=F.layer_norm(x,(2048,),weights['decoder.norm_out.weight'],weights['decoder.norm_out.bias'],1e-5)
        x=linear(x,'decoder.proj_out');torch.cuda.synchronize();seconds=time.monotonic()-start
    result=x.cpu().numpy();result.tofile(out/'projected.f32');reference=np.fromfile(a.fixture+'.projected','<f4').reshape(result.shape);delta=result.astype('float64')-reference
    r=dict(precision=a.precision,torch=torch.__version__,finite=bool(np.isfinite(result).all()),rel_l2=float(np.linalg.norm(delta)/np.linalg.norm(reference.astype('float64'))),max_abs=float(abs(delta).max()),diagnostic_seconds_including_range_synchronization=seconds,max_matrix_operand=max((v for _,v in ranges),default=None),matrix_operand_ranges=ranges,fp16_reduced_precision_reduction=False)
    (out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='matrix_operand_ranges'}))
if __name__=='__main__':main()
