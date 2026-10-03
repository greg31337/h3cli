#!/usr/bin/env python3
"""Compare the first real VAE tile, with lossless dtype/layout adaptation."""
import argparse
import json
from pathlib import Path
import numpy as np
from cuda_sglang_compare import oracle,metric


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('oracle',type=Path);p.add_argument('native',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();prefix='decoder.transformer_blocks.0';results=[]
    pairs={'input':prefix+'.input','norm1':prefix+'.norm1.output','qkv':prefix+'.attn.to_qkv.output',
        'query':'query','key':'key','value':'value','attention':prefix+'.attn.attn.output',
        'projection':prefix+'.attn.to_out.output','residual1':prefix+'.norm2.input','norm2':prefix+'.norm2.output',
        'ff1':prefix+'.ff.w1.output','swiglu':'w2.input','mlp':'w2.output','output':prefix+'.output'}
    gate=json.loads(Path(__file__).with_name('cuda_sglang_contract.json').read_text())['gates']['operation']
    for name,other in pairs.items():
        try:
            x=np.fromfile(a.native/'capture'/('vae.'+name+'.f32'),dtype='<f4');y=oracle(a.oracle,other).astype(np.float32).reshape(-1)
            r=dict(name=name,**metric(x,y))
        except (OSError,ValueError) as exc:r=dict(name=name,error=str(exc))
        r['passed']=bool(r.get('finite') and r.get('compatible') and r['relative_l2']<=gate['relative_l2_max'] and
            r['cosine']>=gate['cosine_min'] and r['max_abs']<=gate['absolute_floor']+gate['absolute_max_per_reference_rms']*r['reference_rms'])
        results.append(r);print(json.dumps(r),flush=True)
    rgb=metric(np.fromfile(a.native/'rgb.f32',dtype='<f4'),np.fromfile(a.oracle/'rgb.f32',dtype='<f4'))
    print('RGB',json.dumps(rgb),flush=True)
    report=dict(domain='first tile of identical clean latents; original full VAE',results=results,rgb=rgb,
        first_failing_boundary=next((r['name'] for r in results if not r['passed']),None),passed=all(r['passed'] for r in results))
    a.output.write_text(json.dumps(report,indent=2)+'\n');raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
