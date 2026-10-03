#!/usr/bin/env python3
"""Check actual prepared AdaLN rows against the official timestep modules on CPU."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import struct
import numpy as np
import torch
from safetensors import safe_open
from diffusers import MiniMaxH3Transformer3DModel
from test_sampler_file import entries
from test_refvideo_dit import compare
ROOT=Path(__file__).resolve().parents[1]


def rows(path):
    data=next(p[-1] for p in entries(path.read_bytes()) if p[0]==30)
    version=struct.unpack_from('<I',data)[0];count=struct.unpack_from('<Q',data,36)[0]
    assert version==1
    cursor=44
    for _ in range(count):
        kind,n=struct.unpack_from('<IQ',data,cursor);cursor+=12
        if kind==100:
            values=np.frombuffer(data,'<u2',count=n,offset=cursor).astype(np.uint32)
            return (values<<16).view(np.float32).reshape(-1,3*6*5376)[[0,-1]]
        cursor+=n*2
    raise ValueError('prepared block 0 is absent')


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(8)
    root=ROOT/'models/MiniMax-H3/Ref2VA/transformer';index=json.loads(next(root.glob('*index.json')).read_text())['weight_map']
    def weight(key):
        with safe_open(root/index[key],framework='pt') as f:return f.get_tensor(key)
    with torch.device('meta'):model=MiniMaxH3Transformer3DModel()
    model.time_embedder.load_state_dict({new+'.'+suffix:weight(old+'.'+suffix)
        for new,old in [('linear_1','time_embedder.proj_in'),('linear_2','time_embedder.proj_out')]
        for suffix in ('weight','bias')},strict=True,assign=True)
    module=model.transformer_blocks[0].adaln_proj
    module.load_state_dict({'linear.'+s:weight('blocks.0.adaln_proj.linear.'+s) for s in ('weight','bias')},strict=True,assign=True)
    temb=model.time_embedder(model.time_proj(torch.tensor([0.,.999],dtype=torch.float32)))
    want=torch.cat(module(temb),dim=-1).reshape(2,-1).float().numpy()
    before=compare(rows(args.before),want);after=compare(rows(args.after),want)
    result={'before':before,'after':after,
        'passed':after['relative_max']<.025 and after['relative_l2']<.025 and after['relative_l2']<before['relative_l2'],
        'timesteps':[0.,.999],'projection':'block 0, all modalities and six slots',
        'source_sha256':hashlib.sha256(Path(inspect.getfile(type(model))).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
    assert result['passed'],'released AdaLN precision correction does not match the official module'
if __name__=='__main__':main()
