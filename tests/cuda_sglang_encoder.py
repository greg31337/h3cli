#!/usr/bin/env python3
"""Pinned video encoder on a real 640x480 image; bounded first-tile captures."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--image',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from sglang.multimodal_gen.runtime.server_args import set_global_server_args
    from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import MiniMaxH3VideoVAEConfig
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3VideoVAE
    set_global_server_args(SimpleNamespace(attention_backend='torch_sdpa',attention_backend_config=None,kv_gather_degree=1))
    config=json.loads((a.model/'config.json').read_text());c=MiniMaxH3VideoVAEConfig();c.update_model_arch(config)
    with torch.device('cpu'):model=MiniMaxH3VideoVAE(c)
    weights=load_file(str(a.model/'source/model.safetensors'),device='cpu');model.load_state_dict(weights,strict=True,assign=True);del weights
    model.eval();model.requires_grad_(False);model.encoder.to('cuda');model.quant_conv.to('cuda')
    pixels=np.array(Image.open(a.image).convert('RGB'))
    if pixels.shape!=(480,640,3):raise ValueError('fixture must be 640x480')
    x=torch.from_numpy(pixels).permute(2,0,1).contiguous().cuda().float().div_(255)[None,:,None]
    mean=torch.tensor([.485,.456,.406],device='cuda').reshape(1,3,1,1,1);std=torch.tensor([.229,.224,.225],device='cuda').reshape(1,3,1,1,1)
    x=x.sub_(mean).div_(std)
    captured={};total=0
    def save(name,t):
        nonlocal total
        if name in captured:return
        t=t.detach().contiguous().cpu();raw=t.reshape(-1).view(torch.uint8).numpy().tobytes();total+=len(raw)
        if total>512*1024**2:raise ValueError('first encoder tile exceeds 512 MiB capture budget')
        meta=dict(dtype=str(t.dtype),shape=list(t.shape),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        (a.out/(name+'.bin')).write_bytes(raw);(a.out/(name+'.json')).write_text(json.dumps(meta)+'\n');captured[name]=meta
    names={'encoder.conv_in','encoder.down.0.block.0.norm1','encoder.down.0.block.0.conv1','encoder.down.0.block.0.norm2','encoder.down.0.block.0.conv2','encoder.down.0.block.0','encoder.norm_out','encoder.conv_out','quant_conv'}
    boundary_names={f'encoder.down.{level}.block.{block}' for level in range(6) for block in range(2)}|{f'encoder.down.{level}.downsample' for level in range(4)}
    for name,module in model.named_modules():
        if name not in names and name not in boundary_names:continue
        def pre(m,args,name=name):save(name+'.input',args[0])
        def post(m,args,out,name=name):save(name+'.output',out)
        if name in names:module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)
    with torch.inference_mode():moments=model._adaptive_encode(x)
    save('moments',moments)
    report=dict(torch=torch.__version__,cudnn=torch.backends.cudnn.version(),cudnn_tf32=torch.backends.cudnn.allow_tf32,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
        image_sha256=hashlib.sha256(a.image.read_bytes()).hexdigest(),captures=captured)
    (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='captures'}),flush=True)
if __name__=='__main__':main()
