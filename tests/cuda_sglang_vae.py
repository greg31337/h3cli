#!/usr/bin/env python3
"""Decode saved clean latents with the pinned, installed SGLang full VAE.

No package edits or model hashes. Raw float RGB and first-tile boundaries
separate decoder arithmetic from denoising and video codec differences.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import time
from types import SimpleNamespace
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tile',action='store_true')
    p.add_argument('--no-capture',action='store_true',help='Disable operator hooks; retain complete raw RGB')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    import torch
    from safetensors.torch import load_file
    from sglang.multimodal_gen.runtime.server_args import set_global_server_args
    from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
    from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import MiniMaxH3VideoVAEConfig
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3VideoVAE
    set_global_server_args(SimpleNamespace(attention_backend='torch_sdpa',attention_backend_config=None,kv_gather_degree=1))
    data=a.state.read_bytes()
    if data[:8]!=b'H3AV\r\n\x1a\n' or hashlib.sha256(data[:128]+data[160:]).digest()!=data[128:160]:
        raise ValueError('invalid saved state')
    width,height,frames=struct.unpack_from('<3I',data,24)
    if (width,height) != (640,480) or frames not in (124,362):raise ValueError('unsupported fixture geometry')
    vt=37 if frames==124 else 107
    count=24*vt*30*40;z=np.frombuffer(data,dtype='<f4',count=count,offset=160).reshape(1,24,vt,30,40).copy()
    if a.tile:z=z[:,:,:7,:16,:16].copy()
    config=json.loads((a.model/'config.json').read_text())
    c=MiniMaxH3VideoVAEConfig();c.update_model_arch(config)
    # Construct on CPU so nonpersistent rotary buffers receive the installed
    # implementation's normal initialization as well as checkpoint parameters.
    with torch.device('cpu'):model=MiniMaxH3VideoVAE(c)
    weights=load_file(str(a.model/'source/model.safetensors'),device='cpu')
    model.load_state_dict(weights,strict=True,assign=True);del weights
    model.eval();model.requires_grad_(False)
    # The encoder is inactive during decode. Keep the pinned decoder class and
    # its original methods, and move only the components it actually executes.
    model.prepare_decoder_autocast_weights(torch.float16)
    model.decoder.to('cuda');model.post_quant_conv.to('cuda')
    model.decoder_tiling=not a.tile;model.parallel_tiling=False
    z=torch.from_numpy(z).to('cuda')
    mean=torch.tensor(config['latents_mean'],device='cuda').reshape(1,24,1,1,1)
    std=torch.tensor(config['latents_std'],device='cuda').reshape(1,24,1,1,1)
    z=z*std+mean
    used=0;records={};handles=[]
    def save(name,value):
        nonlocal used
        if a.no_capture:return
        if name in records:return
        if isinstance(value,(tuple,list)):value=value[0]
        if not isinstance(value,torch.Tensor):return
        tensor=value.detach().contiguous().cpu();raw=tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        used+=len(raw)
        if used>512*1024**2:raise ValueError('VAE capture exceeds 512 MiB')
        (a.out/(name+'.bin')).write_bytes(raw)
        info=dict(dtype=str(tensor.dtype),shape=list(tensor.shape),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        (a.out/(name+'.json')).write_text(json.dumps(info,indent=2)+'\n');records[name]=info
    names={'post_quant_conv','decoder.x_embedder','decoder.norm_out','decoder.proj_out'}
    names.update('decoder.transformer_blocks.0'+n for n in ('','.norm1','.norm2','.attn.to_qkv','.attn.attn','.attn.to_out','.ff','.ff.w1','.ff.w2'))
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import base_module
    old_w2=base_module._unfused_bias_linear
    def capture_w2(linear,hidden):
        save('w2.input',hidden);out=old_w2(linear,hidden);save('w2.output',out);return out
    if not a.no_capture:base_module._unfused_bias_linear=capture_w2
    for name,module in model.named_modules():
        if a.no_capture or name not in names:continue
        def before(m,args,name=name):
            save(name+'.input',args)
            if name.endswith('.attn.attn'):
                for label,tensor in zip(('query','key','value'),args):save(label,tensor)
        def after(m,args,out,name=name):save(name+'.output',out)
        handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
    torch.cuda.synchronize();started=time.perf_counter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16),set_forward_context(current_timestep=0,attn_metadata=None):
        value=model.decode_base(z)
        rgb=model.processor.revert_tensor(value)
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    shape=list(rgb.shape)
    with (a.out/'rgb.f32').open('xb') as f:
        for index in range(rgb.shape[2]):
            frame=rgb[0,:,index].permute(1,2,0).float().contiguous().cpu().numpy()
            if not np.isfinite(frame).all():raise ValueError('nonfinite RGB')
            f.write(frame.astype('<f4',copy=False).tobytes())
    delivered=time.perf_counter()-started
    report=dict(implementation='installed SGLang full video VAE',torch=torch.__version__,tile=a.tile,
                source_state=str(a.state),source_state_sha256=hashlib.sha256(data).hexdigest(),shape=shape,
                decode_seconds=elapsed,decode_and_raw_delivery_seconds=delivered,
                instrumented=not a.no_capture,records=records,
                timing_scope='raw video diagnostic; includes CPU transfer and raw file write in delivery; excludes audio/codec/model load')
    (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__':main()
