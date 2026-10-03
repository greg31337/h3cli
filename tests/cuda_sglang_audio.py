#!/usr/bin/env python3
"""Pinned installed SGLang audio decoder, same clean H3AV latent, raw PCM."""
import argparse
import hashlib
import json
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
    p.add_argument('--capture',action='store_true',help='Retain bounded early decoder operations')
    p.add_argument('--capture-convs',action='store_true',help='One real operation per convolution shape, at most 2 GiB')
    p.add_argument('--capture-stage0',action='store_true',help='First audio stage operations, at most 512 MiB')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    import torch
    from safetensors.torch import load_file
    from sglang.multimodal_gen.runtime.server_args import set_global_server_args
    from sglang.multimodal_gen.configs.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAEConfig
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3AudioVAE
    from sglang.multimodal_gen.runtime.loader.utils import adopt_plain_weight_norm_state
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.decoding import _deterministic_audio_decode_context
    set_global_server_args(SimpleNamespace(attention_backend='torch_sdpa',attention_backend_config=None,kv_gather_degree=1))
    data=a.state.read_bytes()
    if data[:8]!=b'H3AV\r\n\x1a\n' or hashlib.sha256(data[:128]+data[160:]).digest()!=data[128:160]:raise ValueError('invalid state')
    vb,ab=struct.unpack_from('<2Q',data,72)
    if ab%256 or len(data)!=160+vb+ab:raise ValueError('invalid audio payload')
    t=ab//256
    if t not in (207,603):raise ValueError('unsupported frozen audio length')
    z=np.frombuffer(data,dtype='<f4',count=ab//4,offset=160+vb).reshape(32,2,t).transpose(1,0,2).copy()
    config=json.loads((a.model/'config.json').read_text());c=MiniMaxH3AudioVAEConfig();c.update_model_arch(config)
    with torch.device('cpu'):model=MiniMaxH3AudioVAE(c)
    weights=load_file(str(a.model/'model.safetensors'),device='cpu')
    adopt_plain_weight_norm_state(model,weights)
    model.load_state_dict(weights,strict=True,assign=True);del weights
    model.eval();model.requires_grad_(False);model.dec_in_proj.to('cuda');model.decoder.to('cuda')
    captured={};capture_bytes=0;convs={};shapes=set()
    def save(name,t):
        nonlocal capture_bytes
        if name in captured:return
        t=t.detach().contiguous().cpu();raw=t.reshape(-1).view(torch.uint8).numpy().tobytes()
        capture_bytes+=len(raw)
        if capture_bytes>(2048 if a.capture_convs else 512)*1024**2:raise ValueError('audio capture exceeds explicit budget')
        (a.out/(name+'.bin')).write_bytes(raw)
        meta=dict(dtype=str(t.dtype),shape=list(t.shape),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        (a.out/(name+'.json')).write_text(json.dumps(meta)+'\n');captured[name]=meta
    if a.capture or a.capture_stage0:
        for name,module in model.named_modules():
            stage0=a.capture_stage0 and any(name.startswith('decoder.resblocks.'+str(j)+'.') for j in range(3)) and ('.convs' in name or '.activations.' in name) and name.count('.')==4
            if not stage0 and name not in ('dec_in_proj','decoder.conv_pre','decoder.ups.0.0','decoder.resblocks.0.activations.0','decoder.resblocks.0.activations.0.upsample','decoder.resblocks.0.activations.0.act','decoder.resblocks.0.activations.0.downsample','decoder.resblocks.0.convs1.0'):continue
            def pre(m,args,name=name,stage0=stage0):
                save(name+'.input',args[0])
                if hasattr(m,'weight') and not stage0:save(name+'.weight',m.weight)
                for attr in ('alpha','beta','filter'):
                    if hasattr(m,attr):save(name+'.'+attr,getattr(m,attr))
            def post(m,args,out,name=name):save(name+'.output',out)
            module.register_forward_pre_hook(pre);module.register_forward_hook(post)
    if a.capture_convs:
        for name,module in model.named_modules():
            if not name.startswith(('dec_in_proj','decoder.')) or not isinstance(module,(torch.nn.Conv1d,torch.nn.ConvTranspose1d)):continue
            def post_conv(m,args,out,name=name):
                x=args[0];key=(tuple(x.shape),tuple(out.shape),m.kernel_size,m.stride,m.padding,m.dilation,m.groups,isinstance(m,torch.nn.ConvTranspose1d))
                if key in shapes:return
                shapes.add(key)
                meta=dict(batch=x.shape[0],ci=m.in_channels,co=m.out_channels,length=x.shape[2],kernel=m.kernel_size[0],stride=m.stride[0],padding=m.padding[0],dilation=m.dilation[0],groups=m.groups,transpose=isinstance(m,torch.nn.ConvTranspose1d))
                for suffix,t in [('input',x),('weight',m.weight),('output',out)]:save(name+'.'+suffix,t)
                if m.bias is not None:save(name+'.bias',m.bias)
                meta['bias']=m.bias is not None;convs[name]=meta
            module.register_forward_hook(post_conv)
    z=torch.from_numpy(z).to('cuda')
    mean=torch.tensor(config['latents_mean'],device='cuda').reshape(1,32,1)
    std=torch.tensor(config['latents_std'],device='cuda').reshape(1,32,1)
    z=z*std+mean
    torch.cuda.synchronize();started=time.perf_counter()
    with torch.inference_mode(),_deterministic_audio_decode_context():pcm=model.decode(z)
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    pcm=pcm.clamp(-1,1).float().cpu().numpy()
    if pcm.shape!=(2,1,t*800) or not np.isfinite(pcm).all():raise ValueError('invalid decoded audio')
    raw=pcm.astype('<f4',copy=False).tobytes();(a.out/'pcm.f32').write_bytes(raw)
    report=dict(source_state=str(a.state),source_state_sha256=hashlib.sha256(data).hexdigest(),
        channels=2,samples=t*800,sample_rate=model.sample_rate,decode_seconds=elapsed,torch=torch.__version__,
        pcm_sha256=hashlib.sha256(raw).hexdigest(),domain='FP32 full audio decoder, clipped channel-major PCM')
    if a.capture or a.capture_convs or a.capture_stage0:report['capture']=captured
    if a.capture_convs:report['convolutions']=convs
    (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
