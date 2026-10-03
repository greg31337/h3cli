#!/usr/bin/env python3
"""Official Transformers Qwen3-VL decoder, streamed one released layer at a time.

Runs the first 50 layers, without final RMSNorm, as MiniMax H3 conditioning does.
The primary backend is the official SDPA interface. Optional fp32-eager calls
upstream eager attention with F32 Q/K/V and casts its output back to BF16,
exposing the design's explicit FP32(QK)*scale contract for diagnosis.
"""
import argparse,gc,hashlib,inspect,json,time
from pathlib import Path
import numpy as np
import torch,transformers
from safetensors import safe_open
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from memory_generation import digest
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/scalingfix-validation'
def bfread(path,shape):return torch.from_numpy(np.fromfile(path,'<u2')).view(torch.bfloat16).reshape(shape)
def dump(path,x):x.detach().cpu().contiguous().view(torch.uint16).numpy().tofile(path)
def fp32_eager(module,q,k,v,mask,**kw):
    out,weights=qwen.eager_attention_forward(module,q.float(),k.float(),v.float(),mask.float() if mask is not None else None,**kw)
    return out.to(q.dtype),None
@torch.no_grad()
def run(case,backend,device):
    f=OUT/'presentations'/case;dest=OUT/'encoder'/case/('official-'+backend);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json').exists():return
    weights=ROOT/'models/MiniMax-H3'/('Ref2VA' if case in ('image','video') else 'FL2VA')/'text_encoder';index=json.loads((weights/'model.safetensors.index.json').read_text())['weight_map']
    config=Qwen3VLConfig.from_pretrained(weights).text_config;config._attn_implementation=backend
    if backend=='fp32-eager':ALL_ATTENTION_FUNCTIONS.register(backend,fp32_eager)
    def load(name):
        with safe_open(weights/index[name],framework='pt') as sf:return sf.get_tensor(name)
    n,ns,has_pos,has_tags=map(int,np.fromfile(f/'spec.u64','<u8'))
    ids=torch.from_numpy(np.fromfile(f/'ids.u32','<u4').astype(np.int64))
    emb=load('model.language_model.embed_tokens.weight');hidden=emb[ids].unsqueeze(0).contiguous().to(device);del emb
    deep=[torch.zeros_like(hidden) for _ in range(3)]
    for i in range(ns):
        start,count=map(int,np.fromfile(f/f'span-{i}.u64','<u8'))
        hidden[:,start:start+count]=bfread(f/f'vision-{i}.bf16',(count,5120)).to(device)
        for j in range(3):deep[j][:,start:start+count]=bfread(f/f'deepstack-{i}-{j}.bf16',(count,5120)).to(device)
    dump(dest/'layer-00.bf16',hidden)
    positions=torch.from_numpy(np.fromfile(f/'positions.u32','<u4').astype(np.int64).reshape(3,1,n)) if has_pos else torch.arange(n).reshape(1,1,n).expand(3,1,n)
    # Official mRoPE tables are computed on CPU, then transferred unchanged.
    rope=qwen.Qwen3VLTextRotaryEmbedding(config)
    cos,sin=rope(torch.empty(1,n,5120,dtype=torch.bfloat16),positions)
    position_embeddings=(cos.to(device),sin.to(device))
    mask=torch.full((n,n),float('-inf'),device=device).triu(1).reshape(1,1,n,n).to(torch.bfloat16)
    start=time.monotonic();times=[];weight_hash=hashlib.sha256()
    for layer_index in range(50):
        t=time.monotonic()
        with torch.device('meta'):layer=qwen.Qwen3VLTextDecoderLayer(config,layer_index)
        prefix=f'model.language_model.layers.{layer_index}.'
        state={k[len(prefix):]:load(k) for k in index if k.startswith(prefix)}
        for name in sorted(state):weight_hash.update(name.encode());weight_hash.update(state[name].contiguous().view(torch.uint16).numpy().tobytes())
        layer.load_state_dict(state,assign=True,strict=True);del state
        layer.to(device);layer.eval()
        # Save the official first-layer attention's actual post-RoPE inputs.
        if layer_index==0:
            original=ALL_ATTENTION_FUNCTIONS.get_interface(backend,qwen.eager_attention_forward)
            def capture(module,q,k,v,mask,**kw):
                for name,x in [('query',q),('key',k),('value',v)]:dump(dest/f'first-{name}.bf16',x.transpose(1,2))
                out,w=original(module,q,k,v,mask,**kw);dump(dest/'first-attention.bf16',out);return out,w
            ALL_ATTENTION_FUNCTIONS.register('scaling-capture',capture);layer.self_attn.config._attn_implementation='scaling-capture'
        hidden=layer(hidden,position_embeddings=position_embeddings,attention_mask=mask,use_cache=False)
        config._attn_implementation=backend
        if layer_index<3 and ns:hidden=hidden+deep[layer_index]
        if layer_index<5 or layer_index in (9,24,39,49):dump(dest/f'layer-{layer_index+1:02d}.bf16',hidden)
        assert torch.isfinite(hidden).all().item()
        if device=='mps':torch.mps.synchronize()
        times.append(time.monotonic()-t);del layer;gc.collect()
        print(case,backend,layer_index+1,round(times[-1],3),flush=True)
    result={'seconds':time.monotonic()-start,'layer_seconds':times,'torch':torch.__version__,'transformers':transformers.__version__,'backend':backend,'device':device,'tokens':n,'layers':50,
        'weights_sha256':weight_hash.hexdigest(),'source_sha256':digest(Path(inspect.getfile(qwen))),'input_hashes':{p.name:digest(p) for p in f.iterdir() if p.suffix in ('.u64','.u32','.u8','.bf16')}}
    (dest/'result.json').write_text(json.dumps(result,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--only',default='plain,dialogue,image,video');p.add_argument('--backend',choices=['sdpa','fp32-eager'],default='sdpa');p.add_argument('--device',choices=['cpu','mps'],default='mps');a=p.parse_args();torch.set_num_threads(8)
    for case in a.only.split(','):run(case,a.backend,a.device)
if __name__=='__main__':main()
