#!/usr/bin/env python3
"""Optional released Diffusers whole-transformer oracle, using original H3 weights.

Use the pinned official converter via --converter (v0.40.0). Its load-time QKV
reordering and SwiGLU half swap are essential: raw shards are not state_dicts.
The small fixture isolates the 50-block DiT from Qwen and target RNG differences.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import diffusers
from diffusers import MiniMaxH3Transformer3DModel
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3Ref2VAPrepareLayoutStep, MiniMaxH3SetTimestepsStep,
    patchify_video_latents)

ROOT=Path(__file__).resolve().parents[1]


def load_transformer(weights, converter, device):
    spec=importlib.util.spec_from_file_location('official_h3_converter',converter)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=module.MINIMAX_H3_TRANSFORMER_CONFIG
    with torch.device('meta'): model=MiniMaxH3Transformer3DModel(**config)
    expected=model.state_dict(); assigned=set()
    index=json.loads(next(weights.glob('*index.json')).read_text())['weight_map']
    digest=hashlib.sha256()
    for shard in sorted(set(index.values())):
        print('loading official transformer',shard,flush=True)
        with safe_open(weights/shard,framework='pt',device='cpu') as f:
            for key in f.keys():
                value=f.get_tensor(key)
                digest.update(key.encode()+b'\0');digest.update(value.view(torch.uint8).numpy().tobytes())
                if key.endswith('attn.qkv_proj.weight'):
                    value=module.reorder_interleaved_qkv(value,config['num_attention_heads'],config['attention_head_dim'])
                for name,tensor in module.convert_transformer_key(key,value,config):
                    assert name in expected and tensor.shape==expected[name].shape and name not in assigned,name
                    parent,_,leaf=name.rpartition('.')
                    model.get_submodule(parent).register_parameter(leaf,torch.nn.Parameter(tensor.to(device),requires_grad=False))
                    assigned.add(name)
    assert assigned==set(expected),set(expected)-assigned
    # Non-persistent rotary buffer was initialized on meta along with the model.
    model.rope=type(model.rope)(config['rope_freq_dim'],config['rope_theta']).to(device)
    model.eval()
    return model,digest.hexdigest()


def unpatchify_video_latents(rows,batch,channels,t,h,w,patch):
    pt,ph,pw=patch
    return rows.reshape(batch,t//pt,h//ph,w//pw,channels,pt,ph,pw).permute(
        0,4,1,5,2,6,3,7).reshape(batch,channels,t,h,w)


def make_inputs(encoder_fixture):
    condition=load_file(encoder_fixture)['x.normalized']
    text=torch.randn(1,6,5120,generator=torch.Generator().manual_seed(314)).bfloat16()
    video=torch.randn(1,24,7,4,4,generator=torch.Generator().manual_seed(72))
    audio=torch.randn(32,2,37,generator=torch.Generator().manual_seed(73))
    refs=[SimpleNamespace(kind='video',has_audio=False)]
    positions,tags,vi,ai,ti,nv,na=MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        torch.ones(6,dtype=torch.int64),refs,[condition],[],7,4,4,37,(1,2,2),2,2,0)
    return {'x.spec':torch.tensor([6,7,4,4,37,22],dtype=torch.int32),
        'x.references':torch.tensor([[2,*condition.shape[2:],0]],dtype=torch.int32),
        'x.text':text,'x.video':video,'x.audio':audio,
        'x.condition_video':patchify_video_latents(condition,(1,2,2)),
        'x.condition_audio':torch.empty(0,32),'x.positions':positions,'x.tags':tags,
        'x.video_indices':vi,'x.audio_indices':ai,'x.text_indices':ti,
        'x.counts':torch.tensor([nv,na],dtype=torch.int64)}


@torch.no_grad()
def evaluate(model, data, device, steps, full):
    nv,na=map(int,data['x.counts'])
    _,t,h,w,at,_=map(int,data['x.spec'])
    video=patchify_video_latents(data['x.video'],(1,2,2)).to(device)
    audio=data['x.audio'].permute(1,2,0).reshape(-1,32).to(device)
    cv=data['x.condition_video'].to(device);ca=data['x.condition_audio'].to(device)
    layout={name:data['x.'+name].to(device=device,dtype=torch.float32 if name=='positions' else None)
        for name in ('positions','tags','video_indices','audio_indices','text_indices')}
    layout['position_ids']=layout.pop('positions');layout['token_tags']=layout.pop('tags')
    text=data['x.text'].to(device)
    sv=MiniMaxH3Scheduler(shift=12);sa=MiniMaxH3Scheduler(shift=3)
    sv.set_timesteps(steps+1);sa.set_timesteps(steps+1)
    data['x.sigmas_video']=sv.sigmas.clone();data['x.sigmas_audio']=sa.sigmas.clone()
    for step in range(steps if full else 1):
        start=time.monotonic()
        ts,indices=MiniMaxH3SetTimestepsStep.build_row_timesteps(
            data['x.video_indices'],data['x.audio_indices'],nv,na,text.shape[1],
            float(sv.timesteps[step]),float(sa.timesteps[step]),max(float(sv.timesteps[step]),.999),1.)
        pv,pa=model(hidden_states=torch.cat([cv,video])[None],
            audio_hidden_states=torch.cat([ca,audio])[None],encoder_hidden_states=text,
            timestep=ts.to(device),timestep_indices=indices.to(device),return_dict=False,**layout)
        pv=pv[0,nv:].float();pa=pa[0,na:].float()
        if step==0:
            data['x.video_velocity']=unpatchify_video_latents(pv,1,24,t,h,w,(1,2,2)).cpu()
            data['x.audio_velocity']=pa.reshape(2,at,32).permute(2,0,1).cpu()
        if full:
            video=sv.step(pv,sv.timesteps[step],video,return_dict=False)[0]
            audio=sa.step(pa,sa.timesteps[step],audio,return_dict=False)[0]
        if device=='mps':torch.mps.synchronize()
        print(f'official DiT step {step+1}/{steps}: {time.monotonic()-start:.2f}s',flush=True)
    if full:
        data['x.video_final']=unpatchify_video_latents(video,1,24,t,h,w,(1,2,2)).cpu()
        data['x.audio_final']=audio.reshape(2,at,32).permute(2,0,1).cpu()
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,default=ROOT/'models/MiniMax-H3/Ref2VA/transformer')
    p.add_argument('--converter',type=Path,required=True)
    p.add_argument('--encoder-fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','mps'],default='mps')
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--full',action='store_true')
    args=p.parse_args();torch.set_num_threads(8)
    data=make_inputs(args.encoder_fixture)
    model,identity=load_transformer(args.weights,args.converter,args.device)
    start=time.monotonic();data=evaluate(model,data,args.device,args.steps,args.full)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save_file({k:v.contiguous().clone() for k,v in data.items()},args.output)
    manifest={'torch':torch.__version__,'diffusers':diffusers.__version__,'device':args.device,
        'steps':args.steps,'full':args.full,'seconds':time.monotonic()-start,
        'weights_sha256':identity,'converter_sha256':hashlib.sha256(args.converter.read_bytes()).hexdigest(),
        'transformer_source_sha256':hashlib.sha256(Path(inspect.getfile(type(model))).read_bytes()).hexdigest(),
        'fixture_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
