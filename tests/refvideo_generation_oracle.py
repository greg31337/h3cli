#!/usr/bin/env python3
"""Real-media oracle across the new video VAE, packed layout, full DiT and decode.

Qwen embeddings, request noise and existing augmentation draws are held fixed
at the captured native boundary. They are outside this video-encoder change;
this is deliberately not a test of PyTorch versus native request RNG streams.
The source RGB is decoded afresh and the released video VAE/layout are rerun
independently. No native video conditions or positions feed the oracle model.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
from types import SimpleNamespace
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file,save_file
from diffusers import AutoencoderKLMiniMaxH3
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3Ref2VAPrepareLayoutStep,patchify_video_latents
from refvideo_encoder_oracle import load_model,normalize
from refvideo_dit_oracle import load_transformer,evaluate
from test_refvideo_dit import compare
from test_sampler_file import entries
ROOT=Path(__file__).resolve().parents[1]


def request_identity(parts):
    # Section 1's stable v1 scalar prefix; see docs/sampler-state.md.
    names=('width height frames steps reference_image_size denoise_reuse dit_layers core_reuse '
        'token_reduction use_int8_row_fc2 use_reference_rope ssd_streaming render_width render_height '
        'use_slower_bf16_mlp use_slower_bf16_qkv use_slower_bf16_attention_output '
        'use_slower_row_major_attention_output use_slower_unfused_int8_inputs use_slower_unfused_qkv_rope '
        'use_slower_scalar_qkv_rms use_slower_uncached_int8_scales use_slower_dynamic_fc1_k '
        'use_slower_grouped_quantizer continuation_context_frames keep_continuation_prefix '
        'continuation_mode bridge_video_steps bridge_profile').split()
    assert struct.unpack_from('<I',parts[1])[0]==1
    values=dict(zip(names,struct.unpack_from('<'+str(len(names))+'i',parts[1],16)))
    values['seed']=struct.unpack_from('<Q',parts[1],16+len(names)*4)[0]
    return values


def snapshot(path):
    p={x[0]:bytes(x[-1]) for x in entries(path.read_bytes())}
    seq,nseg,nv,ntv,na,nta=struct.unpack_from('<6Q',p[10])
    text,t,h,w,at=struct.unpack_from('<5i',p[10],48)
    refs=np.frombuffer(p[9],'<i4').copy().reshape(-1,5)
    positions=np.frombuffer(p[10],'<f8',offset=76+nseg*20).copy().reshape(seq,3)
    tags=np.ones(text,np.int64) if len(p[6])==8 else np.frombuffer(p[6],np.uint8,offset=8).astype(np.int64)
    data={'x.spec':torch.tensor([text,t,h,w,at,(t-2)//5*17+5],dtype=torch.int32),
        'x.references':torch.from_numpy(refs),'x.text':torch.from_numpy(np.frombuffer(p[5],'<u2').copy()).view(torch.bfloat16).reshape(1,text,5120),
        'x.video':torch.from_numpy(np.frombuffer(p[12],'<f4').copy()).reshape(1,24,t,h,w),
        'x.audio':torch.from_numpy(np.frombuffer(p[13],'<f4').copy()).reshape(32,2,at),
        'x.condition_video':torch.from_numpy(np.frombuffer(p[7],'<f4').copy()).reshape(nv,96),
        'x.condition_audio':torch.from_numpy(np.frombuffer(p[8],'<f4').copy()).reshape(na,32)}
    return p,data,torch.from_numpy(tags),torch.from_numpy(positions)


@torch.no_grad()
def prepare(args):
    p,data,tags,native_positions=snapshot(args.checkpoint)
    assert p[32]==struct.pack('<I',2) and struct.unpack_from('<i',p[1],8)[0]==0
    request=request_identity(p)
    assert request['seed']==args.seed and request['steps']==args.steps
    assert request['dit_layers']==50 and request['denoise_reuse']==request['core_reuse']==1
    assert not request['token_reduction'] and request['use_reference_rope']==1
    assert all(request[n]==1 for n in ('use_slower_bf16_mlp','use_slower_bf16_qkv','use_slower_bf16_attention_output'))
    refs=data['x.references'];assert refs.shape==(1,5) and int(refs[0,0])==2 and int(refs[0,4])==0,'single silent video fixture required'
    _,rt,rh,rw,_=map(int,refs[0]);h=rh*16;w=rw*16
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(args.reference),'-vf',f'fps=24,scale={w}:{h}:flags=lanczos,setsar=1',
        '-frames:v',str(int(data['x.spec'][-1])),'-f','rawvideo','-pix_fmt','rgb24','-'])
    rgb=np.frombuffer(raw,np.uint8).reshape(-1,h,w,3).transpose(3,0,1,2).copy().astype(np.float32)*np.float32(1/255)
    n=rgb.shape[1];selected=(n-5)//17*17+5;assert (selected-5)//17*5+2==rt
    model=load_model(ROOT/'models/MiniMax-H3/Ref2VA/video_vae')
    posterior=model.encode(normalize(torch.from_numpy(rgb[None,:,:selected])),return_dict=False)[0]
    sample=posterior.sample(generator=torch.Generator().manual_seed(42)).half().float()
    mean=torch.tensor(model.config.latents_mean).reshape(1,24,1,1,1);std=torch.tensor(model.config.latents_std).reshape(1,24,1,1,1)
    latent=(sample-mean)/std;rows=patchify_video_latents(latent,(1,2,2))
    noise_path=args.output/'augmentation.f32'
    subprocess.run([str(ROOT/'bin/refvideo_layout_tests'),'--noise',str(rows.numel()),str(args.seed),str(noise_path)],check=True)
    noise=torch.from_numpy(np.fromfile(noise_path,np.float32).reshape(rows.shape))
    augmented=.999*rows+.001*noise
    stats=compare(augmented.numpy(),data['x.condition_video'].numpy())
    assert stats['max_abs']<=3e-3 and stats['relative_l2']<=3e-4,stats
    _,t,th,tw,at,_=map(int,data['x.spec'])
    positions,tags,vi,ai,ti,nv,na=MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        tags,[SimpleNamespace(kind='video',has_audio=False)],[latent],[],t,th,tw,at,(1,2,2),2,2,0)
    layout_error=float((positions-native_positions).abs().max());assert layout_error<=1e-12
    data.update({'x.condition_video':augmented,'x.positions':positions,'x.tags':tags,
        'x.video_indices':vi,'x.audio_indices':ai,'x.text_indices':ti,'x.counts':torch.tensor([nv,na],dtype=torch.int64)})
    save_file({k:v.contiguous().clone() for k,v in data.items()},args.output/'input.safetensors')
    record={'reference':str(args.reference.resolve()),'reference_sha256':hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'normalized_frames':n,'vae_frames':selected,'latent_t':rt,'condition_errors':stats,'layout_max_abs':layout_error,
        'request':request,'prompt':p[4][8:8+struct.unpack_from('<Q',p[4])[0]].decode(),'request_seed':args.seed,
        'fixed_boundaries':['Qwen embeddings','target noise','existing PCG augmentation'],
        'encoder_weights_sha256':model._oracle_encoder_sha256}
    (args.output/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    print('PASS independent real-video encoder, augmentation span and official layout',stats,flush=True)


@torch.no_grad()
def denoise(args):
    data=load_file(args.output/'input.safetensors')
    preparation=json.loads((args.output/'preparation.json').read_text())
    assert preparation['request']['steps']==args.steps
    model,identity=load_transformer(ROOT/'models/MiniMax-H3/Ref2VA/transformer',args.converter,args.device)
    data=evaluate(model,data,args.device,args.steps,True)
    save_file({k:v.contiguous().clone() for k,v in data.items()},args.output/'generation.safetensors')
    record={'weights_sha256':identity,'steps':args.steps,'device':args.device,'full':True,
        'converter_sha256':hashlib.sha256(args.converter.read_bytes()).hexdigest()}
    (args.output/'generation.json').write_text(json.dumps(record,indent=2)+'\n')


@torch.no_grad()
def decode(args):
    data=load_file(args.output/'generation.safetensors')
    spec=importlib.util.spec_from_file_location('converter',args.converter);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=json.loads((ROOT/'models/MiniMax-H3/Ref2VA/video_vae/config.json').read_text())
    with torch.device('meta'):
        model=AutoencoderKLMiniMaxH3(latents_mean=config['latents_mean'],latents_std=config['latents_std'])
    state={}
    with safe_open(ROOT/'models/MiniMax-H3/Ref2VA/video_vae/source/model.safetensors',framework='pt') as f:
        for key in f.keys():
            if key.startswith(('decoder.','post_quant_conv.')):
                state.update(module.convert_video_vae_key(key,f.get_tensor(key),module.MINIMAX_H3_VIDEO_VAE_CONFIG))
    for name in ('decoder','post_quant_conv'):
        getattr(model,name).load_state_dict({k[len(name)+1:]:v for k,v in state.items() if k.startswith(name+'.')},strict=True,assign=True)
    del state
    model.decoder.rope=type(model.decoder.rope)(int(model.config.decoder_attention_head_dim*model.config.decoder_rope_dim_ratio),model.config.decoder_rope_theta)
    model.decoder.to(args.device);model.post_quant_conv.to(args.device)
    # F32 is the official non-CUDA decode path; only decoder modules move.
    model.enable_tiling();model.eval()
    mean=torch.tensor(config['latents_mean']).reshape(1,24,1,1,1);std=torch.tensor(config['latents_std']).reshape(1,24,1,1,1)
    rgb=(model.decode((data['x.video_final']*std+mean).to(args.device),return_dict=False)[0].float().cpu()*torch.tensor([.229,.224,.225]).reshape(1,3,1,1,1)+torch.tensor([.485,.456,.406]).reshape(1,3,1,1,1)).clamp(0,1)
    pixels=(rgb[0].permute(1,2,3,0).numpy()*255).round().astype(np.uint8)
    pixels.tofile(args.output/'oracle.rgb24');del model,rgb;gc.collect()
    config=module.get_audio_vae_config(str(ROOT/'models/MiniMax-H3/Ref2VA'))
    model=AutoencoderKLMiniMaxH3Audio(**config)
    model.load_state_dict(load_file(ROOT/'models/MiniMax-H3/Ref2VA/audio_vae/model.safetensors'),strict=True);model.eval()
    mean=torch.tensor(config['latents_mean']).reshape(1,32,1);std=torch.tensor(config['latents_std']).reshape(1,32,1)
    pcm=model.decode(data['x.audio_final'].permute(1,0,2)*std+mean,return_dict=False)[0][:,0].T.contiguous().numpy()
    pcm.astype(np.float32).tofile(args.output/'oracle.f32le')
    frames,h,w,_=pixels.shape
    subprocess.run(['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}',
        '-framerate','24','-i',str(args.output/'oracle.rgb24'),'-f','f32le','-ar','32000','-ac','2','-i',str(args.output/'oracle.f32le'),
        '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-shortest',str(args.output/'oracle.mp4')],check=True)
    print('decoded official generation',frames,h,w,pcm.shape,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['prepare','denoise','decode'],required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--reference',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--converter',type=Path)
    p.add_argument('--device',choices=['cpu','mps'],default='mps');p.add_argument('--steps',type=int,default=20);p.add_argument('--seed',type=int,default=72)
    args=p.parse_args()
    if args.stage=='prepare' and (args.checkpoint is None or args.reference is None):
        p.error('prepare requires --checkpoint and --reference')
    if args.stage!='prepare' and args.converter is None:
        p.error('denoise and decode require --converter')
    args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(8)
    globals()[args.stage](args)
if __name__=='__main__':main()
