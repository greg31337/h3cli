#!/usr/bin/env python3
"""Official Diffusers MiniMax-H3 F32/MPS decoder, released 256/64 tiling.

Uses the unmodified upstream conversion function for the locally released weights.
The unused encoder stays on meta. No native decoder outputs enter this reference.
"""
import argparse, hashlib, importlib.util, inspect, json, resource, time
from pathlib import Path
import numpy as np
import torch
import diffusers
from diffusers import AutoencoderKLMiniMaxH3
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('latent',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--height',type=int,required=True);p.add_argument('--width',type=int,required=True)
    p.add_argument('--time',type=int,default=7)
    p.add_argument('--converter',type=Path,default=ROOT/'outputs/refvideo-integration-validation/convert_minimax_h3_to_diffusers.py')
    a=p.parse_args();torch.set_num_threads(8)
    spec=importlib.util.spec_from_file_location('converter',a.converter)
    converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)
    directory=ROOT/'models/MiniMax-H3/FL2VA/video_vae';config=json.loads((directory/'config.json').read_text())
    start=time.monotonic()
    with torch.device('meta'):
        model=AutoencoderKLMiniMaxH3(latents_mean=config['latents_mean'],latents_std=config['latents_std'])
    state={};weight_hash=hashlib.sha256()
    with safe_open(directory/'source/model.safetensors',framework='pt') as f:
        for key in f.keys():
            if key.startswith(('decoder.','post_quant_conv.')):
                v=f.get_tensor(key);weight_hash.update(key.encode()+b'\0');weight_hash.update(v.numpy().tobytes())
                state.update(converter.convert_video_vae_key(key,v,converter.MINIMAX_H3_VIDEO_VAE_CONFIG))
    for name in ('decoder','post_quant_conv'):
        getattr(model,name).load_state_dict({k[len(name)+1:]:v for k,v in state.items() if k.startswith(name+'.')},strict=True,assign=True)
    del state
    model.decoder.rope=type(model.decoder.rope)(int(model.config.decoder_attention_head_dim*model.config.decoder_rope_dim_ratio),model.config.decoder_rope_theta)
    model.decoder.to('mps');model.post_quant_conv.to('mps');model.enable_tiling();model.eval()
    # These are the official class defaults; assert instead of assuming parity.
    assert model.tile_sample_min_height==256 and model.tile_sample_min_width==256
    assert model.tile_sample_min_overlap_height==64 and model.tile_sample_min_overlap_width==64
    z=np.fromfile(a.latent,dtype='<f4').reshape(1,24,a.time,a.height//16,a.width//16)
    mean=torch.tensor(config['latents_mean']).reshape(1,24,1,1,1)
    std=torch.tensor(config['latents_std']).reshape(1,24,1,1,1)
    z=(torch.from_numpy(z)*std+mean).to('mps');torch.mps.synchronize();load=time.monotonic()-start
    start=time.monotonic()
    with torch.no_grad():
        decoded=model.decode(z,return_dict=False)[0]
        rgb=(decoded.float().cpu()*torch.tensor([.229,.224,.225]).reshape(1,3,1,1,1)+torch.tensor([.485,.456,.406]).reshape(1,3,1,1,1)).clamp(0,1)
    torch.mps.synchronize();elapsed=time.monotonic()-start
    values=rgb[0].permute(1,2,3,0).contiguous().numpy();assert np.isfinite(values).all();values.tofile(a.output)
    source=Path(inspect.getfile(AutoencoderKLMiniMaxH3))
    record={'shape':list(values.shape),'load_seconds':load,'decode_seconds':elapsed,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'mps_current_allocated_bytes':torch.mps.current_allocated_memory(),
        'mps_driver_allocated_bytes':torch.mps.driver_allocated_memory(),
        'memory_note':'MPS samples at completion are not peak measurements',
        'torch':torch.__version__,'diffusers':diffusers.__version__,'device':'M4 Max MPS/F32',
        'class_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'converter_sha256':hashlib.sha256(a.converter.read_bytes()).hexdigest(),
        'decoder_weights_sha256':weight_hash.hexdigest(),
        'latent_sha256':hashlib.sha256(a.latent.read_bytes()).hexdigest()}
    a.output.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
if __name__=='__main__':main()
