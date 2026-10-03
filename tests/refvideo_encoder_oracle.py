#!/usr/bin/env python3
"""Generate reproducible VAE fixtures using the installed official Diffusers class.

Only encoder/quant weights are materialized; the unused decoder stays on meta.
All oracle operations run in F32 on CPU. Outputs include source/package hashes.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import re
import time

import diffusers
from diffusers import AutoencoderKLMiniMaxH3
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from types import SimpleNamespace
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    patchify_video_latents, MiniMaxH3Ref2VAPrepareLayoutStep)

ROOT = Path(__file__).resolve().parents[1]


def load_model(root):
    config = json.loads((root / 'config.json').read_text())
    with torch.device('meta'):
        model = AutoencoderKLMiniMaxH3(latents_mean=config['latents_mean'],
                                      latents_std=config['latents_std'])
    state = {}
    digest = hashlib.sha256()
    with safe_open(root / 'source/model.safetensors', framework='pt', device='cpu') as f:
        for key in f.keys():
            if not key.startswith(('encoder.', 'quant_conv.')):
                continue
            name = re.sub(r'encoder.down.(\d+).block.(\d+).', r'encoder.down_blocks.\1.resnets.\2.', key)
            name = re.sub(r'encoder.down.(\d+).downsample.', r'encoder.down_blocks.\1.downsamplers.0.', name)
            name = name.replace('.nin_shortcut.', '.conv_shortcut.')
            state[name] = f.get_tensor(key).float()
            digest.update(key.encode() + b'\0')
            digest.update(state[name].numpy().tobytes())
    for prefix in ('encoder', 'quant_conv'):
        getattr(model, prefix).load_state_dict(
            {k[len(prefix)+1:]: v for k, v in state.items() if k.startswith(prefix + '.')},
            strict=True, assign=True)
    model.eval()
    model.enable_tiling()
    model._oracle_encoder_sha256 = digest.hexdigest()
    return model


def pixels(frames, height, width):
    # Integer pattern avoids dependence on a fixture-input RNG or transcoder.
    c, t, y, x = np.indices((3, frames, height, width), dtype=np.int32)
    rgb = ((c*67 + t*11 + x*3 + y*5 + (x*y)%37) % 256).astype(np.float32)
    # Match the native RGB boundary's F32 multiply by 1/255 exactly.
    return torch.from_numpy((rgb * np.float32(1/255))[None])


def normalize(rgb):
    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1, 1)
    return (rgb - mean) / std


def save(out, name, tensors):
    path = out / (name + '.safetensors')
    save_file({k: v.contiguous() for k, v in tensors.items()}, path)
    print(f'saved {name}: ' + ', '.join(f'{k}={list(v.shape)}' for k, v in tensors.items()), flush=True)


@torch.no_grad()
def raw_fixtures(model, out):
    for name, frames, h, w in [('raw17', 17, 64, 64), ('tiled', 1, 288, 304),
                              ('legacy-image', 1, 64, 64), ('legacy-video', 56, 64, 64)]:
        rgb = pixels(frames, h, w)
        moments = model._encode_clip(normalize(rgb))
        mean = torch.tensor(model.config.latents_mean).view(1, 24, 1, 1, 1)
        std = torch.tensor(model.config.latents_std).view(1, 24, 1, 1, 1)
        save(out, name, {'x.pixels': rgb, 'x.moments': moments,
                         'x.legacy': (moments[:, :24] - mean) / std})

    # Arbitrary moments make both axes, corner blend order and logvar channels
    # observable without relying on a CNN's naturally correlated tile outputs.
    h, w, time = 512, 704, 2
    ys, yl, yo = model._split_tiles(h, 256, 64)
    xs, xl, xo = model._split_tiles(w, 256, 64)
    tiles = torch.randn(len(ys)*len(xs), 1, 48, time, 16, 16,
                        generator=torch.Generator().manual_seed(123))
    rows = [[tiles[i*len(xs)+j] for j in range(len(xs))] for i in range(len(ys))]
    stitched = model._stitch_tiles(rows, [n//16 for n in yo], [n//16 for n in xo])
    save(out, 'stitch', {'x.tiles': tiles, 'x.moments': stitched,
        'x.canvas': torch.tensor([h, w], dtype=torch.int32),
        'x.y_starts': torch.tensor(ys, dtype=torch.int32), 'x.y_overlaps': torch.tensor(yo, dtype=torch.int32),
        'x.x_starts': torch.tensor(xs, dtype=torch.int32), 'x.x_overlaps': torch.tensor(xo, dtype=torch.int32)})


@torch.no_grad()
def math_fixtures(root, out):
    config = json.loads((root / 'config.json').read_text())
    tensors = {}
    for n in [1, 2, 7, 15, 16, 17, 24, 96, 216, 1920, 5952, 131071]:
        tensors[f'epsilon.{n}'] = torch.randn(n, generator=torch.Generator().manual_seed(42))
    moments = torch.randn(1, 48, 2, 3, 3, generator=torch.Generator().manual_seed(314))
    moments[:, 24:] *= 20  # Cross both logvar clamp boundaries.
    posterior = DiagonalGaussianDistribution(moments)
    epsilon = torch.randn(posterior.mean.shape, generator=torch.Generator().manual_seed(42))
    sample = posterior.sample(generator=torch.Generator().manual_seed(42))
    mean = torch.tensor(config['latents_mean']).view(1, 24, 1, 1, 1)
    std = torch.tensor(config['latents_std']).view(1, 24, 1, 1, 1)
    rounded = sample.half().float()
    tensors.update({'x.moments': moments, 'x.epsilon': epsilon, 'x.sample': sample,
        'x.rounded': rounded, 'x.normalized': (rounded-mean)/std,
        'x.mean': mean, 'x.std': std, 'x.logvar': posterior.logvar, 'x.sigma': posterior.std})
    # Every finite positive FP16 value, all ties between adjacent values and
    # their immediate F32 neighbours, mirrored negative. This includes underflow.
    half = torch.arange(0x7c00, dtype=torch.int32).to(torch.uint16).view(torch.float16).float()
    midpoint = (half[:-1] + half[1:]) * .5
    values = torch.cat([half, midpoint, torch.nextafter(midpoint, torch.full_like(midpoint, float('inf'))),
                        torch.nextafter(midpoint, torch.full_like(midpoint, -float('inf'))),
                        torch.tensor([65519., 65520., 65521., float('inf')])])
    values = torch.cat([values, -values])
    tensors['half.input'] = values
    tensors['half.output'] = values.half().float()
    save(out, 'math', tensors)


@torch.no_grad()
def temporal_fixtures(model, out):
    mean = torch.tensor(model.config.latents_mean).view(1, 24, 1, 1, 1)
    std = torch.tensor(model.config.latents_std).view(1, 24, 1, 1, 1)
    cases = [(n,n,64,64,f'temporal{n}') for n in (39,56,73,107,124)]
    cases += [(n, (n-5)//17*17+5, 64, 64, f'temporal{n}') for n in (48,60,80,110)]
    cases += [(39,39,288,320,'temporal39-tiled')]
    for frames, selected, height, width, name in cases:
        rgb = pixels(frames, height, width)
        # Capture the actual calls made by the released wrapper, not a reimplementation.
        chunks = []
        encode_clip = model._encode_clip
        def capture(clip):
            output = encode_clip(clip)
            chunks.append(output.clone())
            return output
        model._encode_clip = capture
        try:
            posterior = model.encode(normalize(rgb[:, :, :selected]), return_dict=False)[0]
        finally:
            model._encode_clip = encode_clip
        epsilon = torch.randn(posterior.mean.shape, generator=torch.Generator().manual_seed(42))
        sample = posterior.sample(generator=torch.Generator().manual_seed(42))
        tensors = {'x.pixels': rgb, 'x.moments': posterior.parameters, 'x.epsilon': epsilon,
                   'x.sample': sample, 'x.rounded': sample.half().float(),
                   'x.normalized': (sample.half().float()-mean)/std,
                   'x.vae_frames': torch.tensor([selected], dtype=torch.int32),
                   'x.chunks': torch.stack(chunks),
                   'x.concatenated': torch.cat(chunks, dim=2),
                   'x.rows': patchify_video_latents((sample.half().float()-mean)/std, (1,2,2))}
        assert torch.equal(tensors['x.concatenated'][:, :, :-3], posterior.parameters)
        if frames == 56:
            tensors['x.continuous'] = model._encode_clip(normalize(rgb))
        save(out, name, tensors)


def layout_fixtures(out):
    # kind, T, H, W, soundtrack T. Deliberately mix aspect ratios and audio spans.
    image = (0,1,4,6,0)
    video = (2,17,4,4,0)
    audio = (1,0,0,0,101)
    cases = {'released56':[video],
        'image-video':[image,video], 'video-audio':[video,audio],
        'two-videos':[video,(2,22,6,4,120)],
        'three-videos':[video,(2,22,6,4,120),(2,37,4,8,240)],
        'audio-video-image':[audio,(2,17,4,4,120),image]}
    for name, geometry in cases.items():
        refs = [SimpleNamespace(kind=['image','audio','video'][k],has_audio=bool(a))
                for k,t,h,w,a in geometry]
        visual = [torch.zeros(1,24,t,h,w) for k,t,h,w,a in geometry if k != 1]
        sound = [torch.zeros(a*2,32) for k,t,h,w,a in geometry if a]
        tags = torch.tensor([1,1,0,0,1,1],dtype=torch.int64)
        values = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
            tags,refs,visual,sound,7,4,6,37,(1,2,2),2,2,0)
        positions, tags, vi, ai, ti, nv, na = values
        save(out, 'layout-'+name, {'x.references':torch.tensor(geometry,dtype=torch.int32),
            'x.spec':torch.tensor([6,7,4,6,37,22],dtype=torch.int32),
            'x.positions':positions, 'x.tags':tags, 'x.video_indices':vi,
            'x.audio_indices':ai, 'x.text_indices':ti,
            'x.counts':torch.tensor([nv,na,7*2*3,37*2],dtype=torch.int64),
            'x.target_start':torch.tensor([6+nv+na],dtype=torch.int64),
            'x.cursor':positions[6+nv+na,0].reshape(1).clone()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=ROOT / 'models/MiniMax-H3/Ref2VA/video_vae')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['raw', 'math', 'temporal', 'layout', 'all'], default='all')
    p.add_argument('--compact-host-fixture', type=Path,
                   help='also save a small self-contained host fixture derived from the math oracle')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    source = Path(inspect.getfile(AutoencoderKLMiniMaxH3))
    manifest = {'torch': torch.__version__, 'diffusers': diffusers.__version__,
                'device': 'cpu', 'dtype': 'float32', 'threads': 8,
                'encoder_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'model': str(args.model.resolve()), 'stage': args.stage}
    (args.output / 'oracle-source.py').write_bytes(source.read_bytes())
    start = time.monotonic()
    if args.stage in ('math', 'all'):
        math_fixtures(args.model, args.output)
    if args.stage in ('layout', 'all'): layout_fixtures(args.output)
    if args.stage in ('raw', 'temporal', 'all'):
        model = load_model(args.model)
        manifest['encoder_weights_sha256'] = model._oracle_encoder_sha256
        if args.stage in ('raw', 'all'): raw_fixtures(model, args.output)
        if args.stage in ('temporal', 'all'): temporal_fixtures(model, args.output)
    manifest['seconds'] = time.monotonic() - start
    manifest['fixtures'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in args.output.glob('*.safetensors')}
    (args.output / (args.stage + '-manifest.json')).write_text(json.dumps(manifest, indent=2) + '\n')
    if args.compact_host_fixture:
        data = load_file(args.output/'math.safetensors')
        del data['epsilon.131071']
        length = data['half.input'].numel()
        indices = np.unique(np.concatenate([np.arange(128), np.arange(length-128,length),
                                            np.linspace(0,length-1,2048,dtype=np.int64)]))
        for key in ('half.input','half.output'):
            data[key] = data[key][torch.from_numpy(indices)].contiguous()
        dest = args.compact_host_fixture; dest.parent.mkdir(parents=True, exist_ok=True)
        save_file(data, dest)
        metadata = {k:v for k,v in manifest.items() if k not in ('fixtures','model','seconds')}
        metadata.update({'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
                         'derived_from':manifest['fixtures']['math.safetensors'],
                         'source':'https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py'})
        dest.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
