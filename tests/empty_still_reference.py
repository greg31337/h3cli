#!/usr/bin/env python3
"""Independent F32 stock VAE equations plus pinned ComfyUI spatial assembly.

Model equations follow MiniMax H3; no native decoder functions are called.
ComfyUI methods are loaded from the hash-verified upstream file retained by the
qualification run. That source remains under its upstream GPL-3.0 license.
"""
import argparse
import ast
import hashlib
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CONTRACT = Path(__file__).with_name('empty_still_contract.json')

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def load_latent(path):
    with open(path, 'rb') as file:
        length = struct.unpack('<Q', file.read(8))[0]
        header = json.loads(file.read(length))
        tensor = header['latent']
        file.seek(8 + length + tensor['data_offsets'][0])
        z = np.frombuffer(file.read(tensor['data_offsets'][1] - tensor['data_offsets'][0]), '<f4')
    return z.reshape(tensor['shape']).copy(), header['__metadata__']

class Weights:
    def __init__(self, directory, device):
        self.path = directory / 'model.safetensors'
        self.config = json.loads((directory.parent / 'config.json').read_text())
        self.device = device
        with self.path.open('rb') as file:
            self.start = 8 + struct.unpack('<Q', file.read(8))[0]
            self.header = json.loads(file.read(self.start - 8))
        self.cache = {}

    def __call__(self, name):
        if name not in self.cache:
            tensor = self.header[name]
            assert tensor['dtype'] == 'F32', (name, tensor['dtype'])
            array = np.memmap(self.path, dtype='<f4', mode='r', offset=self.start + tensor['data_offsets'][0], shape=tuple(tensor['shape']))
            self.cache[name] = torch.from_numpy(np.array(array)).to(self.device)
        return self.cache[name]

def decoder(weights, z):
    """Explicit F32 RMSNorm, split-half 3D RoPE, attention and gated FFN."""
    batch, channels, temporal, height, width = z.shape
    assert batch == 1 and channels == 24 and temporal == 5
    x = z.permute(0, 2, 3, 4, 1).reshape(-1, 24)
    x = F.linear(x, weights('post_quant_conv.weight').reshape(24, 24), weights('post_quant_conv.bias'))
    x = F.linear(x, weights('decoder.x_embedder.weight'), weights('decoder.x_embedder.bias'))
    x = torch.cat((x, weights('decoder.register_tokens')[0], x.new_zeros((1, 2048))))
    axes = [2 * (torch.arange(n, device=z.device, dtype=torch.float32) + .5) / n - 1 for n in (temporal, height, width)]
    coords = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    coords = torch.cat((coords, coords.new_zeros((5, 3))))
    angle = (2 * math.pi * coords[:, :, None] * (100 ** (-torch.arange(8, device=z.device) / 8))).reshape(-1, 24)
    cosine, sine = angle.cos()[:, None], angle.sin()[:, None]
    def rms(value, scale=None):
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5)
        return value if scale is None else value * scale
    def rotate(value):
        return torch.cat((value[..., :24] * cosine - value[..., 24:48] * sine,
                          value[..., 24:48] * cosine + value[..., :24] * sine, value[..., 48:]), -1)
    for index in range(36):
        prefix = f'decoder.transformer_blocks.{index}.'
        q, k, v = F.linear(rms(x, weights(prefix + 'norm1.weight')), weights(prefix + 'attn.to_qkv.weight'), weights(prefix + 'attn.to_qkv.bias')).reshape(-1, 32, 192).chunk(3, -1)
        q, k = rotate(rms(q)), rotate(rms(k))
        attention = torch.softmax((q.transpose(0, 1) @ k.permute(1, 2, 0)) / 8, dim=-1) @ v.transpose(0, 1)
        x = x + F.linear(attention.transpose(0, 1).reshape(-1, 2048), weights(prefix + 'attn.to_out.weight'), weights(prefix + 'attn.to_out.bias')) * weights(prefix + 'scale1')
        gate, up = F.linear(rms(x, weights(prefix + 'norm2.weight')), weights(prefix + 'ff.w1.weight'), weights(prefix + 'ff.w1.bias')).chunk(2, -1)
        x = x + F.linear(F.silu(gate) * up, weights(prefix + 'ff.w2.weight'), weights(prefix + 'ff.w2.bias')) * weights(prefix + 'scale2')
    x = F.layer_norm(x, (2048,), weights('decoder.norm_out.weight'), weights('decoder.norm_out.bias'), 1e-5)
    pixels = F.linear(x, weights('decoder.proj_out.weight'), weights('decoder.proj_out.bias'))[:temporal * height * width]
    return pixels.reshape(temporal, height, width, 3, 4, 16, 16).permute(3, 0, 4, 1, 5, 2, 6).reshape(1, 3, temporal * 4, height * 16, width * 16)

def upstream_tiles(path, weights):
    expected = next(item['sha256'] for item in json.loads(CONTRACT.read_text())['files'] if item['path'].endswith('/vae.py'))
    assert sha(path) == expected, 'upstream source identity changed'
    tree = ast.parse(path.read_text())
    source = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MiniMaxH3VideoVAE')
    methods = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in ('split_tiles', 'blend', 'tiled_decode')]
    assert len(methods) == 3
    namespace = {'math': math, 'torch': torch}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
    class Tiles:
        vae_ratio = 16
        tile_size = 256
        tile_overlap_min = 64
        def _decode_tile_row(self, row, starts, lengths):
            for start, length in zip(starts, lengths):
                yield decoder(weights, row[..., start // 16:(start + length) // 16])
    for name in ('split_tiles', 'blend', 'tiled_decode'):
        setattr(Tiles, name, namespace[name])
    return Tiles()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--latent', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    if args.device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    begin = time.monotonic()
    latent, meta = load_latent(args.latent)
    weights = Weights(args.model / meta['h3_model_variant'] / 'video_vae/source', args.device)
    z = torch.from_numpy(latent).to(args.device)
    mean = torch.tensor(weights.config['latents_mean'], device=args.device).reshape(1, 24, 1, 1, 1)
    std = torch.tensor(weights.config['latents_std'], device=args.device).reshape(1, 24, 1, 1, 1)
    raw = (z * std + mean).repeat(1, 1, 5, 1, 1)
    result = upstream_tiles(args.source, weights).tiled_decode(raw, slice(3, 4))
    result = result * result.new_tensor([.229, .224, .225]).reshape(1, 3, 1, 1, 1) + result.new_tensor([.485, .456, .406]).reshape(1, 3, 1, 1, 1)
    reference = result.clamp(0, 1)[0, :, 0].permute(1, 2, 0).cpu().numpy()
    candidate = np.fromfile(args.candidate, '<f4').reshape(reference.shape)
    assert np.isfinite(reference).all() and np.isfinite(candidate).all()
    delta = candidate.astype(np.float64) - reference
    mae, mse = float(np.abs(delta).mean()), float(np.square(delta).mean())
    psnr = float(-10 * np.log10(max(mse, 1e-30)))
    gates = json.loads(CONTRACT.read_text())['gates']
    passed = mae <= gates['rgb_mae_max'] and psnr >= gates['rgb_psnr_min_db']
    report = dict(passed=passed, mae=mae, psnr_db=psnr, max_error=float(np.abs(delta).max()), seconds=time.monotonic()-begin,
                  device=args.device, torch=torch.__version__, shape=list(reference.shape), source_sha256=sha(args.source),
                  latent_sha256=sha(args.latent), candidate_sha256=sha(args.candidate), script_sha256=sha(__file__), contract_sha256=sha(CONTRACT))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out.with_suffix('.npy'), reference)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if passed else 1)

if __name__ == '__main__':
    main()
