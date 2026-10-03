#!/usr/bin/env python3
"""Produce opt-in LBH operator fixtures; Torch is never a runtime dependency.

The frozen contract is checked before loading any weights. The upstream AST
selects only the named network definitions, never ComfyUI installation hooks.
Supply an existing complete golden AV state as the real normalized input.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--av', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    contract = json.loads((Path(__file__).parent / 'upscale/contract.json').read_text())
    assert sha(a.weights) == contract['artifact_sha256'], 'unqualified weights'
    assert sha(a.upstream) == contract['code_sha256'], 'unqualified source'
    a.out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from safetensors.torch import load_file
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1234)
    names = {'normalization', 'zero_module', 'ResBlockEmb3D', 'TemporalConv', 'LatentResizer3D'}
    tree = ast.parse(a.upstream.read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(selected) == len(names)
    scope = dict(torch=torch, nn=torch.nn, F=torch.nn.functional)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(a.upstream), 'exec'), scope)
    device = torch.device('cuda')
    weights = load_file(str(a.weights), device='cpu')
    result = dict(contract_sha256=sha(Path(__file__).parent / 'upscale/contract.json'),
                  torch=torch.__version__, cuda=torch.version.cuda,
                  cudnn=torch.backends.cudnn.version(), artifact_sha256=sha(a.weights),
                  av_sha256=sha(a.av), cases=[], files={})

    def save(name, x):
        data = x.detach().float().cpu().contiguous().numpy().astype('<f4')
        path = a.out / (name + '.f32')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.tobytes())
        result['files'][str(path.relative_to(a.out))] = dict(sha256=sha(path), shape=list(data.shape))

    with a.av.open('rb') as f:
        header = f.read(160)
        assert header[:8] == b'H3AV\r\n\x1a\n' and struct.unpack_from('<I', header, 8)[0] == 3
        width, height, frames, t, h, w, at = struct.unpack_from('<7I', header, 24)
        vb, ab = struct.unpack_from('<2Q', header, 72)
        assert vb == 24*t*h*w*4 and ab == 64*at*4
        payload = f.read()
        assert len(payload) == vb+ab
        assert hashlib.sha256(header[:128]+payload).digest() == header[128:160]
    real = torch.from_numpy(np.frombuffer(payload[:vb], dtype='<f4').copy()).reshape(1, 24, t, h, w)
    audio = torch.from_numpy(np.frombuffer(payload[vb:], dtype='<f4').copy()).reshape(1, 32, 2, at)
    save('domain/audio-native', audio.reshape(64, at))
    save('domain/audio-comfy', audio)
    mean = torch.tensor(contract['normalization']['mean']).reshape(1, 24, 1, 1, 1)
    std = torch.tensor(contract['normalization']['std']).reshape(1, 24, 1, 1, 1)
    save('domain/video-native', real)
    save('domain/video-comfy', real)
    save('domain/video-decoder-input', real*std+mean)
    cases = {}
    for c in contract['fixture_cases']:
        shape = c['shape']; name = c['id']
        x = torch.zeros(shape)
        if name == 'constant': x.fill_(0.25)
        elif name == 'channel-ramp': x += torch.linspace(-2, 2, 24).reshape(1, 24, 1, 1, 1)
        elif name == 'random': x = torch.randn(shape)
        elif name == 'impulse-edge':
            x[0, 0, 0, 0, 0] = 5
            x[0, 23, -1, -1, -1] = -5
        elif name == 'real': x = real[:, :, :shape[2], :shape[3], :shape[4]].contiguous()
        assert list(x.shape) == shape
        cases[name] = x
        save(name+'/input', x)
    for dtype in (torch.bfloat16, torch.float32):
        mode = 'bf16' if dtype == torch.bfloat16 else 'fp32'
        net = scope['LatentResizer3D']().eval().to(device=device, dtype=dtype)
        net.load_state_dict(weights, strict=True)
        m, s = mean.to(device=device, dtype=dtype), std.to(device=device, dtype=dtype)
        for name, x in cases.items():
            records = []
            def hook(label):
                def record(module, args, out):
                    save(name+'/'+mode+'/'+label, out)
                    records.append(label)
                return record
            handles = [module.register_forward_hook(hook(label)) for label, module in net.named_modules()
                       if label in {'conv_in', 'embed', 'norm_out', 'conv_out'}
                       or label.startswith(('in_blocks.', 'out_blocks.')) and label.count('.') == 1]
            begin = time.monotonic()
            with torch.inference_mode():
                xin = x.to(device=device, dtype=dtype)
                normalized = (xin-m)/s
                save(name+'/'+mode+'/normalized', normalized)
                y = net(normalized, scale=2.0, target_size=(x.shape[2], x.shape[3]*2, x.shape[4]*2), enable_chunking=False)
                save(name+'/'+mode+'/output', y*s+m)
            for handle in handles: handle.remove()
            result['cases'].append(dict(id=name, dtype=mode, records=records, seconds=time.monotonic()-begin))
        del net
        torch.cuda.empty_cache()
    result['passed'] = True
    (a.out/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
