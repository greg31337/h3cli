#!/usr/bin/env python3
"""Development-only MLX replay of the pinned reference decoder, without denoising."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import time
import numpy as np

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('state', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--weights', type=Path, default=Path('models/taeh3.safetensors'))
    p.add_argument('--reference', type=Path, default=Path('outputs/preview-vae/reference/minimax_h3_taeh3.py'))
    p.add_argument('--audio', type=Path)
    p.add_argument('--trim', type=int, default=0)
    p.add_argument('--chunk', type=int, default=5)
    p.add_argument('--dtype', default='fp16')
    a = p.parse_args()
    start = time.monotonic()
    blob = a.state.read_bytes()
    assert blob[:8] == b'H3AV\r\n\x1a\n'
    assert hashlib.sha256(blob[:128] + blob[160:]).digest() == blob[128:160]
    w, h, frames, t, lh, lw, *_ = struct.unpack_from('<10I', blob, 24)
    video_bytes, audio_bytes = struct.unpack_from('<QQ', blob, 72)
    assert len(blob) == 160 + video_bytes + audio_bytes
    latent = np.frombuffer(blob, '<f4', count=video_bytes // 4, offset=160).reshape(1,24,t,lh,lw)
    spec = importlib.util.spec_from_file_location('taeh3_reference', a.reference)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    import mlx.core as mx
    decoder = module.MLXTAEH3Decoder(a.weights, dtype=a.dtype)
    loaded = time.monotonic()
    rgb = decoder.decode_ntchw(mx.array(latent.transpose(0,2,1,3,4)), chunk_size=a.chunk)
    mx.eval(rgb)
    rgb = np.asarray(rgb.astype(mx.float32)).transpose(0,1,3,4,2)[0]
    decoded = time.monotonic()
    assert rgb.shape == (frames,h,w,3) and np.isfinite(rgb).all()
    rgb = np.rint(np.clip(rgb[a.trim:],0,1)*255).astype(np.uint8)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}',
           '-r','24','-i','pipe:0']
    if a.audio: cmd += ['-i',str(a.audio),'-map','0:v','-map','1:a','-c:a','copy']
    cmd += ['-c:v','libx264','-preset','fast','-crf','6','-pix_fmt','yuv420p','-movflags','+faststart','-frames:v',str(len(rgb)),str(a.output)]
    subprocess.run(cmd,input=rgb.tobytes(),check=True,timeout=60)
    subprocess.run(['ffmpeg','-v','error','-i',str(a.output),'-f','null','-'],check=True,timeout=60)
    record = dict(state=str(a.state),state_sha256=hashlib.sha256(blob).hexdigest(),
                  weights_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),dtype=a.dtype,
                  chunk=a.chunk,width=w,height=h,frames=len(rgb),load_seconds=loaded-start,
                  decode_seconds=decoded-loaded,wall_seconds=time.monotonic()-start,
                  peak_metal_bytes=mx.get_peak_memory(),mp4_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())
    a.output.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))

if __name__ == '__main__': main()
