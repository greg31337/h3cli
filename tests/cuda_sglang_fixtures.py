#!/usr/bin/env python3
"""Build bounded, checksummed replay inputs from preserved SGLang captures.

The native teacher directory stores each evaluation's INPUT, not its output.
No checkpoint weights are read. Optional .h3av export uses a validated native
container solely for its geometry/compatibility identity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
from cuda_sglang_compare import oracle,pack_video,pack_audio


def unpack_video(rows,h,w):
    if rows.ndim!=2 or rows.shape[1]!=96 or rows.shape[0]%((h//2)*(w//2)):
        raise ValueError('invalid packed video shape')
    t=rows.shape[0]//((h//2)*(w//2))
    return rows.reshape(t,h//2,w//2,24,2,2).transpose(3,0,1,4,2,5).reshape(24,t,h,w)


def unpack_audio(rows):
    if rows.ndim!=2 or rows.shape[1]!=32 or rows.shape[0]%2:raise ValueError('invalid audio shape')
    return rows.reshape(2,-1,32).transpose(2,0,1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('oracle',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--av-template',type=Path)
    a=p.parse_args();root=a.oracle/'capture';spec=json.loads((a.oracle/'command.json').read_text())
    frames,evaluations=spec['frames'],spec['evaluations']
    if frames not in (124,362) or evaluations not in (6,50):raise ValueError('unsupported frozen geometry/schedule')
    vt=37 if frames==124 else 107;at=207 if frames==124 else 603
    a.output.mkdir(exist_ok=False,parents=True);files={}
    def save(name,value):
        data=np.ascontiguousarray(value).tobytes()
        with (a.output/name).open('xb') as f:f.write(data)
        files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    tokens=oracle(root,'text.encode_ids.args.0').reshape(-1).astype('<u4')
    text=oracle(root,'text.prompt_embeds.0')
    if text.shape!=(len(tokens),5120):raise ValueError('text/token shape mismatch')
    bits=np.ascontiguousarray(text,dtype='<f4').view('<u4')
    if np.any(bits&65535):raise ValueError('text capture is not exact BF16')
    save('tokens.u32',tokens);save('text.bf16',(bits>>16).astype('<u2'))
    final={}
    for step in range(evaluations+1):
        for modality in ('video','audio'):
            value=oracle(root,f'initial_{modality}_rows' if not step else f'step-{step-1:03d}.{modality}')
            if value.dtype!=np.float32 or not np.isfinite(value).all():raise ValueError('invalid sampler tensor dtype/value')
            native=unpack_video(value,30,40) if modality=='video' else unpack_audio(value)
            expected=(24,vt,30,40) if modality=='video' else (32,2,at)
            if native.shape!=expected:raise ValueError(f'incompatible {modality} shape {native.shape}')
            packed=pack_video(native) if modality=='video' else pack_audio(native)
            if not np.array_equal(packed,value):raise ValueError('packing round trip failed')
            save(f'step-{step:03d}-{modality}-latent.f32',native.astype('<f4'))
            if step==evaluations:final[modality]=np.ascontiguousarray(native,dtype='<f4').tobytes()
    if a.av_template:
        blob=a.av_template.read_bytes()
        if blob[:8]!=b'H3AV\r\n\x1a\n' or hashlib.sha256(blob[:128]+blob[160:]).digest()!=blob[128:160]:raise ValueError('invalid AV template')
        if struct.unpack_from('<3I',blob,24)!=(640,480,frames):raise ValueError('AV template geometry mismatch')
        if struct.unpack_from('<2Q',blob,72)!=(len(final['video']),len(final['audio'])):raise ValueError('AV payload size mismatch')
        header=bytearray(blob[:160]);payload=final['video']+final['audio'];header[128:160]=hashlib.sha256(header[:128]+payload).digest()
        data=header+payload;(a.output/'oracle-final.h3av').write_bytes(data)
        files['oracle-final.h3av']=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    manifest=dict(source=str(a.oracle),source_command_sha256=hashlib.sha256((a.oracle/'command.json').read_bytes()).hexdigest(),
        frames=frames,evaluations=evaluations,geometry=[640,480,frames],
        indexing='step-N is input to evaluation N, final step is clean output',files=files,
        runtime_env={'H3_TEST_SGLANG_INPUT_DIR':str(a.output.resolve()),'H3_TEST_NATIVE_TEACHER_DIR':str(a.output.resolve()),'H3_TEST_MAX_EVALUATIONS':str(evaluations)})
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'{len(files)} lossless replay artifacts, {sum(f["bytes"] for f in files.values())} bytes')


if __name__=='__main__':main()
