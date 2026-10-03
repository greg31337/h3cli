#!/usr/bin/env python3
"""Historical single-still codec/generation campaign; not the current test suite.

Requires the normal CUDA build, pinned image VAE, original H3 weights, inputs,
archived Metal codec fixtures. The old-source video comparison was removed;
current full decode is covered by the cleanup render matrix.
No accelerated precision/routing settings and at most six evaluations/render.
"""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path('outputs/single-still-cuda')
METAL = Path('outputs/single-still/fixtures')
MODEL = 'models/image-vae/minimax_h3_t1_image_vae_step1597.safetensors'
CONTRACT = json.loads(Path('tests/still_contract.json').read_text())
ENV = dict(os.environ, H3_TEST_MAX_EVALUATIONS='6', H3_PROFILE='1')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def run(args, log, result=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    record = {'argv': [str(a) for a in args], 'cwd': os.getcwd(),
              'env': {'H3_TEST_MAX_EVALUATIONS': '6', 'H3_PROFILE': '1'},
              'started_unix': time.time(), 'log': str(log)}
    print('RUN', record['argv'], flush=True)
    with log.open('w') as out:
        p = subprocess.run(record['argv'], env=ENV, stdout=subprocess.PIPE if result else out,
                           stderr=out, text=True)
    record.update(returncode=p.returncode, wall_seconds=time.time()-record['started_unix'])
    with (ROOT/'commands.jsonl').open('a') as f:
        f.write(json.dumps(record)+'\n')
    if result:
        result.write_text(p.stdout)
    if p.returncode:
        raise RuntimeError(f'command failed: {args}; see {log}')


def metrics(a, b):
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    d = a.astype(np.float64)-b
    return {'mae': float(np.abs(d).mean()),
            'psnr': float(-10*np.log10(max(float(np.square(d).mean()), 1e-30)))}


def gate(m):
    return (m['mae'] <= CONTRACT['gates']['unclamped_rgb_mae_max'] and
            m['psnr'] >= CONTRACT['gates']['unclamped_rgb_psnr_min_db'])


def save(name, obj):
    (ROOT/name).write_text(json.dumps(obj, indent=2)+'\n')


def png(path, rgb):
    Image.fromarray(np.uint8(np.clip(rgb, 0, 1)*255+.5)).save(path)


def codec():
    rows = []
    assert sha(MODEL) == CONTRACT['artifact_sha256']
    for w, h in [(256,256), (512,512), (640,480)]:
        old = METAL/str(w)
        dest = ROOT/'fixtures'/str(w)
        dest.mkdir(parents=True, exist_ok=True)
        # Same normalized latent as Metal: compare before output rounding.
        run(['./bin/still_probe', 'decode', MODEL, old/'latent.safetensors',
             dest/'candidate.f32', '3'], dest/'decode.log')
        candidate = np.fromfile(dest/'candidate.f32', np.float32).reshape(h,w,3)
        reference = np.load(old/'reference_slices.npy')[3]
        metal = np.fromfile(old/'candidate.f32', np.float32).reshape(h,w,3)
        comparison = metrics(candidate, reference)
        assert gate(comparison), comparison
        row = {'size':[w,h], 'same_latent_reference':comparison,
               'same_latent_metal':metrics(candidate, metal)}
        allocations = re.findall(r'gpu_live=(\d+)', (dest/'decode.log').read_text())
        assert len(allocations)==3 and len(set(allocations))==1
        row['repeated_live_bytes'] = [int(x) for x in allocations]
        run(['./bin/h3cli','--decode-still-latent',old/'latent.safetensors','--image-vae',MODEL,
             '-o',dest/'output.png','--frames-dir',dest/'ppm','--profile'], dest/'cli.log')
        assert len(list((dest/'ppm').glob('*.ppm')))==1
        for name in ['source.png','reference.png']:
            shutil.copy2(old/name, dest/name)
        # Fresh CUDA encoder/decoder versus independent CUDA F32 equations.
        fresh = dest/'roundtrip'
        fresh.mkdir(exist_ok=True)
        run(['./bin/still_probe','encode',MODEL,'inputs/2.jpg',str(w),str(h),fresh/'latent.safetensors'],fresh/'encode.log')
        run([sys.executable,'tests/still_reference.py',MODEL,fresh,'--device','cuda','--encoder'],fresh/'reference.log')
        run(['./bin/still_probe','decode',MODEL,fresh/'latent.safetensors',fresh/'candidate.f32','1'],fresh/'decode.log')
        from still_reference import load_latent
        encoder=metrics(load_latent(fresh/'latent.safetensors'),np.load(fresh/'reference_latent.npy'))
        # Report latent differences; the pre-frozen acceptance is on RGB.
        a=np.fromfile(fresh/'candidate.f32',np.float32).reshape(h,w,3)
        b=np.load(fresh/'reference_slices.npy')[3]
        source=np.fromfile(fresh/'source.f32',np.float32).reshape(h,w,3)
        row.update(encoder=encoder,roundtrip_reference=metrics(a,b),source_candidate=metrics(source,a),source_reference=metrics(source,b))
        assert gate(row['roundtrip_reference']),row
        if row['source_reference']['psnr']>25:
            assert row['source_candidate']['psnr']>25,row
        png(fresh/'source.png',source);png(fresh/'reference.png',b);png(fresh/'output.png',a)
        row['pass']=True;rows.append(row);save('codec.json',rows)
        print('PASS codec',row,flush=True)
    run(['./bin/still_controls',MODEL,METAL/'256/latent.safetensors',ROOT/'fixtures/256'],ROOT/'controls.log')
    run(['./bin/still_lifecycle'],ROOT/'lifecycle.log')
    ref=np.load(METAL/'256/reference_slices.npy')
    slices=[]
    for i in range(4):
        a=np.fromfile(ROOT/f'fixtures/256/candidate-slice-{i}.f32',np.float32).reshape(256,256,3)
        m=metrics(a,ref[i]);assert gate(m);slices.append(m)
    save('slice-controls.json',slices)


def generation():
    rows={}
    prompt='A polished red ceramic teapot on a pale wooden table, a single still photograph, soft studio lighting.'
    portrait='A portrait photograph of the woman in <Picture 1>, smiling softly, soft studio lighting, neutral background.'
    for name,text,ref in [('prompt',prompt,False),('repeat',prompt,False),('reference',portrait,True)]:
        dest=ROOT/'generation'/name;dest.mkdir(parents=True,exist_ok=True)
        args=['./bin/still_generate',dest,text]+(['reference'] if ref else [])
        run(args,dest/'render.log',dest/'result.json')
        row=json.loads((dest/'result.json').read_text());assert row['pass']
        row['prompt']=text;row['png_sha256']=sha(dest/'output.png');row['latent_sha256']=sha(dest/'latent.safetensors')
        steps=re.findall(r'"wall_seconds":([0-9.]+)',(dest/'render.log').read_text())
        row['step_wall_seconds']=[float(x) for x in steps]
        rows[name]=row;save('generation.json',rows)
    pairs=[]
    for step in range(7):
        a=sha(ROOT/f'generation/prompt/step-{step}.f32');b=sha(ROOT/f'generation/repeat/step-{step}.f32')
        assert a==b,(step,a,b);pairs.append(a)
    assert rows['prompt']['latent_sha256']==rows['repeat']['latent_sha256']
    assert rows['prompt']['png_sha256']==rows['repeat']['png_sha256']
    save('determinism.json',{'pass':True,'initial_and_step_sha256':pairs})
    dest=ROOT/'generation/prompt'
    run(['./bin/h3cli','--decode-still-latent',dest/'latent.safetensors','--image-vae',MODEL,'-o',dest/'redecoded.png','--profile'],dest/'redecode.log')
    assert sha(dest/'output.png')==sha(dest/'redecoded.png')
    print('PASS generation, repeat and redecode',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['codec','generation']);a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    globals()[a.phase]()

if __name__=='__main__':main()
