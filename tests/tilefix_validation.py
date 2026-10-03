#!/usr/bin/env python3
"""Resumable fixed-latent tile-policy matrix using real image pans and weights."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
ROOT=Path(__file__).resolve().parents[1]
MATRIX=[(320,320),(512,512),(480,864),(576,1024),(768,768),(768,1024),(768,1344)]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def pixels(h,w):
    # A deterministic translating composite includes faces, hair, fabric,
    # straight edges, a smooth gradient and low-contrast stripes.
    wide=w+32;canvas=Image.new('RGB',(wide,h));top=h*3//4
    for i,name in enumerate(('face1.jpg','body1.jpg','2.jpg')):
        x=i*wide//3;right=(i+1)*wide//3
        canvas.paste(ImageOps.fit(Image.open(ROOT/'inputs'/name).convert('RGB'),(right-x,top)),(x,0))
    a=np.asarray(canvas).copy()
    gradient=np.linspace(96,176,wide,dtype=np.float32).astype(np.uint8)
    a[top:]=gradient[None,:,None]
    a[top+10:top+14,:,1]=np.minimum(a[top+10:top+14,:,1]+5,255)
    canvas=Image.fromarray(a);draw=ImageDraw.Draw(canvas)
    for x in range(8,wide,47):draw.line((x,top+24,x,h-8),fill=(121,123,124),width=1)
    return np.stack([np.asarray(canvas.crop((int(32*t/21),0,int(32*t/21)+w,h))) for t in range(22)])
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'outputs/tilefix-validation/matrix')
    p.add_argument('--only',default='');p.add_argument('--oracle',action='store_true');a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in os.environ.items() if not k.startswith('H3_')};env['H3_PROFILE']='1'
    weights=ROOT/'models/MiniMax-H3/FL2VA/video_vae/source'
    for h,w in MATRIX:
        name=f'{h}x{w}'
        if a.only and name not in a.only.split(','):continue
        d=out/name;d.mkdir(exist_ok=True);latent=d/'latent.f32';recordfile=d/'runs.json'
        records=json.loads(recordfile.read_text()) if recordfile.exists() else {}
        if not latent.exists() or 'encode' not in records:
            rgb=pixels(h,w);Image.fromarray(rgb[0]).save(d/'source.png')
            src=d/'pixels.f32';(rgb.transpose(3,0,1,2).astype(np.float32)*np.float32(1/255)).tofile(src)
            cmd=[str(ROOT/'bin/tilefix_decode'),'encode','22',str(h),str(w),str(src),str(latent),str(weights)]
            print('encode',name,flush=True)
            with (d/'encode.log').open('w') as log:r=subprocess.run(cmd,env=env,cwd=ROOT,stdout=subprocess.PIPE,stderr=log,text=True,check=True)
            records['encode']=json.loads(r.stdout);assert records['encode']['latent_time']==7
            src.unlink();records['latent_sha256']=sha(latent);recordfile.write_text(json.dumps(records,indent=2)+'\n')
        assert sha(latent)==records['latent_sha256']
        for mode in ('default','auto','320'):
            if mode in records:continue
            case_env=env.copy()
            if mode!='default':case_env['H3_VAE_TILE_PIXELS']=mode
            cmd=[str(ROOT/'bin/tilefix_decode'),'resident','7',str(h),str(w),str(latent),str(d/(mode+'.f32')),str(weights)]
            print('decode',name,mode,flush=True);start=time.monotonic()
            with (d/(mode+'.log')).open('w') as log:r=subprocess.run(cmd,env=case_env,cwd=ROOT,stdout=subprocess.PIPE,stderr=log,text=True,check=True)
            records[mode]=json.loads(r.stdout);records[mode].update(command=cmd,process_seconds=time.monotonic()-start,sha256=sha(d/(mode+'.f32')),latent_sha256=sha(latent))
            recordfile.write_text(json.dumps(records,indent=2)+'\n')
            print('completed',name,mode,round(records[mode]['decode_seconds'],2),flush=True)
        if a.oracle and 'official' not in records:
            cmd=[sys.executable,str(ROOT/'tests/tilefix_oracle.py'),str(latent),str(d/'official.f32'),'--height',str(h),'--width',str(w)]
            print('official',name,flush=True)
            with (d/'official.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=log,check=True)
            records['official']=json.loads((d/'official.json').read_text());records['official']['sha256']=sha(d/'official.f32')
            recordfile.write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
