#!/usr/bin/env python3
"""Deterministic reference-derived motion/text and reference-encoder fixtures."""
import json,pathlib,hashlib,argparse
import numpy as np
from PIL import Image,ImageDraw
from fast_vae_quality import video
p=argparse.ArgumentParser();p.add_argument('--root',default='outputs/fast-vae/cuda/pixels');args=p.parse_args()
root=pathlib.Path(args.root);root.mkdir(parents=True,exist_ok=True)
for name,path,w,h in [('faces','inputs/2.jpg',384,256),('texture','inputs/1.jpg',512,320)]:
    original=Image.open(path).convert('RGB').resize((w+24,h+16),Image.Resampling.LANCZOS);frames=[]
    for t in range(39):
        im=original.crop((t%25,t%17,t%25+w,t%17+h));draw=ImageDraw.Draw(im)
        draw.rectangle((5,5,190,31),fill='white');draw.text((10,11),f'H3 VAE 012345 / frame {t:02}',fill='black');frames.append(np.array(im))
    data=np.array(frames).astype('float32')/255;data.transpose(3,0,1,2).copy().tofile(root/(name+'.f32'))
    command=video(root/(name+'.mp4'),data,w,h)
    (root/(name+'.json')).write_text(json.dumps(dict(kind='reference-derived pan with text; no denoiser',source=path,source_sha256=hashlib.file_digest(open(path,'rb'),'sha256').hexdigest(),frames=39,width=w,height=h,denoising_steps=None,command=command),indent=2))
# Actual input sizes for the M5 isolated encoder; CLI C5 separately exercises
# the released match/max preprocessing policy without substituting this resize.
im=Image.open('inputs/2.jpg').convert('RGB')
for name,w,h,t in [('image-match',640,480,1),('image-max',im.width//16*16,im.height//16*16,1),('video-short',128,96,22),('video-tail',128,96,39)]:
    x=np.array(im.resize((w,h),Image.Resampling.LANCZOS)).astype('float32')/255
    raw=np.stack([np.roll(x,j%7,axis=1) for j in range(t)]).transpose(3,0,1,2).copy();raw.tofile(root/(name+'.f32'))
    (root/(name+'.json')).write_text(json.dumps(dict(frames=t,width=w,height=h,kind='isolated encoder input, no denoiser')))
