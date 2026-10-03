#!/usr/bin/env python3
"""Serial real image codec qualification. No denoiser or remote CUDA execution."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import json,subprocess,sys,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('outputs/single-still');MODEL='models/image-vae/minimax_h3_t1_image_vae_step1597.safetensors'
def run(cmd,log):
 with log.open('w') as f:
  f.write(json.dumps(cmd)+'\n');f.flush();subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
def metrics(a,b):
 mse=float(np.mean((a.astype(np.float64)-b)**2));return dict(mae=float(np.mean(np.abs(a-b))),psnr=-10*np.log10(max(mse,1e-30)))
def main():
 rows=[]
 for w,h in [(256,256),(512,512),(640,480)]:
  dest=ROOT/'fixtures'/str(w);dest.mkdir(parents=True,exist_ok=True)
  if not (dest/'latent.safetensors').exists():run(['./bin/still_probe','encode',MODEL,'inputs/2.jpg',str(w),str(h),str(dest/'latent.safetensors')],dest/'encode.log')
  run(['./bin/still_probe','decode',MODEL,str(dest/'latent.safetensors'),str(dest/'candidate.f32'),'3'],dest/'decode.log')
  run([sys.executable,'tests/still_reference.py',MODEL,str(dest),'--encoder'],dest/'reference.log')
  if w==512:run([sys.executable,'tests/still_reference.py',MODEL,str(dest),'--whole'],dest/'whole.log')
  run(['./bin/h3cli','--decode-still-latent',str(dest/'latent.safetensors'),'--image-vae',MODEL,'-o',str(dest/'output.png'),'--frames-dir',str(dest/'ppm'),'--profile'],dest/'cli.log')
  source=np.fromfile(dest/'source.f32',np.float32).reshape(h,w,3);a=np.fromfile(dest/'candidate.f32',np.float32).reshape(h,w,3);b=np.load(dest/'reference_slices.npy')[3]
  row={'size':[w,h],'candidate_reference':metrics(a,b),'source_candidate':metrics(source,a),'source_reference':metrics(source,b)}
  row['pass']=row['candidate_reference']['mae']<=.01 and row['candidate_reference']['psnr']>=35
  if row['source_reference']['psnr']>25:row['pass'] &= row['source_candidate']['psnr']>25
  for name,rgb in [('source',source),('reference',b)]:Image.fromarray(np.uint8(np.clip(rgb,0,1)*255+.5)).save(dest/(name+'.png'))
  for i,rgb in enumerate(np.load(dest/'reference_slices.npy')):Image.fromarray(np.uint8(np.clip(rgb,0,1)*255+.5)).save(dest/f'slice-{i}.png')
  row['pass']=bool(row['pass'])
  (dest/'metrics.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row);print(row,flush=True)
 (ROOT/'codec-metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
 if not all(r['pass'] for r in rows):raise SystemExit('codec gates failed')
if __name__=='__main__':main()
