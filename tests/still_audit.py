#!/usr/bin/env python3
import hashlib,json,struct
from pathlib import Path
import numpy as np

def header(path):
 with path.open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];return json.loads(f.read(n)),8+n

def main():
 image=Path('models/image-vae/minimax_h3_t1_image_vae_step1597.safetensors');original=Path('models/MiniMax-H3/FL2VA/video_vae/source/model.safetensors')
 ih,io=header(image);oh,oo=header(original)
 out={'image':str(image),'original':str(original),'tensors':[]}
 for name,t in sorted(ih.items()):
  if name=='__metadata__' or name not in oh:continue
  o=oh[name];assert t['shape']==o['shape']
  a=np.memmap(image,'<f2',mode='r',offset=io+t['data_offsets'][0],shape=tuple(t['shape'])).reshape(-1)
  b=np.memmap(original,{'F32':'<f4','F16':'<f2'}[o['dtype']],mode='r',offset=oo+o['data_offsets'][0],shape=tuple(o['shape'])).reshape(-1)
  equal=rounded=True;maximum=0.;nonfinite=False
  for i in range(0,len(a),1<<20):
   x=np.asarray(a[i:i+(1<<20)],np.float32);y=np.asarray(b[i:i+(1<<20)],np.float32)
   nonfinite |= not bool(np.isfinite(x).all());equal &= bool(np.array_equal(x,y));rounded &= bool(np.array_equal(x,y.astype(np.float16).astype(np.float32)));maximum=max(maximum,float(np.max(np.abs(x-y))))
  out['tensors'].append(dict(name=name,source_dtype=o['dtype'],image_dtype=t['dtype'],value_equal=equal,equal_after_f16_round=rounded,max_abs=maximum,nonfinite=nonfinite))
 cfg=json.loads(ih['__metadata__']['minimax_h3_video_vae']);oc=json.loads(Path('models/MiniMax-H3/FL2VA/video_vae/config.json').read_text())
 out['whitening_equal']=all(np.array_equal(np.array(cfg[k],np.float32),np.array(oc[k],np.float32)) for k in ['latents_mean','latents_std'])
 for prefix in ['encoder.','quant_conv.','post_quant_conv.','decoder.']:
  ts=[x for x in out['tensors'] if x['name'].startswith(prefix)];print(prefix,len(ts),sum(x['value_equal'] for x in ts),sum(x['equal_after_f16_round'] for x in ts),flush=True)
 root=Path('outputs/single-still');(root/'checkpoint-audit.json').write_text(json.dumps(out,indent=2)+'\n')
 files=[Path('tests/still_contract.json'),Path('tests/still_reference.py'),Path('inputs/2.jpg')]+list(Path('models/MiniMax-H3/FL2VA/video_vae').glob('*.py'))+list((root/'source').glob('*'))
 (root/'source-manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},indent=2)+'\n')
if __name__=='__main__':main()
