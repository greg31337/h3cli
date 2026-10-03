#!/usr/bin/env python3
"""Single-pipeline CLI/delivery tests supplementing the golden reference suite."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse,json,os,subprocess,time
from pathlib import Path
from cuda_reference_regression import regression_config,runtime_env,sha,write

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--state',type=Path,required=True)
 p.add_argument('--model',type=Path);p.add_argument('--fixtures',type=Path)
 p.add_argument('--preview-vae-model',type=Path)
 a=p.parse_args();a.source=a.source.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
 config=regression_config(a.source,a.model,a.fixtures);env=runtime_env();model=config['model'];rows=[]
 def run(name,args,needle=None,override=None):
  cmd=[str(a.source/'bin/h3cli'),'-d',model,*map(str,args)];start=time.monotonic()
  r=subprocess.run(cmd,cwd=a.source,env=env|(override or {}),stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
  text=r.stdout.decode(errors='replace');(a.out/(name+'.log')).write_bytes(r.stdout)
  ok=(r.returncode!=0 and needle in text and 'text encoder' not in text and 'load transformer core' not in text) if needle else r.returncode==0
  rows.append(dict(name=name,argv=cmd,returncode=r.returncode,seconds=time.monotonic()-start,passed=ok))
  write(a.out/'result.json',dict(passed=False,complete=False,checks=rows,binary_sha256=sha(a.source/'bin/h3cli')))
  if not ok:raise RuntimeError(name+': '+text[-800:])
 base=['-p','test','--steps','6']
 for flag in ('--cuda-reference','--legacy-ref2va-video-pipeline','--decode-full-state','--fast-cuda','--resume-default-cuda'):
  run('removed-'+flag[2:],base+[flag],'unrecognized option')
 state=a.state.resolve();state_hash=sha(state)
 small=a.out/'cropped-diagnostic.h3av'
 with (a.out/'crop.log').open('w') as log:
  subprocess.run([str(a.source/'bin/cuda_policy_context'),str(state),str(small)],cwd=a.source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=10)
 # Default full delivery uses a bounded cropped-latent diagnostic.
 # Preview remains independent and retains all C0 frames.
 tiny=a.preview_vae_model or a.source/'models/taeh3.safetensors'
 for name,flags in [('preview',['--preview-vae','--preview-vae-model',tiny]),('full',[])]:
  chosen=state if name=='preview' else small
  run('decode-'+name,['--decode-av-state',chosen,*flags,'-o',a.out/(name+'.mp4')])
  assert sha(state)==state_hash
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(a.out/(name+'.mp4'))],env=env))
  assert any(s['codec_type']=='video' and int(s['nb_read_frames'])==(124 if name=='preview' else 22) for s in probe['streams'])
  assert any(s['codec_type']=='audio' for s in probe['streams'])
 assert len(rows)==7 and all(x['passed'] for x in rows)
 write(a.out/'result.json',dict(passed=True,complete=True,checks=rows,binary_sha256=sha(a.source/'bin/h3cli'),state_sha256=state_hash))
 print('PASS:',len(rows),'mode/delivery checks; saved denoising unchanged')
if __name__=='__main__':main()
