#!/usr/bin/env python3
"""Optional larger A/B renders; reports execution, not automatic visual approval."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fold_lora import sha256

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=256)
    parser.add_argument('--base',type=Path,default=ROOT/'models/MiniMax-H3')
    parser.add_argument('--models',type=Path,default=ROOT/'outputs/lora-validation')
    args=parser.parse_args()
    out=args.models/f'quality-{args.size}'
    out.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in os.environ.items() if not k.startswith('H3_')}
    env['H3_CPU_SAMPLER']='1'
    results={}
    for name,model,steps in [('base-20',args.base,20),('realism-20',args.models/'realism',20),
                              ('base-8',args.base,8),('turbo-8',args.models/'turbo',8)]:
        command=[str(ROOT/'bin/h3cli'),'-d',str(model),'-p',
          'r34l1sm, the woman smiles and turns toward the camera. Steady camera, quiet room ambience.',
          '--seed','72','--width',str(args.size),'--height',str(args.size),'--frames','56','--steps',str(steps),
          '--first-frame',str(ROOT/'inputs/face1.jpg'),'-o',str(out/(name+'.mp4'))]
        start=time.monotonic()
        with (out/(name+'.log')).open('w') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=log,check=True)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',out/(name+'.mp4')]))
        assert {s['codec_type'] for s in probe['streams']}=={'audio','video'}
        video=next(s for s in probe['streams'] if s['codec_type']=='video')
        assert (video['width'],video['height'],int(video['nb_frames']))==(args.size,args.size,56)
        results[name]=dict(command=command,seconds=time.monotonic()-start,mp4_sha256=sha256(out/(name+'.mp4')))
        (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print('PASS',name,flush=True)
    assert results['base-20']['mp4_sha256']!=results['realism-20']['mp4_sha256']
    assert results['base-8']['mp4_sha256']!=results['turbo-8']['mp4_sha256']
    command=['ffmpeg','-v','error']
    for name in results: command+=['-i',str(out/(name+'.mp4'))]
    filters=[f"[{i}:v]select='not(mod(n,14))',scale=256:256,tile=4x1[r{i}]" for i in range(4)]
    filters+=['[r0][r1][r2][r3]vstack=inputs=4[out]']
    subprocess.run(command+['-filter_complex',';'.join(filters),'-map','[out]','-frames:v','1','-update','1','-y',str(out/'comparison.png')],check=True)
    print('Comparison rows: base-20, realism-20, base-8, turbo-8; columns: frames 0,14,28,42.',flush=True)


if __name__=='__main__': main()
