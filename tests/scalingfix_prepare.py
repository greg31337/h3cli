#!/usr/bin/env python3
"""Build isolated native capture/replay drivers and capture real presentations."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse,os,shutil,subprocess,sys,tarfile
from pathlib import Path
from scalingfix_instrument import instrument
from source_tree import copy_source_tree, source_path, build_target, build_path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/scalingfix-validation'
FLAGS=['-framework', 'Foundation', '-framework', 'Metal','-framework','MetalPerformanceShaders','-framework','MetalPerformanceShadersGraph','-framework','Accelerate','-licucore','-lm']
def build(label):
    d=OUT/label;d.mkdir(exist_ok=True)
    if label=='baseline' and not (d/'Makefile').exists():
        archive=OUT/'baseline.tar'
        subprocess.run(['git','archive','-o',str(archive),'91546c572f8f6548ddd7da400442b0ffbf8eeae2'],cwd=ROOT,check=True)
        with tarfile.open(archive) as f:f.extractall(d,filter='data')
    if label=='native':
        copy_source_tree(ROOT, d)
    else:
        # Restore only instrumentation target so repeated builds stay valid.
        (source_path(d, 'text_encoder.c')).write_bytes(subprocess.check_output(['git','show','91546c5:h3_text_encoder.c'],cwd=ROOT))
    instrument(d)
    with (d/'build.log').open('w') as log:subprocess.run(['make','-j8',build_target(d,'h3cli'),build_target(d,'libh3.a')],cwd=d,stdout=log,stderr=log,check=True)
    (ROOT/'bin').mkdir(exist_ok=True)
    subprocess.run(['clang','-O3','-std=c11','-D_DARWIN_C_SOURCE','-I',str(d),str(ROOT/'tests/scalingfix_qwen.c'),str(build_path(d,'libh3.a')),*FLAGS,'-o',str(ROOT/'bin'/('scalingfix_qwen_'+label))],check=True)
    return d

def main():
    p=argparse.ArgumentParser();p.add_argument('--build-only',action='store_true');a=p.parse_args()
    for label in ('native','baseline'):build(label)
    if a.build_only:return
    env={k:v for k,v in os.environ.items() if not k.startswith('H3_')};env['H3_TEST_QWEN_CAPTURE_ONLY']='1'
    prompts={'plain':'A woman walks through a quiet room and looks toward the camera.',
        'dialogue':'A woman on the left asks <d>[English] Where are you going?</d> A man on the right replies <d>[English] To the garden.</d>',
        'image':'The woman in <Picture 1> stands to the left of a seated man. She raises her right hand while he looks at a red book on the table.',
        'video':'The woman in <Video 1> turns toward a second woman on her right. A man stands behind them near a window. The camera remains steady.'}
    for case,prompt in prompts.items():
        d=OUT/'presentations'/case;d.mkdir(parents=True,exist_ok=True);env['H3_TEST_QWEN_DUMP']=str(d)
        if case in ('plain','dialogue'):cmd=[str(OUT/'native/qwen'),str(ROOT/'models/MiniMax-H3/FL2VA/text_encoder'),'--plain',prompt]
        else:
            if case=='video':
                reference=OUT/'reference.mp4'
                subprocess.run(['ffmpeg','-v','error','-y','-loop','1','-i',str(ROOT/'inputs/body1.jpg'),'-vf',"scale=512:512,zoompan=z='1+0.0006*on':x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d=96:s=512x512:fps=24",'-frames:v','96','-c:v','libx264','-pix_fmt','yuv420p',str(reference)],check=True)
                refs=['--ref-silent-video',str(reference)]
            else:refs=['--ref-image',str(ROOT/'inputs/2.jpg')]
            cmd=[str(OUT/'native/bin/h3cli'),'-d',str(ROOT/'models/MiniMax-H3'),'-p',prompt,'--width','320','--height','320','--frames','56','--steps','4',*refs,'-o',str(d/'unused.mp4')]
        print('capture',case,flush=True)
        with (d/'capture.log').open('w') as log:r=subprocess.run(cmd,cwd=OUT/'native',env=env,stdout=log,stderr=log)
        assert r.returncode!=0 and 'Qwen presentation captured' in (d/'capture.log').read_text()
        (d/'prompt.txt').write_text(prompt)
        import numpy as np
        spec=np.fromfile(d/'spec.u64','<u8')
        if case=='dialogue':
            ids=np.fromfile(d/'ids.u32','<u4');assert sum(ids==151669)==sum(ids==151670)==2
        if case=='video':assert 512<=spec[0]<7936
        print(case,spec,flush=True)
if __name__=='__main__':main()
