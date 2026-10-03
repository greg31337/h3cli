#!/usr/bin/env python3
"""CLI contract checks, plus an optional real full-schedule scouting render."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile
from test_sampler_file import entries

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--oracle',type=Path,default=ROOT/'outputs/resume-validation/real/t2va.trace')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/resume-validation/cli')
    args=parser.parse_args(); count=0
    def reject(options,phrase):
        nonlocal count
        result=subprocess.run([str(ROOT/'bin/h3cli'),'-d','/missing-model']+options,cwd=ROOT,capture_output=True,text=True)
        assert result.returncode!=0 and phrase in result.stderr,(options,result.stderr); count+=1
    for flag in ('--cuda-reference','--legacy-ref2va-video-pipeline','--decode-full-state','--fast-cuda','--resume-default-cuda'):
        reject([flag], 'option')
    for value in ['','-1','x','4x','2147483648']:
        reject(['-p','test','--stop-after-step',value,'--save-sampler-state','state'],'invalid stop-after-step')
    for value in ['21','1001']:
        reject(['-p','test','--steps','20','--stop-after-step',value,'--save-sampler-state','state'],'absolute boundary')
    for options in [ ['--stop-after-step','4'],['--save-sampler-state','state'],['--preview-on-stop'],
                     ['-p','test','--stop-after-step','4'],['-p','test','--save-sampler-state',''],
                     ['--resume-sampler-state',''],['-p','test','--preview-on-stop','--save-sampler-state','state'] ]:
        reject(options,'sampler state requires')
    reject(['-p','test','--steps','20','--stop-after-step','4','--save-sampler-state','state','--save-av-state','av'],'no complete .h3av')
    changes=[['-p','new'],['--seed','1'],['--width','256'],['--height','256'],['--frames','90'],['--steps','20'],
             ['--reuse','1'],['--core-reuse','1'],['--layers','50'],['--ref-image','missing'],['--first-frame','missing'],
             ['--last-frame','missing'],['--token-reduction'],['--ssd-streaming'],['--continue-from','missing'],
             ['--continue-context','39'],['--use-reference-rope'],['--continue-mode','bridge'],['--render-width','256']]
    for change in changes: reject(['--resume-sampler-state','missing']+change,'checkpoint is authoritative')
    with tempfile.TemporaryDirectory(prefix='h3-sample-cli-') as temp:
        path=Path(temp)/'state.h3sample'
        subprocess.run([str(ROOT/'bin/sampler_tests'),'--fixture',str(path)],cwd=ROOT,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        for stop in ['3','21']:
            reject(['--resume-sampler-state',str(path),'--stop-after-step',stop,'--save-sampler-state',str(path)],'absolute boundary')
        # These overrides pass checkpoint/CLI validation and reach model loading.
        # The deliberately absent model keeps this a cheap parser test.
        for permitted in [ ['-o',str(Path(temp)/'out.mp4')],
                           ['--profile','--frames-dir',str(Path(temp)/'frames')],
                           ['--show','--zoom','2'],
                           ['--stop-after-step','8','--save-sampler-state',str(Path(temp)/'next.h3sample'),'--preview-on-stop'] ]:
            reject(['--resume-sampler-state',str(path)]+permitted,'missing required model file')
        subprocess.run([str(ROOT/'bin/sampler_tests'),'--reference-fixture',str(path)],cwd=ROOT,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        reject(['--resume-sampler-state',str(path)],'missing required model file')
    if args.render:
        args.output.mkdir(parents=True,exist_ok=True)
        state=args.output/'step4.h3sample'
        env={k:v for k,v in os.environ.items() if not k.startswith('H3_')}; env['H3_CPU_SAMPLER']='1'
        command=[str(ROOT/'bin/h3cli'),'-d',str(ROOT/'models/MiniMax-H3'),'-p',
                 'The woman walks slowly through a sunlit garden and waves. Steady camera and quiet outdoor ambience.',
                 '--seed','72','--width','256','--height','256','--frames','90','--steps','20',
                 '--stop-after-step','4','--save-sampler-state',str(state),'--preview-on-stop',
                 '-o',str(args.output/'preview.mp4'),'--frames-dir',str(args.output/'frames')]
        with (args.output/'render.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=log,check=True)
        parts={p[0]:p for p in entries(state.read_bytes())}
        assert struct.unpack_from('<4I',parts[1][-1])==(1,20,4,1)
        assert len(parts[11][-1])==4+21*8
        av=parts[12][-1]+parts[13][-1]; trace=args.oracle.read_bytes()
        assert av==trace[4*len(av):5*len(av)], 'CLI 4-of-20 differs from uninterrupted step four'
        # Full schedule is stored verbatim, and differs from an independent
        # four-pass serving grid (covered additionally by the host test).
        assert 'paused successfully after 4/20 transitions' in (args.output/'render.log').read_text()
        assert len(list((args.output/'frames').glob('*.ppm')))==90, 'full pause-preview frame delivery missing'
        count+=1
    print(f'ok: {count} sampler CLI cases')

if __name__=='__main__': main()
