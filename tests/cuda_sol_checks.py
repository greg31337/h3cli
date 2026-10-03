#!/usr/bin/env python3
"""Bounded additional contract checks. Every subprocess uses the campaign clock."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse,fcntl,json,sys,struct,time,subprocess
from pathlib import Path
import numpy as np
from cuda_sol_campaign import MODEL,QUANT_CACHE,ROOT,ACCEPT,execute,render,verify,save,media,remaining,command
from cuda_sol_oracle import run,metrics

def av(path):
    raw=Path(path).read_bytes();fields=struct.unpack_from('<10I',raw,24);nv,na=struct.unpack_from('<2Q',raw,72)
    return np.frombuffer(raw,dtype='<f4',offset=160,count=nv//4).reshape(24,fields[3],fields[4],fields[5]),np.frombuffer(raw,dtype='<f4',offset=160+nv,count=na//4).reshape(2,32,fields[6])
def synthetic():
    for tool in ['memcheck','racecheck']:
        case=json.loads((ROOT/'operator-final/mixed/result.json').read_text());argv=case['argv']
        target=ROOT/('sanitizer-'+tool);target.mkdir(exist_ok=False)
        for flag,name in [('--output','out.bf16'),('--routes','routes.bin'),('--stats','native.json')]:argv[argv.index(flag)+1]=str(target/name)
        execute('B-'+tool,'B',['/usr/local/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99']+argv,300,allow_failure=True)
    # Extreme finite inputs exercise summary-error cleanup, not approximation quality.
    dest=ROOT/'overflow';a=argparse.Namespace(output=dest,binary=Path('./bin/cuda_sol_native').resolve());dest.mkdir()
    run(a,dict(id='finite-overflow',sequence=65,pattern='overflow'))

def heldout():
    a=ROOT/'diagnostics/R1-default/steps';b=ROOT/'diagnostics/R1-sol/steps';result={}
    for kind in ['video','audio']:
        x=np.fromfile(a/f'step-002-{kind}-input.f32',dtype='<f4');y=np.fromfile(b/f'step-002-{kind}-input.f32',dtype='<f4')
        same=x.tobytes()==y.tobytes();m=metrics(np.fromfile(b/f'step-002-{kind}-velocity.f32',dtype='<f4'),np.fromfile(a/f'step-002-{kind}-velocity.f32',dtype='<f4'))
        result[kind]=dict(identical_evaluation_input=same,velocity=m,screen_pass=same and m['finite'] and m['relative_l2']<=.05 and m['cosine']>=.998)
    save(ROOT/'heldout-velocity.json',result)

def state_checks():
    chosen=json.loads((ROOT/'candidate.json').read_text());minimum=chosen['minimum'];identity=verify()
    prompt=json.loads(ACCEPT.read_text())['held_out']['prompt']
    base=dict(id='E-continuation',width=640,height=480,frames=243,steps=2,seed=4343,prompt=prompt,references=[],image_size='match',save_state=True,continue_from=str(ROOT/'runs/R1-default/output.h3av'))
    if Path(base['continue_from']).exists():
        for mode in ['default','sol']:render(base,mode,minimum,'E',600,dict(H3_CPU_SAMPLER='1'))
        pv,pa=av(base['continue_from']);checks={}
        for mode in ['default','sol']:
            path=ROOT/f'runs/E-continuation-{mode}/output.h3av'
            if path.exists():
                v,a=av(path);checks[mode]=dict(video_prefix_exact=v[:,:12].tobytes()==pv[:,-12:].tobytes(),audio_prefix_exact=a[:,:,:65].tobytes()==pa[:,:,-65:].tobytes())
        save(ROOT/'prefix-checks.json',checks)
    tiny=dict(id='E-uninterrupted',width=128,height=128,frames=22,steps=2,seed=4545,prompt='A yellow sailboat moving gently across a blue lake.',references=[],image_size='match',save_state=True)
    full=render(tiny,'sol',minimum,'E',240)
    paused=dict(tiny,id='E-paused',save_state=False,extra=['--stop-after-step','1','--save-sampler-state',str(ROOT/'paused.h3sample')])
    d=ROOT/'runs/E-paused-sol';execute('E-paused-sol','E',command(paused,'sol',d,minimum,identity),240,monitor=True,allow_failure=True)
    resumed=ROOT/'runs/E-resumed-sol'
    row=execute('E-resumed-sol','E',['./bin/h3cli','-d',str(MODEL),'--resume-sampler-state',str(ROOT/'paused.h3sample'),'--save-av-state',str(resumed/'output.h3av'),'-o',str(resumed/'output.mp4')],240,monitor=True,allow_failure=True)
    if (resumed/'output.h3av').exists():
        x,y=av(ROOT/'runs/E-uninterrupted-sol/output.h3av');xx,yy=av(resumed/'output.h3av');save(ROOT/'resume-equivalence.json',dict(video_exact=x.tobytes()==xx.tobytes(),audio_exact=y.tobytes()==yy.tobytes()))
        save(resumed/'media.json',media(resumed,tiny))
    for mode,flags in [('conflict',['--sol-min-exact','0.123']),('default',['--resume-default-cuda'])]:
        r=execute('E-resume-reject-'+mode,'E',['./bin/h3cli','-d',str(MODEL),'--resume-sampler-state',str(ROOT/'paused.h3sample')]+flags,60,allow_failure=True)
        save(ROOT/('resume-rejection-'+mode+'.json'),dict(rejected=r['returncode']!=0))
    for label,extra in [('fast',dict(fast=True)),('stream',dict(weight_mode='stream')),('floor-one',dict(extra=['--sol-min-exact','1']))]:
        c=dict(tiny,id='E-'+label,**extra);render(c,'sol',minimum,'E',300)
    # Completed-state presentation decoding uses the current non-SOL build.
    target=ROOT/'runs/E-decode'
    execute('E-decode','E',['./default-build/bin/h3cli','-d',str(MODEL),'--decode-av-state',str(ROOT/'runs/E-uninterrupted-sol/output.h3av'),'-o',str(target/'output.mp4')],180,monitor=True,allow_failure=True)
    save(target/'media.json',media(target,tiny))

def quant():
    minimum=json.loads((ROOT/'candidate.json').read_text())['minimum'];verify()
    # Existing native quant tests verify both projection modes, including the
    # BF16 activation/output contract independently of attention dispatch.
    execute('F-build-interfaces','F',['make','CUDA_SOL=1','CUDA_ARCH=120','bin/quant_native'],120,allow_failure=True)
    if Path('bin/quant_native').exists():
        for mode in ['fp8','nvfp4']:execute('F-interface-'+mode,'F',['./bin/quant_native',mode,str(QUANT_CACHE)],180,allow_failure=True)
    c=dict(id='F-nvfp4',width=128,height=128,frames=22,steps=2,seed=4646,prompt='A small red kite flying over a green meadow.',references=[],image_size='match',quant='nvfp4',save_state=True,extra=['--cuda-denoise-quant-cache',str(QUANT_CACHE)])
    for mode in ['default','sol']:render(c,mode,minimum,'F',300)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['synthetic','heldout','state','quant']);a=p.parse_args()
    with (ROOT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'synthetic':synthetic,'heldout':heldout,'state':state_checks,'quant':quant}[a.action]()
if __name__=='__main__':main()
