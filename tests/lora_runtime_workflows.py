#!/usr/bin/env python3
"""Reproducible real-backend runtime LoRA acceptance workflows.

Host arithmetic/fault tests are separate. This driver requires existing local
models/adapters, ffmpeg, and a functioning Metal or CUDA build. No downloads.
"""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROMPT='A person in a blue jacket walks through a sunny park. Birds sing softly.'
p=argparse.ArgumentParser()
p.add_argument('--backend',choices=['metal','cuda'],required=True)
p.add_argument('--model',required=True)
p.add_argument('--cache',required=True)
p.add_argument('--turbo',required=True)
p.add_argument('--ordinary')
p.add_argument('--reference',default='inputs/face1.jpg')
p.add_argument('--preview-model',default='models/taeh3.safetensors')
p.add_argument('--quant-cache')
p.add_argument('--offline-model',help='Existing standalone offline-folded model for legacy resume coverage')
p.add_argument('--output',required=True)
p.add_argument('--moved-directory',help='Directory for a copied adapter and relocated model aliases')
p.add_argument('--allow-evict-generated-variant',action='store_true',
               help='Permit recovery case to remove its recorded baseline cache entry under an exclusive lease')
p.add_argument('--cases',nargs='+',default=['baseline','resume','ref-resume','continuation'])
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
env={**os.environ,'H3_CPU_SAMPLER':'1'}
binary_hash=hashlib.sha256(Path('bin/h3cli').read_bytes()).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def selection(adapter=None,model=None,cache=None):
    return ['-d',model or a.model,'--lora',adapter or a.turbo,'--lora-cache',cache or a.cache]

def preview():return ['--preview-vae','--preview-vae-model',a.preview_model]

def settings(frames=56,steps=8,width=256,height=256):
    return ['-p',PROMPT,'--width',str(width),'--height',str(height),'--frames',str(frames),'--steps',str(steps),'--seed','42']

def run(name,args,expected=None,save=True):
    av=out/(name+'.h3av');media=out/(name+'.mp4');log=out/(name+'.log')
    command=['./bin/h3cli',*args,'--profile','-o',str(media)]
    if save:command+=['--save-av-state',str(av)]
    started=time.monotonic()
    with log.open('w') as f, subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env) as r:
        _,status,usage=os.wait4(r.pid,0)
        r.returncode=os.waitstatus_to_exitcode(status)
    result=dict(argv=command,environment={'H3_CPU_SAMPLER':'1'},binary_sha256=binary_hash,
        returncode=r.returncode,wall_seconds=time.monotonic()-started,
        child_peak_rss_bytes=usage.ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        rss_scope='individual child (wait4)',
        user_seconds=usage.ru_utime,system_seconds=usage.ru_stime,
        expected_error=expected,log=str(log))
    if r.returncode==0 and save:
        result['av_sha256']=sha(av)
        if '--lora' in args:
            assert Path(str(av)+'.lora.json').is_file(), 'missing informational LoRA provenance'
    (out/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(name,json.dumps(result),flush=True)
    text=log.read_text(errors='replace')
    if expected:assert r.returncode and expected.lower() in text.lower(),text[-8000:]
    else:assert r.returncode==0,text[-8000:]
    return av

if 'baseline' in a.cases:
    first=run('baseline',selection()+settings()+preview())
    second=run('warm',selection()+settings()+preview())
    assert sha(first)==sha(second),'cold/warm request latents differ'
if 'resume' in a.cases:
    state=out/'pause.h3sample'
    run('pause',selection()+settings()+preview()+['--stop-after-step','3','--save-sampler-state',str(state)],save=False)
    assert state.is_file() and Path(str(state)+'.lora.json').is_file()
    assert not (out/'pause.mp4').exists()
    resumed=run('resumed',selection()+['--resume-sampler-state',str(state)]+preview())
    assert sha(resumed)==sha(out/'baseline.h3av'),'pause/resume latents differ'
    # Missing selection and offline folding retain exact-model rejection.
    run('missing-selection',['-d',a.model,'--resume-sampler-state',str(state)]+preview(),expected='model content fingerprint',save=False)
if 'ref-resume' in a.cases:
    state=out/'ref-pause.h3sample'
    args=selection()+settings(frames=22,steps=2,width=128,height=128)+['--ref-image',a.reference]+preview()
    complete=run('ref-complete',args)
    run('ref-pause',args+['--stop-after-step','1','--save-sampler-state',str(state)],save=False)
    resumed=run('ref-resumed',selection()+['--resume-sampler-state',str(state)]+preview())
    assert sha(complete)==sha(resumed),'saved Ref2VA mode or latents differ'
    assert 'LoRA Ref2VA' in (out/'ref-resumed.log').read_text()
if 'recovery' in a.cases:
    assert a.allow_evict_generated_variant,'recovery requires explicit cache eviction opt-in'
    baseline=out/'baseline.h3av';state=out/'pause.h3sample'
    assert baseline.is_file() and state.is_file(),'run baseline/resume first with the same binary'
    moved=Path(a.moved_directory) if a.moved_directory else out/'moved-inputs'
    moved.mkdir(parents=True,exist_ok=True)
    model=moved/'model';model.mkdir(exist_ok=True)
    for source in Path(a.model).resolve().iterdir():
        alias=model/source.name
        if not alias.exists():alias.symlink_to(source,target_is_directory=source.is_dir())
    adapter=moved/Path(a.turbo).name
    if not adapter.exists():shutil.copyfile(a.turbo,adapter)
    assert sha(adapter)==sha(a.turbo),'relocated adapter content differs'
    args=selection(str(adapter),str(model))
    relocated=run('moved-inputs',args+settings()+preview())
    assert sha(relocated)==sha(baseline),'relocated request latents differ'
    assert 'cache hit' in (out/'moved-inputs.log').read_text()
    identity=json.loads(Path(str(baseline)+'.lora.json').read_text())
    key=identity['runtime_fold_key']
    assert len(key)==64 and all(c in '0123456789abcdef' for c in key)
    entry=Path(a.cache)/'variants'/key
    manifest=json.loads((entry/'manifest.json').read_text())
    (out/'pre-eviction-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (Path(a.cache)/'locks'/(key+'.lock')).open('rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert entry.is_dir() and not entry.is_symlink()
        shutil.rmtree(entry)
    resumed=run('rebuilt-resumed',args+['--resume-sampler-state',str(state)]+preview())
    assert sha(resumed)==sha(baseline),'cold rebuilt checkpoint resume latents differ'
    after=json.loads((entry/'manifest.json').read_text())
    assert manifest['files']==after['files'],'rebuilt transformer bytes differ'
    rebuilt=run('rebuilt',args+settings()+preview())
    assert sha(rebuilt)==sha(baseline),'rebuilt request latents differ'
if any(case in a.cases for case in ['continuation','continuation-same','continuation-changed']):
    assert a.ordinary,'continuation suite needs --ordinary'
    for adapter,label in [(a.turbo,'same'),(a.ordinary,'changed')]:
        if 'continuation' not in a.cases and 'continuation-'+label not in a.cases:continue
        for mode in ['hard','bridge']:
            run('continue-'+label+'-'+mode,selection(adapter)+settings(frames=90,steps=4,width=256,height=256)+preview()+
                ['--continue-from',str(out/'baseline.h3av'),'--continue-mode',mode])
if 'rejections' in a.cases:
    assert a.ordinary
    state=out/'pause.h3sample'
    for label,adapter in [('adapter',a.ordinary),('strength',a.turbo+':0')]:
        run('reject-'+label,selection(adapter)+['--resume-sampler-state',str(state)]+preview(),
            expected='model content fingerprint',save=False)
    ordered=selection(a.turbo+':0')+['--lora',a.ordinary+':0']
    order_state=out/'order-pause.h3sample'
    run('order-pause',ordered+settings(frames=22,steps=2,width=128,height=128)+preview()+
        ['--stop-after-step','1','--save-sampler-state',str(order_state)],save=False)
    reversed_args=selection(a.ordinary+':0')+['--lora',a.turbo+':0']
    run('reject-order',reversed_args+['--resume-sampler-state',str(order_state)]+preview(),
        expected='model content fingerprint',save=False)
    changed=out/'changed-base';changed.mkdir(exist_ok=True)
    original=Path(a.model).resolve()
    for child in original.iterdir():
        if child.name!='FL2VA' and not (changed/child.name).exists():
            (changed/child.name).symlink_to(child,target_is_directory=child.is_dir())
    mode=changed/'FL2VA';mode.mkdir(exist_ok=True)
    for child in (original/'FL2VA').iterdir():
        if child.name!='transformer' and not (mode/child.name).exists():
            (mode/child.name).symlink_to(child,target_is_directory=child.is_dir())
    transformer=mode/'transformer';transformer.mkdir(exist_ok=True)
    for child in (original/'FL2VA'/'transformer').iterdir():
        target=transformer/child.name
        if child.name=='config.json':
            config=json.loads(child.read_text());config['runtime_lora_validation_marker']=1
            target.write_text(json.dumps(config)+'\n')
        elif not target.exists():target.symlink_to(child)
    changed_args=selection(a.turbo+':0',str(changed))+['--lora',a.ordinary+':0']
    run('reject-base',changed_args+['--resume-sampler-state',str(order_state)]+preview(),
        expected='model content fingerprint',save=False)
if 'legacy' in a.cases:
    models=[('no-lora',a.model)]
    if a.offline_model:models.append(('offline',a.offline_model))
    for label,model in models:
        state=out/(label+'-pause.h3sample')
        args=['-d',model]+settings(frames=22,steps=2,width=128,height=128)+preview()
        full=run(label+'-complete',args)
        run(label+'-pause',args+['--stop-after-step','1','--save-sampler-state',str(state)],save=False)
        resumed=run(label+'-resumed',['-d',model,'--resume-sampler-state',str(state)]+preview())
        assert sha(full)==sha(resumed),f'{label} legacy resume latents differ'
    if a.offline_model:
        run('reject-offline',['-d',a.offline_model,'--resume-sampler-state',str(out/'pause.h3sample')]+preview(),
            expected='model content fingerprint',save=False)
if 'quant' in a.cases:
    assert a.backend=='cuda' and a.quant_cache
    for mode in ['off','fp8','nvfp4']:
        args=selection()+settings(frames=22)+[]
        if mode!='off':args+=['--cuda-denoise-quant',mode,'--cuda-denoise-quant-cache',a.quant_cache]
        first=run('quant-'+mode,args+preview())
        second=run('quant-'+mode+'-warm',args+preview())
        assert sha(first)==sha(second),f'{mode} repeated latents differ'
        if mode=='nvfp4':
            full=run('quant-nvfp4-full',args)
            assert sha(first)==sha(full),'preview/full VAE changed denoising'
            state=out/'quant-pause.h3sample'
            run('quant-pause',args+preview()+['--stop-after-step','3','--save-sampler-state',str(state)],save=False)
            resume_args=selection()+['--resume-sampler-state',str(state),'--cuda-denoise-quant','nvfp4','--cuda-denoise-quant-cache',a.quant_cache]+preview()
            resumed=run('quant-resumed',resume_args)
            assert sha(first)==sha(resumed),'NVFP4 pause/resume latents differ'
if 'late-failure' in a.cases:
    key=json.loads(Path(str(out/'baseline.h3av')+'.lora.json').read_text())['runtime_fold_key']
    manifest=Path(a.cache)/'variants'/key/'manifest.json'
    before=manifest.read_bytes()
    # A directory at the media destination makes atomic publication fail only
    # after successful folding, denoising, and decoding.
    destination=out/'late-delivery-failure.mp4';destination.mkdir(exist_ok=True)
    try:
        run('late-delivery-failure',selection()+settings(frames=22,steps=2,width=128,height=128)+
            preview(),expected='cannot publish completed media',save=False)
        assert 'cache hit' in (out/'late-delivery-failure.log').read_text()
        assert manifest.read_bytes()==before,'later delivery failure changed a valid folded cache'
        assert destination.is_dir() and not list(destination.iterdir())
    finally:
        destination.rmdir()
if 'large' in a.cases:
    assert a.backend=='cuda' and a.quant_cache
    run('1344x768-turbo8-nvfp4-preview',selection()+settings(frames=243,width=1344,height=768)+preview()+
        ['--cuda-denoise-quant','nvfp4','--cuda-denoise-quant-cache',a.quant_cache])
if 'ordinary' in a.cases:
    assert a.ordinary
    first=run('ordinary',selection(a.ordinary)+settings(frames=22,steps=20)+preview())
    second=run('ordinary-warm',selection(a.ordinary)+settings(frames=22,steps=20)+preview())
    assert sha(first)==sha(second),'ordinary cold/warm latents differ'
if 'ordinary-ref' in a.cases:
    assert a.ordinary
    run('ordinary-ref',selection(a.ordinary)+settings(frames=22,steps=20,width=128,height=128)+
        ['--ref-image',a.reference]+preview())
if 'combined' in a.cases:
    assert a.ordinary
    args=selection()+['--lora',a.ordinary+':0.6']+settings(frames=22)+preview()
    first=run('combined',args)
    second=run('combined-warm',args)
    assert sha(first)==sha(second),'combined cold/warm latents differ'
    if a.backend=='cuda' and a.quant_cache:
        run('combined-nvfp4',args+['--cuda-denoise-quant','nvfp4','--cuda-denoise-quant-cache',a.quant_cache])
