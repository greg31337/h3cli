#!/usr/bin/env python3
"""Bounded replay boundaries, placement, delivery, chains and baseline checks."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import json
from pathlib import Path
import re
import struct
import subprocess
import time

import numpy as np
from adaptive_continuation_integration import parts, sha
from bridge_metrics import diagnostics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('model','integration','baseline','campaign','out'): p.add_argument('--'+flag,type=Path,required=True)
    a=p.parse_args(); root=Path(__file__).resolve().parents[1]; out=a.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    model=str(a.model.resolve()); data=a.integration.resolve(); baseline=a.baseline.resolve(); campaign=a.campaign.resolve()
    env=os.environ|{'H3_TEST_MAX_EVALUATIONS':'6','H3_TEST_SCHEDULE_STEPS':'6','H3_EXPERIMENT_TRACE':'1','H3_VERBOSE':'1'}
    records=[];checks=[]
    def publish(complete=False):
        (out/'result.json').write_text(json.dumps(dict(complete=complete,passed=complete and all(r['passed'] for r in records+checks),
            jobs=records,checks=checks,binary_sha256=sha(root/'bin/h3cli')),indent=2)+'\n')
    def check(name,ok):
        checks.append(dict(name=name,passed=bool(ok)));publish()
        if not ok: raise AssertionError(name)
    def job(name,argv,extra=None,expected=0):
        start=time.monotonic()
        with (out/(name+'.log')).open('w') as log:
            result=subprocess.run(list(map(str,argv)),cwd=root,env=env|(extra or {}),stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        record=dict(name=name,argv=list(map(str,argv)),overrides=extra or {},seconds=time.monotonic()-start,
                    returncode=result.returncode,passed=(result.returncode==0 if expected==0 else result.returncode!=0))
        records.append(record);publish();print(name,record['passed'],round(record['seconds'],2),flush=True)
        if not record['passed']: raise RuntimeError(name+' failed')
        return (out/(name+'.log')).read_text()
    def same(label,left,right,kinds=(12,13,40,41,42)):
        x,y=parts(left),parts(right)
        for kind in kinds:
            if kind in x: check(label+f' section {kind}',x[kind][-1]==y[kind][-1])
    def latent(name,cache='conservative',stop=6,resume=None,sparse=True,source=None,mode='hard',refs=None,extra=None,shape=(384,384,124),size='max'):
        more={'H3_TEST_ADAPTIVE_THRESHOLD':'1','H3_TEST_ADAPTIVE_MAX_HITS':'2'} if cache!='off' else {}
        if source and not resume: more|={'H3_TEST_CONTINUE_FROM':str(source),'H3_TEST_CONTINUE_MODE':mode}
        if refs and not resume:
            ref_file=out/(name+'.references');ref_file.write_text('\n'.join(map(str,refs))+'\n');more['H3_TEST_REFERENCES']=str(ref_file)
        more|=extra or {}
        argv=[root/'bin/adaptive_latent',model,out/name,cache,stop,resume or '-', '.75' if sparse else '-',
              *shape,'off','-','2' if cache!='off' else '0','3' if sparse else '0','-','-','-',size]
        text=job(name,argv,more)
        for line in text.splitlines():
            if line.startswith('h3_experiment '):
                t=json.loads(line.split(' ',1)[1])
                if t['blocks']==1: check(name+f' hit traffic {t["step"]}',t['stream_read_layers']<=1 and t['sparse_calls']==t['router_calls']==0)
        return out/(name+'.h3sample')

    source=data/'inputs/source-text.h3av'; ref_source=data/'inputs/source-ref.h3av'; image=data/'inputs/1.jpg'
    # Old-default recipes 1/3 and dense hard/bridge trajectories captured before edits.
    original=json.loads((baseline/'result.json').read_text())
    for old in original:
        if old['name']=='build': continue
        name='baseline-'+old['name']; argv=[str(v).replace(str(baseline.relative_to(root)),str(out.relative_to(root))) for v in old['argv']]
        # Replace output basename separately, retaining the exact original prompt/seed/settings.
        argv=[v.replace(str(out.relative_to(root)/old['name']),str(out/name)) for v in argv]
        extras={'H3_TEST_ADAPTIVE_THRESHOLD':'1','H3_TEST_ADAPTIVE_MAX_HITS':'2'} if old['name']=='reference' else {}
        job(name,argv,extras)
        same(name,baseline/(old['name']+'.h3sample'),out/(name+'.h3sample'))
        old_log=(baseline/(old['name']+'.log')).read_text();new_log=(out/(name+'.log')).read_text()
        decisions=lambda text:[s for s in text.splitlines() if s.startswith('h3cli: adaptive step=')]
        check(name+' decisions',decisions(old_log)==decisions(new_log))
    # A preserved pre-change binary provides a bounded ordinary recipe-2 control.
    old_binary=baseline.parent/'baseline-gate/build/bin/h3cli'
    quant_flags=['-d',model,'-p','A woman plays piano in warm sunlight with flowing piano music.',
                 '--width','256','--height','256','--frames','22','--steps','6','--seed','42',
                 '--adaptive-cache','conservative','--adaptive-cache-warmup','2','--cuda-denoise-quant','fp8','--stop-after-step','4']
    for label,binary in [('before',old_binary),('after',root/'bin/h3cli')]:
        job('recipe2-'+label,[binary,*quant_flags,'--cuda-denoise-quant-cache',out/'quant-cache',
            '--save-sampler-state',out/('recipe2-'+label+'.h3sample'),'-o',out/(label+'.mp4')])
    same('recipe2 preserved',out/'recipe2-before.h3sample',out/'recipe2-after.h3sample')
    # Resume on either side of warmup, actual hit streak and sparse transition,
    # including the last pending transition. Original media are hidden.
    for case,boundaries in [('hard-adaptive',(1,4)),('bridge-combined',(2,4,5))]:
        config=dict(source=source,mode=case.split('-')[0],sparse=case.endswith('combined'))
        direct=latent(case+'-current-full',**config)
        same(case+' unchanged after help/tooling',direct,data/(case+'-full.h3sample'))
        for stop in boundaries:
            name=f'{case}-at{stop}';paused=latent(name,stop=stop,**config)
            assets=data/'inputs';hidden=data/'hidden-supplement-inputs';assets.rename(hidden)
            try: resumed=latent(name+'-resume',resume=paused,**config)
            finally: hidden.rename(assets)
            same(name,resumed,direct)
    mixed=(data/'mixed-bridge-full.references').read_text().splitlines()
    config=dict(source=ref_source,mode='bridge',refs=mixed,size='high')
    direct=latent('mixed-current-full',**config);paused=latent('mixed-current-pause',stop=3,**config)
    assets=data/'inputs';hidden=data/'hidden-mixed-inputs';assets.rename(hidden)
    try: resumed=latent('mixed-current-resume',resume=paused,**config)
    finally: hidden.rename(assets)
    same('mixed current replay',direct,resumed)
    same('mixed preserved after reader validation',direct,data/'mixed-bridge-full.h3sample')
    # CLI policy restoration and rejection must happen before a missing model is loaded.
    state=out/'bridge-combined-at2.h3sample'
    common=[root/'bin/h3cli','-d','/missing-continuation-model','--resume-sampler-state',state,'-o',out/'unused.mp4']
    for name,flags,rejected in [('omitted',[],False),('matching',['--adaptive-cache-threshold','1','--adaptive-cache-max-hits','2','--adaptive-cache-warmup','2','--subblock-warmup','3'],False),
                              ('threshold',['--adaptive-cache-threshold','.04'],True),('streak',['--adaptive-cache-max-hits','1'],True),
                              ('warmup',['--adaptive-cache-warmup','3'],True),('budget-small',['--adaptive-cache-max-mib','1'],True),
                              ('budget-large',['--adaptive-cache-max-mib','8192'],False)]:
        text=job('cli-'+name,common+flags,expected=1)
        policy='differs from checkpoint' in text or 'minimum --adaptive-cache-max-mib' in text
        check('CLI '+name+' before model',policy==rejected)
    for case,stop,flags in [('hard-adaptive',1,[]),('bridge-combined',2,
            ['--adaptive-cache-threshold','1','--adaptive-cache-max-hits','2','--adaptive-cache-warmup','2','--subblock-warmup','3'])]:
        name='cli-resume-'+case;assets=data/'inputs';hidden=data/'hidden-cli-inputs';assets.rename(hidden)
        try:
            job(name,[root/'bin/h3cli','-d',model,'--resume-sampler-state',out/f'{case}-at{stop}.h3sample',
                '--save-sampler-state',out/(name+'.h3sample'),'-o',out/(name+'.mp4'),*flags])
        finally: hidden.rename(assets)
        same(name,out/(name+'.h3sample'),out/(case+'-current-full.h3sample'))
    for mode,extra in [('resident',{'H3_CUDA_WEIGHT_MODE':'resident'}),('partial',{'H3_TEST_CUDA_RESIDENT_BLOCKS':'8'}),('stream',{'H3_CUDA_WEIGHT_MODE':'stream'})]:
        state=latent('placement-'+mode,source=source,extra=extra)
        same(mode+' exact placement',state,data/'hard-combined-full.h3sample')
    # A changed policy/media/source request and multiple recoveries on one live context.
    job('retained-context',[root/'bin/adaptive_continuation_context',model,ref_source,image,data/'inputs/2.jpg'])
    live=job('live-stream-context',[root/'bin/adaptive_continuation_context',model,ref_source,image,data/'inputs/2.jpg','live'])
    check('actual live prepared DiT reuse',live.count('prepared DiT cache hit')==4)
    # Exact normal/debug delivery suffix, both continuation modes, cached context.
    for mode in ('hard','bridge'):
        normal=out/(mode+'-trim');debug=out/(mode+'-debug')
        text=job(mode+'-delivery',[root/'bin/continuation_generate',model,normal,'reference',ref_source,'42','124','6','1','0',
             'A woman plays piano in warm sunlight with flowing piano music.','-','39','384',mode,'stepped','8','.5'],
            {'H3_TEST_IMAGE':str(image),'H3_TEST_ADAPTIVE_MODE':'conservative','H3_TEST_SUBBLOCK':'1',
             'H3_TEST_KEEP_PAIR':str(debug),'H3_BRIDGE_DIAGNOSTICS':'1'})
        if mode=='bridge':
            measured=diagnostics(text,6,2)
            (out/'bridge-velocity.json').write_text(json.dumps(measured,indent=2)+'\n')
            check('bridge velocity scaled exactly once',measured['max_velocity_scale_error']<2e-6)
        check(mode+' keep-prefix AV',normal.with_suffix('.h3av').read_bytes()==debug.with_suffix('.h3av').read_bytes())
        pixels=np.fromfile(debug.with_suffix('.rgb'),dtype='u1').reshape(124,384,384,3)
        check(mode+' RGB suffix',normal.with_suffix('.rgb').read_bytes()==pixels[39:].tobytes())
        sound=np.fromfile(debug.with_suffix('.pcm'),dtype='<f4').reshape(2,-1)
        check(mode+' PCM suffix',normal.with_suffix('.pcm').read_bytes()==sound[:,52000:].tobytes())
    previous=source
    for index,mode in enumerate(('hard','bridge','hard'),1):
        saved=out/f'chain-{index}.h3av'
        state=latent('chain-'+str(index),source=previous,mode=mode,extra={'H3_TEST_SAVE_AV':str(saved)})
        log=(out/f'chain-{index}.log').read_text()
        first=next(s for s in log.splitlines() if s.startswith('h3cli: adaptive step='))
        check(f'chain {index} empty cache','reason=empty' in first)
        job(f'chain-{index}-decode',[root/'bin/h3cli','-d',model,'--decode-av-state',saved,'-o',out/f'chain-{index}.mp4'])
        previous=saved
    ffmpeg=os.environ.get('H3_SGLANG_INPUT_FFMPEG','ffmpeg')
    job('chain-review',[ffmpeg,'-nostdin','-v','error','-i',out/'chain-1.mp4','-i',out/'chain-2.mp4','-i',out/'chain-3.mp4',
        '-filter_complex','[0:v][0:a][1:v][1:a][2:v][2:a]concat=n=3:v=1:a=1[v][a]',
        '-map','[v]','-map','[a]','-c:v','libx264','-crf','18','-c:a','aac',out/'chain-review.mp4'])
    # Large reference geometry: one real transition, allocation/planning and a
    # current checkpoint. This is capacity evidence, not a full 362-frame render.
    large=campaign/'S02.h3av'
    state=latent('capacity362',stop=1,source=large,refs=[image],shape=(1344,768,362))
    record=parts(state);check('capacity362 committed ready',int.from_bytes(record[40][-1][8:12],'little')==1)
    _,vt,lh,lw,at=struct.unpack_from('<5i',record[10][-1],48)
    vp,ap=struct.unpack_from('<2i',record[10][-1],68)
    def capacity_rows(step):
        raw=(out/f'capacity362.step-{step:02d}.bin').read_bytes();nv,na=struct.unpack_from('<2Q',raw)
        values=np.frombuffer(raw,dtype='<u4',offset=16)
        check(f'capacity362 finite step {step}',np.isfinite(values.view('<f4')).all())
        return values[:nv].reshape(24,vt,lh,lw),values[nv:nv+na].reshape(64,at)
    before,after=capacity_rows(0),capacity_rows(1)
    check('capacity362 exact video prefix',np.array_equal(before[0][:,:vp],after[0][:,:vp]))
    check('capacity362 exact stereo prefix',np.array_equal(before[1][:,:ap],after[1][:,:ap]))
    check('capacity362 checkpoint bounded',state.stat().st_size<16*1024**3)
    publish(True);print('PASS continuation supplement',len(records),len(checks),flush=True)


if __name__=='__main__': main()
