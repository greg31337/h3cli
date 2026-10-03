#!/usr/bin/env python3
"""Bounded reference-cache integration, exact resume and state corruption.

Every subprocess performs at most six denoiser evaluations. Expected policy,
condition immutability, and resume bytes are checked independently of helpers.
"""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time
from test_sampler_file import entries,build


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--media',type=Path,required=True);a=parser.parse_args()
    source=Path(__file__).resolve().parents[1];out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    assets=out/'inputs';assets.mkdir()
    for src in [source/'inputs/1.jpg',source/'inputs/2.jpg',*[a.media/name for name in ('embedded.mp4','replacement.wav','separate.wav')]]:shutil.copy2(src,assets/src.name)
    image,image2=assets/'1.jpg',assets/'2.jpg';video=assets/'embedded.mp4';audio=assets/'separate.wav';replacement=assets/'replacement.wav'
    image_refs=[str(image)];embedded=['video:'+str(video)];silent=['silent:'+str(video)]
    pair=['paired:'+str(video)+'|'+str(replacement)]
    audio_refs=[str(image),'audio:'+str(audio)]
    mixed=[str(image),*silent,*pair,'audio:'+str(audio)]
    base_env=dict(os.environ,H3_TEST_MAX_EVALUATIONS='6',H3_EXPERIMENT_TRACE='1',H3_TEST_SCHEDULE_STEPS='6')
    records=[];checks=[];binary=source/'bin/adaptive_latent';identity=sha(binary)

    def publish():
        (out/'result.json').write_text(json.dumps(dict(complete=False,passed=False,binary_sha256=identity,jobs=records,checks=checks),indent=2)+'\n')

    def run(name,refs,stop=6,sparse=False,mode='conservative',threshold='1',hits='2',size='max',resume=None,extra=None,shape=(256,256,22),success=True):
        ref_file=out/(name+'.references');ref_file.write_text('\n'.join(refs)+'\n')
        env=base_env|{'H3_TEST_REFERENCES':str(ref_file),'H3_TEST_ADAPTIVE_THRESHOLD':threshold,'H3_TEST_ADAPTIVE_MAX_HITS':hits}
        if mode=='off':env.pop('H3_TEST_ADAPTIVE_THRESHOLD');env.pop('H3_TEST_ADAPTIVE_MAX_HITS')
        env|=extra or {}
        cmd=[str(binary),str(a.model.resolve()),str(out/name),mode,str(stop),str(resume) if resume else '-', '.75' if sparse else '-',*map(str,shape),'off','-','2' if mode!='off' else '0','3' if sparse else '0','-','-','-',size]
        start=time.monotonic()
        with (out/(name+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        text=(out/(name+'.log')).read_text();decisions=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('h3cli: adaptive step=')]
        trace=[json.loads(l.split('h3_experiment ',1)[1]) for l in text.splitlines() if l.startswith('h3_experiment ')]
        record=dict(name=name,argv=cmd,environment={k:v for k,v in env.items() if k.startswith('H3_TEST_')},seconds=time.monotonic()-start,returncode=r.returncode,passed=(r.returncode==0)==success,decisions=decisions,trace=trace)
        records.append(record);publish();print(name,record['passed'],round(record['seconds'],2),flush=True)
        if not record['passed']:raise RuntimeError(name+' failed; see log')
        return out/(name+'.h3sample')

    def check(name,ok):
        checks.append(dict(name=name,passed=bool(ok)));publish()
        if not ok:raise AssertionError(name)

    def policy(record):
        for d,t in zip(record['decisions'],record['trace']):
            check(record['name']+' selected score '+d['step'],float(d['score'])==max(float(d[k]) for k in ('global_score','video_score','audio_score')))
            if d['decision']=='hit':check(record['name']+' hit dispatch '+d['step'],t['blocks']==1 and not t['sparse_calls'] and not t['router_calls'] and t.get('stream_read_layers',0)<=1)

    def state_parts(path):return {p[0]:p for p in entries(path.read_bytes())}

    # Both attention paths across all four reference categories; actual hit,
    # warmup/phase/final and resume boundaries in the original six-step schedule.
    for label,refs,size in [('image',image_refs,'max'),('video',embedded,'match'),('audio',audio_refs,'max'),('mixed',mixed,'high')]:
        for sparse in (False,True):
            name=label+('-subblock' if sparse else '-dense')
            full=run(name+'-full',refs,sparse=sparse,size=size);full_record=records[-1];policy(full_record)
            check(name+' actual hits',any(d['decision']=='hit' for d in full_record['decisions']))
            check(name+' final refresh',full_record['decisions'][-1]['reason']=='final')
            paused=run(name+'-pause',refs,stop=3,sparse=sparse,size=size)
            before=state_parts(paused)
            hidden=out/'hidden-inputs';assets.rename(hidden)
            try:resumed=run(name+'-resume',[],sparse=sparse,size=size,resume=paused)
            finally:hidden.rename(assets)
            after=state_parts(resumed);direct=state_parts(full)
            for kind in (12,13,40,41,42):check(name+f' exact final state section {kind}',after[kind][-1]==direct[kind][-1])
            for kind in (7,8):
                if kind in before:check(name+f' fixed condition section {kind}',before[kind][-1]==after[kind][-1])
            for step in range(4,7):check(name+f' exact resumed AV step {step}',sha(out/(name+f'-full.step-{step:02d}.bin'))==sha(out/(name+f'-resume.step-{step:02d}.bin')))
            check(name+' resumed decisions',records[-1]['decisions']==full_record['decisions'][3:])
            # Mismatch must precede model loading in the public CLI path.
            for flag,value in [('--adaptive-cache-threshold','0'),('--adaptive-cache-max-hits','16')]:
                r=subprocess.run([str(source/'bin/h3cli'),'-d','/missing-acr-model','--resume-sampler-state',str(paused),flag,value],cwd=source,capture_output=True,text=True)
                check(name+' resume mismatch '+flag,r.returncode!=0 and 'differs from checkpoint' in r.stderr and 'loading model' not in r.stderr.lower())

    # Additional representative encodes, default and zero controls, count/size,
    # soundtrack modes, cancellation and forced streaming.
    run('two-images-high',[str(image),str(image2)],stop=2,sparse=True,size='high')
    run('nine-images-match',[str(image if i%2==0 else image2) for i in range(9)],stop=1,size='match')
    run('silent-video',silent,stop=3,mode='aggressive',size='match')
    run('replacement-video',pair,stop=3,sparse=True,size='match')
    stripped=assets/'no-embedded.mp4'
    subprocess.run([os.environ.get('H3_SGLANG_INPUT_FFMPEG','ffmpeg'),'-nostdin','-v','error','-i',str(video),'-an','-c:v','copy',str(stripped)],check=True)
    original_silent=run('silent-original-zero',silent,stop=0,size='match')
    stripped_silent=run('silent-stripped-zero',['silent:'+str(stripped)],stop=0,size='match')
    original_pair=run('replacement-original-zero',pair,stop=0,size='match')
    stripped_pair=run('replacement-stripped-zero',['paired:'+str(stripped)+'|'+str(replacement)],stop=0,size='match')
    for kind in (7,8):
        check('silent ignores embedded soundtrack '+str(kind),state_parts(original_silent)[kind][-1]==state_parts(stripped_silent)[kind][-1])
        check('replacement excludes embedded soundtrack '+str(kind),state_parts(original_pair)[kind][-1]==state_parts(stripped_pair)[kind][-1])
    check('silent has no audio prefix',not state_parts(original_silent)[8][-1])
    run('invalid-video',['video:'+str(image)],stop=0,size='match',success=False)
    run('missing-video',['video:'+str(assets/'missing.mp4')],stop=0,size='match',success=False)
    run('invalid-audio-context-recovery',image_refs,stop=3,extra={'H3_TEST_BAD_AUDIO':str(image)})
    for step in range(1,4):check('invalid audio recovery exact '+str(step),sha(out/f'invalid-audio-context-recovery.step-{step:02d}.bin')==sha(out/f'image-dense-full.step-{step:02d}.bin'))
    run('video-separate-audio',silent+['audio:'+str(audio)],stop=3,size='match')
    run('multiple-audio',[str(image),'audio:'+str(audio),'audio:'+str(replacement)],stop=3,sparse=True,size='match')
    zero=run('threshold-zero',image_refs,threshold='0',hits='16')
    check('zero always refreshes',all(d['decision']=='refresh' for d in records[-1]['decisions']))
    dense=run('uncached-control',image_refs,mode='off')
    for kind in (12,13):check('zero exact dense latent '+str(kind),state_parts(zero)[kind][-1]==state_parts(dense)[kind][-1])
    run('cancel-retry',image_refs,stop=3,extra={'H3_TEST_CANCEL_ONCE':'1'})
    for step in range(1,4):check('cancellation exact retry '+str(step),sha(out/f'cancel-retry.step-{step:02d}.bin')==sha(out/f'image-dense-full.step-{step:02d}.bin'))
    run('context-budget-recovery',image_refs,stop=2,extra={'H3_TEST_CONTEXT_BUDGET':'1'})
    run('forced-streamed-image',image_refs,extra={'H3_CUDA_WEIGHT_MODE':'stream'})
    for step in range(1,7):check('streamed image exact '+str(step),sha(out/f'forced-streamed-image.step-{step:02d}.bin')==sha(out/f'image-dense-full.step-{step:02d}.bin'))
    run('forced-streamed-mixed',mixed,sparse=True,size='high',extra={'H3_CUDA_WEIGHT_MODE':'stream'})
    for step in range(1,7):check('streamed mixed exact '+str(step),sha(out/f'forced-streamed-mixed.step-{step:02d}.bin')==sha(out/f'mixed-subblock-full.step-{step:02d}.bin'))
    run('large-image-capacity',image_refs,stop=3,shape=(1344,768,124),extra={'H3_TEST_SCHEDULE_STEPS':'50'})
    # The loader runs without source media and rejects rechecksummed forgeries.
    parts=entries((out/'mixed-subblock-pause.h3sample').read_bytes())
    forged=out/'forged.h3sample'
    mutations=[(40,4,'I',1),(40,44,'f',float('nan')),(40,48,'i',17),(40,12,'I',16),(40,24,'i',-1),
               (32,0,'I',0),(40,28,'Q',2**63)]
    for kind,offset,fmt,value in mutations:
        changed=entries(build(parts));struct.pack_into('<'+fmt,next(p[-1] for p in changed if p[0]==kind),offset,value);forged.write_bytes(build(changed))
        r=subprocess.run([str(source/'bin/sampler_tests'),'--load',str(forged)],cwd=source,capture_output=True,text=True)
        check(f'forged {kind}/{offset} rejected',r.returncode!=0)
    # Existing independent corruption campaign covers required sections/bounds.
    bounded=out/'bounded-corruption.h3sample'
    with (out/'bounded-corruption.log').open('w') as log:
        subprocess.run([str(source/'bin/sampler_tests'),'--adaptive-reference-fixture',str(bounded)],cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run(['python3',str(source/'tests/test_approximate_state.py'),str(source/'bin/sampler_tests'),str(bounded),str(out/'corruption.json')],cwd=source,check=True)
    result=dict(complete=True,passed=all(r['passed'] for r in records) and all(c['passed'] for c in checks),binary_sha256=identity,jobs=records,checks=checks)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS bounded reference integration',len(records),len(checks));return 0

if __name__=='__main__':raise SystemExit(main())
