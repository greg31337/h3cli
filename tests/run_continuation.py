#!/usr/bin/env python3
"""Resumable real-checkpoint acceptance matrix. Run from any directory.

Each render uses 20 steps and all 50 blocks by default. No mock weights. Raw
RGB/PCM, F32 states, command lines and per-case logs stay in the output folder.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PROMPT = 'A person turns toward the camera and waves, natural daylight, steady camera, quiet outdoor ambience.'
CHAIN_PROMPT = 'The person in <Picture 1> and <Picture 2> walks slowly through a sunlit garden, turns toward the camera, and waves. Natural skin tone, green background, steady tracking camera, quiet outdoor ambience.'

def cases(out, model, steps):
    def state(name):
        return str(out / (name + '.h3av')) if name else '-'
    def item(name, mode, source=None, frames=90, seed=43, keep=0, prompt=CHAIN_PROMPT, reuse=1, video=None, context=39):
        cmd = [str(ROOT / 'bin/continuation_generate'), model, str(out/name), mode, state(source),
               str(seed), str(frames), str(steps), str(reuse), str(keep), prompt,
               str(video or ROOT/'outputs/continuation-validation/reference.mp4'), str(context)]
        return name, cmd
    result = []
    for label, mode in [('t2va','t2va'), ('fl2va','fl2va'), ('image-ref2va','image1'), ('video-ref2va','video'), ('audio-ref2va','audio')]:
        result.append(item('regression-'+label,mode,frames=56,seed=42,prompt=PROMPT))
    result += [item('t2va-1','t2va',seed=41,prompt=PROMPT), item('t2va-2','t2va','t2va-1',prompt=PROMPT)]
    result.append(item('native-1','image1',seed=42))
    for n in range(2,6):
        result.append(item(f'native-{n}','image1',f'native-{n-1}',seed=41+n))
    for n in range(2,6):
        previous = 'native-1' if n==2 else f'recursive-{n-1}'
        result.append(item(f'recursive-{n}','recursive',seed=41+n,video=out/(previous+'.mp4')))
    result += [item('trim-audit','image1','native-1',seed=43),
               item('changed-reference','image2','native-1',seed=43),
               item('changed-prompt','image1','native-1',seed=43,
                    prompt='The person in <Picture 1> and <Picture 2> stops walking, raises both arms, and turns left as the camera moves around them. Sunlit garden, natural skin tone and quiet outdoor ambience.')]
    for mode in ['imagevideo','imageaudio','video','replace']:
        result.append(item('mixed-'+mode,mode,'native-1'))
    for reuse in [2,3]:
        result.append(item(f'reuse-{reuse}','image1','native-1',reuse=reuse))
    return result

def prepare_fixtures():
    fixture=ROOT/'outputs/continuation-validation/reference.mp4'
    if not fixture.exists():
        fixture.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-loop','1','-i',str(ROOT/'inputs/body1.jpg'),
                        '-f','lavfi','-i','sine=frequency=220:sample_rate=32000','-vf','scale=256:256',
                        '-t','3','-r','24','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',str(fixture)],check=True)
    waveform=fixture.with_suffix('.wav')
    if not waveform.exists():
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(fixture),
                        '-ar','32000','-ac','2',str(waveform)],check=True)
    replacement=fixture.with_name('replacement.wav')
    if not replacement.exists():
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','lavfi','-i',
                        'sine=frequency=440:sample_rate=32000','-t','3','-ac','2',str(replacement)],check=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=str(ROOT/'models/MiniMax-H3'))
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/continuation-validation/acceptance')
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--only',help='Comma-separated case names (dependencies must exist)')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--list',action='store_true')
    args=parser.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    matrix=cases(out,args.model,args.steps)
    if args.only:
        wanted=set(args.only.split(',')); matrix=[x for x in matrix if x[0] in wanted]
        assert wanted=={x[0] for x in matrix},'unknown case'
    if args.list:
        print('\n'.join(name for name,_ in matrix)); return
    prepare_fixtures()
    subprocess.run(['make','-j8','bin/continuation_generate'],cwd=ROOT,check=True)
    manifest=out/'runs.json'
    runs=json.loads(manifest.read_text()) if args.resume and manifest.exists() else {}
    env=os.environ.copy(); env['H3_PROFILE']='1'
    for name,cmd in matrix:
        previous=runs.get(name,{})
        if args.resume and previous.get('returncode')==0 and previous.get('command')==cmd:
            continue
        print(f'running {name}',flush=True); start=time.monotonic()
        case_env=env.copy()
        if name=='trim-audit': case_env['H3_TEST_KEEP_PAIR']=str(out/'debug-prefix')
        with (out/(name+'.log')).open('w') as log:
            run=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=log,env=case_env)
        record={'command':cmd,'returncode':run.returncode,'seconds':time.monotonic()-start,
                'binary_sha256':hashlib.sha256(Path(cmd[0]).read_bytes()).hexdigest()}
        if run.returncode==0:
            for extension in ['mp4','h3av','rgb','pcm']:
                record[extension+'_sha256']=hashlib.sha256((out/(name+'.'+extension)).read_bytes()).hexdigest()
        runs[name]=record
        temporary=manifest.with_suffix('.tmp'); temporary.write_text(json.dumps(runs,indent=2)); temporary.replace(manifest)
        print(f'{name}: exit={run.returncode}, {record["seconds"]:.1f}s',flush=True)
        if run.returncode:
            print((out/(name+'.log')).read_text()[-3000:]); raise SystemExit(run.returncode)
    baseline=ROOT/'outputs/continuation-validation/baseline/results.json'
    if baseline.exists():
        frozen_path=ROOT/'outputs/continuation-validation/frozen-regression/results.json'
        frozen=json.loads(frozen_path.read_text()) if frozen_path.exists() else {}
        comparison={}
        for name,entry in json.loads(baseline.read_text()).items():
            current=runs.get('regression-'+name)
            if current and current.get('returncode')==0:
                exact=current['mp4_sha256']==entry['sha256']
                replay=frozen.get(name,{}).get('passed',False)
                comparison[name]={'cold_output_exact':exact,'fixed_conditioning_exact':replay}
                assert exact or replay, (f'non-continuation regression unresolved: {name}; '
                                         'run tests/continuation_regression.py to isolate existing encoder variability')
        (out/'baseline-comparison.json').write_text(json.dumps(comparison,indent=2))
        print('ok: available baseline cases match directly or with identical captured conditioning',flush=True)

if __name__=='__main__':
    main()
