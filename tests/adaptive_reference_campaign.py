#!/usr/bin/env python3
"""Frozen BF16 reference-cache comparison matrix. No numerical golden updates.

Prepare once from existing qualified media, then run the exact 18 manifest rows.
Only V01--V04 lift the six-evaluation ceiling, for their specified 50-step videos.
"""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import array
import hashlib
import html
import json
import math
from pathlib import Path
import re
import subprocess
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def ffmpeg():return os.environ.get('H3_SGLANG_INPUT_FFMPEG','ffmpeg')
def ffprobe():return os.environ.get('H3_FFPROBE','ffprobe')
def pcm(path):
    data=subprocess.check_output([ffmpeg(),'-nostdin','-v','error','-i',str(path),'-map','0:a:0','-ar','32000','-ac','2','-f','f32le','-'])
    values=array.array('f');values.frombytes(data)
    if not values or not all(math.isfinite(x) for x in values):raise ValueError('empty/nonfinite stereo audio')
    return values


def prepare(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    media=out/'inputs';media.mkdir()
    source=a.source.resolve();model=a.model.resolve();commands=[]
    sources=[a.video.resolve(),a.replacement.resolve(),a.audio.resolve()]
    video,replacement,audio=[media/name for name in ('embedded.mp4','replacement.wav','separate.wav')]
    for src,dst in zip(sources,(video,replacement,audio)):
        cmd=[ffmpeg(),'-nostdin','-v','error','-i',str(src),'-t','2']
        cmd+=['-map','0:v:0','-map','0:a:0','-r','24','-c:v','libx264','-crf','18','-preset','fast','-c:a','aac','-ar','32000','-ac','2'] if dst==video else ['-map','0:a:0','-vn','-ar','32000','-ac','2','-c:a','pcm_s16le']
        cmd += [str(dst)];subprocess.run(cmd,check=True);commands.append(cmd)
    audio_hashes=[hashlib.sha256(pcm(p).tobytes()).hexdigest() for p in (video,replacement,audio)]
    if len(set(audio_hashes))!=3:raise ValueError('soundtracks must be distinguishable')
    images=[source/'inputs/1.jpg',source/'inputs/2.jpg']
    image=['--ref-image',str(images[0]),'--ref-image-size','max']
    two=['--ref-image',str(images[0]),'--ref-image',str(images[1]),'--ref-image-size','high']
    embedded=['--ref-video',str(video)]
    separate=image+['--ref-audio',str(audio)]
    mixed=['--ref-image',str(images[0]),'--ref-silent-video',str(video),'--ref-video-audio',str(video),str(replacement),'--ref-audio',str(audio),'--ref-image-size','high']
    beach='The woman in the white dress walks slowly along the beach, turns toward the camera and smiles. Her hair and dress move gently in the sea breeze. Soft ocean waves and natural outdoor ambience.'
    pair='The two women walk together along a sunny coastal garden path, smiling and talking softly as the camera glides sideways. Gentle wind and soft footsteps.'
    garden='A person explores a sunny garden, moving naturally as the camera slowly glides sideways. Warm daylight and soft outdoor ambience inspired by the reference sound.'
    rows=[]
    custom=['--adaptive-cache','conservative','--adaptive-cache-threshold','0.12','--adaptive-cache-max-hits','2','--adaptive-cache-warmup','2']
    sparse=lambda n:['--cuda-attention','subblock','--subblock-sparsity','0.75','--subblock-warmup',str(n)]
    policies=[[],['--adaptive-cache','conservative'],['--adaptive-cache','aggressive'],['--adaptive-cache','conservative','--adaptive-cache-threshold','0.06','--adaptive-cache-max-hits','2','--adaptive-cache-warmup','4']]
    settings=[(640,480,90,50,image,beach,1,p) for p in policies]
    settings += [(1344,768,124,6,image,beach,5,p) for p in ([],custom,custom+sparse(2))]
    settings += [(640,480,90,6,two,pair,8,p) for p in ([],custom+sparse(3))]
    for refs,prompt,control in [(embedded,garden,10),(separate,beach,13),(mixed,garden,16)]:
        settings += [(640,480,90,6,refs,prompt,control,p) for p in ([],custom,custom+sparse(3))]
    for i,(w,h,frames,steps,refs,prompt,control,policy) in enumerate(settings,1):
        name=f'V{i:02d}';base=out/name
        cmd=[str(source/'bin/h3cli'),'-d',str(model),'-p',prompt,'--width',str(w),'--height',str(h),'--frames',str(frames),'--steps',str(steps),'--seed','42','--reuse','1','--core-reuse','1','--cuda-weight-mode','auto']
        cmd+=refs+policy+['--save-av-state',str(base.with_suffix('.h3av')),'--save-sampler-state',str(base.with_suffix('.h3sample')),'--profile','-o',str(base.with_suffix('.mp4'))]
        rows.append(dict(name=name,control=f'V{control:02d}',shape=[w,h,frames],steps=steps,argv=cmd,evaluation_limit=50 if i<=4 else 6))
    manifest=dict(version=1,source=str(source),model=str(model),seed=42,rows=rows,derivation=commands,
                  inputs={str(p):sha(p) for p in images+[video,replacement,audio]},sources={str(p):sha(p) for p in sources},decoded_audio_sha256=audio_hashes)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Prepared frozen 18-video manifest:',out/'manifest.json')


def publish(out,manifest,records,identity,complete=False):
    result=dict(complete=complete,passed=complete and len(records)==18 and all(r['passed'] for r in records),binary_sha256=identity,
                manifest_sha256=sha(out/'manifest.json'),visual_approval='pending human review',jobs=records)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    cards=['<!doctype html><meta charset="utf-8"><title>Adaptive reference comparison</title>',
           '<style>body{font:16px system-ui;margin:2em}section{display:flex;gap:1em;flex-wrap:wrap}article{width:30%;min-width:280px}video,img{width:100%}pre{white-space:pre-wrap}</style>',
           '<h1>BF16 adaptive cache with references</h1><p>One video per frozen variant. Automated checks and numerical diagnostics do not imply visual acceptance.</p>',
           '<button onclick="document.querySelectorAll(\'video\').forEach(v=>{v.currentTime=0;v.play()})">Play all from start</button> <button onclick="document.querySelectorAll(\'video\').forEach(v=>v.pause())">Pause all</button>']
    by_name={r['name']:r for r in records}
    for control in ('V01','V05','V08','V10','V13','V16'):
        cards.append(f'<h2>Group {control}</h2><section>')
        for spec in manifest['rows']:
            if spec['control']!=control:continue
            name=spec['name'];r=by_name.get(name)
            cards.append(f'<article><h3>{name}</h3>')
            if r:
                cards.append(f'<p>{r["wall_seconds"]:.2f}s total · {r.get("hits",0)} hits · {"PASS" if r["passed"] else "FAILED"}</p><video controls preload="metadata" src="{name}.mp4"></video><img src="{name}-contact.jpg">')
                cards.append(f'<pre>{html.escape(json.dumps(r.get("similarity",{}),indent=2))}</pre><a href="{name}.log">Full log</a>')
            cards.append(f'<details><summary>Command</summary><pre>{html.escape(" ".join(spec["argv"]))}</pre></details></article>')
        cards.append('</section>')
    (out/'review.html').write_text('\n'.join(cards))


def run(a):
    out=a.out.resolve();manifest=json.loads((out/'manifest.json').read_text());source=Path(manifest['source'])
    for path,digest in manifest['inputs'].items():
        if sha(path)!=digest:raise ValueError('changed frozen input: '+path)
    identity=sha(source/'bin/h3cli');records=[]
    if (out/'result.json').exists():
        old=json.loads((out/'result.json').read_text())
        if old['binary_sha256']!=identity or old['manifest_sha256']!=sha(out/'manifest.json'):raise ValueError('changed campaign identity')
        records=old['jobs']
    done={r['name'] for r in records if r['passed']}
    for spec in manifest['rows']:
        name=spec['name']
        if name in done:continue
        if any(r['name']==name for r in records):raise ValueError('failed attempt needs a reviewed new campaign directory; refusing overwrite')
        limit=50 if name in ('V01','V02','V03','V04') else 6
        if limit!=spec['evaluation_limit'] or spec['steps']>limit:raise ValueError('evaluation limit mismatch')
        env=dict(os.environ,H3_TEST_MAX_EVALUATIONS=str(limit),H3_EXPERIMENT_TRACE='1')
        base=out/name;record=dict(name=name,argv=spec['argv'],passed=False);started=time.monotonic()
        memory=base.with_suffix('.memory.csv').open('w')
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-lms','250'],stdout=memory,stderr=subprocess.DEVNULL)
        try:
            with base.with_suffix('.log').open('w') as log:
                proc=subprocess.run(spec['argv'],cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
            record['wall_seconds']=time.monotonic()-started;record['returncode']=proc.returncode
            if proc.returncode:raise RuntimeError('render exited '+str(proc.returncode))
            video=base.with_suffix('.mp4')
            streams=json.loads(subprocess.check_output([ffprobe(),'-v','error','-count_frames','-show_streams','-of','json',str(video)]))['streams']
            v=next(s for s in streams if s['codec_type']=='video');sound=next(s for s in streams if s['codec_type']=='audio')
            if [int(v['width']),int(v['height']),int(v['nb_read_frames'])]!=spec['shape'] or v['r_frame_rate']!='24/1':raise ValueError('incorrect video geometry/rate')
            if sound['channels']!=2 or abs(float(sound['duration'])-spec['shape'][2]/24)>.12:raise ValueError('incorrect audio duration/channels')
            subprocess.run([ffmpeg(),'-nostdin','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            values=pcm(video);rms=math.sqrt(sum(x*x for x in values)/len(values))
            if rms<=0:raise ValueError('silent generated audio')
            text=base.with_suffix('.log').read_text();trace=[json.loads(l.split('h3_experiment ',1)[1]) for l in text.splitlines() if l.startswith('h3_experiment ')]
            decisions=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in text.splitlines() if l.startswith('h3cli: adaptive step=')]
            if len(trace)!=spec['steps']:raise ValueError('missing evaluation trace')
            if '--adaptive-cache' in spec['argv'] and len(decisions)!=spec['steps']:raise ValueError('missing decisions')
            for d,t in zip(decisions,trace):
                if d['decision']=='hit' and (t['blocks']!=1 or t['sparse_calls'] or t['router_calls'] or t.get('stream_read_layers',0)>1):raise ValueError('suffix dispatch on cache hit')
            subprocess.run([ffmpeg(),'-nostdin','-v','error','-i',str(video),'-vf',f'fps=24/{max(1,spec["shape"][2]//6)},scale=240:-1,tile=3x2','-frames:v','1',str(out/(name+'-contact.jpg'))],check=True)
            similarity={}
            if name!=spec['control']:
                control=out/(spec['control']+'.mp4');reference=pcm(control);n=min(len(values),len(reference))
                energy=sum(x*x for x in reference[:n]);mse=sum((values[i]-reference[i])**2 for i in range(n))/n
                similarity['audio_rms_difference']=math.sqrt(mse);similarity['audio_relative_l2']=math.sqrt(mse*n/max(energy,1e-20))
                check=subprocess.run([ffmpeg(),'-nostdin','-v','info','-i',str(control),'-i',str(video),'-lavfi','[0:v][1:v]ssim','-an','-f','null','-'],capture_output=True,text=True,check=True)
                match=re.findall(r'SSIM Y:.*',check.stderr);similarity['video_ssim']=match[-1] if match else 'unavailable'
            record.update(passed=True,streams=streams,sha256=sha(video),audio_rms=rms,steps=trace,decisions=decisions,hits=sum(d['decision']=='hit' for d in decisions),
                          denoise_seconds=sum(t['wall_seconds'] for t in trace),similarity=similarity,
                          residency=[l for l in text.splitlines() if any(k in l for k in ('residency','adaptive ceiling','adaptive budget','persistent='))])
        except (OSError,ValueError,RuntimeError,subprocess.SubprocessError,StopIteration) as exc:record['error']=str(exc)
        finally:monitor.terminate();monitor.wait(timeout=10);memory.close()
        record.setdefault('wall_seconds',time.monotonic()-started)
        record['peak_gpu_mib']=max((int(x) for x in base.with_suffix('.memory.csv').read_text().splitlines() if x.strip().isdigit()),default=0)
        records.append(record);publish(out,manifest,records,identity)
        print(name,record['passed'],round(record['wall_seconds'],2),record.get('error',''),flush=True)
        if not record['passed']:return 1
    publish(out,manifest,records,identity,True);return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('prepare','run'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('.'));p.add_argument('--model',type=Path,default=Path('models/MiniMax-H3'))
    for name in ('video','replacement','audio'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.operation=='prepare':
        if not all((a.video,a.replacement,a.audio)):p.error('prepare requires three existing audiovisual sources')
        prepare(a);return 0
    return run(a)

if __name__=='__main__':raise SystemExit(main())
