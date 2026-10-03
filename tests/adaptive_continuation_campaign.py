#!/usr/bin/env python3
"""Frozen, resumable continuation comparison; only V01--V10 permit 50 steps."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import html
import json
import math
from pathlib import Path
import re
import subprocess
import time

from adaptive_reference_campaign import sha, ffmpeg, ffprobe, pcm


def prepare(a):
    root = Path(__file__).resolve().parents[1]
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    image = root/'inputs/1.jpg'
    prompt = 'A woman walks slowly along a sunlit beach. Her hair and dress move gently in the sea breeze. A steady camera follows her. Soft ocean waves and natural outdoor ambience.'
    bridge = 'The woman slows her walk along the same sunlit beach, gently turns toward the camera and smiles. Her hair and dress move in the sea breeze. A steady camera follows her. Soft ocean waves and natural outdoor ambience.'
    rows = []
    policies = [('dense', []), ('conservative', ['--adaptive-cache', 'conservative']),
                ('aggressive', ['--adaptive-cache', 'aggressive']),
                ('subblock', ['--cuda-attention', 'subblock', '--subblock-sparsity', '.75']),
                ('combined', ['--adaptive-cache', 'conservative', '--cuda-attention', 'subblock', '--subblock-sparsity', '.75'])]

    def add(name, shape, steps, source=None, mode='hard', policy=('dense', []), refs=False, control=None):
        base = out/name
        cmd = [str(root/'bin/h3cli'), '-d', str(a.model.resolve()), '-p', bridge if mode == 'bridge' else prompt,
               '--width', str(shape[0]), '--height', str(shape[1]), '--frames', str(shape[2]),
               '--steps', str(steps), '--seed', '42', '--reuse', '1', '--core-reuse', '1', '--cuda-weight-mode', 'auto']
        if source:
            cmd += ['--continue-from', str(out/(source+'.h3av')), '--continue-context', '39', '--continue-mode', mode]
            if mode == 'bridge':
                cmd += ['--continue-bridge-steps', '8', '--continue-bridge-max-strength', '.5', '--continue-bridge-profile', 'stepped']
        if refs:
            cmd += ['--ref-image', str(image), '--ref-image-size', 'max']
        cmd += policy[1]
        cmd += ['--save-av-state', str(base.with_suffix('.h3av')), '--profile', '-o', str(base.with_suffix('.mp4'))]
        rows.append(dict(name=name, shape=shape, delivered_frames=shape[2]-(39 if source else 0), steps=steps,
                         source=source, mode=mode, policy=policy[0], control=control, argv=cmd,
                         evaluation_limit=50 if name in {f'V{i:02}' for i in range(1, 11)} else 6))

    add('S01', [640, 480, 90], 6)
    add('S02', [1344, 768, 124], 6, refs=True)
    for group, mode in enumerate(('hard', 'bridge')):
        for i, policy in enumerate(policies, 1):
            flags = list(policy[1])
            if '--adaptive-cache' in flags: flags += ['--adaptive-cache-warmup', '4']
            if '--cuda-attention' in flags: flags += ['--subblock-warmup', '10']
            add(f'V{group*5+i:02}', [640, 480, 90], 50, 'S01', mode, (policy[0], flags), control=f'V{group*5+1:02}')
    custom = ('combined-custom', policies[-1][1]+['--adaptive-cache-threshold', '.12', '--adaptive-cache-max-hits', '2',
                                                  '--adaptive-cache-warmup', '2', '--subblock-warmup', '3'])
    for group, mode in enumerate(('hard', 'bridge')):
        for i, policy in enumerate((policies[0], custom)):
            add(f'V{11+group*2+i:02}', [1344, 768, 124], 6, 'S02', mode, policy, True, f'V{11+group*2:02}')
    manifest = dict(version=1, rows=rows, inputs={str(image): sha(image)}, root=str(root),
                    timing='Subprocess loading through MP4 completion; shared source cost reported separately.',
                    source_files={str(p.relative_to(root)): sha(p) for p in [root/'Makefile',*sorted((root/'src').rglob('*'))]
                                  if p.name=='Makefile' or p.suffix in ('.c','.h','.cu','.cuh','.m','.metal','.inc','.cpp')},
                    binary_sha256=sha(root/'bin/h3cli'))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


def publish(out, manifest, records, complete=False):
    result = dict(complete=complete, passed=complete and len(records)==16 and all(r['passed'] for r in records),
                  manifest_sha256=sha(out/'manifest.json'), visual_acceptance='pending', jobs=records)
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    cards = ['<!doctype html><meta charset="utf-8"><title>Continuation comparison</title>',
             '<style>body{font:16px system-ui;margin:2em}section{display:flex;flex-wrap:wrap;gap:1em}article{width:30%;min-width:300px}video,img{width:100%}pre{white-space:pre-wrap}</style>',
             '<h1>Adaptive cache and SubBlock continuation</h1><p>Automated media checks do not imply human visual/listening acceptance.</p>',
             '<button onclick="document.querySelectorAll(\'video\').forEach(v=>{v.currentTime=0;v.play()})">Play all</button> <button onclick="document.querySelectorAll(\'video\').forEach(v=>v.pause())">Pause all</button><section>']
    for spec in manifest['rows']:
        name = spec['name']; record = next((r for r in records if r['name']==name), None)
        cards.append(f'<article><h2>{name}: {spec["mode"]} / {spec["policy"]}</h2>')
        if record and record['passed']:
            cards.append(f'<p>{record["wall_seconds"]:.2f}s total; {record["denoise_seconds"]:.2f}s denoise; {record["hits"]} hits</p><video controls preload="metadata" src="{name}.mp4"></video><img src="{name}-contact.jpg">')
            if spec['source']: cards.append(f'<a href="{name}-join.mp4">Source tail + continuation with audio</a>')
        cards.append(f'<details><summary>Command and evidence</summary><pre>{html.escape(json.dumps(record or spec, indent=2))}</pre></details></article>')
    (out/'review.html').write_text('\n'.join(cards)+ '</section>')


def run(a):
    out = a.out.resolve(); manifest = json.loads((out/'manifest.json').read_text()); root = Path(manifest['root'])
    for path, digest in manifest['inputs'].items():
        if sha(path)!=digest: raise ValueError('changed frozen input '+path)
    if sha(root/'bin/h3cli')!=manifest['binary_sha256']: raise ValueError('changed binary')
    for path, digest in manifest['source_files'].items():
        if sha(root/path)!=digest: raise ValueError('changed source '+path)
    records = []
    if (out/'result.json').exists():
        old = json.loads((out/'result.json').read_text())
        if old['manifest_sha256']!=sha(out/'manifest.json'): raise ValueError('changed manifest')
        records = old['jobs']
    for spec in manifest['rows']:
        name=spec['name']; base=out/name
        previous=next((r for r in records if r['name']==name), None)
        if previous:
            if not previous['passed']: raise ValueError('preserved failed attempt requires a new reviewed campaign')
            for path, digest in previous['artifacts'].items():
                if sha(out/path)!=digest: raise ValueError('changed successful artifact '+path)
            continue
        limit=50 if name in {f'V{i:02}' for i in range(1,11)} else 6
        if spec['evaluation_limit']!=limit or spec['steps']>limit: raise ValueError('invalid evaluation ceiling')
        if base.with_suffix('.log').exists(): raise ValueError('unrecorded attempt, refusing stale output')
        source_hashes = {p.name:sha(p) for p in out.glob(spec['source']+'.h3av*')} if spec['source'] else {}
        record=dict(name=name, passed=False, argv=spec['argv'], source_hashes=source_hashes)
        memory=base.with_suffix('.memory.csv').open('w')
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-lms','250'],stdout=memory,stderr=subprocess.DEVNULL)
        start=time.monotonic()
        try:
            with base.with_suffix('.log').open('w') as log:
                p=subprocess.run(spec['argv'],cwd=root,env=os.environ|{'H3_TEST_MAX_EVALUATIONS':str(limit),'H3_EXPERIMENT_TRACE':'1'},stdout=log,stderr=subprocess.STDOUT,timeout=7200)
            record.update(wall_seconds=time.monotonic()-start, returncode=p.returncode)
            if p.returncode: raise ValueError('render failed')
            video=base.with_suffix('.mp4')
            streams=json.loads(subprocess.check_output([ffprobe(),'-v','error','-count_frames','-show_streams','-of','json',str(video)]))['streams']
            v=next(s for s in streams if s['codec_type']=='video'); audio=next(s for s in streams if s['codec_type']=='audio')
            if [int(v['width']),int(v['height']),int(v['nb_read_frames'])]!=spec['shape'][:2]+[spec['delivered_frames']] or v['r_frame_rate']!='24/1': raise ValueError('wrong delivered geometry')
            if audio['channels']!=2 or abs(float(audio['duration'])-spec['delivered_frames']/24)>.12: raise ValueError('wrong soundtrack duration')
            subprocess.run([ffmpeg(),'-nostdin','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True,capture_output=True)
            samples=pcm(video); rms=math.sqrt(sum(x*x for x in samples)/len(samples))
            if rms==0: raise ValueError('silent audio')
            text=base.with_suffix('.log').read_text()
            traces=[json.loads(s.split('h3_experiment ',1)[1]) for s in text.splitlines() if s.startswith('h3_experiment ')]
            decisions=[dict(re.findall(r'(\w+)=([^ ]+)',s)) for s in text.splitlines() if s.startswith('h3cli: adaptive step=')]
            if len(traces)!=spec['steps'] or ('--adaptive-cache' in spec['argv'] and len(decisions)!=spec['steps']): raise ValueError('missing execution trace')
            for d,t in zip(decisions,traces):
                if d['decision']=='hit' and (t['blocks']!=1 or t['sparse_calls'] or t['router_calls'] or t['stream_read_layers']>1): raise ValueError('suffix executed on hit')
            subprocess.run([ffmpeg(),'-nostdin','-v','error','-i',str(video),'-vf',f'fps=24/{max(1,spec["delivered_frames"]//6)},scale=320:-1,tile=3x2','-frames:v','1',str(out/(name+'-contact.jpg'))],check=True)
            if spec['source']:
                source=out/(spec['source']+'.mp4')
                subprocess.run([ffmpeg(),'-nostdin','-v','error','-sseof','-1','-i',str(source),'-i',str(video),'-filter_complex','[0:v]setpts=PTS-STARTPTS[v0];[0:a]asetpts=PTS-STARTPTS[a0];[v0][a0][1:v][1:a]concat=n=2:v=1:a=1[v][a]','-map','[v]','-map','[a]','-c:v','libx264','-crf','18','-c:a','aac',str(out/(name+'-join.mp4'))],check=True)
            for path,digest in source_hashes.items():
                if sha(out/path)!=digest: raise ValueError('source mutated')
            similarity={}
            if spec['control'] and spec['control']!=name:
                control=out/(spec['control']+'.mp4'); reference=pcm(control); count=min(len(samples),len(reference))
                difference=sum((samples[i]-reference[i])**2 for i in range(count))
                similarity['audio_relative_l2']=math.sqrt(difference/max(sum(x*x for x in reference[:count]),1e-20))
                comparison=subprocess.run([ffmpeg(),'-nostdin','-v','info','-i',str(control),'-i',str(video),
                    '-lavfi','[0:v][1:v]ssim','-an','-f','null','-'],capture_output=True,text=True,check=True)
                values=re.findall(r'SSIM Y:.*',comparison.stderr)
                if not values: raise ValueError('missing SSIM diagnostic')
                similarity['video_ssim']=values[-1]
            record.update(passed=True, streams=streams, audio_rms=rms, steps=traces, decisions=decisions,
                          hits=sum(d['decision']=='hit' for d in decisions), denoise_seconds=sum(t['wall_seconds'] for t in traces),
                          similarity=similarity,
                          diagnostics=[s for s in text.splitlines() if any(k in s for k in ('weight planner:','adaptive continuation','adaptive budget','persistent='))],
                          artifacts={p.name:sha(p) for p in out.glob(name+'.*') if p.suffix in ('.mp4','.h3av','.presentation')})
        except (ValueError,OSError,StopIteration,subprocess.SubprocessError) as e: record['error']=str(e)
        finally:
            monitor.terminate(); monitor.wait(timeout=10); memory.close()
        record.setdefault('wall_seconds',time.monotonic()-start)
        record['peak_gpu_mib']=max((int(s) for s in base.with_suffix('.memory.csv').read_text().splitlines() if s.strip().isdigit()),default=0)
        records.append(record); publish(out,manifest,records)
        print(name,record['passed'],round(record['wall_seconds'],2),record.get('error',''),flush=True)
        if not record['passed']: return 1
    publish(out,manifest,records,True)
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('operation',choices=('prepare','run'))
    p.add_argument('--out',type=Path,required=True); p.add_argument('--model',type=Path,default=Path('models/MiniMax-H3'))
    a=p.parse_args()
    if a.operation=='prepare': prepare(a)
    else: raise SystemExit(run(a))
