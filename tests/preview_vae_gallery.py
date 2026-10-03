#!/usr/bin/env python3
"""Matched local decoder gallery. Every render/reference/validation uses the persistent budget ledger."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('outputs/preview-vae')

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def run_case(name, command, output):
    ledger=BASE/'metal/budget.json'
    runs=json.loads(ledger.read_text())['runs'] if ledger.exists() else []
    for run in runs:
        if run['status']=='passed' and run['command']==command and output.exists() and run.get('output_sha256',{}).get(str(output))==digest(output):
            if all(Path(path).exists() and digest(Path(path))==value for path,value in run.get('input_sha256',{}).items()):
                print(f'Reuse compatible {name}',flush=True);return
    used={r['name'] for r in runs};attempt=1;base=name
    while name in used:
        attempt+=1;name=f'{base}-attempt{attempt}'
    subprocess.run([sys.executable,'tests/preview_vae_run.py','--name',name,'--timeout','120','--',*command],check=True)

def inspect(path, frames, width, height, audio=True):
    metadata = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(path)]))
    v = next(s for s in metadata['streams'] if s['codec_type']=='video')
    assert (v['width'],v['height'],int(v['nb_frames']),v['r_frame_rate'])==(width,height,frames,'24/1'), v
    assert abs(float(v['duration'])-frames/24)<0.00001
    if audio:
        a = next(s for s in metadata['streams'] if s['codec_type']=='audio')
        assert a['sample_rate']=='32000' and a['channels']==2
        assert abs(float(a['duration'])-frames/24)<0.026
    else: assert len(metadata['streams'])==1
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),'-f','null','-'],check=True,timeout=60)
    return metadata

def validate(root):
    records=json.loads((root/'manifest.json').read_text())
    for record in records:
        for mode in ['full','tiny','reference']:
            path=root/f'{record["name"]}-{mode}.mp4'
            if not path.exists(): continue
            record[mode]=dict(path=str(path),sha256=digest(path),media=inspect(path,record['frames'],record['width'],record['height']))
        assert record['full']['media']['streams'][1]['duration']==record['tiny']['media']['streams'][1]['duration']
        # Audio is unchanged, so exact decoded PCM is the appropriate contract.
        audio=[]
        for mode in ['full','tiny']:
            audio.append(subprocess.check_output(['ffmpeg','-v','error','-i',str(root/f'{record["name"]}-{mode}.mp4'),'-vn','-f','f32le','-']))
        assert audio[0]==audio[1], record['name']
    (root/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    cards=[]
    for r in records:
        videos=''.join(f'<figure><figcaption>{mode}</figcaption><video controls preload="metadata" src="{html.escape(r["name"])}-{mode}.mp4"></video></figure>' for mode in ['full','tiny','reference'] if mode in r)
        cards.append(f'<section><h2>{html.escape(r["name"])}</h2><p>{r["width"]}×{r["height"]}, {r["frames"]} frames at 24 fps; trim {r["trim"]}. Same final latents. <a href="manifest.json">Provenance</a></p><div>{videos}</div></section>')
    (root/'review.html').write_text('''<!doctype html><meta charset="utf-8"><title>TAEH3 preview review</title>
<style>body{font:16px system-ui;background:#16191e;color:#eee;margin:24px}section{border-top:1px solid #555;margin-top:28px}section div{display:flex;gap:16px;flex-wrap:wrap}figure{margin:0;max-width:32%}video{width:100%;max-height:540px}a{color:#9cf}button{margin:10px;padding:10px}</style>
<h1>Original VAE / native TAEH3 / pinned reference</h1><p>Quality acceptance is pending. Review identity, action, motion, fine detail loss, color, temporal seams and audio/video synchronization. Play one soundtrack at a time. The face fixture uses six steps and is visibly degraded in the original decoder; it is a limited smoke fixture. Face20 uses a previously reviewed 20-step face-reference sample. Other cases use previously reviewed 20-step B200 latents.</p>
<button onclick="document.querySelectorAll('video').forEach(v=>{v.pause();v.currentTime=0})">Reset all</button>
'''+''.join(cards))
    print(f'PASS {len(records)} matched pairs: complete decode, frame geometry/timing and identical audio')

def joins(root):
    records=[]
    for mode in ['full','tiny','reference']:
        output=root/f'joined-{mode}.mp4'
        command=['ffmpeg','-v','error','-y','-i',str(root/f'chain1-{mode}.mp4'),'-i',str(root/f'chain2-{mode}.mp4'),
            '-filter_complex','[0:a]atrim=duration=3.75,asetpts=PTS-STARTPTS[a0];[1:a]atrim=duration=2.125,asetpts=PTS-STARTPTS[a1];[0:v][a0][1:v][a1]concat=n=2:v=1:a=1[v][a]',
            '-map','[v]','-map','[a]','-c:v','libx264','-preset','fast','-crf','6','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(output)]
        subprocess.run(command,check=True,timeout=60)
        records.append(dict(mode=mode,command=command,sha256=digest(output),media=inspect(output,141,288,384)))
    (root/'joins.json').write_text(json.dumps(records,indent=2)+'\n')
    page=root/'review.html'
    if '<h2>Continuation join</h2>' not in page.read_text():
        with page.open('a') as f:
            f.write('<section><h2>Continuation join</h2><p>90 + 51 frames; transition at 3.75 seconds. <a href="joins.json">Commands and metadata</a></p><div>')
            for mode in ['full','tiny','reference']:f.write(f'<figure><figcaption>{mode}</figcaption><video controls preload="metadata" src="joined-{mode}.mp4"></video></figure>')
            f.write('</div></section>')
    print('PASS complete 141-frame continuation joins')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-dir',default='models/MiniMax-H3');p.add_argument('--reference-python');p.add_argument('--validate',type=Path);p.add_argument('--joins',type=Path);p.add_argument('--case');a=p.parse_args()
    os.chdir(ROOT)
    if a.validate: validate(a.validate);return
    if a.joins: joins(a.joins);return
    root=BASE/'metal/quality';root.mkdir(parents=True,exist_ok=True)
    corpus=json.loads((BASE/'corpus.json').read_text());cases=corpus['cases'][:]
    face=Path('outputs/bugfix1-validation/after-face.h3av')
    cases.append(dict(name='face',state=str(face),state_sha256=digest(face),reference_files={'inputs/face1.jpg':digest(Path('inputs/face1.jpg'))},source_evidence='docs/bugfix1-validation.md; seed 72, 128x128, 56 frames, six steps, FL2VA'))
    face20=BASE/'corpus/face20.h3av'
    face20_info=json.loads(face20.with_suffix('.json').read_text())
    cases.append(dict(name='face20',state=str(face20),state_sha256=digest(face20),reference_files=face20_info['reference_files'],source_sha256=face20_info['source_sha256'],source_evidence='20-step ordinary M4 baseline, Ref2VA face1',ref2va=1))
    if a.case:
        cases=[case for case in cases if case['name']==a.case]
        assert cases, a.case
    records=json.loads((root/'manifest.json').read_text()) if a.case and (root/'manifest.json').exists() else []
    for case in cases:
        source=Path(case['state']);blob=source.read_bytes();assert digest(source)==case['state_sha256']
        width,height,frames=struct.unpack_from('<3I',blob,24)
        trim=39 if case['name']=='chain2' else 0
        # Current files must carry writer-produced provenance; do not manufacture
        # a sidecar to make an archived latent loadable by today's decoder.
        assert struct.unpack_from('<I',blob,8)[0] == 3, 'requires a fresh current AV state'
        sidecar=Path(str(source)+'.presentation')
        assert sidecar.read_text().startswith('H3-PRESENTATION 9\n')
        state=root/(case['name']+'.h3av');shutil.copyfile(source,state)
        shutil.copyfile(sidecar,Path(str(state)+'.presentation'))
        record=dict(case,width=width,height=height,frames=frames-trim,trim=trim,replay_state=str(state),commands={})
        records=[r for r in records if r["name"]!=record["name"]]
        records.append(record)
        for mode in ['full','tiny']:
            output=root/f'{case["name"]}-{mode}.mp4'
            command=['./bin/h3cli','-d',a.model_dir,'--decode-av-state',str(state),'--profile','-o',str(output)]
            if mode=='tiny': command+=['--preview-vae']
            record['commands'][mode]=command
            run_case(f'gallery-{case["name"]}-{mode}',command,output)
        if a.reference_python:
            command=[a.reference_python,'tests/preview_vae_reference.py',str(source),str(root/f'{case["name"]}-reference.mp4'),'--trim',str(trim),'--audio',str(root/f'{case["name"]}-full.mp4')]
            record['commands']['reference']=command
            run_case(f'gallery-{case["name"]}-reference',command,root/f'{case["name"]}-reference.mp4')
        (root/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    run_case('gallery-media-validation',[sys.executable,__file__,'--validate',str(root)],root/'manifest.json')
    run_case('gallery-joins',[sys.executable,__file__,'--joins',str(root)],root/'joins.json')

if __name__=='__main__':main()
