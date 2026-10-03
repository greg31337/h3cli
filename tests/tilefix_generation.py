#!/usr/bin/env python3
"""Pinned old/new FL2VA, Ref2VA and continuation tile-policy regression.

Runs fresh conditioning first. If cold Qwen varies (an existing issue), retains
that attempt and reruns with only baseline text replayed. Conditions stay fresh.
"""
import argparse, json, os, shutil, subprocess, tarfile, time
from pathlib import Path
from source_tree import build_path
from memory_generation import instrument, digest
from source_tree import copy_source_tree

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    out=ROOT/'outputs/tilefix-validation/generation';out.mkdir(parents=True,exist_ok=True)
    for label in ('before','after'):
        d=out/label
        if not (build_path(d, 'h3cli')).exists():
            d.mkdir(exist_ok=True)
            if label=='before':
                archive=out/'before.tar';subprocess.run(['git','archive','-o',str(archive),'59b64c9'],cwd=ROOT,check=True)
                with tarfile.open(archive) as f:f.extractall(d,filter='data')
            else:
                copy_source_tree(ROOT, d)
            instrument(d)
            with (d/'build.log').open('w') as log:subprocess.run(['make','-j8','all'],cwd=d,stdout=log,stderr=log,check=True)
    if a.prepare_only:return
    reference=out/'reference.mp4'
    if not reference.exists() or subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','csv=p=0',str(reference)],text=True).strip()!='48':
        subprocess.run(['ffmpeg','-v','error','-y','-loop','1','-i',str(ROOT/'inputs/body1.jpg'),'-vf','scale=256:256','-frames:v','48','-r','24','-c:v','libx264','-pix_fmt','yuv420p',str(reference)],check=True)
    results=out/'results.json';records=json.loads(results.read_text()) if results.exists() else {}
    for case in ('fl2va','ref2va','continuation'):
        refs=['--first-frame',str(ROOT/'inputs/face1.jpg')] if case=='fl2va' else ['--ref-image',str(ROOT/'inputs/2.jpg'),'--ref-silent-video',str(reference)]
        if case=='continuation':refs+=['--continue-from',str(out/'ref2va-before.h3av'),'--continue-context','39']
        for label in ('before','after'):
            name=case+'-'+label
            if name in records:continue
            d=out/label;base=out/name
            cmd=[str(build_path(d, 'h3cli')),'-d',str(ROOT/'models/MiniMax-H3'),'-p','A woman looks at the camera and slowly turns her head.',
                 '--width','320','--height','320','--frames','56','--steps','4','--seed','72',*refs,'--save-av-state',str(base)+'.h3av','-o',str(base)+'.mp4']
            env={k:v for k,v in os.environ.items() if not k.startswith('H3_')};env['H3_REGRESSION_DUMP']=str(base);env['H3_PROFILE']='1'
            print('generate',name,flush=True);start=time.monotonic()
            with Path(str(base)+'.log').open('w') as log:subprocess.run(cmd,cwd=d,env=env,stdout=log,stderr=log,check=True)
            replay=False
            fresh={f.name[len(name)+1:]:digest(f) for f in out.glob(name+'.*') if f.suffix!='.log'}
            if label=='after' and fresh['text']!=records[case+'-before']['hashes']['text']:
                raw=out/(name+'-fresh');raw.mkdir(exist_ok=True)
                for f in out.glob(name+'.*'):shutil.move(f,raw/f.name)
                env['H3_REGRESSION_REPLAY']=str(out/(case+'-before'));replay=True
                print('replay only text to isolate known cold Qwen variability:',name,flush=True)
                with Path(str(base)+'.log').open('w') as log:subprocess.run(cmd,cwd=d,env=env,stdout=log,stderr=log,check=True)
            hashes={f.name[len(name)+1:]:digest(f) for f in out.glob(name+'.*') if f.suffix!='.log'}
            records[name]={'hashes':hashes,'fresh_hashes':fresh,'text_replayed':replay,'seconds':time.monotonic()-start,'command':cmd,'binary_sha256':digest(build_path(d, 'h3cli'))}
            results.write_text(json.dumps(records,indent=2)+'\n')
        before=records[case+'-before']['hashes'];after=records[case+'-after']['hashes']
        expected=[k for k in before if k not in ('fresh-text','decoded-rgb','mp4')]
        assert all(before[k]==after[k] for k in expected),(case,[k for k in expected if before[k]!=after[k]])
        assert before['decoded-rgb']!=after['decoded-rgb']
        audio_hashes=[]
        for label in ('before','after'):
            movie=out/(case+'-'+label+'.mp4');aac=movie.with_suffix('.aac')
            subprocess.run(['ffmpeg','-v','error','-y','-i',str(movie),'-map','0:a:0','-c','copy','-f','adts',str(aac)],check=True)
            audio_hashes.append(digest(aac))
        assert audio_hashes[0]==audio_hashes[1],case+' encoded audio differs'
        delivered=int(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','csv=p=0',str(out/(case+'-after.mp4'))],text=True))
        assert delivered==(17 if case=='continuation' else 56)
        records[case]={'unchanged':expected,'decoded_rgb_changed':True,'encoded_aac_sha256':audio_hashes[0],'encoded_audio_exact':True}
        results.write_text(json.dumps(records,indent=2)+'\n');print('PASS',case,flush=True)
if __name__=='__main__':main()
