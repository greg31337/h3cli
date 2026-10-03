#!/usr/bin/env python3

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
from source_tree import source_path
"""Fresh deterministic three-mode FL2VA/image/video quality comparisons.

No text replay. Captures initial noise, conditions, presentation, per-layer text,
latents and decoded outputs to separate arithmetic from changed request inputs.
"""
import argparse,json,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
from memory_generation import instrument,digest
from scalingfix_instrument import instrument as instrument_qwen
from source_tree import copy_source_tree

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/scalingfix-validation/generation'
def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--only',default='fl2va,image,video');p.add_argument('--modes',default='legacy,scaled-q,reference');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);build=OUT/'build';build.mkdir(exist_ok=True)
    if a.prepare_only or not (build/'bin/h3cli').exists():
        copy_source_tree(ROOT, build)
        instrument(build);instrument_qwen(build);path=source_path(build, 'engine.c');s=path.read_text();needle='    h3_rng_fill_normal(&audio_rng, audio, audio_count);';assert s.count(needle)==1
        s=s.replace(needle,needle+'\n    regression_buffer("initial-video",video,video_count*4,0);\n    regression_buffer("initial-audio",audio,audio_count*4,0);\n    regression_buffer("schedule",&sigmas,sizeof(sigmas),0);');path.write_text(s)
        with (build/'build.log').open('w') as log:subprocess.run(['make','-j8','bin/h3cli'],cwd=build,stdout=log,stderr=log,check=True)
    if a.prepare_only:return
    reference=OUT/'reference.mp4'
    if not reference.exists():subprocess.run(['ffmpeg','-v','error','-y','-loop','1','-i',str(ROOT/'inputs/body1.jpg'),'-vf',"scale=256:256,zoompan=z='1+0.0008*on':x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d=48:s=256x256:fps=24",'-frames:v','48','-c:v','libx264','-pix_fmt','yuv420p',str(reference)],check=True)
    cases={
        'fl2va':('A close-up of the woman. She looks into the camera and says <d>[English] Where are you going? To the garden.</d> Her lips move naturally as she speaks.', ['--first-frame',str(ROOT/'inputs/face1.jpg')]),
        'image':('The woman in <Picture 1> stands on the left of a man in a blue shirt. A red book rests on the table between them. She lifts her right hand and the man turns toward her. A third person remains behind them near the window. The camera stays steady.', ['--ref-image',str(ROOT/'inputs/2.jpg')]),
        'video':('The woman in <Video 1> walks slowly toward the camera. A second woman stands to her right and a man remains behind them near a window. She turns her head toward the second woman. Keep the same main woman and her clothes. The camera stays steady.', ['--ref-silent-video',str(reference)])}
    results=OUT/'results.json';records=json.loads(results.read_text()) if results.exists() else {}
    for case in a.only.split(','):
        prompt,refs=cases[case]
        for mode in a.modes.split(','):
            name=case+'-'+mode
            if name in records:continue
            base=OUT/name;qwen=OUT/(name+'-qwen');qwen.mkdir(exist_ok=True)
            cmd=[str(build/'bin/h3cli'),'-d',str(ROOT/'models/MiniMax-H3'),'-p',prompt,'--width','320','--height','320','--frames','124','--steps','20','--seed','72',*refs,'--save-av-state',str(base)+'.h3av','-o',str(base)+'.mp4']
            env={k:v for k,v in os.environ.items() if not k.startswith('H3_')};env.update(H3_QWEN_GQA_SCALE_MODE=mode,H3_REGRESSION_DUMP=str(base),H3_TEST_QWEN_DUMP=str(qwen),H3_PROFILE='1')
            print('generate',name,flush=True);start=time.monotonic()
            with Path(str(base)+'.log').open('w') as log:subprocess.run(cmd,cwd=build,env=env,stdout=log,stderr=log,check=True)
            hashes={f.name[len(name)+1:]:digest(f) for f in OUT.glob(name+'.*') if f.is_file() and f.suffix!='.log'}
            for suffix in ('video','audio','decoded-rgb','decoded-pcm'):
                data=np.fromfile(str(base)+'.'+suffix,'<f4');assert data.size and np.isfinite(data).all(),(name,suffix)
            inputs={f.name:digest(f) for f in qwen.iterdir() if not f.name.startswith(('layer-','first-'))}
            records[name]={'hashes':hashes,'qwen_inputs':inputs,'seconds':time.monotonic()-start,'command':cmd,'binary_sha256':digest(build/'bin/h3cli'),'text_replayed':False}
            results.write_text(json.dumps(records,indent=2)+'\n');print('complete',name,records[name]['seconds'],flush=True)
        names=[case+'-'+m for m in ('legacy','scaled-q','reference')]
        if not all(n in records for n in names):continue
        baseline=records[names[0]]
        invariant=[k for k in baseline['hashes'] if k.startswith('pixels-') or k in ('ids','positions','spans','condition-video','condition-audio','initial-video','initial-audio','schedule')]
        for name in names[1:]:
            assert all(baseline['hashes'][k]==records[name]['hashes'][k] for k in invariant),(name,'changed non-Qwen input')
            assert baseline['qwen_inputs']==records[name]['qwen_inputs'],(name,'changed Qwen input')
        records[case]={'invariant_inputs':invariant,'qwen_presentation_exact':True};results.write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
