#!/usr/bin/env python3
"""Two 256x256 / 22-frame / two-evaluation segments, with real frame references.

CUDA uses the existing serial lock, full-depth and 60 GiB development guard.
The complete merged video is 44 frames; no upscale or long render is performed.
"""
import argparse
import fcntl
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from long_video_campaign import script_plan
from long_video_guard import inspect_state, inspect_video, run_case

SCRIPT = '''### opening
#size 256x256
#frames 22
#steps 2
#seed 42
#no-preview-vae
subject_definitions:
A red toy boat with a white sail.
summary:
The boat floats beside the reeds on a quiet pond.
overall_soundscape:
Soft water and a gentle breeze.
non_diegetic_music:
N/A
### next
#use-last-sections
#ref-image #last-segment-frame 1
#ref-image #last-segment-frame -1
summary:
The same boat drifts away from the reeds.
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',required=True,type=Path)
    parser.add_argument('--model',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--cuda',action='store_true')
    args=parser.parse_args()
    binary=args.binary.resolve();out=args.out.resolve();out.mkdir(parents=True)
    source=out/'scenes.txt';source.write_text(SCRIPT)
    env=dict(os.environ,H3_OFFLINE='1',H3_TEST_MAX_EVALUATIONS='6')
    command=[str(binary),'--render',str(source),'-d',str(args.model.resolve()),'--offline',
             '--output-dir',str(out/'render')]
    fingerprint=hashlib.sha256(binary.read_bytes()).hexdigest()
    planned=subprocess.run(command+['--dry-run'],env=env,capture_output=True,text=True,check=True,timeout=60)
    data=json.loads(planned.stdout);(out/'plan.json').write_text(planned.stdout)
    assert not (out/'render').exists()
    text=subprocess.run(command+['--dry-run-text'],env=env,capture_output=True,text=True,check=True,timeout=60)
    (out/'plan.txt').write_text(text.stdout)
    assert all('Prompt:\n'+row['prompt']+'\n' in text.stdout for row in data['segments'])
    budget=script_plan(data)
    assert [row['delivered'] for row in data['segments']]==[22,22]
    assert len(data['segments'][1]['frame_references'])==2
    record=dict(passed=False,binary_sha256=fingerprint,cuda=args.cuda)
    start=time.monotonic()
    try:
        if args.cuda:
            with open('/tmp/h3-long-video-gpu.lock','a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                record['guard']=run_case(command,budget,out/'case.mp4',media=False,env=env)
            log=out/'case.log'
        else:
            log=out/'case.log'
            with log.open('w') as stream:
                process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    code=process.wait(timeout=600)
                    assert code==0,code
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid,signal.SIGKILL);process.wait()
        result=json.loads((out/'render/run.json').read_text())
        assert result['status']=='complete'
        env.update(H3_FFMPEG=result['tools']['ffmpeg'],H3_FFPROBE=result['tools']['ffprobe'])
        rows=result['segments']
        assert [r['prompt'] for r in rows]==[r['prompt'] for r in data['segments']]
        assert not rows[1]['continuing'] and rows[1]['use_last_sections']
        def inspect_clip(path, frames):
            info=inspect_video(path,env=env)
            video,=[s for s in info['streams'] if s['codec_type']=='video']
            assert int(video['nb_read_frames'])==frames
            # Use exact stream ticks: FFprobe's decimal duration rounds 22/24 up.
            assert int(video['duration_ts'])*Fraction(video['time_base'])==Fraction(frames,24)
        for row in rows:
            inspect_clip(row['video'],22)
            inspect_state(row['state'])
        inspect_clip(out/'render/scenes.mp4',44)
        ffmpeg=env.get('H3_FFMPEG','ffmpeg')
        def rgb(path):
            return subprocess.check_output([ffmpeg,'-v','error','-i',str(path),'-map','0:v:0',
                                            '-f','rawvideo','-pix_fmt','rgb24','-'],env=env,timeout=60)
        pixels=rgb(rows[0]['video']);stride=256*256*3
        for ref,expected in zip(rows[1]['frame_references'],(pixels[:stride],pixels[-stride:])):
            assert ref['status']=='complete' and rgb(ref['image'])==expected
            assert ref['image'] in rows[1]['command']
        progress=log.read_text()
        assert re.search(r'denoise\s+\d+/\d+\s+\([^\r\n]*remaining:',progress)
        assert hashlib.sha256(binary.read_bytes()).hexdigest()==fingerprint
        record['passed']=True
    finally:
        record['seconds']=time.monotonic()-start
        (out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print('PASS real frame extraction/conditioning, independent prompt inheritance, dry runs and denoising ETA')

if __name__=='__main__':
    main()
