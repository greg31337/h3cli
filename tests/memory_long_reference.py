#!/usr/bin/env python3
"""Issue-47 memory cleanup stress and bounded long-sequence attention."""
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs/memory-validation/long-reference'; out.mkdir(parents=True,exist_ok=True)
video=out/'reference-15s-1280x720.mp4'
if not video.exists():
    subprocess.run(['ffmpeg','-v','error','-y','-loop','1','-framerate','24',
        '-i',str(ROOT/'inputs/body1.jpg'),'-vf','scale=1280:720','-frames:v','360',
        '-c:v','libx264','-preset','ultrafast','-pix_fmt','yuv420p',str(video)],check=True)
env={k:v for k,v in os.environ.items() if not k.startswith('H3_')}
binary=ROOT/'bin/memory_long_reference'
binary.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['clang','-std=c11','-D_DARWIN_C_SOURCE','-I',str(ROOT),'-O1','-g',
    '-DH3_DECODER_SOURCE="'+str(ROOT/'src/media/ffmpeg.c')+'"',str(ROOT/'tests/memory_decoder_probe.c'),
    str(ROOT/'src/memory.c'),'-o',str(binary)],check=True)
start=time.monotonic()
result=subprocess.run([str(binary),str(video),'1280','704','361','0',str(out/'unused.f32'),'-200'],
    capture_output=True,text=True,env=env,timeout=60)
assert result.returncode==0,(result.returncode,result.stderr)
data=json.loads(result.stdout)
assert not data['ok'] and data['live']==0 and data['queries']>=200,data
assert 'reclaimable physical memory' in result.stderr,result.stderr
match=re.search(r'stress actual first=(\d+) minimum=(\d+) test-floor=(\d+)',result.stderr)
assert match,result.stderr
first,minimum,floor=map(int,match.groups())
assert minimum>10*1024**3,(minimum,result.stderr)
records={'stress':{'result':data,'error':result.stderr,'actual_first':first,'actual_minimum':minimum,
    'test_floor':floor,'seconds':time.monotonic()-start}}
print('ok: live macOS measurement triggered safe override-floor cancellation during long-reference decode',flush=True)
# Long sequences now use bounded tiled GQA instead of early rejection at the
# dynamic-threadgroup boundary. Exercise admission and actual finite dispatch
# below/at/above that boundary; no model render or numerical oracle is needed.
subprocess.run(['make','bin/memory_gpu_tests'],cwd=ROOT,check=True)
result=subprocess.run([str(ROOT/'bin/memory_gpu_tests')],cwd=ROOT,
    capture_output=True,text=True,env=env,timeout=120)
(out/'preflight.log').write_text(result.stdout+result.stderr)
assert result.returncode==0,(result.stdout,result.stderr)
records['gqa_boundary']={'passed':True,'log':result.stdout}
(out/'results.json').write_text(json.dumps(records,indent=2)+'\n')
print('ok: tiled long-sequence attention boundary and memory-pressure cleanup',flush=True)
