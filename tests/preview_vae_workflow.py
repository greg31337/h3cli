#!/usr/bin/env python3
"""Matched fresh-process generation timing; short smoke output, not quality evidence."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import hashlib
import json
from pathlib import Path
import subprocess
import time
from preview_vae_gallery import inspect

root=Path(os.environ.get('H3_PREVIEW_TEST_ROOT','outputs/preview-vae/metal'))/'cold-workflow';root.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,H3_CPU_SAMPLER='1');env.pop('H3_GPU_SAMPLER',None)
results=[]
for mode in ['full','tiny']:
    output=root/(mode+'.mp4');state=root/(mode+'.h3av')
    command=['./bin/h3cli','-d','models/MiniMax-H3','-p','A person smiles gently in a quiet studio. Soft daylight, fixed camera.',
        '--width','128','--height','128','--render-width','64','--render-height','64','--frames','22','--steps','2','--seed','72',
        '--save-av-state',str(state),'-o',str(output),'--profile']
    if mode=='tiny':command.append('--preview-vae')
    start=time.monotonic();r=subprocess.run(command,env=env,capture_output=True,text=True,timeout=110);wall=time.monotonic()-start
    (root/(mode+'.log')).write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
    inspect(output,22,128,128)
    results.append(dict(mode=mode,seconds=wall,command=command,environment={'H3_CPU_SAMPLER':'1'},state_sha256=hashlib.sha256(state.read_bytes()).hexdigest(),mp4_sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
assert results[0]['state_sha256']==results[1]['state_sha256']
(root/'timing.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results))
