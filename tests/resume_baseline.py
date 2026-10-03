#!/usr/bin/env python3
"""Freeze the pre-resume engine and record/replay full 20-step CPU renders."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
from continuation_regression import instrument

from source_tree import copy_source_tree

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/resume-validation/baseline')
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    label = 'current' if args.current else 'original'
    build = out/label; build.mkdir(exist_ok=True)
    if args.current:
        copy_source_tree(ROOT, build)
    else:
        subprocess.run(['git','archive','-o',str(out/'baseline.tar'),'7f4aa56'],cwd=ROOT,check=True)
        with tarfile.open(out/'baseline.tar') as tar: tar.extractall(build,filter='data')
    instrument(build)
    with (build/'build.log').open('w') as log:
        subprocess.run(['make','-j8','all'],cwd=build,stdout=log,stderr=log,check=True)
    env = {k:v for k,v in os.environ.items() if not k.startswith('H3_')}
    env['H3_CPU_SAMPLER']='1'
    records = {'commit':subprocess.check_output(['git','rev-parse','7f4aa56'],cwd=ROOT,text=True).strip(), 'binary_sha256':digest(build/'bin/h3cli'), 'runs':{}, 'passed':False}
    for name in ['t2va','ref2va','continuation']:
        base = out/(name+'-'+label)
        cmd = [str(build/'bin/h3cli'),'-d',str(ROOT/'models/MiniMax-H3'),'-p',
               'The woman walks slowly through a sunlit garden and waves. Steady camera and quiet outdoor ambience.',
               '--width','256','--height','256','--frames','90','--steps','20',
               '--layers','50','--reuse','1','--core-reuse','1','--seed','72',
               '-o',str(base)+'.mp4','--save-av-state',str(base)+'.h3av']
        if name != 't2va':
            cmd += ['--ref-image',str(ROOT/'inputs/face1.jpg'),'--ref-image',str(ROOT/'inputs/body1.jpg')]
        if name == 'continuation':
            cmd += ['--continue-from',str(out/'ref2va-original.h3av'),'--continue-context','39']
        run_env = dict(env,H3_REGRESSION_DUMP=str(base))
        if args.current: run_env['H3_REGRESSION_REPLAY']=str(out/(name+'-original'))
        print('running',name,label,flush=True); start=time.monotonic()
        with Path(str(base)+'.log').open('w') as log:
            subprocess.run(cmd,cwd=build,env=run_env,stdout=log,stderr=log,check=True)
        hashes = {ext:digest(Path(str(base)+'.'+ext)) for ext in
                  ['text','condition-video','condition-audio','video','audio','h3av','mp4']
                  if Path(str(base)+'.'+ext).exists()}
        if args.current:
            original=json.loads((out/'original.json').read_text())
            assert hashes == original['runs'][name]['hashes'], name+' changed'
        records['runs'][name]={'command':cmd,'hashes':hashes,'seconds':time.monotonic()-start}
        (out/(label+'.json')).write_text(json.dumps(records,indent=2)+'\n')
    records['passed']=True
    (out/(label+'.json')).write_text(json.dumps(records,indent=2)+'\n')

if __name__ == '__main__': main()
