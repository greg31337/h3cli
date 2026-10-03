#!/usr/bin/env python3
"""Retain M1 Metal kernel checks, balanced timings, commands and input hashes.

Run serially on an idle GPU. Full-model B1/B5 acceptance is a separate test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--binary-dir',type=Path,default=ROOT)
    p.add_argument('--qkv',type=Path,nargs='*',default=[])
    p.add_argument('--sequence',type=int,default=22426,help='243-frame production fixture with 16 text tokens')
    p.add_argument('--skip-production',action='store_true')
    p.add_argument('--sol-only',action='store_true',help='Check SOL without repeating unchanged dense timing experiments')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'index.json').exists():p.error('use a fresh output directory')
    records=[]
    def run(name,binary,args,balanced=False,source=None):
        command=[str((a.binary_dir/binary).resolve()),*map(str,args)]
        env=os.environ.copy();env.pop('H3_TEST_ATTENTION_BALANCED',None)
        if balanced:env['H3_TEST_ATTENTION_BALANCED']='1'
        start=time.monotonic();r=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
        (a.output/(name+'.json')).write_text(r.stdout)
        (a.output/(name+'.log')).write_text(r.stderr)
        row={'name':name,'command':command,'binary_sha256':sha(command[0]),
             'input_sha256':sha(source) if source else None,'balanced':balanced,
             'wall_seconds':time.monotonic()-start,'returncode':r.returncode,
             'result':json.loads(r.stdout) if r.stdout.strip() else None}
        records.append(row);(a.output/'index.json').write_text(json.dumps(records,indent=2)+'\n')
        print(name,r.returncode,flush=True)
        if r.returncode:raise SystemExit('Kernel check failed: '+name)
    for seq,qb,kb,minimum,layout,pattern in [
        (1,32,32,1,0,'random'),(31,32,32,1,0,'random'),
        (129,32,64,1,1,'outlier'),(1025,32,64,.1,0,'constant-key'),
        (1025,32,64,.1,1,'outlier'),(257,64,128,1,1,'random'),
        (4096,32,64,.1,1,'random'),(4097,64,128,.1,0,'random'),(4097,32,32,.1,1,'random')]:
        run(f'sol-{seq}-{qb}-{kb}-{pattern}','bin/metal_sol',[seq,2,qb,kb,minimum,layout,pattern])
    if not a.sol_only:
        for seq,pattern,il,ol in [(65,'zero',0,0),(128,'outlier',1,1),(129,'outlier',1,0),(257,'constant',0,1),(2049,'random',1,1)]:
            run(f'dense-{seq}-{pattern}','bin/metal_attention',[seq,1,pattern,il,ol])
    if not a.skip_production and not a.sol_only:
        run('dense-production','bin/metal_attention',[a.sequence,1,'random',0,0],True)
    for i,path in enumerate(a.qkv):
        if path.stat().st_size%(3*56*128*2):p.error('invalid QKV file length')
        seq=path.stat().st_size//(3*56*128*2)
        if not a.sol_only:run(f'dense-real-{i}','bin/metal_attention',[seq,1,'file',0,0,path.resolve()],True,path)
        run(f'sol-real-{i}','bin/metal_sol',[seq,56,32,64,.1,0,'file',path.resolve()],False,path)

if __name__=='__main__':main()
