#!/usr/bin/env python3
"""Qualify linked cuDNN algorithms on bounded real audio operation captures."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('probe',type=Path);p.add_argument('capture',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.capture/'result.json').read_text());rows=[]
    # Reject stale inputs before launching a diagnostic against them.
    for name,meta in manifest['capture'].items():
        path=a.capture/(name+'.bin')
        if path.stat().st_size!=meta['bytes'] or hashlib.file_digest(path.open('rb'),'sha256').hexdigest()!=meta['sha256']:raise ValueError('invalid capture: '+name)
    for name,shape in manifest['convolutions'].items():
        out=a.out/name;out.mkdir()
        cmd=[str(a.probe.resolve()),str(a.capture.resolve()),name,str(out.resolve())]+[str(int(shape[k])) for k in ('batch','ci','co','length','kernel','stride','padding','dilation','transpose','groups')]
        with (out/'command.log').open('w') as log:subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        metrics=[json.loads(f.read_text()) for f in sorted(out.glob('algorithm-*.json'))]
        row=dict(name=name,shape=shape,commands=cmd,algorithms=metrics,exact_algorithms=[m['algorithm'] for m in metrics if m['finite'] and not m['different']])
        rows.append(row);print(name,row['exact_algorithms'],flush=True)
        (a.out/'result.json').write_text(json.dumps(dict(capture=str(a.capture),probe_sha256=hashlib.sha256(a.probe.read_bytes()).hexdigest(),results=rows),indent=2)+'\n')
if __name__=='__main__':main()
