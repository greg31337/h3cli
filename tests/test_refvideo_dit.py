#!/usr/bin/env python3
"""Compare native whole Ref2VA DiT with the optional official Diffusers fixture."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1]

def compare(got,want):
    delta=got.astype(np.float64)-want
    return {'max_abs':float(np.max(np.abs(delta))),
        'relative_max':float(np.max(np.abs(delta))/max(np.max(np.abs(want)),1e-30)),
        'relative_l2':float(np.linalg.norm(delta)/max(np.linalg.norm(want),1e-30))}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--binary',type=Path,default=ROOT/'bin/refvideo_dit_test')
    p.add_argument('--full',action='store_true');p.add_argument('--steps',type=int,default=20)
    p.add_argument('--compare-only',action='store_true',help='check retained native outputs without repeating inference')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=args.compare_only)
    command=[str(args.binary.resolve()),str(ROOT/'models/MiniMax-H3/Ref2VA/transformer'),str(args.fixture.resolve()),
        str((args.output/'native').resolve()),str(args.steps),str(int(args.full))]
    env={k:v for k,v in os.environ.items() if not k.startswith('H3_')}
    if not args.compare_only:
        with (args.output/'native.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=log,check=True)
    # BF16 operation boundaries differ across Metal and PyTorch. These bounds
    # are fixed before measurement, following the existing semantic DiT test.
    limit={'relative_max':.15,'relative_l2':.10}
    results={}
    names=['video_velocity','audio_velocity']+(['video_final','audio_final'] if args.full else [])
    for name in names:
        with safe_open(args.fixture,framework='np') as f: want=f.get_tensor('x.'+name)
        got=np.fromfile(args.output/('native.'+name),np.float32).reshape(want.shape)
        metrics=compare(got,want)
        metrics['passed']=bool(np.isfinite(got).all() and all(metrics[k]<=v for k,v in limit.items()))
        results[name]=metrics;print(name,metrics,flush=True)
    record={'passed':all(r['passed'] for r in results.values()),'limits':limit,'cases':results,'command':command,
        'oracle':json.loads(args.fixture.with_suffix('.json').read_text()),
        'binary_sha256':hashlib.sha256(args.binary.read_bytes()).hexdigest()}
    (args.output/'results.json').write_text(json.dumps(record,indent=2)+'\n')
    assert record['passed'],'whole DiT oracle comparison failed'
if __name__=='__main__':main()
