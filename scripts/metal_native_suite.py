#!/usr/bin/env python3
"""Save once, then run repeatable B1/B2/B5/B6 (at most six evaluations per run)."""
import argparse
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from metal_native_bench import sha

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--model', type=Path, default=ROOT/'models/MiniMax-H3')
p.add_argument('--binary', type=Path, default=ROOT/'bin/h3cli')
p.add_argument('--backend', choices=['mpsgraph','metal'], default='mpsgraph')
p.add_argument('--metal-attention', dest='attention', choices=['dense','sol'], default='dense')
p.add_argument('--metal-attention-kernel', choices=['steel','steel-routed'], default='steel')
p.add_argument('--cases', nargs='+', choices=['B1','B2','B5','B6'], default=['B1','B2','B5','B6'])
p.add_argument('--conditioning', type=Path)
p.add_argument('--width', type=int, default=640)
p.add_argument('--height', type=int, default=480)
p.add_argument('--frames', type=int, default=243)
p.add_argument('--full-vae', action='store_true')
a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
cache=a.conditioning.resolve() if a.conditioning else a.output/'conditioning.h3cond'
base=[sys.executable,str(ROOT/'scripts/metal_native_bench.py'),'--model',str(a.model.resolve()),
      '--binary',str(a.binary.resolve()),'--conditioning',str(cache),
      '--width',str(a.width),'--height',str(a.height),'--frames',str(a.frames)]
if a.full_vae:base+=['--full-vae']
records=[]
def run(name,options):
    directory=a.output/name
    # Reuse only successfully completed records with the same exact command.
    expected=base+['--output',str(directory)]+options
    policy={'command':expected,'runner_sha256':sha(ROOT/'scripts/metal_native_bench.py'),
            'environment':{k:v for k,v in os.environ.items() if k.startswith('H3_') or k=='DEVELOPER_DIR'},
            'platform':platform.platform()}
    marker=directory/'suite-command.json'
    if marker.exists():
        if json.loads(marker.read_text())!=policy:raise SystemExit('Stale command, runner or environment: '+str(directory))
        record=json.loads((directory/'record.json').read_text())
        if record['manifest']['binary_sha256']!=sha(a.binary) or record['conditioning_sha256_after']!=sha(cache):
            raise SystemExit('Stale binary or conditioning: '+str(directory))
        if not(record['evaluation_contract_pass'] and record['conditioning_bypass_pass']):
            raise SystemExit('Previous run failed: '+str(directory))
    else:
        subprocess.run(expected,cwd=ROOT,check=True)
        marker.write_text(json.dumps(policy,indent=2)+'\n')
        record=json.loads((directory/'record.json').read_text())
    records.append({'name':name,**record})
if not cache.exists():run('save-b1',['--case','B1','--save-conditioning','--backend','mpsgraph'])
for case in a.cases:run(a.backend+'-'+case.lower(),['--case',case,'--backend',a.backend,
    '--metal-attention',a.attention,'--metal-attention-kernel',a.metal_attention_kernel])
(a.output/'suite.json').write_text(json.dumps(records,indent=2)+'\n')
if a.backend=='mpsgraph' and 'B1' in a.cases and (a.output/'save-b1/result.h3av').exists():
    original=(a.output/'save-b1/result.h3av').read_bytes()
    cached=(a.output/'mpsgraph-b1/result.h3av').read_bytes()
    if original!=cached:raise SystemExit('Cached and uncached B1 AV states differ')
    print('PASS: exact cached/uncached B1 AV state')
