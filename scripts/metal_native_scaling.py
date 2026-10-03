#!/usr/bin/env python3
"""M1 B1 sequence/protection scaling; failed quality remains a failed gate.

The MPSGraph run creates each immutable conditioning cache; SOL then loads it.
Ratios are diagnostic single-step observations, not production qualification.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from metal_native_bench import ROOT,sha

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--binary',type=Path,default=ROOT/'bin/h3cli')
p.add_argument('--source-state',type=Path,required=True)
p.add_argument('--cases',nargs='+',choices=['short','references','continue39','continue192'],
               default=['short','references','continue39','continue192'])
p.add_argument('--capture-layouts',action='store_true',help='Additional reference QKV diagnostics; continuation retains all 50 blocks')
a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
if (a.output/'index.json').exists():p.error('use a fresh output directory')
rows=[]
for case in a.cases:
    root=a.output/case;root.mkdir(exist_ok=True)
    options=['--frames','90' if case=='short' else '243']
    if case=='references':
        for name in ('1.jpg','2.jpg','face1.jpg'):options+=['--reference',str(ROOT/'inputs'/name)]
    if case.startswith('continue'):
        options+=['--continue-from',str(a.source_state.resolve()),'--continue-context',case.removeprefix('continue')]
    base=[sys.executable,str(ROOT/'scripts/metal_native_bench.py'),'--binary',str(a.binary.resolve()),
          '--conditioning',str(root/'conditioning.h3cond'),'--case','B1',*options]
    modes=[('reference',['--save-conditioning']),('sol',['--backend','metal','--metal-attention','sol'])]
    if a.capture_layouts and case in ('references','continue192'):
        # Continuation validates a complete 50-block workload before inference.
        modes.append(('qkv',['--blocks','50' if case.startswith('continue') else '1',
                             '--capture-qkv','--capture-boundaries']))
    for name,extra in modes:
        command=base+['--output',str(root/name),*extra]
        with (root/(name+'.driver.log')).open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(case,name,flush=True)
    compare=root/'compare.json'
    subprocess.run([sys.executable,str(ROOT/'scripts/metal_native_compare.py'),str(root/'reference'),str(root/'sol'),
                    '--max-relative-l2','.05','--min-cosine','.998','--output',str(compare)],cwd=ROOT,check=False)
    if not compare.exists():raise SystemExit('Comparison did not produce a valid record')
    r=json.loads((root/'reference/record.json').read_text());s=json.loads((root/'sol/record.json').read_text())
    exact=sum(x['exact'] for x in s['sol']);approx=sum(x['approximate'] for x in s['sol'])
    rows.append({'case':case,'reference_seconds':r['timing']['first_step_seconds'],
                 'sol_seconds':s['timing']['first_step_seconds'],
                 'routed_exact_fraction':exact/(exact+approx) if exact+approx else 1,
                 'protected_rows':s['sol'][0]['protected_rows'] if s['sol'] else None,
                 'comparison':json.loads(compare.read_text()),'diagnostic_scaling_only':True})
    (a.output/'index.json').write_text(json.dumps({'limits_sha256':sha(ROOT/'tests/metal_native_limits.json'),'cases':rows},indent=2)+'\n')
