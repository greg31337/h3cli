#!/usr/bin/env python3
"""Serial, reproducible conservative mixed SOL checks on the 243-frame fixtures."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from metal_native_bench import ROOT, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.frozen=a.frozen.resolve();a.output=a.output.resolve()
    if a.output.exists():p.error('use a fresh output directory')
    provenance=json.loads((a.frozen/'build-provenance.json').read_text())
    if provenance['binary_sha256']!=sha(a.frozen/'bin/h3cli'):p.error('changed frozen executable')
    binary=a.frozen/'bin/metal_sol'
    source=Path(provenance['source_snapshot'])
    for name in ('src/metal/fp16_attention_host.inc','src/metal/native_attention_host.inc','src/metal/sol.metal','tests/metal_sol.c'):
        if sha(source/name)!=provenance['source_sha256'][name]:p.error('changed snapshot')
    a.output.mkdir(parents=True)
    limits_path=ROOT/'tests/metal_sol_mixed_limits.json';limits=json.loads(limits_path.read_text())
    ledger={'version':1,'binary_sha256':sha(binary),'build_provenance_sha256':sha(a.frozen/'build-provenance.json'),
            'script_sha256':sha(__file__),'limits':limits,'limits_sha256':sha(limits_path),'records':[],
            'production_qualified':False,'protection_layout':'synthetic protected blocks over real QKV; model layout tested separately'}
    def run(name,args,fixture=None,repeats=0):
        command=[str(binary),*map(str,args)]
        env={**os.environ,'H3_TEST_SOL_FP16':'1','H3_SOL_BENCH_REPEATS':str(repeats),'H3_TEST_MAX_EVALUATIONS':'6'}
        env.pop('MTL_SHADER_VALIDATION',None)
        if repeats:env.pop('MTL_DEBUG_LAYER',None)
        else:env['MTL_DEBUG_LAYER']='1'
        start=time.monotonic()
        result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
        (a.output/(name+'.log')).write_text(result.stdout+result.stderr)
        rows=[json.loads(x) for x in result.stdout.splitlines() if x.startswith('{')]
        row={'name':name,'command':command,'returncode':result.returncode,'wall_seconds':time.monotonic()-start,
             'fixture_sha256':sha(fixture) if fixture else None,'results':rows,'log_sha256':sha(a.output/(name+'.log'))}
        if repeats and not result.returncode:
            metric=next(x for x in rows if 'correctness_pass' in x);timing=next(x for x in rows if 'dense_seconds' in x)
            speed=statistics.median(timing['dense_seconds'])/statistics.median(timing['sol_seconds'])
            conservative=min(timing['dense_seconds'])/max(timing['sol_seconds'])
            gate=limits['component_screen']
            row.update(speedup=speed,conservative_speedup=conservative,
                       component_screen_pass=metric['relative_l2']<=gate['relative_l2_max'] and metric['cosine']>=gate['cosine_min'] and
                       speed>=gate['incremental_median_speedup_min'] and conservative>1)
        ledger['records'].append(row)
        (a.output/'index.json').write_text(json.dumps(ledger,indent=2)+'\n')
        print(name,'rc',result.returncode,'speedup',row.get('speedup'),'pass',row.get('component_screen_pass'),flush=True)
        if result.returncode:raise SystemExit(result.returncode)
    cases=[(17,2,32,64,1,0,'random'),(257,2,32,64,.5,0,'random'),(513,2,32,32,.5,1,'random'),
           (1025,2,64,64,.5,1,'random'),(1025,2,64,128,1,0,'random'),(1025,2,32,64,.1,1,'constant-key'),
           (1025,2,32,64,.5,0,'outlier'),(513,2,32,64,.5,1,'recovery'),(257,2,32,64,.5,0,'nonfinite'),
           (4097,2,32,64,.75,0,'random')]
    for i,case in enumerate(cases):run(f'operator-{i:02d}',case)
    for block in (0,24,49):
        file=ROOT/'outputs/metal-native-m1-243/reference-profile'/('qkv.bf16'+(f'.block{block}' if block else ''))
        if file.stat().st_size!=3*22426*56*128*2:raise ValueError('wrong 243-frame QKV size')
        for floor in limits['initial_sweep']['minimum_exact']:
            run(f'block{block}-exact{floor}',[22426,56,32,64,floor,0,'file',file],file,3)
    candidates=limits['initial_sweep']['minimum_exact']
    ledger['component_screen_pass']=any(all(next(r for r in ledger['records'] if r['name']==f'block{b}-exact{x}')['component_screen_pass'] for b in (0,24,49)) for x in candidates)
    (a.output/'index.json').write_text(json.dumps(ledger,indent=2)+'\n')

if __name__=='__main__':main()
