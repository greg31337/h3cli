#!/usr/bin/env python3
"""Serial M2E differential validation at 243 frames, never more than six evaluations.

The unchanged adapter is the arithmetic oracle for fusion. This does not replace
or reclassify the retained MPSGraph quality screens or deferred held-out corpus.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
from metal_native_bench import ROOT,sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--conditioning',type=Path,required=True)
    a=p.parse_args();frozen=a.frozen.resolve();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((frozen/'build-provenance.json').read_text())
    if sha(frozen/'bin/h3cli')!=provenance['binary_sha256']:p.error('changed frozen binary')
    env={**os.environ,'H3_SHADER_PATH':str(Path(provenance['source_snapshot'])/'src/metal/shaders.metal'),'MTL_DEBUG_LAYER':'1'}
    with (out/'operators.jsonl').open('x') as result,(out/'operators.log').open('x') as log:
        subprocess.run([str(frozen/'bin/metal_layout')],cwd=ROOT,env=env,stdout=result,stderr=log,check=True)
    operators=[json.loads(line) for line in (out/'operators.jsonl').read_text().splitlines()]
    if len(operators)!=12 or not all(r['pass'] and r['byte_identical'] for r in operators):raise ValueError('operator failure')
    index={'geometry':[640,480,243],'binary_sha256':sha(frozen/'bin/h3cli'),'script_sha256':sha(__file__),
           'operators_sha256':sha(out/'operators.jsonl'),'operator_binary_sha256':sha(frozen/'bin/metal_layout'),
           'pairs':[],'scope':'Dense B1/B5/B6; conservative SOL B5 cumulative ablation; held-out corpus and SOL B6 remain deferred.'}
    for attention,case,order in [('dense','B1',['adapter','fused']),('dense','B5',['fused','adapter']),
                                 ('dense','B6',['adapter','fused']),('sol','B5',['adapter','fused'])]:
        records={}
        for layout in order:
            name=f'{attention}-{case}-{layout}';run=out/name
            command=['python3',str(ROOT/'scripts/metal_native_bench.py'),'--binary',str(frozen/'bin/h3cli'),
                '--output',str(run),'--conditioning',str(a.conditioning.resolve()),'--backend','metal','--metal-attention',attention,
                '--metal-attention-kernel','steel-routed','--metal-attention-dtype','fp16','--metal-tier','reference',
                '--metal-attention-layout',layout,'--case',case,'--frames','243']
            if not a.conditioning.exists():command+=['--save-conditioning']
            if case=='B1':command+=['--regions','--capture-boundaries','--capture-steps']
            if case=='B6':command+=['--capture-steps','--full-vae']
            if attention=='sol':command+=['--sol-min-exact','.75','--sol-dense-steps','1','--sol-dense-sigma','.99']
            print('START '+name,flush=True)
            with (out/(name+'.driver.log')).open('x') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            r=json.loads((run/'record.json').read_text());records[layout]=r
            if not r['evaluation_contract_pass'] or r['geometry']!=[640,480,243] or r['blocks']!=50:raise ValueError('model contract failure')
            print('DONE '+name+' '+json.dumps(r['timing']),flush=True)
        left,right=records['adapter'],records['fused']
        for key in ('geometry','seed','prompt','evaluations','blocks','conditioning_sha256_after','preview_vae'):
            if left[key]!=right[key]:raise ValueError('unmatched '+key)
        equal={key:left[key]==right[key] for key in ('av_sha256','media_sha256')}
        if case in ('B1','B6'):equal['step_tensor_sha256']=left['step_tensor_sha256']==right['step_tensor_sha256']
        if case=='B1':
            captures={layout:{p.name:sha(p) for p in (out/f'{attention}-{case}-{layout}'/'boundaries').glob('*.bf16')} for layout in records}
            if not captures['adapter']:raise ValueError('missing boundaries')
            equal['block_boundaries']=captures['adapter']==captures['fused']
        # MP4 container metadata may vary; decoded state and every captured
        # sampler tensor must be identical. Media hashes are still retained.
        if not all(v for k,v in equal.items() if k!='media_sha256'):raise ValueError('fusion changed model arithmetic: '+str(equal))
        pair={'attention':attention,'case':case,'identity':equal,
              'record_sha256':{layout:sha(out/f'{attention}-{case}-{layout}'/'record.json') for layout in records},
              'timing':{layout:r['timing'] for layout,r in records.items()},
              'peak_gib':{layout:max(s['metal_peak_bytes'] for s in r['steps'])/2**30 for layout,r in records.items()}}
        if case=='B5':pair['steady_speedup']=left['timing']['steady_median']/right['timing']['steady_median']
        index['pairs'].append(pair);(out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
        print('PAIR PASS '+attention+' '+case+' '+json.dumps(equal),flush=True)
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
