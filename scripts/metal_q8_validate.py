#!/usr/bin/env python3
"""Serial Q8 ablations, fixed 640x480/243 frames, 50 blocks, <=6 evaluations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from metal_native_bench import ROOT, sha

MODES = ('reference-plus', 'dense-bf16', 'dense-q8', 'sol-bf16', 'sol-q8')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, default=ROOT/'models/MiniMax-H3')
    p.add_argument('--cases', nargs='+', choices=['B1','B5','B6'], default=['B1','B5','B6'])
    p.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    a = p.parse_args(); frozen=a.frozen.resolve(); out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((frozen/'build-provenance.json').read_text())
    if sha(frozen/'bin/h3cli')!=provenance['binary_sha256']:p.error('changed frozen executable')
    for name,digest in provenance['source_sha256'].items():
        if sha(Path(provenance['source_snapshot'])/name)!=digest:p.error('changed frozen source')
    index_path=out/'index.json'
    identity={'schema':1,'geometry':[640,480,243],'maximum_evaluations':6,
              'binary_sha256':sha(frozen/'bin/h3cli'),'script_sha256':sha(__file__),
              'bench_sha256':sha(ROOT/'scripts/metal_native_bench.py'),
              'contract_sha256':sha(ROOT/'tests/metal_q8_limits.json')}
    index=json.loads(index_path.read_text()) if index_path.exists() else {**identity,'runs':{}}
    if any(index[k]!=v for k,v in identity.items()):p.error('changed suite identity; use a new output')
    env={**os.environ,'H3_TEST_MAX_EVALUATIONS':'6','H3_DIT_COMMAND_BLOCKS':'5'}
    for key in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'):env.pop(key,None)
    conditioning=out/'conditioning.h3cond'
    index_path.write_text(json.dumps(index,indent=2)+'\n')
    for case in a.cases:
        # Rotate direction for steady-state runs; no concurrent GPU work.
        order=list(a.modes)
        if case=='B5':order.reverse()
        for mode in order:
            name=case+'-'+mode; run=out/name
            if name in index['runs']:
                saved=index['runs'][name]
                if any(sha(run/f)!=v for f,v in saved['sha256'].items()):raise ValueError('changed retained run '+name)
                continue
            if (run/'run.log').exists():raise ValueError('unindexed run exists; inspect before retry: '+name)
            command=[sys.executable,str(ROOT/'scripts/metal_native_bench.py'),
                     '--binary',str(frozen/'bin/h3cli'),'--model',str(a.model.resolve()),
                     '--output',str(run),'--conditioning',str(conditioning),
                     '--case',case,'--width','640','--height','480','--frames','243']
            if not conditioning.exists():command+=['--save-conditioning']
            if mode=='reference-plus':command+=['--backend','mpsgraph']
            else:
                attention,weight=mode.split('-')
                command+=['--backend','metal','--metal-attention',attention,
                          '--metal-attention-kernel','steel-routed','--metal-attention-dtype','fp16',
                          '--metal-attention-layout','adapter','--metal-tier','reference',
                          '--metal-weight-format',weight,'--metal-q8-kernel','mpsgraph',
                          '--sol-min-exact','.75','--sol-dense-steps','1','--sol-dense-sigma','.99']
            if case=='B1':command+=['--capture-steps']
            if case=='B6':command+=['--full-vae','--capture-steps']
            print('START '+name,flush=True)
            with (out/(name+'.driver.log')).open('x') as log:
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if not (run/'record.json').exists():raise ValueError('missing run record '+name)
            r=json.loads((run/'record.json').read_text())
            if result.returncode or not r['evaluation_contract_pass'] or r['geometry']!=[640,480,243] or r['blocks']!=50:
                raise ValueError('execution contract failed '+name)
            index['runs'][name]={'sha256':{f:sha(run/f) for f in ('record.json','run.log','bin/h3cli','result.h3av','result.mp4')},
                                 'timing':r['timing']}
            index_path.write_text(json.dumps(index,indent=2)+'\n')
            print('DONE '+name+' '+json.dumps(r['timing']),flush=True)

if __name__=='__main__':main()
