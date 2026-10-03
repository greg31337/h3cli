#!/usr/bin/env python3
"""Serial M3C QKV split ablation; 243 frames and at most six evaluations/run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from metal_native_bench import ROOT,sha

MODES=('off','serial','static','dynamic')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--conditioning',type=Path,required=True)
    a=p.parse_args();frozen=a.frozen.resolve();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((frozen/'build-provenance.json').read_text())
    if sha(frozen/'bin/h3cli')!=provenance['binary_sha256']:p.error('changed frozen executable')
    index_path=out/'index.json'
    index=json.loads(index_path.read_text()) if index_path.exists() else {
        'schema':1,'geometry':[640,480,243],'binary_sha256':sha(frozen/'bin/h3cli'),
        'script_sha256':sha(__file__),'runs':{},'scope':'T2VA calibration B1/B5/B6; selected QKV row shards only',
        'unrun':['held-out conditioning/continuation corpus','50-step rendering','M5','MLP/output projection offload']}
    if index['binary_sha256']!=sha(frozen/'bin/h3cli') or index['script_sha256']!=sha(__file__):p.error('changed suite identity')
    env={**os.environ,'H3_SHADER_PATH':str(Path(provenance['source_snapshot'])/'src/metal/shaders.metal'),
         'H3_ANE_CACHE_DIR':str(out/'graph-cache'),'DEVELOPER_DIR':'/Applications/Xcode.app/Contents/Developer'}
    env.pop('MTL_DEBUG_LAYER',None)
    if not index.get('operators_sha256'):
        with (out/'operators.jsonl').open('x') as result,(out/'operators.log').open('x') as log:
            subprocess.run([str(frozen/'bin/ane_tests')],cwd=ROOT,env={**env,'MTL_DEBUG_LAYER':'1'},stdout=result,stderr=log,check=True)
        records=[json.loads(line) for line in (out/'operators.jsonl').read_text().splitlines()]
        if len(records)!=4 or not all(r['pass'] for r in records):raise ValueError('operator failure')
        index['operators_sha256']=sha(out/'operators.jsonl');index['operator_binary_sha256']=sha(frozen/'bin/ane_tests')
        index_path.write_text(json.dumps(index,indent=2)+'\n')
    # Rotate order across gates; B5 has no capture/component fences or tracing.
    for case,order in [('B1',MODES),('B5',('dynamic','static','serial','off')),('B6',MODES)]:
        for mode in order:
            name=f'{case}-{mode}';run=out/name
            if name in index['runs']:
                if sha(run/'record.json')!=index['runs'][name]['record_sha256']:raise ValueError('changed retained run')
                continue
            if (run/'run.log').exists():raise ValueError('unindexed run exists; inspect before restarting: '+name)
            command=['python3',str(ROOT/'scripts/metal_native_bench.py'),'--binary',str(frozen/'bin/h3cli'),
                '--output',str(run),'--conditioning',str(a.conditioning.resolve()),'--backend','metal','--metal-attention','sol',
                '--metal-attention-kernel','steel-routed','--metal-attention-dtype','fp16','--metal-tier','reference',
                '--metal-attention-layout','adapter','--metal-ane',mode,'--metal-ane-rows','4096','--metal-ane-chunk','512',
                '--sol-min-exact','.75','--sol-dense-steps','1','--sol-dense-sigma','.99','--case',case,'--frames','243']
            if not a.conditioning.exists():command+=['--save-conditioning']
            if case=='B1':command+=['--regions','--capture-boundaries','--capture-steps']
            if case=='B1' and mode=='static':command+=['--trace','--trace-window','30']
            if case=='B6':command+=['--full-vae','--capture-steps']
            print('START '+name,flush=True)
            with (out/(name+'.driver.log')).open('x') as log:
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if not (run/'record.json').exists():raise ValueError('missing run record: '+name)
            r=json.loads((run/'record.json').read_text())
            if result.returncode or not r['evaluation_contract_pass'] or r['geometry']!=[640,480,243] or r['blocks']!=50:
                raise ValueError('model contract failure: '+name)
            index['runs'][name]={'record_sha256':sha(run/'record.json'),'log_sha256':sha(run/'run.log'),
                'timing':r['timing'],'hardware_ready':r.get('ane_hardware_ready'),
                'offloaded_blocks':r.get('ane_offloaded_blocks',0),'recovered_blocks':r.get('ane_recovered_blocks',0)}
            index_path.write_text(json.dumps(index,indent=2)+'\n')
            print('DONE '+name+' '+json.dumps(index['runs'][name]),flush=True)
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
