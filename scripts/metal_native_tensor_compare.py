#!/usr/bin/env python3
"""Compare retained M1 projection/block or per-step velocity/latent tensors."""
import argparse
import json
from pathlib import Path
import subprocess
from metal_native_bench import ROOT,sha

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('reference',type=Path)
p.add_argument('candidate',type=Path)
p.add_argument('--sol',action='store_true')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
tool=ROOT/'bin/metal_tensor_metrics'
tool.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['cc','-O3',str(ROOT/'tests/metal_tensor_metrics.c'),'-lm','-o',str(tool)],check=True)
limits=json.loads((ROOT/'tests/metal_native_limits.json').read_text())
files=sorted(p for p in a.reference.rglob('*') if p.suffix in ('.bf16','.f32'))
if not files:raise SystemExit('No captured tensors')
rows=[]
for path in files:
    relative=path.relative_to(a.reference);other=a.candidate/relative
    if not other.is_file() or path.stat().st_size!=other.stat().st_size:raise SystemExit('Missing/mismatched tensor: '+str(relative))
    size=4 if path.suffix=='.f32' else 2
    if path.stat().st_size%size:raise SystemExit('Misaligned tensor: '+str(relative))
    cmd=[str(tool.resolve()),str(path),str(other)]+(['f32'] if size==4 else [])
    metrics=json.loads(subprocess.run(cmd,capture_output=True,text=True,check=True).stdout)
    gate=('sol_model_audio' if 'audio' in path.name else 'sol_model_video') if size==4 and a.sol else 'sol_block' if a.sol else 'dense_model'
    if path.name.startswith('qkv'):gate='dense_component'
    limit=limits[gate]
    passed=metrics['relative_l2']<=limit['relative_l2'] and metrics['cosine']>=limit.get('min_cosine',-1)
    rows.append({'tensor':str(relative),'reference_sha256':sha(path),'candidate_sha256':sha(other),
                 'gate':gate,'metrics':metrics,'pass':passed})
result={'reference':str(a.reference),'candidate':str(a.candidate),'limits_sha256':sha(ROOT/'tests/metal_native_limits.json'),
        'pass':all(row['pass'] for row in rows),'tensors':rows}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if not result['pass']:raise SystemExit('Captured tensor numerical gate failed')
