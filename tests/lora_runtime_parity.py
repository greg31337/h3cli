#!/usr/bin/env python3
"""Bounded real-model oracle: independently recompute sampled rows with NumPy.

This validation-only script imports the existing offline parser/folder. Runtime
h3cli and the host folder do not invoke it or require Python dependencies.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lora'))
import numpy as np
import fold_lora as oracle

p=argparse.ArgumentParser()
p.add_argument('--base',required=True)
p.add_argument('--native',required=True)
p.add_argument('--lora',action='append',required=True)
p.add_argument('--offline')
p.add_argument('--output',required=True)
a=p.parse_args()
start=time.monotonic()
base=oracle.Checkpoint(Path(a.base).resolve())
native=oracle.Checkpoint(Path(a.native).resolve())
offline=oracle.Checkpoint(Path(a.offline).resolve()) if a.offline else None
groups=defaultdict(list)
for i,arg in enumerate(a.lora):
    path,scale=oracle.parse_lora(arg)
    targets,record,stamp=oracle.adapter_targets(path,scale,base,adapter=i)
    for target in targets:groups[target.tensor].append(target)
metrics=oracle.ProbeMetrics();offline_metrics=oracle.ProbeMetrics()
recipe_metrics=oracle.ProbeMetrics();different_bf16_elements=0
rows=0;elements=0;max_ratio=0;records=[]
for name,targets in groups.items():
    tensor=base.tensors[name]
    selected=sorted(set([0,tensor.shape[0]//3,tensor.shape[0]//2,tensor.shape[0]-1]+[t.start for t in targets]+[t.stop-1 for t in targets]))
    arrays=[(t,t.a.read()) for t in targets]
    local=oracle.ProbeMetrics()
    for row in selected:
        expected=tensor.read(row,row+1);magnitude=np.abs(expected.astype(np.float64))
        for t,A in arrays:
            if not t.start<=row<t.stop or t.scale==0:continue
            B=t.b.read(row-t.start,row-t.start+1)
            delta=(B@A)*np.float32(t.scale)
            magnitude+=abs(t.scale)*(np.abs(B.astype(np.float64))@np.abs(A.astype(np.float64)))
            expected+=delta
        actual=native.tensors[name].read(row,row+1)
        bound=2e-5*magnitude+np.abs(expected.astype(np.float64))/256+1e-7
        difference=np.abs(actual.astype(np.float64)-expected)
        if not np.isfinite(actual).all() or np.any(difference>bound):raise RuntimeError(f'native matrix oracle mismatch: {name}, row {row}, max {difference.max()}')
        max_ratio=max(max_ratio,float(np.max(difference/bound)))
        metrics.add(actual,expected);local.add(actual,expected)
        # Execute the offline folder's final BF16 conversion as well as its
        # ordered NumPy deltas, without materializing a second 62-GiB tree.
        offline_recipe=(oracle.round_bf16(expected).astype(np.uint32)<<16).view(np.float32)
        recipe_metrics.add(actual,offline_recipe)
        different_bf16_elements+=int(np.count_nonzero(actual!=offline_recipe))
        if offline:offline_metrics.add(actual,offline.tensors[name].read(row,row+1))
        rows+=1;elements+=actual.size
    records.append({'tensor':name,'rows':selected,**local.finish()})
manifest=json.loads((Path(a.native).parent/'manifest.json').read_text())
result=dict(base=str(Path(a.base).resolve()),native=str(Path(a.native).resolve()),adapters=a.lora,
    tensors=len(groups),rows=rows,elements=elements,maximum_bound_fraction=max_ratio,
    native_vs_fp32=metrics.finish(),native_vs_offline=offline_metrics.finish() if offline else None,
    native_vs_offline_recipe=recipe_metrics.finish(),different_bf16_elements=different_bf16_elements,
    native_key=manifest['key'],native_shard_hashes=manifest['files'],matrix_results=records,
    seconds=time.monotonic()-start,quality='Not assessed by numerical validation')
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('matrix_results','native_shard_hashes')},indent=2))
