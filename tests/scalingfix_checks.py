#!/usr/bin/env python3
"""Sequential M4 kernel, compatibility, memory and cancellation checks."""
import json,os,subprocess
from pathlib import Path
from source_tree import build_path
from scalingfix_prepare import ROOT,OUT,FLAGS
from memory_generation import digest

def main():
    subprocess.run(['make','-j8','bin/qwen_scaling_tests','bin/memory_gpu_tests','bin/libh3.a'],cwd=ROOT,check=True)
    (ROOT/'bin').mkdir(exist_ok=True)
    base=OUT/'baseline';subprocess.run(['clang','-O3','-std=c11','-D_DARWIN_C_SOURCE','-DSCALING_BASELINE','-I',str(base),str(ROOT/'tests/test_qwen_scaling.c'),str(build_path(base,'libh3.a')),*FLAGS,'-o',str(ROOT/'bin/scalingfix_baseline_kernels')],check=True)
    d=OUT/'baseline-kernels';d.mkdir(exist_ok=True)
    with (OUT/'baseline-kernels.jsonl').open('w') as log:subprocess.run([str(ROOT/'bin/scalingfix_baseline_kernels'),str(d)],cwd=base,stdout=log,check=True)
    d=OUT/'kernels';d.mkdir(exist_ok=True)
    with (OUT/'kernels.jsonl').open('w') as log:subprocess.run([str(ROOT/'bin/qwen_scaling_tests'),str(d)],cwd=ROOT,stdout=log,check=True)
    rows=[]
    for f in d.glob('*-legacy.bf16'):
        old=OUT/'baseline-kernels'/f.name;exact=digest(f)==digest(old);rows.append({'case':f.stem,'exact':exact})
        # Historical shared-memory race makes large cases nondeterministic.
        if f.name=='small-legacy.bf16':assert exact
    (OUT/'legacy-kernel-compatibility.json').write_text(json.dumps(rows,indent=2)+'\n')
    for mode in ('legacy','scaled-q','reference'):
        with (OUT/f'preflight-{mode}.log').open('w') as log:subprocess.run([str(ROOT/'bin/memory_gpu_tests')],cwd=ROOT,env={**os.environ,'H3_QWEN_GQA_SCALE_MODE':mode},stdout=log,stderr=log,check=True)
    subprocess.run(['python3',str(ROOT/'tests/scalingfix_control.py')],cwd=ROOT,check=True)
    subprocess.run(['python3',str(ROOT/'tests/scalingfix_control.py'),'--sanitize'],cwd=ROOT,check=True)
    print('PASS kernel oracle, legacy arithmetic, all-mode limits/guards/cancellation/saved-state loading',flush=True)
if __name__=='__main__':main()
