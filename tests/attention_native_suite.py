#!/usr/bin/env python3
"""Bounded functional CUDA cases. Timing qualification is a separate run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--binary',default='./bin/attention_native')
p.add_argument('--output',required=True);p.add_argument('--lengths',nargs='+',type=int,default=[1,17,65,129,257,2281])
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);results=[]
for mode in ('sage2++','sage3'):
    for length in a.lengths:
        for pattern in ('zero','constant','random'):
            for layout in (0,1):
                name=f'{mode}-{length}-{pattern}-{layout}'
                command=[a.binary,mode,str(length),'-',str(layout),'1',pattern]
                r=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
                (out/(name+'.log')).write_text(r.stdout)
                row={'name':name,'argv':command,'returncode':r.returncode};results.append(row)
                print(json.dumps(row),flush=True)
                (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
                if r.returncode:print(r.stdout[-4000:],file=sys.stderr);raise SystemExit(1)
    command=[a.binary,mode,'129','-','0','1','nan']
    r=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
    (out/(mode+'-nan.log')).write_text(r.stdout);results.append({'name':mode+'-nan','argv':command,'returncode':r.returncode})
    if r.returncode:print(r.stdout[-4000:],file=sys.stderr);raise SystemExit(1)
(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print(f'PASS {len(results)} native functional cases')
