#!/usr/bin/env python3
"""Checksummed corruption and CLI override checks on a real custom-warmup state."""
import argparse
import copy
import json
from pathlib import Path
import struct
import subprocess
import tempfile
from test_sampler_file import entries,build


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','cli','state','out'):parser.add_argument(name,type=Path)
    args=parser.parse_args();binary=args.binary.resolve();cli=args.cli.resolve()
    original=args.state.read_bytes();parts=entries(original);policy=next(p for p in parts if p[0]==44)
    assert policy[1:3]==[2,1]
    warmups=struct.unpack_from('<2i',policy[-1],len(policy[-1])-8)
    assert any(warmups) and all(x==0 or 2<=x<=16 for x in warmups)
    total=struct.unpack_from('<i',next(p[-1] for p in parts if p[0]==1),4)[0]
    records=[]
    with tempfile.TemporaryDirectory(prefix='h3cli-warmup-state-') as temp:
        path=Path(temp)/'state.h3sample'
        def check(name,data,accepted=False):
            path.write_bytes(data);p=subprocess.run([str(binary),'--load',str(path)],capture_output=True,text=True)
            records.append(dict(name=name,accepted=p.returncode==0,expected=accepted,error=p.stderr.strip()))
            assert (p.returncode==0)==accepted,(name,p.stderr)
        check('original',original,True)
        check('missing-warmup-policy',build([p for p in parts if p[0]!=44]))
        for index,value in [(1,1),(1,99),(2,0)]:
            changed=copy.deepcopy(parts);next(p for p in changed if p[0]==44)[index]=value
            check(f'policy-header-{index}-{value}',build(changed))
        for index in (0,1):
            for value in (-1,1,17,total-1):
                changed=copy.deepcopy(parts);data=next(p[-1] for p in changed if p[0]==44)
                struct.pack_into('<i',data,len(data)-8+index*4,value)
                check(f'warmup-{index}-{value}',build(changed))
        changed=copy.deepcopy(parts);p=next(p for p in changed if p[0]==44);p[-1]=p[-1][:-8];p[4]=len(p[-1])
        check('truncated-warmups',build(changed))
        if 43 in [p[0] for p in parts]:
            changed=copy.deepcopy(parts);data=next(p[-1] for p in changed if p[0]==43)
            old=struct.unpack_from('<I',data,16)[0];struct.pack_into('<I',data,16,3 if old==2 else 2)
            check('subblock-policy-disagreement',build(changed))
        # Resuming without warmup flags restores them. Explicit conflicts fail
        # before model loading; the original schedule governs range checking.
        for index,flag in enumerate(('--adaptive-cache-warmup','--subblock-warmup')):
            value=warmups[index];wrong=3 if value==2 else 2
            p=subprocess.run([str(cli),'-d','/missing-warmup-model','--resume-sampler-state',str(args.state.resolve()),flag,str(wrong)],capture_output=True,text=True)
            assert p.returncode!=0 and 'differs from checkpoint' in p.stderr,(flag,p.stderr)
            records.append(dict(name=flag+'-conflict',passed=True))
    args.out.write_text(json.dumps(dict(passed=True,warmups=warmups,cases=records),indent=2)+'\n')
    print(f'PASS {len(records)} warmup checkpoint/override checks')


if __name__=='__main__':main()
