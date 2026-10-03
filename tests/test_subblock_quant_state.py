#!/usr/bin/env python3
"""Rechecksummed corruption checks for real SubBlock/quantized checkpoints."""
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
    for name in ('binary','state','out'):parser.add_argument(name,type=Path)
    args=parser.parse_args();binary=args.binary.resolve();original=args.state.read_bytes();parts=entries(original)
    ids={p[0] for p in parts};assert {34,35,43,44}<=ids and not {40,41,42}&ids
    assert struct.unpack_from('<2I',next(p[-1] for p in parts if p[0]==34))[1]==4
    assert struct.unpack_from('<3I',next(p[-1] for p in parts if p[0]==35))==(4,2,1)
    records=[]
    with tempfile.TemporaryDirectory(prefix='h3cli-subblock-quant-state-') as temp:
        path=Path(temp)/'state.h3sample'
        def check(name,data,accepted=False):
            path.write_bytes(data);r=subprocess.run([str(binary),'--load',str(path)],capture_output=True,text=True)
            records.append(dict(name=name,accepted=r.returncode==0,expected=accepted,error=r.stderr.strip()))
            assert (r.returncode==0)==accepted,(name,r.stderr)
        check('original',original,True)
        for kind in [34,35,43,44]:
            check(f'missing-{kind}',build([p for p in parts if p[0]!=kind]))
            for field,value in [(1,99),(2,0)]:
                changed=copy.deepcopy(parts);next(p for p in changed if p[0]==kind)[field]=value
                check(f'section-{kind}-field-{field}',build(changed))
        check('missing-attention-and-plan',build([p for p in parts if p[0] not in (35,43)]))
        for kind,offset,fmt,value in [
            (34,0,'I',0),(34,0,'I',99),(34,4,'I',2),(34,4,'I',3),
            (35,0,'I',0),(35,0,'I',3),(35,4,'I',1),(35,8,'I',2),
            (43,0,'I',1),(43,4,'I',99),(43,8,'I',32),(43,12,'I',8),
            (43,16,'I',9),(43,20,'f',float('nan')),(43,20,'f',float('inf')),(43,20,'f',1.),
        ]:
            changed=copy.deepcopy(parts);struct.pack_into('<'+fmt,next(p[-1] for p in changed if p[0]==kind),offset,value)
            check(f'payload-{kind}-{offset}-{value}',build(changed))
        changed=bytearray(original);changed[-1]^=1;check('checksum',changed)
    args.out.write_text(json.dumps(dict(passed=True,cases=records),indent=2)+'\n')
    print(f'PASS {len(records)} SubBlock/quantization checkpoint integrity cases')


if __name__=='__main__':main()
