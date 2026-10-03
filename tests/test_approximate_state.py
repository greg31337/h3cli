#!/usr/bin/env python3
"""Rechecksummed adversarial tests of real adaptive/SubBlock checkpoints."""
import argparse
import copy
import json
from pathlib import Path
import struct
import subprocess
import tempfile
from test_sampler_file import entries,build


def main():
    p=argparse.ArgumentParser();p.add_argument('binary',type=Path);p.add_argument('state',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    binary=a.binary.resolve();original=a.state.read_bytes();parts=entries(original);ids={x[0] for x in parts};assert {35,40,41,42,43,44}<=ids
    records=[]
    with tempfile.TemporaryDirectory(prefix='h3cli-approximate-state-') as temp:
        path=Path(temp)/'state.h3sample'
        def check(name,data,accepted=False):
            path.write_bytes(data);r=subprocess.run([str(binary),'--load',str(path)],capture_output=True,text=True)
            records.append(dict(name=name,accepted=r.returncode==0,expected=accepted,error=r.stderr.strip()))
            assert (r.returncode==0)==accepted,(name,r.stderr)
        check('original',original,True)
        for kind in [35,40,41,42,43,44]:
            check(f'missing-{kind}',build([p for p in parts if p[0]!=kind]))
            for field,value in [(1,99),(2,0)]:
                altered=copy.deepcopy(parts);next(p for p in altered if p[0]==kind)[field]=value
                check(f'section-{kind}-field-{field}',build(altered))
        for kind,offset,fmt,value in [(40,0,'I',99),(40,4,'I',99),(40,8,'I',2),(40,12,'I',99),(40,16,'I',2),(40,20,'i',-1),
                (40,24,'i',-1),(40,28,'Q',2**63),(40,44,'f',float('nan')),(40,44,'f',-1.),(40,48,'i',0),(40,48,'i',17),(41,0,'H',0x7fc0),(42,0,'H',0x7f80),
                (43,0,'I',99),(43,4,'I',99),(43,8,'I',32),(43,12,'I',8),(43,16,'I',9),(43,20,'f',float('nan')),(43,20,'f',1.)]:
            altered=copy.deepcopy(parts);payload=next(p[-1] for p in altered if p[0]==kind);struct.pack_into('<'+fmt,payload,offset,value)
            check(f'payload-{kind}-{offset}',build(altered))
        descriptors=next(p[-1] for p in parts if p[0]==9)
        for ref in range(len(descriptors)//20):
            for offset,value in [(0,99),(4,10000),(8,4096),(12,4096),(16,1000000)]:
                altered=copy.deepcopy(parts);payload=next(p[-1] for p in altered if p[0]==9)
                struct.pack_into('<i',payload,ref*20+offset,value)
                check(f'reference-{ref}-{offset}',build(altered))
        if descriptors:
            altered=copy.deepcopy(parts);payload=next(p[-1] for p in altered if p[0]==10)
            segments=struct.unpack_from('<Q',payload,8)[0];offset=76+segments*20
            struct.pack_into('<d',payload,offset,struct.unpack_from('<d',payload,offset)[0]+.125)
            check('forged-rope-time',build(altered))
            altered=copy.deepcopy(parts);payload=next(p[-1] for p in altered if p[0]==4)
            prompt_bytes=struct.unpack_from('<Q',payload)[0];provenance=16+prompt_bytes
            struct.pack_into('<Q',payload,provenance+8,99)
            check('forged-provenance-kind',build(altered))
            for version in (1,2):
                altered=copy.deepcopy(parts);next(p for p in altered if p[0]==40)[1]=version
                check(f'removed-adaptive-section-{version}',build(altered))
        altered=bytearray(original);altered[-1]^=1;check('checksum',altered)
    a.out.write_text(json.dumps(dict(passed=True,cases=records),indent=2)+'\n');print(f'PASS {len(records)} approximate checkpoint integrity cases')

if __name__=='__main__':main()
