#!/usr/bin/env python3
"""Destructive artifact tests use only a fresh temporary cache."""
import os
import hashlib
import struct
from pathlib import Path
import subprocess
import tempfile

# This suite tests content-addressed recipe-1 corruption detection. Metadata
# cache structure/invalidation/no-hash behavior has its own host-only suite.
os.environ['H3_QUANT_VERIFY']='1'

def run(mode,cache,*args,env=None,success=True):
    r=subprocess.run(['./bin/quant_native',mode,str(cache),*args],env=env,capture_output=True,text=True,timeout=30)
    assert (r.returncode==0)==success,(args,r.returncode,r.stdout,r.stderr)
    return r

for mode in ('fp8','nvfp4'):
    with tempfile.TemporaryDirectory(prefix='h3-artifact-test-') as directory:
        cache=Path(directory)/'packed'
        # Two creators contend for the same lock and must publish one complete file.
        jobs=[subprocess.Popen(['./bin/quant_native',mode,str(cache)],stdout=subprocess.PIPE,stderr=subprocess.PIPE) for _ in range(2)]
        for p in jobs:
            out,err=p.communicate(timeout=30);assert p.returncode==0,(out,err)
        artifacts=list(cache.glob('*.h3q'));assert len(artifacts)==1
        artifact=artifacts[0];original=artifact.read_bytes()
        for name,payload in [('truncated',original[:-1]),('version',original[:8]+b'\xff'+original[9:]),('payload',original[:-1]+bytes([original[-1]^1]))]:
            artifact.write_bytes(payload);result=run(mode,cache,success=False);assert 'corrupt/incompatible' in result.stderr,name
            artifact.write_bytes(original)
        # Recompute the checksum to exercise semantic validation, not just hashing.
        scale,global_scale=struct.unpack_from('<QQ',original,32)
        for offset,value in [(128+global_scale,b'\x00\x00\xc0\x7f'),(128+(0 if mode=='fp8' else scale),b'\x7f')]:
            payload=bytearray(original);payload[offset:offset+len(value)]=value
            payload[80:112]=hashlib.sha256(payload[128:]).digest();artifact.write_bytes(payload)
            result=run(mode,cache,success=False);assert 'scale' in result.stderr or 'nonfinite' in result.stderr
            artifact.write_bytes(original)
        run(mode,cache,'source-shift');assert len(list(cache.glob('*.h3q')))==2
        run(mode,cache,'zero-weight');run(mode,cache,'nonfinite')
        run(mode,cache,'bad-range');run(mode,cache,'bad-shape')
        run(mode,cache,env={**os.environ,'H3_CUDA_WEIGHT_MODE':'stream'})
        run(mode,cache,env={**os.environ,'H3_QUANT_TEST_DEFAULT':'1'})
        assert not list(cache.glob('*.tmp.*'))
        print('PASS',mode,'concurrent create, corruption/truncation/version, source identity, zero, nonfinite, shape/range, compressed stream, default CUDA')
