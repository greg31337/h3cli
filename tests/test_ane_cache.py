#!/usr/bin/env python3
"""Exercise actual CoreML graph cache integrity and cleanup on an ANE Mac."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,default=Path('bin/ane_probe'))
    a=p.parse_args();binary=a.binary.resolve();results=[]
    with tempfile.TemporaryDirectory(prefix='h3-ane-cache-') as temporary:
        root=Path(temporary)/'cache'
        def run(expected=True,shape=('1024','1024','257','256')):
            r=subprocess.run([str(binary),*shape],env={**os.environ,'H3_ANE_CACHE_DIR':str(root)},capture_output=True,text=True)
            if (r.returncode==0)!=expected:raise AssertionError(r.stderr)
            return r
        cold=run();assert 'cache 0' in cold.stderr
        warm=run();assert 'cache 1' in warm.stderr
        results.append({'case':'cold-warm','pass':True,'cold':cold.stderr,'warm':warm.stderr})
        run(shape=('1024','1024','513','512'))
        assert len(list(root.glob('*.mlmodelc')))==2
        path=next(root.glob('*.mlmodelc'));manifest=path/'h3-manifest.json';original=manifest.read_bytes()
        m=json.loads(original);m['identity']+='/other-os-or-recipe';manifest.write_text(json.dumps(m))
        # The selected path depends on shape; both shapes are checked.
        failures=0
        for chunk in ('256','512'):
            r=subprocess.run([str(binary),'1024','1024','257',chunk],env={**os.environ,'H3_ANE_CACHE_DIR':str(root)},capture_output=True,text=True)
            if r.returncode:
                assert 'integrity/recipe mismatch' in r.stderr;failures+=1
        assert failures==1;manifest.write_bytes(original)
        mil=path/'model.mil';original_mil=mil.read_bytes();mil.write_bytes(original_mil+b'\nchanged')
        failures=0
        for chunk in ('256','512'):
            r=subprocess.run([str(binary),'1024','1024','257',chunk],env={**os.environ,'H3_ANE_CACHE_DIR':str(root)},capture_output=True,text=True)
            if r.returncode:
                assert 'integrity/recipe mismatch' in r.stderr;failures+=1
        assert failures==1;mil.write_bytes(original_mil)
        results.append({'case':'shape-and-recipe-and-content-invalidation','pass':True})
        root.chmod(0o755);assert 'private directory' in run(False).stderr;root.chmod(0o700)
        filler=root/'quota-test';filler.write_bytes(b'0'*(65*1024*1024))
        assert 'cache is full' in run(False,('1024','1024','1025','1024')).stderr
        filler.unlink();run(shape=('1024','1024','1025','1024'))
        assert not list(root.glob('.stage-*'))
        results.append({'case':'private-cache-quota-cleanup','pass':True})
    assert not Path(temporary).exists()
    print(json.dumps({'tests':results,'pass':True},indent=2))

if __name__=='__main__':main()
