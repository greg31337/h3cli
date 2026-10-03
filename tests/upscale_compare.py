#!/usr/bin/env python3
"""Compare native layer captures against the previously frozen BF16 contract."""
import argparse
import hashlib
import json
from pathlib import Path
import array
import math

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--native', type=Path, required=True)
a=p.parse_args()
root=Path(__file__).resolve().parent
contract_path=root/'upscale/contract.json'
c=json.loads(contract_path.read_text())
m=json.loads((a.reference/'manifest.json').read_text())
assert hashlib.sha256(contract_path.read_bytes()).hexdigest()==m['contract_sha256']
rows=[]
for case in c['fixture_cases']:
    for ref in sorted((a.reference/case['id']/'bf16').glob('*.f32')):
        isolated=ref.stem.startswith(('in_blocks.', 'out_blocks.')) or ref.stem in ('norm_out','conv_out')
        native=a.native/case['id']/('isolated' if isolated else '')/ref.name
        rb=ref.read_bytes()
        assert hashlib.sha256(rb).hexdigest()==m['files'][str(ref.relative_to(a.reference))]['sha256']
        nb=native.read_bytes()
        assert len(nb)==len(rb)
        r=array.array('f');r.frombytes(rb)
        x=array.array('f');x.frombytes(nb)
        assert all(math.isfinite(v) for v in x)
        rms=max(math.sqrt(sum(v*v for v in r)/len(r)),c['tolerances']['rms_floor'])
        nrmse=math.sqrt(sum((u-v)**2 for u,v in zip(x,r))/len(r))/rms
        maximum=max(abs(u-v) for u,v in zip(x,r))/rms
        tol=c['tolerances']['bf16_network' if ref.stem=='output' else 'bf16_layer']
        passed=nrmse<=tol['nrmse'] and maximum<=tol['max_rms_error']
        rows.append(dict(case=case['id'],layer=ref.stem,scope='isolated layer' if isolated else 'end-to-end',nrmse=nrmse,max_rms_error=maximum,passed=passed,
                         native_sha256=hashlib.sha256(nb).hexdigest()))
result=dict(passed=all(r['passed'] for r in rows),layers=len(rows),results=rows,
            contract_sha256=m['contract_sha256'],reference_manifest_sha256=hashlib.sha256((a.reference/'manifest.json').read_bytes()).hexdigest())
(a.native/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(passed=result['passed'],layers=len(rows),worst=sorted(rows,key=lambda r:r['nrmse'],reverse=True)[:10]),indent=2))
raise SystemExit(0 if result['passed'] else 1)
