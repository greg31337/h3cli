#!/usr/bin/env python3
"""Run independent/upstream parity on captured real DiT QKV across evaluations."""
import argparse,json,re,sys
from pathlib import Path
import attention_oracle
from attention_run import sha
p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path)
p.add_argument('--log',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
p.add_argument('--binary',default='./bin/attention_native');p.add_argument('--remove-validated-inputs',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
record=a.output/'results.json'
rows=json.loads(record.read_text()) if record.exists() else []
provenance=a.output/'provenance.json'
identity={'binary_sha256':sha(a.binary),'segments_log_sha256':sha(a.log),
          'oracle_sha256':sha(Path(attention_oracle.__file__))}
if provenance.exists():
    if json.loads(provenance.read_text())!=identity:
        raise RuntimeError('real-QKV evidence identity changed; use a fresh output directory')
elif rows:
    raise RuntimeError('existing real-QKV evidence lacks provenance; use a fresh output directory')
else:provenance.write_text(json.dumps(identity,indent=2)+'\n')
passed={(r['step'],r['block'],r['sequence']):r for r in rows if r['pass']}
files=sorted(a.input.glob('step-*-block-*-s-*.qkv'))
if not files and not rows:raise RuntimeError('no captured QKV files')
for path in files:
    match=re.fullmatch(r'step-(\d+)-block-(\d+)-s-(\d+)\.qkv',path.name)
    if not match:raise RuntimeError(path)
    step,block,seq=map(int,match.groups());dest=a.output/path.stem
    if (step,block,seq) in passed:
        previous=passed[step,block,seq]['results']
        input_hash=sha(path)
        if len(previous)!=4 or any(r['input_sha256']!=input_hash or r.get('binary_sha256')!=identity['binary_sha256'] for r in previous):
            raise RuntimeError('captured input or native build changed: '+str(path))
        if a.remove_validated_inputs:path.unlink()
        continue
    sys.argv=['attention_oracle.py','--binary',a.binary,'--output',str(dest),'--input',str(path),
        '--lengths',str(seq),'--patterns','captured','--segments-log',str(a.log)]
    code=attention_oracle.main()
    results=json.loads((dest/'results.json').read_text())
    rows=[r for r in rows if (r['step'],r['block'],r['sequence'])!=(step,block,seq)]
    rows.append({'step':step,'block':block,'sequence':seq,'pass':code==0,'results':results})
    (a.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    if code:raise RuntimeError(f'port parity failed at {path}')
    if a.remove_validated_inputs:path.unlink()
print(f'PASS {len(rows)} real block/evaluation QKV captures')
