#!/usr/bin/env python3
"""Replay actual oracle QKV through the native reference attention API."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
from cuda_sglang_compare import oracle,native,metric

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('oracle',type=Path);p.add_argument('output',type=Path);p.add_argument('--library',required=True);a=p.parse_args()
 a.output.mkdir(exist_ok=False,parents=True);results=[]
 contract_path=Path(__file__).with_name('cuda_sglang_contract.json');contract=json.loads(contract_path.read_text())
 gate=contract['gates']['operation']
 env=os.environ.copy();env['H3_SGLANG_CUBLAS_LIBRARY']=a.library
 binary=a.source.resolve()/'bin/cuda_sglang_replay'
 for label,prefix,mode in [('qwen-0','qwen-0','gqa'),('dit-0','step-000','attention')]:
  root=a.output/label;root.mkdir();shapes=[];files={}
  for key in ('q','k','v'):
   x=oracle(a.oracle/'capture',prefix+'.attention.'+key)
   if x.ndim!=4 or x.shape[0]!=1:raise ValueError('unsupported captured attention batch')
   # Head-major oracle -> native row-major ABI, without changing dtype.
   x=np.ascontiguousarray(x[0].transpose(1,0,2),dtype='<f4');shapes.append(x.shape)
   bits=x.view('<u4')
   if np.any(bits&65535):raise ValueError('expected exact BF16 QKV')
   data=(bits>>16).astype('<u2').tobytes();path=root/(key+'.bf16');path.write_bytes(data);files[path.name]=hashlib.sha256(data).hexdigest()
  rows,heads,dim=shapes[0];kv=shapes[1][1]
  if shapes[1]!=shapes[2] or shapes[1]!=(rows,kv,dim):raise ValueError('incompatible QKV shape')
  command=[str(binary),mode,*[str(root/(k+'.bf16')) for k in ('q','k','v')],str(root/'output.bf16'),str(rows),str(heads),str(kv),str(dim)]
  with (root/'render.log').open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  x=native(root,'output.bf16','<u2').reshape(rows,heads,dim)
  y=oracle(a.oracle/'capture',prefix+'.attention.output')[0].transpose(1,0,2)
  metrics=metric(x,y);limit=gate['absolute_floor']+gate['absolute_max_per_reference_rms']*metrics['reference_rms']
  passed=metrics['finite'] and metrics['relative_l2']<=gate['relative_l2_max'] and metrics['cosine']>=gate['cosine_min'] and metrics['max_abs']<=limit
  result=dict(name=label,command=command,inputs=files,metrics=metrics,absolute_limit=limit,passed=passed);results.append(result);print(result,flush=True)
 (a.output/'metrics.json').write_text(json.dumps(dict(domain='same-input attention replay',binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),contract_sha256=hashlib.sha256(contract_path.read_bytes()).hexdigest(),passed=all(r['passed'] for r in results),results=results),indent=2)+'\n')
if __name__=='__main__':main()
