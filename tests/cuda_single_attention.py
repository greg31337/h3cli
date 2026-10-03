import argparse,sys,json,subprocess,re,time
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser(description='Compare CUDA attention against recorded production QKV')
parser.add_argument('source',type=Path);parser.add_argument('out',type=Path)
parser.add_argument('--qkv',type=Path,required=True);parser.add_argument('--layout',type=Path)
args=parser.parse_args();src=args.source.resolve();out=args.out.resolve();out.mkdir()
sys.path.insert(0,str(src/'tests'))
from cuda_reference_regression import runtime_env,sha,write
env=runtime_env()|{'H3_TEST_SHARED_POLICY':'1'}
records=[]
def run(name,mode,seq,layout,pattern='random',meta=None,extra=None):
 command=[str(src/'bin/attention_native'),mode,str(seq),str(out/(name+'.bf16')),str(layout),'1',str(pattern)]
 if meta:command.append(str(meta))
 start=time.monotonic()
 with (out/(name+'.log')).open('w') as f:r=subprocess.run(command,cwd=src,env=env|(extra or {}),stdout=f,stderr=subprocess.STDOUT,timeout=180)
 log=(out/(name+'.log')).read_text();rows=[json.loads(x) for x in log.splitlines() if x.startswith('{')]
 item=dict(name=name,command=command,returncode=r.returncode,seconds=time.monotonic()-start,statistics=rows,passed=r.returncode==0 and len(rows)==1)
 if (out/(name+'.bf16')).exists():item['sha256']=sha(out/(name+'.bf16'))
 records.append(item);write(out/'progress.json',dict(complete=False,records=records));assert item['passed'],name
 return out/(name+'.bf16')
# Same real QKV is used by the native shared oracle and every approximate engine.
qkv=args.qkv.resolve();seq=int(re.search(r'-s-(\d+)\.',qkv.name)[1]);meta=args.layout.resolve() if args.layout else qkv.parent/'layout.bin'
for layout in (0,1):
 reference=run('real-reference-'+str(layout),'default',seq,layout,qkv,extra={'H3_TEST_ATTENTION_FAST':'0'})
 for mode in ['default','sage2++','sage3','sol']:
  candidate=run('real-'+mode+'-'+str(layout),mode,seq,layout,qkv,meta if mode=='sol' else None)
  norm=diff=dot=cnorm=0.;peak=maxdiff=0.
  a=np.memmap(reference,dtype='<u2',mode='r');b=np.memmap(candidate,dtype='<u2',mode='r');assert a.shape==b.shape
  for start in range(0,a.size,1<<20):
   x=(np.array(a[start:start+(1<<20)],dtype=np.uint32)<<16).view(np.float32).astype(np.float64);y=(np.array(b[start:start+(1<<20)],dtype=np.uint32)<<16).view(np.float32).astype(np.float64)
   assert np.isfinite(x).all() and np.isfinite(y).all();d=y-x;norm+=x@x;diff+=d@d;dot+=x@y;cnorm+=y@y;peak=max(peak,float(abs(x).max()));maxdiff=max(maxdiff,float(abs(d).max()))
  records[-1]['diagnostic_error']=dict(relative_l2=float(np.sqrt(diff/max(norm,1e-30))),cosine=float(dot/max(np.sqrt(norm*cnorm),1e-30)),max_relative=maxdiff/max(peak,1e-30))
  if mode=='default':assert sha(reference)==sha(candidate)
 forced=run('real-sol-forced-dense-'+str(layout),'sol',seq,layout,qkv,meta,{'H3_TEST_SOL_MIN_EXACT':'1'})
 assert sha(forced)==sha(reference)
for seq in [1,63,65,129,257]:
 for layout in (0,1):
  for mode in ['default','sage2++','sage3','sol']:run('tail-'+str(seq)+'-'+mode+'-'+str(layout),mode,seq,layout)
for mode in ['sage2++','sage3','sol']:
 for pattern in ['zero','constant']:run(mode+'-'+pattern,mode,129,0,pattern)
write(out/'result.json',dict(passed=True,complete=True,records=records,real_qkv=dict(path=str(qkv),sha256=sha(qkv),layout_sha256=sha(meta)),binary_sha256=sha(src/'bin/attention_native'),notes=['Real operator errors are diagnostic; full-video promotion uses the unchanged fast-quality contract.','Q/K/V inputs are sequence-major; the head-major flag selects only output layout.']))
print('PASS',len(records),'operator cases')
