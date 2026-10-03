#!/usr/bin/env python3
"""Serial native placement qualification; compare payloads, never provenance headers.

Run with the pinned CUDA environment already active. Each run is limited to six
full-depth evaluations. All failed attempts and physical resource samples remain
in the output directory. Test caps reduce residency without bypassing admission.
"""
import argparse,hashlib,json,os,re,struct,subprocess,threading,time
from pathlib import Path
PROMPT='Cinematic medium shot of a woman passionately playing a grand piano in a sunlit concert hall. Her fingers move across the keys as the camera slowly glides sideways. Warm natural lighting, realistic details, flowing piano music.'
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def command(args,env):return subprocess.check_output(args,env=env,text=True,stderr=subprocess.STDOUT).strip()
def decoded(path,env):
 out={}
 for name,flags in [('rgb',['-map','0:v:0','-pix_fmt','rgb24','-f','rawvideo']),('pcm',['-map','0:a:0','-acodec','pcm_f32le','-f','f32le'])]:
  proc=subprocess.Popen(['ffmpeg','-v','error','-i',str(path),*flags,'-'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  h=hashlib.sha256();n=0
  for b in iter(lambda:proc.stdout.read(1<<20),b''):h.update(b);n+=len(b)
  err=proc.stderr.read();assert proc.wait()==0,err
  out[name]={'sha256':h.hexdigest(),'bytes':n}
 return out
def payload(p):
 b=p.read_bytes();assert b[:8]==b'H3AV\r\n\x1a\n'
 nv,na=struct.unpack_from('<2Q',b,72);assert len(b)==160+nv+na
 return {'video':hashlib.sha256(b[160:160+nv]).hexdigest(),'audio':hashlib.sha256(b[160+nv:]).hexdigest()}
def numeric(d,env):return {'state':payload(d/'final.h3av'),'steps':{str(p.relative_to(d/'steps')):sha(p) for p in sorted((d/'steps').rglob('*.f32'))},'decoded':decoded(d/'video.mp4',env)}
def monitor(proc,env,samples):
 start=time.monotonic();last=-2
 while proc.poll() is None:
  row={'seconds':time.monotonic()-start}
  for file in ['status','io']:
   try:
    for line in Path('/proc/%d/%s'%(proc.pid,file)).read_text().splitlines():
     key,_,value=line.partition(':')
     if key in ('VmRSS','VmHWM','RssAnon','RssFile','VmSwap','read_bytes','rchar'):row[key]=int(value.strip().split()[0])*(1024 if 'kB' in value else 1)
   except OSError:pass
  if row['seconds']-last>=1:
   last=row['seconds']
   try:row['gpu']=command(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,pcie.link.gen.current,pcie.link.width.current','--format=csv,noheader,nounits'],env)
   except subprocess.CalledProcessError:pass
  samples.append(row);time.sleep(.2)
def run(root,source,binary,model,env,ident,w,h,mode,cap=None,baseline=None,expected=None,extra=(),steps=6,frames=90):
 d=root/ident;d.mkdir();(d/'steps').mkdir();(d/'preparation').mkdir()
 e=env.copy();e.update(H3_CUDA_WEIGHT_MODE=mode,H3_TEST_NATIVE_STEP_DIR=str(d/'steps'),H3_TEST_SGLANG_DIR=str(d/'preparation'),H3_SGLANG_CAPTURE_STEPS='none',H3_EXPERIMENT_TRACE='1',H3_EXPERIMENT_TIMING='1')
 e.pop('H3_TEST_CUDA_RESIDENT_BLOCKS',None)
 if cap is not None:e['H3_TEST_CUDA_RESIDENT_BLOCKS']=str(cap)
 cmd=[str(binary),'-d',str(model),'-p',PROMPT,'--width',str(w),'--height',str(h),'--frames',str(frames),'--steps',str(steps),'--seed','42','--profile','--save-av-state',str(d/'final.h3av'),'-o',str(d/'video.mp4'),*extra]
 result={'id':ident,'argv':cmd,'weight_mode':mode,'cap':cap,'passed':False};write(d/'command.json',result)
 begin=time.monotonic();samples=[]
 with (d/'run.log').open('wb') as f:
  proc=subprocess.Popen(cmd,cwd=source,env=e,stdout=f,stderr=subprocess.STDOUT)
  t=threading.Thread(target=monitor,args=(proc,e,samples));t.start()
  try:ret=proc.wait(timeout=1200)
  except subprocess.TimeoutExpired:proc.kill();ret=proc.wait();result['timeout']=True
  t.join()
 result.update(returncode=ret,seconds=time.monotonic()-begin);write(d/'monitor.json',samples)
 log=(d/'run.log').read_text(errors='replace');result['plans']=re.findall(r'h3cli: CUDA (?:BF16|packed) weight planner: ([^\n]+)',log)
 result['phases']=[l for l in log.splitlines() if any(x in l for x in ['weight placement ','CUDA profile ','CUDA transfer events:','BF16 SSD stream','reference generation including','reference host weights:'])]
 result['steps']=[json.loads(l.split('h3_experiment ',1)[1]) for l in log.splitlines() if 'h3_experiment {' in l]
 result['maxima']={k:max((r.get(k,0) for r in samples),default=0) for k in ['VmRSS','VmHWM','RssAnon','RssFile','VmSwap','read_bytes','rchar']}
 if ret and mode=='resident' and 'resident weights need' in log:
  result.update(expected_capacity_rejection=True,passed=True);write(d/'result.json',result);return result
 write(d/'result.json',result);assert ret==0,ident+' failed'
 result['numeric']=numeric(d,e)
 assert result['numeric']['decoded']['rgb']['bytes']==w*h*frames*3
 result['preparation']={str(p.relative_to(d/'preparation')):sha(p) for p in sorted((d/'preparation').rglob('*')) if p.is_file()}
 if baseline is not None:assert result['numeric']==baseline,(ident,'baseline mismatch')
 if expected is not None:
  assert result['numeric']==expected['numeric'],(ident,'placement mismatch')
  assert result['preparation']==expected['preparation'],(ident,'preparation mismatch')
 if not extra:
  assert len(result['steps'])==steps and all(x['blocks']==50 for x in result['steps'])
 if cap is not None:
  counts=re.findall(r'resident=(\d+)/(\d+)', '\n'.join(result['plans']));assert counts and 0<int(counts[-1][0])<=cap and int(counts[-1][1])==50
 result['passed']=True;write(d/'result.json',result);print(ident,result['seconds'],result['plans'],flush=True);return result

def main():
    from cuda_reference_regression import source_files, fingerprint
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','binary','model','baseline','out','golden-gate','manifest'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--hardware',choices=('pro','5090'),required=True)
    p.add_argument('--timing',action='store_true')
    a=p.parse_args();env=dict(os.environ);out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    source=a.source.resolve();model=a.model.resolve();binary=a.binary.resolve()
    manifest=json.loads(a.manifest.read_text());gate=json.loads(a.golden_gate.read_text())
    identity=fingerprint(source_files(source))
    assert gate['passed'] and gate['source_sha256']==identity, 'a complete passing golden gate for this source is required'
    assert manifest['prompt']==PROMPT and manifest['steps']==6 and manifest['frames']==90 and manifest['seed']==42
    write(out/'identity.json',{'source_sha256':identity,'binary_sha256':sha(binary),'manifest_sha256':sha(a.manifest),'gate_sha256':sha(a.golden_gate),
        'gpu':command(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,pcie.link.gen.max,pcie.link.width.max','--format=csv'],env),
        'meminfo':Path('/proc/meminfo').read_text(),
        'cgroup':{str(p):p.read_text() for p in [Path('/proc/self/cgroup'),Path('/sys/fs/cgroup/memory.max'),Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')] if p.exists()},
        'storage':command(['df','-T',str(model)],env),
        'model_metadata':{str(p.relative_to(model)):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in sorted(model.rglob('*')) if p.is_file()}})
    result={'passed':False,'runs':[]};write(out/'result.json',result)
    for row in manifest['rows']:
        if not row['id'].startswith(a.hardware+'-'):continue
        w,h=row['width'],row['height'];baseline=numeric(a.baseline/('baseline-%dx%d'%(w,h)),env)
        write(out/(row['id']+'-baseline.json'),baseline);control=None
        cases=[('stream',None),('auto',None),('resident',None)]+[('auto',c) for c in row['partial_caps']]
        if a.timing:cases=[(m,None) for pair in manifest['timing_pairs'] for m in pair]
        for i,(mode,cap) in enumerate(cases):
            ident='%s-%s%s-attempt%d'%(row['id'],mode,'' if cap is None else '-cap%d'%cap,i+1)
            q=run(out,source,binary,model,env,ident,w,h,mode,cap,baseline,control)
            if q.get('expected_capacity_rejection'):
                assert row.get('resident_expected')=='capacity-rejection','unexpected strict capacity rejection'
            else:
                plan=q['plans'][-1]
                if mode=='auto' and cap is None:
                    assert ('effective=partial' in plan) if a.hardware=='5090' else ('effective=resident' in plan or w==1344)
                control=q
            result['runs'].append({'id':ident,'seconds':q['seconds'],'passed':q['passed']});write(out/'result.json',result)
    assert fingerprint(source_files(source))==identity,'source changed during qualification'
    result['passed']=True;write(out/'result.json',result)
if __name__=='__main__':main()
