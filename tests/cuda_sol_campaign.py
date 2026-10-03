#!/usr/bin/env python3
"""One durable clock and serial ownership for the bounded CUDA SOL campaign."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse,fcntl,hashlib,json,os,re,shutil,signal,subprocess,sys,time
from pathlib import Path
from cuda_multi_reference import Monitor,sha,digest,save,terminate,probe,cg_memory
ROOT=Path('outputs/cuda-sol')
ACCEPT=Path('tests/cuda_sol_acceptance.json')
MODEL=Path(os.environ.get('H3_MODEL_DIR','models/MiniMax-H3')).expanduser().resolve()
REFERENCE_ROOT=Path(os.environ.get('H3_TEST_REFERENCE_ROOT','.')).expanduser().resolve()
QUANT_CACHE=Path(os.environ.get('H3_TEST_QUANT_CACHE','outputs/cuda-sol/packed')).expanduser().resolve()

def gpu_uuid():
    """Bind each local campaign to device zero without publishing its serial ID."""
    value=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid',
                                   '--format=csv,noheader'],text=True).strip()
    assert re.fullmatch(r'GPU-[0-9a-fA-F-]+',value),'Cannot identify qualification GPU'
    return value

def clock():
    p=ROOT/'clock.json';now=time.time()
    if not p.exists():
        save(p,dict(start_epoch=now,gpu_deadline=now+450*60,final_deadline=now+480*60,start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(now))))
    c=json.loads(p.read_text());assert now>=c['start_epoch']-5,'Clock moved backwards'
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if 'start_monotonic' not in c:
        c.update(boot_id=boot,start_monotonic=time.monotonic()-(now-c['start_epoch']));save(p,c)
    assert c['boot_id']==boot,'Node rebooted; preserve the original deadline and audit before recovery'
    return c

def remaining(final=False):
    c=clock();return min(c['final_deadline' if final else 'gpu_deadline']-time.time(),(480 if final else 450)*60-(time.monotonic()-c['start_monotonic']))
def records():return [json.loads(p.read_text()) for p in sorted(ROOT.glob('runs/*/record.json'))]
def spent(bucket):return sum(r['wall_seconds'] for r in records() if r['bucket']==bucket)
def controlled(extra=None):
    env={k:v for k,v in os.environ.items() if not k.startswith('H3_')};env.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    env.update(extra or {});return env

def source_identity():
    files=[]
    for pat in ['src/**/*.c','src/**/*.cu','src/**/*.cuh','src/**/*.h','Makefile','scripts/cuda_arch.sh','tests/cuda_sol*','tests/test_cuda_sol*']:
        files.extend(p for p in Path('.').glob(pat) if p.is_file() and p.suffix not in ('.o','.d'))
    return {str(p):sha(p) for p in sorted(set(files))}

def freeze():
    if (ROOT/'identity.json').exists():return verify()
    old=json.loads(Path('tests/archive/multi-reference/multi-reference.json').read_text())
    assets={k:v for k,v in old['assets'].items() if k in [f'I{i:02d}' for i in range(1,10)]+['V3a','V3b','V3c']}
    target=ROOT/'fixtures';target.mkdir(exist_ok=True)
    for a in assets.values():
        src=REFERENCE_ROOT/a['path'];assert sha(src)==a['sha256'];dst=target/src.name;shutil.copy2(src,dst);a['path']=str(dst)
    model=MODEL;prior=REFERENCE_ROOT/'outputs/multi-reference-cuda/model-identity.json'
    model_files=json.loads(prior.read_text())
    for rel,identity in model_files.items():
        s=(model/rel).stat();assert s.st_size==identity['size'] and s.st_mtime_ns==identity['mtime_ns'],f'Model changed: {rel}'
    save(ROOT/'model-identity.json',model_files)
    env={'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,memory.free','--format=csv,noheader'],text=True).strip(),'nvcc':subprocess.check_output(['/usr/local/cuda/bin/nvcc','--version'],text=True),'cgroup':cg_memory(),'model_identity_basis':'Prior full SHA-256 audit with matching sizes and nanosecond mtimes; no model rewrite'}
    save(ROOT/'environment.json',env)
    identity=dict(binary=sha('bin/h3cli'),native=sha('bin/cuda_sol_native'),dense_native=sha('bin/attention_native'),source=source_identity(),acceptance=sha(ACCEPT),manifest=sha('tests/cuda_sol_manifest.json'),assets=assets,model=digest(model_files),model_path=str(MODEL),gpu_uuid=gpu_uuid())
    identity['identity']=digest(identity);save(ROOT/'identity.json',identity);return identity

def verify():
    i=json.loads((ROOT/'identity.json').read_text());assert sha('bin/h3cli')==i['binary'] and sha('bin/cuda_sol_native')==i['native'] and sha('bin/attention_native')==i['dense_native'],'Binary changed: frozen result cannot be reused'
    assert all(Path(p).exists() and sha(p)==h for p,h in i['source'].items()),'Source/test identity changed after freeze'
    assert sha(ACCEPT)==i['acceptance'] and sha('tests/cuda_sol_manifest.json')==i['manifest']
    assert i.get('model_path')==str(MODEL),'Model location changed; start a fresh campaign'
    assert i.get('gpu_uuid')==gpu_uuid(),'Qualification GPU changed; start a fresh campaign'
    for a in i['assets'].values():assert sha(a['path'])==a['sha256']
    return i

def execute(ident,bucket,argv,cap,env=None,monitor=False,allow_failure=False):
    assert bucket in json.loads(ACCEPT.read_text())['buckets_minutes']
    cap=min(cap,remaining(bucket=='G'));assert cap>0,'Campaign deadline reached'
    limit=json.loads(ACCEPT.read_text())['buckets_minutes'][bucket]*60
    assert spent(bucket)<limit,'Bucket limit reached';cap=min(cap,limit-spent(bucket))
    d=ROOT/'runs'/ident;d.mkdir(parents=True,exist_ok=False)
    row=dict(id=ident,bucket=bucket,argv=list(map(str,argv)),environment=env or {},timeout=cap,start_epoch=time.time(),cwd=str(Path.cwd()),identity=json.loads((ROOT/'identity.json').read_text())['identity'] if (ROOT/'identity.json').exists() else None)
    save(d/'request.json',row);mon=None;reason=None
    if monitor:
        settings=dict(gpu_uuid=verify()['gpu_uuid'],gpu_headroom_bytes=4<<30,host_headroom_bytes=10<<30,process_limit_bytes=110000000000)
        assert shutil.disk_usage('.').free>15<<30,'Disk headroom below 15 GiB'
        mon=Monitor(d/'telemetry.jsonl.gz',settings);mon.start();assert mon.ready.wait(10) and not mon.error and not mon.abort
    start=time.monotonic();log_offset=0;phase_events=[];last_disk=0
    with (d/'stdout.log').open('wb') as out,(d/'stderr.log').open('wb') as err:
        process=subprocess.Popen(row['argv'],stdout=out,stderr=err,env=controlled(env),start_new_session=True)
        save(d/'owner.json',dict(pid=process.pid,start_epoch=time.time(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
        if mon:mon.pid=process.pid
        while process.poll() is None:
            elapsed=time.monotonic()-start
            if elapsed-last_disk>1:
                last_disk=elapsed
                if shutil.disk_usage('.').free<10<<30:reason='Disk headroom below 10 GiB'
                with (d/'stderr.log').open('rb') as phases:
                    phases.seek(log_offset);chunk=phases.read();log_offset=phases.tell()
                for match in re.finditer(rb'(denoise|video VAE decoder|audio VAE decoder)[^\r\n]*(starting|last step[^\r\n]*)',chunk):
                    phase_events.append(dict(seconds=elapsed,line=match[0].decode(errors='replace')))
            if time.monotonic()-start>=cap or remaining(bucket=='G')<=0:reason='deadline'
            if mon and (mon.error or mon.abort):reason=mon.error or mon.abort
            if reason:terminate(process)
            else:time.sleep(.05)
    row.update(phase_events=phase_events,wall_seconds=time.monotonic()-start,returncode=process.returncode,termination_reason=reason,finished_epoch=time.time())
    if mon:
        released=False
        for _ in range(100):
            if not mon.nv.nvmlDeviceGetComputeRunningProcesses(mon.handle) and mon.nv.nvmlDeviceGetMemoryInfo(mon.handle).used<=mon.idle+(256<<20):released=True;break
            time.sleep(.1)
        row['gpu_released']=released;row['telemetry']=mon.finish()
    log=(d/'stderr.log').read_text(errors='replace')
    row['step_seconds']=[float(x) for x in re.findall(r'denoise\s+\d+\s*/\s*\d+[^\r\n]*?([0-9]+\.[0-9]+)\s*s',log)]
    row['sol_counters']=[dict((k,float(v) if '.' in v else int(v)) for k,v in re.findall(r'(\w+)=([0-9.]+)',s)) for s in re.findall(r'h3(?:cli)?: SOL counters ([^\n]+)',log)]
    row['status']='pass' if process.returncode==0 and not reason and (not mon or row['gpu_released']) else 'failed'
    save(d/'record.json',row);print(ident,row['status'],round(row['wall_seconds'],2),flush=True)
    if row['status']!='pass' and not allow_failure:raise RuntimeError(f'{ident} failed; retained logs at {d}')
    return row

def command(c,mode,d,minimum,identity):
    assert c['steps']==2,'All generated cases require exactly two steps'
    args=['./bin/h3cli','-d',str(MODEL),'-p',c['prompt'],'--width',str(c['width']),'--height',str(c['height']),'--frames',str(c['frames']),'--steps','2','--seed',str(c['seed']),'--reuse','1','--core-reuse','1','--cuda-device','0','--cuda-weight-mode',c.get('weight_mode','resident'),'--cuda-attention',mode,'--cuda-denoise-quant',c.get('quant','off'),'--ref-image-size',c['image_size']]
    if mode=='sol':args+=['--sol-min-exact',str(minimum)]
    for key in c.get('references',[]):
        a=identity['assets'][key];args+=['--ref-image' if a['kind']=='image' else '--ref-silent-video',a['path']]
    if c.get('save_state'):args+=['--save-av-state',str(d/'output.h3av')]
    if c.get('continue_from'):args+=['--continue-from',c['continue_from'],'--continue-context','39']
    return args+c.get('extra',[])+['-o',str(d/'output.mp4')]

def media(d,c):
    path=d/'output.mp4';result={'valid':False}
    if path.exists():
        data=probe(path,True);streams=data['streams'];v=next(x for x in streams if x['codec_type']=='video')
        frames=c['frames']-(39 if c.get('continue_from') else 0)
        with (d/'decode.log').open('w') as f:p=subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),'-f','null','-'],stdout=f,stderr=f,timeout=120)
        result=dict(valid=p.returncode==0 and int(v['nb_read_frames'])==frames and (v['width'],v['height'])==(c['width'],c['height']) and any(s['codec_type']=='audio' for s in streams),probe=data,sha256=sha(path),expected_frames=frames)
    save(d/'media.json',result);return result

def render(c,mode,minimum,bucket,cap,extra_env=None):
    identity=verify();ident=c['id']+'-'+mode;d=ROOT/'runs'/ident
    env={};diagnostics=c.get('diagnostics',False)
    if diagnostics:
        for part in ['steps','qkv']:(ROOT/'diagnostics'/ident/part).mkdir(parents=True,exist_ok=False)
        env=dict(H3_TEST_MAX_EVALUATIONS='6',H3_TEST_NATIVE_STEP_DIR=str(ROOT/'diagnostics'/ident/'steps'))
        if c.get('capture'):env.update(H3_TEST_ATTENTION_CAPTURE_DIR=str(ROOT/'diagnostics'/ident/'qkv'),H3_TEST_ATTENTION_CAPTURE_MAX_SEQUENCE='131072',H3_TEST_ATTENTION_CAPTURE_BLOCKS='1,25',H3_TEST_ATTENTION_CAPTURE_STEPS='1')
    env.update(extra_env or {})
    row=execute(ident,bucket,command(c,mode,d,minimum,identity),cap,env,True,True)
    row['case']=c;row['mode']=mode;row['minimum']=minimum if mode=='sol' else None
    row['media']=media(d,c);row['status']='pass' if row['status']=='pass' and row['media']['valid'] else 'failed';save(d/'record.json',row);return row

def matrix():
    verify();candidate=json.loads((ROOT/'candidate.json').read_text());assert candidate['screen_pass'],'Calibration did not qualify a candidate'
    contract=json.loads(ACCEPT.read_text());pairs=json.loads(Path('tests/cuda_sol_manifest.json').read_text())['render_pairs']
    for c in pairs:
        c['prompt']=contract['held_out']['prompt'];c['save_state']=c['id']=='R1';c['diagnostics']=c['id']=='R1'
        # Pair admission uses the entire frozen cap, independent of expected SOL speed.
        if remaining()<c['pair_minutes']*60+60:
            save(ROOT/(c['id']+'-blocked.json'),dict(reason='Insufficient remaining paired budget'));continue
        done={r['id'] for r in records()};start=time.monotonic()
        for mode in ['default','sol']:
            if c['id']+'-'+mode in done:continue
            cap=min(c['pair_minutes']*30,c['pair_minutes']*60-(time.monotonic()-start))
            row=render(c,mode,candidate['minimum'],'D',cap,dict(H3_CPU_SAMPLER='1'))
            if row['status']!='pass':break

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','freeze','exec','matrix','status']);p.add_argument('--id');p.add_argument('--bucket',default='A');p.add_argument('--minutes',type=float,default=10);a,cmd=p.parse_known_args();ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.action=='start':print(json.dumps(clock()))
        elif a.action=='freeze':freeze()
        elif a.action=='matrix':matrix()
        elif a.action=='status':print(json.dumps(dict(clock=clock(),remaining=remaining(),spent={b:spent(b) for b in json.loads(ACCEPT.read_text())['buckets_minutes']})))
        else:
            if cmd and cmd[0]=='--':cmd=cmd[1:]
            execute(a.id,a.bucket,cmd,a.minutes*60)
if __name__=='__main__':main()
