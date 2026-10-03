#!/usr/bin/env python3
"""Fixed twelve-video experiment; ordinary regression budgets remain unchanged."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import ctypes
import fcntl
import json
import math
from pathlib import Path
import re
import signal
import struct
import hashlib
import subprocess
import threading
import time
from cuda_reference_regression import fingerprint, sha, source_files, write, PROMPT
from cuda_sglang import NVML

IDS = ['D0','R2','R3','C4','C6','A1','A3','S75','S80','A1S75','A3S75','A3S80']
ARGS = [[],['--reuse','2'],['--reuse','3'],['--core-reuse','4'],['--core-reuse','6'],
        ['--adaptive-cache','conservative'],['--adaptive-cache','aggressive'],
        ['--cuda-attention','subblock','--subblock-sparsity','0.75'],
        ['--cuda-attention','subblock','--subblock-sparsity','0.80']]
ARGS += [ARGS[5]+ARGS[7], ARGS[6]+ARGS[7], ARGS[6]+ARGS[8]]


def manifest(path):
    m=json.loads(path.read_text())
    assert [m[k] for k in ('schema','recipe','width','height','frames','steps','fps','seed','timeout_seconds')]==[1,1,640,480,90,50,24,42,7200]
    assert m['prompt']==PROMPT and m['projection']=='bf16' and m['decoder']=='full'
    assert m['variants']==[dict(id=i,args=a) for i,a in zip(IDS,ARGS)], 'changed fixed comparison matrix'
    return m


def trace(log):
    steps=[json.loads(s.split('h3_experiment ',1)[1]) for s in log.splitlines() if 'h3_experiment {' in s]
    assert [s['step'] for s in steps]==list(range(50)), 'missing/duplicate scheduler transition'
    assert all(s['total']==50 and all(not isinstance(v,float) or math.isfinite(v) for v in s.values()) for s in steps)
    assert all(s['blocks'] in (0,1,50) and 0<=s['sparse_calls']<=s['blocks'] for s in steps)
    pattern=r'adaptive step=(\d+).*?decision=(\S+) reason=(\S+) score=(\S+) video_score=(\S+) audio_score=(\S+) blocks=(\d+) streak=(\d+) bytes=(\d+)'
    cache=[dict(step=int(a),decision=b,reason=c,score=float(d),video_score=float(e),audio_score=float(f),blocks=int(g),streak=int(h),bytes=int(i))
           for a,b,c,d,e,f,g,h,i in re.findall(pattern,log)]
    return steps,cache


def dispatch(variant,steps,cache):
    args=variant['args'];adaptive='--adaptive-cache' in args;sparse='subblock' in args
    assert bool(cache)==adaptive
    if adaptive:
        assert [x['step'] for x in cache]==list(range(50))
        assert cache[-1]['decision']=='refresh' and cache[-1]['reason']=='final'
        for s,c in zip(steps,cache):
            assert s['blocks']==c['blocks'] and c['bytes']<=512*1024**2
            if c['decision']=='hit':assert s['blocks']==1 and s['sparse_calls']==0 and s['dense_calls']==1
            assert all(math.isfinite(c[k]) for k in ('score','video_score','audio_score'))
        if sparse:assert cache[10]['decision']=='refresh' and cache[10]['reason']=='attention-phase'
    if sparse:
        assert all(s['sparse_calls']==0 for s in steps[:10])
        assert all(s['dense_calls']>=1 for s in steps if s['evaluated'])
    else:assert all(s['sparse_calls']==0 for s in steps)
    if variant['id']=='D0':assert all(s['evaluated']==1 and s['blocks']==50 for s in steps)
    return dict(transitions=50,forwards=sum(s['evaluated'] for s in steps),blocks=sum(s['blocks'] for s in steps),
        hits=sum(c['decision']=='hit' for c in cache),refreshes=sum(c['decision']=='refresh' for c in cache),
        sparse_calls=sum(s['sparse_calls'] for s in steps),dense_calls=sum(s['dense_calls'] for s in steps),
        selected=sum(s['selected'] for s in steps),possible=sum(s['possible'] for s in steps))


def media(path,env):
    p=json.loads(subprocess.check_output([env.get('H3_FFPROBE','ffprobe'),'-v','error','-count_frames','-show_streams','-of','json',str(path)],env=env))
    video=[s for s in p['streams'] if s['codec_type']=='video'];audio=[s for s in p['streams'] if s['codec_type']=='audio']
    assert len(video)==len(audio)==1
    v,a=video[0],audio[0]
    assert (v['width'],v['height'],int(v['nb_read_frames']),v['avg_frame_rate'])==(640,480,90,'24/1')
    assert a['channels']==2 and int(a['sample_rate'])==32000
    # AAC packets can include up to one 1024-sample padding frame.
    assert abs(float(v['duration'])-90/24)<1/24
    assert abs(float(a['duration'])-90/24)<=1024/32000+1/32000
    result=subprocess.run([env.get('H3_FFMPEG','ffmpeg'),'-v','error','-i',str(path),'-f','null','-'],env=env,capture_output=True,text=True)
    assert result.returncode==0 and not result.stderr.strip(), result.stderr
    return p


def model_metadata(model):
    files={str(p.relative_to(model)):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(model.rglob('*')) if p.is_file()}
    assert files and any(k.endswith('.safetensors') for k in files)
    return files


def av_state(path):
    data=path.read_bytes()
    assert data[:8]==b'H3AV\r\n\x1a\n' and struct.unpack_from('<4I',data,8)==(3,160,0x01020304,1)
    width,height,frames,vt,lh,lw,at,vc,ac,channels=struct.unpack_from('<10I',data,24)
    nv,na=24*vt*lh*lw,64*at
    assert (width,height,frames,vc,ac,channels)==(640,480,90,24,32,2)
    assert struct.unpack_from('<2Q',data,72)==(nv*4,na*4) and len(data)==160+4*(nv+na)
    assert hashlib.sha256(data[:128]+data[160:]).digest()==data[128:160]
    assert all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',data[160:]))
    assert Path(str(path)+'.presentation').is_file()


def prepare(model,metadata):
    start=time.monotonic();total=0
    for name,info in metadata.items():
        p=model/name;s=p.stat();assert (s.st_size,s.st_mtime_ns)==(info['bytes'],info['mtime_ns'])
        if p.suffix=='.safetensors' and Path(name).parts[0]=='FL2VA':
            # Bounded header readahead; never enqueue the whole model or evict
            # other workloads. Full weight paging remains in measured wall time.
            length=min(s.st_size,1024*1024)
            fd=os.open(p,os.O_RDONLY)
            try:os.posix_fadvise(fd,0,length,os.POSIX_FADV_WILLNEED)
            finally:os.close(fd)
            total+=length
    return dict(seconds=time.monotonic()-start,advised_bytes=total,policy='stat metadata then first 1 MiB of each FL2VA weight file via POSIX_FADV_WILLNEED; no hashing or eviction; full weight paging is measured; OS residency not guaranteed')


def runtime(env):
    device=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    cudnn=ctypes.CDLL(env['H3_SGLANG_CUDNN_LIBRARY']);cudnn.cudnnGetVersion.restype=ctypes.c_size_t
    assert cudnn.cudnnGetVersion()==92000
    toolkit=subprocess.check_output([str(Path(env['CUDA_PATH'])/'bin/nvcc'),'--version'],text=True)
    assert 'release 13.0' in toolkit
    settings=subprocess.check_output(['nvidia-smi','--query-gpu=power.limit,persistence_mode,compute_mode','--format=csv,noheader'],text=True).strip()
    return dict(device=device,cudnn=int(cudnn.cudnnGetVersion()),toolkit=toolkit,device_settings=settings,cpu_affinity=sorted(os.sched_getaffinity(0)),
        environment={k:v for k,v in env.items() if k.startswith(('CUDA','CUDNN','SGLANG','H3_')) or k in ('LD_LIBRARY_PATH','LIBRARY_PATH','CPATH','PATH','OMP_NUM_THREADS')})


def render(source,out,model,m,variant,env):
    out.mkdir(parents=True,exist_ok=False)
    cmd=[str(source/'bin/h3cli'),'-d',str(model),'-p',m['prompt'],'--seed','42','--width','640','--height','480','--frames','90','--steps','50',
         '--save-av-state',str(out/'final.h3av'),'-o',str(out/'video.mp4'),*variant['args']]
    record=dict(passed=False,variant=variant,command=cmd,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write(out/'result.json',record)
    monitor=NVML();samples=[];errors=[];stop=threading.Event()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not active, 'GPU has active compute processes: '+active
    start=time.monotonic()
    with (out/'render.log').open('x') as log,(out/'memory.jsonl').open('x') as memory:
        child=subprocess.Popen(cmd,cwd=source,env=env|{'H3_TEST_MAX_EVALUATIONS':'50'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        def sample():
            try:
                while not stop.is_set():
                    s=monitor.sample(child.pid)|dict(elapsed_seconds=time.monotonic()-start);samples.append(s)
                    memory.write(json.dumps(s)+'\n');memory.flush();stop.wait(1)
            except BaseException as e:errors.append(str(e))
        worker=threading.Thread(target=sample);worker.start()
        try:record['returncode']=child.wait(timeout=m['timeout_seconds'])
        except BaseException as e:
            record['error']=str(e);os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        finally:
            stop.set();worker.join();record.update(wall_seconds=time.monotonic()-start,telemetry_errors=errors)
            write(out/'result.json',record)
    assert record.get('returncode')==0 and samples and not errors, 'render or telemetry failed'
    log=(out/'render.log').read_text();steps,cache=trace(log);counts=dispatch(variant,steps,cache)
    probe=media(out/'video.mp4',env);av_state(out/'final.h3av')
    write(out/'ffprobe.json',probe);write(out/'steps.json',steps);write(out/'cache.json',cache)
    gaps=[b['elapsed_seconds']-a['elapsed_seconds'] for a,b in zip(samples,samples[1:])]
    record.update(passed=True,counts=counts,stage_seconds={k:float(v) for k,v in re.findall(r'h3cli: phase duration ([^\r\n:]+): ([0-9.]+) s',log)},
        peak_vram_bytes=max(s['gpu_used_bytes'] for s in samples),peak_host_rss_bytes=max(s.get('process_rss_bytes',0) for s in samples),
        maximum_sample_gap_seconds=max(gaps,default=0),missed_samples=sum(max(0,int(g)-1) for g in gaps),
        artifacts={p.name:sha(p) for p in (out/'video.mp4',out/'final.h3av',out/'final.h3av.presentation')})
    write(out/'result.json',record);return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    source=a.source.resolve();model=a.model.resolve();out=a.out.resolve();m=manifest(a.manifest)
    out.mkdir(parents=True,exist_ok=a.resume)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(os.environ);env['H3_EXPERIMENT_TRACE']='1';env['H3_EXPERIMENT_TIMING']='1';env['H3_CUDA_WEIGHT_MODE']='stream'
        assert not any(k.startswith('H3_TEST_') and k!='H3_TEST_MAX_EVALUATIONS' for k in env), 'diagnostic tensor capture is forbidden'
        assert not any(k.startswith('H3_PROFILE') for k in env), 'heavy profiling is forbidden in timed generation'
        env.pop('H3_TEST_MAX_EVALUATIONS',None)
        metadata=model_metadata(model)
        identity=dict(schema=1,source_sha256=fingerprint(source_files(source)),binary_sha256=sha(source/'bin/h3cli'),manifest_sha256=sha(a.manifest),
                      model_metadata=metadata,runtime=runtime(env))
        frozen=out/'identity.json'
        if frozen.exists():assert json.loads(frozen.read_text())==identity,'source/build/model/runtime changed'
        else:write(frozen,identity)
        ledger=[]
        for variant in m['variants']:
            case=out/variant['id'];case.mkdir(exist_ok=True)
            attempts=sorted(case.glob('attempt-*'))
            existing=[json.loads((d/'result.json').read_text()) for d in attempts]
            successes=[(d,r) for d,r in zip(attempts,existing) if r['passed']]
            assert len(successes)<=1
            if successes:
                d,r=successes[0]
                for name,digest in r['artifacts'].items():assert sha(d/name)==digest
            else:
                assert not any(r.get('returncode')==0 for r in existing), 'completed render requires inspection; do not silently repeat it'
                prep=prepare(model,metadata);d=case/f'attempt-{len(attempts)+1:03d}'
                try:r=render(source,d,model,m,variant,env)
                except BaseException as e:
                    ledger.append(dict(id=variant['id'],passed=False,artifact_directory=str(d.relative_to(out)),error=str(e)))
                    write(out/'ledger.json',dict(complete=False,planned=IDS,cases=ledger));raise
                r['preparation']=prep;write(d/'result.json',r)
            assert fingerprint(source_files(source))==identity['source_sha256'] and sha(source/'bin/h3cli')==identity['binary_sha256']
            ledger.append(dict(id=variant['id'],passed=True,artifact_directory=str(d.relative_to(out)),attempt_count=len(list(case.glob('attempt-*')))))
            write(out/'ledger.json',dict(complete=len(ledger)==12,planned=IDS,cases=ledger))
            print(json.dumps(dict(id=variant['id'],wall_seconds=r['wall_seconds'],counts=r['counts'])),flush=True)
        assert len(ledger)==12 and [x['id'] for x in ledger]==IDS

if __name__=='__main__':main()
