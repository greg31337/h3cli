#!/usr/bin/env python3
"""Run one recorded attention qualification job with a real per-job timeout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import re

def profile(log):
    text=Path(log).read_text(errors='replace')
    result={
        'cuda_profiles':re.findall(r'h3(?:cli)?: CUDA profile ([^\n]+)',text),
        'category_times':re.findall(r'h3(?:cli)?: CUDA category seconds: ([^\n]+)',text),
        'execution_profiles':re.findall(r'h3(?:cli)?: execution profile ([^\n]+)',text),
        'sage_profiles':re.findall(r'h3(?:cli)?: Sage profile ([^\n]+)',text),
        'quant_profiles':re.findall(r'h3(?:cli)?: quant profile ([^\n]+)',text),
        'total_wall_seconds':[float(x) for x in re.findall(r'h3(?:cli)?: total wall time: ([0-9.]+) s',text)],
        'sequence_lengths':sorted(set(map(int,re.findall(r'(?:S=|sequence=)([0-9]+)',text)))),
        'main_dit_sequence_lengths':sorted(set(map(int,re.findall(r'attention shape phase=main-DiT [^\n]* S=(\d+)',text)))),
        'denoise_wall_seconds':[float(x) for x in re.findall(r'CUDA profile H3 DiT / (?:GPU Euler|Euler|RES) denoise: wall ([0-9.]+)s',text)],
        'phase_durations':[{ 'phase':phase,'seconds':float(seconds)} for phase,seconds in re.findall(r'h3(?:cli)?: phase duration ([^\n]+): ([0-9.]+) s',text)],
        'phase_starts':[{ 'phase':phase,'monotonic':float(seconds)} for phase,seconds in re.findall(r'h3(?:cli)?: phase start ([^\n]+): monotonic ([0-9.]+)',text)],
    }
    if not result['main_dit_sequence_lengths']:
        # The default route logs refiners and main blocks under H3 DiT. The
        # joint sequence adds video/audio/reference rows to the refiner input.
        dit_shapes=list(map(int,re.findall(r'attention shape phase=H3 DiT [^\n]* S=(\d+)',text)))
        if dit_shapes:result['main_dit_sequence_lengths']=[max(dit_shapes)]
    starts=result['phase_starts'];durations=result['phase_durations']
    enqueue=next((x for x in starts if x['phase']=='denoise enqueue'),None)
    complete=next((x for x in starts if x['phase']=='denoise'),None)
    tail=next((x['seconds'] for x in durations if x['phase']=='denoise'),0.)
    if enqueue and complete:result['denoising_seconds']=complete['monotonic']-enqueue['monotonic']+tail
    elif complete and tail:result['denoising_seconds']=tail
    # CUDA counters are cumulative within a component. Subtract its load
    # snapshot to isolate denoising transfers and cache behavior; retain peak
    # memory as an absolute high-water mark rather than subtracting peaks.
    sections=[]
    snapshots=list(re.finditer(r'h3(?:cli)?: CUDA profile ([^\n]+)',text))
    for index,snapshot in enumerate(snapshots):
        line=snapshot[1]
        body=text[snapshot.end():snapshots[index+1].start() if index+1<len(snapshots) else len(text)]
        header=re.fullmatch(r'(.+): wall ([0-9.]+)s peak ([0-9.]+) GiB, GEMM (\d+) attention (\d+) conv (\d+), H2D ([0-9.]+) GiB D2H ([0-9.]+) GiB stream ([0-9.]+) GiB copy ([0-9.]+)s wait ([0-9.]+)s',line)
        if not header:continue
        section={'label':header[1],'cumulative_wall_seconds':float(header[2]),'tracked_peak_device_gib':float(header[3])}
        section.update(zip(('gemm_calls','attention_calls','conv_calls'),map(int,header.group(4,5,6))))
        section.update(zip(('h2d_gib','d2h_gib','stream_gib','copy_seconds','wait_seconds'),map(float,header.group(7,8,9,10,11))))
        category=re.search(r'h3(?:cli)?: CUDA category seconds: GEMM ([0-9.]+) attention ([0-9.]+) conv ([0-9.]+);',body)
        if category:section.update(zip(('gemm_seconds','attention_seconds','conv_seconds'),map(float,category.groups())))
        transfer=re.search(r'h3(?:cli)?: CUDA transfer events: H2D ([0-9.]+)s \([0-9.]+ GiB/s\), D2H ([0-9.]+)s; uncovered upload wait ([0-9.]+)s; normalization/elementwise ([0-9.]+)s; source read/upload wall ([0-9.]+)s',body)
        if transfer:section.update(zip(('h2d_seconds','d2h_seconds','uncovered_upload_wait_seconds','normalization_seconds','source_read_upload_seconds'),map(float,transfer.groups())))
        cache=re.search(r'h3(?:cli)?: CUDA weight cache: ([0-9.]+) GiB, hits (\d+) / ([0-9.]+) GiB, misses (\d+), releases (\d+)',body)
        if cache:section.update(weight_cache_gib=float(cache[1]),weight_cache_hits=int(cache[2]),weight_cache_hit_gib=float(cache[3]),weight_cache_misses=int(cache[4]),weight_cache_releases=int(cache[5]))
        sections.append(section)
    result['cuda_sections']=sections
    quant_sections=[]
    for line in result['quant_profiles']:
        match=re.fullmatch(r'(.+): mode=(\S+) recipe=(\d+) native_calls=(\d+) cache_hits=(\d+) prepared=(\d+) conversion=([0-9.]+)s \(included in GEMM\) stream=(compressed|resident)',line)
        if match:
            quant_sections.append({'label':match[1],'mode':match[2],'recipe':int(match[3]),
                'native_calls':int(match[4]),'artifact_cache_hits':int(match[5]),
                'prepared_weights':int(match[6]),'conversion_seconds':float(match[7]),
                'stream':match[8]})
    result['quant_sections']=quant_sections
    loaded=next((x for x in sections if x['label']=='H3 DiT / load'),None)
    denoised=next((x for x in sections if re.fullmatch(r'H3 DiT / (?:GPU Euler|Euler|RES) denoise',x['label'])),None)
    if loaded and denoised:
        gauges=('label','cumulative_wall_seconds','tracked_peak_device_gib','weight_cache_gib')
        result['denoising_cuda_counters']={k:v-loaded[k] for k,v in denoised.items() if k not in gauges and k in loaded}
        result['denoising_cuda_counters'].update({k:denoised[k] for k in ('tracked_peak_device_gib','weight_cache_gib') if k in denoised})
    return result

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def run(name,command,output,timeout,environment=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    log=output/(name+'.log');record=output/(name+'.json')
    if record.exists():raise RuntimeError(f'refusing to overwrite recorded job {record}')
    env={**os.environ,**(environment or {})}
    start=time.monotonic();started=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    try:binary_hash=sha(command[0])
    except (OSError,IsADirectoryError):binary_hash=None
    with log.open('w') as f:
        p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        timed_out=False;next_sample=0;samples=[];telemetry_error=None
        observed={};carry='';last_observation=start;max_observation_gap=0
        watcher=log.open('rb')
        while True:
            pid,status,usage=os.wait4(p.pid,os.WNOHANG)
            now=time.monotonic();max_observation_gap=max(max_observation_gap,now-last_observation);last_observation=now
            chunk=carry+watcher.read().decode('utf-8',errors='replace')
            for match in re.finditer(r'\r(denoise enqueue|denoise|audio VAE|tiny video VAE)\s+[^\r\n]*',chunk):
                phase=match[1];entry=observed.setdefault(phase,{'first_seconds':now-start})
                entry['last_seconds']=now-start
            # Only retain an incomplete trailing progress line; complete lines
            # must not be observed again when the child writes nothing new.
            carry=chunk[chunk.rfind('\r'):] if chunk.endswith('\r') else ''
            if pid:break
            if time.monotonic()>=next_sample and sys.platform!='darwin':
                next_sample=time.monotonic()+1
                try:
                    t=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,check=True)
                    values=[x.strip().split(',') for x in t.stdout.splitlines() if x.strip()]
                    mine=sum(int(mem.strip()) for child,mem in values if int(child.strip())==p.pid and mem.strip().isdigit())
                    others=[int(child.strip()) for child,mem in values if int(child.strip())!=p.pid]
                    samples.append({'seconds':time.monotonic()-start,'process_mib':mine,'other_cuda_pids':others})
                except (OSError,ValueError,subprocess.SubprocessError) as e:telemetry_error=str(e)
            if time.monotonic()-start>timeout and not timed_out:
                timed_out=True;os.killpg(p.pid,signal.SIGTERM)
            if time.monotonic()-start>timeout+10:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            time.sleep(.2)
        watcher.close()
        p.returncode=os.waitstatus_to_exitcode(status)
    result=dict(record_version=2,name=name,argv=command,started_utc=started,started_monotonic=start,binary_sha256=binary_hash,
        environment={k:v for k,v in env.items() if k.startswith(('H3_','CUDA_','CUBLAS_'))},
        returncode=p.returncode,timed_out=timed_out,timeout_seconds=timeout,
        wall_seconds=time.monotonic()-start,user_seconds=usage.ru_utime,system_seconds=usage.ru_stime,
        child_peak_rss_bytes=usage.ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        rss_scope='individual child (wait4)',log=str(log))
    result['progress_observed']=observed
    result['progress_max_observation_gap_seconds']=max_observation_gap
    begin=observed.get('denoise enqueue',observed.get('denoise'))
    end=observed.get('audio VAE')
    if begin and end:result['observed_denoising_seconds']=end['first_seconds']-begin['first_seconds']
    result['profile']=profile(log)
    denoise=next((x for x in result['profile']['phase_starts'] if x['phase'].startswith('denoise')),None)
    if denoise:result['startup_seconds']=denoise['monotonic']-start
    result['gpu_telemetry']={'interval_seconds':1,'scope':'child CUDA process',
        'peak_process_bytes':max((x['process_mib'] for x in samples),default=0)*1048576,
        'concurrent_cuda_pids':sorted({p for x in samples for p in x['other_cuda_pids']}),
        'error':telemetry_error,'samples':len(samples)}
    if samples:(output/(name+'.nvml.json')).write_text(json.dumps(samples)+'\n')
    if p.returncode==0 and '--save-av-state' in command:
        av=command[command.index('--save-av-state')+1]
        if Path(av).exists():result['av_sha256']=sha(av)
    record.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True);p.add_argument('--output',required=True)
    p.add_argument('--timeout',type=float,default=2400)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command or a.timeout<=0:p.error('positive timeout and command required')
    sys.exit(0 if run(a.name,command,a.output,a.timeout)['returncode']==0 else 1)
