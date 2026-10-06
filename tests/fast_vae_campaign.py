#!/usr/bin/env python3
"""Serial CUDA/Metal qualification; one persistent deadline includes failures.

Metal defaults to four hours and reserves 25 minutes for metrics/reporting.
Jobs can supply a measured estimated_seconds to avoid starting a test that
cannot finish. RSS is sampled on macOS; unified memory is not called VRAM.
"""
import argparse,json,os,signal,subprocess,time,pathlib,hashlib,fcntl,platform,ctypes

class MacUsage(ctypes.Structure):
    _fields_=[('uuid',ctypes.c_ubyte*16)]+[(name,ctypes.c_uint64) for name in
        ['user_time','system_time','pkg_idle_wkups','interrupt_wkups','pageins',
         'wired_size','resident_size','phys_footprint','proc_start_abstime','proc_exit_abstime']]

def mac_usage(pid):
    usage=MacUsage();lib=ctypes.CDLL('/usr/lib/libproc.dylib')
    lib.proc_pid_rusage.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
    if lib.proc_pid_rusage(pid,0,ctypes.byref(usage))!=0:return None
    return usage

def save(path,value):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def main():
    metal=platform.system()=='Darwin'
    p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('--wait-for');p.add_argument('--root',default='outputs/fast-vae/metal' if metal else 'outputs/fast-vae/cuda')
    p.add_argument('--budget-minutes',type=float,default=240 if metal else 480)
    p.add_argument('--reserve-minutes',type=float,default=25 if metal else 35)
    a=p.parse_args()
    if not 0<a.reserve_minutes<a.budget_minutes:p.error('reserve must be positive and smaller than budget')
    root=pathlib.Path(a.root);root.mkdir(parents=True,exist_ok=True);ledgerpath=root/'ledger.json'
    lock=(root/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    if not ledgerpath.exists():
        now=time.time();save(ledgerpath,dict(started_unix=now,deadline_unix=now+a.budget_minutes*60,budget_seconds=a.budget_minutes*60,reporting_reserve_seconds=a.reserve_minutes*60,platform=platform.platform(),entries=[]))
    ledger=json.loads(ledgerpath.read_text())
    if ledger.get('finished_unix'):raise SystemExit('Campaign is closed; use a new root for a new campaign')
    # Existing deadlines never move when the harness restarts.
    ledger.setdefault('reporting_reserve_seconds',a.reserve_minutes*60)
    if a.wait_for:
        fcntl.flock(lock,fcntl.LOCK_UN)
        while not pathlib.Path(a.wait_for).exists():
            if time.time()>ledger['deadline_unix']-ledger['reporting_reserve_seconds']:return
            time.sleep(5)
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=json.loads(ledgerpath.read_text())
        if ledger.get('finished_unix'):raise SystemExit('Campaign is closed')
    jobs=json.loads(pathlib.Path(a.manifest).read_text())
    for job in jobs:
        d=root/job['id'];d.mkdir(parents=True,exist_ok=True)
        if (d/'record.json').exists():continue
        now=time.time();budget=min(job.get('timeout',1800),ledger['deadline_unix']-now-ledger['reporting_reserve_seconds'])
        missing=[]
        for dependency in job.get('requires',[]):
            p=root/dependency/'record.json'
            if not p.is_file() or json.loads(p.read_text()).get('status')!='pass':missing.append(dependency)
        if missing:
            save(d/'record.json',dict(job,status='deferred',started_unix=now,wall_seconds=0,reason='dependency did not pass',dependencies=missing))
            ledger['entries'].append(dict(id=job['id'],started_unix=now,wall_seconds=0,status='deferred'));save(ledgerpath,ledger);continue
        if budget<=0 or job.get('estimated_seconds',0)>budget:
            record=dict(job,status='deferred',started_unix=now,wall_seconds=0,reason='measured estimate exceeds remaining budget/reporting reserve')
            save(d/'record.json',record);ledger['entries'].append(dict(id=job['id'],started_unix=now,wall_seconds=0,status='deferred'));save(ledgerpath,ledger);continue
        env=os.environ.copy();env.update(job.get('env',{}));env['H3_TEST_MAX_EVALUATIONS']='6';env['H3_VERBOSE']='1'
        argv=job['argv'];record=dict(job,started_unix=now,cwd=os.getcwd(),status='running',measurement=('main-process RSS and physical footprint sampled at 1 Hz via proc_pid_rusage; unified GPU allocations reported by decoder, not discrete VRAM' if metal else 'whole-device VRAM and process RSS sampled at 1 Hz')+'; peaks between samples may be missed')
        binary=pathlib.Path(argv[0]);record['binary_sha256']=hashlib.file_digest(binary.open('rb'),'sha256').hexdigest() if binary.is_file() else None
        save(d/'request.json',record);peak=None if metal else 0;rss=0;footprint=0;wired=0;samples=[]
        with (d/'stdout.log').open('w') as stdout,(d/'stderr.log').open('w') as stderr:
            proc=subprocess.Popen(argv,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            while proc.poll() is None:
                used=None;current_rss=None;physical=None
                try:
                    if metal:
                        usage=mac_usage(proc.pid)
                        if usage:
                            current_rss=usage.resident_size;physical=usage.phys_footprint
                            footprint=max(footprint,physical);wired=max(wired,usage.wired_size)
                        else:current_rss=int(subprocess.check_output(['ps','-o','rss=','-p',str(proc.pid)],text=True,timeout=3).strip())*1024
                        rss=max(rss,current_rss)
                    else:
                        used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5).splitlines()[0])*1048576;peak=max(peak,used)
                        status=pathlib.Path(f'/proc/{proc.pid}/status').read_text()
                        for line in status.splitlines():
                            if line.startswith('VmRSS:'):current_rss=int(line.split()[1])*1024;rss=max(rss,current_rss)
                except (OSError,ValueError,subprocess.SubprocessError):pass
                samples.append([time.time()-now,used,current_rss,physical])
                save(d/'progress.json',dict(pid=proc.pid,elapsed=time.time()-now,peak_vram_bytes=peak,peak_rss_bytes=rss,peak_physical_footprint_bytes=footprint if metal else None))
                if time.time()-now>budget:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
                    record['timed_out']=True;break
                time.sleep(1)
            record.update(returncode=proc.wait(),wall_seconds=time.time()-now,peak_vram_bytes=peak,peak_rss_bytes=rss,status='pass' if proc.returncode==0 else 'failed')
            if metal:record.update(peak_physical_footprint_bytes=footprint,peak_wired_bytes=wired)
        if '--steps' in argv:
            import re
            record['requested_steps']=int(argv[argv.index('--steps')+1]);log=(d/'stderr.log').read_text(errors='replace')
            counts=[int(x) for x in re.findall(r'denoise\s+(\d+)/6\b',log)]
            record['completed_steps']=max(counts,default=0)
            if record['requested_steps']!=6 or record['completed_steps']!=6:record['status']='failed-provenance'
        save(d/'memory.json',samples);save(d/'record.json',record)
        ledger['entries'].append(dict(id=job['id'],started_unix=now,wall_seconds=record['wall_seconds'],status=record['status']));save(ledgerpath,ledger)
if __name__=='__main__':main()
