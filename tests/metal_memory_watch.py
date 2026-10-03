#!/usr/bin/env python3
"""Run a bounded Metal regression with an independent macOS footprint watchdog.

The watcher queries public proc_pid_rusage every 50 ms; no GPU allocations.
It terminates its own child if monitoring fails or the lower test cap is hit.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cap-gb',type=float,default=100)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    a=parser.parse_args();command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command or not 1<=a.cap_gb<=105:parser.error('command and watchdog cap 1..105 GB required')
    def option(name,default):
        value=default
        for i,arg in enumerate(command):
            if arg==name:
                if i+1==len(command):parser.error('missing '+name+' value')
                value=command[i+1]
            elif arg.startswith(name+'='):value=arg[len(name)+1:]
        return value
    steps=int(option('--steps','50'));stop=int(option('--stop-after-step',str(steps)))
    evaluations=min(steps,stop) if stop>=0 else steps
    if steps<1 or stop< -1 or evaluations>6:parser.error('tests execute at most six evaluations')
    width=int(option('--width','0'));height=int(option('--height','0'))
    if max(width,height)>=1344 and evaluations>2:parser.error('large-resolution tests execute at most two evaluations')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    lib=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
    lib.proc_pid_rusage.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
    lib.proc_pid_rusage.restype=ctypes.c_int
    # Prefix shared by public rusage_info_v0..v4; allocate room for full v4.
    class Usage(ctypes.Structure):
        _fields_=[('uuid',ctypes.c_uint8*16)]+[(n,ctypes.c_uint64) for n in
            ('user','system','idle','interrupts','pageins','wired','resident','footprint')]
    buffer=ctypes.create_string_buffer(4096)
    env={**os.environ,'H3_TEST_MAX_EVALUATIONS':'6'}
    binary_before=hashlib.sha256(Path(command[0]).read_bytes()).hexdigest()
    watcher_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    started=time.monotonic();peak=0;reason=None;count=0
    with (out/'run.log').open('w') as log,(out/'memory.jsonl').open('w') as samples:
        child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while child.poll() is None:
                valid=lib.proc_pid_rusage(child.pid,4,buffer)==0
                if not valid:
                    if child.poll() is not None:break
                    reason='watchdog query failed'
                else:
                    r=Usage.from_buffer(buffer);peak=max(peak,r.footprint);count+=1
                    samples.write(json.dumps({'seconds':time.monotonic()-started,'footprint_bytes':r.footprint,
                        'resident_bytes':r.resident,'wired_bytes':r.wired})+'\n')
                    if count%20==0:samples.flush()
                    if r.footprint>=a.cap_gb*1e9:reason='watchdog footprint cap'
                if reason:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=3)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
                time.sleep(.05)
        finally:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait()
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
        return h.hexdigest()
    result={'command':command,'environment':{k:v for k,v in env.items() if k.startswith('H3_')},
        'returncode':child.returncode,'watchdog_reason':reason,'watchdog_cap_bytes':int(a.cap_gb*1e9),
        'peak_footprint_bytes':peak,'samples':count,'wall_seconds':time.monotonic()-started,
        'binary_sha256':binary_before,'binary_unchanged':binary_before==sha(command[0]),
        'watcher_sha256':watcher_sha,'log_sha256':sha(out/'run.log'),'memory_sha256':sha(out/'memory.jsonl')}
    (out/'record.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if reason:raise SystemExit(3)
    if child.returncode:raise SystemExit(1)

if __name__=='__main__':main()
