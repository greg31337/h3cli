#!/usr/bin/env python3
"""Persistent, serialized experiment budget for the CUDA denoiser work."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import shutil
import time


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def interrupted(signum, frame):
    raise InterruptedError(f'received signal {signum}')

def file_arguments(command):
    result={}
    for value in command:
        try:
            path=Path(value)
            if path.is_file() and path.stat().st_size<200*1024*1024:result[value]=sha(path)
        except OSError:
            pass # Long prompt strings are arguments, not filenames.
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='outputs/quant-5090')
    p.add_argument('--name', required=True)
    p.add_argument('--kind', choices=['component', 'render', 'prepare'], default='component')
    p.add_argument('--timeout', type=float)
    p.add_argument('--extend', type=float, default=0, help='explicit additional total seconds')
    p.add_argument('--unlimited', action='store_true', help='explicitly remove the cumulative limit; keep case timeouts')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    cap = dict(component=120, render=300, prepare=600)[a.kind]
    timeout = a.timeout if a.timeout is not None else cap
    if not 0 < timeout <= cap or a.extend < 0 or (a.unlimited and a.extend):
        p.error('invalid timeout/extension')
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command or '/' in a.name or a.name in ('.', '..'):
        p.error('provide a command and simple case name')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with (out/'budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = out/'budget.json'
        ledger = json.loads(path.read_text()) if path.exists() else dict(limit=3600, seconds=0, runs=[])
        if ledger.get('pending'):
            pending = ledger.pop('pending')
            pending.update(status='interrupted-run-reservation-charged', seconds=pending['timeout'])
            ledger['seconds'] += pending['seconds']; ledger['runs'].append(pending)
        if a.extend:
            if ledger['limit'] is None:
                p.error('the cumulative allowance is already unlimited')
            ledger['limit'] += a.extend
            ledger.setdefault('extensions', []).append(dict(seconds=a.extend, time=time.time()))
        if a.unlimited and ledger['limit'] is not None:
            ledger.setdefault('extensions', []).append(dict(type='unlimited', previous_limit=ledger['limit'],
                seconds_spent=ledger['seconds'], time=time.time(), case=a.name))
            ledger['limit'] = None
        atomic(path, ledger)
        if ledger['limit'] is not None and ledger['seconds'] + timeout > ledger['limit']:
            p.error(f'budget exhausted: {ledger["seconds"]:.3f}/{ledger["limit"]}s; cannot reserve {timeout}s')
        if (out/(a.name+'.json')).exists():
            p.error('case already recorded; use a new name for an explicit repeat')
        record = dict(name=a.name, command=command, cwd=os.getcwd(), timeout=timeout,
                      kind=a.kind, started=time.time(), environment={k:v for k,v in os.environ.items()
                      if k.startswith(('H3_', 'CUDA_', 'CUDNN_'))})
        binary = Path(shutil.which(command[0]) or command[0])
        if binary.is_file():
            record['binary_sha256'] = sha(binary)
        record['file_arguments_sha256']=file_arguments(command[1:])
        sources=list(Path('.').glob('src/**/*'))+list(Path('tests').glob('quant*'))+[Path('Makefile')]
        record['workspace_sources_sha256_at_launch']={str(x):sha(x) for x in sorted(sources)
            if x.is_file() and (x.suffix in ('.c','.h','.m','.cu','.cuh','.metal','.py') or x.name=='Makefile')}
        identity=out/'environment.json'
        if identity.exists():record['environment_manifest_sha256']=sha(identity)
        ledger['pending'] = record; atomic(path, ledger)
        signal.signal(signal.SIGTERM,interrupted)
        start = time.monotonic(); code = -1; failure = None; process = None
        try:
            with (out/(a.name+'.log')).open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                code = process.wait(timeout=timeout)
                if code: failure = f'exit {code}'
        except BaseException as e:
            failure = f'{type(e).__name__}: {e}'
            if process is not None:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait()
        record.update(seconds=time.monotonic()-start, returncode=code, failure=failure,
                      status='failed' if failure else 'passed')
        ledger.pop('pending'); ledger['seconds'] += record['seconds']; ledger['runs'].append(record)
        atomic(out/(a.name+'.json'), record); atomic(path, ledger)
        allowance='unlimited' if ledger['limit'] is None else f'{ledger["limit"]}s'
        print(f'{a.name}: {record["status"]}, {record["seconds"]:.3f}s; total {ledger["seconds"]:.3f}s / {allowance}')
        return 1 if failure else 0


if __name__ == '__main__':
    raise SystemExit(main())
