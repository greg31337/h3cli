#!/usr/bin/env python3
"""Run one bounded local preview experiment, charging all attempts to one ledger."""
import argparse
import fcntl
import hashlib
import json
import os
import platform
from pathlib import Path
import signal
import subprocess
import time

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--ledger', type=Path, default=Path('outputs/preview-vae/metal/budget.json'))
    p.add_argument('--budget', type=float, default=1200)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if not 0 < a.timeout <= 300 or not 0 < a.budget <= 1800:
        p.error('timeout must be <=300s and budget <=1800s')
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command or Path(a.name).name != a.name:
        p.error('a command and simple case name are required')
    a.ledger.parent.mkdir(parents=True, exist_ok=True)
    with a.ledger.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(a.ledger.read_text()) if a.ledger.exists() else {'seconds': 0, 'runs': []}
        if data.setdefault('budget_seconds', a.budget) != a.budget:
            p.error('an existing ledger has a fixed budget; do not change it between cases')
        if any(r['name'] == a.name for r in data['runs']):
            p.error('case already recorded; use a distinct name for another charged attempt')
        if data['seconds'] + a.timeout > a.budget:
            p.error(f'cannot reserve {a.timeout}s; {data["seconds"]:.2f}s already used')
        files = {}
        for arg in command:
            candidate = Path(arg)
            if candidate.is_file() and candidate.stat().st_size <= 64 << 20:
                files[arg] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        record = dict(name=a.name, command=command, timeout=a.timeout,
                      status='running', started=time.time(), cwd=str(Path.cwd()),
                      platform=platform.platform(), input_sha256=files,
                      environment={k: v for k, v in os.environ.items() if k.startswith('H3_')})
        # Reserve persistently so interruption of the harness cannot erase GPU work.
        data['seconds'] += a.timeout
        data['runs'].append(record)
        a.ledger.write_text(json.dumps(data, indent=2) + '\n')
        start = time.monotonic()
        code = -1
        with (a.ledger.parent / (a.name + '.log')).open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=a.timeout)
                record['status'] = 'passed' if code == 0 else 'failed'
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                record['status'] = 'timeout'
            finally:
                elapsed = time.monotonic() - start
                record.update(seconds=elapsed, returncode=code)
                record['output_sha256'] = {arg: hashlib.sha256(Path(arg).read_bytes()).hexdigest()
                    for arg in command if Path(arg).is_file() and Path(arg).stat().st_size <= 64 << 20}
                data['seconds'] += elapsed - a.timeout
                a.ledger.write_text(json.dumps(data, indent=2) + '\n')
        print(json.dumps(record), flush=True)
        raise SystemExit(0 if code == 0 else 1)

if __name__ == '__main__':
    main()
