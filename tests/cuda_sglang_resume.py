#!/usr/bin/env python3
"""Real reference checkpoint/resume and conditioning-cache round trips.

Run serially on the qualification GPU. The saved full schedule remains
authoritative on resume. Compare clean AV payloads to the independently
completed, unmodified primary baseline; never regenerate that baseline.
"""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import struct
import subprocess
import time

from cuda_sglang import PROMPT, validate


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def clean_state(path):
    data = path.read_bytes()
    if len(data) < 160 or data[:8] != b'H3AV\r\n\x1a\n':
        raise ValueError('invalid AV header')
    if hashlib.sha256(data[:128] + data[160:]).digest() != data[128:160]:
        raise ValueError('invalid generated AV checksum')
    video, audio = struct.unpack_from('<2Q', data, 72)
    if 160 + video + audio != len(data):
        raise ValueError('unexpected AV payload length')
    return dict(geometry=struct.unpack_from('<3I', data, 24),
                video_bytes=video, audio_bytes=audio,
                video_sha256=hashlib.sha256(data[160:160 + video]).hexdigest(),
                audio_sha256=hashlib.sha256(data[160 + video:]).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'baseline-state', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--case', choices=('C0', 'C1', 'C2'), required=True)
    p.add_argument('--model', default=os.environ.get('H3_MODEL_DIR', 'models/MiniMax-H3'))
    p.add_argument('--build-env', help='Optional shell environment file')
    for name in ('reference-cublas', 'reference-cudnn', 'reference-jpeg', 'reference-ffmpeg'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    contract_path = Path(__file__).with_name('cuda_sglang_contract.json')
    contract = json.loads(contract_path.read_text())
    case = next(c for c in contract['primary'] if c['id'] == a.case)
    gold = clean_state(a.baseline_state)
    if gold['geometry'] != (case['width'], case['height'], case['frames']):
        raise ValueError('baseline geometry differs from the frozen manifest')
    a.out.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('H3_', 'SGLANG_'))}
    env.update(H3_SGLANG_CUBLAS_LIBRARY=a.reference_cublas,
               H3_SGLANG_CUDNN_LIBRARY=a.reference_cudnn,
               H3_SGLANG_JPEG_LIBRARY=a.reference_jpeg, H3_FFMPEG=a.reference_ffmpeg,
               H3_TEST_MAX_EVALUATIONS=str(case['evaluations']))
    env['LD_LIBRARY_PATH']=str(Path(a.reference_cudnn).resolve(strict=True).parent)+':'+env.get('LD_LIBRARY_PATH','')
    binary = str(a.source.resolve() / 'bin/h3cli')
    checkpoint = a.out.resolve() / 'paused.h3sample'
    conditioning = a.out.resolve() / 'conditioning.h3cond'
    base = [binary, '-d', a.model, '-p', PROMPT, '--seed', '42',
            '--width', '640', '--height', '480', '--frames', str(case['frames']),
            '--steps', str(case['evaluations']), ]
    records = []

    def run(name, command, complete=False):
        root = a.out / name
        root.mkdir()
        command += ['-o', str(root.resolve() / 'video.mp4')]
        if complete:
            command += ['--save-av-state', str(root.resolve() / 'final.h3av')]
        spec = dict(argv=command, environment={k: v for k, v in env.items() if k.startswith('H3_')},
                    binary_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest())
        write(root / 'command.json', spec)
        start = time.monotonic()
        with (root / 'render.log').open('x') as log:
            r = subprocess.run(['bash', '-c', 'set -e; ' + (('source ' + shlex.quote(a.build_env) + '; ') if a.build_env else '') +
                                'exec ' + shlex.join(command)], cwd=a.source, env=env,
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        record = dict(name=name, wall_seconds=time.monotonic() - start, exit_code=r.returncode, passed=False)
        records.append(record)
        if r.returncode:
            write(root / 'result.json', record)
            raise RuntimeError(name + ' failed')
        if complete:
            record['validation'] = validate(root, 'native', case['frames'], case['evaluations'])
            record['state'] = clean_state(root / 'final.h3av')
            record['passed'] = record['state'] == gold
        else:
            record['passed'] = checkpoint.is_file() and checkpoint.stat().st_size > 1024
        write(root / 'result.json', record)
        return root

    error = None
    try:
        flags = ['--save-conditioning', str(conditioning), '--conditioning-schedule'] if a.case == 'C0' else []
        run('pause', base + flags + ['--stop-after-step', str(case['evaluations'] // 2), '--save-sampler-state', str(checkpoint)])
        run('resume', [binary, '-d', a.model, '--resume-sampler-state', str(checkpoint)], True)
        if a.case == 'C0':
            root = run('conditioning-hit', base + ['--load-conditioning', str(conditioning)], True)
            log = (root / 'render.log').read_text()
            records[-1]['cache_log'] = [line for line in log.splitlines() if 'conditioning' in line.lower()]
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        error = str(exc)
    result = dict(comparison='native reference interruption and cache replay', case=a.case,
                  baseline_state=str(a.baseline_state), baseline_payload=gold, records=records,
                  error=error, passed=not error and all(r['passed'] for r in records),
                  contract_sha256=hashlib.sha256(contract_path.read_bytes()).hexdigest())
    write(a.out / 'result.json', result)
    print(json.dumps(result), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
