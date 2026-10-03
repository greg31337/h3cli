#!/usr/bin/env python3
"""Replay every captured oracle state through native reference denoising.

The complete six/50-point evaluation schedule is retained. Every evaluation
receives an independently captured oracle input, so this cannot be mistaken
for the separate free-running trajectory qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from cuda_sglang_compare import metric, native, oracle, pack_audio, pack_video


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def unpack(value, modality):
    if modality == 'video':
        if value.ndim != 2 or value.shape[1] != 96 or value.shape[0] % 300:
            raise ValueError('unexpected primary video geometry')
        return value.reshape(-1, 15, 20, 24, 2, 2).transpose(3, 0, 1, 4, 2, 5).copy()
    if value.ndim != 2 or value.shape[1] != 32 or value.shape[0] % 2:
        raise ValueError('unexpected primary audio geometry')
    return value.reshape(2, -1, 32).transpose(2, 0, 1).copy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'oracle', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('reference-cublas', 'reference-cudnn', 'reference-jpeg', 'reference-ffmpeg'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    contract_path = Path(__file__).with_name('cuda_sglang_contract.json')
    contract = json.loads(contract_path.read_text())
    spec = json.loads((a.oracle / 'command.json').read_text())
    case = next(c for c in contract['primary'] if c['id'] == spec['case'])
    if any(case[k] != spec[k] for k in ('frames', 'evaluations')):
        raise ValueError('oracle differs from the frozen primary manifest')
    if json.loads((a.oracle / 'result.json').read_text())['status'] != 'media_valid':
        raise ValueError('oracle render is incomplete')
    a.out.mkdir(parents=True, exist_ok=False)
    inputs = a.out / 'inputs'
    inputs.mkdir()
    capture = a.oracle / 'capture'
    records = []
    for step in range(case['evaluations']):
        for modality in ('video', 'audio'):
            # Primary cases have no immutable condition prefix.
            prefix = json.loads((capture / f'positive.{modality}_target_start.json').read_text())
            if prefix != 0:
                raise ValueError('teacher campaign requires an unconditioned primary case')
            key = f'step-{step - 1:03d}.{modality}' if step else f'initial_{modality}_rows'
            value = oracle(capture, key)
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('invalid oracle state')
            path = inputs / f'step-{step:03d}-{modality}-latent.f32'
            raw = unpack(value, modality).astype('<f4', copy=False).tobytes()
            path.write_bytes(raw)
            records.append(dict(step=step, modality=modality, source=key,
                                bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    write(a.out / 'inputs.json', records)
    root = a.out / 'native'
    command = [sys.executable, str(a.source / 'tests/cuda_sglang.py'),
               '--engine', 'native', '--reference', '--case', case['id'],
               '--source', str(a.source), '--out', str(root), '--capture',
               '--env', 'H3_TEST_SGLANG_DIR=',
               '--env', f'H3_TEST_NATIVE_TEACHER_DIR={inputs.resolve()}']
    for name in ('reference_cublas', 'reference_cudnn', 'reference_jpeg', 'reference_ffmpeg'):
        command += ['--' + name.replace('_', '-'), getattr(a, name)]
    write(a.out / 'command.json', dict(argv=command))
    with (a.out / 'render.log').open('x') as log:
        run = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    results = []
    for step in range(case['evaluations']):
        for index, (modality, pack) in enumerate((('video', pack_video), ('audio', pack_audio))):
            for kind in ('input', 'velocity', 'latent'):
                label = f'step-{step}.{modality}-{kind}'
                key = (f'step-{step:03d}.velocity.{index}' if kind == 'velocity' else
                       f'step-{step:03d}.{modality}' if kind == 'latent' else
                       f'step-{step - 1:03d}.{modality}' if step else f'initial_{modality}_rows')
                try:
                    row = dict(name=label, kind=kind, **metric(
                        pack(native(root / 'steps', f'step-{step + 1:03d}-{modality}-{kind}.f32')),
                        oracle(capture, key)))
                except (OSError, ValueError, KeyError) as exc:
                    row = dict(name=label, kind=kind, error=str(exc))
                gate = contract['gates']['full_velocity' if kind == 'velocity' else 'trajectory']
                row['passed'] = bool(row.get('finite') and row.get('compatible') and
                    (row['exact'] if kind == 'input' else row['relative_l2'] <= gate['relative_l2_max'] and row['cosine'] >= gate['cosine_min']))
                results.append(row)
    report = dict(comparison='teacher_forced_every_evaluation', case=case['id'],
                  evaluations=case['evaluations'], selected_early_middle_final=[0, case['evaluations'] // 2, case['evaluations'] - 1],
                  results=results, exit_code=run.returncode,
                  passed=run.returncode == 0 and all(r['passed'] for r in results),
                  first_failing_boundary=next((r['name'] for r in results if not r['passed']), None),
                  contract_sha256=hashlib.sha256(contract_path.read_bytes()).hexdigest())
    write(a.out / 'result.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
