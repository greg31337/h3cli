#!/usr/bin/env python3
"""Summarize complete reference profiles without adding overlapping GPU times."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    contract_path = Path(__file__).with_name('cuda_sglang_contract.json')
    contract = json.loads(contract_path.read_text())
    cases = []
    for spec in contract['primary']:
        case = spec['id']
        root = a.root / f'profile-full-{case}-v12'
        result = json.loads((root / 'result.json').read_text())
        log = (root / 'render.log').read_text()
        steps = [json.loads(value) for value in re.findall(r'h3_cuda_reference_step (\{[^\r\n]+\})', log)]
        complete = (result['status'] == 'media_valid' and
                    [s['step'] for s in steps] == list(range(1, spec['evaluations'] + 1)) and
                    all(s['total'] == spec['evaluations'] and s['evaluated'] for s in steps))
        stable = bool(steps) and all(len({s[key] for s in steps}) == 1 for key in ('live_device_bytes', 'pinned_bytes'))
        teacher = a.root / f'teacher-full-{case}-v10/native/video.mp4'
        same = digest(root / 'video.mp4') == digest(teacher)
        categories = {}
        for key in ('wall_seconds', 'gemm_seconds', 'attention_seconds', 'elementwise_seconds',
                    'h2d_seconds', 'd2h_seconds', 'upload_wait_seconds'):
            values = [s[key] for s in steps]
            if values:
                categories[key] = dict(total=sum(values), median=statistics.median(values),
                                       minimum=min(values), maximum=max(values))
        memory = {key: dict(first=steps[0][key], last=steps[-1][key], minimum=min(s[key] for s in steps),
                            maximum=max(s[key] for s in steps))
                  for key in ('live_device_bytes', 'peak_tensor_bytes', 'pinned_bytes', 'resident_bytes', 'swap_used_bytes', 'tensor_allocations')} if steps else {}
        cases.append(dict(case=case, complete_schedule=complete, live_allocation_plateau=stable,
                          media_unchanged_from_teacher_run=same, steps=steps, categories=categories,
                          memory=memory, generation_seconds=result['validation'].get('generation_seconds'),
                          phases=result['validation'].get('stage_seconds'),
                          passed=complete and stable and same,
                          log_sha256=digest(root / 'render.log'), command_sha256=digest(root / 'command.json')))
    report = dict(kind='complete instrumented reference profiles', cases=cases,
                  contract_sha256=digest(contract_path), passed=all(c['passed'] for c in cases),
                  notes=['Instrumented runs do not qualify latency.',
                         'H2D and compute overlap: category totals must not be added as serial wall time.',
                         'Cumulative tensor allocation counts include released views; live device/pinned bytes measure retention.',
                         'Native tensor counters exclude driver/context/library overhead; matched NVML totals remain separate.'])
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({c['case']: {k: v for k, v in c.items() if k not in ('steps', 'phases')} for c in cases}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
