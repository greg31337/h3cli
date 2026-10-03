#!/usr/bin/env python3
"""Bounded, calibration-only mixed-recipe study using a separate plan-9001 build.

Apply attention_sensitivity.patch to an isolated source copy and rebuild it.
This is diagnostic evidence, not a public policy or a qualified mixed recipe.
The delivered binary and frozen pure-mode limits remain unchanged.
"""
import argparse
import json
import re
from pathlib import Path
from attention_run import run, sha
from attention_workflows import compare_av
from attention_cache import prepare, clear_generated


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', required=True)
    p.add_argument('--calibration', required=True, type=Path)
    p.add_argument('--cache', required=True)
    p.add_argument('--cache-source', required=True)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    prepare(a.cache, a.cache_source)
    ledger = json.loads((a.calibration / 'ledger.json').read_text())
    references = [r for r in ledger if re.fullmatch(r'calibration-(fl|ref)-(base|turbo)-nvfp4-100[12]-r0-default', r['name'])]
    if len(references) != 8:
        raise RuntimeError('requires the complete fixed NVFP4 calibration subset')
    if sha(a.binary) in {r['binary_sha256'] for r in references}:
        raise RuntimeError('the sensitivity binary must be a separate diagnostic build')
    rows = []
    for index,reference in enumerate(references):
        source = reference['argv'][reference['argv'].index('--save-av-state') + 1]
        steps = int(reference['argv'][reference['argv'].index('--steps') + 1])
        for schedule in ('endpoints', 'endpoints-and-edge-blocks'):
            name = reference['name'].replace('calibration-', 'sensitivity-').replace('-default', '-' + schedule)
            command = reference['argv'].copy(); command[0] = a.binary
            command[command.index('--cuda-attention') + 1] = 'sage3'
            command[command.index('--cuda-denoise-quant-cache') + 1] = a.cache
            output = a.output / (name + '.h3av')
            command[command.index('--save-av-state') + 1] = str(output)
            command[command.index('-o') + 1] = str(a.output / (name + '.mp4'))
            environment = {k:v for k,v in reference['environment'].items()
                           if not k.startswith(('H3_TEST_', 'H3_DEBUG_DIT'))}
            environment.update(H3_TEST_ATTENTION_SENSITIVITY=schedule, H3_TEST_ATTENTION_LAST_STEP=str(steps - 1))
            record = a.output / (name + '.json')
            if record.exists():
                job = json.loads(record.read_text())
                if job['argv'] != command or job['binary_sha256'] != sha(a.binary) or job['returncode']:
                    raise RuntimeError('changed or failed sensitivity record')
            else:
                job = run(name, command, a.output, 3000, environment)
            if job['returncode']:
                raise RuntimeError('sensitivity render failed: ' + job['log'])
            profiles = job['profile']['sage_profiles']
            dispatch = next(x for x in reversed(profiles) if 'plan=9001 ' in x)
            counts = [int(re.search(key + r'=(\d+)', dispatch)[1]) for key in ('sage2', 'sage3')]
            expected2 = 100 + ((steps - 2) * 8 if schedule == 'endpoints-and-edge-blocks' else 0)
            if counts != [expected2, steps * 50 - expected2]:
                raise RuntimeError('diagnostic schedule did not execute its declared dispatches')
            metrics = compare_av(Path(source), output)
            rows.append({'name': name, 'schedule': schedule, 'diagnostic_plan': 9001,
                         'metrics': metrics, 'dispatches': counts, 'record': job,
                         'hard_ceiling_pass': all(m['finite'] and m['relative_l2'] <= .1 and m['max_abs'] <= 1 for m in metrics.values()),
                         'qualified_public_policy': False})
            (a.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
        # Keep this family's identical weight artifacts across its two seeds.
        if index+1==len(references) or references[index+1]['name'].split('-nvfp4-')[0]!=reference['name'].split('-nvfp4-')[0]:
            clear_generated(a.cache)
    (a.output / 'complete.json').write_text(json.dumps({'cases': len(rows), 'qualified_public_policy': False,
        'selection': 'calibration only; no held-out threshold changes', 'hard_ceiling_passes': sum(r['hard_ceiling_pass'] for r in rows)}) + '\n')


if __name__ == '__main__':
    main()
