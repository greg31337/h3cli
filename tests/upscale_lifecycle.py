#!/usr/bin/env python3
"""Serial bounded model-backed refinement tests; no full comparison videos."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--binary', type=Path, default=Path('bin/upscale_lifecycle'))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env['H3_TEST_MAX_EVALUATIONS'] = '6'
    result = {'passed': False, 'runs': []}

    def run(name, args):
        begin = time.monotonic()
        with (a.out/(name+'.log')).open('w') as log:
            proc = subprocess.run([str(a.binary.resolve()), *map(str, args)], env=env,
                                  stdout=log, stderr=subprocess.STDOUT, timeout=720)
        result['runs'].append({'name': name, 'seconds': time.monotonic()-begin,
                               'returncode': proc.returncode})
        assert proc.returncode == 0, name

    def sample(name):
        return a.out/(name+'.h3sample')

    try:
        run('prepare', ['prepare', a.model, a.source, a.weights, sample('zero'), a.out/'K0.h3av'])
        trace = a.out/'trajectory.f32'
        run('full', ['run', a.model, sample('zero'), a.out/'full', trace, -1, -1, 0, 0])
        expected = hashlib.sha256((a.out/'full.h3av').read_bytes()).hexdigest()
        for boundary in (0, 1, 3):
            run(f'stop-{boundary}', ['run', a.model, sample('zero'), a.out/f'stop-{boundary}',
                                     trace, boundary, -1, 1, 0])
            run(f'resume-{boundary}', ['run', a.model, sample(f'stop-{boundary}'),
                                       a.out/f'resume-{boundary}', trace, -1, -1, 1, boundary])
            assert hashlib.sha256((a.out/f'resume-{boundary}.h3av').read_bytes()).hexdigest() == expected
        for boundary in (0, 1, 3):
            run(f'cancel-{boundary}', ['run', a.model, sample('zero'), a.out/f'cancel-{boundary}',
                                       trace, -1, boundary, 1, 0])
            run(f'recover-{boundary}', ['run', a.model, sample(f'cancel-{boundary}'),
                                        a.out/f'recover-{boundary}', trace, -1, -1, 1, boundary])
            assert hashlib.sha256((a.out/f'recover-{boundary}.h3av').read_bytes()).hexdigest() == expected
        result.update(passed=True, final_av_sha256=expected,
                      trajectory_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
    finally:
        (a.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
