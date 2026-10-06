#!/usr/bin/env python3
"""Opt-in model/GPU check of concise logs and quiet/verbose render identity."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-path', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith('H3_PROFILE') and key != 'H3_VERBOSE'}
    env.update(H3_OFFLINE='1', H3_TEST_MAX_EVALUATIONS='6')
    common = [str(root / 'bin/h3cli'), '--models-path', str(args.models_path.resolve()),
              '-p', 'A quiet Alpine valley, a smooth forward camera glide. Gentle wind.',
              '--ref-image', str(root / 'tests/fixtures/cuda-reference/first.png'),
              '--ref-image-size', 'match', '--width', '256', '--height', '256',
              '--frames', '90', '--steps', '2', '--seed', '42']
    results = []
    for stage in ('source', 'continuation'):
        for mode in ('quiet', 'verbose'):
            name = f'{stage}-{mode}'
            command = common + ['--save-av-state', str(out / f'{name}.h3av'),
                                '-o', str(out / f'{name}.mp4')]
            if mode == 'verbose':
                command += ['--verbose']
            if stage == 'continuation':
                command += ['--continue-from', str(out / 'source-quiet.h3av'),
                            '--continue-context', '39']
            log = out / f'{name}.log'
            with log.open('x') as stream:
                subprocess.run(command, cwd=root, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True, timeout=600)
            text = log.read_text()
            assert b'\r' in log.read_bytes(), name
            assert re.search(r'denoise\s+2/2\s+\([0-9.]+ s', text), name
            assert re.search(r'DiT initialization\s+1/1\s+\([0-9.]+ s\)', text), name
            assert re.search(r'h3cli: total wall time: [0-9.]+ s', text), name
            assert f'h3cli: wrote {out / (name + ".mp4")}' in text, name
            if mode == 'quiet':
                for detail in ('phase start', 'phase duration', 'monotonic',
                               'allocation counters', 'weight planner',
                               'generation including lazy weights', 'continuation source:'):
                    assert detail not in text, (name, detail)
            else:
                assert 'phase start denoise:' in text, name
                assert 'phase duration DiT initialization:' in text, name
                if stage == 'continuation':
                    assert text.count('h3cli: continuation source:') == 1, name
            results.append(dict(name=name, command=command, log_lines=len(text.splitlines()),
                                hashes={ext: sha(out / f'{name}.{ext}') for ext in ('h3av', 'mp4')}))
            print(f'PASS {name}: progress and diagnostics', flush=True)
        assert results[-2]['hashes'] == results[-1]['hashes'], stage
        print(f'PASS {stage}: byte-identical quiet/verbose MP4 and AV state', flush=True)
    (out / 'result.json').write_text(json.dumps(dict(passed=True, runs=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
