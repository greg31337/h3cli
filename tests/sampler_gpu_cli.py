#!/usr/bin/env python3
"""GPU CLI checkpoint at a skipped step, without per-step callback synchronization."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from test_sampler_file import entries

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle', type=Path, required=True, help='Matching GPU reuse-3 oracle base')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=ROOT/'bin/h3cli')
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    oracle = args.oracle.resolve(); binary = args.binary.resolve()
    env = {k: v for k, v in os.environ.items() if not k.startswith('H3_')}
    env['H3_GPU_SAMPLER'] = '1'
    checkpoint = out/'step5.h3sample'
    command = [str(binary), '-d', str(ROOT/'models/MiniMax-H3'), '-p',
        'The woman walks slowly through a sunlit garden and waves. Steady camera and quiet outdoor ambience.',
        '--width', '256', '--height', '256', '--frames', '90', '--seed', '72', '--steps', '20',
        '--reuse', '3', '--stop-after-step', '5', '--save-sampler-state', str(checkpoint)]
    commands = [command]
    with (out/'pause.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=log, check=True)
    parts = {p[0]: p[-1] for p in entries(checkpoint.read_bytes())}
    expected = {p[0]: p[-1] for p in entries(Path(str(oracle)+'.step5.h3sample').read_bytes())}
    for kind in [11, 12, 13, 16, 25, 26, 27, 28]:
        assert parts[kind] == expected[kind], f'GPU CLI section {kind} changed without callbacks'
    command = [str(binary), '-d', str(ROOT/'models/MiniMax-H3'), '--resume-sampler-state', str(checkpoint),
               '-o', str(out/'resumed.mp4'), '--save-av-state', str(out/'resumed.h3av')]
    commands.append(command)
    with (out/'resume.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=log, check=True)
    hashes = {}
    for extension in ['h3av', 'mp4']:
        a = (out/('resumed.'+extension)).read_bytes(); b = Path(str(oracle)+'.'+extension).read_bytes()
        assert a == b, f'GPU CLI final {extension} changed without callbacks'
        hashes[extension] = hashlib.sha256(a).hexdigest()
    (out/'results.json').write_text(json.dumps(dict(passed=True, commands=commands, hashes=hashes), indent=2)+'\n')
    print('ok: GPU CLI skipped-step pause, native BF16 histories and completion exact without latent callbacks')


if __name__ == '__main__':
    main()
