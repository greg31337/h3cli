#!/usr/bin/env python3
"""Exercise bridge parsing, warnings and rejection of unvalidated sampler settings."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import json
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/MiniMax-H3')
    parser.add_argument('--state', type=Path, required=True, help='Fresh current AV state; its canvas sets the validation request')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/bridge-validation/cli.json')
    args = parser.parse_args()
    header=args.state.read_bytes()[:160]
    assert header[:8]==b'H3AV\r\n\x1a\n' and struct.unpack_from('<I',header,8)[0]==3
    width,height=struct.unpack_from('<2I',header,24)
    base = [str(ROOT / 'bin/h3cli'), '-d', str(args.model.resolve()), '-p', 'test',
            '--width', str(width), '--height', str(height), '--frames', '90', '--steps', '6']
    source = ['--continue-from', str(args.state.resolve())]
    bridge = source + ['--continue-mode', 'bridge']
    cases = [
        ('mode', ['--continue-mode', 'typo'], 'invalid continuation mode'),
        ('mode-empty', ['--continue-mode', ''], 'invalid continuation mode'),
        ('profile', ['--continue-bridge-profile', 'typo'], 'invalid bridge profile'),
        ('no-source', ['--continue-mode', 'bridge'], 'requires --continue-from'),
        ('no-exact-row', bridge + ['--continue-bridge-steps', '12'], 'at least one exact row'),
        ('too-long', bridge + ['--continue-bridge-steps', '13'], 'at least one exact row'),
        ('zero-steps', bridge + ['--continue-bridge-steps', '0'], 'positive and shorter'),
        ('negative-steps', ['--continue-bridge-steps', '-1'], 'invalid bridge steps'),
        ('empty-steps', ['--continue-bridge-steps', ''], 'steps cannot be empty'),
        ('overflow-steps', ['--continue-bridge-steps', '2147483648'], 'invalid bridge steps'),
        ('fractional-steps', ['--continue-bridge-steps', '1.5'], 'invalid bridge steps'),
        ('context', bridge + ['--continue-context', '40'], '39 + 51*k'),
        ('one-exact-warning', bridge + ['--continue-bridge-steps', '11', '--core-reuse', '2'], 'warning: bridge retains only one exact'),
        ('one-bridge-row', bridge + ['--continue-bridge-steps', '1', '--core-reuse', '2'], 'supports --core-reuse 1, 4, or 6'),
        ('reuse-4', bridge + ['--reuse', '4'], 'denoise reuse must be in [1, 3]'),
        ('core-reuse-2', bridge + ['--core-reuse', '2'], 'supports --core-reuse 1, 4, or 6'),
        ('core-reuse-3', bridge + ['--core-reuse', '3'], 'supports --core-reuse 1, 4, or 6'),
        ('core-reuse-5', bridge + ['--core-reuse', '5'], 'supports --core-reuse 1, 4, or 6'),
        ('mixed-reuse', bridge + ['--reuse', '2', '--core-reuse', '4'], 'cannot be combined'),
        ('reduced-layers', bridge + ['--layers', '45'], 'layers 50'),
        ('token-reduction', bridge + ['--token-reduction'], 'token reduction off'),
        ('anchor', bridge + ['--first-frame', str(ROOT/'inputs/face1.jpg')], 'anchors'),
    ]
    for strength in ['', 'nan', 'inf', '-inf', '-0.1', '1.1', '0.5junk', '1e999']:
        cases.append((f'strength-{strength}', ['--continue-bridge-max-strength', strength], 'invalid bridge maximum strength'))
    for profile in ['stepped', 'linear', 'ease-out']:
        for strength in ['0', '0.5', '1']:
            cases.append((f'valid-profile-reuse-guard-{profile}-{strength}', bridge + [
                '--continue-bridge-profile', profile, '--continue-bridge-max-strength', strength, '--core-reuse', '2'], 'supports --core-reuse 1, 4, or 6'))
    records = []
    with tempfile.TemporaryDirectory(prefix='h3-bridge-cli-') as temp:
        output = Path(temp) / 'must-not-render.mp4'
        state = Path(temp) / 'must-not-save.h3av'
        for name, flags, expected in cases:
            cmd = base + flags + ['-o', str(output), '--save-av-state', str(state)]
            result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60)
            assert result.returncode != 0 and expected in result.stderr, (name, result.returncode, result.stderr)
            if name == 'one-exact-warning':
                assert 'supports --core-reuse 1, 4, or 6' in result.stderr, 'valid configuration must reach the reuse guard'
            assert not output.exists() and not state.exists(), name
            records.append({'case': name, 'returncode': result.returncode, 'stderr': result.stderr})
        for name, extra, flags, expected in [
            ('environment-token-reduction',{'H3_TOKEN_REDUCTION':'1'},[], 'token reduction off'),
            ('custom-velocity-reuse',{'H3_REUSE_STEPS':'0,5'},['--reuse','2'],'does not support custom H3_REUSE_STEPS')]:
            env={k:v for k,v in os.environ.items() if not k.startswith('H3_')}; env.update(extra)
            result=subprocess.run(base+bridge+flags+['-o',str(output),'--save-av-state',str(state)],
                                  cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
            assert result.returncode!=0 and expected in result.stderr,(name,result.stderr)
            assert not output.exists() and not state.exists()
            records.append({'case':name,'returncode':result.returncode,'stderr':result.stderr,'environment':extra})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + '\n')
    print(f'ok: {len(cases)+2} bridge CLI validation cases; no incomplete renders or states written')


if __name__ == '__main__':
    main()
