#!/usr/bin/env python3
"""Opt-in batch CLI progress and unchanged-output check against saved baseline."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/progress-validation'
BASE = ROOT / 'outputs/bugfix1-validation/baseline'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def terminal_lines(raw):
    line, cursor, lines = [], 0, []
    for char in raw:
        if char == '\r':
            cursor = 0
        elif char == '\n':
            lines.append(''.join(line).strip())
            line, cursor = [], 0
        else:
            if cursor == len(line):
                line.append(char)
            else:
                line[cursor] = char
            cursor += 1
    if line:
        lines.append(''.join(line).strip())
    return lines


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    baseline = json.loads((BASE / 'results.json').read_text())['face']
    cmd = baseline['command'].copy()
    cmd[0] = str(ROOT / 'bin/h3cli')
    cmd[cmd.index('--save-av-state') + 1] = str(OUT / 'face.h3av')
    cmd[cmd.index('-o') + 1] = str(OUT / 'face.mp4')
    env = {k: v for k, v in os.environ.items() if not k.startswith('H3_')}
    env['H3_CPU_SAMPLER'] = '1'
    # No profiling chatter: validate the actual terminal view of progress.
    started = time.monotonic()
    with (OUT / 'cli.log').open('wb') as log:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=log, check=True)
    wall_seconds = time.monotonic() - started
    raw = (OUT / 'cli.log').read_bytes().decode()
    lines = terminal_lines(raw)
    (OUT / 'terminal.txt').write_text('\n'.join(lines) + '\n')
    assert not re.search(r'\b0/[01]\b', raw)
    assert '\r' in raw and 'phase start' not in raw
    for phase in ['reference vision preparation', 'DiT initialization']:
        assert any(line.startswith(phase) and re.search(r'\b1/1\s+\([0-9.]+ s\)$', line) for line in lines), phase
    assert any(line.startswith('video VAE encoder') and re.search(r'\b1/1\s+\([0-9.]+ s\)$', line) for line in lines)
    assert 'loading...' in raw and 'starting...' in raw
    for phase in ['video VAE load', 'video VAE decode']:
        rows = [line for line in lines if line.startswith(phase)]
        assert len(rows) == 1, (phase, rows)
    assert any(line.startswith('video VAE load') and re.search(r'\b36/36\s+\([0-9.]+ s\)$', line) for line in lines)
    # 56 frames = three temporal chunks; 128px fits one spatial tile.
    assert any(line.startswith('video VAE decode') and re.search(r'\b108/108\s+\([0-9.]+ s\)$', line) for line in lines)
    timing = re.fullmatch(r'h3(?:cli)?: total wall time: (\d+\.\d{2}) s', lines[-1])
    assert timing, lines[-1]
    reported_seconds = float(timing[1])
    assert 0 < reported_seconds <= wall_seconds + 0.02
    assert abs(reported_seconds - wall_seconds) < max(1.0, wall_seconds * 0.05)
    assert any(line == f'h3cli: wrote {OUT / "face.mp4"}' for line in lines[:-1])
    assert sha(OUT / 'face.mp4') == baseline['mp4_sha256']
    assert sha(OUT / 'face.h3av') == sha(BASE / 'face.h3av')
    (OUT / 'cli-results.json').write_text(json.dumps(dict(command=cmd,
        mp4_sha256=sha(OUT / 'face.mp4'), av_state_sha256=sha(OUT / 'face.h3av'),
        mp4_identical=True, av_state_identical=True, completed_progress=True,
        vae_load_lines=sum(line.startswith('video VAE load') for line in lines),
        vae_decode_lines=sum(line.startswith('video VAE decode') for line in lines), vae_decode_blocks=108,
        reported_wall_seconds=reported_seconds, measured_wall_seconds=wall_seconds), indent=2) + '\n')
    print('PASS real CLI progress, total wall time, and bit-identical MP4/AV state')


if __name__ == '__main__':
    main()
