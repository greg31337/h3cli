#!/usr/bin/env python3
"""Real-model regression against a separately captured pre-change baseline."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/bugfix1-validation')
    parser.add_argument('--unaligned', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    baseline = json.loads((out / 'baseline/results.json').read_text())
    install = out / 'installed binary with spaces'
    install.mkdir(exist_ok=True)
    (install/'bin').mkdir(exist_ok=True)
    shutil.copy2(ROOT / 'bin/h3cli', install / 'bin/h3cli')
    (install/'src/metal').mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT / 'src/metal/shaders.metal', install / 'src/metal/shaders.metal')
    cwd = out / 'unrelated cwd'
    cwd.mkdir(exist_ok=True)
    assert not (cwd / 'src/metal/shaders.metal').exists()
    records = {}
    filename = out / ('unaligned-integration.json' if args.unaligned else 'integration.json')
    env = {k: v for k, v in os.environ.items() if not k.startswith('H3_')}
    env.update(H3_CPU_SAMPLER='1', H3_PROFILE='1')
    if args.unaligned:
        env['H3_ZERO_COPY_WEIGHTS'] = '1'
    for case in (['face'] if args.unaligned else ['face', 'image']):
        name = ('unaligned-' if args.unaligned else 'after-') + case
        cmd = baseline[case]['command'].copy()
        cmd[0] = str(install / 'bin/h3cli')
        cmd[cmd.index('--save-av-state') + 1] = str(out / (name + '.h3av'))
        cmd[cmd.index('-o') + 1] = str(out / (name + '.mp4'))
        if args.unaligned:
            cmd[cmd.index('-d') + 1] = str(out / 'unaligned-model')
        start = time.monotonic()
        with (out / (name + '.log')).open('w') as log:
            subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=log, check=True)
        record = dict(command=cmd, cwd=str(cwd), seconds=time.monotonic()-start,
                      mp4_sha256=sha(out / (name + '.mp4')))
        record['baseline_mp4_sha256'] = baseline[case]['mp4_sha256']
        record['mp4_identical'] = record['mp4_sha256'] == record['baseline_mp4_sha256']
        # Model representation fingerprints can differ in AV metadata; compare
        # decoded video/audio through the MP4, plus the aligned complete state.
        record['av_sha256'] = sha(out / (name + '.h3av'))
        record['baseline_av_sha256'] = sha(out / 'baseline' / (case + '.h3av'))
        record['av_identical'] = record['av_sha256'] == record['baseline_av_sha256']
        subprocess.run([str(ROOT / 'bin/bugfix1_probe'), 'state-check', str(out / 'baseline' / (case + '.h3av')), str(out / (name + '.h3av'))], check=True, capture_output=True)
        record['video_audio_latents_identical'] = True
        records[name] = record
        filename.write_text(json.dumps(records, indent=2)+'\n')
        assert record['mp4_identical'], record
        if not args.unaligned:
            assert record['av_identical'], record
        print('PASS', name, 'MP4 and video/audio latent parity', flush=True)


if __name__ == '__main__':
    main()
