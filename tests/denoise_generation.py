#!/usr/bin/env python3
"""Sequential released-weight preview comparisons and visual fixtures on M4."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from memory_generation import digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/denoise-validation'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quality', action='store_true')
    parser.add_argument('--default', action='store_true', help='also compare unset and empty preview modes')
    parser.add_argument('--force', action='store_true', help='repeat even an unchanged recorded case')
    parser.add_argument('--only', default='cpu-fl2va,cpu-image,cpu-video,cpu-hard,cpu-bridge,gpu-fl2va,gpu-image,gpu-video,gpu-hard,gpu-bridge')
    args = parser.parse_args()
    assert not (args.quality and args.default)
    out = OUT / ('quality' if args.quality else 'default' if args.default else 'invariance')
    out.mkdir(parents=True, exist_ok=True)
    results = out / 'results.json'
    records = json.loads(results.read_text()) if results.exists() else {}
    reference = OUT / 'reference.mp4'
    if not reference.exists():
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-loop', '1', '-i', str(ROOT / 'inputs/body1.jpg'),
                        '-vf', "scale=256:256,zoompan=z='1+0.0008*on':x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d=48:s=256x256:fps=24",
                        '-frames:v', '48', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(reference)], check=True)
    binary = ROOT / 'bin/preview_capture_generate' if args.quality else ROOT / 'bin/preview_generate'
    if args.quality:
        subprocess.run(['python3', str(ROOT / 'tests/denoise_capture.py')], check=True)
    binary_hash = digest(binary)
    shader_hash = digest(ROOT / 'src/metal/shaders.metal')
    for name in args.only.split(','):
        backend, case = name.split('-')
        if backend not in ('cpu', 'gpu'):
            parser.error('backend must be cpu or gpu')
        if args.quality and backend != 'cpu':
            parser.error('quality effective-velocity capture requires CPU state')
        chained = case == 'chain'
        if chained:
            case = 'bridge'
        directory = out / name
        directory.mkdir(exist_ok=True)
        base = directory / 'run'
        source = '-'
        if case in ('hard', 'bridge'):
            source = out / f'{backend}-{"hard" if chained else "fl2va"}' / ('run.denoised.h3av' if args.quality else 'run.off.h3av')
            assert source.is_file(), source
        reuse = {'fl2va': 1, 'image': 2, 'video': 3, 'hard': 2, 'bridge': 3}[case]
        cmd = [str(binary), str(ROOT / 'models/MiniMax-H3'), str(base), case, backend,
               '256' if args.quality else '128', '56', '20' if args.quality else '6', str(reuse), str(source)]
        prior = records.get(name, {})
        if (not args.force and prior.get('command') == cmd and
                prior.get('binary_sha256') == binary_hash and prior.get('shader_sha256') == shader_hash):
            print('CACHED', name, flush=True)
            continue
        env = {k: v for k, v in os.environ.items() if not k.startswith('H3_')}
        env['H3_PROFILE'] = '1'
        if args.default:
            env['H3_TEST_PREVIEW_DEFAULT'] = '1'
        if args.quality:
            env['H3_TEST_PREVIEW_ONLY'] = 'denoised'
            env['H3_TEST_PREVIEW_CAPTURE'] = str(base)
        else:
            env['H3_TEST_PREVIEW_PAUSE'] = '1'
            if case == 'fl2va':
                env['H3_TEST_PREVIEW_CANCEL'] = '1'
        print('RUN', name, 'quality' if args.quality else 'default' if args.default else 'invariance', flush=True)
        start = time.monotonic()
        with (directory / 'run.log').open('w') as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=log, check=True)
        record = {'command': cmd, 'seconds': time.monotonic() - start,
                  'binary_sha256': binary_hash, 'shader_sha256': shader_hash,
                  'runs': {f.stem: json.loads(f.read_text()) for f in directory.glob('run.*.json')}}
        if not args.quality:
            for extension in ('mp4', 'h3av', 'h3sample', 'pause.h3sample'):
                modes = ('off', 'noisy', 'denoised', 'default', 'empty') if args.default else ('off', 'noisy', 'denoised')
                hashes = {mode: digest(directory / f'run.{mode}.{extension}') for mode in modes}
                assert len(set(hashes.values())) == 1, (name, extension, hashes)
                record[extension + '_sha256'] = hashes
            for extension in ('mp4', 'h3av'):
                assert digest(directory / f'run.resumed.{extension}') == digest(directory / f'run.off.{extension}')
        records[name] = record
        results.write_text(json.dumps(records, indent=2) + '\n')
        print('PASS', name, round(record['seconds'], 1), 'seconds', flush=True)


if __name__ == '__main__':
    main()
