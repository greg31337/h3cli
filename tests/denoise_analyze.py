#!/usr/bin/env python3
"""Independent arithmetic checks and matched preview galleries (numpy/Pillow)."""
import argparse
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from test_sampler_file import entries

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/denoise-validation'
STEPS = (1, 2, 3, 5, 10, 15, 20)


def metrics(actual, expected):
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return {'max_abs': float(np.max(np.abs(delta))),
            'rmse': float(np.sqrt(np.mean(delta * delta))),
            'relative_l2': float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-30))}


def gallery(base, output, kinds, steps=STEPS):
    width, height = 224, 248
    canvas = Image.new('RGB', (width * len(steps), height * len(kinds)), '#20242a')
    draw = ImageDraw.Draw(canvas)
    for row, (kind, title) in enumerate(kinds):
        for column, step in enumerate(steps):
            frame = Image.open(f'{base}.{kind}-step-{step:02d}.ppm').convert('RGB')
            frame.thumbnail((width, width))
            canvas.paste(frame, (column * width, row * height + 24))
            draw.text((column * width + 5, row * height + 6), f'{title} / step {step}', fill='white')
    canvas.save(output)


def analyze(name, decode):
    directory = OUT / 'quality' / name
    base = directory / 'run'
    if decode:
        with (directory / 'decode.log').open('w') as log, (directory / 'decode.json').open('w') as result:
            subprocess.run([str(ROOT / 'bin/preview_decode'), str(ROOT / 'models/MiniMax-H3'), str(base)],
                           cwd=ROOT, stdout=result, stderr=log, check=True)
    stats = json.loads((directory / 'run.denoised.json').read_text())
    parts = {e[0]: e[-1] for e in entries(Path(f'{base}.denoised.h3sample').read_bytes())}
    total = struct.unpack_from('<i', parts[1], 4)[0]
    vt, lh, lw = struct.unpack_from('<3i', parts[1], 172)
    prefix = struct.unpack_from('<i', parts[10], 68)[0]
    nv, na = stats['video_elements'], stats['audio_elements']
    assert nv == 24 * vt * lh * lw
    trace = np.fromfile(f'{base}.trace', '<f4').reshape(total + 1, nv + na)
    sigmas = np.frombuffer(parts[11], '<f4', count=total + 1, offset=4)
    report = {'shape': [24, vt, lh, lw], 'prefix_latent_steps': prefix,
              'generation': stats, 'steps': {}, 'decoder': json.loads((directory / 'decode.json').read_text())}
    for step in range(1, total + 1):
        velocity = np.fromfile(f'{base}.velocity-step-{step:02d}.f32', '<f4')
        x0 = np.fromfile(f'{base}.x0-step-{step:02d}.f32', '<f4')
        before, after = trace[step - 1, :nv], trace[step, :nv]
        # Float64 multiplication/addition rounds once to float32, independently
        # of the production helper's fmaf; zero velocity/sigma preserves bits.
        expected = (after.astype(np.float64) + float(sigmas[step]) * velocity.astype(np.float64)).astype('<f4')
        expected[velocity == 0] = after[velocity == 0]
        if sigmas[step] == 0:
            expected = after.copy()
        assert x0.tobytes() == expected.tobytes(), (name, step, metrics(x0, expected))
        pre = (before.astype(np.float64) + float(sigmas[step - 1]) * velocity.astype(np.float64)).astype('<f4')
        identity = metrics(x0, pre)
        assert identity['max_abs'] < 2e-4 and identity['relative_l2'] < 2e-6, (name, step, identity)
        if name.endswith('-hard'):
            shape = (24, vt, lh, lw)
            assert np.all(velocity.reshape(shape)[:, :prefix] == 0)
            assert x0.reshape(shape)[:, :prefix].tobytes() == trace[0, :nv].reshape(shape)[:, :prefix].tobytes()
        if step == total:
            assert x0.tobytes() == after.tobytes()
        record = {'pre_post_identity': identity, 'post_formula_byte_exact': True}
        if step in STEPS:
            actual = np.asarray(Image.open(f'{base}.denoised.step-{step:02d}.ppm'))
            offline = np.asarray(Image.open(f'{base}.decoded-x0-step-{step:02d}.ppm'))
            assert np.array_equal(actual, offline), (name, step, 'offline/public decoder mismatch')
            noisy = np.asarray(Image.open(f'{base}.noisy-step-{step:02d}.ppm'))
            record['noisy_denoised_pixel_rmse'] = metrics(noisy, offline)['rmse']
            if step == total:
                assert np.array_equal(noisy, offline)
        report['steps'][str(step)] = record
    gallery(base, directory / 'comparison.jpg', [('noisy', 'Noisy'), ('decoded-x0', 'Denoised')])
    if name.endswith(('-hard', '-bridge')):
        gallery(base, directory / 'continuation.jpg',
                [('noisy-prefix', 'Noisy prefix'), ('x0-prefix', 'Denoised prefix'),
                 ('noisy-transition', 'Noisy transition'), ('x0-transition', 'Denoised transition'),
                 ('noisy-suffix', 'Noisy suffix'), ('x0-suffix', 'Denoised suffix')], (1, 10, 20))
        if name.endswith('-hard'):
            frames = [np.asarray(Image.open(f'{base}.x0-prefix-step-{step:02d}.ppm')) for step in (1, 10, 20)]
            assert all(np.array_equal(frames[0], f) for f in frames[1:])
            report['preserved_prefix_pixels_byte_exact'] = True
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decode', action='store_true')
    parser.add_argument('--only', default='cpu-fl2va,cpu-image,cpu-video,cpu-hard,cpu-bridge')
    args = parser.parse_args()
    output = OUT / 'quality/analysis.json'
    records = json.loads(output.read_text()) if output.exists() else {}
    for name in args.only.split(','):
        print('ANALYZE', name, flush=True)
        records[name] = analyze(name, args.decode)
        output.write_text(json.dumps(records, indent=2) + '\n')
        print('PASS', name, flush=True)
    sections = []
    for name in sorted(records):
        extra = (f'<img src="{name}/continuation.jpg" alt="Prefix and transition previews">'
                 if name.endswith(('-hard', '-bridge')) else '')
        sections.append(f'<section><h2>{name}</h2><img src="{name}/comparison.jpg" '
                        f'alt="Noisy and denoised previews from identical sampler states">{extra}'
                        f'<p><a href="{name}/run.denoised.mp4">Final video</a></p></section>')
    (output.parent / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Denoised preview comparisons</title>'
        '<style>body{background:#181c22;color:#eee;font:16px system-ui;margin:2rem}'
        'img{display:block;max-width:100%;margin:1rem 0}a{color:#8acaff}'
        'section{margin:3rem 0}</style><h1>Denoised preview comparisons</h1>'
        '<p>Matched raw-state and clean-estimate VideoVAE decodes from identical '
        '20-step trajectories. Final previews match exactly. Early estimates may change.</p>'
        + ''.join(sections))


if __name__ == '__main__':
    main()
