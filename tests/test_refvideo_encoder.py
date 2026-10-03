#!/usr/bin/env python3
"""Run native encoders sequentially against retained official Diffusers fixtures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--binary', type=Path, default=ROOT/'bin/refvideo_encoder_test')
    p.add_argument('--smoke', action='store_true', help='raw/tiled/stitch and 60-frame cases for sanitizer runs')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    results = []; commands = []

    def run(mode, name, key, binary=None, label=None, seed=None):
        label = label or f'{mode}-{name}'
        output = args.output/(label+'.f32')
        command = [str((binary or args.binary).resolve()), mode,
                   str(ROOT/'models/MiniMax-H3/Ref2VA/video_vae/source'),
                   str((args.fixtures/(name+'.safetensors')).resolve()), str(output.resolve())]
        commands.append(command)
        env = {k:v for k,v in os.environ.items() if not k.startswith('H3_')}
        if seed is not None: env['H3_TEST_REQUEST_SEED'] = str(seed)
        print('running', label, flush=True)
        process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        (args.output/(label+'.log')).write_text(process.stdout+process.stderr)
        assert process.returncode == 0, (label, process.stderr)
        shape = json.loads(process.stdout)
        want = load_file(args.fixtures/(name+'.safetensors'))[key]
        assert list(want.shape) == [1,shape['channels'],shape['time'],shape['height'],shape['width']]
        got = np.fromfile(output, dtype=np.float32).reshape(want.shape)
        assert np.isfinite(got).all()
        delta = got.astype(np.float64)-want
        maximum = float(np.abs(delta).max())
        relative = float(np.linalg.norm(delta)/max(np.linalg.norm(want), 1e-30))
        # CNN F32 error may cross an FP16 midpoint. Bound both the individual
        # quantized steps and whole-tensor error; never demand bitwise CNN parity.
        absolute_limit, relative_limit = (3e-3, 3e-4) if mode == 'released' else (2e-3, 2e-4)
        if mode == 'stitch': absolute_limit, relative_limit = 1e-6, 1e-7
        assert maximum <= absolute_limit and relative <= relative_limit, (label,maximum,relative)
        if mode == 'released':
            rows = load_file(args.fixtures/(name+'.safetensors'))['x.rows']
            packed = np.fromfile(str(output)+'.rows', dtype=np.float32).reshape(rows.shape)
            diff = packed.astype(np.float64)-rows
            assert np.max(np.abs(diff)) <= absolute_limit
            assert np.linalg.norm(diff)/np.linalg.norm(rows) <= relative_limit
            # Packing must introduce no additional error or permutation.
            expected = got.reshape(1,24,shape['time'],shape['height']//2,2,shape['width']//2,2)
            expected = expected.transpose(0,2,3,5,1,4,6).reshape(rows.shape)
            assert np.array_equal(packed, expected)
        results.append({'case':label, 'max_abs':maximum, 'relative_l2':relative,
            'absolute_limit':absolute_limit, 'relative_l2_limit':relative_limit,
            'shape':list(want.shape), 'sha256':hashlib.sha256(output.read_bytes()).hexdigest(), 'request_seed':seed})
        print(f'passed {label}: max_abs={maximum:.7g} relative_l2={relative:.7g}', flush=True)
        return output

    run('raw', 'raw17', 'x.moments')
    run('raw', 'tiled', 'x.moments')
    run('stitch', 'stitch', 'x.moments')
    if args.smoke:
        run('temporal', 'temporal60', 'x.moments')
        run('released', 'temporal60', 'x.normalized', seed=987)
    else:
        for name in ['legacy-image','legacy-video','tiled']:
            current = run('legacy', name, 'x.legacy')
            if args.baseline:
                old = run('legacy', name, 'x.legacy', args.baseline, label='baseline-'+name)
                assert old.read_bytes() == current.read_bytes(), f'legacy changed: {name}'
        temporal56 = None
        for frames in [39,48,56,60,73,80,107,110,124]:
            name = f'temporal{frames}'
            moments = run('temporal', name, 'x.moments')
            released = run('released', name, 'x.normalized', seed=72)
            if frames == 56:
                temporal56 = moments
                again = run('released', name, 'x.normalized', label='released56-seed987', seed=987)
                assert again.read_bytes() == released.read_bytes()
                continuous = run('raw', name, 'x.continuous')
                a = np.fromfile(moments, dtype=np.float32).reshape(48,17,4,4)
                b = np.fromfile(continuous, dtype=np.float32).reshape(48,14,4,4)
                assert np.max(np.abs(a[:,5:14]-b[:,5:14])) > 1e-2
            if frames == 60:
                assert temporal56.read_bytes() == moments.read_bytes(), 'normalized RGB stride/prefix mismatch'
        run('temporal', 'temporal39-tiled', 'x.moments')
        run('released', 'temporal39-tiled', 'x.normalized', seed=72)
    manifest = {'passed':True,'cases':results,'commands':commands,
        'oracle_manifests':{p.name:json.loads(p.read_text()) for p in args.fixtures.glob('*-manifest.json')},
        'fixture_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.fixtures.glob('*.safetensors')},
        'binary_sha256':hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        'legacy_exact':bool(args.baseline) and not args.smoke}
    (args.output/'results.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'ok: {len(results)} native encoder oracle comparisons', flush=True)


if __name__ == '__main__': main()
