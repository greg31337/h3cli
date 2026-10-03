#!/usr/bin/env python3
"""Development-only 960x544 fixed-latent comparison; oversized tiles use an
explicit pre-change binary, never a production environment-variable bypass.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
from tilefix_validation import pixels, sha
from tilefix_metrics import compare, seams, media

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/bugfix1-validation/quality-960x544')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('H3_')}
    env['H3_PROFILE'] = '1'
    weights = ROOT / 'models/MiniMax-H3/FL2VA/video_vae/source'
    h, w = 544, 960
    latent = out / 'latent.f32'
    records = {}
    recordfile = out / 'runs.json'
    if recordfile.exists():
        records = json.loads(recordfile.read_text())

    def run(label, binary, mode, t, src, dest, tile=None):
        if label in records:
            return
        case_env = dict(env)
        if tile is not None:
            case_env['H3_VAE_TILE_PIXELS'] = tile
        cmd = [str(binary), mode, str(t), str(h), str(w), str(src), str(dest), str(weights)]
        with (out / (label + '.log')).open('w') as log:
            r = subprocess.run(cmd, env=case_env, cwd=ROOT, stdout=subprocess.PIPE, stderr=log, text=True, check=True)
        records[label] = dict(json.loads(r.stdout), command=cmd, sha256=sha(dest))
        recordfile.write_text(json.dumps(records, indent=2)+'\n')
        print('PASS', label, flush=True)

    if 'encode' not in records:
        rgb = pixels(h, w)
        Image.fromarray(rgb[0]).save(out / 'source.png')
        src = out / 'pixels.f32'
        (rgb.transpose(3, 0, 1, 2).astype(np.float32) * np.float32(1/255)).tofile(src)
        run('encode', ROOT / 'bin/tilefix_decode', 'encode', 22, src, latent)
        src.unlink()
    for label, binary, tile in [('default', ROOT / 'bin/tilefix_decode', None),
                                 ('auto', ROOT / 'bin/tilefix_decode', 'auto'),
                                 ('320', ROOT / 'bin/tilefix_decode', '320'),
                                 ('before-default', args.baseline.resolve(), None),
                                 ('before-auto', args.baseline.resolve(), 'auto'),
                                 ('legacy-512', args.baseline.resolve(), '512')]:
        run(label, binary, 'resident', 7, latent, out / (label + '.f32'), tile)
    for mode in ['default', 'auto']:
        assert records[mode]['sha256'] == records['before-' + mode]['sha256'], mode
    arrays = {label: np.memmap(out / (label + '.f32'), dtype='<f4', mode='r').reshape(-1,h,w,3)
              for label in ['default', 'auto', '320', 'legacy-512']}
    metrics = dict(default_and_auto_bitwise_unchanged=True,
                   comparisons={label: compare(arrays[label], arrays['default']) for label in ['auto', '320', 'legacy-512']},
                   seams={label: seams(array, records[label]) for label, array in arrays.items()})
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    media(out, arrays, h, w)
    print('PASS default/auto exact parity and quality evidence', flush=True)


if __name__ == '__main__':
    main()
