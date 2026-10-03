#!/usr/bin/env python3
"""Serial full-canvas and raw-condition probes; each process uses <=6 evaluations."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--group', choices=['all', 'canvases', 'conditions'], default='all')
    p.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parent/'fixtures/cuda-reference')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy() | {'H3_TEST_MAX_EVALUATIONS': '6', 'H3_PROFILE': '1',
                               'H3_EXPERIMENT_TRACE': '1'}
    root = Path(__file__).resolve().parents[1]
    result = {'passed': False, 'runs': []}

    def run(name, mode, out, extra=()):
        args = [str(root/'bin/upscale_probe'), mode, str(a.model), str(a.weights), str(out), *map(str, extra)]
        begin = time.monotonic()
        with (a.out/(name+'.log')).open('x') as log:
            proc = subprocess.run(args, cwd=root, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=2400)
        result['runs'].append({'name': name, 'command': args, 'seconds': time.monotonic()-begin,
                               'returncode': proc.returncode})
        (a.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        assert proc.returncode == 0, name

    try:
        if a.group in ('all', 'canvases'):
            for w, h in ((672, 384), (960, 544)):
                d = a.out/f'{w}x{h}'
                d.mkdir()
                run(d.name, 'canvas', d, (w, h))
                run(d.name+'-direct', 'direct', d, (2*w, 2*h))
        if a.group in ('all', 'conditions'):
            for kind in ('first', 'last', 'both', 'mixed', 'max', 'cancel'):
                d = a.out/kind
                d.mkdir()
                for file in ('first.png', 'portrait.jpg', 'reference.mp4'):
                    shutil.copyfile(a.fixtures/file, d/file)
                if kind in ('mixed', 'max'):
                    # Standalone/native paired reference audio requires >=2 s.
                    # Derive an owned 56-frame fixture; never alter the goldens.
                    (d/'reference.mp4').rename(d/'short.mp4')
                    subprocess.run([env.get('H3_FFMPEG','ffmpeg'),'-v','error','-stream_loop','-1','-i',str(d/'short.mp4'),
                                    '-frames:v','56','-t',str(56/24),'-c:v','libx264','-c:a','aac',str(d/'reference.mp4')],env=env,check=True)
                    (d/'short.mp4').unlink()
                subprocess.run([env.get('H3_FFMPEG', 'ffmpeg'), '-v', 'error', '-i', str(d/'reference.mp4'),
                                '-vn', '-ar', '32000', '-ac', '2', str(d/'audio.wav')], env=env, check=True)
                run('capture-'+kind, 'capture', d, (kind,))
                for file in ('first.png', 'portrait.jpg', 'reference.mp4', 'audio.wav'):
                    (d/file).unlink()
                if kind != 'cancel':
                    (d/'source.h3up').rename(d/'moved.h3up')
                    run('retarget-'+kind, 'retarget', d)
        result['passed'] = True
    finally:
        (a.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
