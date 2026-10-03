#!/usr/bin/env python3
"""Bounded CUDA replays of the approved Metal saved-state corpus, without sampling."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys

from preview_vae_gallery import inspect

ROOT = Path('outputs/preview-vae/cuda-5090')
QUALITY = ROOT/'quality'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def charge(name, timeout, command, env=None):
    subprocess.run([sys.executable, 'tests/preview_vae_run.py', '--ledger',
                    str(ROOT/'budget.json'), '--budget', '1800', '--name', name,
                    '--timeout', str(timeout), '--', *command], env=env, check=True)


def render():
    QUALITY.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path('outputs/preview-vae/metal/quality/manifest.json').read_text())
    records = []
    ledger = json.loads((ROOT/'budget.json').read_text())
    for case in source:
        name = case['name']
        state = QUALITY/(name+'.h3av')
        assert digest(case['replay_state']) == case['state_sha256']
        for suffix in ['', '.presentation']:
            shutil.copy2(case['replay_state']+suffix, str(state)+suffix)
        row = {k: v for k, v in case.items() if k not in ['full', 'tiny', 'reference', 'commands']}
        row.update(replay_state=str(state), outputs={}, commands={})
        for mode in ['metal-tiny', 'reference']:
            original = case['tiny' if mode == 'metal-tiny' else mode]
            assert digest(original['path']) == original['sha256']
            destination = QUALITY/f'{name}-{mode}.mp4'
            shutil.copy2(original['path'], destination)
            row['outputs'][mode] = dict(path=str(destination), sha256=digest(destination), reused=True)
        for mode in ['full', 'tiny', 'cublas']:
            output = QUALITY/f'{name}-{mode}.mp4'
            command = ['./bin/h3cli', '-d', 'models/MiniMax-H3', '--decode-av-state', str(state),
                       '--profile', '-o', str(output)]
            env = dict(os.environ, H3_PREVIEW_CUDA_CONV='cublas' if mode == 'cublas' else 'auto')
            if mode != 'full':
                command.append('--preview-vae')
            baseline = {'full': ('baseline-fast-image05', 'image05-before-fast.mp4'),
                        'tiny': ('tiny-pilot', 'image05-tiny.mp4')}
            if name == 'image05' and mode in baseline:
                run_name, file = baseline[mode]
                record = next(r for r in ledger['runs'] if r['name'] == run_name)
                assert record['status'] == 'passed' and digest(ROOT/file) == record['output_sha256'][str(ROOT/file)]
                shutil.copy2(ROOT/file, output)
                row['commands'][mode] = dict(command=record['command'], environment=record['environment'], reused=True)
            else:
                charge(f'gallery-{name}-{mode}', 120, command, env)
                row['commands'][mode] = dict(command=command, environment={'H3_PREVIEW_CUDA_CONV': env['H3_PREVIEW_CUDA_CONV']})
            row['outputs'][mode] = dict(path=str(output), sha256=digest(output))
        records.append(row)
        (QUALITY/'manifest.json').write_text(json.dumps(records, indent=2)+'\n')


def pcm(path):
    return subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-vn', '-f', 'f32le', '-'])


def validate():
    records = json.loads((QUALITY/'manifest.json').read_text())
    assert len(records) == 8
    for row in records:
        for mode, artifact in row['outputs'].items():
            assert digest(artifact['path']) == artifact['sha256']
            artifact['media'] = inspect(Path(artifact['path']), row['frames'], row['width'], row['height'])
        audio = pcm(row['outputs']['full']['path'])
        for mode in ['tiny', 'cublas']:
            assert pcm(row['outputs'][mode]['path']) == audio, (row['name'], mode)
    (QUALITY/'manifest.json').write_text(json.dumps(records, indent=2)+'\n')
    joins = []
    for mode in ['full', 'tiny', 'cublas']:
        output = QUALITY/f'joined-{mode}.mp4'
        command = ['ffmpeg', '-v', 'error', '-y', '-i', str(QUALITY/f'chain1-{mode}.mp4'),
                   '-i', str(QUALITY/f'chain2-{mode}.mp4'), '-filter_complex',
                   '[0:a]atrim=duration=3.75,asetpts=PTS-STARTPTS[a0];[1:a]atrim=duration=2.125,asetpts=PTS-STARTPTS[a1];[0:v][a0][1:v][a1]concat=n=2:v=1:a=1[v][a]',
                   '-map', '[v]', '-map', '[a]', '-c:v', 'libx264', '-preset', 'fast', '-crf', '6',
                   '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', str(output)]
        subprocess.run(command, check=True, timeout=60)
        joins.append(dict(mode=mode, command=command, sha256=digest(output), media=inspect(output, 141, 288, 384)))
    (QUALITY/'joins.json').write_text(json.dumps(joins, indent=2)+'\n')
    cards = []
    for row in records:
        videos = ''.join(f'<figure><figcaption>{html.escape(mode)}</figcaption><video controls preload="metadata" src="{Path(row["outputs"][mode]["path"]).name}"></video></figure>'
                         for mode in ['full', 'tiny', 'cublas', 'metal-tiny', 'reference'])
        cards.append(f'<section><h2>{html.escape(row["name"])}</h2><p>{row["width"]}×{row["height"]}, {row["frames"]} frames; trim {row["trim"]}. Same saved latents.</p><div>{videos}</div></section>')
    joined = ''.join(f'<figure><figcaption>{j["mode"]}</figcaption><video controls preload="metadata" src="joined-{j["mode"]}.mp4"></video></figure>' for j in joins)
    (QUALITY/'review.html').write_text('''<!doctype html><meta charset="utf-8"><title>RTX 5090 TAEH3 preview review</title>
<style>body{font:16px system-ui;background:#16191e;color:#eee;margin:24px}section{border-top:1px solid #555;margin-top:28px}section div{display:flex;gap:16px;flex-wrap:wrap}figure{margin:0;width:280px}video{width:100%;max-height:500px}a{color:#9cf}</style>
<h1>RTX 5090: full VAE / cuDNN tiny / cuBLAS tiny / accepted Metal tiny / pinned reference</h1>
<p>CUDA preview quality acceptance is pending. Review identity, action, motion, fine detail, color, continuation joins and audiovisual sync. Play one soundtrack at a time. The six-step face fixture is broken in the original decoder too; face20 is the higher-quality face comparison. These are decoder replays, with no new denoising.</p>
<p><a href="manifest.json">Commands, hashes and media metadata</a></p>
<button onclick="document.querySelectorAll('video').forEach(v=>{v.pause();v.currentTime=0})">Reset all</button>
'''+''.join(cards)+f'<section><h2>Continuation join</h2><p>90+51 frames, join at 3.75 s. <a href="joins.json">Provenance</a></p><div>{joined}</div></section>\n')
    print('PASS eight full/cuDNN/cuBLAS/Metal/reference comparisons, unchanged CUDA audio and three 141-frame joins')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['render', 'validate'])
    args = parser.parse_args()
    if args.operation == 'render':
        render()
        charge('gallery-media-validation', 120, [sys.executable, __file__, 'validate'])
    else:
        validate()
