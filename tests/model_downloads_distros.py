#!/usr/bin/env python3
"""Opt-in standalone model download/render checks in three clean userlands."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/linux'))
import environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    release, models, cache, out = (p.resolve() for p in (a.release, a.models, a.cache, a.out))
    out.mkdir(parents=True, exist_ok=False)
    binary = release/'h3cli-linux-x86_64'
    # A private CoW clone permits true read-only permissions without modifying
    # qualified weights or the download receipts and saved-state identities.
    readonly = out/'readonly-main'
    subprocess.run(['cp', '-a', '--reflink=always', str(models/'MiniMaxH3'), str(readonly)], check=True)
    for path in readonly.rglob('*'):
        path.chmod(0o555 if path.is_dir() else 0o444)
    readonly.chmod(0o555)
    rows = []
    record = {'binary_sha256': environment.digest(binary), 'checks': rows, 'passed': False}
    env = {'PATH': '/usr/sbin:/usr/bin:/sbin:/bin', 'HOME': str(out), 'LANG': 'C.UTF-8',
           'H3CLI_RUNTIME_CACHE': str(out/'runtime-cache'), 'H3_TEST_MAX_EVALUATIONS': '6'}

    def run(name, command, offline=False, expected=0):
        start = time.monotonic()
        with (out/(name+'.log')).open('w') as log:
            proc = subprocess.run(list(map(str, command)), cwd=out, env=dict(env, H3_OFFLINE='1' if offline else '0'), stdout=log, stderr=log, timeout=600)
        rows.append({'name': name, 'argv': list(map(str, command)), 'seconds': time.monotonic()-start, 'returncode': proc.returncode, 'passed': proc.returncode == expected})
        (out/'results.json').write_text(json.dumps(record, indent=2)+'\n')
        assert proc.returncode == expected, (out/(name+'.log')).read_text()[-5000:]

    for distro in ('ubuntu2204', 'ubuntu2404', 'debian12'):
        root = environment.prepare(cache, distro)
        prefix = environment.command(cache, root, [str(release)+':'+str(release), str(out)+':'+str(out)], gpu=True, cwd=str(out))
        run(distro+'-inventory', [*prefix, 'sh', '-c', 'test ! -e /usr/bin/python3 && test ! -e /usr/bin/curl && test ! -e /usr/bin/ffmpeg && test ! -e /usr/local/cuda && cat /etc/os-release'])
        selected = out/(distro+' models')
        run(distro+'-https', [*prefix, binary, '--models-path', selected, '--download-models', 'preview'])
        preview = selected/'preview-vae/taeh3.safetensors'
        assert environment.digest(preview) == '4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13'
        preview.chmod(0o444);preview.parent.chmod(0o555);selected.chmod(0o555)
        movie = out/(distro+'.mp4')
        run(distro+'-offline', [*prefix, binary, '--models-path', selected, '-d', readonly, '--preview-vae', '-p', 'A red wooden toy boat floating on a quiet pond. Gentle ripples and soft birdsong.', '--width', '256', '--height', '256', '--frames', '22', '--steps', '2', '--seed', '42', '-o', movie], offline=True)
        tools = release/'linux-runtime/tools'
        probe = json.loads(subprocess.check_output([str(tools/'ffprobe'), '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', str(movie)]))
        video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        assert (video['width'], video['height'], int(video['nb_read_frames'])) == (256, 256, 22)
        subprocess.run([str(tools/'ffmpeg'), '-v', 'error', '-xerror', '-i', str(movie), '-f', 'null', '-'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        rows[-1]['media_sha256'] = environment.digest(movie)
        rows[-1]['probe'] = probe
    assert len({r['media_sha256'] for r in rows if 'media_sha256' in r}) == 1
    record['passed'] = True
    (out/'results.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
