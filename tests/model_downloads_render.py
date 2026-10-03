#!/usr/bin/env python3
"""Opt-in real-model download qualification on Metal or CUDA.

Existing models stay read-only. Inference is offline; --server additionally
downloads a fresh preview VAE into this run's private auxiliary root.
"""
import argparse
import hashlib
import html
import http.client
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PROMPT = 'A red wooden toy boat floating on a quiet pond. Gentle ripples and soft birdsong.'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT/'bin/h3cli')
    parser.add_argument('--models-path', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--manual-model', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backend', choices=('metal', 'cuda'), required=True)
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--ffmpeg', default=os.environ.get('H3_FFMPEG', 'ffmpeg'))
    parser.add_argument('--ffprobe', default=os.environ.get('H3_FFPROBE', 'ffprobe'))
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    binary, model, root = args.binary.resolve(), args.model.resolve(), args.models_path.resolve()
    identity = sha(binary)
    record = out/'results.json'
    result = json.loads(record.read_text()) if args.resume and record.exists() else {
        'binary_sha256': identity, 'backend': args.backend, 'cases': {}, 'comparisons': {}}
    if result['binary_sha256'] != identity:
        raise ValueError('Resume requires the identical executable')
    env = dict(os.environ, H3_OFFLINE='1', H3_TEST_MAX_EVALUATIONS='6')
    for key in ('H3_REFERENCE_MEDIA', 'H3_REFERENCE_MODEL', 'H3_REFERENCE_REGRESSION_OUT'):
        env.pop(key, None)
    common = [str(binary), '--models-path', str(root), '-d', str(model)]
    generated = ['-p', PROMPT, '--width', '256', '--height', '256', '--steps', '2', '--seed', '42']

    def save():
        record.write_text(json.dumps(result, indent=2)+'\n')
        cards = ['<!doctype html><meta charset="utf-8"><title>Model download qualification</title>',
                 '<style>body{font:16px system-ui;max-width:1000px;margin:30px auto}video,img{max-width:512px}article{margin:2em 0}</style>']
        for name, row in result['cases'].items():
            cards.append(f'<article><h2>{html.escape(name)}</h2><p>{row.get("wall_seconds",0):.3f} s · {row.get("passed",False)}</p>')
            if row.get('media'):
                file = html.escape(row['media'])
                cards.append(f'<img src="{file}">' if file.endswith('.png') else f'<video controls preload="metadata" src="{file}"></video>')
            cards.append('</article>')
        (out/'review.html').write_text('\n'.join(cards))

    def media_check(path, shape):
        probe = json.loads(subprocess.check_output([args.ffprobe, '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', str(path)]))
        video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        assert (int(video['width']), int(video['height'])) == shape[:2], probe
        if path.suffix == '.mp4':
            assert int(video['nb_read_frames']) == shape[2] and video['r_frame_rate'] == '24/1', probe
            audio = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
            assert abs(float(audio['duration'])-shape[2]/24) < .12, probe
        subprocess.run([args.ffmpeg, '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return {'sha256': sha(path), 'probe': probe}

    def run(name, flags=(), *, frames=22, operation=False, shape=None, still=False, deliver=True, command=None):
        if name in result['cases'] and result['cases'][name].get('passed'):
            return out/result['cases'][name].get('media', name+'.mp4')
        path = out/(name+('.png' if still else '.mp4'))
        cmd = list(command or common)
        if not operation:
            cmd += generated + ([] if still else ['--frames', str(frames)])
        cmd += list(map(str, flags)) + ['-o', str(path)]
        start = time.monotonic()
        row = {'argv': cmd, 'passed': False}
        result['cases'][name] = row
        try:
            with (out/(name+'.log')).open('w') as log:
                proc = subprocess.run(cmd, cwd=out, env=env, stdout=log, stderr=log, timeout=1200)
            row['wall_seconds'] = time.monotonic()-start
            row['returncode'] = proc.returncode
            if proc.returncode:
                raise RuntimeError((out/(name+'.log')).read_text()[-6000:])
            if deliver:
                row['media'] = path.name
                row.update(media_check(path, shape or (256, 256, frames)))
            row['passed'] = True
            print(name, f'{row["wall_seconds"]:.3f}s', flush=True)
        finally:
            save()
        return path

    def same(label, a, b):
        equal = sha(a) == sha(b)
        result['comparisons'][label] = equal
        save()
        assert equal, label

    base = run('text-full', ['--save-av-state', out/'base.h3av', '--save-upscale-state', out/'base.h3up', '--save-conditioning', out/'base.h3cond'], frames=56)
    run('preview', ['--preview-vae'])
    run('first-last', ['--first-frame', ROOT/'inputs/1.jpg', '--last-frame', ROOT/'inputs/2.jpg'])
    wav = out/'reference.wav'
    if not wav.exists():
        subprocess.run([args.ffmpeg, '-v', 'error', '-i', str(base), '-vn', '-ac', '2', '-ar', '32000', str(wav)], check=True)
    run('references', ['--ref-image', ROOT/'inputs/1.jpg', '--ref-image-size', 'match', '--ref-silent-video', base, '--ref-audio', wav], frames=56)
    still = run('still', ['--still', '--save-still-latent', out/'still.safetensors'], still=True)
    decoded = run('still-decode', ['--decode-still-latent', out/'still.safetensors'], still=True, operation=True)
    same('still-roundtrip', still, decoded)
    decoded = run('av-decode', ['--decode-av-state', out/'base.h3av'], operation=True, frames=56)
    same('av-roundtrip', base, decoded)
    run('av-preview-decode', ['--decode-av-state', out/'base.h3av', '--preview-vae'], operation=True, frames=56)
    run('conditioning', ['--load-conditioning', out/'base.h3cond'], frames=56)
    # A decoder-only tree has just the eight identity/decoder files. Symlinks
    # retain the original inodes required by the existing saved-state contract.
    minimal = out/'minimal-model'
    for file in ('transformer/config.json', 'video_vae/config.json', 'video_vae/source/config.json', 'video_vae/source/model.safetensors', 'audio_vae/config.json', 'audio_vae/config.yaml', 'audio_vae/metadata.json', 'audio_vae/model.safetensors'):
        dest = minimal/'FL2VA'/file
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.is_symlink():
            dest.symlink_to(model/'FL2VA'/file)
    minimal_command = common.copy()
    minimal_command[minimal_command.index('-d')+1] = str(minimal)
    decoded = run('minimal-av-decode', ['--decode-av-state', out/'base.h3av'], operation=True, frames=56, command=minimal_command)
    same('minimal-av-roundtrip', base, decoded)
    for mode in ('hard', 'bridge'):
        flags = ['--continue-from', out/'base.h3av', '--continue-context', '39', '--continue-mode', mode]
        if mode == 'bridge':
            flags += ['--continue-bridge-steps', '1']
        run('continue-'+mode, flags, frames=56, shape=(256, 256, 17))
    uninterrupted = run('sampler-uninterrupted')
    run('sampler-pause', ['--stop-after-step', '1', '--save-sampler-state', out/'paused.h3sample'], deliver=False)
    resumed = run('sampler-resume', ['--resume-sampler-state', out/'paused.h3sample'], operation=True)
    same('sampler-roundtrip', uninterrupted, resumed)
    upscale = run('upscale', ['--upscale-state', out/'base.h3up', '--upscale-refine-steps', '2'], operation=True, shape=(512, 512, 56))
    run('upscale-pause', ['--upscale-state', out/'base.h3up', '--upscale-refine-steps', '2', '--stop-after-step', '1', '--save-sampler-state', out/'upscale.h3sample'], operation=True, deliver=False)
    resume_command = common.copy()
    resume_command[resume_command.index('--models-path')+1] = str(out/'unused-empty-models')
    resumed = run('upscale-resume', ['--resume-sampler-state', out/'upscale.h3sample'], operation=True, shape=(512, 512, 56), command=resume_command)
    same('upscale-roundtrip', upscale, resumed)
    if args.manual_model:
        manual = common.copy()
        manual[manual.index('-d')+1] = str(args.manual_model.resolve())
        manual_output = run('manual-identical-weights', command=manual)
        same('managed-versus-manual', uninterrupted, manual_output)

    if args.server and not result.get('server', {}).get('passed'):
        qualify_server(args, out, common, env, result, save, media_check, run, same)
    result['passed'] = all(r['passed'] for r in result['cases'].values()) and all(result['comparisons'].values()) and (not args.server or result['server']['passed'])
    save()
    return 0 if result['passed'] else 1


def qualify_server(args, out, common, env, result, save, media_check, run, same):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    state, models = out/'server-state', out/'server-models'
    cmd = [common[0], '--server', '-d', str(args.model.resolve()), '--models-path', str(models), '--server-port', str(port), '--server-state-dir', str(state)]
    row = {'passed': False, 'jobs': [], 'argv': cmd}
    result['server'] = row

    def request(method, path, data=None):
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=30)
        conn.request(method, path, json.dumps(data) if data is not None else None, {'Content-Type': 'application/json'})
        response = conn.getresponse()
        body = json.loads(response.read())
        conn.close()
        assert 200 <= response.status < 300, (response.status, body)
        return body

    with (out/'server.log').open('w') as log:
        for offline in (False, True):
            server = subprocess.Popen(cmd, cwd=out, env=dict(env, H3_OFFLINE='1' if offline else '0'), stdout=log, stderr=log, start_new_session=True)
            try:
                deadline = time.monotonic()+30
                while True:
                    if server.poll() is not None:
                        raise RuntimeError((out/'server.log').read_text())
                    try:
                        request('GET', '/health')
                        break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise
                        time.sleep(.1)
                flags = '--width 256 --height 256 --frames 22 --steps 2 --seed 42 --preview-vae'
                start = time.monotonic()
                job = request('POST', '/v1/videos', {'prompt': PROMPT, 'h3cli': flags})
                history = []
                while time.monotonic()-start < 900:
                    job = request('GET', '/v1/h3/jobs/'+job['id'])
                    history.append({'status': job['status'], 'h3': job.get('h3')})
                    if job['status'] not in ('queued', 'running'):
                        break
                    time.sleep(.1)
                row['jobs'].append({'offline': offline, 'wall_seconds': time.monotonic()-start, 'result': job, 'observations': history})
                save()
                assert job['status'] == 'completed', job
                if not offline:
                    assert any(i['h3'].get('model_preparation') for i in history), history
                art = next(i for i in job['h3']['artifacts'] if i['name'].endswith('.mp4'))
                path = out/('server-offline.mp4' if offline else 'server-cold.mp4')
                conn = http.client.HTTPConnection('127.0.0.1', port, timeout=30)
                conn.request('GET', art['url']);response = conn.getresponse()
                assert response.status == 200
                path.write_bytes(response.read());conn.close()
                row['jobs'][-1]['media'] = media_check(path, (256, 256, 22))
            finally:
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGTERM)
                    server.wait(timeout=30)
        same('server-cold-offline', out/'server-cold.mp4', out/'server-offline.mp4')
        cli = run('server-cli-equivalent', ['--preview-vae-model', models/'preview-vae/taeh3.safetensors', '--preview-vae'])
        same('server-cli', out/'server-cold.mp4', cli)
        row['passed'] = True
        save()


if __name__ == '__main__':
    raise SystemExit(main())
