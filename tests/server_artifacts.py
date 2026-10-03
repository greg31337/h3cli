#!/usr/bin/env python3
"""Validate an existing real-M4 campaign; does not run inference."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import contextlib
import hashlib
import json
import socket
import sqlite3
import struct
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('campaign', nargs='?', type=Path, default=ROOT / 'outputs/server-validation/m4')
p.add_argument('--http', action='store_true', help='restart the local service to check repeated downloads and the example client')
args = p.parse_args()
cases = json.loads((args.campaign / 'results.json').read_text())['cases']
report = {'native_loaders': [], 'media': [], 'skipped_cases': [], 'cuda_tested': False}

for name, case in cases.items():
    if case.get('result', {}).get('status') != 'completed':
        report['skipped_cases'].append({'case': name, 'status': case.get('result', {}).get('status')})
        continue
    result = case['result']['h3']['variants'][0]['result']
    for artifact in case['artifacts']:
        path = ROOT / artifact['local_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact['sha256'], path
        if path.suffix in ('.h3av', '.h3up', '.h3sample', '.h3cond', '.safetensors'):
            subprocess.run([str(ROOT / 'bin/server_artifacts'), str(path)], check=True)
            report['native_loaders'].append(artifact['local_path'])
        if path.suffix not in ('.png', '.mp4', '.ppm'):
            continue
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
                         '-show_streams', '-of', 'json', str(path)]))
        video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        assert (video['width'], video['height']) == (result['width'], result['height']), path
        frames = int(video['nb_read_frames'])
        if path.suffix == '.mp4':
            assert frames == result['frames'], (path, frames, result['frames'])
            assert Fraction(video['avg_frame_rate']) == 24, path
            audio = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
            if result['kind'] == 'paused':
                assert audio is None, path  # Native preview-on-stop deliberately has no audio decode.
            else:
                assert audio and int(audio['sample_rate']) == 32000 and audio['channels'] == 2, path
                # AAC framing can round the final packet; the worker reports exact PCM samples.
                assert abs(float(audio['duration']) - result['audio_samples'] / 32000) <= .033, path
        else:
            assert frames == 1 and not any(s['codec_type'] == 'audio' for s in probe['streams']), path
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        report['media'].append({'case': name, 'path': artifact['local_path'], 'frames': frames,
                                'width': video['width'], 'height': video['height']})

def artifact_bytes(case, suffix):
    artifacts = [a for a in cases[case]['artifacts'] if a['name'].endswith(suffix)]
    assert len(artifacts) == 1, (case, suffix)
    return (ROOT / artifacts[0]['local_path']).read_bytes()

def audio_payload(case):
    data = artifact_bytes(case, '.h3av')
    assert data[:8] == b'H3AV\r\n\x1a\n' and struct.unpack_from('<I', data, 8)[0] == 3
    header = struct.unpack_from('<I', data, 12)[0]
    video, audio = struct.unpack_from('<QQ', data, 72)
    payload = data[header + video:]
    assert len(payload) == audio
    return payload

source = audio_payload('M01a')
for name in ('M08-zero', 'M08-refine'):
    assert audio_payload(name) == source, ('upscale changed audio latent', name)
report['upscale_audio_byte_identical'] = True
report['upscale_audio_sha256'] = hashlib.sha256(source).hexdigest()
assert artifact_bytes('M09-still', '.png') == artifact_bytes('M09-decode', '.png')
report['still_decode_byte_identical'] = True
assert cases['M01a']['cli_equivalence']['mp4_byte_identical']
assert cases['M01a']['cli_equivalence']['av_byte_identical']
report['cli_server_byte_identical'] = True

# Both endpoints must resemble their own center-cropped anchor more closely
# than the other anchor. This checks placement/order, not SGLang parity.
video = next(a for a in cases['M03']['artifacts'] if a['name'].endswith('.mp4'))
frames = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(ROOT / video['local_path']),
                                  '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
anchors = [subprocess.check_output(['ffmpeg', '-v', 'error', '-i',
           str(args.campaign / 'fixtures' / f'{i}.png'), '-vf',
           'scale=256:256:force_original_aspect_ratio=increase,crop=256:256',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']) for i in (1, 2)]
n = 256 * 256 * 3
distance = [[sum(abs(x-y) for x, y in zip(frame, anchor)) / n for anchor in anchors]
            for frame in (frames[:n], frames[-n:])]
assert distance[0][0] < distance[0][1] and distance[1][1] < distance[1][0], distance
report['anchor_endpoint_mae'] = distance

if args.http:
    sys.path.insert(0, str(ROOT / 'scripts'))
    from server_client import Client
    state = args.campaign.resolve() / 'state'
    with contextlib.closing(sqlite3.connect(state / 'queue.sqlite3')) as db:
        assert db.execute("SELECT count(*) FROM variants WHERE status IN ('queued','running')").fetchone()[0] == 0
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    client = Client(f'http://127.0.0.1:{port}')
    with (args.campaign / 'download-audit.log').open('w') as log:
        server = subprocess.Popen([str(ROOT / 'bin/h3cli'), '--server', '--server-port', str(port),
                                   '--server-state-dir', str(state)], cwd=ROOT, stdout=log, stderr=log)
        try:
            for _ in range(100):
                assert server.poll() is None
                try:
                    client.request('GET', '/health'); break
                except OSError:
                    time.sleep(.1)
            else:
                raise TimeoutError('local service startup')
            report['repeated_downloads'] = []
            for name, suffix in (('M01a', '.mp4'), ('M01a', '.h3av.h3bundle'), ('M09-still', '.png')):
                job = client.wait(cases[name]['id'])
                artifact = next(a for a in job['h3']['artifacts'] if a['name'].endswith(suffix))
                target = args.campaign / ('repeat-download' + suffix)
                for _ in range(2):
                    client.download(artifact['url'], target)
                    assert hashlib.sha256(target.read_bytes()).hexdigest() == artifact['sha256']
                report['repeated_downloads'].append({'case': name, 'name': artifact['name'], 'sha256': artifact['sha256']})
        finally:
            server.terminate(); server.wait(timeout=20)
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', port)) != 0
    with contextlib.closing(sqlite3.connect(state / 'queue.sqlite3')) as db:
        report['remaining_leases'] = db.execute('SELECT count(*) FROM leases').fetchone()[0]
        assert report['remaining_leases'] == 0
    report['service_reaped_and_port_closed'] = True

(args.campaign / 'artifact-validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(f"Validated {len(report['native_loaders'])} native states and {len(report['media'])} media files")
print('Upscale audio, still decode and same-build CLI/server bytes match')
print('Non-completed cases:', report['skipped_cases'])
