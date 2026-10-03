#!/usr/bin/env python3
"""Independent functional image-sizing campaign; no numerical parity oracle."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import time
import zlib

MANIFEST = Path(__file__).with_name('reference_image_matrix.json')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(root, manifest):
    root.mkdir(parents=True, exist_ok=False)
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    identities = {}
    for name, spec in manifest['fixtures'].items():
        w, h, variant = spec
        raw = bytearray()
        for y in range(h):
            raw.append(0)
            for x in range(w):
                raw.extend(((x * 191 // w + variant * 31) % 256,
                            (y * 193 // h + variant * 67) % 256,
                            ((x // 40 + y // 40 + variant) % 2) * 127 + 40))
        path = root / name
        path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) +
                         chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))
        identities[name] = digest(path)
    write(root / 'identities.json', identities)


def finite(path, still):
    with path.open('rb') as f:
        if still:
            header = json.loads(f.read(struct.unpack('<Q', f.read(8))[0]))
            tensors = [v for k, v in header.items() if k != '__metadata__']
            assert tensors and all(t['dtype'] == 'F32' for t in tensors), header
        else:
            header = f.read(160)
            assert header[:8] == b'H3AV\r\n\x1a\n'
        values = array.array('f')
        values.frombytes(f.read())
        if sys.byteorder != 'little':
            values.byteswap()
        assert values and all(math.isfinite(v) for v in values), 'nonfinite/empty saved latent'
        return len(values)


def run(a, manifest, case):
    out = a.out / case['id']
    out.mkdir(parents=True, exist_ok=False)
    still = case['output'] == 'still'
    media = out / ('image.png' if still else 'video.mp4')
    latent = out / ('latent.safetensors' if still else 'latent.h3av')
    cmd = [str(a.source / 'bin/h3cli'), '-d', str(a.model), '-p', manifest['prompt'],
           '--width', '640', '--height', '480', '--steps', '2', '--seed', '42',
           '--layers', '50', '--reuse', '1', '--core-reuse', '1', '--profile',
           '--ref-image-size', case['mode'], '-o', str(media)]
    for ref in case['references']:
        cmd += ['--ref-image', str(a.fixtures / ref)]
    if still:
        cmd += ['--still', '--image-vae', str(a.image_vae), '--save-still-latent', str(latent)]
    else:
        # Existing metadata identity path avoids scanning/hashing base weights.
        cmd += ['--frames', '90', '--save-av-state', str(latent), '--save-upscale-state', str(out / 'source.h3up')]
    env = os.environ.copy()
    # The existing safety override accepts 6 or 50. Each request still asks
    # for exactly the manifest's two evaluations.
    env['H3_TEST_MAX_EVALUATIONS'] = '6'
    record = dict(case=case, backend=a.backend, argv=cmd, manifest_sha256=digest(MANIFEST),
                  binary_sha256=digest(a.source / 'bin/h3cli'), timeout_seconds=manifest['timeout_seconds'],
                  fixture_sha256={n: digest(a.fixtures / n) for n in case['references']}, passed=False)
    write(out / 'result.json', record)
    start = time.monotonic()
    samples = []
    try:
        with (out / 'render.log').open('x') as log:
            p = subprocess.Popen(cmd, cwd=a.source, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while p.poll() is None:
                    rss = subprocess.run(['ps', '-o', 'rss=', '-p', str(p.pid)], capture_output=True, text=True)
                    sample = dict(seconds=time.monotonic() - start, rss_bytes=int(rss.stdout.strip() or '0') * 1024)
                    if a.backend == 'cuda':
                        gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True)
                        sample['gpu_bytes'] = sum(int(line.split(',')[1]) * 1048576 for line in gpu.stdout.splitlines() if line.split(',')[0].strip() == str(p.pid))
                    samples.append(sample)
                    if time.monotonic() - start > manifest['timeout_seconds']:
                        raise TimeoutError('functional row exceeded its deadline')
                    time.sleep(1)
                record['returncode'] = p.returncode
                assert p.returncode == 0, f'render failed: {p.returncode}'
            finally:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
                    try:
                        p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGKILL)
                        p.wait()
                record['process_reaped'] = p.poll() is not None
        text = (out / 'render.log').read_text()
        geometry = re.findall(r'reference image (\d+) size=(\w+) source=(\d+)x(\d+) canvas=(\d+)x(\d+) patches=(\d+) image-batch-patches=(\d+)', text)
        total = 0
        assert len(geometry) == len(case['references']), geometry
        for i, (ref, expected, actual) in enumerate(zip(case['references'], case['canvases'], geometry)):
            sw, sh, _ = manifest['fixtures'][ref]
            w, h = expected
            patches = w * h // 256
            total += patches
            assert actual == tuple(map(str, [i + 1, case['mode'], sw, sh, w, h, patches, total])), actual
        record['geometry'] = geometry
        record['phase_seconds'] = [(name, float(seconds)) for name, seconds in
                                   re.findall(r'phase duration ([^\n]+): ([0-9.]+) s', text)]
        memory = [json.loads(line[len('h3_memory '):]) for line in text.splitlines() if line.startswith('h3_memory ')]
        record['memory_boundaries'] = memory
        record['peak_sampled_footprint_bytes'] = max((m['footprint_bytes'] for m in memory), default=0)
        record['peak_sampled_metal_bytes'] = max((m['metal_allocated_bytes'] for m in memory), default=0)
        record['finite_latent_values'] = finite(latent, still)
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', str(media)], env=env))
        video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        assert (int(video['width']), int(video['height']), int(video['nb_read_frames'])) == (640, 480, 1 if still else 90), video
        if not still:
            assert video['r_frame_rate'] == '24/1', video
            assert any(s['codec_type'] == 'audio' for s in probe['streams'])
        subprocess.run([env.get('H3_FFMPEG', 'ffmpeg'), '-v', 'error', '-xerror', '-i', str(media), '-f', 'null', '-'], env=env, check=True, timeout=120)
        record.update(passed=True, media=probe, media_sha256=digest(media))
    except Exception as exc:
        record['error'] = repr(exc)
        raise
    finally:
        record.update(seconds=time.monotonic() - start, peak_rss_bytes=max((s['rss_bytes'] for s in samples), default=0),
                      peak_gpu_bytes=max((s.get('gpu_bytes', 0) for s in samples), default=0))
        write(out / 'resources.json', samples)
        write(out / 'result.json', record)
        print(json.dumps({k: record[k] for k in ['backend', 'passed', 'seconds', 'peak_rss_bytes', 'peak_gpu_bytes']}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--source', type=Path, default=Path('.'))
    p.add_argument('--model', type=Path)
    p.add_argument('--image-vae', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--backend', choices=['metal', 'cuda'])
    p.add_argument('--case', action='append', help='Independent functional row, never a golden-gate selection')
    a = p.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    a.fixtures = a.fixtures.resolve()
    if a.prepare:
        prepare(a.fixtures, manifest)
        return
    assert a.out and a.model and a.image_vae and a.backend
    for key in ['source', 'model', 'image_vae', 'out']:
        setattr(a, key, getattr(a, key).resolve())
    identities = json.loads((a.fixtures / 'identities.json').read_text())
    assert all(digest(a.fixtures / name) == value for name, value in identities.items())
    a.out.mkdir(parents=True, exist_ok=False)
    for case in manifest['cases']:
        if not a.case or case['id'] in a.case:
            run(a, manifest, case)


if __name__ == '__main__':
    main()
