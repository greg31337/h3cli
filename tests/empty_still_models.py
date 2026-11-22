#!/usr/bin/env python3
"""Offline real-model still qualification with only the 53 planned model files.

Run against a source binary or a standalone package. Existing weights stay
read-only; each fresh output directory contains owned symlinks and run records.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'bin/h3cli')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    binary, model, out = args.binary.resolve(), args.model.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    catalog = json.loads((ROOT / 'src/models/catalog.json').read_text())['artifacts']
    env = dict(os.environ, H3_OFFLINE='1', H3_PROFILE='1', H3_VERBOSE='1')
    env.pop('H3_TEST_MAX_EVALUATIONS', None)
    records = []

    def run(name, flags, good=True):
        start = time.monotonic()
        with (out / (name + '.log')).open('w') as log:
            result = subprocess.run([str(binary), '--offline', *map(str, flags)],
                                    cwd=out, env=env, stdout=log, stderr=log)
        record = dict(name=name, returncode=result.returncode,
                      seconds=time.monotonic() - start,
                      passed=(result.returncode == 0) == good)
        records.append(record)
        (out / 'result.json').write_text(json.dumps(dict(passed=False, checks=records), indent=2) + '\n')
        assert record['passed'], (name, out / (name + '.log'))
        return (out / (name + '.log')).read_text()

    for variant in ('FL2VA', 'Ref2VA'):
        models_root = out / variant / 'models'
        minimal = models_root / 'MiniMaxH3'
        files = [item['path'] for item in catalog if item['path'].startswith(variant + '/')
                 and '/audio_vae/' not in item['path']]
        assert len(files) == 53, (variant, len(files))
        for name in files:
            source, target = model / name, minimal / name
            assert source.is_file(), source
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)
        assert not (minimal / variant / 'audio_vae').exists()
        # Exercise default model discovery and an explicit model directory.
        location = ['--models-path', models_root] if variant == 'FL2VA' else ['-d', minimal]
        image, latent = out / (variant + '.png'), out / (variant + '.safetensors')
        refs = [] if variant == 'FL2VA' else ['--ref-image', ROOT / 'tests/fixtures/cuda-reference/portrait.jpg']
        run(variant + '-generate', [*location, '--still', '--frames', '1', '--width', '256',
            '--height', '256', '--steps', '2', '--seed', '42', *refs, '-p',
            'A red ceramic teapot on a wooden table.' if not refs else
            'A photograph of the person in image 1 wearing a blue jacket in a garden.',
            '--save-still-latent', latent, '-o', image])
        data = image.read_bytes()
        assert data[:8] == b'\x89PNG\r\n\x1a\n' and struct.unpack('>II', data[16:24]) == (256, 256)
        codec_root = out / variant / 'codec'
        for name in ('config.json', 'source/config.json', 'source/model.safetensors'):
            target = codec_root / 'MiniMaxH3' / variant / 'video_vae' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(model / variant / 'video_vae' / name)
        decoded = out / (variant + '-decoded.png')
        run(variant + '-decode', ['--models-path', codec_root, '--decode-still-latent', latent, '-o', decoded])
        assert decoded.read_bytes() == data, variant + ' codec-only PNG mismatch'
        run(variant + '-video-missing-audio', [*location, '--width', '256', '--height', '256',
            '--frames', '22', '--steps', '2', *refs, '-p', 'A teapot.', '-o', out / 'missing-audio.mp4'], good=False)
        assert not (out / 'missing-audio.mp4').exists()
        # An unrelated incomplete audio download must not break still startup.
        audio = minimal / variant / 'audio_vae'
        audio.mkdir()
        (audio / 'model.safetensors').write_bytes(b'incomplete audio download')
        run(variant + '-partial-audio-info', ['-d', minimal, '--info'])
        config = codec_root / 'MiniMaxH3' / variant / 'video_vae/config.json'
        config.unlink()  # Owned symlink only; installed weights remain untouched.
        run(variant + '-missing-vae-config', ['--models-path', codec_root,
            '--decode-still-latent', latent, '-o', out / 'missing-vae.png'], good=False)
        assert not (out / 'missing-vae.png').exists()
    result = dict(passed=True, binary_sha256=sha(binary),
                  model_files_per_variant=53, codec_files_per_variant=3, checks=records)
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: both variants generate without audio/image VAEs; minimal codec PNGs match; missing dependencies fail.')


if __name__ == '__main__':
    main()
