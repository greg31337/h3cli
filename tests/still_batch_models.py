#!/usr/bin/env python3
"""Bounded real-model still batches: exact independent-seed parity and cache reuse."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=ROOT / 'bin/h3cli')
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, H3_OFFLINE='1', H3_TEST_MAX_EVALUATIONS='6')
    common = [str(args.binary.resolve()), '--offline', '-d', str(args.model.resolve()),
              '--still', '--width', '256', '--height', '256', '--steps', '2', '--verbose']
    report = {'passed': False, 'binary_sha256': digest(args.binary), 'runs': []}

    def save():
        (args.out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    for name, prompt, references in (
        ('text', 'A red ceramic teapot on a pale wooden table.', []),
        ('reference', 'A still portrait of the person in Picture 1 in soft daylight.',
         ['--ref-image', str(ROOT / 'tests/fixtures/cuda-reference/portrait.jpg'),
          '--ref-image-size', 'match']),
    ):
        folder = args.out / name
        folder.mkdir()
        commands = [
            ('batch', ['--still-count', '2', '--seed', '42',
                       '--frames-dir', str(folder / 'frames')]),
            ('single-42', ['--still-count', '1', '--seed', '42']),
            ('single-43', ['--seed', '43']),
        ]
        for label, flags in commands:
            command = [*common, '-p', prompt, *references, *flags,
                       '--save-still-latent', str(folder / (label + '.safetensors')),
                       '-o', str(folder / (label + '.png'))]
            started = time.monotonic()
            with (folder / (label + '.log')).open('w') as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            report['runs'].append({'case': name, 'label': label, 'command': command,
                                   'seconds': time.monotonic() - started, 'returncode': result.returncode})
            save()
            if result.returncode:
                raise RuntimeError(f'{name}/{label} failed; see its log')
            print(f'{name}/{label}: completed', flush=True)
        hashes = {}
        for seed in (42, 43):
            for extension in ('png', 'safetensors'):
                batched = folder / f'batch-seed-{seed}.{extension}'
                independent = folder / f'single-{seed}.{extension}'
                assert digest(batched) == digest(independent), (name, seed, extension)
                hashes[f'{seed}.{extension}'] = digest(batched)
            png = (folder / f'batch-seed-{seed}.png').read_bytes()
            assert png[:8] == b'\x89PNG\r\n\x1a\n'
            assert struct.unpack('>II', png[16:24]) == (256, 256)
        assert hashes['42.safetensors'] != hashes['43.safetensors']
        assert not (folder / 'batch.png').exists()
        assert sorted(p.name for p in (folder / 'frames').iterdir()) == ['frame-0000.ppm', 'frame-0001.ppm']
        log = (folder / 'batch.log').read_text()
        for message in ('conditioning cache hit', 'prepared DiT cache hit', 'stock still VAE cache hit'):
            assert log.count(message) == 1, (name, message)
        assert log.count('phase start stock still VAE load:') == 1
        report[name] = {'seed_outputs': hashes, 'independent_seed_parity': True, 'cache_hits': True}
        save()
        print(f'{name}: identical PNG/latent outputs for both independent seeds; cache reuse verified', flush=True)
    report['passed'] = True
    save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
