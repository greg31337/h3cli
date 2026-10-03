#!/usr/bin/env python3
"""Real Qwen/DiT/VAE tensor parity, official mapping audit, and storage benchmark."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/bugfix1-validation'
PAGE = os.sysconf('SC_PAGESIZE')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def header(path):
    with path.open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return n + 8, json.loads(f.read(n))


def main():
    env = {k: v for k, v in os.environ.items() if not k.startswith('H3_')}
    env.update(H3_ZERO_COPY_WEIGHTS='1', H3_PROFILE='1')
    records, audit = {}, {}
    for component in ['text_encoder', 'transformer', 'video_vae/source', 'audio_vae']:
        base = ROOT / 'models/MiniMax-H3/FL2VA' / component
        entries = []
        for path in sorted(base.glob('*.safetensors')):
            data, tensors = header(path)
            subprocess.run([str(ROOT / 'bin/bugfix1_probe'), 'header', str(path)], check=True, stdout=subprocess.DEVNULL)
            for name, value in tensors.items():
                if name == '__metadata__':
                    continue
                start, end = value['data_offsets']
                entries.append((path, name, value, data + start, end-start))
        eligible = [e for e in entries if e[3] % PAGE == 0 and e[4] > 0]
        audit[component] = dict(tensors=len(entries), page_aligned_tensors=len(eligible),
                                page_aligned_bytes=sum(e[4] for e in eligible))
        # Verify every eligible released tensor, not just the benchmark sample.
        verified = []
        for path, name, desc, offset, size in eligible:
            raw_out = OUT / 'eligible-readback.raw'
            cmd = [str(ROOT / 'bin/bugfix1_probe'), 'load', str(path), name, '1', str(raw_out)]
            result = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True, check=True)
            with path.open('rb') as source_file:
                source_file.seek(offset)
                expected_hash = hashlib.sha256(source_file.read(size)).hexdigest()
            assert sha(raw_out) == expected_hash
            assert 'released ' in result.stderr
            verified.append(dict(tensor=name, bytes=size, mapped=True, sha256=expected_hash))
            raw_out.unlink()
        audit[component]['verified_eligible_tensors'] = verified
        # Representative actual weight; Qwen also proves released eligible tensors stay mapped.
        selected = max(eligible or [e for e in entries if 8*1024*1024 <= e[4] <= 256*1024*1024], key=lambda e: e[4])
        source, name, desc, offset, size = selected
        directory = OUT / 'weights' / component
        directory.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as f:
            f.seek(offset)
            payload = f.read(size)
        expected = hashlib.sha256(payload).hexdigest()
        generated = []
        for variant, extra in [('mapped', 0), ('copied', 1)]:
            d = directory / variant
            d.mkdir(exist_ok=True)
            path = d / 'model.safetensors'
            raw = json.dumps({name: dict(desc, data_offsets=[0, size])}, separators=(',', ':')).encode()
            data_start = ((len(raw) + 8 + PAGE - 1) // PAGE) * PAGE + extra
            raw += b' ' * (data_start-8-len(raw))
            path.write_bytes(struct.pack('<Q', len(raw)) + raw + payload)
            generated.append((variant, path, int(not extra)))
        unaligned = OUT / 'unaligned-model/FL2VA' / source.relative_to(ROOT / 'models/MiniMax-H3/FL2VA')
        cases = [('official', source, int(offset % PAGE == 0)), ('unaligned_model', unaligned, 0)] + generated
        runs = {}
        for variant, path, mapped in cases:
            measurements = []
            for i in range(3 if variant in ('mapped', 'copied') else 1):
                raw_out = directory / 'readback.raw'
                cmd = [str(ROOT / 'bin/bugfix1_probe'), 'weight', str(path), name, str(mapped), str(raw_out)]
                with (directory / f'{variant}-{i}.log').open('w') as log:
                    r = subprocess.run(cmd, env=env, cwd=ROOT, stdout=subprocess.PIPE, stderr=log, text=True, check=True)
                measure = json.loads(r.stdout)
                assert sha(raw_out) == expected
                measurements.append(measure)
                raw_out.unlink()
            runs[variant] = dict(measurements=measurements, median_load_seconds=statistics.median(m['load_seconds'] for m in measurements))
        records[component] = dict(tensor=name, source=str(source), source_offset=offset, bytes=size, sha256=expected, runs=runs)
        (OUT / 'weights.json').write_text(json.dumps(dict(audit=audit, parity=records), indent=2)+'\n')
        print('PASS', component, name, size, flush=True)


if __name__ == '__main__':
    main()
