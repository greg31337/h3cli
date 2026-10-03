#!/usr/bin/env python3
"""Check all production modes twice against pre-fix deterministic Qwen captures.

Requires presentations and encoder captures from scalingfix_prepare/encoder.
Legacy must match the previously synchronized legacy fixture, while both other
modes must preserve their outputs exactly. No production instrumentation is used.
"""
import json
import os
import subprocess

from memory_generation import digest
from scalingfix_prepare import FLAGS, OUT, ROOT


def main():
    out = OUT / 'legacy-sync'
    out.mkdir(exist_ok=True)
    binary = ROOT / 'bin/scalingfix_legacy_sync'
    binary.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([
        'clang', '-O3', '-std=c11', '-D_DARWIN_C_SOURCE', '-I', str(ROOT),
        str(ROOT / 'tests/scalingfix_qwen.c'), str(ROOT / 'bin/libh3.a'),
        *FLAGS, '-o', str(binary),
    ], check=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('H3_')}
    records = {}
    for case in ('plain', 'dialogue', 'image', 'video'):
        fixture = OUT / 'presentations' / case
        weights = ROOT / 'models/MiniMax-H3' / (
            'Ref2VA' if case in ('image', 'video') else 'FL2VA'
        ) / 'text_encoder'
        for mode in ('legacy', 'scaled-q', 'reference'):
            expected_mode = 'legacy-synchronized' if mode == 'legacy' else mode
            expected = OUT / 'encoder' / case / expected_mode / 'layer-50.bf16'
            expected_hash = digest(expected)
            runs = []
            for repeat in range(2):
                dest = out / f'{case}-{mode}-{repeat}'
                dest.mkdir(exist_ok=True)
                result = subprocess.run(
                    [str(binary), str(weights), str(fixture), str(dest)],
                    cwd=ROOT, env={**env, 'H3_QWEN_GQA_SCALE_MODE': mode},
                    capture_output=True, text=True, check=True,
                )
                (dest / 'run.log').write_text(result.stderr + result.stdout)
                actual = digest(dest / 'final.bf16')
                assert actual == expected_hash, (case, mode, repeat, actual, expected_hash)
                runs.append({'sha256': actual, 'stats': json.loads(result.stdout)})
            records[f'{case}-{mode}'] = {
                'expected': str(expected.relative_to(ROOT)),
                'expected_sha256': expected_hash, 'exact': True, 'runs': runs,
            }
            print(f'PASS {case} {mode}: both runs match {expected_mode} exactly', flush=True)
    report = {
        'binary_sha256': digest(binary),
        'shader_sha256': digest(ROOT / 'src/metal/shaders.metal'),
        'cases': records,
    }
    (out / 'encoder-results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
