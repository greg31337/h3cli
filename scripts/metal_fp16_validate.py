#!/usr/bin/env python3
"""Serial M2A operator checks and adapter-inclusive 243-frame performance gate.

Snapshots sources and binaries before testing. A failed performance gate is
retained with exit 3; it does not authorize DiT/Reference integration.
"""
import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def timing_gate(result, limits):
    timings = result['balanced_seconds']
    if set(timings) != {'mpsgraph', 'fp16-32x16', 'fp16-64x32'} or any(
        len(values) != limits['balanced_rounds'] or
        any(not math.isfinite(v) or v <= 0 for v in values) for values in timings.values()
    ):
        raise ValueError('Incomplete or nonfinite balanced timing record')
    reference = timings['mpsgraph']
    rows = {}
    for name, values in timings.items():
        if name == 'mpsgraph':
            continue
        speedup = statistics.median(reference) / statistics.median(values)
        conservative = min(reference) / max(values)
        rows[name] = {'median_seconds': statistics.median(values),
                      'reference_median_seconds': statistics.median(reference),
                      'speedup': speedup, 'conservative_speedup': conservative,
                      'pass': (speedup >= limits['adapter_inclusive_median_speedup_min']
                               and conservative >= limits['conservative_min_reference_over_max_candidate_min'])}
    return rows


def performance_pass(performance, limits):
    if set(performance) != set(limits['required_fixtures']):
        return False
    return any(all(row[candidate]['pass'] for row in performance.values())
               for candidate in ('fp16-32x16', 'fp16-64x32'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, default=ROOT/'outputs/metal-native-m1-243')
    parser.add_argument('--operators-only', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error('use an empty output directory to preserve prior evidence')
    frozen = out/'frozen'
    frozen.mkdir()
    sources = {p for pattern in ['src/**/*.c', 'src/**/*.h', 'src/**/*.m', 'src/**/*.cu', 'src/**/*.cuh', 'src/**/*.metal', 'src/**/*host.inc',
                                 'tests/metal*', 'tests/test_metal*', 'scripts/*metal*.py', 'third_party/mlx-attention/**/*']
               for p in ROOT.glob(pattern) if p.is_file() and p.suffix not in ('.o', '.d')}
    sources.add(ROOT/'Makefile')
    source_hashes = {}
    for source in sorted(sources):
        relative = source.relative_to(ROOT)
        destination = frozen/'source'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[str(relative)] = sha(destination)
    binaries = ('bin/metal_fp16', 'bin/metal_attention', 'bin/metal_sol')
    # Compile the snapshot itself: a working-tree hash beside an unrelated or
    # stale executable is not build provenance. No model/GPU work runs here.
    build_command = ['make', '-j4', *binaries]
    with (out/'build.log').open('w') as log:
        subprocess.run(build_command, cwd=frozen/'source', stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    for name in binaries:
        shutil.copy2(frozen/'source'/name, frozen/name)
    limits = json.loads((ROOT/'tests/metal_fp16_limits_v2.json').read_text())
    recipe = {'version': 4, 'tier': 'Reference-candidate-unqualified',
              'weight_dtype': 'bf16-unchanged', 'state_dtype': 'bf16/fp32-unchanged',
              'interface_dtype': 'bf16', 'attention_storage': 'fp16-scaled-per-head',
              'half_layout': 'head-major-packed-by-conversion', 'recovery_layout': 'head-major-bit-exact-bf16-copy',
              'range_policy_version': 2,
              'qk_operands': 'fp16', 'pv_operands': 'fp16-probability/fp16-value',
              'score_accumulator': 'fp32', 'softmax_statistics': 'fp32',
              'output_accumulator': 'fp32', 'routing': 'dense', 'device_path': 'simdgroup',
              'math': 'Metal-3.1-safe/explicit-fast-exp2', 'range_recovery': 'bf16-storage/fp32-operands-per-head',
              'adapters': 'GPU-scan/pow2-convert/half+conditional-recovery/output-check/global-check/bf16-commit',
              'production_selectable': False, 'm5_nax_qualified': False}
    manifest = {'platform': platform.platform(), 'sources': source_hashes,
                'source_hash_scope': 'Compiled from this immutable source snapshot',
                'build_command': build_command,
                'compiler': subprocess.run(['clang', '--version'], capture_output=True, text=True, check=True).stdout,
                'recipe': recipe, 'limits': limits, 'limits_sha256': sha(ROOT/'tests/metal_fp16_limits_v2.json'),
                'hardware': subprocess.run(['system_profiler', 'SPDisplaysDataType', '-json'],
                                           capture_output=True, text=True, check=True).stdout,
                'records': [], 'performance': {}, 'performance_pass': False}
    def save():
        (out/'index.json').write_text(json.dumps(manifest, indent=2)+'\n')

    def run(name, binary, params, balanced=False, source=None):
        command = [str(frozen/binary), *map(str, params)]
        env = os.environ.copy()
        env['H3_TEST_MAX_EVALUATIONS'] = '6'
        env.pop('H3_TEST_ATTENTION_BALANCED', None)
        env.pop('H3_TEST_FP16_TILE_COUNT', None)
        if balanced:
            env['H3_TEST_ATTENTION_BALANCED'] = '1'
        start = time.monotonic()
        completed = subprocess.run(command, cwd=frozen/'source', env=env, capture_output=True, text=True)
        (out/(name+'.json')).write_text(completed.stdout)
        (out/(name+'.log')).write_text(completed.stderr)
        record = {'name': name, 'command': command, 'cwd': str(frozen/'source'),
                  'binary_sha256': sha(command[0]), 'input_sha256': sha(source) if source else None,
                  'returncode': completed.returncode, 'wall_seconds': time.monotonic()-start,
                  'balanced': balanced, 'result': None}
        try:
            record['result'] = json.loads(completed.stdout)
        except ValueError:
            pass
        manifest['records'].append(record)
        save()
        print(name, completed.returncode, flush=True)
        pass_key = 'correctness_pass' if binary == 'bin/metal_sol' else 'pass'
        if completed.returncode or not record['result'] or not record['result'].get(pass_key):
            raise SystemExit('Operator check failed: '+name)
        if balanced:
            manifest['performance'][name] = timing_gate(record['result'], limits['performance'])
            save()
        return record['result']

    for seq, pattern, il, ol in [
        (1, 'random', 0, 0), (31, 'random', 1, 1), (65, 'zero', 0, 1),
        (128, 'outlier', 1, 0), (129, 'constant', 0, 0), (257, 'random', 1, 1),
        (65, 'overflow', 1, 0), (65, 'underflow', 0, 1), (65, 'recovery', 1, 1), (65, 'bounded-loss', 0, 1),
        (65, 'reciprocal', 1, 0), (65, 'bf16-subnormal', 0, 1),
        (65, 'nonfinite', 0, 0), (65, 'score-overflow', 1, 1), (2049, 'random', 0, 0)]:
        run(f'{seq}-{pattern}-{il}-{ol}', 'bin/metal_fp16', [seq, 1, pattern, il, ol])
    # Existing template instantiations must survive the generic operand change.
    run('bf16-regression', 'bin/metal_attention', [129, 1, 'outlier', 1, 0])
    run('sol-regression', 'bin/metal_sol', [257, 56, 32, 64, .1, 1, 'outlier'])
    if not args.operators_only:
        run('synthetic-22426', 'bin/metal_fp16', [22426, 1, 'random', 0, 0], True)
        fixtures = args.fixtures.resolve()
        for name, relative in [
            ('block0', 'reference-profile/qkv.bf16'),
            ('block24', 'reference-profile/qkv.bf16.block24'),
            ('block49', 'reference-profile/qkv.bf16.block49'),
            ('references-243', 'scaling/references/qkv/qkv.bf16'),
            ('continuation192-243', 'scaling/continue192/qkv-full/qkv.bf16')]:
            path = fixtures/relative
            nbytes = path.stat().st_size
            if nbytes % (3*56*128*2):
                raise ValueError('Malformed QKV fixture: '+str(path))
            seq = nbytes//(3*56*128*2)
            if seq != (24250 if name == 'references-243' else 22426):
                raise ValueError('Fixture is not the required 243-frame shape')
            run(name, 'bin/metal_fp16', [seq, 1, 'file', 0, 0, path], True, path)
        # One fixed candidate must pass every fixture; no per-input cherry-picking.
        manifest['performance_pass'] = performance_pass(manifest['performance'], limits['performance'])
    manifest['operator_pass'] = True
    manifest['integration_authorized_by_gate'] = manifest['performance_pass']
    save()
    rows = []
    for fixture, values in manifest['performance'].items():
        for candidate, v in values.items():
            rows.append(f'<tr><td>{html.escape(fixture)}</td><td>{candidate}</td>'
                        f'<td>{v["reference_median_seconds"]:.6f}</td><td>{v["median_seconds"]:.6f}</td>'
                        f'<td>{v["speedup"]:.3f}×</td><td>{"PASS" if v["pass"] else "FAIL"}</td></tr>')
    (out/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>M2 FP16 results</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:3rem auto;background:#fafafa}'
        'td,th{padding:.5rem;text-align:left;border-bottom:1px solid #ddd}</style>'
        '<h1>M2 FP16 attention — 243-frame gate</h1>'
        '<p>Local M4 Max. Operator tests: PASS. This is an isolated benchmark, not a production-render qualification.</p>'
        f'<p>Adapter-inclusive 1.10× performance gate: {"PASS" if manifest["performance_pass"] else "NOT PASSED"}.</p>'
        '<p>All timings include GPU range scans, conversion, conditional recovery, output validation and BF16 commit. '
        'No candidate clips are presented: DiT integration and perceptual qualification require the performance gate.</p>'
        '<table><tr><th>Fixture</th><th>Candidate</th><th>MPSGraph seconds</th><th>Candidate seconds</th>'
        '<th>Speedup</th><th>Gate</th></tr>'+''.join(rows)+'</table><p><a href="index.json">Commands, hashes, ranges and raw timings</a></p>')
    if not args.operators_only and not manifest['performance_pass']:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
