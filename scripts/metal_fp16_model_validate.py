#!/usr/bin/env python3
"""Run gated M2B 243-frame B1/B5 and diagnostic region/velocity captures.

Legacy BF16 trajectory errors are retained separately from implementation and
performance gates. No B6/Reference promotion is opened by this tool alone.
"""
import argparse
import html
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from metal_native_bench import ROOT, sha


def performance_gate(records, limits):
    """Acceptance uses complete, unfenced runs, never their cached statistics."""
    timings = {}
    for name in ('reference-B1', 'fp16-B1', 'reference-B5', 'fp16-B5'):
        r = records[name]
        count = 1 if name.endswith('B1') else 5
        if (not r['evaluation_contract_pass'] or r['geometry'] != limits['geometry'] or
            r['evaluations'] != count or r['blocks'] != 50 or
            r['weight_precision'] != 'bf16' or
            any(r.get(k) for k in ('component_fences', 'region_fences', 'capture_qkv',
                                   'capture_boundaries', 'capture_steps', 'capture_ranges', 'teacher_from', 'trace_command')) or
            r['manifest']['environment'].get('H3_DIT_COMMAND_BLOCKS') != str(limits['command_blocks'])):
            raise ValueError('Incomplete, incompatible or diagnostic acceptance run: '+name)
        steps = r['steps']
        if len(steps) != count or [s['step'] for s in steps] != list(range(1, count+1)):
            raise ValueError('Missing/duplicate evaluations: '+name)
        for s in steps:
            if not math.isfinite(s['wall_seconds']) or s['wall_seconds'] <= 0:
                raise ValueError('Invalid timing: '+name)
            if not isinstance(s['metal_peak_bytes'], int) or s['metal_peak_bytes'] <= 0:
                raise ValueError('Missing memory measurement: '+name)
        timings[name] = [s['wall_seconds'] for s in steps]
    reference, candidate = timings['reference-B5'][1:], timings['fp16-B5'][1:]
    speedup = statistics.median(reference)/statistics.median(candidate)
    conservative = min(reference)/max(candidate)
    memory_pass = all(s['metal_peak_bytes'] <= limits['maximum_tracked_metal_bytes']
                      for r in records.values() for s in r['steps'])
    return {'steady_speedup': speedup, 'conservative_speedup': conservative,
            'memory_pass': memory_pass,
            'pass': speedup >= limits['minimum_b5_steady_speedup'] and
                    conservative >= limits['minimum_nonoverlap_speedup'] and memory_pass}


def region_gate(records):
    """Matched diagnostic fences provide attribution, not acceptance timings."""
    names = ('DiT complete attention region', 'DiT complete block')
    rows = {}
    for name in ('reference-profile', 'fp16-profile'):
        row = {}
        r = records[name]
        if not r['evaluation_contract_pass'] or not r['region_fences']:
            raise ValueError('Missing diagnostic region run: '+name)
        for component in names:
            values = [v for v in r['components'] if v['name'] == component and v['block'] > 0]
            if (len(values) != 49 or {v['block'] for v in values} != set(range(1,50)) or
                any(not math.isfinite(v['wall_seconds']) or v['wall_seconds'] <= 0 for v in values)):
                raise ValueError('Incomplete/invalid warm region coverage: '+name+'/'+component)
            row[component] = {'count': len(values), 'sum_seconds': sum(v['wall_seconds'] for v in values)}
        rows[name] = row
    ratios = {k: rows['reference-profile'][k]['sum_seconds']/rows['fp16-profile'][k]['sum_seconds']
              for k in names}
    return {'warm_regions_excluding_block0': rows, 'speedups': ratios,
            'no_warm_block_regression': ratios['DiT complete block'] >= 1.0,
            'timing_comparable_to_unfenced': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--conditioning', type=Path, required=True)
    p.add_argument('--standalone', type=Path, required=True)
    p.add_argument('--block-proof', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.binary = a.binary.resolve(); a.conditioning = a.conditioning.resolve()
    a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    if any(a.output.iterdir()):
        p.error('use an empty output directory')
    standalone = json.loads(a.standalone.read_text())
    if not standalone.get('operator_pass') or not standalone.get('performance_pass'):
        p.error('T139 standalone operator/performance gate has not passed')
    provenance = json.loads((a.binary.parent/'build-provenance.json').read_text())
    if sha(a.binary) != provenance['binary_sha256']:
        p.error('stale integrated binary provenance')
    for name in ('src/metal/fp16_attention.metal', 'src/metal/routed_attention.metal', 'scripts/embed_metal_attention.py'):
        if provenance['source_sha256'][name] != standalone['sources'][name]:
            p.error('attention shader differs from qualified standalone: '+name)
    proof = json.loads(a.block_proof.read_text())
    if not proof['pass']:
        p.error('one-block numerical gate has not passed')
    for key in ('reference', 'candidate'):
        directory = (ROOT/proof[key]).resolve()
        record = json.loads((directory.parent/'record.json').read_text())
        if record['manifest']['binary_sha256'] != sha(a.binary) or not record['evaluation_contract_pass']:
            p.error('block proof has stale binary or incomplete evaluation')
        for tensor in proof['tensors']:
            if sha(directory/tensor['tensor']) != tensor[key+'_sha256']:
                p.error('block proof tensor changed')
    limits_path = ROOT/'tests/metal_fp16_model_limits.json'
    limits = json.loads(limits_path.read_text())
    ledger = {'limits': limits, 'limits_sha256': sha(limits_path),
              'driver_sha256': sha(Path(__file__)),
              'standalone_sha256': sha(a.standalone), 'block_proof_sha256': sha(a.block_proof),
              'binary_sha256': sha(a.binary), 'jobs': [], 'comparisons': {}}
    def save():
        (a.output/'index.json').write_text(json.dumps(ledger, indent=2)+'\n')
    def run(name, case, native, diagnostic=False):
        command = [sys.executable, str(ROOT/'scripts/metal_native_bench.py'),
                   '--binary', str(a.binary), '--conditioning', str(a.conditioning),
                   '--output', str(a.output/name), '--case', case]
        if native:
            command += ['--backend', 'metal', '--metal-attention-kernel', 'steel-routed',
                        '--metal-attention-dtype', 'fp16', '--metal-tier', 'reference']
        if diagnostic:
            command += ['--regions', '--capture-steps']
        print('Running', name, flush=True)
        with (a.output/(name+'.driver.log')).open('w') as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        row = {'name': name, 'command': command, 'returncode': completed.returncode}
        ledger['jobs'].append(row); save()
        if completed.returncode:
            raise SystemExit('Model execution failed: '+name)
        return json.loads((a.output/name/'record.json').read_text())
    # Opposite ordering for B1/B5. A passing B5 would still require repeatability
    # and the separate T153/T152 quality corpus; failures remain recorded.
    records = {}
    for name, case, native in [('reference-B1', 'B1', False), ('fp16-B1', 'B1', True),
                               ('fp16-B5', 'B5', True), ('reference-B5', 'B5', False)]:
        records[name] = run(name, case, native)
    for case in ('B1', 'B5'):
        output = a.output/(case+'-legacy-numerics.json')
        command = [sys.executable, str(ROOT/'scripts/metal_native_compare.py'),
                   str(a.output/('reference-'+case)), str(a.output/('fp16-'+case)),
                   '--max-relative-l2', '.02', '--min-cosine', '.999', '--output', str(output)]
        with (a.output/(case+'-comparison.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        # The comparer writes only after finite/identity/coverage checks. A
        # nonzero exit with retained finite metrics is an old numerical failure,
        # not an instruction to relabel the new perceptual tier as failed/passed.
        result = json.loads(output.read_text())
        ledger['comparisons'][case] = result; save()
    ledger['performance'] = performance_gate(records, limits)
    speedup = ledger['performance']['steady_speedup']
    save()
    # Attribution and matched first-evaluation velocities are always diagnostic.
    # Their fences and file I/O cannot improve an acceptance timing above.
    for name, native in [('reference-profile', False), ('fp16-profile', True)]:
        records[name] = run(name, 'B1', native, True)
    ledger['regions'] = region_gate(records)
    ledger['performance']['pass'] &= ledger['regions']['no_warm_block_regression']
    ledger['reference_qualified'] = False
    ledger['perceptual_qualification'] = 'not performed; requires calibrated T153 contract and T142'
    save()
    parts = ['<!doctype html><meta charset="utf-8"><title>M2 integrated comparison</title>',
             '<style>body{font:16px system-ui;background:#15171c;color:#eee;max-width:1300px;margin:2rem auto}'
             '.pair{display:grid;grid-template-columns:1fr 1fr;gap:1rem}video{width:100%}a{color:#9cd3ff}'
             'td,th{padding:.5rem;text-align:left}code{color:#aed}</style>',
             '<h1>M2 FP16 attention</h1><p>640×480, 243 frames, original BF16 weights, CPU Euler, '
             '50 blocks. These clips use the preview video decoder. They do not establish production-VAE or 50-step Reference quality.</p>',
             f'<p>B5 steady speedup: <b>{speedup:.3f}×</b>; 1.10× gate: '
             f'<b>{"PASS (quality still unqualified)" if ledger["performance"]["pass"] else "FAIL"}</b>.</p>',
             '<p>Original strict BF16 numerical screens are diagnostics. Perceptual qualification has not been performed. '
             'No human-review approval is required by this page.</p>']
    for case in ('B1', 'B5'):
        parts.append(f'<h2>{case}</h2><div class="pair">')
        for prefix, label in [('reference', 'MPSGraph BF16'), ('fp16', 'Native FP16')]:
            path = prefix+'-'+case
            parts.append(f'<div><p>{label}</p><video controls preload="metadata" src="{path}/result.mp4"></video>'
                         f'<p><a href="{path}/record.json">Run record</a></p></div>')
        parts.append('</div><pre>'+html.escape(json.dumps(ledger['comparisons'][case]['metrics'], indent=2))+'</pre>')
    parts.append('<p><a href="index.json">All commands, metrics and performance disposition</a></p>')
    (a.output/'review.html').write_text('\n'.join(parts)+'\n')
    print(json.dumps(ledger['performance'], indent=2), flush=True)
    if not ledger['performance']['pass']:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
