#!/usr/bin/env python3
"""Audit M2 records and compare matched first-evaluation M1/M2 captures offline.

This tool cannot qualify perceptual quality. It preserves legacy numerical
failures and distinguishes matched first-evaluation arithmetic from later free
trajectory divergence. No rendering or additional denoising is performed.
"""
import argparse
import json
from pathlib import Path
import subprocess
from metal_native_bench import ROOT, sha
from metal_fp16_model_validate import performance_gate, region_gate


def load_run(path):
    record = json.loads((path/'record.json').read_text())
    if not record['evaluation_contract_pass'] or sha(path/'result.h3av') != record['av_sha256']:
        raise ValueError('Incomplete or changed run: '+str(path))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflows', type=Path, required=True)
    parser.add_argument('--m1', type=Path, default=ROOT/'outputs/metal-native-m1-243')
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        parser.error('use a new output file to preserve prior evidence')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    index = json.loads((a.workflows/'index.json').read_text())
    records = {name: load_run(a.workflows/name) for name in
               ('reference-B1', 'fp16-B1', 'reference-B5', 'fp16-B5', 'reference-profile', 'fp16-profile')}
    old = {name: load_run(a.m1/name) for name in ('reference-b1', 'dense-a-b1', 'dense-a-profile')}
    # Fences/capture I/O must leave the first-evaluation state unchanged. Match
    # conditioning and schedule identity too; equal geometry alone is insufficient.
    for left, right in [(records['reference-B1'], old['reference-b1']),
                        (records['reference-B1'], records['reference-profile']),
                        (records['fp16-B1'], records['fp16-profile']),
                        (old['dense-a-b1'], old['dense-a-profile'])]:
        for key in ('av_sha256', 'conditioning_sha256_after', 'conditioning_identity',
                    'geometry', 'seed', 'prompt', 'evaluations', 'blocks', 'sampler', 'weight_precision'):
            if left[key] != right[key]:
                raise ValueError('Capture/baseline equivalence failed: '+key)
    for r in (*records.values(), *old.values()):
        for key in ('conditioning_sha256_after', 'conditioning_identity', 'geometry', 'seed',
                    'prompt', 'sampler', 'weight_precision'):
            if r[key] != records['reference-B1'][key]:
                raise ValueError('Unmatched workload: '+key)
    tool = ROOT/'bin/metal_tensor_metrics'
    tool.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['cc', '-O3', str(ROOT/'tests/metal_tensor_metrics.c'), '-lm', '-o', str(tool)], check=True)
    metrics = {}
    for name, directory in [('m1-dense-a', a.m1/'dense-a-profile'),
                            ('m2-fp16', a.workflows/'fp16-profile')]:
        rows = {}
        for tensor in ('video-velocity', 'audio-velocity', 'video-latent', 'audio-latent'):
            filename = 'step-001-'+tensor+'.f32'
            ref, candidate = a.workflows/'reference-profile'/'steps'/filename, directory/'steps'/filename
            if not ref.is_file() or not candidate.is_file() or ref.stat().st_size != candidate.stat().st_size:
                raise ValueError('Missing/mismatched capture: '+filename)
            command = [str(tool.resolve()), str(ref.resolve()), str(candidate.resolve()), 'f32']
            values = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
            rows[tensor] = {'metrics': values, 'reference_sha256': sha(ref),
                            'candidate_sha256': sha(candidate), 'command': command}
        metrics[name] = rows
    ranges = {}
    for name in ('fp16-B1', 'fp16-B5'):
        r = records[name]
        expected = {(s,b) for s in range(1,r['evaluations']+1) for b in range(50)}
        if len(r['mixed']) != len(expected) or {(x['step'],x['block']) for x in r['mixed']} != expected:
            raise ValueError('Incomplete range coverage: '+name)
        if any(x['heads'] != 56 or x['invalid_heads'] for x in r['mixed']):
            raise ValueError('Invalid range report: '+name)
        ranges[name] = {
            'blocks': len(r['mixed']), 'head_evaluations': len(r['mixed'])*56,
            'recovered_heads': sum(x['recovered_heads'] for x in r['mixed']),
            'operand_underflows': sum(x['underflows'] for x in r['mixed']),
            'invalid_heads': 0,
            'maximum': {k:max(x[k] for x in r['mixed']) for k in
                        ('q_max','k_max','v_max','output_max','score_error_bound')},
            'per_step_recovered_heads': {str(s):sum(x['recovered_heads'] for x in r['mixed'] if x['step']==s)
                                         for s in range(1,r['evaluations']+1)}}
    acceptance = performance_gate({k:v for k,v in records.items() if k.endswith(('B1','B5'))}, index['limits'])
    regions = region_gate(records)
    acceptance['pass'] &= regions['no_warm_block_regression']
    result = {
        'version': 1, 'workflows_index_sha256': sha(a.workflows/'index.json'),
        'diagnostic_script_sha256': sha(Path(__file__)),
        'gate_script_sha256': sha(ROOT/'scripts/metal_fp16_model_validate.py'),
        'tensor_metrics_source_sha256': sha(ROOT/'tests/metal_tensor_metrics.c'),
        'record_sha256': {name:sha(a.workflows/name/'record.json') for name in records},
        'm1_record_sha256': {name:sha(a.m1/name/'record.json') for name in old},
        'reference_plus_unchanged': True, 'diagnostic_capture_state_equivalence': True,
        'first_evaluation_metrics': metrics, 'range_coverage': ranges,
        'regions': regions, 'performance': acceptance, 'reference_qualified': False,
        'limitations': [
            'First-evaluation inputs/noise and outputs are matched. Later B5 trajectories evolve independently.',
            'These captures do not isolate later-noise, teacher-forced error or fully explain M1 B5 drift.',
            'Range telemetry covers post-normalization/RoPE QKV and attention output, not all raw QKV, score, softmax, MLP or residual values.',
            'Real reference/continuation QKV has standalone coverage; full mixed model Ref2VA/continuation qualification remains conditional.',
            'No calibrated perceptual contract, production-VAE B6, SOL/ANE composition or M5 qualification is claimed.']}
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'performance':acceptance, 'regions':regions, 'range_coverage':ranges}, indent=2))


if __name__ == '__main__':
    main()
