#!/usr/bin/env python3
"""Summarize recorded qualification evidence without turning missing tests into passes."""
import argparse
import json
import re
import statistics
from pathlib import Path
from attention_run import sha, profile
from attention_limits import measurements


def read(path, fallback=None):
    return json.loads(path.read_text()) if path.exists() else fallback


def status(passed, count, expected):
    return 'failed' if not passed else 'passed' if count == expected else 'incomplete'


def spread(values):
    return {'samples': values, 'mean': statistics.mean(values),
            'minimum': min(values), 'maximum': max(values),
            'sample_std': statistics.stdev(values) if len(values) > 1 else None}


def component(root):
    rows = read(root / 'results.json', [])
    baseline = {r['sequence']: r['mean_seconds'] for r in rows if r['route'] == 'cudnn'}
    return [{'sequence': r['sequence'], 'mode': r['route'] or r['mode'],
             'seconds': spread(r['samples_seconds']),
             'fraction_of_cudnn': r['mean_seconds'] / baseline[r['sequence']],
             'tracked_peak_device_bytes': r['peak_device_bytes'],
             'observed_process_peak_bytes': r['record']['gpu_telemetry']['peak_process_bytes'],
             'binary_sha256': r['record']['binary_sha256'],
             'concurrent_cuda_pids': r['record']['gpu_telemetry']['concurrent_cuda_pids'],
             'command': r['record']['argv']} for r in rows]


def production(root):
    rows = read(root / 'ledger.json', [])
    groups = {}
    for row in rows:
        # Derive added summary fields from immutable original logs as well as
        # from jobs recorded after the reporting fields were introduced.
        log = root / Path(row['log']).name
        if not log.exists():
            log = Path(row['log'])
        if log.exists():
            row['profile'] = profile(log)
        match = re.fullmatch(r'production-(fl-turbo|ref-turbo)-(off|nvfp4)-42-r[01]-(default|sage2\+\+|sage3)', row['name'])
        if not match:
            raise ValueError('unexpected production workload: ' + row['name'])
        groups.setdefault(match.groups(), []).append(row)
    results = []
    for (family, projection, mode), runs in groups.items():
        baseline = groups.get((family, projection, 'default'), [])
        item = {'family': family, 'projection': projection, 'mode': mode,
                'status': status(all(r['returncode'] == 0 and r['main_attention_coverage']['pass'] for r in runs), len(runs), 2),
                'wall_seconds': spread([r['wall_seconds'] for r in runs]),
                'startup_seconds': spread([r['startup_seconds'] for r in runs]),
                'denoising_seconds': spread([r['profile']['denoising_seconds'] for r in runs]),
                'peak_rss_bytes': max(r['child_peak_rss_bytes'] for r in runs),
                'peak_process_device_bytes': max(r['gpu_telemetry']['peak_process_bytes'] for r in runs),
                'coverage': [r['main_attention_coverage'] for r in runs],
                'binary_sha256': sorted({r['binary_sha256'] for r in runs}),
                'commands': [r['argv'] for r in runs],
                'logs': [r['log'] for r in runs],
                'phase_durations': [r['profile']['phase_durations'] for r in runs],
                'cuda_profiles': [r['profile']['cuda_profiles'] for r in runs],
                'sage_profiles': [r['profile']['sage_profiles'] for r in runs]}
        item['denoising_cuda_counters'] = [r['profile'].get('denoising_cuda_counters') for r in runs]
        item['projection_quantization'] = [r['profile'].get('quant_sections', []) for r in runs]
        state_hashes = [r.get('av_sha256') for r in runs]
        item['repeated_av_state'] = {'sha256': state_hashes,
            'identical': len(set(state_hashes)) == 1 if len(runs) == 2 and all(state_hashes) else None}
        item['main_dit_sequence_lengths'] = sorted({seq for r in runs for seq in r['profile'].get('main_dit_sequence_lengths', [])})
        item['concurrent_cuda_pids'] = sorted({pid for r in runs for pid in r['gpu_telemetry']['concurrent_cuda_pids']})
        if len(baseline) == 2:
            item['fraction_of_default_denoising'] = item['denoising_seconds']['mean'] / statistics.mean(r['profile']['denoising_seconds'] for r in baseline)
            item['fraction_of_default_wall'] = item['wall_seconds']['mean'] / statistics.mean(r['wall_seconds'] for r in baseline)
        results.append(item)
    return {'status': status(all(r['status'] != 'failed' for r in results), len(rows), 24),
            'completed_runs': len(rows), 'expected_runs': 24, 'groups': results}


def numeric(root):
    rows = read(root / 'comparisons.json', [])
    groups = {}
    for row in rows:
        for domain, metrics in row['metrics'].items():
            groups.setdefault((row['mode'], row['projection'], domain), []).append((row, metrics))
    results = []
    for (mode, projection, domain), values in groups.items():
        results.append({'mode': mode, 'projection': projection, 'domain': domain,
                        'cases': len(values),
                        'maximum_relative_l2': max(v['relative_l2'] for _, v in values),
                        'maximum_abs_error': max(v['max_abs'] for _, v in values),
                        'nonfinite_cases': sum(not v['finite'] for _, v in values),
                        'failed_domain_pairs': [r['name'] for r, _ in values if r.get('domain_numeric_pass', {}).get(domain) is False]})
    probes = read(root / 'probe-acceptance.json', [])
    capture_groups = read(root / 'probes.json', [])
    probe_groups = {}
    for row in measurements(capture_groups):
        key = (row['stage'], row['mode'], row['projection'], row['domain'])
        probe_groups.setdefault(key, []).append(row)
    probe_summaries = []
    for (stage, mode, projection, domain), values in sorted(probe_groups.items()):
        probe_summaries.append({'stage': stage, 'mode': mode, 'projection': projection,
            'domain': domain, 'cases': len(values),
            'reference': 'independent FP32 SDPA' if stage == 'attention' else 'default full-core hidden state',
            'worst_relative_l2': max(values, key=lambda r: r['metrics']['relative_l2']),
            'worst_max_abs': max(values, key=lambda r: r['metrics']['max_abs'])})
    captures = [capture for group in capture_groups for capture in group['attention']]
    port_rows = [row for capture in captures for row in capture['results']]
    port = {}
    for mode in ('sage2++', 'sage3'):
        values = [row for row in port_rows if row['mode'] == mode]
        errors = [row['upstream']['relative_l2'] for row in values if row.get('oracle_finite')]
        domain_errors = [domain['upstream']['relative_l2'] for row in values if row.get('oracle_finite')
                         for domain in row.get('query_domains', {}).values()]
        port[mode] = {'cases': len(values), 'failed_cases': sum(not row['port_pass'] for row in values),
                      'maximum_global_relative_l2': max(errors, default=None),
                      'maximum_query_domain_relative_l2': max(domain_errors, default=None)}
    return {'complete': (root / 'complete.json').exists(), 'final_latent_pairs': len(rows),
            'groups': results, 'probe_measurements': len(probes),
            'failed_probe_measurements': sum(not r['numeric_pass'] for r in probes),
            'failed_probe_measurements_by_mode': {mode: sum(r['mode'] == mode and not r['numeric_pass'] for r in probes)
                                                   for mode in ('sage2++', 'sage3')},
            'failed_final_latent_pairs': [r['name'] for r in rows if r.get('numeric_pass') is False],
            'real_qkv_captures': len(captures),
            'full_core_comparisons': sum(len(mode['results']) for group in capture_groups for mode in group['core']),
            'query_domain_measurements': sum(len(values) for values in probe_groups.values()),
            'port_parity': port,
            'approximation_by_stage_and_query_domain': probe_summaries,
            'calibration_hard_ceiling_exceeded_pairs': [r['name'] for r in rows if any(not m['finite'] or m['relative_l2'] > .1 or m['max_abs'] > 1. for m in r['metrics'].values())] if root.name.startswith('calibration') else None,
            'probe_failure_examples': [r for r in probes if not r['numeric_pass']][:20]}


def equality(root, expected):
    files = sorted(root.glob('*-equality.json'))
    records = {r['name']: r for r in read(root / 'ledger.json', [])}
    pairs = []
    for path in files:
        name = path.name.removesuffix('-equality.json')
        metrics = read(path)
        hashes = [records.get(name + '-' + part, {}).get('av_sha256')
                  for part in ('whole', 'resumed')]
        jobs = [records.get(name + '-' + part) for part in ('whole', 'pause', 'resumed')]
        dispatch = []
        if all(jobs):
            command = jobs[0]['argv']
            steps = int(command[command.index('--steps') + 1])
            mode = command[command.index('--cuda-attention') + 1]
            pause = jobs[1]['argv']
            split = int(pause[pause.index('--stop-after-step') + 1])
            selected, other = ('sage2', 'sage3') if mode == 'sage2++' else ('sage3', 'sage2')
            for row, evaluations in zip(jobs, (steps, split, steps - split)):
                counters = row['profile']['sage_profiles']
                def calls(field):
                    values = [int(m[1]) for line in counters
                              if (m := re.search(field + r'=(\d+)', line))]
                    return max(values, default=0)
                observed = calls(selected)
                dispatch.append({'name': row['name'], 'mode': mode,
                    'expected': evaluations * 50, 'observed': observed,
                    'other_sage_calls': calls(other),
                    'pass': observed == evaluations * 50 and calls(other) == 0})
        pairs.append({'name': name, 'metrics': metrics,
            'finite_and_exact': set(metrics) == {'video', 'audio'} and
                                all(v['exact'] and v['finite'] for v in metrics.values()),
            'av_state_sha256': hashes,
            'av_file_exact': hashes[0] == hashes[1] if all(hashes) else None,
            'dispatch': dispatch,
            'dispatch_pass': all(d['pass'] for d in dispatch) if len(dispatch) == 3 else None})
    failed = [p['name'] for p in pairs if not p['finite_and_exact'] or
              p['av_file_exact'] is False or p['dispatch_pass'] is False]
    complete = len(files) == expected and all(p['av_file_exact'] is not None and
                                             p['dispatch_pass'] is not None for p in pairs)
    return {'status': 'failed' if failed else 'passed' if complete else 'incomplete',
            'completed_pairs': len(files), 'expected_pairs': expected, 'failed': failed,
            'pairs': pairs}


def continuation(root, expected):
    files = sorted(root.glob('*-prefix.json'))
    failed = [str(p) for p in files if not all(read(p)[key] for key in ('video_augmented_prefix_exact', 'audio_prefix_exact'))]
    return {'status': status(not failed, len(files), expected),
            'completed_segments': len(files), 'expected_segments': expected, 'failed': failed}


def decoding(root):
    result = read(root / 'complete.json', {'status': 'incomplete'})
    result['comparisons'] = []
    for row in read(root / 'results.json', []):
        result['comparisons'].append({
            'source': row['source'], 'source_sha256': row['source_sha256'],
            'presentation_version': row['presentation_version'], 'vae': row['vae'],
            'exact_across_builds': row['exact_across_builds'],
            'outputs': {build: {
                'binary_sha256': value['record']['binary_sha256'],
                'command': value['record']['argv'],
                'returncode': value['record']['returncode'],
                'video': value['video'], 'audio': value['audio'],
                'media': value['media'],
                'gpu_telemetry': value['record']['gpu_telemetry']}
                for build, value in row['outputs'].items()}})
    return result


def sensitivity(root):
    result = read(root / 'complete.json', {'status': 'incomplete'})
    rows = read(root / 'results.json', [])
    result['schedules'] = []
    for schedule in sorted({r['schedule'] for r in rows}):
        values = [r for r in rows if r['schedule'] == schedule]
        result['schedules'].append({
            'schedule': schedule, 'cases': len(values),
            'hard_ceiling_passes': sum(r['hard_ceiling_pass'] for r in values),
            'worst_by_domain': {domain: {
                'relative_l2': max(r['metrics'][domain]['relative_l2'] for r in values),
                'max_abs': max(r['metrics'][domain]['max_abs'] for r in values),
                'nonfinite_cases': sum(not r['metrics'][domain]['finite'] for r in values)}
                for domain in ('video', 'audio')},
            'jobs': [{'name': r['name'], 'diagnostic_plan': r['diagnostic_plan'],
                'dispatches': r['dispatches'], 'metrics': r['metrics'],
                'binary_sha256': r['record']['binary_sha256'],
                'command': r['record']['argv'], 'environment': r['record']['environment']}
                for r in values]})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--suffix', default='v2', help='Workflow evidence suffix')
    p.add_argument('--kernel-stage', default='rounding-fixed', help='Native qualification directory')
    a = p.parse_args(); root = a.root; kernel = root / a.kernel_stage
    def stage(name):
        return root / (name + '-' + a.suffix)
    environment = read(kernel / 'environment-workflows.json', read(kernel / 'environment.json', {}))
    human_review = read(root / ('human-review-status-' + a.suffix + '.json'), {'status': 'pending'})
    human_status = human_review['status']
    if human_status not in ('pending', 'skipped-by-user'):
        raise ValueError('unrecognized human-review status')
    result = {'schema': 2, 'status': 'qualification-in-progress',
              'workflow_binary_sha256': environment.get('h3_sha256'),
              'evidence_root': str(root), 'human_review': human_status, 'human_review_record': human_review,
              'production_recommendation': None, 'checks': {}}
    checks = result['checks']
    for name, expected in [('native-functional', 74), ('oracle-short', 240), ('oracle-long', 56), ('oracle-regression', 4), ('oracle-sage3-regression', 4), ('oracle-kmean-boundaries', 48), ('oracle-ref-production', 8), ('oracle-ram-scratch-regression', 4)]:
        rows = read(kernel / name / 'results.json', [])
        failures = [r for r in rows if r.get('returncode', 0) or not r.get('port_pass', True)]
        checks[name] = {'status': status(not failures, len(rows), expected), 'cases': len(rows), 'expected': expected, 'failures': failures}
    for mode in ('sage2', 'sage3'):
        packs = read(kernel / (mode + '-packs129') / 'results.json')
        expected = 7 if mode == 'sage2' else 10
        checks[mode + '-intermediates'] = {'status': 'incomplete'} if packs is None else {
            'status': status(all(v['byte_equal_fraction'] == 1. for v in packs.values()), len(packs), expected),
            'tensors': packs, 'expected_tensors': expected}
        sanitizer = kernel / ('sanitizer-' + mode + '.log')
        checks[mode + '-sanitizer'] = {'status': 'incomplete' if not sanitizer.exists() else 'passed' if 'ERROR SUMMARY: 0 errors' in sanitizer.read_text() else 'failed'}
    for name, marker in [('host-sampler', '1552 sampler state checks'), ('host-presentation', 'ok: defaults, temporal contract, presentation'),
                         ('containers', '165 adversarial container cases'), ('failures', 'PASS')]:
        log = kernel / (name + '.log')
        checks[name] = {'status': 'incomplete' if not log.exists() else 'passed' if marker in log.read_text() else 'failed'}
    for name, log in [('cli', kernel / 'cli-resume-preflight.log'),
                      ('non_sage_cli', root / 'cli-non-sage-final.log'),
                      ('non_sage_cudnn_cli', root / 'cli-non-sage-cudnn-final.log')]:
        value = log.read_text() if log.exists() else ''
        checks[name] = {'status': 'incomplete' if not value else 'passed' if 'Ran 8 tests' in value and '\nOK' in value else 'failed'}
    result['component'] = component(kernel / 'component')
    result['component_actual_ref_production'] = component(kernel / 'component-ref-production')
    result['component_without_profile'] = component(kernel / 'component-no-profile')
    result['calibration'] = numeric(stage('calibration'))
    result['heldout'] = numeric(stage('heldout'))
    limits = root / ('frozen-limits-' + a.suffix + '.json')
    result['frozen_limits'] = {'path': str(limits), 'sha256': sha(limits), 'values': read(limits),
        'provenance': read(root / ('limits-freeze-provenance-' + a.suffix + '.json'))} if limits.exists() else None
    result['production'] = production(stage('production'))
    result['production']['counter_notes'] = {
        'compressed_weight_uploads': 'NVFP4 compressed weight uploads on the compute stream increment H2D/stream byte counters but are outside the legacy transfer-event and source-read/upload timers. Denoising and process wall times include them; do not derive compressed-upload bandwidth from those incomplete timers.',
        'cache_scopes': 'CUDA weight-cache counters describe the BF16 device cache. Projection-quantization artifact_cache_hits/prepared_weights describe disk artifact reuse during loading; these are separate caches.'}
    checks['cpu_resume'] = equality(kernel / 'resume-cpu', 2)
    defaults = read(kernel / 'default-build-regression.json')
    checks['default_build_regression'] = {'status': 'incomplete'} if defaults is None else {
        'status': status(defaults['all_latents_exact'], defaults['cases'], 10),
        'record': defaults}
    context = read(kernel / 'context' / 'context.json')
    checks['context_reuse'] = {'status': 'incomplete' if context is None else 'passed' if context['returncode'] == 0 else 'failed',
                               'record': context}
    checks['gpu_resume'] = equality(stage('resume-gpu'), 12)
    checks['target_resume'] = equality(stage('resume-target'), 8)
    checks['cross_mode_continuation'] = continuation(stage('continuation-short'), 18)
    for mode in ('sage2', 'sage3'):
        checks['target_continuation_' + mode] = continuation(stage('continuation-target-' + mode), 2)
    continuation_skip = read(root / ('continuation-skip-' + a.suffix + '.json'))
    result['continuation_skip'] = continuation_skip
    if continuation_skip:
        if continuation_skip['status'] != 'remaining-continuation-skipped-by-user':
            raise ValueError('unrecognized continuation skip status')
        for mode in ('sage2', 'sage3'):
            check = checks['target_continuation_' + mode]
            if check['status'] == 'incomplete':
                check.update(status='skipped-by-user', reason=continuation_skip['instruction'])
    result['decode_replay'] = decoding(stage('decode-replay'))
    result['decode_review_pairs'] = decoding(stage('decode-review'))
    result['decode_audio_pairs'] = decoding(stage('decode-audio'))
    audio_jobs = read(stage('heldout-audio') / 'ledger.json', [])
    result['extended_audio_review'] = {'status': status(all(r['returncode'] == 0 and r['main_attention_coverage']['pass'] for r in audio_jobs), len(audio_jobs), 6),
        'completed_runs': len(audio_jobs), 'expected_runs': 6,
        'comparisons': read(stage('heldout-audio') / 'comparisons.json', []), 'human_review': human_status}
    result['profiling_overhead'] = read(stage('profiling-overhead') / 'results.json', [])
    result['mixed_sensitivity'] = sensitivity(stage('sensitivity'))
    result['validation_driver_corrections'] = read(root / ('warmup-cli-fix-' + a.suffix + '.json'), [])
    result['queue_restart'] = read(root / ('qualification-restart-' + a.suffix + '.json'))
    result['continuation_skip_restart'] = read(root / ('qualification-continuation-skip-restart-' + a.suffix + '.json'))
    result['completion_markers'] = {name: (root / name).read_text().strip() if (root / name).exists() else None
                                    for name in ('qualification-' + a.suffix + '-complete.txt', 'qualification-extended-' + a.suffix + '-complete.txt')}
    if all(result['completion_markers'].values()):
        result['status'] = 'automated-qualification-complete'
    tests_skip = read(root / ('tests-skip-' + a.suffix + '.json'))
    result['tests_skip'] = tests_skip
    if tests_skip:
        if tests_skip['status'] != 'remaining-tests-skipped-by-user':
            raise ValueError('unrecognized test skip status')
        result['status'] = 'qualification-closed-with-skipped-tests'
        for key in ('decode_replay', 'decode_review_pairs', 'decode_audio_pairs',
                    'extended_audio_review', 'mixed_sensitivity'):
            if result[key].get('status') == 'incomplete':
                result[key].update(status='skipped-by-user', reason=tests_skip['instruction'])
        result['profiling_overhead_status'] = {
            'status': 'completed' if len(result['profiling_overhead']) == 3 else 'skipped-by-user',
            'reason': tests_skip['instruction'],
            'jobs': [row for path in sorted(stage('profiling-overhead').glob('overhead-*.json'))
                     if 'returncode' in (row := read(path))]}
    result['mode_acceptance'] = {}
    for mode in ('sage2++', 'sage3'):
        failed_domains = [g for g in result['heldout']['groups'] if g['mode'] == mode and (g['nonfinite_cases'] or g['failed_domain_pairs'])]
        result['mode_acceptance'][mode] = {'status': 'unqualified',
            'numerical_quality': 'failed' if failed_domains or result['heldout']['failed_probe_measurements_by_mode'][mode] else 'passed' if result['heldout']['complete'] else 'incomplete',
            'failed_final_latent_domains': failed_domains, 'human_review': human_status}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
