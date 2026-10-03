#!/usr/bin/env python3
"""Independent Metal closing checks. Speed targets are reported separately."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
from fractions import Fraction
import hashlib
import json
import pathlib
import subprocess
import time
import numpy as np


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def sha(path):
    if not path.is_file():
        return None
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def lines(path):
    result = []
    if path.is_file():
        for line in path.read_text(errors='replace').splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='outputs/fast-vae/metal')
    args = parser.parse_args()
    root = pathlib.Path(args.root)
    checks, hashes, media, sampling = [], {}, [], []
    records = {p.parent.name: read(p, {}) for p in root.glob('*/record.json')}
    extended = read(root / 'max-image-extended-jobs.json', [])
    replacements = {job['replaces']: job['id'] for job in extended}

    def check(name, passed, **detail):
        checks.append(dict(name=name, passed=bool(passed), **detail))

    def equal(name, paths):
        paths = [str(pathlib.Path(replacements.get(pathlib.Path(p).parts[0], pathlib.Path(p).parts[0]),
                                  *pathlib.Path(p).parts[1:])) for p in paths]
        values = [sha(root / p) for p in paths]
        hashes.update(zip(paths, values))
        check(name, all(values) and len(set(values)) == 1, paths=paths, sha256=values)

    manifest = read(root / 'qualification-jobs.json', [])
    manifest += read(root / 'ane-memory-retest-jobs.json', [])
    manifest += extended
    for job in manifest:
        record = records.get(job['id'], {})
        replacement = replacements.get(job['id']) or record.get('superseded_by')
        if replacement:
            record = records.get(replacement, {})
        check('Job: ' + job['id'], record.get('status') == 'pass',
              status=record.get('status', 'missing'), reason=record.get('reason'), replacement=replacement)

    equal('Complete C1 decoder policy leaves latents unchanged',
          ['C1-final-' + m + '/output.h3av' for m in ['reference', 'balanced']])
    for size in ['match', 'max']:
        equal('Ref2VA ' + size + ' decoder policy leaves latents unchanged',
              ['C5-image-' + size + '-' + m + '/output.h3av' for m in ['reference', 'balanced']])
    equal('Cached reference / balanced / preview latents',
          ['final-session/' + m + '.h3av' for m in ['reference', 'balanced', 'preview']])
    equal('Cached shape-change / repeat latents',
          ['final-session/shape-' + m + '.h3av' for m in ['change', 'repeat']])
    equal('Exact resume across decoder policy change',
          ['C5-resume/' + m + '.h3av' for m in ['reference', 'resumed']])
    equal('Global reference override preserves original FP32 output',
          ['final-tile-' + m + '/output.f32' for m in ['reference', 'forced-reference']])
    for suffix in ['f32', 'f32.projected']:
        equal('Fused / unfused QKV bitwise: ' + suffix,
              ['final-tile-' + m + '/output.' + suffix for m in ['default', 'unfused-qkv']])
    tile_quality = []
    reference = root / 'final-tile-reference/output.f32.projected'
    if reference.is_file():
        ref = np.fromfile(reference, '<f4').astype('float64')
        for path in sorted(root.glob('final-tile-*/output.f32.projected')):
            value = np.fromfile(path, '<f4').astype('float64')
            finite = value.shape == ref.shape and bool(np.isfinite(value).all())
            relative = float(np.linalg.norm(value-ref)/max(np.linalg.norm(ref), 1e-30)) if finite else None
            tile_quality.append(dict(candidate=path.parent.name, relative_l2=relative, finite=finite))
            if path.parent.name == 'final-tile-default':
                check('Selected recipe raw projected-tile relative L2 <= 0.02', finite and relative <= .02,
                      relative_l2=relative)
    for case in ['image-match', 'image-max', 'video-short', 'video-tail']:
        for suffix in ['moments', 'epsilon', 'sample']:
            equal('Encoder ' + case + ': ' + suffix,
                  ['encoder-' + case + '-' + m + '/output.' + suffix for m in ['reference', 'balanced']])

    session = [x for x in lines(root / 'final-session/stdout.log') if 'completed_steps' in x]
    check('Five cached API renders each complete six steps',
          len(session) == 5 and all(x['completed_steps'] == 6 for x in session), records=session)
    resume = [x for x in lines(root / 'C5-resume/stdout.log') if 'completed_steps' in x]
    check('Resume outputs complete six steps; paused checkpoint is diagnostic only',
          len(resume) == 1 and resume[0].get('reference_steps') == resume[0].get('resumed_total_steps') == 6,
          records=resume)
    for name, record in records.items():
        if record.get('requested_steps') is not None and name not in replacements:
            check('CLI completed six steps: ' + name,
                  record.get('requested_steps') == record.get('completed_steps') == 6 and record.get('status') == 'pass',
                  requested=record.get('requested_steps'), completed=record.get('completed_steps'))
        samples = read(root / name / 'memory.json', [])
        times = [x[0] for x in samples]
        logged = []
        log = root / name / 'stderr.log'
        if log.is_file():
            for line in log.read_text(errors='replace').splitlines():
                if line.startswith(('h3_step ', 'h3_memory ')):
                    try:
                        logged.append(json.loads(line[line.index('{'):]))
                    except (ValueError, json.JSONDecodeError):
                        pass
        sampling.append(dict(job=name, samples=len(samples),
                             max_gap_seconds=max((b-a for a, b in zip(times, times[1:])), default=None),
                             sampled_physical_footprint_bytes=record.get('peak_physical_footprint_bytes'),
                             sampled_rss_bytes=record.get('peak_rss_bytes'),
                             discrete_vram_bytes=record.get('peak_vram_bytes'),
                             application_logged_peak_physical_footprint_bytes=max((x.get('physical_footprint_bytes', x.get('footprint_bytes', 0)) for x in logged), default=None),
                             application_logged_peak_metal_allocated_bytes=max((x.get('metal_current_allocated_bytes', x.get('metal_allocated_bytes', 0)) for x in logged), default=None)))

    provenance = read(root / 'C1-fixture/provenance.json', {})
    origin = read(root / 'C1-fixture/source-record.json', {})
    check('Retained C1 fixture has six-step provenance', provenance.get('completed_steps') == 6 and
          provenance.get('sha256') == sha(root / 'C1-fixture/result.h3av') and
          origin.get('returncode') == 0 and origin.get('evaluations') == 6 and
          origin.get('evaluation_contract_pass') is True and origin.get('teacher_from') is None,
          provenance=provenance, source_evaluation_contract=origin.get('evaluation_contract_pass'))
    c2 = records.get('C2-fixture', {})
    check('New C2 fixture completes six steps', c2.get('status') == 'pass' and c2.get('completed_steps') == 6)
    quality = {}
    for case in ['C0', 'C1', 'C2', 'C3', 'C4-faces', 'C4-texture']:
        data = read(root / (case + '-quality/metrics.json'), {})
        quality[case] = data.get('gates', {})
        check('Frozen absolute / better-than-preview quality: ' + case,
              bool(quality[case]) and all(quality[case].values()), gates=quality[case])

    for path in sorted(root.rglob('*.mp4')):
        if 'venv' in path.parts:
            continue
        rel = str(path.relative_to(root))
        streams = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-count_frames', '-show_entries',
            'stream=codec_type,width,height,nb_read_frames,duration,r_frame_rate', '-of', 'json', str(path)
        ], text=True))['streams']
        video = next(x for x in streams if x['codec_type'] == 'video')
        parent, stem = path.parent.name, path.stem
        expected, dimensions = None, None
        if parent.startswith(('C1-', 'C2-')):
            expected = 243
            dimensions = (640, 480) if parent.startswith('C1-') else (1344, 768)
        elif parent.startswith('C3-'):
            expected, dimensions = 362, (1344, 768)
        elif parent.startswith(('C0-', 'C4-')) or parent == 'pixels':
            expected = 39
        elif parent == 'final-session':
            expected = 39 if stem.startswith('shape-') else 22
        elif parent.startswith(('C5-image-', 'C5-resume')):
            expected = 22
        elif parent.startswith('C5-continuation'):
            expected = 51
        frames, duration = int(video['nb_read_frames']), float(video['duration'])
        delta = max((abs(float(x['duration']) - duration) for x in streams if x['codec_type'] == 'audio'), default=0.)
        passed = frames > 0 and (expected is None or frames == expected) and Fraction(video['r_frame_rate']) == 24
        passed = passed and abs(duration - frames / 24) < .001 and delta < .085
        passed = passed and (dimensions is None or dimensions == (video['width'], video['height']))
        check('Media: ' + rel, passed)
        media.append(dict(path=rel, sha256=sha(path), expected_frames=expected, streams=streams,
                          expected_dimensions=dimensions, audio_video_delta_seconds=delta, passed=passed))

    # Decode payloads to compare pixels, independently of container metadata.
    for label, paths in [
        ('Continuation generation / decode-only', ['C5-continuation-corrected/output.mp4', 'C5-continuation-decode-corrected/output.mp4']),
        ('Cached same-shape repeat', ['final-session/shape-change.mp4', 'final-session/shape-repeat.mp4'])
    ]:
        payloads = []
        for relative in paths:
            path = root / relative
            if not path.is_file():
                payloads.append(None)
                continue
            result = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:v:0',
                                              '-f', 'hash', '-hash', 'sha256', '-'], text=True).strip()
            payloads.append(result)
        check(label + ' video payload identity', all(payloads) and payloads[0] == payloads[1], payload_sha256=payloads)

    frozen = read(root / 'source.json', {})
    changed = [name for name, value in frozen.get('production_sha256', {}).items() if sha(pathlib.Path(name)) != value]
    check('Final production source matches frozen benchmark source', bool(frozen) and not changed, changed=changed)
    check('Final executable matches frozen benchmark binary',
          bool(frozen) and sha(pathlib.Path('bin/h3cli')) == frozen.get('binary_sha256'))
    check('Frozen source archive checksum', bool(frozen) and sha(root / 'source.tar.gz') == frozen.get('archive_sha256'))
    validation = read(root / 'validation-source.json', {})
    changed = [name for name, value in validation.get('files_sha256', {}).items() if sha(pathlib.Path(name)) != value]
    check('Retained validation source and package inventory', bool(validation) and not changed and
          sha(root / 'validation-source.tar.gz') == validation.get('archive_sha256') and
          sha(root / 'requirements-metal.txt') == validation.get('python_packages_sha256'), changed=changed)
    for name in ['C1-decode', 'C2-decode', 'C3-decode']:
        record = records.get(name, {})
        values = lines(root / name / 'stdout.log')
        check('Bounded unified memory measurement: ' + name,
              record.get('peak_physical_footprint_bytes', 0) > 0 and
              record.get('peak_physical_footprint_bytes', 0) < 110_000_000_000 and
              bool(values) and all(x.get('peak_tensor_bytes', 0) > 0 for x in values),
              peak_physical_footprint_bytes=record.get('peak_physical_footprint_bytes'),
              note='Process footprint includes GPU allocations; sampled peaks can miss transients.')
    ledger = read(root / 'ledger.json', {})
    check('Campaign remains within original four-hour deadline',
          bool(ledger) and (ledger.get('finished_unix') or time.time()) <= ledger.get('deadline_unix', 0),
          elapsed_minutes=((ledger.get('finished_unix') or time.time())-ledger.get('started_unix', time.time()))/60)
    failures = [x for x in checks if not x['passed']]
    result = dict(created_unix=time.time(), checks=checks, state_and_encoder_sha256=hashes,
                  retained_prior_attempts={name: records.get(name, {}) for name in replacements},
                  media=media, quality=quality, raw_tile_quality=tile_quality, sampling=sampling,
                  summary=f'{len(checks)-len(failures)}/{len(checks)} closing checks passed; performance targets reported separately.',
                  C3_provenance='Decode-only repeated-tail expansion of the complete six-step C2 state to 362 frames.',
                  C4_provenance='Denoiser-free photo pan/text fixtures passed through the unchanged original encoder.')
    (root / 'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(result['summary'])
    if failures:
        print(json.dumps(failures, indent=2))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
