#!/usr/bin/env python3
"""Compare pre-bridge/default-hard/explicit-hard real-model results.

Runs sequentially, with all 50 DiT layers and 20 Euler transitions by default.
Replays captured conditioning to isolate pre-existing multimodal text-encoder
variability. Hooks are installed only in isolated build copies, using the
existing continuation regression helper. No production instrumentation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from source_tree import build_path
import shutil
import subprocess
import tarfile
import time

from continuation_regression import instrument

from source_tree import copy_source_tree

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='6c09804')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/MiniMax-H3')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/bridge-validation/regression')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--reference', type=Path, help='Reuse verified source-before/hard-before artifacts from a passing prior baseline run')
    args = parser.parse_args()
    out = args.output.resolve()
    if args.reference:
        args.reference = args.reference.resolve()
        assert out != args.reference, 'reference artifacts must remain separate from this run'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps({'passed': False, 'status': 'running'}, indent=2) + '\n')
    original, current = out / 'original', out / 'current'
    # Always rebuild current source. Optional baseline reuse is hash- and
    # configuration-verified below; current results are never reused.
    if not args.reference:
        original.mkdir(exist_ok=True)
        archive = out / 'baseline.tar'
        subprocess.run(['git', 'archive', '-o', str(archive), args.baseline], cwd=ROOT, check=True)
        with tarfile.open(archive) as tar:
            tar.extractall(original, filter='data')
    current.mkdir(exist_ok=True)
    copy_source_tree(ROOT, current)
    for directory in ([current] if args.reference else [original, current]):
        instrument(directory)
        with (directory / 'build.log').open('w') as log:
            subprocess.run(['make', '-B', '-j8', 'all'], cwd=directory, stdout=log, stderr=log, check=True)
    common = ['-d', str(args.model.resolve()), '--width', '256', '--height', '256',
              '--frames', '90', '--steps', str(args.steps), '--reuse', '1', '--core-reuse', '1', '--layers', '50',
              '--ref-image', str(ROOT / 'inputs/face1.jpg'), '--ref-image', str(ROOT / 'inputs/body1.jpg')]
    prompt = ('The woman with the face in <Picture 1> and the outfit in <Picture 2> '
              'walks slowly through a sunlit garden and waves. Steady camera and quiet outdoor ambience.')
    env = {k: v for k, v in os.environ.items() if not k.startswith('H3_')}
    records = {'passed': False, 'baseline_commit': subprocess.check_output(['git', 'rev-parse', args.baseline], cwd=ROOT, text=True).strip(),
               'steps': args.steps, 'inputs': {name: digest(ROOT / 'inputs' / name) for name in ['face1.jpg', 'body1.jpg']}, 'runs': {}}
    record_path = out / 'results.json'

    def reference(name, seed, source=None):
        manifest_path = args.reference / 'results.json'
        prior = json.loads(manifest_path.read_text())
        assert prior['passed'] and all(prior[k] == records[k] for k in ['baseline_commit','steps','inputs'])
        record = prior['runs'][name]
        base = args.reference / name
        expected = common + ['-p',prompt,'--seed',str(seed),'-o',str(base)+'.mp4',
                             '--save-av-state',str(base)+'.h3av']
        if source: expected += ['--continue-from',str(args.reference/source)]
        assert record['command'][1:] == expected, 'baseline render configuration changed'
        assert digest(Path(record['command'][0])) == record['binary_sha256'], 'baseline binary changed'
        for suffix, expected_hash in record['hashes'].items():
            path = Path(str(base)+'.'+suffix)
            assert digest(path) == expected_hash, 'baseline artifact changed: '+str(path)
            shutil.copy2(path, out/(name+'.'+suffix))
        assert set(record['hashes']) == {'text','condition-video','video','audio','h3av','mp4'}
        shutil.copy2(Path(str(base)+'.log'),out/(name+'.log'))
        records['runs'][name] = dict(record, reused_from=str(manifest_path),
                                    reference_manifest_sha256=digest(manifest_path))
        record_path.write_text(json.dumps(records,indent=2)+'\n')
        print('verified baseline artifacts',name,flush=True)
        return record['hashes']

    def run(name, directory, seed, source=None, replay=None, flags=()):
        base = out / name
        cmd = [str(build_path(directory, 'h3cli'))] + common + ['-p', prompt, '--seed', str(seed),
               '-o', str(base) + '.mp4', '--save-av-state', str(base) + '.h3av']
        if source:
            cmd += ['--continue-from', str(source)]
        cmd += list(flags)
        run_env = dict(env, H3_REGRESSION_DUMP=str(base))
        if replay:
            run_env['H3_REGRESSION_REPLAY'] = str(out / replay)
        print('running', name, flush=True)
        started = time.monotonic()
        with Path(str(base) + '.log').open('w') as log:
            subprocess.run(cmd, cwd=directory, env=run_env, stdout=log, stderr=log, check=True)
        hashes = {suffix: digest(Path(str(base) + '.' + suffix)) for suffix in
                  ['text', 'condition-video', 'video', 'audio', 'h3av', 'mp4']}
        records['runs'][name] = {'command': cmd, 'seconds': time.monotonic() - started,
                                 'hashes': hashes, 'binary_sha256': digest(build_path(directory, 'h3cli'))}
        record_path.write_text(json.dumps(records, indent=2) + '\n')
        return hashes

    before = reference('source-before',42) if args.reference else run('source-before', original, 42)
    after = run('source-after', current, 42, replay='source-before')
    assert before == after, 'unmasked Ref2VA regression'
    source = out / 'source-before.h3av'
    before = reference('hard-before',43,'source-before.h3av') if args.reference else run('hard-before', original, 43, source=source)
    after = run('hard-default', current, 43, source=source, replay='hard-before')
    assert before == after, 'default hard continuation regression'
    explicit = run('hard-explicit', current, 43, source=source, replay='hard-before', flags=[
        '--continue-mode', 'hard', '--continue-bridge-steps', '11',
        '--continue-bridge-max-strength', '0.65', '--continue-bridge-profile', 'ease-out'])
    assert before == explicit, 'explicit hard or ignored bridge tuning changes generation'
    # Exercise the bridge video preparation helpers on a generated clean AV
    # state as well as the five encoder-derived image fixtures.
    with (out / 'bridge-preparation.log').open('w') as log:
        subprocess.run([str(ROOT / 'bin/bridge_tests'), '--state', str(source)], cwd=ROOT,
                       stdout=log, stderr=log, check=True)
    for name in records['runs']:
        probe = subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(out / (name + '.mp4'))], text=True)
        streams = json.loads(probe)['streams']
        expected = 90 / 24 if name.startswith('source-') else 51 / 24
        assert {s['codec_type'] for s in streams} == {'video', 'audio'}
        assert all(abs(float(s['duration']) - expected) < .001 and float(s['start_time']) == 0 for s in streams)
        records['runs'][name]['av_seconds'] = expected
    records['passed'] = True
    record_path.write_text(json.dumps(records, indent=2) + '\n')
    print('ok: unmasked/default-hard/explicit-hard final AV latents, h3av states and MP4 bytes match pre-bridge baseline', flush=True)


if __name__ == '__main__':
    main()
