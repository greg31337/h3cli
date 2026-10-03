#!/usr/bin/env python3
"""Sequential T033–T070 model oracle; M5 is optional, never simulated as certified."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import time
from test_sampler_file import entries, build

from source_tree import copy_source_tree

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    't2va': dict(mode='t2va', boundaries=[1, 2, 4, 10, 19]),
    'ref2va': dict(mode='ref2va'),
    'mixed': dict(mode='mixed', legacy_video=True),
    'continuation141': dict(mode='ref2va', continuation=True, frames=141),
    'bridge': dict(mode='bridge', continuation=True),
    'reuse2': dict(mode='t2va', reuse=2, boundaries=[1, 2, 3, 4, 5]),
    'reuse3': dict(mode='t2va', reuse=3, boundaries=[1, 2, 3, 4, 5]),
    'custom': dict(mode='t2va', reuse=3, custom='0,2,5,9,19', boundaries=[2, 3, 4, 5, 6]),
    'core4': dict(mode='t2va', core=4, boundaries=[3, 4, 5]),
    'reduction': dict(mode='t2va', reduction=True),
    'reduction_core4': dict(mode='t2va', core=4, reduction=True, boundaries=[3, 4, 5]),
    'gpu': dict(mode='t2va', gpu=True),
    'gpu_reuse3': dict(mode='t2va', gpu=True, reuse=3, boundaries=[3, 4, 5]),
    'gpu_core4': dict(mode='t2va', gpu=True, core=4, boundaries=[3, 4, 5]),
    'gpu_reduction': dict(mode='t2va', gpu=True, reduction=True),
}


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', default=','.join(CASES))
    parser.add_argument('--output', type=Path, required=True, help='Fresh artifact directory')
    parser.add_argument('--require-m5', action='store_true', help='Fail unless running on actual M5 hardware')
    args = parser.parse_args()
    machine = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    if args.require_m5 and 'M5' not in machine:
        raise SystemExit('M5 certification requires actual M5 hardware; use the normal M4 run here')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = out/'build'; frozen.mkdir()
    (frozen/'src/metal').mkdir(parents=True,exist_ok=True)
    (frozen/'bin').mkdir()
    for name in ['bin/h3cli', 'bin/sampler_generate', 'src/metal/shaders.metal']:
        shutil.copy2(ROOT/name, frozen/name)
    sources = frozen/'source'; sources.mkdir(); (sources/'tests').mkdir()
    copy_source_tree(ROOT, sources)
    shutil.copy2(ROOT/'tests/sampler_generate.c', sources/'tests/sampler_generate.c')
    manifest = dict(passed=False, device=machine, m5_certified='M5' in machine,
                    binary_sha256=digest(frozen/'bin/sampler_generate'), runs={})
    (frozen/'source-sha256.json').write_text(json.dumps({str(p.relative_to(sources)): digest(p)
        for p in sources.rglob('*') if p.is_file()}, indent=2)+'\n')

    def save():
        (out/'results.json').write_text(json.dumps(manifest, indent=2)+'\n')

    def run(label, command, cwd, env):
        print('running', label, flush=True); start = time.monotonic()
        with (out/(label+'.log')).open('w') as log:
            subprocess.run([str(v) for v in command], cwd=cwd, env=env, stdout=log, stderr=log, check=True)
        return dict(command=[str(v) for v in command], seconds=time.monotonic()-start)

    def same(a, b):
        for extension in ['h3av', 'mp4']:
            assert digest(Path(str(a)+'.'+extension)) == digest(Path(str(b)+'.'+extension)), (a, b, extension)

    for name in args.only.split(','):
        config = CASES[name]; directory = out/name; directory.mkdir()
        (directory/'inputs').mkdir()
        (directory/'src/metal').mkdir(parents=True,exist_ok=True)
        for image in ['face1.jpg', 'body1.jpg']:
            shutil.copy2(ROOT/'inputs'/image, directory/'inputs'/image)
        (directory/'outputs').symlink_to(ROOT/'outputs')
        (directory/'src/metal/shaders.metal').symlink_to(frozen/'src/metal/shaders.metal')
        env = {k: v for k, v in os.environ.items() if not k.startswith('H3_')}
        env['H3_GPU_SAMPLER' if config.get('gpu') else 'H3_CPU_SAMPLER'] = '1'
        for field, variable in [('core', 'H3_TEST_CORE'), ('frames', 'H3_TEST_FRAMES'), ('reduction', 'H3_TEST_REDUCTION')]:
            if config.get(field): env[variable] = str(int(config[field]))
        if config.get('custom'): env['H3_REUSE_STEPS'] = config['custom']
        if config.get('legacy_video'): env['H3_TEST_LEGACY_REFVIDEO'] = '1'
        boundaries = config.get('boundaries', [4])
        if 'boundaries' in config: env['H3_TEST_BOUNDARIES'] = ','.join(map(str, boundaries))
        source = '-'
        if config.get('continuation'):
            source = directory/'source.h3av'
            shutil.copy2(ROOT/'outputs/resume-validation/baseline/ref2va-original.h3av', source)
        base = directory/'oracle'
        record = run(name+'-oracle', [frozen/'bin/sampler_generate', ROOT/'models/MiniMax-H3', base,
                     config['mode'], source, config.get('reuse', 1)], directory, env)
        record['config'] = config; record['checks'] = []
        manifest['runs'][name] = record; save()
        parts = {p[0]: p[-1] for p in entries(Path(str(base)+'.h3sample').read_bytes())}
        identity = parts[1]
        assert struct.unpack_from('<I', identity, len(identity)-4)[0] == int(config.get('gpu', False))
        assert 30 in parts and len(parts[30]) > 1000, 'prepared cache was not exported'
        if config.get('gpu') and config.get('reuse', 1) > 1:
            assert all(k in parts for k in range(25, 29))
        if config.get('core'): assert 29 in parts and len(parts[29]) > 0
        if config.get('reduction'): assert struct.unpack_from('<I', parts[24], 36)[0] == 1
        assert Path(str(base)+'.h3sample').read_bytes() == Path(str(base)+'.after-preview.h3sample').read_bytes()
        if 'boundaries' in config: same(base, Path(str(base)+'.same'))
        (directory/'inputs').rename(directory/'unavailable-inputs')
        if source != '-': source.unlink()
        if name == 'mixed':
            (directory/'outputs').unlink(); (directory/'outputs').mkdir()
            (directory/'outputs/.h3-model-hashes').symlink_to(ROOT/'outputs/.h3-model-hashes')

        def resume(label, checkpoint, stop=None, expected=True):
            destination = directory/label
            command = [frozen/'bin/sampler_generate', ROOT/'models/MiniMax-H3', destination,
                       'resume', checkpoint, Path(str(base)+'.trace')]
            if stop is not None: command.append(stop)
            result = run(name+'-'+label, command, directory, env)
            if expected: same(base, destination)
            record['checks'].append(dict(label=label, **result)); save()
            return destination

        for boundary in boundaries:
            checkpoint = Path(str(base)+('.h3sample' if boundary == 4 else f'.step{boundary}.h3sample'))
            resume('resume'+str(boundary), checkpoint)
        log = (out/(name+'-resume4.log')).read_text()
        assert 'prepared refined text restored' in log and 'prepared timestep/AdaLN tensors restored' in log
        if name == 't2va':
            current = Path(str(base)+'.h3sample')
            for stop in [8, 12, 20]:
                destination = resume('chain'+str(stop), current, stop, expected=stop == 20)
                current = Path(str(destination)+'.h3sample')
                if stop < 20:
                    provenance = next(p[-1] for p in entries(current.read_bytes()) if p[0] == 31)
                    assert struct.unpack_from('<I', provenance)[0] == (1 if stop == 8 else 2)
            source_parts = entries(Path(str(base)+'.h3sample').read_bytes())
            for variant in ['removed', 'incompatible-version', 'incompatible-key']:
                altered = [[*p[:-1], bytearray(p[-1])] for p in source_parts]
                if variant == 'removed': altered = [p for p in altered if p[0] != 30]
                elif variant == 'incompatible-version': next(p for p in altered if p[0] == 30)[1] = 2
                else: next(p for p in altered if p[0] == 30)[-1][4] ^= 1
                checkpoint = directory/(variant+'.h3sample'); checkpoint.write_bytes(build(altered))
                resume(variant, checkpoint)
                fallback_log = (out/(name+'-'+variant+'.log')).read_text()
                assert 'prepared refined text restored' not in fallback_log and 'prepared timestep/AdaLN tensors restored' not in fallback_log
                checkpoint.unlink()
        if name == 'continuation141':
            # Segment 3 consumes segment 2's uninterrupted and resumed AV states
            # in one conditioning context, avoiding unrelated cold-encoder noise.
            (directory/'unavailable-inputs').rename(directory/'inputs')
            env['H3_TEST_ALT_SOURCE'] = str(directory/'resume4.h3av')
            third = directory/'segment3'
            record['segment3'] = run(name+'-segment3', [frozen/'bin/sampler_generate', ROOT/'models/MiniMax-H3',
                third, 'ref2va', Path(str(base)+'.h3av'), 1], directory, env)
            same(third, Path(str(third)+'.alternate'))
        record['hashes'] = {ext: digest(Path(str(base)+'.'+ext)) for ext in ['h3av', 'mp4', 'trace', 'h3sample']}
        record['passed'] = True; save()
    manifest['passed'] = True; save()
    print('ok: all requested advanced sampler comparisons passed on', machine, flush=True)


if __name__ == '__main__':
    main()
