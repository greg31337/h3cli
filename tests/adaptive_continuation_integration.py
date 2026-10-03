#!/usr/bin/env python3
"""Bounded native continuation qualification; no SGLang goldens are generated.

Each model subprocess executes at most six original-schedule evaluations.
Approximate trajectories are checked for exact protected history and resume,
not numerical equality with a fully evaluated trajectory.
"""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time

import numpy as np
from test_sampler_file import entries, build


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def parts(path):
    return {p[0]: p for p in entries(Path(path).read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--media', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    assets = out/'inputs'; assets.mkdir()
    for name in ('1.jpg', '2.jpg'):
        shutil.copy2(root/'inputs'/name, assets/name)
    ffmpeg = os.environ.get('H3_SGLANG_INPUT_FFMPEG', 'ffmpeg')
    subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-i', str(args.media.resolve()),
                    '-t', '2', '-vf', 'scale=256:256', '-r', '24', '-c:v', 'libx264',
                    '-c:a', 'aac', '-ar', '32000', '-ac', '2', str(assets/'embedded.mp4')], check=True)
    for name, frequency in [('replacement.wav', 523), ('separate.wav', 880)]:
        subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                        f'sine=frequency={frequency}:sample_rate=32000:duration=2',
                        '-ac', '2', str(assets/name)], check=True)
    model = str(args.model.resolve()); binary = root/'bin/adaptive_latent'
    identities = {str(p.relative_to(root)): sha(p) for p in (binary, root/'bin/h3cli')}
    records, checks = [], []
    base_env = os.environ | {'H3_TEST_MAX_EVALUATIONS': '6', 'H3_TEST_SCHEDULE_STEPS': '6',
                             'H3_EXPERIMENT_TRACE': '1'}

    def publish(complete=False):
        (out/'result.json').write_text(json.dumps(dict(complete=complete,
            passed=complete and all(x['passed'] for x in checks+records),
            binaries=identities, jobs=records, checks=checks), indent=2)+'\n')

    def check(name, ok):
        checks.append(dict(name=name, passed=bool(ok))); publish()
        if not ok:
            raise AssertionError(name)

    def run(name, refs=(), mode='hard', cache='conservative', sparse=False, stop=6,
            resume=None, extra=None, threshold='1', hits='2', size='max', source=None,
            shape=(384, 384, 124)):
        env = dict(base_env)
        if cache != 'off':
            env |= {'H3_TEST_ADAPTIVE_THRESHOLD': threshold, 'H3_TEST_ADAPTIVE_MAX_HITS': hits}
        if refs and not resume:
            ref_file = out/(name+'.references'); ref_file.write_text('\n'.join(refs)+'\n')
            env['H3_TEST_REFERENCES'] = str(ref_file)
        if source and not resume:
            env |= {'H3_TEST_CONTINUE_FROM': str(source), 'H3_TEST_CONTINUE_MODE': mode}
        env |= extra or {}
        argv = [str(binary), model, str(out/name), cache, str(stop), str(resume) if resume else '-',
                '.75' if sparse else '-', *map(str, shape), 'off', '-',
                '2' if cache != 'off' else '0', '3' if sparse else '0', '-', '-', '-', size]
        start = time.monotonic()
        with (out/(name+'.log')).open('w') as log:
            r = subprocess.run(argv, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        text = (out/(name+'.log')).read_text()
        decisions = [dict(re.findall(r'(\w+)=([^ ]+)', line)) for line in text.splitlines()
                     if line.startswith('h3cli: adaptive step=')]
        traces = [json.loads(line.split('h3_experiment ', 1)[1]) for line in text.splitlines()
                  if line.startswith('h3_experiment ')]
        rec = dict(name=name, argv=argv, overrides={k: v for k, v in env.items() if k.startswith('H3_TEST_')},
                   returncode=r.returncode, seconds=time.monotonic()-start, passed=r.returncode == 0,
                   decisions=decisions, trace=traces)
        records.append(rec); publish(); print(name, rec['passed'], round(rec['seconds'], 2), flush=True)
        if r.returncode:
            raise RuntimeError(name+' failed; see its log')
        state = out/(name+'.h3sample')
        check(name+' saved state', state.is_file())
        if source and not resume:
            audit(name, state)
        for decision, trace in zip(decisions, traces):
            if decision['decision'] == 'hit':
                check(name+' hit dispatch '+decision['step'], trace['blocks'] == 1 and
                      not trace['sparse_calls'] and not trace['router_calls'] and trace['stream_read_layers'] <= 1)
        if source and cache != 'off':
            component_lines = [s for s in text.splitlines() if s.startswith('h3cli: adaptive continuation step=')]
            check(name+' class diagnostics', len(component_lines) == len(decisions))
            for line, decision in zip(component_lines, decisions):
                values = line.split('components=', 1)[1].split(' ', 1)[0].strip(',').split(',')
                check(name+' maximum score '+decision['step'],
                      float(decision['score']) == max(float(v.split(':')[2]) for v in values))
        return state

    def audit(name, state):
        p = parts(state); layout = p[10][-1]
        _, vt, lh, lw, at = struct.unpack_from('<5i', layout, 48)
        vp, ap = struct.unpack_from('<2i', layout, 68)
        video_exact = np.arange(vt) < vp; audio_exact = np.arange(at) < ap
        bridge = p[15][-1]
        if struct.unpack_from('<i', bridge, 116)[0]:
            masks = np.array(struct.unpack_from('<24f', bridge, 188))
            video_exact[:vp] = masks[np.frombuffer(bridge, dtype=np.uint8, count=vp, offset=284)] == 0
            audio_exact[:ap] = masks[np.frombuffer(bridge, dtype=np.uint8, count=ap, offset=391)] == 0
        def read(path):
            raw = Path(path).read_bytes(); nv, na = struct.unpack_from('<2Q', raw)
            arr = np.frombuffer(raw, dtype='<u4', offset=16)
            assert nv == 24*vt*lh*lw and na == 64*at
            return arr[:nv].reshape(24, vt, lh, lw), arr[nv:].reshape(64, at)
        original = read(out/(name+'.step-00.bin'))
        for path in sorted(out.glob(name+'.step-*.bin')):
            v, a = read(path)
            check(name+' exact rows '+path.stem, np.array_equal(v[:, video_exact], original[0][:, video_exact]) and
                  np.array_equal(a[:, audio_exact], original[1][:, audio_exact]))
            check(name+' finite '+path.stem, np.isfinite(v.view('<f4')).all() and np.isfinite(a.view('<f4')).all())

    image = [str(assets/'1.jpg')]
    embedded = ['video:'+str(assets/'embedded.mp4')]
    separate = image+['audio:'+str(assets/'separate.wav')]
    mixed = image+['silent:'+str(assets/'embedded.mp4'),
                   'paired:'+str(assets/'embedded.mp4')+'|'+str(assets/'replacement.wav'),
                   'audio:'+str(assets/'separate.wav')]
    source_text, source_ref = assets/'source-text.h3av', assets/'source-ref.h3av'
    run('source-text', cache='off', extra={'H3_TEST_SAVE_AV': str(source_text)})
    run('source-ref', refs=image, cache='off', extra={'H3_TEST_SAVE_AV': str(source_ref)})
    source_hashes = {p.name: sha(p) for p in assets.glob('source-*')}
    cases = []
    for mode in ('hard', 'bridge'):
        for feature in ('adaptive', 'subblock', 'combined'):
            cases.append((mode+'-'+feature, mode, [], feature, 'max', feature != 'subblock'))
    cases += [('image-hard', 'hard', image, 'adaptive', 'max', False),
              ('image-bridge', 'bridge', image, 'combined', 'high', False),
              ('video-hard', 'hard', embedded, 'combined', 'match', True),
              ('audio-bridge', 'bridge', separate, 'adaptive', 'max', True),
              ('mixed-hard', 'hard', mixed, 'combined', 'match', True),
              ('mixed-bridge', 'bridge', mixed, 'combined', 'high', True),
              ('mixed-subblock', 'bridge', mixed, 'subblock', 'match', False)]
    full_states = {}
    for name, mode, refs, feature, size, replay in cases:
        config = dict(refs=refs, mode=mode, cache='off' if feature == 'subblock' else 'conservative',
                      sparse=feature != 'adaptive', size=size, source=source_ref if refs else source_text)
        full = run(name+'-full', **config); record = records[-1]; full_states[name] = full
        if feature != 'subblock':
            check(name+' real cache hit', any(d['decision'] == 'hit' for d in record['decisions']))
            check(name+' final refresh', record['decisions'][-1]['reason'] == 'final')
        if feature != 'adaptive':
            check(name+' sparse execution', sum(t['sparse_calls'] for t in record['trace']) > 0)
        if replay:
            pause = run(name+'-pause', stop=3, **config)
            hidden = out/'hidden-inputs'; assets.rename(hidden)
            try:
                resumed = run(name+'-resume', resume=pause, **config)
            finally:
                hidden.rename(assets)
            before, after, direct = parts(pause), parts(resumed), parts(full)
            for kind in (7, 8, 12, 13, 40, 41, 42):
                if kind in direct:
                    check(name+f' exact state {kind}', after[kind][-1] == direct[kind][-1])
                if kind in (7, 8) and kind in before:
                    check(name+f' fixed conditioning {kind}', before[kind][-1] == after[kind][-1])
            check(name+' exact decisions', records[-1]['decisions'] == record['decisions'][3:])
            for step in range(4, 7):
                check(name+f' resumed step {step}', sha(out/(name+f'-full.step-{step:02d}.bin')) ==
                      sha(out/(name+f'-resume.step-{step:02d}.bin')))
    for name, source in [('text', source_text), ('ref', source_ref)]:
        for sparse in (False, True):
            common = dict(refs=image if name == 'ref' else [], source=source, sparse=sparse)
            dense = run(f'{name}-control-{sparse}', cache='off', **common)
            zero = run(f'{name}-zero-{sparse}', threshold='0', **common)
            a, b = parts(dense), parts(zero)
            for kind in (12, 13):
                check(f'{name} threshold zero {sparse} {kind}', a[kind][-1] == b[kind][-1])
    zero_bridge = run('zero-bridge', mode='bridge', source=source_text, sparse=True,
                      extra={'H3_TEST_BRIDGE_STRENGTH': '0'})
    for kind in (12, 13, 40, 41, 42):
        check(f'zero bridge hard identity {kind}', parts(zero_bridge)[kind][-1] == parts(full_states['hard-combined'])[kind][-1])
    run('context90-linear', source=source_text, mode='bridge', sparse=True, cache='aggressive',
        extra={'H3_TEST_CONTINUE_CONTEXT': '90', 'H3_TEST_BRIDGE_PROFILE': '1'})
    run('context90-ease', refs=image, source=source_ref, mode='bridge', sparse=True,
        extra={'H3_TEST_CONTINUE_CONTEXT': '90', 'H3_TEST_BRIDGE_PROFILE': '2'})
    cancel = run('cancel-retry', source=source_text, sparse=True, stop=3, extra={'H3_TEST_CANCEL_ONCE': '1'})
    for kind in (12, 13, 40, 41, 42):
        check(f'cancel retry {kind}', parts(cancel)[kind][-1] == parts(out/'hard-combined-pause.h3sample')[kind][-1])
    raw = (out/'bridge-combined-pause.h3sample').read_bytes()
    mutations = [(15, 0, 'i', 0), (15, 4, 'i', 90), (15, 120, 'i', 999), (15, 160, 'i', 999),
                 (15, 188+4*4, 'f', float('nan')), (10, 68, 'i', 0), (10, 72, 'i', 999),
                 (40, 4, 'I', 3), (40, 12, 'I', 17), (40, 28, 'Q', 2**63), (40, 36, 'Q', 1)]
    for index, (kind, offset, fmt, value) in enumerate(mutations):
        altered = entries(raw); payload = next(p[-1] for p in altered if p[0] == kind)
        struct.pack_into('<'+fmt, payload, offset, value)
        path = out/f'corrupt-{index}.h3sample'; path.write_bytes(build(altered))
        with (out/f'corrupt-{index}.log').open('w') as log:
            result = subprocess.run([str(root/'bin/sampler_tests'), '--load', str(path)], stdout=log, stderr=subprocess.STDOUT)
        check(f'forged state rejected {index}', result.returncode != 0); path.unlink()
    check('borrowed source files unchanged', all(sha(assets/name) == digest for name, digest in source_hashes.items()))
    publish(True)
    print(f'PASS: {len(records)} model jobs, {len(checks)} checks', flush=True)


if __name__ == '__main__':
    main()
