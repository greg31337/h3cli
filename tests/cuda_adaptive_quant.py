#!/usr/bin/env python3
"""Two fixed three-way comparisons, with one video per row and resident weights."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import fcntl
import json
from pathlib import Path
import re
from cuda_reference_regression import PROMPT, fingerprint, sha, source_files, write
from cuda_adaptive_subblock import render, prepare, model_metadata, runtime


CASES = []
for group, precision in [('F8', 'fp8'), ('F4', 'nvfp4')]:
    for suffix, args in [('D0', []), ('Q', ['--cuda-denoise-quant', precision]),
                         ('A1', ['--cuda-denoise-quant', precision, '--adaptive-cache', 'conservative'])]:
        CASES.append(dict(id=f'{group}-{suffix}', group=precision, baseline=f'{group}-D0', args=args))
IDS = [v['id'] for v in CASES]


def manifest(path):
    value = json.loads(path.read_text())
    assert [value[k] for k in ('schema', 'recipe', 'width', 'height', 'frames', 'steps', 'fps', 'seed', 'timeout_seconds')] == [1, 1, 640, 480, 90, 50, 24, 42, 7200]
    assert value['prompt'] == PROMPT and value['decoder'] == 'full'
    assert value['weight_mode'] == 'resident'
    assert [value[k] for k in ('quantized_execution_recipe', 'adaptive_quant_recipe', 'packing_recipe')] == [3, 2, 2]
    assert value['variants'] == CASES, 'changed fixed six-case comparison'
    return value


def validate_dispatch(variant, steps, cache, log, presentation):
    quant = '--cuda-denoise-quant' in variant['args']
    adaptive = '--adaptive-cache' in variant['args']
    assert [s['step'] for s in steps] == list(range(50))
    assert len(cache) == (50 if adaptive else 0)
    assert all(s['evaluated'] == 1 and s['dense_calls'] == s['blocks'] and s['sparse_calls'] == 0 for s in steps)
    if not adaptive:
        assert all(s['blocks'] == 50 for s in steps)
    else:
        assert [c['step'] for c in cache] == list(range(50))
        assert all(s['blocks'] == (1 if c['decision'] == 'hit' else 50) for s, c in zip(steps, cache))
        assert all(c['decision'] == 'refresh' for c in cache[:4])
        assert cache[-1]['reason'] == 'final' and cache[-1]['decision'] == 'refresh'
        assert all(c['streak'] <= 1 for c in cache)
    expected = [0 if not quant or s['blocks'] == 1 else 196 if adaptive else 200 for s in steps]
    assert [s['quant_calls'] for s in steps] == expected, 'wrong quantized projection dispatch'
    if quant:
        precision = variant['group']
        mode = 1 if precision == 'fp8' else 2
        recipe = 3 if adaptive else 2
        assert f'DiT quantization={precision} recipe={recipe}' in log
        assert 'weights=compressed-resident' in log and 'weights=compressed-stream' not in log
        counters = re.findall(r'projection counters requested=' + precision + r' recipe=(\d+) native_calls=(\d+) cache_hits=(\d+) prepared=(\d+)', log)
        assert counters == [(str(recipe), str(sum(expected)), str(196 if adaptive else 200), '0')], counters
        assert re.search(r'^denoise_quant ' + str(mode) + ' ' + str(recipe) + r' [0-9a-f]{64}$', presentation, re.M)
        if adaptive:
            assert 'block 0 projections BF16, blocks 1-49 ' + precision in log
            assert re.search(r'^adaptive 1 2$', presentation, re.M)
    else:
        assert not re.search(r'^denoise_quant [12] ', presentation, re.M)
    assert 'CUDA weight planner: resident' in log
    assert 'CUDA weight planner: stream' not in log
    return dict(quantized_calls=sum(expected),quantized_projection_matrices=196 if adaptive else 200 if quant else 0,
                quantization_recipe=3 if adaptive else 2 if quant else 0,
                adaptive_recipe=2 if adaptive else 0, weight_mode='resident')


def packed_metadata(cache):
    entries = {p.name: dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
               for p in sorted(cache.glob('m2-*.h3q'))}
    for precision in ('fp8', 'nvfp4'):
        assert sum(name.startswith('m2-' + precision + '-') for name in entries) == 200
    return entries


def main(*, comparison=None):
    # The ledger, timing and artifact rules are shared by both fixed triplets.
    cases = comparison.CASES if comparison else CASES
    ids = [v['id'] for v in cases]
    load_manifest = comparison.manifest if comparison else manifest
    check_dispatch = comparison.validate_dispatch if comparison else validate_dispatch
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'model', 'manifest', 'out', 'cache', 'preparation'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    source, model, out, cache = (getattr(args, n).resolve() for n in ('source', 'model', 'out', 'cache'))
    config = load_manifest(args.manifest)
    preparation = json.loads(args.preparation.read_text())
    assert preparation['passed'], 'hardware qualification and cache preparation must pass first'
    prepared = [r for r in preparation['records'] if r['name'] in ('prepare-fp8', 'prepare-nvfp4')]
    assert len(prepared) == 2 and all(r['passed'] for r in prepared)
    out.mkdir(parents=True, exist_ok=args.resume)
    with (out / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = dict(os.environ)
        env.update(H3_EXPERIMENT_TRACE='1', H3_EXPERIMENT_TIMING='1', H3_CUDA_WEIGHT_MODE='resident', H3_QUANT_VERIFY='0')
        assert not any(k.startswith('H3_TEST_') and k != 'H3_TEST_MAX_EVALUATIONS' for k in env)
        assert not any(k.startswith('H3_PROFILE') for k in env)
        assert env.get('H3_QUANT_DIAGNOSTICS', '0') == '0'
        env.pop('H3_TEST_MAX_EVALUATIONS', None)
        metadata = model_metadata(model)
        packed = packed_metadata(cache)
        identity = dict(schema=1, source_sha256=fingerprint(source_files(source)), binary_sha256=sha(source/'bin/h3cli'),
                        manifest_sha256=sha(args.manifest), model_metadata=metadata, runtime=runtime(env),
                        packed_weight_metadata=packed, packed_cache=str(cache), cache_preparation=prepared,
                        preparation_record_sha256=sha(args.preparation))
        frozen = out/'identity.json'
        if frozen.exists():
            assert json.loads(frozen.read_text()) == identity, 'source/build/runtime/model/cache changed'
        else:
            write(frozen, identity)
        ledger = []
        for variant in cases:
            directory = out/variant['id']
            directory.mkdir(exist_ok=True)
            attempts = sorted(directory.glob('attempt-*'))
            existing = [(d, json.loads((d/'result.json').read_text())) for d in attempts]
            successes = [(d, r) for d, r in existing if r['passed']]
            assert len(successes) <= 1
            if successes:
                attempt, result = successes[0]
                for name, digest in result['artifacts'].items():
                    assert sha(attempt/name) == digest
            else:
                assert not any(r.get('returncode') == 0 for _, r in existing), 'inspect completed output; never silently render it again'
                prep = prepare(model, metadata)
                assert packed_metadata(cache) == packed
                attempt = directory/f'attempt-{len(attempts)+1:03d}'
                effective = dict(variant, args=variant['args'] + (['--cuda-denoise-quant-cache', str(cache)] if variant['args'] else []))
                try:
                    result = render(source, attempt, model, config, effective, env)
                    steps = json.loads((attempt/'steps.json').read_text())
                    history = json.loads((attempt/'cache.json').read_text())
                    result['mixed_precision'] = check_dispatch(variant, steps, history, (attempt/'render.log').read_text(),
                                                                  (attempt/'final.h3av.presentation').read_text())
                    result.update(preparation=prep, manifest_variant=variant)
                    write(attempt/'result.json', result)
                except BaseException as error:
                    if (attempt/'result.json').exists():
                        failed = json.loads((attempt/'result.json').read_text())
                        failed.update(passed=False, validation_error=str(error))
                        write(attempt/'result.json', failed)
                    ledger.append(dict(id=variant['id'], passed=False, artifact_directory=str(attempt.relative_to(out)), error=str(error)))
                    write(out/'ledger.json', dict(complete=False, planned=ids, cases=ledger))
                    raise
            assert fingerprint(source_files(source)) == identity['source_sha256']
            assert sha(source/'bin/h3cli') == identity['binary_sha256'] and packed_metadata(cache) == packed
            ledger.append(dict(id=variant['id'], passed=True, artifact_directory=str(attempt.relative_to(out)),
                               attempt_count=len(list(directory.glob('attempt-*'))), baseline=variant['baseline'], group=variant['group']))
            write(out/'ledger.json', dict(complete=len(ledger) == 6, planned=ids, cases=ledger))
            print(json.dumps(dict(id=variant['id'], wall_seconds=result['wall_seconds'], counts=result['counts'],
                                  mixed_precision=result['mixed_precision'])), flush=True)
        assert [r['id'] for r in ledger] == ids


if __name__ == '__main__':
    main()
