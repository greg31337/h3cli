#!/usr/bin/env python3
"""Supplement make test with current CUDA feature/lifecycle checks.

Uses newly generated current AV state and real installed models. Historical
campaigns and external/live numerical oracles are not part of this suite.
"""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

TARGETS = ['bin/cuda_resume_test','bin/continuation_generate','bin/cuda_single_matrix',
           'bin/attention_failures','bin/cuda_sol_context','bin/subblock_context',
           'bin/conditioning_context','bin/weight_residency_context','bin/full_vae_tests',
           'bin/fast_vae_lifetime','bin/fast_vae_delivery','bin/cuda_policy_context',
           'bin/still_tests','bin/cuda_sglang_patch_test','bin/cuda_memory_test','bin/lora_host']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path(os.environ.get('H3_MODEL_DIR', 'models/MiniMax-H3')))
    p.add_argument('--cow-directory', type=Path, required=True,
                   help='Disposable scratch directory on a filesystem with reflink support')
    p.add_argument('--prebuilt',type=Path,help='Verified feature-probe manifest; never rebuild in this mode')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out, model, state = a.out.resolve(), a.model.resolve(), a.state.resolve()
    out.mkdir(parents=True, exist_ok=False)
    a.cow_directory.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    env = dict(os.environ, H3_TEST_MAX_EVALUATIONS='6', H3_MODEL_DIR=str(model),
               H3_CUDA_OUTPUT=str(out/'integration'), H3_TEST_REFERENCE_WORKERS='1')
    for key in ('H3_TEST_REFERENCE_MEDIA','H3_TEST_REFERENCE_MEDIA_SHA256'):
        env.pop(key,None)
    records = []
    binary_hash = hashlib.sha256((root/'bin/h3cli').read_bytes()).hexdigest()

    def run(name, command, extra=None):
        start = time.monotonic()
        argv = list(map(str, command))
        with (out/(name+'.log')).open('w') as log:
            try:
                proc = subprocess.run(argv, env=env | (extra or {}), stdout=log,
                                      stderr=subprocess.STDOUT, timeout=1800)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                rc = 124
        if name == 'cow-lora-runtime' and 'skipped=' in (out/(name+'.log')).read_text():
            rc = 1
        record = dict(name=name, argv=argv, overrides=extra or {}, returncode=rc,
                      seconds=time.monotonic()-start, passed=rc==0)
        records.append(record)
        (out/'result.json').write_text(json.dumps(dict(complete=False, passed=False,
            binary_sha256=binary_hash, checks=records), indent=2)+'\n')
        print(name, rc, round(record['seconds'],2), flush=True)
        return rc==0

    prebuilt=None
    if a.prebuilt:
        import cuda_reference_regression as gate
        prebuilt=json.loads(a.prebuilt.read_text())
        if prebuilt.get('schema')!=1 or prebuilt.get('source_sha256')!=gate.fingerprint(gate.source_files(root)) or set(prebuilt.get('binaries',{}))!=set(TARGETS+['bin/h3cli']):
            raise ValueError('Prebuilt feature probes do not match the complete source/target set')
        gate.checked_files(root,prebuilt['binaries'])
        runtime=Path(os.environ['H3CLI_RUNTIME_ROOT']).resolve()
        artifact=gate.artifact_config(root,runtime,a.prebuilt.resolve().parent,root/'bin/h3cli',reference=False)
        env.update(artifact['_runtime_env'])
        env['PATH']=str(runtime/'tools')+os.pathsep+env.get('PATH','/usr/bin:/bin')
    elif not run('build', ['make','-j4',*TARGETS]):return 1
    run('cow-lora-runtime', ['python3','tests/test_lora_runtime.py'],
        {'TMPDIR':str(a.cow_directory.resolve())})
    run('bridge-cli', ['python3','tests/bridge_cli.py','--model',model,
                       '--state',state,'--output',out/'bridge-cli.json'])
    run('feature-integration', ['python3','tests/cuda_integration.py','features'])
    run('option-matrix', ['bin/cuda_single_matrix',out/'quant-cache'])
    run('attention-failures', ['bin/attention_failures'])
    run('sol-context', ['bin/cuda_sol_context'])
    run('subblock-context', ['bin/subblock_context'])
    run('conditioning-context', ['bin/conditioning_context',model,out/'conditioning'])
    run('subblock-conditioning-context', ['bin/conditioning_context',model,out/'subblock-conditioning'],
        {'H3_TEST_SUBBLOCK_CONTEXT':'1'})
    (out/'residency').mkdir()
    run('residency-context', ['bin/weight_residency_context','context',model,out/'residency'])
    run('full-vae-gpu', ['bin/full_vae_tests','gpu'])
    run('still-gpu', ['bin/still_tests','gpu'])
    run('patch-operators', ['bin/cuda_sglang_patch_test'])
    run('cuda-memory', ['bin/cuda_memory_test'])
    cropped = out/'current-cropped.h3av'
    if run('current-state-crop', ['bin/cuda_policy_context',state,cropped]):
        run('vae-lifetime', ['bin/fast_vae_lifetime',model/'FL2VA/video_vae/source',cropped])
        run('decode-failure-recovery', ['bin/fast_vae_delivery',model,cropped])
    run('policy-context', ['bin/cuda_policy_context'])
    passed = all(r['passed'] for r in records)
    if prebuilt:
        gate.checked_files(root,prebuilt['binaries'])
        gate.verify_artifact_identities(artifact['_identities'])
        if prebuilt['source_sha256']!=gate.fingerprint(gate.source_files(root)):raise ValueError('Source changed during prebuilt qualification')
    (out/'result.json').write_text(json.dumps(dict(complete=True, passed=passed,
        binary_sha256=binary_hash, checks=records), indent=2)+'\n')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
