#!/usr/bin/env python3
"""Internal native build/test worker; Linux runs in the qualified toolchain root."""
import argparse
import os
from pathlib import Path
import re
import shutil
import shlex
import subprocess
import sys

from common import ROOT, read, write, run, require, sha, identities, verify
sys.path.insert(0, str(ROOT/'tests'))
import cuda_reference_regression as gate
import current_cuda_suite as current


def targets(make):
    # Ask make for its expanded dependency graph, without executing recipes.
    p = subprocess.run([*make, '-qp', 'test', 'test-server-http'], cwd=ROOT,
                       capture_output=True, text=True)
    require(p.returncode in (0, 1), 'Cannot inspect native test prerequisites: '+p.stderr)
    graph = {}
    for line in p.stdout.splitlines():
        m = re.match(r'^([\w./-]+):\s+(.*)$', line)
        if m:
            graph.setdefault(m[1], set()).update(m[2].split())
    seen = set()
    def visit(name):
        if name in seen: return
        seen.add(name)
        for child in graph.get(name, ()):
            visit(child)
    for name in ('test', 'test-server-http'): visit(name)
    return sorted(n for n in seen if n.startswith('bin/') and n != 'bin')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('build', 'test'))
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); job = read(a.job); out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    model = Path(job['model']); models = Path(job['models'])
    (ROOT/'models').mkdir(exist_ok=True)
    for name, target in [('MiniMax-H3', model), ('MiniMaxH3', model),
                         *[(n, models/n) for n in ('preview-vae','image-vae','latent-upscale')]]:
        link = ROOT/'models'/name
        if not link.exists(): link.symlink_to(target, target_is_directory=True)
    os.environ.update(H3_MODEL_DIR=str(model), H3_MODELS_PATH=str(models), H3_CPU_SAMPLER='1')
    make = ['make', '-j'+str(job['jobs']), 'H3_GIT_COMMIT='+job['commit']]
    if job['target']=='linux':
        arch=run(['nvidia-smi','-i','0','--query-gpu=compute_cap','--format=csv,noheader'],capture=True).replace('.','').strip()
        require(arch in ('86','89','90','100','120'),'Unsupported native CUDA architecture: '+arch)
        make.append('CUDA_ARCH='+arch)
    if a.action == 'build':
        bins = set(targets(make)) | {'bin/h3cli', 'bin/libh3.a'}
        if job['target'] == 'linux': bins.update(gate.BUILD + current.TARGETS)
        run([*make, *sorted(bins)], log=out/'compile.log')
        binaries = {str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'bin').iterdir() if f.is_file()}
        write(out/'native.json', dict(schema=1, source_sha256=gate.fingerprint(gate.source_files(ROOT)),
              binaries=binaries, make=make))
        return
    manifest = Path(job['native_manifest'])
    row = read(manifest)
    require(row['source_sha256'] == gate.fingerprint(gate.source_files(ROOT)), 'Native source changed')
    files = {str(ROOT/n):v for n,v in row['binaries'].items()}
    verify(files)
    tools=out/'tools';tools.mkdir()
    wrapper=tools/'python3'
    interpreter=shlex.quote(sys.executable); checked=shlex.quote(str(ROOT/'release/checked.py'))
    wrapper.write_text('#!/bin/sh\ncase "${1:-}" in *.py) exec '+interpreter+' '+checked+' "$@" ;; *) exec '+interpreter+' "$@" ;; esac\n')
    wrapper.chmod(0o755)
    os.environ['PATH']=str(tools)+os.pathsep+os.environ['PATH']
    # Every executable is already compiled. Mark recorded binaries old so test
    # recipes cannot silently replace them, then verify all of them afterwards.
    frozen = [arg for n in row['binaries'] for arg in ('-o', n)]
    run([*row['make'], *frozen, 'test', 'test-server-http'], log=out/'host.log')
    run([sys.executable, '-m', 'unittest', 'discover', '-s', 'release/tests'], log=out/'release.log')
    run([sys.executable, 'tests/test_render_script.py'], log=out/'render-script.log')
    # This separate result fails on skipped cloning tests on either platform.
    run([sys.executable, 'tests/test_lora_runtime.py'], log=out/'lora.log')
    require('skipped=' not in (out/'lora.log').read_text(), 'Required LoRA tests skipped; configure a cloning-capable work/cache/models volume')
    if job['target'] == 'linux':
        run([sys.executable, 'tests/cuda_reference_regression.py', '--source', ROOT,
             '--native-manifest', manifest, '--model', model, '--out', out/'parity'], log=out/'parity.log')
        require(read(out/'parity/result.json')['passed'] and len(read(out/'parity/result.json')['files']) == 204,
                'The complete 204-output native CUDA gate did not pass')
    features = [sys.executable, 'tests/model_downloads_render.py', '--binary', ROOT/'bin/h3cli',
                '--backend', 'cuda' if job['target']=='linux' else 'metal', '--models-path', models,
                '--model', model, '--ffmpeg', os.environ['H3_FFMPEG'], '--ffprobe', os.environ['H3_FFPROBE'],
                '--server', '--out', out/'features']
    run(features, log=out/'features.log')
    require(read(out/'features/results.json').get('passed') is True, 'Native feature tests did not pass')
    if job['target'] == 'linux':
        run([sys.executable, 'tests/current_cuda_suite.py', '--native-manifest', manifest,
             '--out', out/'current', '--state', out/'features/base.h3av', '--model', model,
             '--cow-directory', os.environ['TMPDIR']], log=out/'current.log')
        require(read(out/'current/result.json').get('passed') is True, 'Current CUDA suite did not pass')
    verify(files)
    require(row['source_sha256'] == gate.fingerprint(gate.source_files(ROOT)), 'Source changed during tests')
    write(out/'result.json', dict(passed=True, binary_sha256=sha(ROOT/'bin/h3cli'),
                                 native_manifest_sha256=sha(manifest), source_sha256=row['source_sha256']))


if __name__ == '__main__':
    main()
