#!/usr/bin/env python3
"""Exercise real sampler loops with deterministic DiT predictions, no weights.

Only the expensive prediction calls are replaced in a test-only source copy.
Euler, callbacks, continuation masks, reuse caches, range/resume, and Metal
preview operations are the production implementation.
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from scalingfix_prepare import FLAGS, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--progress-only', action='store_true', help='Short GPU progress checks without model inference')
    args = parser.parse_args()
    out = ROOT / 'outputs/denoise-validation' / ('sampler-sanitize' if args.sanitize else 'sampler')
    out.mkdir(parents=True, exist_ok=True)
    source = (ROOT / 'src/denoise/dit.c').read_text()
    cpu = 'ok = h3_dit_forward(dit, step, video_latent, audio_latent,\n                                video_velocity, audio_velocity,'
    gpu = 'if (ok) ok = encode_forward(dit, step, 0, 0,'
    assert source.count(cpu) == source.count(gpu) == 1
    source = source.replace(cpu, cpu.replace('h3_dit_forward', 'preview_test_forward'))
    source = source.replace(gpu, gpu.replace('encode_forward', 'preview_test_encode'))
    (out / 'preview_test_dit.c').write_text(source)
    darwin = sys.platform == 'darwin'
    flags = ['-std=c11', '-D_DARWIN_C_SOURCE' if darwin else '-D_GNU_SOURCE', '-g', '-O1' if args.sanitize else '-O3']
    if darwin:
        libraries = [*FLAGS, '-framework', 'CoreML', '-framework', 'CoreVideo', '-lc++']
    else:
        cuda = os.environ.get('CUDA_PATH', '/usr/local/cuda')
        libraries = [f'-L{cuda}/lib64', f'-Wl,-rpath,{cuda}/lib64', '-lcudart', '-lcublas', '-lcublasLt',
                     '-lstdc++', '-licuuc', '-ljson-c', '-lpthread', '-lm']
        if os.environ.get('CUDA_CUDNN') == '1':
            libraries += ['-lcudnn']
        if os.environ.get('CUDA_SAGE') == '1':
            libraries += ['-lcuda']
        if os.environ.get('CUDA_OPENSSL') != '0':
            crypto = subprocess.run(['pkg-config', '--libs', 'libcrypto'], text=True, capture_output=True)
            if crypto.returncode == 0:
                libraries += shlex.split(crypto.stdout)
    if args.sanitize:
        flags += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    binary = ROOT / 'bin' / ('preview_sampler_sanitize' if args.sanitize else 'preview_sampler')
    binary.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['clang' if darwin else 'gcc', *flags, '-I', str(ROOT), '-I', str(out),
                    str(ROOT / 'tests/test_preview_sampler.c'), str(ROOT / 'src/media/preview.c'),
                    str(ROOT / 'src/cli/cli_progress.c'), str(ROOT / 'bin/libh3.a'), *libraries, '-o', str(binary)], check=True)
    with (out / 'run.log').open('w') as log:
        subprocess.run([str(binary), *(['--progress-only'] if args.progress_only else [])],
                       cwd=ROOT, stdout=log, stderr=log, check=True,
                       timeout=60 if args.progress_only else None)
    print((out / 'run.log').read_text(), end='')


if __name__ == '__main__':
    main()
