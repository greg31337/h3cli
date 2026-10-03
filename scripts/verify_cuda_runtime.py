#!/usr/bin/env python3
"""Check the pinned native runtime without importing Torch or model weights."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


# Probe in a subprocess so loader search paths match the generated environment.
PROBE = r'''
import ctypes as c, json, os, sys
kind, path, gpu = sys.argv[1:]
lib = c.CDLL(path, mode=os.RTLD_LOCAL | os.RTLD_NOW | os.RTLD_DEEPBIND)
if kind == 'cudnn':
    lib.cudnnGetVersion.restype = c.c_size_t
    result = {'version': lib.cudnnGetVersion()}
elif kind == 'cublas':
    parts = []
    for field in range(3):
        value = c.c_int()
        status = lib.cublasGetProperty(field, c.byref(value))
        if status: raise RuntimeError('cublasGetProperty status ' + str(status))
        parts.append(value.value)
    result = {'version': parts}
    if gpu == '1':
        handle = c.c_void_p()
        status = lib.cublasCreate_v2(c.byref(handle))
        if status: raise RuntimeError('cuBLAS 13 GPU initialization status ' + str(status))
        version = c.c_int()
        try:
            status = lib.cublasGetVersion_v2(handle, c.byref(version))
            if status or version.value != 130101:
                raise RuntimeError('Unexpected GPU cuBLAS identity: ' + str(version.value))
        finally: lib.cublasDestroy_v2(handle)
        result['gpu_version'] = version.value
elif kind == 'nvrtc':
    major, minor = c.c_int(), c.c_int()
    status = lib.nvrtcVersion(c.byref(major), c.byref(minor))
    if status: raise RuntimeError('nvrtcVersion status ' + str(status))
    result = {'version': [major.value, minor.value]}
else:
    for symbol in ('tjInitDecompress', 'tjDecompressHeader3', 'tjDecompress2', 'tjDestroy'):
        getattr(lib, symbol)
    result = {'loaded': True}
print(json.dumps(result))
'''


def require_file(name):
    value = os.environ.get(name, '')
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f'{name} must point to an existing absolute file: {value}')
    return path


def probe(kind, path, expected=None, gpu=False):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Missing runtime library: {path}')
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(path.parent) + ':' + env.get('LD_LIBRARY_PATH', '')
    result = subprocess.run([sys.executable, '-c', PROBE, kind, str(path), str(int(gpu))],
                            env=env, text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise ValueError(f'Cannot use {path}: {result.stderr.strip()}')
    record = json.loads(result.stdout)
    if expected is not None and record.get('version') != expected:
        raise ValueError(f'{path}: expected {expected}, found {record.get("version")}')
    return dict(path=str(path), **record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    try:
        blas = require_file('H3_SGLANG_CUBLAS_LIBRARY')
        cudnn = require_file('H3_SGLANG_CUDNN_LIBRARY')
        record = {
            'cudnn': probe('cudnn', cudnn, 92000),
            'cublas': probe('cublas', blas, [13, 1, 1], args.gpu),
            'toolkit_cublas': probe('cublas', Path(os.environ['CUDA_PATH']) / 'lib64/libcublas.so.13', [13, 1, 1]),
            'nvrtc': probe('nvrtc', blas.parent / 'libnvrtc.so.13', [13, 0]),
            'jpeg': probe('jpeg', require_file('H3_SGLANG_JPEG_LIBRARY')),
            'gpu_initialized': args.gpu,
        }
        for stem in ('graph', 'ops', 'cnn', 'adv', 'heuristic',
                     'engines_precompiled', 'engines_runtime_compiled'):
            if not (cudnn.parent / f'libcudnn_{stem}.so.9').is_file():
                raise ValueError(f'Missing cuDNN component: {stem}')
        for name, version in (('H3_FFMPEG', 'ffmpeg version 9.0.2'),
                              ('H3_SGLANG_INPUT_FFMPEG', 'ffmpeg version 9.0.2'),
                              ('H3_FFPROBE', 'ffprobe version 9.0.2')):
            path = require_file(name)
            output = subprocess.check_output([str(path), '-version'], text=True).splitlines()[0]
            if not output.startswith(version):
                raise ValueError(f'{name}: expected {version}, found {output}')
            record[name] = dict(path=str(path), version=output)
        record['passed'] = True
        args.out.write_text(json.dumps(record, indent=2) + '\n')
        print('Verified pinned CUDA/cuDNN/media runtime' + (' and cuBLAS 13 GPU access.' if args.gpu else ' (GPU not checked).'))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f'CUDA runtime verification failed: {exc}\n')


if __name__ == '__main__':
    main()
