"""Small shared primitives for the five release stages (private state only)."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import build_process


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write('\n')
    os.replace(f.name, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def identities(paths):
    return {str(Path(p).resolve()): sha(p) for p in sorted(map(Path, paths))}


def verify(files):
    for name, expected in files.items():
        require(Path(name).is_file() and sha(name) == expected, 'Missing or changed recorded file: '+name)


def run(argv, *, log=None, cwd=ROOT, env=None, capture=False, **kwargs):
    argv = list(map(str, argv))
    if len(argv)>1 and Path(argv[0]).name.startswith('python') and argv[1].endswith('.py') and Path(argv[1]).name!='checked.py':
        # Test programs often own subprocesses in separate groups. Let their
        # finally blocks run on cancellation instead of killing Python abruptly.
        argv=[argv[0],str(ROOT/'release/checked.py'),*argv[1:]]
    print('+ '+shlex.join(argv), flush=True)
    if capture:
        return build_process.run(argv, cwd=cwd, env=env, check=True,
                                 stdout=subprocess.PIPE, stderr=kwargs.pop('stderr', subprocess.PIPE),
                                 text=True, **kwargs).stdout.strip()
    if log is None:
        return build_process.run(argv, cwd=cwd, env=env, check=True, **kwargs)
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    stopped = threading.Event()
    def progress():
        while not stopped.wait(30):
            print('  Working; log: '+str(log), flush=True)
    thread = threading.Thread(target=progress, daemon=True)
    thread.start()
    print('  Log: '+str(log), flush=True)
    try:
        with log.open('a') as output:
            output.write('+ '+shlex.join(argv)+'\n'); output.flush()
            return build_process.run(argv, cwd=cwd, env=env, stdout=output,
                                     stderr=subprocess.STDOUT, check=True, **kwargs)
    except BaseException:
        print(log.read_text(errors='replace')[-5000:], file=sys.stderr, flush=True)
        raise
    finally:
        stopped.set(); thread.join()


def receipt(path, inputs, result, files):
    row = dict(schema=1, inputs=digest(inputs), result=result,
               files=identities(files), passed=True, completed=time.time())
    write(path, row)
    return row


def completed(path, inputs):
    path = Path(path)
    if not path.exists():
        return None
    row = read(path)
    if row.get('schema') != 1 or row.get('inputs') != digest(inputs) or row.get('passed') is not True:
        return None
    try:
        verify(row['files'])
    except (ValueError, OSError):
        return None
    return row


def attempt(base, stage):
    base = Path(base); base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=stage+'-', dir=base))


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('H3_', 'H3CLI_', 'LD_', 'DYLD_', 'CUDA_', 'CUDNN_', 'SGLANG_', 'SAGE_'))
           and k not in ('MAKEFLAGS','MFLAGS','CC','CXX','CFLAGS','CXXFLAGS','CPPFLAGS','LDFLAGS','LDLIBS',
                         'CPATH','LIBRARY_PATH','PKG_CONFIG_PATH','NVCC','NVCCFLAGS')}
    return env | {'H3_OFFLINE': '1', 'H3_TEST_MAX_EVALUATIONS': '6'} | extra
