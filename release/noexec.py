#!/usr/bin/env python3
"""Check noexec-cache diagnostics with the exact launcher from a final Linux asset."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def scratch(candidates=('/dev/shm','/run/lock')):
    for name in candidates:
        path=Path(name)
        if path.is_dir() and os.access(path,os.W_OK) and os.statvfs(path).f_flag & getattr(os,'ST_NOEXEC',8):
            return path
    raise ValueError('Required launcher test needs writable noexec scratch (/dev/shm or /run/lock)')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','package-module','report'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    args=parser.parse_args(argv)
    if sha(args.binary)!=args.sha256:raise ValueError('Final Linux executable changed before launcher test')
    noexec=scratch();args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.binary.open('rb') as source:
        source.seek(-128,2);footer=struct.unpack('<16sQQQQ32s32sQQ',source.read(128))
        if footer[0].rstrip(b'\0')!=b'H3CLI_BUNDLE_V1' or not 0<footer[3]<args.binary.stat().st_size-128:
            raise ValueError('Invalid final Linux bundle footer')
        source.seek(0);launcher=source.read(footer[3])
    spec=importlib.util.spec_from_file_location('noexec_bundle_writer',args.package_module)
    package=importlib.util.module_from_spec(spec);spec.loader.exec_module(package)
    # Only the payload is replaced with a tiny host fixture. The launcher bytes
    # are extracted unchanged from the final executable; no compiler is used.
    with tempfile.TemporaryDirectory(prefix='launcher-check-',dir=args.report.parent) as folder:
        work=Path(folder);stub=work/'launcher';stub.write_bytes(launcher);stub.chmod(0o755)
        runtime=work/'runtime';(runtime/'bin').mkdir(parents=True)
        shutil.copy2('/usr/bin/true',runtime/'bin/h3cli')
        probe=work/'probe';package.package(runtime,stub,probe)
        env={k:v for k,v in os.environ.items() if not k.startswith(('H3CLI_','LD_','DYLD_'))}
        normal=subprocess.run([str(probe)],env=env|{'H3CLI_RUNTIME_CACHE':str(work/'cache')},capture_output=True,text=True,timeout=30)
        if normal.returncode:raise ValueError('Launcher fixture failed on executable storage: '+normal.stderr)
        with tempfile.TemporaryDirectory(prefix='h3-release-noexec-',dir=noexec) as cache:
            rejected=subprocess.run([str(probe)],env=env|{'H3CLI_RUNTIME_CACHE':cache},capture_output=True,text=True,timeout=30)
        if rejected.returncode==0 or 'check cache mount permits execution' not in rejected.stderr:
            raise ValueError('Launcher did not diagnose a noexec cache: '+rejected.stderr)
    if sha(args.binary)!=args.sha256:raise ValueError('Final Linux executable changed during launcher test')
    args.report.write_text(json.dumps(dict(passed=True,binary_sha256=args.sha256,
        launcher_sha256=hashlib.sha256(launcher).hexdigest(),script_sha256=sha(__file__),
        noexec_mount=str(noexec),exit_code=rejected.returncode),sort_keys=True,indent=2)+'\n')
    print('PASS final Linux launcher noexec-cache diagnostics')


if __name__=='__main__':main()
