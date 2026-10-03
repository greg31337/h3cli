#!/usr/bin/env python3
"""Run pinned OCI filesystems on providers that prohibit nested namespaces.

PRoot isolates filesystem resolution, not hostile code. Only trusted build/test
inputs belong here. The distributed executable itself does not use this tool.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE/'lock.json').read_text())

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def fetch(url, path, sha):
    path = Path(path)
    if path.is_file():
        if digest(path) != sha:
            raise ValueError(f'Cached download checksum mismatch: {path}')
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.partial')
    try:
        with urllib.request.urlopen(url, timeout=120) as src, temporary.open('wb') as dst:
            shutil.copyfileobj(src, dst)
        if digest(temporary) != sha:
            raise ValueError(f'Download checksum mismatch: {url}')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path

def prepare(cache, name, offline=False):
    image = LOCK['images'][name]
    oci = cache/'images'/name
    if not (oci/'index.json').is_file():
        if offline:raise ValueError('Missing offline OCI image: '+str(oci))
        oci.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['skopeo', '--override-arch', 'amd64', 'copy',
                        'docker://'+image, 'oci:'+str(oci)+':base'], check=True)
        (oci/'origin.json').write_text(json.dumps({'image':image})+'\n')
    origin=oci/'origin.json'
    if not origin.is_file() or json.loads(origin.read_text()).get('image')!=image:
        raise ValueError('OCI cache has no matching locked origin; fetch into a fresh cache: '+str(oci))
    for blob in (oci/'blobs/sha256').iterdir():
        if not blob.is_file() or digest(blob)!=blob.name:raise ValueError('Corrupt OCI blob: '+str(blob))
    bundle = cache/'roots'/name
    if not (bundle/'rootfs').is_dir():
        bundle.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['umoci', 'unpack', '--rootless', '--image',
                        str(oci)+':base', str(bundle)], check=True)
    return bundle/'rootfs'

def driver_bindings():
    # Only NVIDIA driver objects. Never bind the host toolkit or library tree.
    found = {}
    for folder in ('/usr/lib/x86_64-linux-gnu', '/usr/lib/x86_64-linux-gnu/nvidia/current'):
        p = Path(folder)
        if p.is_dir():
            for path in sorted(p.iterdir()):
                if path.name.startswith(('libcuda.so', 'libnvidia-', 'libcudadebugger.so')) and path.is_file():
                    found['/usr/lib/x86_64-linux-gnu/'+path.name] = str(path.resolve())
    if not any(p.endswith('/libcuda.so.1') for p in found):
        raise ValueError('No host libcuda.so.1 found; expose the NVIDIA driver first')
    if shutil.which('nvidia-smi'):
        found['/usr/bin/nvidia-smi'] = str(Path(shutil.which('nvidia-smi')).resolve())
    return [source+':'+dest for dest, source in sorted(found.items())]

def command(cache, root, bindings=(), gpu=False, cwd='/', identity='0'):
    record = LOCK['proot']
    proot = fetch(record['url'], cache/'tools/proot', record['sha256'])
    proot.chmod(0o755)
    args = [str(proot), '-i', identity, '-r', str(root), '-b', '/dev', '-b', '/proc',
            '-b', '/sys', '-b', '/etc/resolv.conf', '-w', cwd]
    for binding in list(bindings)+(driver_bindings() if gpu else []):
        args += ['-b', binding]
    return args

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--image', choices=LOCK['images'], default='builder')
    p.add_argument('--root', type=Path, help='Existing writable extracted root instead of the cached image')
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--bind', action='append', default=[])
    p.add_argument('--cwd', default='/')
    p.add_argument('--identity', default='0')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    args = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not args: p.error('a command after -- is required')
    cache = a.cache.resolve()
    root = a.root.resolve() if a.root else prepare(cache, a.image)
    env = {'PATH':'/usr/sbin:/usr/bin:/sbin:/bin', 'HOME':'/root', 'LANG':'C.UTF-8',
           'TERM':'dumb', 'SOURCE_DATE_EPOCH':str(LOCK['source_date_epoch'])}
    return subprocess.call(command(cache, root, a.bind, a.gpu, a.cwd, a.identity)+args, env=env)

if __name__ == '__main__':
    try: raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f'Linux environment: {exc}')
