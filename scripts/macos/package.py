"""Signature-covered macOS archive construction and dependency auditing."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import zlib

ROOT = Path(__file__).resolve().parents[2]
MAGIC = b'H3CLI_MACOS_V1\0\0\0'
SYSTEM = ('/usr/lib/', '/System/Library/Frameworks/')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')


def sign(path, identity='-', identifier=None):
    command = ['/usr/bin/codesign', '--force', '--sign', identity, '--options', 'runtime',
               '--timestamp=none' if identity == '-' else '--timestamp',
               '--identifier', identifier or 'org.h3cli.'+Path(path).name, str(path)]
    subprocess.run(command, check=True)
    subprocess.run(['/usr/bin/codesign', '--verify', '--strict', str(path)], check=True)


def audit(root):
    result = {}
    root = Path(root)
    for p in sorted(root.rglob('*')):
        if p.is_symlink(): raise ValueError('Runtime symlinks are forbidden: '+str(p))
        if not p.is_file(): continue
        with p.open('rb') as f: magic = f.read(4)
        if magic != b'\xcf\xfa\xed\xfe':
            if p.stat().st_mode & 0o111: raise ValueError('Non-Mach-O executable: '+str(p))
            continue
        arch = subprocess.check_output(['/usr/bin/lipo', '-archs', str(p)], text=True).strip()
        if arch != 'arm64': raise ValueError('Non-arm64 runtime member: '+str(p))
        commands = subprocess.check_output(['/usr/bin/otool', '-l', str(p)], text=True)
        minos = re.findall(r'^\s+minos (\S+)', commands, re.M)
        if minos != ['26.0']: raise ValueError('Unexpected minimum OS: '+str(p)+' '+str(minos))
        if 'LC_RPATH' in commands: raise ValueError('Unexpected runtime search path: '+str(p))
        loads = re.findall(r'^\s+name (.+?) \(offset \d+\)', commands, re.M)
        for load in loads:
            if not load.startswith(SYSTEM[:2]): raise ValueError('Non-system dynamic dependency: '+load)
        result[str(p.relative_to(root))] = {'sha256':sha(p), 'architecture':arch, 'minos':minos[0], 'loads':loads}
    if not result: raise ValueError('No Mach-O files to audit')
    return result


def inventory(root):
    result = {}
    for p in sorted(Path(root).rglob('*')):
        if p.is_symlink(): raise ValueError('Symlink in runtime: '+str(p))
        if not p.is_file(): continue
        name = p.relative_to(root).as_posix()
        if name in ('.ready.json', 'share/h3cli/runtime.json'): continue
        mode = 0o755 if p.stat().st_mode & 0o111 else 0o644
        p.chmod(mode)
        result[name] = {'size':p.stat().st_size, 'sha256':sha(p), 'mode':mode}
    return result


def seal(root):
    root = Path(root)
    manifest = json.loads((root/'share/h3cli/runtime.json').read_text())
    files = inventory(root)
    if files != manifest['files']: raise ValueError('Runtime inventory differs from manifest')
    stamps = {}
    for p in sorted(root.rglob('*')):
        if p.is_dir(): p.chmod(0o700)
        if not p.is_file() or p.name == '.ready.json': continue
        st = p.stat()
        stamps[p.relative_to(root).as_posix()] = [st.st_size,st.st_mode & 0o7777,st.st_ino,st.st_dev,
                st.st_mtime_ns//10**9,st.st_mtime_ns%10**9,st.st_ctime_ns//10**9,st.st_ctime_ns%10**9]
    root.chmod(0o700)
    # The direct runtime uses the same core readiness checks as extraction.
    write(root/'.ready.json', {'manifest':{}, 'files':stamps})
    (root/'.ready.json').chmod(0o600)


def payload(root, destination, source=ROOT):
    root = Path(root)
    runtime = json.loads((root/'share/h3cli/runtime.json').read_text())
    if inventory(root) != runtime['files']: raise ValueError('Unsealed runtime contents')
    data = bytearray(); records = []; total = 0
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.name == '.ready.json': continue
        name = p.relative_to(root).as_posix()
        raw = p.read_bytes(); packed = zlib.compress(raw, 9)
        records.append({'path':name, 'size':len(raw), 'compressed_size':len(packed),
                        'offset':len(data), 'mode':0o755 if p.stat().st_mode & 0o111 else 0o644,
                        'sha256':hashlib.sha256(raw).hexdigest()})
        data.extend(packed); total += len(raw)
    launcher = hashlib.sha256()
    for name in ('launcher_macos.m','macos.m','macos.h'):
        launcher.update((Path(source)/'src/runtime'/name).read_bytes())
    manifest = {'schema':1, 'protocol':1, 'platform':'macos-arm64',
                'launcher_source_sha256':launcher.hexdigest(), 'files':records}
    encoded = canonical(manifest)
    mh, ph = hashlib.sha256(encoded).digest(), hashlib.sha256(data).digest()
    Path(destination).write_bytes(struct.pack('<16sQQQ32s32s',MAGIC,len(encoded),len(data),total,mh,ph)+encoded+data)
    return {'runtime_id':hashlib.sha256(mh+ph).hexdigest(), 'compressed_bytes':len(data),
            'expanded_bytes':total, 'manifest':manifest, 'payload_sha256':sha(destination)}


def embed(source, archive, output, env=None):
    command = ['xcrun','clang','-I.','-arch','arm64','-mmacosx-version-min=26.0','-O2','-fobjc-arc',
        '-Wall','-Wextra','-Wpedantic','-Wshadow','-Wconversion','-Wno-sign-conversion','-Werror',
        'src/runtime/launcher_macos.m','src/runtime/macos.m','-framework','Foundation','-lz',
        '-Wl,-sectcreate,__H3CLI,__payload,'+str(Path(archive).resolve()),
        '-Wl,-segprot,__H3CLI,r,r','-o',str(Path(output).resolve())]
    subprocess.run(command,cwd=source,env=env,check=True)


def dmg(output, executable, label='h3cli development (unnotarized)'):
    output = Path(output)
    staging = output/'dmg-contents';staging.mkdir()
    shutil.copy2(executable, staging/'h3cli-macos-arm64')
    (staging/'READ ME.txt').write_text(label+'\nApple Silicon, macOS 26+. Copy h3cli-macos-arm64 to a writable location and run it in Terminal.\nNo PATH changes or models are installed.\nSee macos-distribution.md and the accompanying corresponding-source archive.\n')
    shutil.copy2(ROOT/'docs/build/macos-distribution.md', staging/'macos-distribution.md')
    image = output/'h3cli-macos-arm64.dmg'
    subprocess.run(['/usr/bin/hdiutil','create','-quiet','-format','UDZO','-volname','h3cli',
                    '-srcfolder',str(staging),str(image)],check=True)
    shutil.rmtree(staging)
    return image
