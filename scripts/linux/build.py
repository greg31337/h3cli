#!/usr/bin/env python3
"""Build the portable CUDA release in a locked Linux x86-64 userland.

Host prerequisites: Python 3.10+, skopeo, umoci, GNU cp and ptrace permission.
The builder uses PRoot because GPU providers often prohibit nested namespaces.
It does not install a toolkit, compiler or libraries into the host. PRoot is
not a security sandbox: only build trusted source. No GPU is needed to build.
"""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request

import environment
import package

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[1]
LOCK=environment.LOCK


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def wheels():
    text='\n'.join((SOURCE/'scripts'/name).read_text() for name in ('requirements-sglang-runtime.txt','requirements-reference-media.txt')).replace('\\\n',' ')
    pins=[]
    for line in text.splitlines():
        line=line.strip()
        if not line or line.startswith('#'):continue
        match=re.fullmatch(r'([A-Za-z0-9_-]+)==([^\s]+)\s+--hash=sha256:([0-9a-f]{64})',line)
        if not match:raise ValueError('Unsupported native dependency pin: '+line)
        pins.append(match.groups())
    if not pins:raise ValueError('Native wheel requirements are empty')
    return pins


def cached(url,path,sha,offline):
    if offline and not path.is_file():raise ValueError('Missing offline input: '+str(path))
    return environment.fetch(url,path,sha)


def fetch(cache,offline):
    inputs=cache/'inputs';inputs.mkdir(parents=True,exist_ok=True)
    pending=[]
    for name,record in (LOCK['sources']|LOCK.get('materials',{})).items():
        pending.append((record['url'],inputs/name,record['sha256']))
    for record in LOCK['debs']:
        pending.append((record['url'].replace('http:','https:',1),inputs/'debs'/record['name'],record['sha256']))
    for record in LOCK['source_packages'].values():
        for artifact in record['artifacts']:
            pending.append((artifact['url'],inputs/'source-packages'/artifact['name'],artifact['sha256']))
    metadata=inputs/'wheels.json'
    known=json.loads(metadata.read_text()) if metadata.is_file() else {}
    for name,version,sha in wheels():
        key=name+'=='+version
        if key not in known:
            if offline:raise ValueError('Missing offline wheel metadata: '+key)
            with urllib.request.urlopen('https://pypi.org/pypi/'+name+'/'+version+'/json',timeout=120) as response:
                candidates=json.load(response)['urls']
            matches=[r for r in candidates if r['digests']['sha256']==sha]
            if len(matches)!=1:raise ValueError('Pinned wheel not found: '+key)
            known[key]={'name':matches[0]['filename'],'url':matches[0]['url'],'sha256':sha}
        record=known[key]
        if record['sha256']!=sha:raise ValueError('Stale wheel metadata: '+key)
        pending.append((record['url'],inputs/'wheels'/record['name'],sha))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(cached,url,path,sha,offline) for url,path,sha in pending]
        for future in futures:future.result()
    write(metadata,known)
    proot=LOCK['proot'];cached(proot['url'],cache/'tools/proot',proot['sha256'],offline)
    environment.prepare(cache,'builder',offline=offline)
    return inputs


def snapshot(source,destination):
    try:
        names=subprocess.check_output(['git','-C',str(source),'ls-files','--cached','--others','--exclude-standard','-z'],stderr=subprocess.DEVNULL).decode().split('\0')
        commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        dirty=bool(subprocess.check_output(['git','-C',str(source),'status','--porcelain']))
        names=[n for n in names if n and (source/n).is_file()]
    except subprocess.CalledProcessError:
        excluded={'.git','bin','outputs','__pycache__','.venv','.cache'}
        names=[str(p.relative_to(source)) for p in source.rglob('*') if p.is_file() and not excluded.intersection(p.relative_to(source).parts) and p.relative_to(source).parts[0]!='models'
               and p.suffix not in ('.o','.d','.pyc') and not p.name.startswith('.cuda-build-config') and 'lora/downloads/' not in str(p.relative_to(source))
               and p!=source/'src/metal/native_attention.inc']
        commit='unknown';dirty=None
    names=sorted(set(names))
    for name in names:
        if (source/name).is_symlink():raise ValueError('Source snapshots must not contain symlinks: '+name)
    before={name:(environment.digest(source/name),(source/name).stat().st_mode&0o111) for name in names}
    inventory={}
    for name in names:
        p=source/name
        if p.is_symlink():raise ValueError('Source snapshots must not contain symlinks: '+name)
        dest=destination/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        dest.chmod(0o755 if p.stat().st_mode&0o111 else 0o644)
        os.utime(dest,(LOCK['source_date_epoch'],LOCK['source_date_epoch']))
        inventory[name]=environment.digest(dest)
        if inventory[name]!=before[name][0]:raise ValueError('Source changed during snapshot: '+name)
    for name in names:
        if not (source/name).is_file() or before[name]!=(environment.digest(source/name),(source/name).stat().st_mode&0o111):
            raise ValueError('Source changed during snapshot: '+name)
    if commit!='unknown' and subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=commit:
        raise ValueError('Checkout revision changed during snapshot')
    if commit!='unknown':
        after=[n for n in subprocess.check_output(['git','-C',str(source),'ls-files','--cached','--others','--exclude-standard','-z']).decode().split('\0') if n and (source/n).is_file()]
    else:
        after=[str(p.relative_to(source)) for p in source.rglob('*') if p.is_file() and not excluded.intersection(p.relative_to(source).parts) and p.relative_to(source).parts[0]!='models'
               and p.suffix not in ('.o','.d','.pyc') and not p.name.startswith('.cuda-build-config') and 'lora/downloads/' not in str(p.relative_to(source))
               and p!=source/'src/metal/native_attention.inc']
    if sorted(set(after))!=names:raise ValueError('Source file list changed during snapshot')
    return {'commit':commit,'dirty':dirty,'files':inventory,
            'snapshot_sha256':hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest()}


def run_builder(cache,root,source,inputs,work,jobs,command):
    bindings=[str(source)+':/source',str(inputs)+':/inputs',str(work/'build')+':/build',str(work/'evidence')+':/evidence']
    env={'PATH':'/usr/sbin:/usr/bin:/sbin:/bin','HOME':'/root','LANG':'C.UTF-8','TERM':'dumb',
         'SOURCE_DATE_EPOCH':str(LOCK['source_date_epoch'])}
    args=environment.command(cache,root,bindings,cwd='/source')
    return subprocess.run(args+['env','H3_BUILD_JOBS='+str(jobs),*command],env=env,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache',type=Path,default=SOURCE/'outputs/linux-build/cache',help='Verified dependency download cache')
    p.add_argument('--work-dir',type=Path,default=SOURCE/'outputs/linux-build/work',help='Fresh directory for isolated build and evidence; must not exist')
    p.add_argument('--output',type=Path,default=SOURCE/'bin/linux-release',help='Fresh caller-owned release directory')
    p.add_argument('--jobs',type=int,default=int(os.environ.get('H3_BUILD_JOBS','8')))
    p.add_argument('--fetch-only',action='store_true',help='Populate and verify dependencies without compiling')
    p.add_argument('--offline',action='store_true',help='Require all locked inputs in cache; never fetch missing inputs')
    a=p.parse_args()
    if platform.system()!='Linux' or platform.machine()!='x86_64':p.error('The builder requires Linux x86-64; ordinary make supports macOS')
    if not 1<=a.jobs<=32:p.error('--jobs must be between 1 and 32')
    for tool in ('skopeo','umoci','cp'):
        if not shutil.which(tool):p.error('Missing '+tool+'; install host build helpers first (Debian/Ubuntu: apt-get install skopeo umoci)')
    cache=a.cache.expanduser().resolve();work=a.work_dir.expanduser().resolve();output=a.output.expanduser().resolve()
    if not a.fetch_only and (work.exists() or output.exists()):p.error('--work-dir and --output must be fresh; existing builds are never overwritten')
    started=time.monotonic();inputs=fetch(cache,a.offline)
    if a.fetch_only:return 0
    work.mkdir(parents=True);(work/'build').mkdir();(work/'evidence').mkdir()
    source=work/'source';record=snapshot(SOURCE,source);write(work/'evidence/source.json',record)
    # Never compile in, or clone, a mutable cached root filesystem. Every build
    # is unpacked again from the checksum-verified OCI blobs.
    bundle=work/'builder';root=bundle/'rootfs'
    subprocess.run(['umoci','unpack','--rootless','--image',str(cache/'images/builder')+':base',str(bundle)],check=True)
    run_builder(cache,root,source,inputs,work,a.jobs,['bash','-c','export DEBIAN_FRONTEND=noninteractive; cp /inputs/debs/*.deb /var/cache/apt/archives/ && apt-get --no-download --no-install-recommends -y install /inputs/debs/*.deb'])
    run_builder(cache,root,source,inputs,work,a.jobs,['bash','scripts/linux/build_dependencies.sh'])
    run_builder(cache,root,source,inputs,work,a.jobs,['bash','scripts/linux/build_release.sh'])
    output.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source/'bin/linux-release',output)
    package.seal(output/'linux-runtime')
    write(work/'evidence/timing.json',{'seconds':time.monotonic()-started,'jobs':a.jobs,'offline':a.offline})
    print('Release prepared at '+str(output)+'; qualification is separate from compilation.')
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,subprocess.CalledProcessError) as exc:raise SystemExit('Linux build: '+str(exc))
