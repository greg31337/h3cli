#!/usr/bin/env python3
"""Build a self-contained Apple Silicon CLI using a locked, installed Apple SDK."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import build_process
import dependency_bundle as bundles
import package
ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = Path(__file__).with_name('lock.json')
EXCLUDED = {'.git','.agents','.codex','bin','outputs','__pycache__','.venv','.cache','.DS_Store','misc','.pytest_cache'}


def toolchain():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('Build requires an Apple Silicon Mac')
    def capture(*args): return subprocess.check_output(args,text=True).strip()
    sdk = Path(capture('xcrun','--sdk','macosx','--show-sdk-path')).resolve()
    inventory = {}
    for p in sorted(sdk.rglob('*')):
        if p.is_symlink(): inventory[str(p.relative_to(sdk))] = 'link:'+os.readlink(p)
        elif p.is_file(): inventory[str(p.relative_to(sdk))] = package.sha(p)
    tools = {}
    for name in ('clang','ld','ar','strip','make'):
        p = Path(capture('xcrun','--find',name)).resolve()
        tools[name] = package.sha(p)
    return {'clang':capture('xcrun','clang','--version').splitlines()[0],
        'linker':json.loads(capture('xcrun','ld','-version_details')),
        'sdk_version':capture('xcrun','--sdk','macosx','--show-sdk-version'),
        'sdk_build':capture('xcrun','--sdk','macosx','--show-sdk-build-version'),
        'sdk_inventory_sha256':hashlib.sha256(package.canonical(inventory)).hexdigest(),
        'tools':tools, 'pkg_config':capture('pkg-config','--version')}


def names(source):
    try:
        top=subprocess.check_output(['git','-C',str(source),'rev-parse','--show-toplevel'],stderr=subprocess.DEVNULL,text=True).strip()
        if Path(top).resolve()!=source.resolve():raise subprocess.CalledProcessError(1,'git snapshot root')
        values = subprocess.check_output(['git','-C',str(source),'ls-files','--cached','--others','--exclude-standard','-z'],stderr=subprocess.DEVNULL).decode().split('\0')
    except subprocess.CalledProcessError:
        values = [str(p.relative_to(source)) for p in source.rglob('*') if p.is_file() or p.is_symlink()]
    result=[]
    for name in values:
        p=Path(name)
        if not name or EXCLUDED.intersection(p.parts) or p.parts[0]=='models' or p.suffix in ('.o','.d','.pyc') or p.name.startswith('.cuda-build-config') or name=='src/metal/native_attention.inc' or name.startswith(('lora/downloads/','lora/output/')):
            continue
        if p.name in ('.env','.netrc') or p.suffix in ('.key','.p12','.pfx','.pem'):
            raise ValueError('Credential-like file in source snapshot: '+name)
        if (source/name).is_symlink(): raise ValueError('Source symlinks are forbidden: '+name)
        if (source/name).is_file(): result.append(name)
    return sorted(set(result))


def snapshot(source,destination,epoch):
    selected=names(source)
    before={n:package.sha(source/n) for n in selected}
    for name in selected:
        p=destination/name;p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,p);p.chmod(0o755 if (source/name).stat().st_mode&0o111 else 0o644)
        os.utime(p,(epoch,epoch))
        if package.sha(p)!=before[name]: raise ValueError('Source changed during snapshot')
    if names(source)!=selected or any(package.sha(source/n)!=before[n] for n in selected):
        raise ValueError('Source changed during snapshot')
    return {'files':before,'sha256':hashlib.sha256(package.canonical(before)).hexdigest()}


def inputs(lock):
    ffmpeg=json.loads((ROOT/'scripts/linux/lock.json').read_text())['sources']['ffmpeg-9.0.2.tar.xz']
    return {'ffmpeg-9.0.2.tar.xz':ffmpeg,lock['x264']['name']:lock['x264']}


def fetch(cache,lock,offline):
    cache.mkdir(parents=True,exist_ok=True)
    for name,pin in inputs(lock).items():
        p=cache/name
        if not p.exists():
            if offline: raise ValueError('Missing offline dependency: '+name)
            temporary=p.with_suffix('.download')
            try:
                with urllib.request.urlopen(pin['url'],timeout=120) as source,temporary.open('wb') as target:
                    shutil.copyfileobj(source,target)
                if package.sha(temporary)!=pin['sha256']: raise ValueError('Source hash mismatch: '+name)
                temporary.replace(p)
            finally: temporary.unlink(missing_ok=True)
        if package.sha(p)!=pin['sha256'] or p.stat().st_size!=pin['size']:
            raise ValueError('Source hash/size mismatch: '+name)


def unpack(path,directory):
    directory.mkdir()
    with tarfile.open(path) as tar:
        # Hash-verified upstream archives; still refuse links and path escapes.
        for m in tar.getmembers():
            if m.issym() or m.islnk() or m.isdev() or Path(m.name).is_absolute() or '..' in Path(m.name).parts:
                raise ValueError('Unsafe dependency archive member: '+m.name)
        tar.extractall(directory)
    children=list(directory.iterdir())
    if len(children)!=1 or not children[0].is_dir():raise ValueError('Dependency archive layout')
    return children[0]


def run(command,cwd,env,log):
    print('Running '+str(command[0])+' in '+str(cwd),flush=True)
    with log.open('a') as output:
        output.write('\n'+repr(command)+'\n');output.flush()
        build_process.run(command,cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)


def sources(source,cache,lock,inventory,output,epoch,dependency=None):
    materials=output/'source-materials';materials.mkdir()
    snapshot(source,materials/'h3cli',epoch)
    (materials/'dependencies').mkdir()
    for name in inputs(lock): shutil.copyfile(cache/name,materials/'dependencies'/name)
    package.write(materials/'source-inventory.json',inventory)
    if dependency:package.write(materials/'dependency-bundle.json',dependency)
    shutil.copyfile(source/'scripts/macos/lock.json',materials/'macos-lock.json')
    (materials/'BUILD.txt').write_text('Corresponding h3cli, FFmpeg and x264 sources and recipes.\nApple SDK/frameworks are host prerequisites and are not redistributed.\nSee h3cli/docs/build/macos-distribution.md.\nRebuild/relink with the locked Apple tools: cd h3cli; bash scripts/build_macos.sh --source-dependencies --offline --cache ../dependencies --work-dir outputs/rebuild --output bin/rebuilt\nModified application/media sources may be rebuilt by updating the relevant source pin and requalifying the package.\n')
    archive=output/'h3cli-macos-arm64-sources.tar.gz'
    with archive.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=epoch) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
        for p in sorted(materials.rglob('*')):
            info=tar.gettarinfo(str(p),str(p.relative_to(materials)));info.uid=info.gid=0;info.uname=info.gname='';info.mtime=epoch
            if p.is_file():
                with p.open('rb') as f:tar.addfile(info,f)
            else:tar.addfile(info)
    shutil.rmtree(materials)
    return archive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--doctor',action='store_true');parser.add_argument('--fetch-only',action='store_true')
    parser.add_argument('--offline',action='store_true');parser.add_argument('--cache',type=Path,default=ROOT/'outputs/macos-build/cache')
    parser.add_argument('--work-dir',type=Path);parser.add_argument('--output',type=Path,default=ROOT/'bin/macos-release')
    parser.add_argument('--jobs',type=int,default=8)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--dependencies',type=Path,help='Pinned qualified dependency bundle reference')
    group.add_argument('--dependency-candidate',type=Path,help='Unqualified bundle for explicit qualification builds only')
    group.add_argument('--source-dependencies',action='store_true',help='Explicitly rebuild dependencies from source')
    parser.add_argument('--dependency-cache',type=Path,default=ROOT/'outputs/dependencies/cache',help='Shared verified bundle cache')
    args=parser.parse_args()
    if args.jobs<1 or args.jobs>256:parser.error('--jobs must be between 1 and 256')
    if args.doctor and args.fetch_only:parser.error('--doctor and --fetch-only are mutually exclusive')
    if not args.doctor and not args.fetch_only:
        if args.work_dir is None:args.work_dir=ROOT/'outputs/macos-build'/time.strftime('run-%Y%m%d-%H%M%S')
        for path in (args.work_dir,args.output):
            if path.exists():parser.error('Use a fresh directory: '+str(path))
        paths=[args.work_dir.resolve(),args.output.resolve(),args.cache.resolve()]
        if any(a==b or a in b.parents or b in a.parents for i,a in enumerate(paths) for b in paths[i+1:]):parser.error('Cache/work/output directories must not overlap')
    lock=json.loads(LOCK_PATH.read_text());actual=toolchain()
    if actual!=lock['toolchain']:raise ValueError('Unqualified Apple toolchain/SDK; compare --doctor inventory and requalify an updated lock: '+json.dumps(actual,sort_keys=True))
    if args.doctor:print(json.dumps({'qualified':True,'toolchain':actual},indent=2));return
    cache=args.cache.resolve();dependency=None;payload=None
    pin=args.dependencies or args.dependency_candidate
    if pin:
        payload,dependency=bundles.load(pin,args.dependency_cache.expanduser().resolve(),
            bundles.recipe(ROOT,'macos-arm64'),args.offline,bool(args.dependency_candidate))
        cache=payload/'inputs'
    elif args.source_dependencies:fetch(cache,lock,args.offline)
    else:parser.error('Select --dependencies QUALIFIED.json or explicitly use --source-dependencies; see scripts/dependencies.py')
    if args.fetch_only:print('All pinned dependency inputs verified.');return
    work=args.work_dir.resolve();output=args.output.resolve();work.mkdir(parents=True);output.mkdir(parents=True)
    epoch=lock['source_date_epoch'];source=work/'source';inv=snapshot(ROOT,source,epoch)
    package.write(work/'source.json',inv)
    producer=bundles.adapter(ROOT,'macos-arm64')
    env=producer.build_env(work,lock);log=work/'build.log'
    if args.source_dependencies:
        payload=work/'source-dependencies'
        producer.produce(ROOT,cache,work/'dependency-build',payload,args.jobs,args.offline)
    else:print('Reusing dependency bundle '+dependency['archive_sha256']+'; dependency compilation skipped.',flush=True)
    # Each release owns its copies; codesign must never touch the shared cache.
    deps=work/'dependencies';deps.mkdir()
    shutil.copytree(payload/'h3cli-media',deps/'h3cli-media')
    runtime=output/'macos-runtime'
    for p in ('bin','tools','src/metal','share/h3cli','share/licenses'): (runtime/p).mkdir(parents=True,exist_ok=True)
    native=work/'native';snapshot(source,native,epoch)
    run(['make','-j'+str(args.jobs),'H3_GIT_COMMIT=source-archive','CPPFLAGS=-arch arm64 -mmacosx-version-min=26.0','LDFLAGS=-arch arm64 -mmacosx-version-min=26.0','all'],native,env,log)
    shutil.copyfile(native/'bin/libh3.a',output/'libh3.a')
    probes=['bin/model_downloads','bin/metal_attention','bin/metal_layout','bin/metal_fp16','bin/metal_diagnostics','bin/metal_sol','bin/metal_q8','bin/ane_probe','bin/ane_tests','bin/h3cli-server-test']
    run(['make','-j'+str(args.jobs),'H3_GIT_COMMIT=source-archive','CPPFLAGS=-arch arm64 -mmacosx-version-min=26.0','LDFLAGS=-arch arm64 -mmacosx-version-min=26.0',*probes],native,env,log)
    validation=output/'macos-validation';validation.mkdir()
    for probe in probes:shutil.copy2(native/probe,validation/Path(probe).name)
    run(['make','-j'+str(args.jobs),'PACKAGE_RUNTIME=1','H3_GIT_COMMIT=source-archive','CPPFLAGS=-DH3_PACKAGE_RUNTIME -arch arm64 -mmacosx-version-min=26.0','LDFLAGS=-arch arm64 -mmacosx-version-min=26.0','all'],source,env,log)
    shutil.copyfile(source/'bin/h3cli',runtime/'bin/h3cli');(runtime/'bin/h3cli').chmod(0o755)
    for name in ('ffmpeg','ffprobe'):
        shutil.copyfile(deps/'h3cli-media/bin'/name,runtime/'tools'/name);(runtime/'tools'/name).chmod(0o755)
    shutil.copyfile(source/'src/metal/shaders.metal',runtime/'src/metal/shaders.metal')
    shutil.copyfile(source/'LICENSE',runtime/'share/licenses/h3cli.txt')
    shutil.copyfile(payload/'notices/FFmpeg.txt',runtime/'share/licenses/FFmpeg.txt')
    shutil.copyfile(payload/'notices/x264.txt',runtime/'share/licenses/x264.txt')
    for p in sorted((source/'third_party').rglob('*')):
        if p.is_file() and p.name.lower().startswith(('license','copying','copyright','notice')):
            target=runtime/'share/licenses/third_party'/p.relative_to(source/'third_party');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    for p in (runtime/'bin/h3cli',runtime/'tools/ffmpeg',runtime/'tools/ffprobe'): package.sign(p)
    audit=package.audit(runtime);package.write(output/'audit.json',audit)
    code={n:h for n,h in inv['files'].items() if n.startswith(('src/','third_party/','scripts/')) or n=='Makefile'}
    code_sha=hashlib.sha256(package.canonical(code)).hexdigest()
    manifest={'schema':1,'platform':'macos-arm64','files':package.inventory(runtime),'source_sha256':code_sha,'recipe':lock,
              'dependencies':dependency or {'mode':'source'}}
    package.write(runtime/'share/h3cli/runtime.json',manifest);package.seal(runtime)
    archive=work/'payload.bin';bundle=package.payload(runtime,archive,source)
    executable=output/'h3cli-macos-arm64';package.embed(source,archive,executable,env);package.sign(executable,identifier='org.h3cli.launcher')
    package.write(output/'audit.json',package.audit(output))
    materials=sources(source,cache,lock,inv,output,epoch,dependency)
    image=package.dmg(output,executable)
    record={'schema':1,'signing':'ad-hoc; unnotarized development build','source':inv,'toolchain':actual,'bundle':bundle,
            'dependencies':dependency or {'mode':'source'},'dependency_compilation':args.source_dependencies,
            'os':platform.platform(),'artifacts':{p.name:{'sha256':package.sha(p),'size':p.stat().st_size} for p in (executable,materials,image,output/'libh3.a')}}
    package.write(output/'build.json',record)
    (output/'SHA256SUMS').write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in record['artifacts'].items()))
    print('Built '+str(executable),flush=True)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:sys.exit(130)
    except (ValueError,subprocess.CalledProcessError,OSError) as e:sys.exit(str(e))
