#!/usr/bin/env python3
"""Internal per-host stages. Called locally or through the SSH cancellation lifeline."""
import argparse
import ctypes
import fcntl
import hashlib
import json
import importlib.util
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

from common import ROOT, read, write, run, sha, digest, require, identities, verify, receipt, completed, attempt, clean_env
import dependency_bundle as bundles

PLATFORMS = {'macos':'macos-arm64', 'linux':'linux-x86_64'}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def system_name(target):
    command=['sysctl','-n','machdep.cpu.brand_string'] if target=='macos' else [
        'nvidia-smi','-i','0','--query-gpu=name,driver_version,compute_cap','--format=csv,noheader']
    detail=run(command,capture=True)
    return platform.platform()+'; '+('CPU: ' if target=='macos' else 'GPU / driver / compute capability: ')+detail


def scratch_root(job):
    """Choose scratch automatically; existing release roots can be on ext4."""
    candidates=[Path(job['root'])/'native-scratch',Path(job['dependency_cache']).parent/'native-scratch',
                Path(job['models'])/'.h3cli-release-scratch']
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True,exist_ok=True)
            # Clear inherited setgid on this release-owned scratch directory;
            # archive fixtures intentionally reject special permission bits.
            candidate.chmod(0o700)
            with tempfile.TemporaryDirectory(prefix='clone-check-',dir=candidate) as folder:
                src=Path(folder)/'source';dst=Path(folder)/'clone';src.write_bytes(b'h3cli clone check')
                if job['target']=='macos':
                    clone=ctypes.CDLL(None,use_errno=True).clonefile
                    clone.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int]
                    if clone(os.fsencode(src),os.fsencode(dst),0):raise OSError(ctypes.get_errno(),'clonefile')
                else:
                    with src.open('rb') as source,dst.open('wb') as target:
                        fcntl.ioctl(target.fileno(),0x40049409,source.fileno())
                require(dst.read_bytes()==src.read_bytes(),'Cloned scratch data differs')
            return candidate
        except OSError:
            continue
    raise ValueError('Required LoRA tests need a cloning-capable scratch volume; configure a work/cache/models path on APFS (Mac) or XFS with reflinks/Btrfs (Linux)')


def host_libraries(source,work):
    """Expose the locked non-system library closure to native host tests.

    Keep the host loader/glibc and compiler support libraries, so host Python,
    Git and compiler test fixtures remain usable. CUDA/math providers retain
    their original distinct directories from the qualified environment.
    """
    package=module('native_linux_package',ROOT/'scripts/linux/package.py')
    root=work/'builder/rootfs';out=work/'host-libraries';out.mkdir(exist_ok=True)
    site=root/'opt/h3deps/venv/lib/python3.10/site-packages'
    search=[root/'opt/h3deps/lib',root/'opt/h3deps/lib64',site/'nvidia/cudnn/lib',site/'nvidia/cu13/lib',
            root/'usr/local/cuda-13.0/lib64',root/'usr/lib/x86_64-linux-gnu',root/'lib/x86_64-linux-gnu']
    pending=[p for p in (source/'bin').iterdir() if p.is_file() and package.elf(p)]
    pending += [root/'opt/h3deps/bin/ffmpeg',root/'opt/h3deps/bin/ffprobe',work/'media/input/ffmpeg',work/'media/input/ffprobe',work/'media/output/ffmpeg']
    seen=set();libraries={}
    while pending:
        file=pending.pop()
        if str(file) in seen:continue
        seen.add(str(file))
        for name in re.findall(r'\(NEEDED\).*?\[(.*?)\]',package.dynamic(file)):
            if package.HOST.fullmatch(name) or name.startswith(('libstdc++.so','libgcc_s.so')):continue
            found=None
            for folder in search:
                candidate=folder/name
                for _ in range(10):
                    if not candidate.is_symlink():break
                    link=Path(os.readlink(candidate));candidate=root/str(link).lstrip('/') if link.is_absolute() else candidate.parent/link
                if candidate.is_file():found=candidate.resolve();break
            require(found is not None,'Qualified native library missing: '+name)
            pending.append(found)
            if package.nvidia_library(name):continue
            require(name not in libraries or libraries[name]==str(found),'Conflicting native library: '+name)
            libraries[name]=str(found)
            link=out/name
            if link.is_symlink():require(link.resolve()==found,'Native library link changed: '+name)
            else:link.symlink_to(found)
    tools=work/'host-tools';tools.mkdir(exist_ok=True)
    openssl=root/'usr/bin/openssl'
    require(openssl.is_file(),'Qualified OpenSSL test helper is missing')
    link=tools/'openssl'
    if not link.exists():link.symlink_to(openssl)
    require(link.is_symlink() and link.resolve()==openssl.resolve(),'OpenSSL test helper changed')
    write(out/'manifest.json',dict(files=identities([*libraries.values(),openssl])))
    return out


def provision(job):
    """Install missing host helpers and locked models, retaining existing weights."""
    target = job['target']
    require((platform.system(), platform.machine()) ==
            (('Darwin','arm64') if target=='macos' else ('Linux','x86_64')), 'Wrong native host: '+target)
    needed=('git','skopeo','umoci') if target=='linux' else ('git','pkg-config','xcrun')
    install=['--install'] if any(not shutil.which(n) for n in needed) else []
    run(['bash', ROOT/'release/prerequisites.sh', '--platform', target, *install])
    if target == 'linux':
        run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'])
    print('Native test scratch: '+str(scratch_root(job)),flush=True)
    catalog = read(ROOT/'src/models/catalog.json')
    for item in catalog['artifacts']:
        base = Path(job['model'] if item['group'] not in ('preview','image-vae','upscale') else job['models'])
        path = base/item['path']
        if path.is_file() and path.stat().st_size == item['bytes']:
            continue
        require(not job.get('offline'), 'Offline model missing or wrong size: '+str(path))
        require(not path.exists(), 'Existing model differs from catalog; select a separate models directory: '+str(path))
        print('Provisioning model '+item['path']+f" ({item['bytes']/1e9:.2f} GB)", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name+'.release-download')
        try:
            with urllib.request.urlopen(item['url'], timeout=120) as src, part.open('wb') as dst:
                shutil.copyfileobj(src, dst, 4 << 20)
            require(part.stat().st_size == item['bytes'] and sha(part) == item['sha256'], 'Model download checksum mismatch')
            part.replace(path)
        finally:
            part.unlink(missing_ok=True)
    print('Host tools and required models ready.', flush=True)


def dependencies(job, base):
    target = PLATFORMS[job['target']]
    cache = Path(job['dependency_cache'])
    expected = bundles.recipe(ROOT, target)
    registry = cache.parent/'qualified'/target/bundles.recipe_id(expected)
    choices = [Path(job['dependencies'])] if job.get('dependencies') else []
    choices += sorted(registry.glob('*/bundle.json'))
    for pin in choices:
        if not pin.is_file(): continue
        row = bundles.reference(pin)
        if row['recipe_id'] != bundles.recipe_id(expected) or 'qualification' not in row: continue
        print('Verifying reusable dependencies: '+str(pin), flush=True)
        payload, record = bundles.load(pin, cache, expected, job.get('offline', False))
        return pin, payload, record
    require(not job.get('offline'), 'No compatible qualified dependency bundle in the offline cache')
    print('No compatible bundle: building and qualifying locked dependencies once.', flush=True)
    work = attempt(registry, 'qualification')
    args = ['--cache', job['build_cache'], '--jobs', job['jobs']]
    run([sys.executable, ROOT/'scripts/dependencies.py', 'build', '--platform', target,
         '--work-dir', work/'build', '--output', work/'bundle', *args], log=work/'build.log')
    run([sys.executable, ROOT/'scripts/dependencies.py', 'qualify', '--bundle', work/'bundle/candidate.json',
         '--out', work/'qualified', '--model', job['model'], '--models-path', job['models'],
         '--dependency-cache', cache, *args], log=work/'qualification.log')
    pin = work/'qualified/bundle.json'
    # Registry entries only point at completed, verified qualifications.
    write(work/'bundle.json', bundles.reference(pin))
    payload, record = bundles.load(pin, cache, expected, offline=True)
    return pin, payload, record


def snapshot(destination):
    run(['git', 'clone', '--quiet', '--no-hardlinks', ROOT, destination])
    run(['git', '-C', destination, 'checkout', '--quiet', '--detach', 'HEAD'])


def native_command(job, result, action, out):
    source, work = Path(result['source']), Path(result['work'])
    scratch=Path(result.get('scratch',str(work/'scratch')))
    native_job = dict(job)
    native_job['native_manifest'] = str(work/'native/native.json')
    path = out/'job.json'; write(path, native_job)
    env = clean_env()
    if job['target'] == 'macos':
        payload = Path(result['payload'])
        env.update(H3_FFMPEG=str(payload/'h3cli-media/bin/ffmpeg'),
                   H3_SGLANG_INPUT_FFMPEG=str(payload/'h3cli-media/bin/ffmpeg'),
                   H3_FFPROBE=str(payload/'h3cli-media/bin/ffprobe'), TMPDIR=str(scratch))
        env['PATH']=str(payload/'h3cli-media/bin')+os.pathsep+env['PATH']
        run([sys.executable, source/'release/native.py', action, '--job', path, '--out', out],
            cwd=source, env=env, log=out/'native.log')
    elif action=='test':
        # Native host tests exercise real filesystem permissions/metadata. PRoot
        # fake-root permission emulation itself changes ctime for mode-000 files.
        # Reuse the qualified libraries without emulating the runtime filesystem.
        root=work/'builder/rootfs'
        script=(root/'opt/h3deps/environment.sh').read_text()
        for prefix in ('/opt/h3deps','/usr/local/cuda-13.0'):
            script=script.replace(prefix,str(root)+prefix)
        environment_file=out/'native-environment.sh';environment_file.write_text(script)
        safe={'PATH':'/usr/bin:/bin','HOME':str(Path.home()),'LANG':'C.UTF-8'}
        captured=run(['bash','-c','source "$1"; env -0','h3-native-environment',environment_file],env=safe,capture=True)
        env=dict(item.split('=',1) for item in captured.split('\0') if '=' in item)
        env['LD_LIBRARY_PATH']=str(result['host_libraries'])+':'+env['LD_LIBRARY_PATH']
        env['PATH']=str(work/'host-tools')+':'+env['PATH']
        env.update(H3_OFFLINE='1',H3_CPU_SAMPLER='1',H3_TEST_MAX_EVALUATIONS='6',CUDA_ARCH='auto',
                   TMPDIR=str(scratch),H3_REFERENCE_MEDIA=str(work/'media/profile.json'))
        run([sys.executable,source/'release/native.py',action,'--job',path,'--out',out],
            cwd=source,env=env,log=out/'native.log')
    else:
        environment = module('release_linux_environment', ROOT/'scripts/linux/environment.py')
        # Same absolute paths inside and outside the userland keep recorded
        # provenance relocatable between the worker and test runner.
        bindings = [str(source)+':/source', str(work)+':'+str(work), str(out)+':'+str(out),
                    str(scratch)+':'+str(scratch),job['models']+':'+job['models'], job['model']+':'+job['model']]
        command = environment.command(Path(result['payload']), work/'builder/rootfs', bindings,
                                      gpu=True, cwd='/source')
        env.update(PATH='/usr/bin:/bin', HOME='/root', LANG='C.UTF-8')
        program = ('set -euo pipefail; source /opt/h3deps/environment.sh; '
                   'export NVCC=/source/scripts/linux/nvcc.sh CUDA_ARCH=auto H3_OFFLINE=1 H3_CPU_SAMPLER=1; '
                   'export TMPDIR="$1" H3_REFERENCE_MEDIA="$2"; shift 2; exec python3 release/native.py "$@"')
        run([*command, 'bash', '-c', program, 'h3-native', scratch, work/'media/profile.json',
             action, '--job', path, '--out', out], env=env, log=out/'native.log')


def build_local(job, work):
    pin, payload, dependency = dependencies(job, work)
    source = work/'source'; snapshot(source)
    scratch=attempt(scratch_root(job),'native')
    result = dict(work=str(work), source=str(source), payload=str(payload), pin=str(pin), dependencies=dependency,
                  binary=str(source/'bin/h3cli'), scratch=str(scratch),system=system_name(job['target']))
    if job['target'] == 'linux':
        run(['umoci', 'unpack', '--rootless', '--image', str(payload/'images/builder')+':base', work/'builder'], log=work/'unpack.log')
        root = work/'builder/rootfs'
        media = work/'media'
        paths = [(root/'opt/h3deps/venv/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2','output/ffmpeg'),
                 (root/'opt/h3deps/reference-media/bin/ffmpeg','input/ffmpeg'),
                 (root/'opt/h3deps/reference-media/bin/ffprobe','input/ffprobe')]
        for src, rel in paths:
            dst=media/rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src,dst)
        # The decoder's dynamic dependencies are installed inside this userland.
        environment = module('release_linux_environment', ROOT/'scripts/linux/environment.py')
        command = environment.command(payload, root, [str(source)+':/source', str(work)+':'+str(work)], cwd='/source')
        run([*command,'bash','-c','source /opt/h3deps/environment.sh; PYTHONPATH=scripts python3 -c \'import reference_media,sys; reference_media.write_profile(sys.argv[1])\' "$1"',
             'h3-media',media], env=clean_env(PATH='/usr/bin:/bin', HOME='/root'), log=work/'media.log')
    native_command(job, result, 'build', work/'native')
    files=[work/'native/native.json', *[source/n for n in read(work/'native/native.json')['binaries']], pin]
    if job['target']=='linux':
        libs=host_libraries(source,work);result['host_libraries']=str(libs)
        files += [libs/'manifest.json',*read(libs/'manifest.json')['files']]
    return result, files


def test_local(job, previous, work):
    require(previous['system']==system_name(job['target']),'Native host changed; rerun build-local to select its current architecture')
    native_command(job, previous, 'test', work)
    row=read(work/'result.json'); require(row.get('passed') is True, 'Native tests did not pass')
    return dict(report=str(work/'result.json'), binary_sha256=sha(previous['binary'])), list(work.rglob('*.json'))


def build_packages(job, previous, work):
    target = job['target']; output=work/'unsigned'
    run(['bash', ROOT/('scripts/build_'+target+'.sh'), '--dependencies', previous['pin'],
         '--dependency-cache', job['dependency_cache'], '--cache', job['build_cache'], '--offline',
         '--jobs', job['jobs'], '--work-dir', work/'build', '--output', output], log=work/'package.log')
    result=dict(directory=str(output), binary=str(output/('h3cli-'+PLATFORMS[target])),
                dependencies=previous['dependencies'])
    return result, [output/'build.json', *[p for p in output.iterdir() if p.is_file()]]


def extracted_runtime(info,target):
    # Mac reports the cache base; Linux reports the installed runtime itself.
    root=Path(info['cache'])
    return root/info['runtime_id'] if target=='macos' else root


def test_packages(job, previous, work):
    target=job['target']; folder=Path(previous['directory'])
    binary=Path(previous['binary']); runtime=folder/(target+'-runtime')
    external=work/'outside checkout'; external.mkdir()
    moved=external/binary.name; shutil.copy2(binary,moved)
    before=sha(binary)
    env=clean_env(H3CLI_RUNTIME_CACHE=str(work/'cache'), PATH='/usr/bin:/bin:/usr/sbin:/sbin')
    run([moved,'--help'],cwd=external,env=env,log=work/'help.log')
    info=json.loads(run([moved],cwd=external,env=env|{'H3CLI_BUNDLE_INFO':'1'},capture=True))
    extracted=extracted_runtime(info,target)
    if target=='linux':
        run([sys.executable, ROOT/'tests/cuda_reference_regression.py', '--source', ROOT,
             '--runtime',runtime,'--validation',folder/'linux-validation','--artifact',moved,
             '--model',job['model'],'--out',work/'parity'], env=env, log=work/'parity.log')
        row=read(work/'parity/result.json')
        require(row.get('passed') is True and len(row.get('files',{}))==204, 'Final artifact CUDA gate failed')
    run([sys.executable,ROOT/'tests/model_downloads_render.py','--binary',moved,
         '--backend','cuda' if target=='linux' else 'metal','--models-path',job['models'],
         '--model',job['model'],'--ffmpeg',extracted/'tools/ffmpeg','--ffprobe',extracted/'tools/ffprobe',
         '--server','--out',work/'features'], cwd=external, env=env, log=work/'features.log')
    require(read(work/'features/results.json').get('passed') is True,'Packaged feature tests did not pass')
    if target=='macos':
        # Mount the actual stapled DMG; test the binary extracted from it.
        mount=work/'mounted'; mount.mkdir()
        run(['hdiutil','attach','-readonly','-nobrowse','-mountpoint',mount,folder/'h3cli-macos-arm64.dmg'],log=work/'mount.log')
        try:
            candidates=list(mount.rglob('h3cli-macos-arm64'))
            require(len(candidates)==1 and sha(candidates[0])==before,'DMG executable differs from standalone')
            run([candidates[0],'--help'],cwd=external,env=env,log=work/'dmg-help.log')
        finally:
            run(['hdiutil','detach',mount],log=work/'unmount.log')
    archive=folder/('h3cli-'+PLATFORMS[target]+'-sources.tar.gz')
    with tarfile.open(archive) as tar:
        members=tar.getmembers();names=[m.name for m in members]
        require(any(n.endswith('src/engine.c') for n in names) and any(n.endswith('LICENSE') for n in names), 'Missing source/license materials')
        require(not any(Path(n).is_absolute() or '..' in Path(n).parts or '/outputs/' in '/'+n for n in names), 'Invalid source materials')
        expected=set(run(['git','ls-files'],capture=True).splitlines())
        checked=set()
        for member in members:
            relative=member.name.split('h3cli/',1)[-1]
            if relative not in expected:continue
            require(member.isfile(),'Source archive replaced a tracked file: '+relative)
            with tar.extractfile(member) as src:
                actual=hashlib.file_digest(src,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(src.read()).hexdigest()
            require(actual==sha(ROOT/relative),'Source archive does not match selected commit: '+relative)
            checked.add(relative)
        require(checked==expected,'Source archive omits tracked application files: '+', '.join(sorted(expected-checked)[:10]))
    require(sha(binary)==before and sha(moved)==before,'Tested package changed')
    report=work/'result.json'
    system=system_name(target)
    write(report,dict(passed=True,binary_sha256=before,source_archive_sha256=sha(archive),system=system,
                     cuda_outputs=204 if target=='linux' else None,scope='Existing account; native GPU; isolated runtime cache; CLI/server/media/rendering'))
    return dict(report=str(report),binary_sha256=before,system=system), list(work.rglob('*.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('provision','build-local','test-local','build-packages','test-packages','downloaded'))
    p.add_argument('--job',type=Path,required=True)
    a=p.parse_args(); job=read(a.job); base=Path(job['base']); base.mkdir(parents=True,exist_ok=True)
    if a.stage=='provision': provision(job); return
    require(run(['git','rev-parse','HEAD'],capture=True)==job['commit'] and
            not run(['git','status','--porcelain'],capture=True), 'Worker checkout differs from the selected commit')
    if a.stage=='downloaded':
        binary=Path(job['downloaded'])
        require(sha(binary)==job['downloaded_sha256'],'Downloaded Linux binary changed in transfer')
        work=attempt(base,'downloaded-test')
        env=clean_env(H3CLI_RUNTIME_CACHE=str(work/'cache'))
        run([binary,'--models-path',job['models'],'-d',job['model'],'-p','A red toy boat on a quiet pond.',
             '--width','256','--height','256','--frames','22','--steps','2','-o',work/'video.mp4'],cwd=work,env=env,log=work/'render.log')
        require((work/'video.mp4').stat().st_size>0,'Downloaded binary produced no video')
        info=run([binary],env=env|{'H3CLI_BUNDLE_INFO':'1'},capture=True,cwd=work)
        extracted=Path(json.loads(info)['cache'])
        probe=json.loads(run([extracted/'tools/ffprobe','-v','error','-count_frames','-show_streams','-of','json',work/'video.mp4'],capture=True,cwd=work,env=env))
        video=next(s for s in probe['streams'] if s['codec_type']=='video')
        require((int(video['width']),int(video['height']),int(video['nb_read_frames']))==(256,256,22),'Downloaded render has wrong geometry/frame count')
        run([extracted/'tools/ffmpeg','-v','error','-xerror','-i',work/'video.mp4','-f','null','-'],env=env,cwd=work,log=work/'decode.log')
        write(base/'downloaded.json',dict(passed=True,binary_sha256=sha(binary),video=video))
        return
    previous=None; inputs=dict(job=job)
    if a.stage=='build-local':inputs['native_host']=system_name(job['target'])
    dependencies_by_stage={'test-local':'build-local','build-packages':'test-local','test-packages':'build-packages'}
    if a.stage in dependencies_by_stage:
        predecessor=base/(dependencies_by_stage[a.stage]+'.json')
        require(predecessor.is_file(),'Run '+dependencies_by_stage[a.stage]+' first')
        row=read(predecessor); verify(row['files']); inputs['previous']=sha(predecessor)
        if a.stage=='build-packages':
            row=read(base/'build-local.json');verify(row['files'])
        previous=row['result']
    if a.stage=='test-packages' and job.get('signed'):
        previous=dict(previous, directory=job['signed'],binary=str(Path(job['signed'])/'h3cli-macos-arm64'))
        inputs['signed']=identities(Path(job['signed'])/n for n in ('build.json','h3cli-macos-arm64','h3cli-macos-arm64.dmg'))
    marker=base/(a.stage+'.json')
    row=completed(marker,inputs)
    if row:
        print('Reusing verified '+a.stage+' result: '+str(marker),flush=True);return
    work=attempt(base/'attempts',a.stage)
    function={'build-local':build_local,'test-local':test_local,'build-packages':build_packages,'test-packages':test_packages}[a.stage]
    result,files=function(job,work) if a.stage=='build-local' else function(job,previous,work)
    receipt(marker,inputs,result,files)
    print('Completed '+a.stage+': '+str(marker),flush=True)


if __name__=='__main__':
    try:
        import build_process
        with build_process.cancellation(): main()
    except KeyboardInterrupt:sys.exit(130)
    except (OSError,ValueError,subprocess.CalledProcessError) as exc:sys.exit(str(exc))
