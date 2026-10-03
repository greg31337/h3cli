#!/usr/bin/env python3
"""Assemble runtime, validation probes and provenance inside the pinned builder."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import package
import source_materials

ROOT=Path('/source')
sys.path.insert(0,str(ROOT/'tests'))
sys.path.insert(0,str(ROOT/'scripts'))
import reference_media
import cuda_reference_regression as gate
from current_cuda_suite import TARGETS


def output(command):return subprocess.check_output(command,text=True).strip()


def main():
    release=ROOT/'bin/linux-release';release.mkdir(exist_ok=True)
    runtime=release/'linux-runtime';validation=release/'linux-validation'
    lock=json.loads((ROOT/'scripts/linux/lock.json').read_text())
    source=json.loads(Path('/evidence/source.json').read_text())
    site=Path('/opt/h3deps/venv/lib/python3.10/site-packages')
    provenance={'schema':1,'source_sha256':gate.fingerprint(gate.source_files(ROOT)),
        'commit':source['commit'],'dirty':source['dirty'],
        'lock_sha256':package.sha(ROOT/'scripts/linux/lock.json'),'lock':lock,
        'requirements_sha256':package.sha(ROOT/'scripts/requirements-sglang-runtime.txt'),
        'compiler':output(['gcc','--version']).splitlines()[0],
        'linker':output(['ld','--version']).splitlines()[0],
        'nvcc':output(['/usr/local/cuda-13.0/bin/nvcc','--version']),
        'packages':output(['dpkg-query','-W','-f=${binary:Package}\t${Version}\n']),
        'features':{'sglang':True,'cudnn':True,'sage':True,'sol':True,'subblock':True},
        'cuda_gencode':output(['bash','scripts/cuda_arch.sh','fat','/usr/local/cuda-13.0/bin/nvcc']),
        'embedded_cuda_code':output(['/usr/local/cuda-13.0/bin/cuobjdump','--list-elf',str(ROOT/'bin/h3cli')]),
        'input_ffmpeg':output([os.environ['H3_SGLANG_INPUT_FFMPEG'],'-version']),
        'output_ffmpeg':output([os.environ['H3_FFMPEG'],'-version']),
        'source_date_epoch':lock['source_date_epoch'],
        'runtime_requirements':{'glibc':lock['glibc_max'],'driver':lock['driver_min'],'architecture':'x86_64'}}
    search=[Path('/usr/local/cuda-13.0/lib64'),Path('/opt/h3deps/lib'),site/'nvidia/cudnn/lib',site/'nvidia/cu13/lib',Path('/usr/lib/x86_64-linux-gnu')]
    package.assemble(runtime,ROOT/'bin/h3cli',search,os.environ['H3_SGLANG_INPUT_FFMPEG'],os.environ['H3_FFPROBE'],os.environ['H3_FFMPEG'],provenance,site/'nvidia/cu13/lib')
    # License and source provenance are part of the immutable runtime manifest.
    notices=runtime/'share/licenses';notices.mkdir(parents=True)
    shutil.copyfile(ROOT/'LICENSE',notices/'h3cli-LICENSE')
    shutil.copytree('/usr/share/common-licenses',notices/'common-licenses',symlinks=False)
    package.write_json(notices/'ffmpeg-source.json',{
        'version':'9.0.2','source':lock['sources']['ffmpeg-9.0.2.tar.xz'],
        'build_recipe':'scripts/linux/build_dependencies.sh',
        'status':'Built from the pinned upstream source. Complete patched dependency '
                 'sources and build recipes accompany h3cli-linux-x86_64-sources.tar.gz.'})
    for folder in (Path('/usr/share/doc'),site,Path('/usr/local/cuda-13.0')):
        for path in sorted(folder.rglob('*')):
            if 'imageio_ffmpeg' in str(path):continue
            if path.is_file() and (path.name.lower() in ('copyright','license','license.txt','eula.txt') or path.name.startswith('License')):
                dest=notices/folder.name/path.relative_to(folder);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    for name in ('icu','jpeg','ffmpeg','cudnn-frontend','sglang-cutlass','sage-cutlass'):
        folder=Path('/build/dependencies')/name
        for path in sorted(folder.glob('*')):
            if path.is_file() and path.name.lower().startswith(('license','copying','copyright')):
                dest=notices/name/path.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    for path in sorted((ROOT/'third_party').rglob('*')):
        if path.is_file() and path.name.lower().startswith(('license','copying','copyright','notice')):
            dest=notices/'third_party'/path.relative_to(ROOT/'third_party');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    # This vendored header carries its zlib notice in the source itself.
    dest=notices/'third_party/torch-rng/avx_mathfun.h';dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/'third_party/torch-rng/avx_mathfun.h',dest)
    materials=source_materials.prepare(ROOT,runtime,release,lock,source)
    manifest_path=runtime/'share/h3cli/runtime.json';manifest=json.loads(manifest_path.read_text())
    for path in sorted(notices.rglob('*')):
        if path.is_file():
            path.chmod(0o644);manifest['files'][str(path.relative_to(runtime))]={'sha256':package.sha(path),'size':path.stat().st_size,'mode':0o644}
    package.write_json(manifest_path,manifest)
    probes={}
    for name in sorted(set(gate.BUILD[1:]+TARGETS)):
        dest=validation/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest);dest.chmod(0o755)
        subprocess.run(['patchelf','--force-rpath','--set-rpath',package.library_rpath(runtime,dest),str(dest)],check=True)
        probes[name]=package.sha(dest)
    # These historical binaries accompany the private validation kit only.
    # The standalone executable contains exclusively FFmpeg/FFprobe 9.0.2.
    media=validation/'media'
    for media_source,relative in ((site/'imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2','output/ffmpeg'),
                            (Path('/opt/h3deps/reference-media/bin/ffmpeg'),'input/ffmpeg'),
                            (Path('/opt/h3deps/reference-media/bin/ffprobe'),'input/ffprobe')):
        dest=media/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(media_source,dest);dest.chmod(0o755)
        if relative.startswith('input/'):
            subprocess.run(['patchelf','--force-rpath','--set-rpath','$ORIGIN/../../../linux-runtime/lib',str(dest)],check=True)
    profile=reference_media.write_profile(media)
    result=package.package(runtime,ROOT/'bin/linux-launcher',release/'h3cli-linux-x86_64')
    package.write_json(validation/'manifest.json',{'schema':1,'source_sha256':provenance['source_sha256'],
        'reference_media_sha256':package.sha(profile),
        'runtime_manifest_sha256':package.sha(manifest_path),'probes':{p:probes[p] for p in gate.BUILD[1:]},'artifact_sha256':result['sha256']})
    package.write_json(validation/'features.json',{'schema':1,'source_sha256':provenance['source_sha256'],
        'binaries':{'bin/h3cli':result['sha256'],**{p:probes[p] for p in TARGETS}}})
    # Local marker does not enter the payload, hashes, or reproducible result.
    package.seal(runtime)
    shutil.copyfile(ROOT/'bin/libh3.a',release/'libh3.a')
    source_materials.finish(materials,release,lock)
    # The complete checkout includes reports that may quote the release hash.
    # Keep that outer snapshot record out of the payload to avoid a hash cycle.
    package.write_json(release/'build.json',provenance|{'snapshot_sha256':source['snapshot_sha256']})
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
