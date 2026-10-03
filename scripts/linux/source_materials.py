#!/usr/bin/env python3
"""Bind every shipped component to original binaries, notices and source materials."""
import gzip
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import package


def debian_provider(path,lock):
    candidates=[str(path)]
    if str(path).startswith('/usr/lib/'):candidates.append(str(path).removeprefix('/usr'))
    for candidate in candidates:
        found=subprocess.run(['dpkg-query','-S',candidate],text=True,capture_output=True)
        if found.returncode==0:
            binary=found.stdout.split(': /',1)[0]
            fields=subprocess.check_output(['dpkg-query','-W','-f=${binary:Package}\t${Version}\t${source:Package}\t${source:Version}',binary],text=True).split('\t')
            binary,version,source,source_version=fields
            if source not in lock['source_packages'] or lock['source_packages'][source]['version']!=source_version:
                raise ValueError('Missing exact source package for '+binary+' '+source_version)
            return {'kind':'ubuntu-source-package','binary_package':binary,'binary_version':version,
                    'source_package':source,'source_version':source_version,
                    'notice':'doc/'+binary.split(':')[0]+'/copyright'}
    raise ValueError('No recorded package owns '+str(path))


def tar(directory,destination,epoch):
    with destination.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w',format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(directory.rglob('*')):
            if path.is_symlink() or (not path.is_file() and not path.is_dir()):
                raise ValueError('Unsafe source material '+str(path))
            item=archive.gettarinfo(str(path),arcname=str(Path(directory.name)/path.relative_to(directory)))
            item.uid=item.gid=0;item.uname=item.gname='';item.mtime=epoch
            item.mode=0o755 if path.is_dir() or path.stat().st_mode&0o111 else 0o644
            with path.open('rb') if path.is_file() else nullcontext() as file:
                archive.addfile(item,file)


def prepare(root,runtime,release,lock,snapshot):
    notices=runtime/'share/licenses';materials=release/'source-materials';materials.mkdir()
    manifest=json.loads((runtime/'share/h3cli/runtime.json').read_text())
    components={}
    for name,record in manifest['files'].items():
        source=Path(record['builder_path'])
        if package.nvidia_library(source.name):
            if record['sha256']!=record['input_sha256']:raise ValueError('Modified NVIDIA file '+name)
            if source.name.startswith('libcudnn'):
                license_path='site-packages/nvidia_cudnn_cu13-9.20.0.48.dist-info/licenses/License.txt'
            elif '/site-packages/' in str(source):
                wheel='nvidia_cuda_nvrtc-13.0.88' if source.name.startswith('libnvrtc') else 'nvidia_cublas-13.1.1.3'
                license_path='site-packages/'+wheel+'.dist-info/licenses/License.txt'
            else:
                component='libcublas-13-0' if source.name.startswith('libcublas') else 'cuda-cudart-13-0'
                license_path='doc/'+component+'/copyright'
            row={'kind':'nvidia-redistributable','notice':license_path,'unmodified':True,
                 'input_sha256':record['input_sha256'],'source':'Proprietary binary; source disclosure is not a redistribution condition.'}
        elif name=='bin/h3cli':row={'kind':'project-source','source':'h3cli/','notice':'h3cli-LICENSE'}
        elif name.startswith('tools/'):
            row={'kind':'upstream-source','source':'ffmpeg-9.0.2.tar.xz','notice':'ffmpeg/COPYING.GPLv2',
                 'license':'GPL-2.0-or-later','recipe':'h3cli/scripts/linux/build_dependencies.sh','changes':'No upstream source changes; relative ELF search paths set by package.py.'}
        elif source.name.startswith('libicu'):
            row={'kind':'upstream-source','source':'icu4c-74_2-src.tgz','notice':'icu/LICENSE'}
        elif source.name.startswith('libturbojpeg'):
            row={'kind':'upstream-source','source':'libjpeg-turbo-2.1.5.tar.gz','notice':'jpeg/LICENSE.md'}
        else:
            row=debian_provider(source,lock)
            notice=Path('/usr/share')/row['notice']
            dest=notices/row['notice'];dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(notice,dest)
        if not (notices/row['notice']).is_file():raise ValueError('Missing notice for '+name+': '+row['notice'])
        row['notice']='share/licenses/'+row['notice'];components[name]=row
    # The static launcher contains libc and compiler support in addition to these
    # three libraries. Their complete patched sources and relink recipe accompany it.
    static={}
    for binary in ('libc6-dev','libgcc-11-dev','libssl-dev','libjson-c-dev','zlib1g-dev'):
        fields=subprocess.check_output(['dpkg-query','-W','-f=${source:Package}\t${source:Version}',binary],text=True).split('\t')
        source,version=fields
        if lock['source_packages'].get(source,{}).get('version')!=version:raise ValueError('Missing launcher source '+source)
        static[binary]={'source_package':source,'source_version':version}
    audit={'schema':1,'components':components,'launcher':{'source':'h3cli/src/runtime/launcher.c',
           'recipe':'h3cli/scripts/linux/build_release.sh','static_dependencies':static},
           'source_packages':lock['source_packages'],'upstream_sources':{k:v for k,v in lock['sources'].items() if k!='ffmpeg-6.1.1.tar.xz'},
           'source_archive':'h3cli-linux-x86_64-sources.tar.gz',
           'distribution':'Supply this source archive beside the executable and its checksum. Historical linux-validation codecs are private test assets and excluded.',
           'nvidia':'This software contains source code provided by NVIDIA Corporation. NVIDIA binaries retain their own license terms; the project MIT license does not relicense them.'}
    package.write_json(notices/'redistribution.json',audit)
    shutil.copyfile(root/'scripts/linux/SOURCE-MATERIALS.md',notices/'SOURCE-MATERIALS.md')
    for name in audit['upstream_sources']:
        shutil.copyfile(Path('/inputs')/name,materials/name)
    source_packages=materials/'ubuntu';source_packages.mkdir()
    for record in lock['source_packages'].values():
        for artifact in record['artifacts']:
            original=Path('/inputs/source-packages')/artifact['name']
            if package.sha(original)!=artifact['sha256'] or original.stat().st_size!=artifact['size']:
                raise ValueError('Source material changed: '+artifact['name'])
            shutil.copyfile(original,source_packages/artifact['name'])
    checkout=materials/'h3cli'
    for name,expected in sorted(snapshot['files'].items()):
        path=root/name
        if package.sha(path)!=expected:raise ValueError('Source snapshot changed before source-material assembly: '+name)
        dest=checkout/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        dest.chmod(0o755 if path.stat().st_mode&0o111 else 0o644)
    recipes=materials/'configuration';recipes.mkdir()
    for name in ('ffmpeg','jpeg-build','icu/source'):
        folder=Path('/build/dependencies')/name
        for file in ('config.h','config.log','ffbuild/config.mak','CMakeCache.txt'):
            path=folder/file
            if path.is_file():
                # Configure logs can contain wall times. Keep exact configure
                # scripts/options via the source and stable config.h/config.mak.
                if file in ('config.log','CMakeCache.txt'):continue
                dest=recipes/name/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    shutil.copyfile(notices/'redistribution.json',materials/'redistribution.json')
    shutil.copyfile(root/'scripts/linux/SOURCE-MATERIALS.md',materials/'README.md')
    return materials


def finish(materials,release,lock):
    inventory={str(p.relative_to(materials)):{'sha256':package.sha(p),'size':p.stat().st_size}
               for p in sorted(materials.rglob('*')) if p.is_file()}
    package.write_json(materials/'manifest.json',{'schema':1,'files':inventory})
    archive=release/'h3cli-linux-x86_64-sources.tar.gz';tar(materials,archive,lock['source_date_epoch'])
    (release/(archive.name+'.sha256')).write_text(package.sha(archive)+'  '+archive.name+'\n')
