"""Build the reusable macOS media/development payload with the locked SDK."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_process
import dependency_bundle as bundles


def build_env(work, lock):
    env = {k: v for k, v in os.environ.items() if k in ('HOME', 'TMPDIR', 'DEVELOPER_DIR')}
    env.update(PATH='/usr/bin:/bin:/usr/sbin:/sbin:'+str(Path(shutil.which('pkg-config')).parent),
        LC_ALL='C', SOURCE_DATE_EPOCH=str(lock['source_date_epoch']), ZERO_AR_DATE='1',
        MACOSX_DEPLOYMENT_TARGET=lock['deployment_target'], H3_OFFLINE='1')
    env['SDKROOT'] = subprocess.check_output(['xcrun', '--sdk', 'macosx', '--show-sdk-path'], text=True).strip()
    env['CC']='clang';env['CXX']='clang++'
    wrappers = work/'tools';wrappers.mkdir()
    for name in ('clang', 'clang++'):
        tool = subprocess.check_output(['xcrun', '--find', name], text=True).strip()
        (wrappers/name).write_text('#!/bin/sh\nexec '+shlex.quote(tool)+' '+
            shlex.quote('-ffile-prefix-map='+str(work)+'=/h3cli-build')+' "$@"\n')
        (wrappers/name).chmod(0o755)
    env['PATH']=str(wrappers)+':'+env['PATH']
    return env


def produce(source, cache, work, payload, jobs, offline=False):
    import build
    lock = bundles.read(source/'scripts/macos/lock.json')
    if build.toolchain() != lock['toolchain']:
        raise ValueError('Unqualified Apple toolchain/SDK')
    build.fetch(cache, lock, offline)
    work.mkdir(parents=True);payload.mkdir(parents=True)
    env = build_env(work, lock)
    deps = work/'dependencies';deps.mkdir()
    log = work/'dependencies.log'
    x264 = build.unpack(cache/lock['x264']['name'], deps/'x264')
    build.run(['./configure', '--prefix=/h3cli-deps', *lock['x264_configure']], x264, env, log)
    config = x264/'x264_config.h'
    config.write_text(config.read_text().replace('#define X264_VERSION ""', '#define X264_VERSION " r3222 b35605a"')
        .replace('#define X264_POINTVER "0.165.x"', '#define X264_POINTVER "0.165.3222 b35605a"'))
    build.run(['make', '-j'+str(jobs)], x264, env, log)
    build.run(['make', 'install', 'DESTDIR='+str(deps)], x264, env, log)
    ffmpeg = build.unpack(cache/'ffmpeg-9.0.2.tar.xz', deps/'ffmpeg')
    env['PKG_CONFIG_LIBDIR']=str(deps/'h3cli-deps/lib/pkgconfig')
    env['PKG_CONFIG_SYSROOT_DIR']=str(deps)
    build.run(['./configure', '--prefix=/h3cli-media', *lock['ffmpeg_configure']], ffmpeg, env, log)
    build.run(['make', '-j'+str(jobs)], ffmpeg, env, log)
    build.run(['make', 'install', 'DESTDIR='+str(deps)], ffmpeg, env, log)
    for name in ('h3cli-deps', 'h3cli-media'):
        shutil.copytree(deps/name, payload/name, symlinks=False)
    notices = payload/'notices';notices.mkdir()
    shutil.copyfile(ffmpeg/'COPYING.GPLv2', notices/'FFmpeg.txt')
    shutil.copyfile(x264/'COPYING', notices/'x264.txt')
    sources = payload/'inputs';sources.mkdir()
    for name in build.inputs(lock):shutil.copyfile(cache/name, sources/name)
    configs = payload/'configuration';configs.mkdir()
    shutil.copyfile(config, configs/'x264_config.h')
    shutil.copyfile(ffmpeg/'config.h', configs/'ffmpeg-config.h')
    shutil.copyfile(ffmpeg/'ffbuild/config.mak', configs/'ffmpeg-config.mak')
    # The runtime audit deliberately excludes static development archives.
    audit = build.package.audit(payload/'h3cli-media/bin')
    return {'platform': 'macos-arm64', 'toolchain': lock['toolchain'], 'media': audit}
