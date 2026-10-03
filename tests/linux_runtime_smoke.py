#!/usr/bin/env python3
"""Real packaged runtime checks for relocation, state, isolation and server life."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import resource
import shlex
import shutil
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/linux'))
import environment
import package

PROMPT='A red wooden toy boat floating on a quiet pond. Gentle ripples and soft birdsong.'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();release=a.release.resolve();model=a.model.resolve();cache=a.cache.resolve();out=a.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    binary=release/'h3cli-linux-x86_64';runtime=release/'linux-runtime';rows=[]
    manifest=json.loads((runtime/'share/h3cli/runtime.json').read_text())
    for key,tool in (('H3_FFMPEG','ffmpeg'),('H3_SGLANG_INPUT_FFMPEG','ffmpeg'),('H3_FFPROBE','ffprobe')):
        selected=runtime/manifest['environment'][key]
        first=subprocess.check_output([str(selected),'-version'],text=True).splitlines()[0]
        assert first.split()[:3]==[tool,'version','9.0.2'],first
    assert manifest['environment']['H3_FFMPEG']==manifest['environment']['H3_SGLANG_INPUT_FFMPEG']
    env={k:v for k,v in os.environ.items() if not k.startswith(('H3_','H3CLI_','LD_'))}
    env.update(H3_TEST_MAX_EVALUATIONS='6',H3CLI_RUNTIME_CACHE=str(out/'cache-a'))
    def run(name,args,extra=None,expected=0):
        start=time.monotonic();command=[str(x) for x in args]
        with (out/(name+'.log')).open('w') as log:
            result=subprocess.run(command,cwd=ROOT,env=env|(extra or {}),stdout=log,stderr=subprocess.STDOUT,timeout=900)
        row={'name':name,'argv':command,'seconds':time.monotonic()-start,'returncode':result.returncode,'passed':result.returncode==expected}
        rows.append(row);package.write_json(out/'results.json',{'complete':False,'checks':rows})
        if not row['passed']:raise RuntimeError(name+': '+(out/(name+'.log')).read_text()[-5000:])
    base=[str(binary),'-d',str(model),'-p',PROMPT,'--width','256','--height','256','--frames','22','--steps','4','--seed','42']
    run('cold-help',[binary,'--help']);info=json.loads(subprocess.check_output([str(binary)],env=env|{'H3CLI_BUNDLE_INFO':'1'}))
    extracted=Path(info['cache']);stamp={str(p):p.stat().st_mtime_ns for p in extracted.rglob('*')}
    run('warm-help',[binary,'--help']);assert stamp=={str(p):p.stat().st_mtime_ns for p in extracted.rglob('*')}
    run('missing-regression-profile-fails',[binary,'--help'],
        {'H3_TEST_REFERENCE_MEDIA':str(out/'missing-profile.json'),'H3_TEST_REFERENCE_MEDIA_SHA256':'0'*64},expected=2)
    # Output paths and inputs remain relative to the caller. The bundle pins all
    # media/math resources even when stale setup variables and PATH conflict.
    fake=out/'fake-path';fake.mkdir()
    for name in ('ffmpeg','ffprobe'):
        path=fake/name;path.write_text('#!/bin/sh\nexit 99\n');path.chmod(0o755)
    stale={'PATH':str(fake),'LD_LIBRARY_PATH':'/must-not-load','LD_PRELOAD':'/must-not-load.so',
           'H3_SGLANG_CUBLAS_LIBRARY':'/stale/cublas','H3_SGLANG_CUDNN_LIBRARY':'/stale/cudnn',
           'H3_SGLANG_JPEG_LIBRARY':'/stale/jpeg','H3_FFMPEG':'/stale/ffmpeg','H3_FFPROBE':'/stale/ffprobe','H3_SGLANG_INPUT_FFMPEG':'/stale/input'}
    run('pause-stale-environment',[*base,'--stop-after-step','2','--save-sampler-state',out/'paused.h3sample','--save-conditioning',out/'prepared.h3cond'],stale)
    run('resume-cache-a',[binary,'-d',model,'--resume-sampler-state',out/'paused.h3sample','--save-av-state',out/'resumed-a.h3av','-o',out/'resumed-a.mp4'])
    run('resume-cache-b',[binary,'-d',model,'--resume-sampler-state',out/'paused.h3sample','--save-av-state',out/'resumed-b.h3av','-o',out/'resumed-b.mp4'],{'H3CLI_RUNTIME_CACHE':str(out/'cache-b')})
    assert package.sha(out/'resumed-a.h3av')==package.sha(out/'resumed-b.h3av')
    assert package.sha(out/'resumed-a.mp4')==package.sha(out/'resumed-b.mp4')
    run('conditioning-cache-b',[*base,'--load-conditioning',out/'prepared.h3cond','--save-av-state',out/'conditioned.h3av','-o',out/'conditioned.mp4'],{'H3CLI_RUNTIME_CACHE':str(out/'cache-b')})
    assert package.sha(out/'conditioned.h3av')==package.sha(out/'resumed-a.h3av')
    # The core is what server workers execute; changing a loaded resource must
    # fail there too, before GPU startup, not just at the outer launcher.
    cache_b_info=json.loads(subprocess.check_output([str(binary)],env=env|{'H3CLI_RUNTIME_CACHE':str(out/'cache-b'),'H3CLI_BUNDLE_INFO':'1'}))
    cache_b=Path(cache_b_info['cache']);changed=cache_b/'tools/ffprobe'
    with changed.open('r+b') as file:file.seek(-1,2);byte=file.read(1);file.seek(-1,2);file.write(bytes([byte[0]^1]))
    run('direct-worker-rejects-corruption',[cache_b/'bin/h3cli','--help'],{'H3CLI_RUNTIME_ROOT':str(cache_b),'LD_LIBRARY_PATH':str(cache_b/'lib')},expected=2)
    # Pin all distro child paths without providing Python/toolkit/host FFmpeg.
    for distro in ('ubuntu2204','ubuntu2404','debian12'):
        root=environment.prepare(cache,distro)
        bindings=[str(release)+':'+str(release),str(out)+':'+str(out),str(model.parent)+':'+str(model.parent)]
        prefix=environment.command(cache,root,bindings,gpu=True,cwd=str(out))
        run(distro+'-clean-inventory',[*prefix,'sh','-c','test ! -e /usr/local/cuda && test ! -e /usr/bin/python3 && test ! -e /usr/bin/ffmpeg && cat /etc/os-release'])
        run(distro+'-clean-help',[*prefix,binary,'--help'],stale)
    # Real non-root credentials; PRoot's identity emulation alone is not proof.
    uid,gid=(65534,65534) if os.geteuid()==0 else (os.geteuid(),os.getegid())
    nobody=out/'nonroot';nobody.mkdir()
    if os.geteuid()==0:os.chown(nobody,uid,gid)
    root=environment.prepare(cache,'ubuntu2204')
    # umoci's OCI bundle parent is deliberately private. Test a real unprivileged
    # process in a separate stock rootfs rather than relaxing the build cache.
    nonroot_root=out/'nonroot-userland'
    subprocess.run(['cp','-a','--reflink=auto',str(root),str(nonroot_root)],check=True)
    nonroot_root.chmod(0o755);root=nonroot_root
    prefix=environment.command(cache,root,[str(release)+':'+str(release),str(out)+':'+str(out),str(model.parent)+':'+str(model.parent)],gpu=True,cwd=str(nobody),identity=f'{uid}:{gid}')
    drop=['setpriv',f'--reuid={uid}',f'--regid={gid}','--clear-groups'] if os.geteuid()==0 else []
    run('nonroot-id',[*drop,*prefix,'id'])
    run('nonroot-render',[*drop,*prefix,binary,'-d',model,'-p',PROMPT,'--width','256','--height','256','--frames','22','--steps','2','-o',nobody/'video.mp4'],{'H3CLI_RUNTIME_CACHE':str(nobody/'runtime-cache'),'HOME':str(nobody)})
    streams=json.loads(subprocess.check_output([str(runtime/'tools/ffprobe'),'-v','error','-show_streams','-of','json',str(nobody/'video.mp4')]))['streams']
    video=next(s for s in streams if s['codec_type']=='video');assert (video['width'],video['height'],int(video['nb_frames']))==(256,256,22)
    package.write_json(out/'results.json',{'complete':True,'passed':all(r['passed'] for r in rows),'checks':rows,'release_sha256':package.sha(binary),'cache_id':info['runtime_id'],'relocated_state_exact':True,'nonroot_uid':uid,'nonroot_streams':streams})

if __name__=='__main__':main()
