#!/usr/bin/env python3
"""Replay the complete CUDA reference suite against recorded golden outputs.

Model/fixture locations are configurable; build and runtime dependencies come
from the caller's environment in source mode. Artifact mode verifies a previously
built release and its matching probes. GPU time limit starts after build/verification.
"""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import reference_media

PROMPT = ('Cinematic medium shot of a woman passionately playing a grand piano in a '
          'sunlit concert hall. Her fingers move across the keys as the camera slowly '
          'glides sideways. Warm natural lighting, realistic details, flowing piano music.')
BUILD = ['bin/h3cli', 'bin/cuda_reference_probe', 'bin/cuda_sglang_patch_test',
         'bin/cuda_sglang_media_input', 'bin/cuda_sglang_vision', 'bin/cuda_sglang_encoder',
         'bin/cuda_sglang_soundtrack_input', 'bin/cuda_sglang_audio_encode',
         'bin/cuda_sglang_vae', 'bin/cuda_sglang_audio']
EXT = {'.c','.h','.cu','.cuh','.cpp','.m','.metal','.inc','.def','.py','.sh','.json','.hpp','.txt'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def write(p,v):
    Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def source_files(root):
    paths=[p for p in root.iterdir() if p.is_file() and (p.suffix in EXT or p.name=='Makefile')]
    for d in ('src','tests','scripts','third_party'):
        paths += [p for p in (root/d).rglob('*') if p.is_file() and (d=='third_party' or p.suffix in EXT or p.parent==root/'scripts/linux')
                  and p.suffix not in ('.o','.d','.pyc') and '__pycache__' not in p.parts
                  and p!=root/'src/metal/native_attention.inc']
    return {str(p.relative_to(root)):sha(p) for p in sorted(paths)}

def fingerprint(files):
    return hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def checked_files(root,files):
    for name,expected in files.items():
        p=root/name
        if not p.is_file() or sha(p)!=expected: raise ValueError('missing or changed fixture: '+str(p))

def regression_config(source,model=None,fixtures=None):
    return dict(model=str(Path(model or os.environ.get('H3_REFERENCE_MODEL') or source/'models/MiniMax-H3').expanduser().resolve()),
                fixtures=str(Path(fixtures or os.environ.get('H3_REFERENCE_FIXTURES') or source/'tests/fixtures/cuda-reference').expanduser().resolve()))

def runtime_env():
    env=dict(os.environ)
    env.setdefault('LANG','C.UTF-8')
    env['H3_TEST_MAX_EVALUATIONS']='6'
    env['H3_OFFLINE']='1'
    env['H3_TEST_REFERENCE_ENCODING']='1'
    return env

def source_media_config(source):
    profile=Path(os.environ.get('H3_REFERENCE_MEDIA') or source/'outputs/setup/reference-media/profile.json')
    if not profile.is_file():
        raise ValueError('Missing regression media profile; run python3 scripts/setup_reference_media.py '
                         'or set H3_REFERENCE_MEDIA to its profile.json')
    env,identities=reference_media.read_profile(profile)
    return dict(_runtime_env=env,_identities=identities,_decoder=env['H3_SGLANG_INPUT_FFMPEG'])

def execute(cmd,cwd,env,log,deadline):
    remain=deadline-time.monotonic()
    if remain<=0: raise TimeoutError('12-minute gate budget exhausted')
    with log.open('xb') as f:
        p=subprocess.Popen([str(x) for x in cmd],cwd=cwd,env=env,stdin=subprocess.DEVNULL,
                           stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            rc=p.wait(timeout=remain)
        except BaseException:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
    if rc: raise RuntimeError(f'{log.name}: command exited {rc}')

def artifact_config(source, runtime, validation, artifact=None, reference=True):
    runtime=Path(runtime).resolve();validation=Path(validation).resolve()
    manifest_path=runtime/'share/h3cli/runtime.json'
    record_path=validation/'manifest.json'
    record=json.loads(record_path.read_text())
    manifest=json.loads(manifest_path.read_text())
    if record.get('schema')!=1 or manifest.get('schema')!=1:
        raise ValueError('unsupported artifact provenance schema')
    if record.get('source_sha256')!=fingerprint(source_files(source)):
        raise ValueError('artifact does not match source snapshot')
    if record.get('runtime_manifest_sha256')!=sha(manifest_path):
        raise ValueError('runtime manifest differs from validation build')
    def checked_path(root,name):
        path=Path(name)
        if path.is_absolute() or not path.parts or any(x in ('.','..') for x in path.parts):
            raise ValueError('unsafe artifact manifest path')
        resolved=(root/path).resolve(strict=True)
        if not resolved.is_relative_to(root):raise ValueError('artifact escapes its root')
        return resolved
    identities={str(manifest_path):sha(manifest_path),str(record_path):sha(record_path)}
    for name,item in manifest['files'].items():
        path=checked_path(runtime,name)
        if not path.is_file() or path.stat().st_size!=item['size'] or sha(path)!=item['sha256']:
            raise ValueError('missing or changed runtime file: '+name)
        identities[str(path)]=item['sha256']
    if set(record.get('probes',{}))!=set(BUILD[1:]):
        raise ValueError('incomplete or unexpected validation probe set')
    binaries={'bin/h3cli':str(runtime/'bin/h3cli')}
    for name,expected in record['probes'].items():
        path=checked_path(validation,name)
        if sha(path)!=expected:raise ValueError('changed validation probe: '+name)
        binaries[name]=str(path);identities[str(path)]=expected
    if artifact is not None:
        artifact=Path(artifact).resolve(strict=True)
        if sha(artifact)!=record.get('artifact_sha256'):raise ValueError('changed release executable')
        identities[str(artifact)]=record['artifact_sha256'];binaries['bin/h3cli']=str(artifact)
    if str(runtime/'bin/h3cli') not in identities:raise ValueError('runtime core missing from manifest')
    env={}
    for name,relative in manifest['environment'].items():
        path=checked_path(runtime,relative)
        if str(path) not in identities:raise ValueError('unrecorded runtime resource: '+name)
        env[name]=str(path)
    for name in ('H3_FFMPEG','H3_FFPROBE','H3_SGLANG_INPUT_FFMPEG','H3_SGLANG_CUBLAS_LIBRARY','H3_SGLANG_CUDNN_LIBRARY','H3_SGLANG_JPEG_LIBRARY'):
        if name not in env:raise ValueError('missing runtime selection: '+name)
    env.update(LD_LIBRARY_PATH=str(runtime/'lib'),H3CLI_RUNTIME_ROOT=str(runtime))
    if reference:
        if not record.get('reference_media_sha256'):raise ValueError('missing regression media provenance')
        media_env,media_ids=reference_media.read_profile(validation/'media/profile.json',record['reference_media_sha256'])
        env.update(media_env);identities.update(media_ids)
    return dict(_binaries=binaries,_runtime_env=env,_identities=identities,
                _decoder=env['H3_SGLANG_INPUT_FFMPEG'])

def verify_artifact_identities(identities):
    for path,expected in identities.items():
        if not Path(path).is_file() or sha(path)!=expected:
            raise ValueError('artifact changed during validation: '+path)

def collect(source,out,config):
    """Run every recorded production probe and the complete reference render."""
    started=time.monotonic();deadline=started+720
    env=runtime_env();fixtures=Path(config['fixtures']);model=Path(config['model'])
    if '_runtime_env' in config:
        for key in ('LD_PRELOAD','LD_AUDIT','LD_DEBUG','LD_DEBUG_OUTPUT','LD_PROFILE','H3CLI_BUNDLE_INFO'):
            env.pop(key,None)
        env.update(config['_runtime_env'])
    binaries=config.get('_binaries',{x:str(source/x) for x in BUILD})
    data=out/'data';data.mkdir();commands=[]
    executor=config.get('_executor',[])
    def run(name,args,extra=None):
        t=time.monotonic();cmd=[binaries[args[0]],*args[1:]]
        record=dict(name=name,argv=[str(x) for x in cmd]);commands.append(record)
        write(out/'commands.json',commands)
        execute([*executor,*cmd],source,env|(extra or {}),out/(name+'.log'),deadline)
        record['seconds']=time.monotonic()-t;write(out/'commands.json',commands)
    run('R0-R1-R4-probe',['bin/cuda_reference_probe',data,fixtures])
    run('R1-patch',['bin/cuda_sglang_patch_test'])
    rows=[json.loads(s) for s in (out/'R1-patch.log').read_text().splitlines() if s.startswith('{')]
    if [(r['rows'],r['mapped']) for r in rows]!=[(257,True),(257,False),(49932,False),(49933,True),(113296,True),(107856,False)] or not all(r['passed'] and r['exact_bf16'] for r in rows):
        raise ValueError('missing/failed patch geometry')
    for name,w,h in [('match',640,480),('max',2048,2720)]:
        run('R2-image-'+name,['bin/cuda_sglang_media_input',fixtures/'portrait.jpg',1365,1821,w,h,data/('image-'+name+'.f32')])
    vision=data/'vision';vision.mkdir()
    run('R2-vision',['bin/cuda_sglang_vision',model/'Ref2VA/text_encoder',fixtures/'portrait.jpg',vision])
    run('R2-image-encoder',['bin/cuda_sglang_encoder',model/'FL2VA/video_vae/source',fixtures/'first.png',data/'image-moments.f32'])
    run('R2-video-encoder',['bin/cuda_sglang_encoder',model/'Ref2VA/video_vae/source',fixtures/'reference.mp4',data/'video-moments.f32','video'])
    run('R2-soundtrack',['bin/cuda_sglang_soundtrack_input',fixtures/'reference.mp4',data/'soundtrack.f32'])
    run('R2-audio-encoder',['bin/cuda_sglang_audio_encode',model/'Ref2VA/audio_vae',data/'soundtrack.f32',data/'audio-latent.f32'])
    run('R2-vae-tile',['bin/cuda_sglang_vae',model/'FL2VA/video_vae/source',fixtures/'final.h3av','default',data/'tile.f32','1','tile'])
    run('R2-audio-decode',['bin/cuda_sglang_audio',model/'FL2VA/audio_vae',fixtures/'final.h3av',data/'audio-decode.f32'])
    steps=data/'steps';steps.mkdir();capture=data/'preparation';capture.mkdir()
    run('R3-C0',['bin/h3cli','-d',model,'-p',PROMPT,'--seed','42','--width','640','--height','480',
        '--frames','124','--steps','6','--save-av-state',out/'final.h3av','-o',out/'video.mp4'],
        {'H3_TEST_NATIVE_STEP_DIR':str(steps),'H3_TEST_SGLANG_DIR':str(capture),'H3_SGLANG_CAPTURE_STEPS':'none'})
    if 'denoise' not in (out/'R3-C0.log').read_text() or {x.name for x in steps.iterdir()}!={f'step-{i:03d}-{m}-{r}.f32' for i in range(1,7) for m in ('video','audio') for r in ('input','velocity','latent')}:
        raise ValueError('incomplete C0 trajectory')
    # Compare decoded content, never container metadata. The saved-state header
    # includes build identity and is intentionally excluded from numeric goldens.
    for name,stream,fmt,codec in [('rgb','0:v:0','rawvideo','rgb24'),('pcm','0:a:0','f32le','pcm_f32le')]:
        cmd=[config.get('_decoder','ffmpeg'),'-v','error','-i',str(out/'video.mp4'),'-map',stream]
        cmd+=['-pix_fmt',codec] if name=='rgb' else ['-acodec',codec]
        cmd+=['-f',fmt,str(data/name)]
        execute([*executor,*cmd],source,env,out/('decode-'+name+'.log'),deadline)
    if (data/'rgb').stat().st_size!=640*480*3*124:raise ValueError('incomplete video frames')
    files={str(p.relative_to(data)):sha(p) for p in sorted(data.rglob('*')) if p.is_file()}
    if source_files(source)!=config['_source_files']:raise ValueError('source changed during test')
    if time.monotonic()>deadline:raise TimeoutError('12-minute gate budget exhausted during validation')
    return dict(files=files,seconds=time.monotonic()-started,commands=commands,patch_cases=rows,
                binary_sha256={x:sha(binaries[x]) for x in BUILD})

def verify_outputs(actual,expected):
    if set(actual)!=set(expected):raise ValueError('missing/unexpected regression artifacts')
    bad=[p for p in expected if actual[p]!=expected[p]]
    if bad:raise ValueError('reference drift: '+', '.join(bad))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',type=Path,help='Model directory (H3_REFERENCE_MODEL or SOURCE/models/MiniMax-H3)')
    p.add_argument('--fixtures',type=Path,help='Recorded inputs (H3_REFERENCE_FIXTURES or SOURCE/tests/fixtures/cuda-reference)')
    p.add_argument('--runtime',type=Path,help='Validate a prebuilt runtime instead of rebuilding')
    p.add_argument('--validation',type=Path,help='Matching validation probes and manifest')
    p.add_argument('--artifact',type=Path,help='Final single-file entry point (requires --runtime/--validation)')
    p.add_argument('--executor-config',type=Path,help='Artifact-only JSON argv prefix for a clean userland; applies to every probe, C0 and decoder')
    a=p.parse_args()
    if bool(a.runtime)!=bool(a.validation) or (a.artifact and not a.runtime):
        p.error('--runtime and --validation are required together; --artifact also requires both')
    if a.executor_config and not a.runtime:p.error('--executor-config requires prebuilt artifact mode')
    source=a.source.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    result={'passed':False};begin=time.monotonic()
    try:
        manifest=source/'tests/cuda_reference/manifest.json'
        m=json.loads(manifest.read_text())
        if m.get('schema')!=2 or not m.get('fixtures') or not m.get('goldens'):
            raise ValueError('recorded fixture and output hashes are required (manifest schema 2)')
        config=regression_config(source,a.model,a.fixtures)
        checked_files(Path(config['fixtures']),m['fixtures'])
        if not Path(config['model']).is_dir():raise ValueError('missing model directory: '+config['model'])
        original=source_files(source);result['source_sha256']=fingerprint(original)
        result['golden_manifest_sha256']=sha(manifest);result['source_files']=original
        result['config']=config
        if a.runtime:
            build=source
            config=config|artifact_config(source,a.runtime,a.validation,a.artifact)
            result['mode']='single-file' if a.artifact else 'runtime'
            result['artifact_identities']=dict(config['_identities'])
            if a.executor_config:
                executor=json.loads(a.executor_config.read_text())
                if not isinstance(executor,list) or not executor or not all(isinstance(x,str) and '\0' not in x for x in executor):
                    raise ValueError('executor configuration must be a nonempty argv array')
                config['_executor']=executor
                config['_identities'][str(a.executor_config.resolve())]=sha(a.executor_config)
                result['executor']=executor
                result['artifact_identities']=dict(config['_identities'])
        else:
            config=config|source_media_config(source)
            result['media_identities']=dict(config['_identities'])
            build=out/'build';build.mkdir()
            for name in original:
                target=build/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
            build_env=dict(os.environ)
            execute(['make','-j8',*BUILD],build,build_env,out/'build.log',time.monotonic()+1800)
            result['mode']='source'
        config=config|{'_source_files':source_files(build)}
        result.update(collect(build,out,config))
        verify_artifact_identities(config['_identities'])
        verify_outputs(result['files'],m['goldens'])
        if source_files(source)!=original:raise ValueError('input source changed while testing')
        checked_files(Path(config['fixtures']),m['fixtures'])
        result['passed']=True
        result['total_with_build_seconds']=time.monotonic()-begin
    except Exception as exc:
        result.update(passed=False,error=str(exc));print(str(exc),flush=True)
    write(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k in ('passed','error','seconds','source_sha256','golden_manifest_sha256')},indent=2))
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
