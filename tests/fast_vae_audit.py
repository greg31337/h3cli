#!/usr/bin/env python3
# Historical evidence renderer only. This is not a current qualification gate.
"""Retain corpus checksums, invariance, peaks and final implementation identity."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse, hashlib, json, pathlib, platform, shutil, statistics, subprocess, tarfile, time
import numpy as np
from fractions import Fraction

def sha(p):
    with pathlib.Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p,default=None):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default
def save(p,value):p.write_text(json.dumps(value,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='outputs/fast-vae/cuda');a=p.parse_args();r=pathlib.Path(a.root)
    records={p.parent.name:read(p,{}) for p in r.glob('*/record.json')}
    hashes={str(p.relative_to(r)):sha(p) for suffix in ['*.h3av','*.moments','*.epsilon','*.sample'] for p in r.rglob(suffix)}
    invariance=[]
    def equal(name,paths):
        values=[hashes.get(x) for x in paths];invariance.append(dict(name=name,paths=paths,sha256=values,passed=all(values) and len(set(values))==1))
    for case in ['C1','C2']:
        equal(case+' initial decoder-only policy',[case+'-'+mode+'/output.h3av' for mode in ['reference','balanced']])
        equal(case+' final six-step state',[case+'-final-'+mode+'/output.h3av' for mode in ['reference','balanced','tensorrt','legacy']])
    for size in ['match','max']:equal('Ref2VA '+size+' encoder/posterior conditioning, fixed GEMM selection',['C5-image-'+size+'-'+mode+'-fixed/output.h3av' for mode in ['reference','balanced']])
    for session in ['session-invariance','final-session']:
        equal(session,[session+'/'+mode+'.h3av' for mode in ['reference','balanced','preview']])
    resume='C5-resume-retest' if (r/'C5-resume-retest/record.json').exists() else 'C5-resume'
    equal('Decoder-policy change on exact sampler resume',[resume+'/reference.h3av',resume+'/resumed.h3av'])
    equal('Cached same-shape repeat',['session-final-repeat/shape-change.h3av','session-final-repeat/shape-repeat.h3av'])
    equal('Final cached decoder-policy isolation',['session-final-repeat/'+mode+'.h3av' for mode in ['reference','balanced','preview']])
    encoder=[]
    for case in ['image-match','image-max','video-short','video-tail']:
        entry={'case':case}
        for suffix in ['moments','epsilon','sample']:
            paths=['encoder-'+case+'-'+variant+'/output.'+suffix for variant in ['baseline','reuse']]
            values=[hashes.get(x) for x in paths];entry[suffix+'_bitwise']=all(values) and values[0]==values[1]
        for variant in ['baseline','reuse']:
            path=r/('encoder-'+case+'-'+variant)/'stdout.log'
            if path.exists():
                for line in path.read_text().splitlines():
                    try:entry[variant]=json.loads(line)
                    except json.JSONDecodeError:pass
        encoder.append(entry)
    exact=[]
    anchor=r/'final-tile-fused/output.f32'
    for candidate in ['unfused','pageable','fast']:
        path=r/('final-tile-'+candidate)/'output.f32'
        if not anchor.exists() or not path.exists():exact.append(dict(candidate=candidate,passed=False,reason='missing output'));continue
        x=np.fromfile(anchor,'<f4');y=np.fromfile(path,'<f4');delta=x.astype('float64')-y
        exact.append(dict(candidate=candidate,passed=bool(np.array_equal(x,y)),max_abs=float(abs(delta).max()),rel_l2=float(np.linalg.norm(delta)/np.linalg.norm(x.astype('float64'))),sha256=[sha(anchor),sha(path)]))
    media=[]
    for path in sorted(r.rglob('*.mp4')):
        command=['ffprobe','-v','error','-count_frames','-show_entries','stream=codec_type,width,height,nb_read_frames,duration,r_frame_rate','-of','json',str(path)]
        streams=json.loads(subprocess.check_output(command,text=True))['streams'];video=next(x for x in streams if x['codec_type']=='video')
        parent=path.parent.name;expected=None
        if parent.startswith(('C1-','C2-')):expected=243
        elif parent.startswith('C3-'):expected=362
        elif parent.startswith(('C0-','C4-')) or parent=='pixels':expected=39
        elif parent in ['final-session','session-invariance','session-final-repeat']:expected=39 if path.stem.startswith('shape-') else 22
        elif parent.startswith('C5-image') or parent=='C5-first-last':expected=22
        elif parent.startswith('C5-videos') or parent=='C5-continue-source':expected=90
        elif parent=='C5-video-inputs':expected=48
        elif parent.startswith('C5-continuation'):expected=51
        elif parent.startswith('C5-resume'):expected=22
        elif parent=='trt-quality':expected=22
        count=int(video['nb_read_frames']);fps=float(Fraction(video['r_frame_rate']))
        duration=float(video['duration']);audio=[s for s in streams if s['codec_type']=='audio']
        audio_delta=max((abs(float(s['duration'])-duration) for s in audio),default=0.)
        dimensions=(640,480) if parent.startswith('C1-') else (1344,768) if parent.startswith(('C2-','C3-')) else None
        dimensions_ok=not dimensions or (video['width'],video['height'])==dimensions
        passed=(count==expected if expected else count>0) and fps==24 and abs(duration-count/24)<.001 and audio_delta<.085 and dimensions_ok
        media.append(dict(path=str(path.relative_to(r)),sha256=sha(path),streams=streams,expected_frames=expected,expected_dimensions=dimensions,audio_video_duration_delta=audio_delta,passed=passed))
    steps=[];aborted_attempts=[];sampling=[]
    for name,record in records.items():
        if record.get('requested_steps') is not None:
            step=dict(id=name,requested=record['requested_steps'],completed=record.get('completed_steps'),passed=record['requested_steps']==record.get('completed_steps')==6 and record.get('returncode')==0)
            if record.get('returncode')!=0 and not list((r/name).glob('*.mp4')):aborted_attempts.append(step)
            else:steps.append(step)
        values=read(r/name/'memory.json',[]);times=[x[0] for x in values]
        sampling.append(dict(id=name,samples=len(values),max_gap_seconds=max((b-a for a,b in zip(times,times[1:])),default=None),sampled_vram_peak=record.get('peak_vram_bytes'),sampled_main_rss_peak=record.get('peak_rss_bytes')))
    api_steps=[]
    for session in ['session-invariance','final-session','session-final-repeat',resume]:
        path=r/session/'stdout.log'
        if path.exists():
            for line in path.read_text().splitlines():
                try:x=json.loads(line)
                except json.JSONDecodeError:continue
                if 'completed_steps' in x:api_steps.append(dict(session=session,**x,passed=x['completed_steps']==6))
    quality={str(p.parent.relative_to(r)):read(p)['gates'] for p in r.glob('*/metrics.json') if 'gates' in read(p,{})}
    required_jobs=['final-build','final-session','final-operators','final-policy','final-preview',
        'final-posterior','final-tiles','final-memory','final-continuation-budget-retest',
        'final-bridge-budget-retest','final-sampler','final-refvideo','final-reference-layout',
        'final-encoder-conv','final-delivery','reference-repeatability','final-lifetime',
        'final-graph-fallback','trt-identity','balanced-streaming','trt-lifetime',
        'trt-shape-fallback','final-tile-forced-reference','C5-resume-retest',
        'C5-first-last','C5-image-match-reference','C5-image-match-balanced',
        'C5-image-max-reference','C5-image-max-balanced','C5-image-tensorrt','C5-videos-retest',
        'C5-continue-source','C5-continuation','C5-continuation-decode','optional-build',
        'C3-repeat','C3-trt-decode','still-regression','torch-fp32','torch-fp16',
        'trt-fp32','trt-fp16','batch-1','batch-2','batch-4','C1-tiny-bench','C2-tiny-bench','model-inventory',
        'session-final-repeat','C5-image-match-reference-fixed','C5-image-match-balanced-fixed',
        'C5-image-max-reference-fixed','C5-image-max-balanced-fixed','C5-image-match-reference-repeat']
    uncontrolled=[]
    for size in ['match','max']:
        for variant in ['balanced']+(['reference-repeat'] if size=='match' else []):
            paths=[r/('C5-image-'+size+'-'+mode)/'output.h3av' for mode in ['reference',variant]]
            if not all(x.is_file() for x in paths):continue
            x,y=[np.fromfile(path,'<f4',offset=160).astype('float64') for path in paths]
            uncontrolled.append(dict(case=size,comparison=variant,rel_l2=float(np.linalg.norm(x-y)/np.linalg.norm(x)),max_abs=float(abs(x-y).max()),bitwise=bool(np.array_equal(x,y)),note='Original timed-autotuning run; retained diagnostic, separate from fixed-selection qualification.'))
    qualification=[dict(id=name,status=records.get(name,{}).get('status','missing'),
        passed=records.get(name,{}).get('status')=='pass') for name in required_jobs]
    result=dict(created_unix=time.time(),state_and_encoder_hashes=hashes,latent_invariance=invariance,encoder=encoder,delivery_and_fusion_bitwise=exact,media=media,generated_steps=steps,aborted_attempts_without_media=aborted_attempts,cached_api_steps=api_steps,sampling=sampling,quality=quality,qualification_records=qualification,uncontrolled_ref2va_repeats=uncontrolled,
        C3_provenance='Decode-only repeated-tail expansion of verified six-step C2 state, 243 to 362 frames; not independent 362-frame generation.',
        C4_provenance='Denoiser-free deterministic pans of inputs/2.jpg and inputs/1.jpg with overlaid text, encoded through the original reference-video encoder.')
    save(r/'audit.json',result)
    src={str(p):sha(p) for pattern in ['src/**/*.c','src/**/*.h','src/**/*.cu','src/**/*.cuh','src/**/*.cpp','src/**/*.m','Makefile','tests/fast_vae*','tests/test_fast_vae.c','scripts/build_vae_engine.py'] for p in pathlib.Path('.').glob(pattern) if p.is_file() and p.suffix not in ['.o','.d']}
    def output(args):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT).strip()
    engine_path=r/'engine-path.txt';engine=read(pathlib.Path(engine_path.read_text().strip()+'.json')) if engine_path.exists() else None
    inventory=dict(platform=platform.platform(),gpu=output(['nvidia-smi','--query-gpu=name,uuid,compute_cap,memory.total,driver_version','--format=csv']),cuda=output(['/usr/local/cuda/bin/nvcc','--version']),compiler=output(['cc','--version']),ffmpeg=output(['ffmpeg','-version']).splitlines()[0],python=platform.python_version(),source_sha256=src,source_archive_sha256=sha('final-source.tar.gz'),engine=engine,base_git_commit='2abc877cd65e8b2977772b38731480d26c4f8274',build_flags='CUDA_ARCH=120 CUDA_SOL=1 CUDA_VAE_TRT=1; native-only build checked separately',local_metal='M4 Max, 128 GiB; build/host compatibility only, M7 not implemented',source_provenance=dict(vpipe='aad3a10e654af71ae46cb627e5f4f7da22ff0593',minimax='d21241f0a4b3acbb34c97dae47fa417b7065e438',comfy_h3_trt='4360e00867eca86ab61b3899216c0ec281367b46',tensorrt_headers='b8db91e15be2cae4465ac17fab19e0f969e45407',implementation='Original thin CUDA/C++/ONNX implementation; no VPIPE or ComfyUI code copied.'))
    with tarfile.open('final-source.tar.gz') as archive:
        members={m.name.removeprefix('./'):m for m in archive if m.isfile()}
        production=[name for name in src if name.startswith('bin/h3cli')]
        changed=[name for name in production if name not in members or hashlib.sha256(archive.extractfile(members[name]).read()).hexdigest()!=src[name]]
    inventory['production_archive_check']=dict(files=len(production),changed=changed,passed=not changed,note='Production source frozen before final build; closing test harnesses and documentation may be newer than the archive.')
    inventory['model_file_identities']=read(r/'model-files.json')
    disk=shutil.disk_usage('.')
    inventory['closing_capacity']=dict(measured_unix=time.time(),disk_total_bytes=disk.total,disk_free_bytes=disk.free,gpu_memory=output(['nvidia-smi','--query-gpu=memory.total,memory.free,memory.used','--format=csv']))
    save(r/'inventory.json',inventory)
    required=invariance+exact+media+steps+api_steps+qualification
    failures=[x for x in required if not x['passed']]+[x for x in encoder if not all(x.get(k+'_bitwise') for k in ['moments','epsilon','sample'])]
    required_quality=['C0-quality','C1-quality','C2-quality','C3-quality','C1-trt-quality','C2-trt-quality','C3-trt-quality','C4-faces-balanced-quality','C4-faces-tensorrt-quality','C4-texture-balanced-quality','C4-texture-tensorrt-quality']
    failures += [dict(case=case,quality_gates=quality.get(case),passed=False) for case in required_quality if not quality.get(case) or not all(quality[case].values())]
    if changed:failures.append(inventory['production_archive_check'])
    print('Audit:',len(required),'invariance/media/provenance checks;',len(encoder),'encoder pairs;',len(failures),'failures')
    if failures:print(json.dumps(failures,indent=2));raise SystemExit(1)
if __name__=='__main__':main()
