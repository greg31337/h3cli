#!/usr/bin/env python3
"""Mutable single-pipeline complete-request runner. Frozen reference tests stay separate."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse,json,os,re,signal,subprocess,sys,threading,time
from pathlib import Path
from cuda_reference_regression import regression_config,runtime_env,sha,write,source_files,fingerprint,PROMPT
from cuda_sglang import NVML,CONDITIONED,HELD_OUT,validate

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--attention',choices=['default','sage2++','sage3','sol'],default='default')
 p.add_argument('--quant',choices=['off','fp8','nvfp4'],default='off');p.add_argument('--quant-cache',type=Path)
 p.add_argument('--case',choices=['C0',*CONDITIONED,*HELD_OUT],default='C0');p.add_argument('--frames',type=int,choices=[124,243,362],default=124)
 p.add_argument('--width',type=int,choices=[640,1344],default=640);p.add_argument('--steps',type=int,choices=range(1,7),default=6)
 p.add_argument('--capture',action='store_true');p.add_argument('--qkv',action='store_true');p.add_argument('--profile',action='store_true')
 p.add_argument('--decode',type=Path);p.add_argument('--continue-from',type=Path);p.add_argument('--env',action='append',default=[])
 p.add_argument('--reference-alias',action='store_true');p.add_argument('--ref-image-size',choices=['match','high','max'],default='max')
 p.add_argument('--model',type=Path);p.add_argument('--fixtures',type=Path)
 a=p.parse_args()
 if a.width==1344 and (a.steps>2 or a.frames!=362):p.error('high-resolution stress is limited to 362 frames and at most two evaluations')
 src=a.source.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
 config=regression_config(src,a.model,a.fixtures);env=runtime_env();env['H3_TEST_MAX_EVALUATIONS']=str(a.steps)
 case=CONDITIONED.get(a.case,HELD_OUT.get(a.case,dict(prompt=PROMPT,seed=42)))
 cmd=[str(src/'bin/h3cli'),'-d',config['model'],'-p',case['prompt'],'--seed',str(case['seed']),'--width',str(a.width),'--height',str(768 if a.width==1344 else 480),'--frames',str(a.frames),'--steps',str(a.steps),'-o',str(out/'video.mp4')]
 if a.decode:cmd=[str(src/'bin/h3cli'),'-d',config['model'],'--decode-av-state',str(a.decode.resolve()),'-o',str(out/'video.mp4')]
 if a.reference_alias:
  cmd+=[]
 if a.continue_from:
  assert not a.decode
  cmd+=['--continue-from',str(a.continue_from.resolve()),'--keep-continuation-prefix']
 if a.attention!='default':cmd+=['--cuda-attention',a.attention]
 if a.quant!='off':
  assert not a.decode and a.quant_cache
  cmd+=['--cuda-denoise-quant',a.quant,'--cuda-denoise-quant-cache',str(a.quant_cache.resolve())]
 inputs=[]
 if a.continue_from:inputs.append(dict(path=str(a.continue_from.resolve()),sha256=sha(a.continue_from)))
 for c in case.get('conditions',[]):
  path=Path(config['fixtures'])/c['uri'];inputs.append(dict(path=str(path),sha256=sha(path)))
  flag=('--first-frame' if c['frame_index']==0 else '--last-frame') if c['role']=='keyframe' else '--ref-image' if c['type']=='image' else '--ref-video'
  cmd+=[flag,str(path)]
 if case.get('task')=='ref2va':cmd+=['--ref-image-size',a.ref_image_size]
 if a.capture:
  for folder in ['steps','preparation']:(out/folder).mkdir()
  env.update(H3_TEST_NATIVE_STEP_DIR=str(out/'steps'),H3_TEST_SGLANG_DIR=str(out/'preparation'),H3_SGLANG_CAPTURE_STEPS='none')
  cmd+=['--save-av-state',str(out/'final.h3av')]
 if a.qkv:
  (out/'qkv').mkdir();env.update(H3_TEST_ATTENTION_CAPTURE_DIR=str(out/'qkv'),H3_TEST_ATTENTION_CAPTURE_MAX_SEQUENCE='131072',H3_TEST_ATTENTION_CAPTURE_BLOCKS='24',H3_TEST_ATTENTION_CAPTURE_STEPS='3')
 if a.profile:env['H3_PROFILE']='1'
 for item in a.env:k,v=item.split('=',1);assert k.startswith('H3_');env[k]=v
 identity=fingerprint(source_files(src));binary=sha(src/'bin/h3cli')
 record=dict(passed=False,source_sha256=identity,binary_sha256=binary,case=a.case,frames=a.frames,evaluations=a.steps,width=a.width,height=768 if a.width==1344 else 480,quant=a.quant,attention=a.attention,command=cmd,inputs=inputs,environment=env,instrumented=a.capture or a.qkv or a.profile or bool(a.env))
 write(out/'command.json',record);write(out/'result.json',record)
 monitor=NVML();idle=monitor.sample();stop=threading.Event();errors=[];samples=[]
 active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 if active:raise RuntimeError('GPU has active compute processes: '+active)
 start=time.monotonic()
 with (out/'render.log').open('w') as log,(out/'telemetry.jsonl').open('w') as telemetry:
  child=subprocess.Popen(cmd,cwd=src,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  def collect():
   try:
    host={};last_host=0
    while not stop.is_set():
     tick=time.monotonic()
     if tick-last_host>.2:host=monitor.host_sample(child.pid);last_host=tick
     row=monitor.sample(host=False)|host|dict(elapsed_s=time.monotonic()-start)
     samples.append(row);telemetry.write(json.dumps(row)+'\n');telemetry.flush();stop.wait(.02)
   except BaseException as exc:errors.append(str(exc))
  thread=threading.Thread(target=collect);thread.start()
  try:rc=child.wait(timeout=1200)
  except BaseException:
   os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=10);raise
  finally:stop.set();thread.join()
 record.update(returncode=rc,wall_seconds=time.monotonic()-start,complete_playable_seconds=time.monotonic()-start,idle=idle,telemetry_errors=errors)
 write(out/'result.json',record)
 assert rc==0 and not errors and samples
 if a.decode:
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(out/'video.mp4')],env=env))
  assert any(s['codec_type']=='video' and int(s['nb_read_frames'])==a.frames for s in probe['streams'])
  assert any(s['codec_type']=='audio' for s in probe['streams']);validation=dict(media_valid=True)
 else:validation=validate(out,'native',a.frames,a.steps,a.width,768 if a.width==1344 else 480)
 log=(out/'render.log').read_text()
 dispatch=re.findall(r'main DiT attention counters requested=(\S+) dense=(\d+) sage2=(\d+) sage3=(\d+) sol=(\d+) protected_bypass=(\d+) workspace=(\d+)',log)
 projections=re.findall(r'projection counters requested=(\S+) recipe=(\d+) native_calls=(\d+) cache_hits=(\d+) prepared=(\d+)',log)
 if not a.decode:
  assert len(dispatch)==len(projections)==1
  d=dispatch[0];q=projections[0];assert d[0]==a.attention and q[0]==a.quant
  if a.attention!='default':assert int(d[{'sage2++':2,'sage3':3,'sol':4}[a.attention]])>0
  if a.quant=='off':assert q[2:]==('0','0','0')
  else:assert int(q[2])>0
 record.update(attention_dispatch=dispatch,projection_dispatch=projections)
 record.update(validation=validation,peak_vram_bytes=max(x['gpu_used_bytes'] for x in samples),peak_host_rss_bytes=max(x.get('process_rss_bytes',0) for x in samples),largest_sample_gap_seconds=max(b['elapsed_s']-x['elapsed_s'] for x,b in zip(samples,samples[1:])))
 assert record['largest_sample_gap_seconds']<1.0
 assert sha(src/'bin/h3cli')==binary and fingerprint(source_files(src))==identity
 record.update(passed=True,video_sha256=sha(out/'video.mp4'))
 write(out/'result.json',record);print(json.dumps({k:record[k] for k in ['passed','case','frames','quant','attention','wall_seconds','peak_vram_bytes']}))
if __name__=='__main__':main()
