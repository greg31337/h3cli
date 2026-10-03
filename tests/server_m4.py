#!/usr/bin/env python3
"""Explicit small real-device server qualification. Never used by host/fake tests."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse, hashlib, http.client, json, os, shlex, signal, socket, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/server-validation/m4');p.add_argument('--cases',default='all')
p.add_argument('--binary',type=Path,default=ROOT/'bin/h3cli');p.add_argument('--backend',choices=('metal','cuda'),default='metal')
p.add_argument('--model',type=Path,default=Path('models/MiniMax-H3'))
p.add_argument('--lora',type=Path,default=Path('lora/downloads/minimax_h3_turbo_v4_step600_ema.safetensors'))
p.add_argument('--executor-config',type=Path,help='Optional clean-userland argv prefix for server and CLI children')
p.add_argument('--https-url',help='Opt in to a public HTTPS reference-image import test')
p.add_argument('--ffmpeg',default=os.environ.get('H3_SGLANG_INPUT_FFMPEG','ffmpeg'));p.add_argument('--ffprobe',default=os.environ.get('H3_FFPROBE','ffprobe'));a=p.parse_args()
OUT=Path(a.output).resolve();OUT.mkdir(parents=True,exist_ok=True)
BINARY=a.binary.resolve();MODEL=a.model.resolve();LORA=a.lora.resolve();PROMPT='A red wooden toy boat floating on a quiet pond. Gentle ripples and soft birdsong.'
BINARY_HASH=hashlib.sha256(BINARY.read_bytes()).hexdigest()
COMMAND=(json.loads(a.executor_config.read_text()) if a.executor_config else [])+[str(BINARY)]
ENV={**os.environ,'H3_TEST_MAX_EVALUATIONS':'6'}
with socket.socket() as sock:sock.bind(('127.0.0.1',0));PORT=sock.getsockname()[1]
BASE=f'http://127.0.0.1:{PORT}'
state=OUT/'state';record=OUT/'results.json';cases=json.loads(record.read_text())['cases'] if record.exists() else {}
def persist():record.write_text(json.dumps({'binary_sha256':BINARY_HASH,'prompt':PROMPT,'seed':42,'cuda_tested':a.backend=='cuda','cases':cases},indent=2)+'\n')
def req(method,path,body=None):
 headers={}
 if body is not None:body=json.dumps(body).encode();headers['Content-Type']='application/json'
 conn=http.client.HTTPConnection('127.0.0.1',PORT,timeout=120);conn.request(method,path,body,headers);r=conn.getresponse();data=r.read();conn.close()
 try:d=json.loads(data)
 except ValueError:d=data.decode(errors='replace')
 if not 200<=r.status<300:raise RuntimeError((r.status,d))
 return d

def upload(path):
 r=subprocess.run(['curl','--fail-with-body','-sS','--max-time','180','-F',f'file=@{path}',BASE+'/v1/h3/assets'],text=True,capture_output=True)
 if r.returncode:raise RuntimeError(r.stderr+r.stdout)
 return json.loads(r.stdout)['uri']
def settings(frames=22,steps=2):return f'--width 256 --height 256 --frames {frames} --steps {steps} --seed 42'
def submit(name,flags='',body=None,native=False):
 data=body if body is not None else {'prompt':PROMPT,'h3cli':flags}
 endpoint='/v1/h3/jobs' if native else '/v1/videos';start=time.monotonic();d=req('POST',endpoint,data)
 prior=cases.get(name);cases[name]={'request':data,'endpoint':endpoint,'id':d['id'],'binary_sha256':BINARY_HASH,'admission_seconds':time.monotonic()-start,'started_monotonic':start};
 if prior:cases[name]['prior_attempts']=prior.get('prior_attempts',[])+[{k:v for k,v in prior.items() if k!='prior_attempts'}]
 persist();print(name,'accepted',d['id'],flush=True);return d['id']
def poll(name,expected='completed'):
 id=cases[name]['id'];deadline=time.monotonic()+900;previous=None
 while time.monotonic()<deadline:
  d=req('GET','/v1/h3/jobs/'+id)
  if d['status']!=previous:print(name,d['status'],d['progress'],flush=True);previous=d['status']
  if d['status'] not in ('queued','running'):
   cases[name]['result']=d;cases[name]['client_total_seconds']=time.monotonic()-cases[name]['started_monotonic'];persist()
   if d['status']!=expected:
    log=state/'jobs'/id/'v0/result/worker.log';log=log if log.exists() else state/'jobs'/id/'v0/work/worker.log';print(log.read_text(errors='replace')[-12000:] if log.exists() else '',flush=True);raise AssertionError((name,d['status'],d['error']))
   if expected=='completed':download(name,d)
   return d
  time.sleep(1)
 raise TimeoutError(name)
def download(name,d):
 folder=OUT/name;folder.mkdir(exist_ok=True);artifacts=[]
 for art in d['h3']['artifacts']:
  path=folder/art['name'];path.parent.mkdir(parents=True,exist_ok=True)
  subprocess.run(['curl','-sS','--fail','--max-time','120',BASE+art['url'],'-o',str(path)],check=True)
  assert hashlib.sha256(path.read_bytes()).hexdigest()==art['sha256'],path
  if path.suffix in ('.mp4','.png'):
   probe=json.loads(subprocess.check_output([a.ffprobe,'-v','error','-show_streams','-show_format','-of','json',str(path)]))
   subprocess.run([a.ffmpeg,'-v','error','-i',str(path),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
   art={**art,'probe':probe}
  artifacts.append({**art,'local_path':str(path.relative_to(ROOT))})
 cases[name]['artifacts']=artifacts;persist()
def artifact(name,suffix):
 matches=[v for v in cases[name]['artifacts'] if v['name'].endswith(suffix)]
 assert len(matches)==1,(name,suffix,[v['name'] for v in matches]);return matches[0]
def uri(name,suffix):return artifact(name,suffix)['uri']
def run(name,flags='',body=None,native=False):submit(name,flags,body,native);return poll(name)
def selected(case):return a.cases=='all' or case in a.cases.split(',')
args=[*COMMAND,'-d',str(MODEL),'--server','--server-port',str(PORT),'--server-state-dir',str(state),'--server-read-root',str(LORA.parent),'--server-job-timeout','900']
if a.https_url:args+=['--server-allow-url-inputs']
log=open(OUT/'server.log','w')
def start_server(env=ENV):return subprocess.Popen(args,stdout=log,stderr=log,env=env,start_new_session=True)
def stop_server(kill=False):
 if server.poll() is None:
  os.killpg(server.pid,signal.SIGKILL if kill else signal.SIGTERM)
  try:server.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=10);raise
server=start_server()
try:
 for _ in range(100):
  if server.poll() is not None:raise RuntimeError((OUT/'server.log').read_text())
  try:req('GET','/health');break
  except OSError:time.sleep(.1)
 (OUT/'capabilities.json').write_text(json.dumps(req('GET','/v1/h3/capabilities'),indent=2)+'\n')
 if selected('M01'):
  body={'prompt':'ignored prompt','model':'missing-overridden-model','task':'t2va','quality':'high','seed':[1],'num_inference_steps':51,'target':{'short_edge':768,'aspect_ratio':'16:9','duration_seconds':10},'h3cli':f'-d {shlex.quote(str(MODEL))} -p {shlex.quote(PROMPT)} --quality extra-high {settings()} --save-av-state source.h3av --save-conditioning source.h3cond --save-upscale-state source.h3up'}
  submit('M01a',body=body);submit('M01b',settings());poll('M01a');poll('M01b')
  command=[*COMMAND,'-d',str(MODEL),*(['--backend','metal'] if a.backend=='metal' else []),'-p',PROMPT,'--quality','extra-high',*shlex.split(settings()),'-o',str(OUT/'M01-cli.mp4'),'--save-av-state',str(OUT/'M01-cli.h3av')]
  with (OUT/'M01-cli.log').open('w') as f:subprocess.run(command,stdout=f,stderr=f,env=ENV,check=True)
  for suffix,local in [('.mp4','M01-cli.mp4'),('.h3av','M01-cli.h3av')]:
   remote=artifact('M01a',suffix);assert hashlib.sha256((OUT/local).read_bytes()).hexdigest()==remote['sha256'],('CLI/server mismatch',suffix)
  cases['M01a']['cli_equivalence']={'argv':command,'mp4_byte_identical':True,'av_byte_identical':True};persist()
 if selected('M02'):
  run('M02',body={'prompt':PROMPT,'task':'t2va','conditions':[],'target':{'short_edge':256,'aspect_ratio':'1:1','duration_seconds':4},'num_inference_steps':3,'quality':'extra-high','seed':42})
 if selected('M03') or selected('M04'):
  fixtures=OUT/'fixtures';fixtures.mkdir(exist_ok=True)
  for i in (1,2):subprocess.run([a.ffmpeg,'-v','error','-y','-i',f'inputs/{i}.jpg','-vf','scale=176:256','-frames:v','1',str(fixtures/f'{i}.png')],check=True)
  first=upload(fixtures/'1.png');last=upload(fixtures/'2.png')
 if selected('M03'):
  run('M03',body={'prompt':PROMPT,'task':'fl2va','conditions':[{'type':'image','role':'keyframe','frame_index':0,'uri':'/must-not-open'},{'type':'image','role':'keyframe','frame_index':-1,'uri':last}],'h3cli':settings()+f' --first-frame {first}'})
 if selected('M04'):
  subprocess.run([a.ffmpeg,'-v','error','-y','-f','lavfi','-i','testsrc2=size=96x96:rate=24:duration=2','-f','lavfi','-i','sine=frequency=440:sample_rate=32000:duration=2','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-shortest',str(fixtures/'reference.mp4')],check=True)
  subprocess.run([a.ffmpeg,'-v','error','-y','-f','lavfi','-i','sine=frequency=330:sample_rate=32000:duration=2','-ac','2',str(fixtures/'reference.wav')],check=True)
  video=upload(fixtures/'reference.mp4');audio=upload(fixtures/'reference.wav')
  run('M04',body={'prompt':PROMPT,'task':'ref2va','conditions':[{'type':'image','role':'reference','uri':'http://127.0.0.1/must-not-open'}],'h3cli':settings(frames=56)+f' --ref-image {first} --ref-silent-video {video} --ref-audio {audio} --ref-image-size match'})
 if selected('M05'):
  bundle=upload(ROOT/artifact('M01a','.h3cond.h3bundle')['local_path'])
  run('M05',settings()+f' --load-conditioning {bundle} --preview-vae --show --frames-dir frames --zoom 2')
 if selected('M06'):
  run('M06-pause',settings(steps=4)+' --stop-after-step 2 --save-sampler-state paused.h3sample --preview-on-stop --preview-vae',native=True)
  bundle=upload(ROOT/artifact('M06-pause','.h3sample.h3bundle')['local_path'])
  run('M06-resume',body={'h3cli':f'--resume-sampler-state {bundle} --preview-vae --save-av-state resumed.h3av'},native=True)
  run('M06-state-only',settings()+' --state-only --save-upscale-state only.h3up --save-av-state only.h3av',native=True)
  bundle=upload(ROOT/artifact('M01a','.h3av.h3bundle')['local_path'])
  run('M06-decode',body={'h3cli':f'--decode-av-state {bundle}'},native=True)
 if selected('M07'):
  run('M07-source',settings(frames=39)+' --save-av-state context.h3av')
  context=uri('M07-source','.h3av')
  run('M07-hard',settings(frames=56)+f' --continue-from {context} --continue-context 39 --save-av-state hard.h3av')
  run('M07-bridge',settings(frames=56)+f' --continue-from {context} --continue-context 39 --continue-mode bridge --continue-bridge-steps 1 --save-av-state bridge.h3av')
 if selected('M08'):
  source=uri('M01a','.h3up')
  run('M08-inspect',body={'h3cli':f'--inspect-upscale-state {source}'},native=True)
  run('M08-zero',body={'h3cli':f'--upscale-state {source} --upscale-refine-steps 0 --save-av-state zero.h3av --state-only'},native=True)
  run('M08-refine',body={'h3cli':f'--upscale-state {source} --upscale-refine-steps 2 --save-av-state upscale.h3av'},native=True)
 if selected('M09'):
  run('M09-still','--still --width 256 --height 256 --steps 2 --seed 42 --save-still-latent still.safetensors',native=True)
  latent=upload(ROOT/artifact('M09-still','.safetensors.h3bundle')['local_path'])
  run('M09-decode',body={'h3cli':f'--decode-still-latent {latent}'},native=True)
 if selected('M10'):
  run('M10-lora',settings()+f' --lora {shlex.quote(str(LORA)+":0.5")} --save-av-state lora.h3av')
  run('M10-high',body={'prompt':PROMPT,'quality':'high','h3cli':settings(steps=6)})
  run('M10-fast',settings(steps=6)+' --quality fast-preview')
 if selected('M11'):
  flags=' --backend metal --metal-attention dense --metal-attention-kernel steel-routed --metal-attention-dtype fp16 --metal-tier reference' if a.backend=='metal' else ' --cuda-attention sage3 --cuda-denoise-quant nvfp4'
  run('M11',settings()+flags)
 if selected('M12'):
  id=submit('M12-cancel',settings());end=time.monotonic()+30
  while time.monotonic()<end:
   if req('GET','/v1/h3/jobs/'+id)['status']=='running':break
   time.sleep(.1)
  time.sleep(2);req('POST','/v1/h3/jobs/'+id+'/cancel',{});poll('M12-cancel','cancelled');run('M12-next',settings())
 if selected('M13'):
  active=submit('M13-active',settings(frames=56,steps=6));queued=submit('M13-queued',settings())
  deadline=time.monotonic()+30
  while req('GET','/v1/h3/jobs/'+active)['status']!='running':
   if time.monotonic()>deadline:raise TimeoutError('M13 active job did not start')
   time.sleep(.05)
  stop_server(kill=True)
  server=start_server()
  for _ in range(100):
   if server.poll() is not None:raise RuntimeError('restart failed')
   try:req('GET','/health');break
   except OSError:time.sleep(.1)
  poll('M13-active','interrupted');poll('M13-queued')
 if selected('M14') and a.https_url:
  run('M14-https',settings()+f' --first-frame {shlex.quote(a.https_url)}')
  stop_server()
  server=start_server(ENV|{'CURL_CA_BUNDLE':'/missing-h3-test-ca-store'})
  for _ in range(100):
   if server.poll() is not None:raise RuntimeError('CA-failure server did not start')
   try:req('GET','/health');break
   except OSError:time.sleep(.1)
  try:submit('M14-invalid-ca',settings()+f' --first-frame {shlex.quote(a.https_url)}')
  except RuntimeError as error:
   assert 'certificate verification failed' in str(error),error
   cases['M14-invalid-ca']={'passed':True,'rejected':str(error)};persist()
  else:raise AssertionError('HTTPS import accepted an unavailable CA store')
finally:
 stop_server();log.close();persist()
 print('Saved evidence:',record,flush=True)
