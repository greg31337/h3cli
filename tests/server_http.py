#!/usr/bin/env python3
"""Local HTTP lifecycle tests. Outputs from the test-only worker are FAKE, not videos."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import contextlib
import struct, base64, concurrent.futures, hashlib, http.client, json, os, signal, socket, sqlite3, subprocess, time, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
BINARY=os.environ.get('H3_SERVER_HTTP_BINARY','bin/h3cli-server-test')
OUT=ROOT/'outputs/server-validation/http';OUT.mkdir(parents=True,exist_ok=True)
PNG=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')
class Server:
 def __init__(self,name,**kw):
  self.state=OUT/(name+'-'+str(time.time_ns()));self.state.mkdir();self.models_root=kw.pop("models_path",None);self.offline=kw.pop("offline",False);self.model=kw.pop("model",None if self.models_root else "models/MiniMax-H3");self.environment=kw.pop("env",{});self.extra=kw;self.p=None
  with socket.socket() as sock:sock.bind(('127.0.0.1',0));self.port=sock.getsockname()[1]
  self.start()
 def start(self):
  args=[BINARY,'--server','--server-port',str(self.port),'--server-state-dir',str(self.state)]
  if self.model is not None:args+=['-d',str(self.model)]
  if self.models_root is not None:args+=['--models-path',str(self.models_root)]
  if self.offline:args+=['--offline']
  for k,v in self.extra.items():
   args+=['--server-'+k.replace('_','-')];
   if v is not True:args+=[str(v)]
  env=os.environ.copy();env['H3_SERVER_TEST_WORKER']='1';env['H3_TEST_MAX_EVALUATIONS']='6';env.update(self.environment)
  self.log=open(self.state/'server.log','ab');self.p=subprocess.Popen(args,stdout=self.log,stderr=self.log,env=env)
  for _ in range(100):
   if self.p.poll() is not None:raise AssertionError((self.p.returncode,(self.state/'server.log').read_text()))
   try:
    if self.req('GET','/health')[0]==200:return
   except OSError:pass
   time.sleep(.05)
  raise AssertionError('listener did not start')
 def close(self,kill=False):
  if self.p and self.p.poll() is None:
   self.p.kill() if kill else self.p.terminate();self.p.wait(timeout=20)
  self.log.close()
 def req(self,method,path,body=None,headers=None,raw=False):
  headers=dict(headers or {})
  if isinstance(body,dict):body=json.dumps(body).encode();headers.setdefault('Content-Type','application/json')
  if body is not None:headers['Content-Length']=str(len(body))
  c=http.client.HTTPConnection('127.0.0.1',self.port,timeout=30)
  c.request(method,path,body,headers);r=c.getresponse();data=r.read();result=(r.status,data if raw or not data else json.loads(data),dict(r.getheaders()));c.close();return result
 def submit(self,prompt='fake normal',flags='',**fields):
  body={'prompt':prompt,'h3cli':'--width 256 --height 256 --frames 22 --steps 2 '+flags,**fields}
  code,d,_=self.req('POST','/v1/videos',body);assert code==200,(code,d);return d
 def wait(self,id,terminal=True,timeout=15):
  end=time.monotonic()+timeout
  while time.monotonic()<end:
   code,d,_=self.req('GET','/v1/h3/jobs/'+id);assert code==200,(code,d)
   if d['status'] not in ('queued','running') if terminal else d['status']=='running':return d
   time.sleep(.03)
  raise AssertionError(('job timed out',d))
 def multipart(self,path,fields=None,file=None,headers=None):
  boundary='H3ServerTestBoundary';parts=[]
  for key,value in (fields or {}).items():
   if not isinstance(value,str):value=json.dumps(value)
   parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n'.encode()+value.encode()+b'\r\n')
  if file:
   key,filename,data=file;parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()+data+b'\r\n')
  body=b''.join(parts)+f'--{boundary}--\r\n'.encode();h={'Content-Type':'multipart/form-data; boundary='+boundary,**(headers or {})};return self.req('POST',path,body,h)
class HTTP(unittest.TestCase):
 def setUp(self):self.s=Server(self._testMethodName)
 def tearDown(self):self.s.close()
 def test_discovery_and_guards(self):
  for path in ('/health','/liveness','/readiness','/v1/models','/v1/h3/capabilities','/v1/h3/schema'):
   code,d,_=self.s.req('GET',path);self.assertEqual(code,200,(path,d))
  d=self.s.req('GET','/v1/h3/capabilities')[1]
  expected=[line.split('"',2)[1] for line in (ROOT/'src/cli/options.def').read_text().splitlines() if line.startswith('H3_OPTION(')]
  self.assertCountEqual([option['name'] for option in d['options']],expected)
  self.assertEqual(d['quality'],['lossless','extra-high','high'])
  for method,path,status in [('POST','/health',405),('PUT','/v1/videos',405),('GET','/nope',404),('GET','/v1/videos/job_missing/content',404),('GET','/queue.sqlite3',404)]:self.assertEqual(self.s.req(method,path)[0],status)
  for body in ({'h3cli':[]},{'h3cli':'--server'},{'h3':{}},{'h3cli':'--seed -1'},{'h3cli':'--cuda-reference'},{'h3cli':'--width 999999 --height 256 --frames 22 -p x'},{'h3cli':'--cuda-device 0 -p x'},{'h3cli':'--width 256 --height 256 --frames 22 --steps 2'}):self.assertEqual(self.s.req('POST','/v1/videos',body)[0],400,body)
 def test_queue_variants_downloads(self):
  a=self.s.submit('fake slow',n=2,seed=[41,99]);b=self.s.submit()
  a=self.s.wait(a['id']);b=self.s.wait(b['id']);self.assertEqual(a['status'],'completed');self.assertEqual(b['status'],'completed')
  self.assertEqual([v['seed'] for v in a['h3']['variants']],['41','99']);self.assertGreaterEqual(b['h3']['queue_seconds'],7)
  for i in range(2):
   path=f"/v1/videos/{a['id']}/content?variant={i}";code,data,h=self.s.req('GET',path,raw=True);self.assertEqual(code,200);self.assertIn(b'FAKE SERVER',data)
   self.assertEqual(self.s.req('HEAD',path,raw=True)[2]['Content-Length'],str(len(data)))
   code,part,_=self.s.req('GET',path,headers={'Range':'bytes=5-12'},raw=True);self.assertEqual(code,206);self.assertEqual(part,data[5:13])
   self.assertEqual(self.s.req('GET',path,headers={'Range':'bytes=99999-'},raw=True)[0],416)
  self.assertEqual(self.s.req('GET',f"/v1/videos/{a['id']}/content?variant=2",raw=True)[0],404)
  listing=self.s.req('GET','/v1/videos?order=asc&limit=1')[1];self.assertEqual(listing['data'][0]['id'],a['id'])
  self.assertEqual(self.s.req('GET','/v1/videos?order=asc&after='+a['id'])[1]['data'][0]['id'],b['id'])
 def test_idempotency_transports(self):
  body={'prompt':'fake normal','h3cli':'--quality fast-preview --steps 2 --no-preview-vae --width 256 --height 256 --frames 22 --seed 42','quality':'high','num_inference_steps':51};h={'Idempotency-Key':'transport'}
  first=self.s.req('POST','/v1/videos',body,h);self.assertEqual(first[0],200,first)
  for data in (dict(reversed(list(body.items()))),{'extra_body':body},{'extra_json':json.dumps(body)},{'extra_params':body}):
   r=self.s.req('POST','/v1/videos',data,h);self.assertEqual(r[0],200,r);self.assertEqual(r[1]['id'],first[1]['id'])
  r=self.s.multipart('/v1/videos',body,headers=h);self.assertEqual(r[0],200,r);self.assertEqual(r[1]['id'],first[1]['id'])
  changed={**body,'h3cli':body['h3cli']+' --seed 43'};self.assertEqual(self.s.req('POST','/v1/videos',changed,h)[0],409)
  def send(_):return self.s.req('POST','/v1/videos',body,{'Idempotency-Key':'race'})
  with concurrent.futures.ThreadPoolExecutor(6) as pool:results=list(pool.map(send,range(6)))
  self.assertTrue(all(v[0]==200 for v in results),results);self.assertEqual(len({v[1]['id'] for v in results}),1)
 def test_override_inputs_and_upload(self):
  code,asset,_=self.s.multipart('/v1/h3/assets',file=('file','../../escape.png',PNG));self.assertEqual(code,201,asset)
  uri=asset['uri'];fields={'model':'missing','task':'fl2va','conditions':[{'type':'image','role':'keyframe','frame_index':0,'uri':'/forbidden/missing'}],'target':{'short_edge':4096,'aspect_ratio':'auto','duration_seconds':15},'prompt':'fake normal','h3cli':f'-d models/MiniMax-H3 --width 256 --height 256 --frames 22 --steps 2 --first-frame {uri}'}
  code,d,_=self.s.req('POST','/v1/videos',fields);self.assertEqual(code,200,d);self.assertEqual(self.s.wait(d['id'])['status'],'completed')
  r=self.s.multipart('/v1/videos',{'prompt':'fake normal','h3cli':'--width 256 --height 256 --frames 22 --steps 2'},('input_reference','image.png',PNG));self.assertEqual(r[0],200,r)
  self.assertEqual(self.s.req('DELETE','/v1/h3/assets/'+asset['id'])[0],200)
  self.assertFalse((ROOT/'escape.png').exists())
 def test_paths_urls_and_shell(self):
  for flag in ('--first-frame /etc/passwd','--first-frame http://127.0.0.1/a','--output /tmp/escape.mp4','--output ../escape.mp4','--lora-cache ../../bad','--output completion.json','--save-av-state video.mp4','--save-av-state state --output state.presentation','--frames-dir frames --output frames/movie.mp4','--show --output preview.ppm'):
   code,d,_=self.s.req('POST','/v1/videos',{'prompt':'fake normal','h3cli':'--width 256 --height 256 --frames 22 --steps 2 '+flag});self.assertEqual(code,400,(flag,d))
  marker=OUT/'shell-marker';d=self.s.submit(f'fake $(touch {marker}) `touch {marker}` $HOME *.png');self.assertEqual(self.s.wait(d['id'])['status'],'completed');self.assertFalse(marker.exists())
 def test_cancel_and_next(self):
  active=self.s.submit('fake slow');self.s.wait(active['id'],False);queued=self.s.submit('fake normal')
  self.assertEqual(self.s.req('POST',f"/v1/h3/jobs/{queued['id']}/cancel",{})[0],200)
  self.assertEqual(self.s.wait(queued['id'])['status'],'cancelled')
  self.assertEqual(self.s.req('POST',f"/v1/h3/jobs/{active['id']}/cancel",{})[0],200);self.assertEqual(self.s.wait(active['id'])['status'],'cancelled')
  next=self.s.submit();self.assertEqual(self.s.wait(next['id'])['status'],'completed')
 def test_events_and_failure(self):
  d=self.s.submit('fake normal');code,data,_=self.s.req('GET',f"/v1/h3/jobs/{d['id']}/events",raw=True);self.assertEqual(code,200);self.assertIn(b'data:',data)
  events=[json.loads(line[6:]) for line in data.splitlines() if line.startswith(b'data: ')];self.assertTrue(any(v['type']=='completed' for v in events))
  event_id=events[-1]['event_id'];r=self.s.req('GET',f"/v1/h3/jobs/{d['id']}/events",headers={'Last-Event-ID':str(event_id)},raw=True);self.assertNotIn(b'data:',r[1])
  d=self.s.submit('fake crash');self.assertEqual(self.s.wait(d['id'])['status'],'failed');self.assertEqual(self.s.req('GET',f"/v1/videos/{d['id']}/content",raw=True)[0],404)
 def test_restart_and_idempotency(self):
  first=self.s.submit('fake slow');self.s.wait(first['id'],False)
  body={'h3cli':'--width 256 --height 256 --frames 22 --steps 2 -p "fake normal"'};queued=self.s.req('POST','/v1/videos',body,{'Idempotency-Key':'restart'})[1]
  self.s.close(kill=True);self.s.start();self.assertEqual(self.s.wait(first['id'])['status'],'interrupted');self.assertEqual(self.s.wait(queued['id'])['status'],'completed')
  again=self.s.req('POST','/v1/videos',body,{'Idempotency-Key':'restart'});self.assertEqual(again[1]['id'],queued['id'])
 def test_gc(self):
  d=self.s.submit();d=self.s.wait(d['id']);art=d['h3']['artifacts'][0];self.assertEqual(self.s.req('DELETE','/v1/videos/'+d['id'])[0],200)
  self.assertEqual(self.s.req('GET',art['url'],raw=True)[0],404)
  end=time.monotonic()+5
  while (self.s.state/'jobs'/d['id']).exists() and time.monotonic()<end:time.sleep(.1)
  self.assertFalse((self.s.state/'jobs'/d['id']).exists())
 def test_limits_and_auth(self):
  self.s.close();key=OUT/'key';key.write_text('local-test-api-key-0123456789\n');self.s=Server('auth',api_key_file=key,max_upload_mib=1)
  self.assertEqual(self.s.req('GET','/v1/models')[0],401);self.assertEqual(self.s.req('GET','/health')[0],200)
  h={'Authorization':'Bearer local-test-api-key-0123456789'};self.assertEqual(self.s.req('GET','/v1/models',headers=h)[0],200)
  self.assertEqual(self.s.multipart('/v1/h3/assets',file=('file','huge',b'x'*(1024*1024+1)),headers=h)[0],413)
 def test_queue_full_timeout_and_owner(self):
  self.s.close();self.s=Server('limits',queue_limit=2,job_timeout=1)
  d=self.s.submit('fake hang');self.s.wait(d['id'],False);q=self.s.submit('fake normal',n=2)
  code,err,h=self.s.req('POST','/v1/videos',{'h3cli':'--help'});self.assertEqual(code,400) # wrong endpoint before admission
  code,err,h=self.s.req('POST','/v1/videos',{'prompt':'x','h3cli':'--width 256 --height 256 --frames 22 --steps 2'});self.assertEqual(code,429);self.assertIn('Retry-After',h)
  self.assertEqual(self.s.wait(d['id'])['error']['code'],'deadline_exceeded');self.assertEqual(self.s.wait(q['id'])['status'],'completed')
  p=subprocess.run([BINARY,'--server','--server-state-dir',str(self.s.state),'--server-port',str(self.s.port+1)],capture_output=True,text=True);self.assertEqual(p.returncode,2);self.assertIn('owned',p.stderr)
 def test_url_connection_policy(self):
  self.s.close();self.s=Server('url',allow_url_inputs=True)
  for url in ('http://127.0.0.1:1/a','http://[::1]/a','http://169.254.169.254/latest/meta-data','http://10.0.0.1/a','http://localhost/a'):
   code,d,_=self.s.req('POST','/v1/videos',{'prompt':'fake normal','h3cli':f'--width 256 --height 256 --frames 22 --steps 2 --first-frame {url}'});self.assertEqual(code,400,(url,d))
  self.assertFalse(any((self.s.state/'jobs').iterdir()))
 def test_concurrent_fifo(self):
  def send(i):return self.s.submit('fake normal '+str(i))
  with concurrent.futures.ThreadPoolExecutor(6) as pool:jobs=list(pool.map(send,range(8)))
  for j in jobs:self.assertEqual(self.s.wait(j['id'])['status'],'completed')
  with contextlib.closing(sqlite3.connect(self.s.state/'queue.sqlite3')) as db:
   rows=db.execute('SELECT j.created,v.started,v.finished FROM variants v JOIN jobs j ON j.id=v.job ORDER BY j.created,j.id,v.idx').fetchall()
  for previous,current in zip(rows,rows[1:]):self.assertLessEqual(previous[2],current[1])
 def test_bundle_upload_and_corruption(self):
  def bundle(files):
   manifest=json.dumps({'schema':1,'files':[{'suffix':suffix,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for suffix,data in files]}).encode()
   return b'H3BNDL1\n'+struct.pack('<Q',len(manifest))+manifest+b''.join(data for _,data in files)
  # Container-valid state header, deliberately not a numerical render fixture.
  state=b'H3AV\r\n\x1a\n'+struct.pack('<II',3,128)+bytes(112)
  good=bundle([('',state),('.presentation',b'fixture presentation')])
  code,a,_=self.s.multipart('/v1/h3/assets',file=('file','state.h3bundle',good));self.assertEqual(code,201,a);self.assertTrue(a['bundle'])
  managed=self.s.state/'assets'/a['id'];self.assertEqual(managed.read_bytes(),state);self.assertEqual(Path(str(managed)+'.presentation').read_bytes(),b'fixture presentation')
  for data in (good[:-1],good+b'garbage',bundle([('../escape',state)]),bundle([('',state),('',state)])):
   self.assertEqual(self.s.multipart('/v1/h3/assets',file=('file','bad',data))[0],400)
  self.assertEqual(self.s.multipart('/v1/h3/assets',file=('file','junk',b'not media'))[0],400)
  self.assertEqual(self.s.req('DELETE','/v1/h3/assets/'+a['id'])[0],200);self.assertFalse(Path(str(managed)+'.presentation').exists())
 def test_lease_and_snapshot(self):
  code,asset,_=self.s.multipart('/v1/h3/assets',file=('file','ref.png',PNG));self.assertEqual(code,201)
  active=self.s.submit('fake slow');self.s.wait(active['id'],False);queued=self.s.submit(flags='--first-frame '+asset['uri'])
  self.assertEqual(self.s.req('DELETE','/v1/h3/assets/'+asset['id'])[0],409)
  snapshots=list((self.s.state/'jobs'/queued['id']/'inputs').iterdir());self.assertEqual(len(snapshots),1);self.assertEqual(snapshots[0].read_bytes(),PNG)
  self.s.req('POST',f"/v1/h3/jobs/{active['id']}/cancel",{});self.s.wait(active['id']);self.s.wait(queued['id'])
  self.assertEqual(self.s.req('DELETE','/v1/h3/assets/'+asset['id'])[0],200)
 def test_publication_restart(self):
  self.s.close();self.s=Server('publish-delay',env={'H3_SERVER_TEST_PUBLICATION_DELAY':'1'})
  d=self.s.submit();manifest=self.s.state/'jobs'/d['id']/'v0/work/completion.json';end=time.monotonic()+10
  while not manifest.exists() and time.monotonic()<end:time.sleep(.005)
  self.assertTrue(manifest.exists());self.s.close(kill=True);self.s.environment={};self.s.start();d=self.s.wait(d['id']);self.assertEqual(d['status'],'completed',d)
  self.assertEqual(self.s.req('GET',f"/v1/videos/{d['id']}/content",raw=True)[0],200)
 def test_rename_restart(self):
  self.s.close();self.s=Server('publish-rename',env={'H3_SERVER_TEST_CRASH_AFTER_RENAME':'1'})
  d=self.s.submit();self.s.p.wait(timeout=10);self.assertEqual(self.s.p.returncode,86);self.s.close();self.s.environment={};self.s.start()
  self.assertEqual(self.s.wait(d['id'])['status'],'completed')
 def test_orphans_and_bad_schemas(self):
  self.s.close();orphan=self.s.state/'jobs/job_orphan';orphan.mkdir();(orphan/'partial').write_text('unacknowledged');(self.s.state/'incoming/upload_orphan').write_text('unacknowledged')
  self.s.start();self.assertFalse(orphan.exists());self.assertFalse((self.s.state/'incoming/upload_orphan').exists());self.s.close()
  for name,content in [('corrupt',b'not a sqlite database'),('version',None),('unversioned',None)]:
   folder=OUT/(name+str(time.time_ns()));folder.mkdir();db=folder/'queue.sqlite3'
   if content:db.write_bytes(content)
   else:
    with contextlib.closing(sqlite3.connect(db)) as con:con.execute('PRAGMA user_version=999' if name=='version' else 'CREATE TABLE unknown(a)');con.commit()
   p=subprocess.run([BINARY,'--server','--server-state-dir',str(folder),'--server-port',str(self.s.port)],capture_output=True,text=True);self.assertEqual(p.returncode,2,(name,p.stderr))
 def test_quota_and_failed_commit(self):
  self.s.close();self.s=Server('quota',max_storage_mib=1)
  # Preflight failure cleans imported files and leaves no acknowledged job.
  with contextlib.closing(sqlite3.connect(self.s.state/'queue.sqlite3')) as db:db.execute("CREATE TRIGGER fail_admission BEFORE INSERT ON jobs BEGIN SELECT RAISE(FAIL,'injected write failure'); END");db.commit()
  code,d,_=self.s.req('POST','/v1/videos',{'prompt':'fake normal','h3cli':'--width 256 --height 256 --frames 22 --steps 2'});self.assertEqual(code,500,d)
  self.assertFalse(any((self.s.state/'jobs').iterdir()));self.assertEqual(self.s.req('GET','/readiness')[0],503)
  self.assertEqual(self.s.req('GET','/liveness')[0],200)
  with contextlib.closing(sqlite3.connect(self.s.state/'queue.sqlite3')) as db:db.execute('DROP TRIGGER fail_admission');db.commit()
  with (self.s.state/'cache/fill').open('wb') as f:f.truncate(1024*1024)
  code,d,_=self.s.req('POST','/v1/videos',{'h3cli':'--help'});self.assertEqual(code,413,d)
 def test_disconnected_stream_and_logs(self):
  d=self.s.submit('fake logflood');c=http.client.HTTPConnection('127.0.0.1',self.s.port,timeout=10);c.request('GET',f"/v1/h3/jobs/{d['id']}/events");c.getresponse();c.close()
  d=self.s.wait(d['id']);self.assertEqual(d['status'],'completed');log=next(a for a in d['h3']['artifacts'] if a['name']=='worker.log');self.assertLessEqual(log['bytes'],2*1024*1024)
  failed=self.s.wait(self.s.submit('fake crash')['id']);self.assertEqual(failed['status'],'failed');self.assertTrue(failed['h3']['artifacts'])
  code,data,_=self.s.req('GET',failed['h3']['artifacts'][0]['url'],raw=True);self.assertEqual(code,200);self.assertNotIn(str(ROOT).encode(),data)
 def test_model_identity_and_symlink(self):
  self.s.close();root=OUT/('read-root-'+str(time.time_ns()));root.mkdir();model=root/'model';model.mkdir();metadata=model/'FL2VA/transformer/config.json';metadata.parent.mkdir(parents=True);metadata.write_text('{}');(root/'escape').symlink_to('/etc/passwd')
  self.s=Server('model-identity',model=model,read_root=root)
  body={'prompt':'fake normal','h3cli':'--width 256 --height 256 --frames 22 --steps 2 --first-frame '+str(root/'escape')}
  self.assertEqual(self.s.req('POST','/v1/videos',body)[0],400)
  first=self.s.submit('fake slow');self.s.wait(first['id'],False);queued=self.s.submit();metadata.write_text('{"changed":true}')
  self.s.wait(first['id']);failed=self.s.wait(queued['id']);self.assertEqual(failed['status'],'failed')
  next=self.s.submit();self.assertEqual(self.s.wait(next['id'])['status'],'completed')
 def wait_preparing(self,id):
  until=time.monotonic()+5
  while time.monotonic()<until:
   d=self.s.req('GET','/v1/h3/jobs/'+id)[1]
   if d['h3'].get('phase')=='downloading_models':return d
   time.sleep(.02)
  self.fail(('no model preparation progress',d))
 def test_model_preparation_root_cancel_timeout_and_restart(self):
  self.s.close();root=OUT/('empty-models-'+str(time.time_ns()));self.s=Server('preparation',models_path=root,model_download_timeout=1)
  d=self.s.submit('fake prepare-slow');active=self.wait_preparing(d['id']);self.assertEqual(active['status'],'queued');self.assertEqual(active['h3']['run_seconds'],0)
  start=time.monotonic();self.assertEqual(self.s.req('GET','/health')[0],200);self.assertLess(time.monotonic()-start,.5)
  self.assertEqual(self.s.wait(d['id'])['error']['code'],'model_download_timeout')
  with contextlib.closing(sqlite3.connect(self.s.state/'queue.sqlite3')) as db:
   request=json.loads(db.execute('SELECT request FROM variants WHERE job=?',(d['id'],)).fetchone()[0]);options={a['name']:a['value'] for a in request['options']}
  self.assertEqual(options['models-path'],str(root.resolve()));self.assertEqual(options['model-dir'],str(root.resolve()/'MiniMaxH3'))
  d=self.s.submit('fake prepare-slow');self.wait_preparing(d['id']);self.s.req('POST','/v1/h3/jobs/'+d['id']+'/cancel',{});self.assertEqual(self.s.wait(d['id'])['status'],'cancelled')
  d=self.s.submit('fake prepare-slow');self.wait_preparing(d['id']);self.s.close(kill=True);self.s.start();self.assertEqual(self.s.wait(d['id'])['status'],'interrupted')
  self.assertEqual(self.s.wait(self.s.submit()['id'])['status'],'completed')
 def test_model_root_authority_offline_and_unrelated_additions(self):
  self.s.close();root=OUT/('model-policy-'+str(time.time_ns()));main=root/'MiniMaxH3';required=main/'FL2VA/transformer/config.json';required.parent.mkdir(parents=True);required.write_text('{}')
  read=OUT/('readonly-model-'+str(time.time_ns()));read.mkdir();self.s=Server('model-policy',models_path=root,read_root=read,offline=True,allow_url_inputs=True)
  for flag in ('--models-path anywhere','--download-models all','--list-models','-d '+str(read/'missing'),'--offline --first-frame https://example.invalid/image.png'):
   code,d,_=self.s.req('POST','/v1/videos',{'prompt':'fake normal','h3cli':'--width 256 --height 256 --frames 22 --steps 2 '+flag});self.assertEqual(code,400,(flag,d))
  first=self.s.submit('fake slow');self.s.wait(first['id'],False);queued=self.s.submit()
  unrelated=main/'Ref2VA/transformer/config.json';unrelated.parent.mkdir(parents=True);unrelated.write_text('{}')
  self.assertEqual(self.s.wait(first['id'])['status'],'completed');self.assertEqual(self.s.wait(queued['id'])['status'],'completed')
if __name__=='__main__':unittest.main(verbosity=2)
