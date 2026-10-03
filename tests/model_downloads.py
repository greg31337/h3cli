#!/usr/bin/env python3
"""Native downloader fault fixtures. No production URLs are contacted."""
import contextlib, copy, email.utils, hashlib, http.server, importlib.util, json, os, resource, shutil, signal, socket, ssl, subprocess, tempfile, threading, time, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BINARY=ROOT/'bin/model_downloads';CLI=ROOT/'bin/h3cli'
DATA=bytes(range(256))*4096
SHA=hashlib.sha256(DATA).hexdigest()
class Handler(http.server.BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def log_message(self,*args):pass
 def do_GET(self):
  path=self.path.split('?')[0];key=path.rsplit('/',1)[-1]
  with self.server.lock:
   self.server.requests.append((path,dict(self.headers)));count=sum(p==path for p,_ in self.server.requests)
  if key.startswith('status'):
   code=int(key[6:]);self.send_response(code);self.send_header('Content-Length','0');self.end_headers();return
  if key in ('retry','rate','retry-date') and count==1:
   self.send_response(429 if key=='rate' else 503);self.send_header('Retry-After',email.utils.formatdate(time.time()+2,usegmt=True) if key=='retry-date' else '1');self.send_header('Content-Length','0');self.end_headers();return
  if key=='redirect' or key=='token':
   self.send_response(307);self.send_header('Location',self.server.redirect);self.send_header('Content-Length','100');self.end_headers();self.wfile.write(b'r'*100);return
  if key=='loop':
   self.send_response(302);self.send_header('Location','/loop');self.send_header('Content-Length','0');self.end_headers();return
  if key=='unsafe':
   self.send_response(302);self.send_header('Location','file:///etc/passwd');self.send_header('Content-Length','0');self.end_headers();return
  length=len(DATA)*(64 if key=='large' else 1);start=0
  partial=self.headers.get('Range')
  if partial and key!='ignore':start=int(partial.split('=')[1].split('-')[0])
  if start>=length:
   self.send_response(416);self.send_header('Content-Length','0');self.send_header('Content-Range',f'bytes */{length}');self.end_headers();return
  self.send_response(206 if start else 200)
  if start:self.send_header('Content-Range',f'bytes {start+(1 if key=="bad-range" else 0)}-{length-1}/{length}')
  self.send_header('Content-Length',str(length-start+(1 if key=='oversize' else 0)));self.end_headers()
  if key=='timeout':time.sleep(1)
  until=length
  if key=='truncated' and count==1:until=min(start+len(DATA)//3,length)
  try:
   at=start
   while at<until:
    n=min(16384,until-at);self.wfile.write(DATA[at%len(DATA):at%len(DATA)+n]);self.wfile.flush();at+=n
    if key in ('slow','cancel'):time.sleep(.03)
   if key=='oversize':self.wfile.write(b'x')
  except (OSError,ssl.SSLError):pass
  if until<length:self.close_connection=True
class Service(http.server.ThreadingHTTPServer):
 daemon_threads=True
 def __init__(self,tls=None):
  super().__init__(('127.0.0.1',0),Handler);self.lock=threading.Lock();self.requests=[];self.redirect='/good'
  if tls:self.socket=tls.wrap_socket(self.socket,server_side=True)
  self.url=('https' if tls else 'http')+'://127.0.0.1:'+str(self.server_port)
  self.thread=threading.Thread(target=self.serve_forever,daemon=True);self.thread.start()
 def handle_error(self,*args):pass  # Expected disconnects in cancellation/truncation fixtures.
 def close(self):self.shutdown();self.server_close();self.thread.join()
class Native(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.s=Service();cls.tmp=tempfile.TemporaryDirectory(prefix='h3-model-tests-');cls.base=Path(cls.tmp.name)
 @classmethod
 def tearDownClass(cls):cls.s.close();cls.tmp.cleanup()
 def setUp(self):self.root=self.base/(self._testMethodName+'-'+str(time.time_ns()));self.root.mkdir();self.install=self.root/'models'
 def entry(self,path='weight.bin',endpoint='good',**kw):return dict(path=path,url=self.s.url+'/'+self._testMethodName+'/'+endpoint,sha256=SHA,bytes=len(DATA),**kw)
 def run_fetch(self,files=None,good=True,extra=None,env=None,root=None,preexec=None):
  f=self.root/('request-'+str(time.time_ns())+'.json');f.write_text(json.dumps(dict(root=str(root or self.install),files=files or [self.entry()],**(extra or {}))))
  e={**os.environ,'H3_OFFLINE':'0','NO_PROXY':'*','no_proxy':'*',**(env or {})}
  p=subprocess.run([str(BINARY),'fetch',str(f)],capture_output=True,text=True,env=e,timeout=45,preexec_fn=preexec)
  self.assertEqual(p.returncode,0 if good else 1,(p.stdout,p.stderr))
  return p
 def records(self,key=None):return [(p,h) for p,h in self.s.requests if self._testMethodName in p and (not key or p.endswith('/'+key))]
 def partial(self,data,entry=None):
  a=entry or self.entry();root=str(self.install.resolve());dest=root+'/'+a['path']
  meta=self.install.parent/'.h3cli-downloads'/hashlib.sha256(root.encode()).hexdigest();meta.mkdir(parents=True,exist_ok=True)
  part=meta/(hashlib.sha256(dest.encode()).hexdigest()+'-'+a['sha256']+'.part');part.write_bytes(data);return part
 def test_success_warm_offline_receipt_and_custom(self):
  self.run_fetch();self.assertEqual((self.install/'weight.bin').read_bytes(),DATA);before=len(self.records());ino=(self.install/'weight.bin').stat().st_ino
  self.run_fetch(env={'H3_OFFLINE':'1'});self.assertEqual(len(self.records()),before);self.assertEqual(ino,(self.install/'weight.bin').stat().st_ino)
  receipts=list(self.root.rglob('*.receipt'));self.assertEqual(len(receipts),1);receipts[0].unlink();self.run_fetch(env={'H3_OFFLINE':'1'});self.assertEqual(len(self.records()),before)
  (self.install/'weight.bin').write_bytes(b'corrupt');self.run_fetch(good=False);self.assertEqual((self.install/'weight.bin').read_bytes(),b'corrupt')
  custom=self.root/'custom';custom.mkdir();(custom/'weight.bin').write_bytes(b'custom')
  self.run_fetch(root=custom,env={'H3_OFFLINE':'1'});self.run_fetch(root=custom,extra={'verify':True},good=False)
 def test_missing_offline_and_conflict(self):
  self.run_fetch(env={'H3_OFFLINE':'1'},good=False);self.assertEqual(self.records(),[]);self.assertFalse(self.install.exists())
  self.install.mkdir();(self.install/'weight.bin').write_bytes(b'custom')
  self.run_fetch([self.entry(),self.entry('second.bin')],good=False);self.assertEqual(self.records(),[])
 def test_custom_neighbor_is_not_silently_adopted(self):
  existing=self.install/'FL2VA/audio_vae/config.json';existing.parent.mkdir(parents=True);existing.write_bytes(b'custom mode')
  self.run_fetch([self.entry('Ref2VA/tokenizer/tokenizer.json')],good=False)
  self.assertEqual(self.records(),[]);self.assertEqual(existing.read_bytes(),b'custom mode')
  self.assertFalse(list(self.root.rglob('catalog')))
 def test_dedup_and_no_writable_hardlinks(self):
  self.run_fetch([self.entry(),self.entry('second.bin')]);self.assertEqual(len(self.records()),1)
  a=self.install/'weight.bin';b=self.install/'second.bin';self.assertNotEqual(a.stat().st_ino,b.stat().st_ino);a.write_bytes(b'changed');self.assertEqual(b.read_bytes(),DATA)
 def test_truncation_and_resume(self):
  self.run_fetch([self.entry(endpoint='truncated')]);self.assertEqual(len(self.records()),2);self.assertEqual(self.records()[1][1].get('Range'),'bytes=349525-')
 def test_persistent_range_and_ignored_range(self):
  a=self.entry();self.partial(DATA[:12345],a);self.run_fetch([a]);self.assertEqual(self.records()[0][1].get('Range'),'bytes=12345-')
  b=self.entry('other.bin','ignore');self.partial(DATA[:12345],b);self.run_fetch([b]);self.assertEqual(len(self.records('ignore')),2)
  c=self.entry('redirect.bin','redirect');self.partial(DATA[:12345],c);self.run_fetch([c]);self.assertEqual((self.install/'redirect.bin').read_bytes(),DATA)
 def test_bad_range_and_complete_partial(self):
  a=self.entry(endpoint='bad-range');self.partial(DATA[:99],a);self.run_fetch([a],good=False);self.assertFalse((self.install/'weight.bin').exists())
  b=self.entry('complete.bin');self.partial(DATA,b);self.run_fetch([b]);self.assertEqual(len(self.records('good')),0)
 def test_retry_and_terminal_http(self):
  for endpoint in ('retry','rate','retry-date'):
   self.run_fetch([self.entry(endpoint+'.bin',endpoint)]);self.assertEqual(len(self.records(endpoint)),2)
  for code,n in ((400,1),(401,1),(403,2),(404,1),(500,5)):
   endpoint='status'+str(code);self.run_fetch([self.entry(endpoint+'.bin',endpoint)],good=False);self.assertEqual(len(self.records(endpoint)),n)
 def test_timeout_and_incomplete_416(self):
  start=time.monotonic();self.run_fetch([self.entry(endpoint='timeout')],good=False,env={'H3_MODEL_TEST_TIMEOUT':'1'})
  self.assertEqual(len(self.records('timeout')),5);self.assertLess(time.monotonic()-start,22)
  a=self.entry('range.bin','status416');self.partial(DATA[:100],a);self.run_fetch([a],good=False)
  self.assertEqual(len(self.records('status416')),5);self.assertFalse((self.install/'range.bin').exists())
 def test_redirect_hash_and_size_failure(self):
  self.run_fetch([self.entry(endpoint='redirect')]);self.assertEqual((self.install/'weight.bin').read_bytes(),DATA)
  for endpoint in ('loop','unsafe','oversize'):
   self.run_fetch([self.entry(endpoint+'.bin',endpoint)],good=False);self.assertFalse((self.install/(endpoint+'.bin')).exists())
  a=self.entry('bad-hash.bin');a['sha256']='0'*64;self.run_fetch([a],good=False);self.assertFalse((self.install/'bad-hash.bin').exists())
 def test_cancel_then_resume(self):
  a=self.entry(endpoint='cancel');start=time.monotonic();self.run_fetch([a],good=False,extra={'cancel_after':.25});self.assertLess(time.monotonic()-start,2.0)
  self.assertFalse((self.install/'weight.bin').exists());self.assertTrue(any(p.stat().st_size for p in self.root.rglob('*.part')))
  self.run_fetch([a]);self.assertTrue(any(h.get('Range') for _,h in self.records()))
 def test_simultaneous_processes(self):
  a=self.entry(endpoint='slow');f=self.root/'parallel.json';f.write_text(json.dumps(dict(root=str(self.install),files=[a])))
  env={**os.environ,'H3_OFFLINE':'0','NO_PROXY':'*'}
  jobs=[subprocess.Popen([str(BINARY),'fetch',str(f)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env) for _ in range(3)]
  for p in jobs:out,err=p.communicate(timeout=30);self.assertEqual(p.returncode,0,(out,err))
  self.assertEqual(len(self.records()),1)
 def test_lock_crash_recovery(self):
  f=self.root/'crash.json';f.write_text(json.dumps(dict(root=str(self.install),files=[self.entry(endpoint='slow')])))
  p=subprocess.Popen([str(BINARY),'fetch',str(f)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,env={**os.environ,'H3_OFFLINE':'0','NO_PROXY':'*'})
  deadline=time.monotonic()+5
  while not list(self.root.rglob('*.part')) and time.monotonic()<deadline:time.sleep(.01)
  time.sleep(.2);p.kill();p.communicate();self.run_fetch([self.entry(endpoint='slow')]);self.assertEqual((self.install/'weight.bin').read_bytes(),DATA)
 def test_readonly_and_storage_error(self):
  self.run_fetch();before=len(self.records());(self.install/'weight.bin').chmod(0o444);self.install.chmod(0o555)
  try:self.run_fetch(env={'H3_OFFLINE':'1'})
  finally:self.install.chmod(0o755)
  self.assertEqual(len(self.records()),before)
  def cap():resource.setrlimit(resource.RLIMIT_FSIZE,(131072,131072))
  self.run_fetch([self.entry('limited.bin')],good=False,preexec=cap);self.assertFalse((self.install/'limited.bin').exists())
 def test_symlink_destination(self):
  self.install.mkdir();outside=self.root/'outside';outside.write_bytes(b'preserve');(self.install/'weight.bin').symlink_to(outside)
  self.run_fetch([self.entry(),self.entry('second.bin')],good=False);self.assertEqual(outside.read_bytes(),b'preserve')
 def test_large_bounded_transfer(self):
  a=self.entry(endpoint='large');a['bytes']=64*len(DATA);h=hashlib.sha256()
  for _ in range(64):h.update(DATA)
  a['sha256']=h.hexdigest();self.run_fetch([a]);self.assertEqual((self.install/'weight.bin').stat().st_size,a['bytes'])
 def test_tls_and_ca(self):
  key=self.root/'key.pem';cert=self.root/'ca.pem'
  subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-keyout',str(key),'-out',str(cert),'-days','1','-subj','/CN=127.0.0.1','-addext','subjectAltName=IP:127.0.0.1'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);ctx.load_cert_chain(cert,key);s=Service(ctx)
  try:
   a=self.entry();a['url']=s.url+'/good';self.run_fetch([a],good=False);self.run_fetch([a],env={'CURL_CA_BUNDLE':str(cert)})
  finally:s.close()
 def test_proxy(self):
  a=self.entry();a['url']='http://fixture.invalid/good'
  self.run_fetch([a],env={'http_proxy':self.s.url,'HTTP_PROXY':self.s.url,'NO_PROXY':'','no_proxy':''});self.assertEqual((self.install/'weight.bin').read_bytes(),DATA)
 def test_token_containment_and_publication_crashes(self):
  key=self.root/'key.pem';cert=self.root/'ca.pem'
  subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-keyout',str(key),'-out',str(cert),'-days','1','-subj','/CN=huggingface.co','-addext','subjectAltName=DNS:huggingface.co,IP:127.0.0.1'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);ctx.load_cert_chain(cert,key);origin=Service(ctx);cdn=Service(ctx);origin.redirect=cdn.url+'/good'
  try:
   a=self.entry();a['url']='https://huggingface.co/token'
   result=self.run_fetch([a],env={'HF_TOKEN':'private-fixture-token','CURL_CA_BUNDLE':str(cert),'H3_MODEL_TEST_CONNECT':f'huggingface.co:443:127.0.0.1:{origin.server_port}'})
   self.assertEqual(origin.requests[0][1].get('Authorization'),'Bearer private-fixture-token');self.assertNotIn('Authorization',cdn.requests[0][1]);self.assertNotIn('private-fixture-token',result.stdout+result.stderr)
  finally:origin.close();cdn.close()
  for label,exitcode in (('BEFORE',71),('AFTER',72)):
   a=self.entry(label+'.bin');file=self.root/(label+'.json');file.write_text(json.dumps(dict(root=str(self.install),files=[a])))
   p=subprocess.run([str(BINARY),'fetch',str(file)],env={**os.environ,'H3_OFFLINE':'0','NO_PROXY':'*','H3_MODEL_TEST_CRASH_'+label+'_PUBLISH':'1'},capture_output=True)
   self.assertEqual(p.returncode,exitcode,p.stderr);self.run_fetch([a]);self.assertEqual((self.install/(label+'.bin')).read_bytes(),DATA);self.assertEqual((self.install/(label+'.bin')).stat().st_nlink,1)
 def test_copy_fallback_prefix_corruption_and_symlink_swap(self):
  self.run_fetch([self.entry(),self.entry('copy.bin')],env={'H3_MODEL_TEST_COPY':'1'});self.assertEqual((self.install/'copy.bin').read_bytes(),DATA)
  a=self.entry('cancelled.bin','cancel');self.run_fetch([a],good=False,extra={'cancel_after':.2})
  part=next(p for p in self.root.rglob('*.part') if p.stat().st_size)
  with part.open('r+b') as f:f.write(b'changed')
  before=len(self.records());self.run_fetch([a],good=False);self.assertEqual(len(self.records()),before)
  outside=self.root/'outside';outside.mkdir();file=self.root/'swap.json';file.write_text(json.dumps(dict(root=str(self.install),files=[self.entry('nested/swap.bin','slow')])))
  p=subprocess.Popen([str(BINARY),'fetch',str(file)],env={**os.environ,'H3_OFFLINE':'0','NO_PROXY':'*'},stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
  until=time.monotonic()+5
  while not (self.install/'nested').exists() and time.monotonic()<until:time.sleep(.01)
  (self.install/'nested').rename(self.install/'old');(self.install/'nested').symlink_to(outside,target_is_directory=True)
  _,err=p.communicate(timeout=10);self.assertEqual(p.returncode,1,err);self.assertEqual(list(outside.iterdir()),[])
 def plan(self,*args,good=True):
  p=subprocess.run([str(BINARY),'plan','--models-path',str(self.root/'shared root'),*args],capture_output=True,text=True)
  self.assertEqual(p.returncode,0 if good else 1,(args,p.stderr));return json.loads(p.stdout) if good else None
 def test_resolution_groups_quality_and_overrides(self):
  base=self.plan('-p','x');self.assertTrue(base);self.assertTrue(all(a['path'].startswith('FL2VA/') for a in base))
  for flag in ('ref-image','ref-audio','ref-video','ref-silent-video'):
   p=self.plan('-p','x','--'+flag,'input');self.assertTrue(all(a['path'].startswith('Ref2VA/') for a in p))
  for q in ('lossless','extra-high','high','preview','fast-preview'):
   p=self.plan('-p','x','--quality',q);self.assertEqual(any('taeh3' in a['path'] for a in p),q in ('preview','fast-preview'))
   p=self.plan('-p','x','--quality',q,'--no-preview-vae');self.assertFalse(any('taeh3' in a['path'] for a in p))
  for flag,group in (('preview-vae-model','preview'),('image-vae','image-vae'),('upscale-model','upscale')):
   for order in (['--'+flag,str(self.root/'explicit'),'--download-models',group],['--download-models',group,'--'+flag,str(self.root/'explicit')]):
    p=self.plan(*order);self.assertEqual(len(p),1);self.assertEqual(p[0]['destination'],str((self.root/'explicit').resolve()))
   p=self.plan('--download-models',group);self.assertIn('/shared root/',p[0]['destination'])
  p=self.plan('--download-models','all','-d',str(self.root/'custom'));self.assertEqual(len(p),119)
  self.assertTrue(any('/shared root/preview-vae/' in a['destination'] for a in p))
  self.plan('--download-models','',good=False);self.plan('--download-models','base,',good=False);self.plan('--download-models','base','-p','x',good=False)
  self.assertEqual(self.plan('--info'),[]);self.assertEqual(self.plan('--help'),[])
  self.plan('--resume-sampler-state',str(self.root/'missing'),good=False)
  for opt in ('continue-from','load-conditioning'):
   self.assertTrue(self.plan('-p','x','--'+opt,'state'))
  self.assertTrue(any('image_vae' in a['path'] for a in self.plan('--still','-p','x')))
  self.assertFalse(any('taeh3' in a['path'] for a in self.plan('-p','x','--show')))
 def test_root_selection(self):
  root=self.root/'shared root';root.mkdir();old=root/'MiniMax-H3';old.mkdir()
  self.assertIn('/MiniMax-H3/',self.plan('-p','x')[0]['destination'])
  (root/'MiniMaxH3').mkdir();self.assertIn('/MiniMaxH3/',self.plan('-p','x')[0]['destination'])
  explicit=self.root/'chosen';p=self.plan('-d',str(explicit),'-p','x');self.assertIn('/chosen/',p[0]['destination'])
  link=self.root/'link';link.symlink_to(root,target_is_directory=True);p=self.plan('--models-path',str(link)+'///','--download-models','preview');self.assertIn('/shared root/preview-vae/',p[0]['destination'])
 def test_saved_operations_and_minimal_closures(self):
  for fixture,operation,ref,upscale in (('reference','resume-sampler-state',True,False),('upscale','upscale-state',False,True),('refinement','resume-sampler-state',False,False)):
   state=self.root/(fixture+'.state')
   subprocess.run([str(ROOT/'bin/sampler_tests'),'--'+fixture+'-fixture',str(state)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   p=self.plan('--'+operation,str(state));main=[i for i in p if i['path'].startswith(('FL2VA/','Ref2VA/'))]
   self.assertEqual(len(main),57);self.assertTrue(all(i['path'].startswith('Ref2VA/' if ref else 'FL2VA/') for i in main))
   self.assertEqual(any('upscaler' in i['path'] for i in p),upscale)
  state=self.root/'decode.h3av';state.write_bytes(b'planner-only fixture')
  for ref in (0,1):
   state.with_name(state.name+'.presentation').write_text('H3-PRESENTATION 9\nstate '+'0'*64+'\nvariant '+str(ref)+'\n')
   p=self.plan('--decode-av-state',str(state));self.assertEqual(len(p),8);self.assertTrue(all(i['path'].startswith('Ref2VA/' if ref else 'FL2VA/') for i in p))
   self.assertEqual(len(self.plan('--decode-av-state',str(state),'--preview-vae')),9)
  self.assertEqual(len(self.plan('--decode-still-latent','latent')),1)
  self.assertEqual(len(self.plan('-p','x','--state-only','--save-av-state','out')),57)
  self.assertEqual(len(self.plan('-p','x','--first-frame','first','--last-frame','last')),57)
 def test_catalog_validation(self):
  spec=importlib.util.spec_from_file_location('model_catalog',ROOT/'scripts/embed_model_catalog.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  source=ROOT/'src/models/catalog.json';catalog=json.loads(source.read_text())
  self.assertEqual(module.render(source),(ROOT/'src/models/catalog.inc').read_text())
  for mutation in ('path','hash','revision','url','duplicate','identity'):
   bad=copy.deepcopy(catalog);item=bad['artifacts'][0]
   if mutation=='path':item['path']='../outside'
   if mutation=='hash':item['sha256']='bad'
   if mutation=='revision':item['revision']='main'
   if mutation=='url':item['url']=item['url'].replace('https://','http://')
   if mutation=='duplicate':bad['artifacts'].append(dict(item))
   if mutation=='identity':bad['artifacts'].append(dict(item,path='conflict',bytes=item['bytes']+1))
   path=self.root/'catalog.json';path.write_text(json.dumps(bad))
   with self.assertRaises(ValueError):module.render(path)
 def test_cli_no_network(self):
  args=[str(CLI),'--models-path',str(self.root/'absent')]
  env={**os.environ,'H3_OFFLINE':'1','https_proxy':self.s.url,'HTTPS_PROXY':self.s.url}
  for flags,code in ((['--help'],0),(['--list-models'],0),(['--download-models','preview'],1),(['-p','x','--width','257'],2),(['-p','x','--first-frame',str(self.root/'missing')],2)):
   p=subprocess.run(args+flags,capture_output=True,text=True,env=env);self.assertEqual(p.returncode,code,(flags,p.stderr))
  self.assertEqual(self.records(),[])
if __name__=='__main__':unittest.main(verbosity=2)
