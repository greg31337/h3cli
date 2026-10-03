#!/usr/bin/env python3
"""Native Mach-O bootstrap tests using tiny signed fixtures, never render evidence."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/macos'))
import package
import build
CORE=r'''
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
int main(int argc,char **argv){
 if(argc>1&&!strcmp(argv[1],"sleep")){puts("ready");fflush(stdout);for(;;)pause();}
 if(argc>1&&!strcmp(argv[1],"status"))return 19;
 if(argc>1&&!strcmp(argv[1],"stdin")){int c=getchar();putchar(c);return 0;}
 char cwd[4096];getcwd(cwd,sizeof(cwd));printf("cwd=%s\n",cwd);
 printf("root=%s\n",getenv("H3CLI_RUNTIME_ROOT"));
 printf("offline=%s\n",getenv("H3_OFFLINE"));
 printf("dyld=%s\n",getenv("DYLD_LIBRARY_PATH")?"set":"unset");
 for(int i=1;i<argc;i++)printf("arg=%s\n",argv[i]);return 0;
}
'''

@unittest.skipUnless(sys.platform=='darwin','native macOS')
class Bundle(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory(prefix='macos-tests-',dir=ROOT/'bin');cls.base=Path(cls.temp.name)
  (cls.base/'core.c').write_text(CORE);cls.core=cls.base/'core'
  subprocess.run(['xcrun','clang','-arch','arm64','-mmacosx-version-min=26.0',str(cls.base/'core.c'),'-o',str(cls.core)],check=True)
  package.sign(cls.core)
 @classmethod
 def tearDownClass(cls):cls.temp.cleanup()
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=self.base);self.work=Path(self.temp.name)
  self.runtime=self.work/'runtime';(self.runtime/'bin').mkdir(parents=True);(self.runtime/'share/h3cli').mkdir(parents=True)
  shutil.copy2(self.core,self.runtime/'bin/h3cli');(self.runtime/'sample').write_bytes(b'payload'*16000)
  package.write(self.runtime/'share/h3cli/runtime.json',{'schema':1,'platform':'macos-arm64','files':package.inventory(self.runtime)})
  self.archive=self.work/'payload';self.binary=self.work/'bundle';self.cache=self.work/'cache'
  self.env=dict(os.environ,H3CLI_RUNTIME_CACHE=str(self.cache),H3_OFFLINE='1')
  self.env.pop('H3CLI_BUNDLE_INFO',None);self.pack()
 def tearDown(self):self.temp.cleanup()
 def pack(self):
  package.payload(self.runtime,self.archive);self.embed()
 def embed(self):
  # Link a fresh inode; do not mutate an executable still cached by macOS AMFI.
  self.binary.unlink(missing_ok=True)
  package.embed(ROOT,self.archive,self.binary);package.sign(self.binary,identifier='org.h3cli.test')
 def run_bundle(self,*args,**kwargs):
  return subprocess.run([str(self.binary),*args],env=self.env,cwd=self.work,capture_output=True,text=True,timeout=30,**kwargs)
 def info(self):
  r=subprocess.run([str(self.binary)],env=self.env|{'H3CLI_BUNDLE_INFO':'1'},capture_output=True,text=True,check=True)
  return json.loads(r.stdout)
 def cache_root(self):return self.cache/self.info()['runtime_id']
 def edit(self,fn):
  raw=self.archive.read_bytes();magic,m,z,n,mh,ph=struct.unpack('<16sQQQ32s32s',raw[:104]);record=json.loads(raw[104:104+m]);fn(record)
  encoded=package.canonical(record);self.archive.write_bytes(struct.pack('<16sQQQ32s32s',magic,len(encoded),z,n,hashlib.sha256(encoded).digest(),ph)+encoded+raw[104+m:]);self.embed()
 def test_inspection_cold_warm_argv_cwd(self):
  self.assertEqual(self.info()['platform'],'macos-arm64');self.assertFalse(self.cache.exists())
  r=self.run_bundle('a b','$(touch escaped)','é 海','')
  self.assertEqual(r.returncode,0,r.stderr);self.assertIn('cwd='+str(self.work),r.stdout);self.assertIn('offline=1',r.stdout)
  self.assertIn('arg=$(touch escaped)',r.stdout);self.assertFalse((self.work/'escaped').exists())
  stamps={str(p):p.stat().st_mtime_ns for p in self.cache.rglob('*')};self.assertEqual(self.run_bundle().returncode,0)
  self.assertEqual(stamps,{str(p):p.stat().st_mtime_ns for p in self.cache.rglob('*')})
  self.assertEqual(self.run_bundle('status').returncode,19)
 def test_concurrent(self):
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(lambda _:self.run_bundle(),range(8)))
  self.assertTrue(all(r.returncode==0 for r in rows),[r.stderr for r in rows]);self.assertEqual(len(list(self.cache.glob('*/.ready.json'))),1)
 def test_relocation_readonly_symlink_unicode(self):
  moved=self.work/'renamed 海 cli';self.binary.rename(moved);moved.chmod(0o555);self.binary.symlink_to(moved.name)
  self.assertEqual(self.run_bundle().returncode,0);first=self.cache_root()
  self.cache=self.work/'other cache';self.env['H3CLI_RUNTIME_CACHE']=str(self.cache)
  self.assertEqual(self.run_bundle().returncode,0);self.assertTrue(first.is_dir())
 def test_signal_and_stdio(self):
  self.assertEqual(self.run_bundle('stdin',input='q').stdout,'q')
  for sig in (signal.SIGINT,signal.SIGTERM):
   p=subprocess.Popen([str(self.binary),'sleep'],env=self.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
   try:self.assertEqual(p.stdout.readline().strip(),'ready');p.send_signal(sig);self.assertEqual(p.wait(timeout=10),-sig)
   finally:
    if p.poll() is None:p.kill();p.wait()
    p.stdout.close();p.stderr.close()
 def test_changed_missing_ready_and_symlink_members(self):
  self.assertEqual(self.run_bundle().returncode,0);root=self.cache_root();sample=root/'sample'
  sample.write_bytes(b'z'*sample.stat().st_size);self.assertIn('runtime changed',self.run_bundle().stderr)
  shutil.rmtree(root);self.assertEqual(self.run_bundle().returncode,0);sample.unlink();sample.symlink_to('/etc/hosts')
  self.assertNotEqual(self.run_bundle().returncode,0)
  shutil.rmtree(root);self.assertEqual(self.run_bundle().returncode,0);(root/'.ready.json').unlink();self.assertNotEqual(self.run_bundle().returncode,0)
 def test_cache_lock_permissions_and_paths(self):
  self.cache.symlink_to(self.runtime,target_is_directory=True);self.assertNotEqual(self.run_bundle().returncode,0);self.cache.unlink()
  self.cache.mkdir(mode=0o777);self.cache.chmod(0o777);self.assertIn('unsafe cache ancestor',self.run_bundle().stderr);self.cache.chmod(0o700)
  (self.cache/'.extract.lock').symlink_to('/dev/null');self.assertNotEqual(self.run_bundle().returncode,0)
  self.env['H3CLI_RUNTIME_CACHE']='relative';self.assertIn('absolute',self.run_bundle().stderr)
 def test_writable_cache_ancestor_rejected(self):
  parent=self.work/'unsafe-parent';parent.mkdir();parent.chmod(0o777)
  self.env['H3CLI_RUNTIME_CACHE']=str(parent/'cache')
  self.assertIn('unsafe cache ancestor',self.run_bundle().stderr)
 def test_paths_duplicates_ancestors_modes_numbers(self):
  for path in ('../outside','/tmp/outside','bin//file','bin/./file','bin/../file','bin/h3cli\x00/hidden','bin\\file'):
   with self.subTest(path=path):self.pack();self.edit(lambda m:m['files'][0].update(path=path));self.assertNotEqual(self.run_bundle().returncode,0)
  for edit in (lambda m:m['files'][1].update(path=m['files'][0]['path']),lambda m:m['files'][1].update(path='bin'),
               lambda m:m['files'][0].update(size=1<<62),lambda m:m['files'][0].update(mode=0o4755),
               lambda m:m['files'][0].update(size=True),lambda m:m.update(platform='linux')):
   self.pack();self.edit(edit);self.assertNotEqual(self.run_bundle().returncode,0)
 def test_truncation_hash_and_bomb(self):
  for kind in ('truncated','manifest','payload'):
   self.pack();raw=bytearray(self.archive.read_bytes())
   if kind=='truncated':raw=raw[:-1]
   elif kind=='manifest':raw[104]^=1
   else:raw[-5]^=1
   self.archive.write_bytes(raw);self.embed();self.assertNotEqual(self.run_bundle().returncode,0)
  self.pack();self.edit(lambda m:m['files'][1].update(size=1));self.assertNotEqual(self.run_bundle().returncode,0)
 def test_noncanonical_and_duplicate_json(self):
  raw=self.archive.read_bytes();magic,m,z,n,mh,ph=struct.unpack('<16sQQQ32s32s',raw[:104])
  for encoded in (b' '+raw[104:104+m],raw[104:104+m-1]+b',"schema":1}'):
   self.archive.write_bytes(struct.pack('<16sQQQ32s32s',magic,len(encoded),z,n,hashlib.sha256(encoded).digest(),ph)+encoded+raw[104+m:]);self.embed()
   self.assertIn('canonical JSON',self.run_bundle().stderr)
 def test_signature_coverage_and_reproducible_bytes(self):
  before=package.sha(self.binary);self.embed();self.assertEqual(before,package.sha(self.binary))
  broken=self.work/'broken';raw=bytearray(self.binary.read_bytes());offset=raw.index(self.archive.read_bytes());raw[offset+105]^=1;broken.write_bytes(raw);broken.chmod(0o755)
  r=subprocess.run(['codesign','--verify','--strict',str(broken)],capture_output=True);self.assertNotEqual(r.returncode,0)
 def test_partial_write_retry(self):
  def limited():resource.setrlimit(resource.RLIMIT_FSIZE,(4096,4096));signal.signal(signal.SIGXFSZ,signal.SIG_IGN)
  self.assertNotEqual(self.run_bundle(preexec_fn=limited).returncode,0);self.assertFalse(self.cache_root().exists())
  self.assertTrue(list(self.cache.glob('.stage-*')));self.assertEqual(self.run_bundle().returncode,0)
 def test_kill_extraction_retry(self):
  (self.runtime/'sample').write_bytes(os.urandom(16*1024*1024))
  package.write(self.runtime/'share/h3cli/runtime.json',{'schema':1,'platform':'macos-arm64','files':package.inventory(self.runtime)});self.pack()
  p=subprocess.Popen([str(self.binary)],env=self.env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
  try:
   deadline=time.monotonic()+10
   while not list(self.cache.glob('.stage-*')) and p.poll() is None and time.monotonic()<deadline:time.sleep(.001)
   self.assertIsNone(p.poll());p.kill();p.wait(timeout=10)
  finally:
   if p.poll() is None:p.kill();p.wait()
   p.stderr.close()
  self.assertFalse(self.cache_root().exists());self.assertEqual(self.run_bundle().returncode,0)
 def test_old_and_new_runtimes(self):
  self.assertEqual(self.run_bundle().returncode,0);old=self.cache_root();self.edit(lambda m:m.update(recipe='changed'))
  self.assertNotEqual(old,self.cache_root());self.assertEqual(self.run_bundle().returncode,0);self.assertTrue(old.is_dir())

class Source(unittest.TestCase):
 def test_archive_exclusions_and_source_models(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'source';root.mkdir()
   for n in ('src/models/catalog.c','src/a.c','models/weights','bin/app','outputs/token','a.o'):
    p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(n)
   out=Path(d)/'out';record=build.snapshot(root,out,1)
   self.assertEqual(set(record['files']),{'src/models/catalog.c','src/a.c'})
 def test_secret_and_symlink_rejection(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'.env').write_text('secret')
   with self.assertRaises(ValueError):build.names(root)
   (root/'.env').unlink();(root/'link.c').symlink_to('/etc/hosts')
   with self.assertRaises(ValueError):build.names(root)
 def test_no_silent_notary_fallback(self):
  r=subprocess.run([sys.executable,str(ROOT/'scripts/macos/release.py'),'--input','absent','--output','absent','--identity','-','--keychain-profile','profile'],capture_output=True)
  self.assertNotEqual(r.returncode,0);self.assertIn(b'Developer ID',r.stderr)

if __name__=='__main__':unittest.main(verbosity=2)
