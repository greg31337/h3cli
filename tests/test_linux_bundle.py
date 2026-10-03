#!/usr/bin/env python3
"""Real Linux launcher tests with a small native payload, never H3 render evidence."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import resource
import struct
import subprocess
import sys
import tempfile
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('package',ROOT/'scripts/linux/package.py')
package=importlib.util.module_from_spec(spec);spec.loader.exec_module(package)
CORE=r'''
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
int main(int argc,char **argv){
 if(argc>1&&!strcmp(argv[1],"sleep")){puts("ready");fflush(stdout);for(;;)pause();}
 if(argc>1&&!strcmp(argv[1],"status"))return 19;
 if(argc>2&&!strcmp(argv[1],"fd")){char c=0;if(read(atoi(argv[2]),&c,1)!=1)return 20;return c=='x'?0:21;}
 char cwd[4096];getcwd(cwd,sizeof(cwd));printf("cwd=%s\n",cwd);
 printf("root=%s\n",getenv("H3CLI_RUNTIME_ROOT"));
 printf("preload=%s\n",getenv("LD_PRELOAD")?getenv("LD_PRELOAD"):"unset");
 for(int i=1;i<argc;i++)printf("arg=%s\n",argv[i]);return 0;
}
'''

@unittest.skipUnless(sys.platform.startswith('linux'),'native Linux launcher')
class Bundle(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.launcher=Path(os.environ['H3_TEST_BUNDLE_LAUNCHER']).resolve()
  (ROOT/'bin').mkdir(exist_ok=True)
  cls.scratch=tempfile.TemporaryDirectory(prefix='bundle-tests-',dir=ROOT/'bin');cls.base=Path(cls.scratch.name)
  (cls.base/'core.c').write_text(CORE)
  cls.core=cls.base/'core'
  subprocess.run(['gcc',str(cls.base/'core.c'),'-o',str(cls.core)],check=True)
 @classmethod
 def tearDownClass(cls):cls.scratch.cleanup()
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=self.base);self.work=Path(self.tmp.name);self.runtime=self.work/'runtime';(self.runtime/'bin').mkdir(parents=True)
  shutil.copy2(self.core,self.runtime/'bin/h3cli');(self.runtime/'lib').mkdir();(self.runtime/'lib/sample').write_bytes(b'a'*100000)
  self.binary=self.work/'h3cli';package.package(self.runtime,self.launcher,self.binary)
  self.cache=self.work/'cache'
  self.env={**os.environ,'H3CLI_RUNTIME_CACHE':str(self.cache)}
  for name in ('H3CLI_BUNDLE_INFO','LD_PRELOAD','LD_AUDIT'):self.env.pop(name,None)
 def tearDown(self):self.tmp.cleanup()
 def run_bundle(self,*args,**kwargs):return subprocess.run([str(self.binary),*args],env=self.env,cwd=self.work,text=True,capture_output=True,timeout=30,**kwargs)
 def info(self):
  r=subprocess.run([str(self.binary)],env=self.env|{'H3CLI_BUNDLE_INFO':'1'},text=True,capture_output=True,check=True)
  return json.loads(r.stdout)
 def rewrite_manifest(self,edit):
  data=self.binary.read_bytes();parts=list(struct.unpack('<16sQQQQ32s32sQQ',data[-128:]));m=json.loads(data[parts[1]:parts[1]+parts[2]]);edit(m)
  encoded=json.dumps(m,sort_keys=True,separators=(',',':')).encode();parts[2]=len(encoded);parts[5]=hashlib.sha256(encoded).digest()
  self.binary.write_bytes(data[:parts[1]]+encoded+struct.pack('<16sQQQQ32s32sQQ',*parts));self.binary.chmod(0o755)
 def test_cold_warm_cwd_argv_and_exit_status(self):
  self.env['LD_PRELOAD']='/must-not-load.so'
  r=self.run_bundle('a b','$(touch escaped)','',"'quoted'",'é 海')
  self.assertEqual(r.returncode,0,r.stderr);self.assertIn('preload=unset',r.stdout);self.assertIn('cwd='+str(self.work),r.stdout)
  self.assertIn('arg=$(touch escaped)',r.stdout);self.assertFalse((self.work/'escaped').exists())
  root=Path(self.info()['cache']);before={str(p):p.stat().st_mtime_ns for p in root.rglob('*')}
  self.assertEqual(self.run_bundle().returncode,0)
  self.assertEqual(before,{str(p):p.stat().st_mtime_ns for p in root.rglob('*')})
  self.assertEqual(self.run_bundle('status').returncode,19)
 def test_parallel_first_launches(self):
  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
   results=list(pool.map(lambda _:self.run_bundle(),range(6)))
  self.assertTrue(all(r.returncode==0 for r in results),[r.stderr for r in results])
  self.assertEqual(len(list(self.cache.glob('*/.ready.json'))),1)
 def test_symlink_rename_readonly_download_and_relocated_cache(self):
  renamed=self.work/'renamed tool';self.binary.rename(renamed);renamed.chmod(0o555);self.binary.symlink_to(renamed.name)
  self.assertEqual(self.run_bundle().returncode,0)
  self.env['H3CLI_RUNTIME_CACHE']=str(self.work/'new cache')
  self.assertEqual(self.run_bundle().returncode,0)
 def test_same_size_cache_corruption_fails_closed(self):
  self.assertEqual(self.run_bundle().returncode,0)
  p=Path(self.info()['cache'])/'lib/sample';p.write_bytes(b'b'*p.stat().st_size)
  r=self.run_bundle();self.assertNotEqual(r.returncode,0);self.assertIn('cached runtime changed',r.stderr)
 def test_inherited_descriptor_and_signal(self):
  read,write=os.pipe();os.write(write,b'x');os.close(write)
  try:self.assertEqual(self.run_bundle('fd',str(read),pass_fds=(read,)).returncode,0)
  finally:os.close(read)
  p=subprocess.Popen([str(self.binary),'sleep'],env=self.env,cwd=self.work,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
  try:
   self.assertEqual(p.stdout.readline().strip(),'ready');p.send_signal(signal.SIGTERM);self.assertEqual(p.wait(timeout=10),-signal.SIGTERM)
  finally:
   if p.poll() is None:p.kill();p.wait()
   p.stdout.close();p.stderr.close()
 def test_info_needs_neither_cache_nor_cuda(self):
  self.assertEqual(self.info()['requirements']['glibc'],'2.35');self.assertFalse(self.cache.exists())
 def test_corrupt_payload_and_footer(self):
  data=bytearray(self.binary.read_bytes());parts=struct.unpack('<16sQQQQ32s32sQQ',data[-128:]);data[parts[3]]^=1;self.binary.write_bytes(data)
  self.assertIn('payload checksum mismatch',self.run_bundle().stderr)
  data[-128]^=1;self.binary.write_bytes(data);self.assertIn('missing bundle trailer',self.run_bundle().stderr)
 def test_traversal_duplicate_and_overflow(self):
  for path in ('../outside','/tmp/outside','bin/../outside','bin//outside','bin/./outside'):
   with self.subTest(path=path):
    package.package(self.runtime,self.launcher,self.binary);self.rewrite_manifest(lambda m:m['files'][0].update(path=path));self.assertNotEqual(self.run_bundle().returncode,0)
  package.package(self.runtime,self.launcher,self.binary)
  self.rewrite_manifest(lambda m:m['files'][1].update(path=m['files'][0]['path']))
  self.assertIn('duplicate',self.run_bundle().stderr)
  package.package(self.runtime,self.launcher,self.binary)
  self.rewrite_manifest(lambda m:m['files'][0].update(size=1<<62))
  self.assertNotEqual(self.run_bundle().returncode,0)
 def test_symlink_payload_and_cache_rejected(self):
  (self.runtime/'lib/link').symlink_to('/etc/passwd')
  with self.assertRaises(ValueError):package.package(self.runtime,self.launcher,self.binary)
  self.cache.symlink_to(self.runtime,target_is_directory=True)
  self.assertNotEqual(self.run_bundle().returncode,0)
 def test_reproducible_bytes(self):
  other=self.work/'second';package.package(self.runtime,self.launcher,other)
  self.assertEqual(package.sha(self.binary),package.sha(other))
 def test_metadata_only_change_gets_a_separate_cache(self):
  self.assertEqual(self.run_bundle().returncode,0)
  first=self.info()['cache']
  self.rewrite_manifest(lambda m:m['files'][1].update(mode=0o755))
  self.assertNotEqual(first,self.info()['cache'])
  self.assertEqual(self.run_bundle().returncode,0)
  self.assertTrue((Path(first)/'.ready.json').is_file())
 def test_missing_file_marker_and_writable_cache_fail(self):
  self.assertEqual(self.run_bundle().returncode,0)
  root=Path(self.info()['cache']);(root/'.ready.json').write_text('{}')
  self.assertNotEqual(self.run_bundle().returncode,0)
  shutil.rmtree(root);self.assertEqual(self.run_bundle().returncode,0)
  (root/'lib/sample').unlink();self.assertNotEqual(self.run_bundle().returncode,0)
  self.cache.chmod(0o777)
  self.assertIn('not writable by others',self.run_bundle().stderr)
 def test_failed_write_then_retry_preserves_incomplete_stage(self):
  def limited():
   resource.setrlimit(resource.RLIMIT_FSIZE,(4096,4096));signal.signal(signal.SIGXFSZ,signal.SIG_IGN)
  failed=self.run_bundle(preexec_fn=limited)
  self.assertNotEqual(failed.returncode,0)
  stages=list(self.cache.glob('.extract-*'));self.assertTrue(stages)
  self.assertFalse(Path(self.info()['cache']).exists())
  self.assertEqual(self.run_bundle().returncode,0)
  self.assertTrue(all(p.exists() for p in stages))
 def test_sigkill_during_extraction_then_retry(self):
  # Incompressible data keeps the actual extractor busy without a test-only hook.
  (self.runtime/'lib/sample').write_bytes(os.urandom(16*1024*1024))
  package.package(self.runtime,self.launcher,self.binary)
  p=subprocess.Popen([str(self.binary)],env=self.env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
  deadline=time.monotonic()+10
  while not list(self.cache.glob('.extract-*')) and p.poll() is None and time.monotonic()<deadline:time.sleep(.001)
  try:
   self.assertIsNone(p.poll(),'fixture finished before extraction could be interrupted')
   p.kill();p.wait(timeout=5)
  finally:
   if p.poll() is None:p.kill();p.wait()
   p.stderr.close()
  self.assertFalse(Path(self.info()['cache']).exists())
  self.assertEqual(self.run_bundle().returncode,0)
 def test_manifest_nul_and_symlink_lock_rejected(self):
  self.rewrite_manifest(lambda m:m['files'][0].update(path='bin/h3cli\x00/hidden'))
  self.assertIn('embedded NUL',self.run_bundle().stderr)
  package.package(self.runtime,self.launcher,self.binary)
  self.cache.mkdir(exist_ok=True);(self.cache/'.extract.lock').symlink_to('/dev/null')
  self.assertNotEqual(self.run_bundle().returncode,0)
 def test_noexec_cache_has_actionable_error(self):
  base=Path('/dev/shm')
  if not base.is_dir() or not os.statvfs(base).f_flag&getattr(os,'ST_NOEXEC',8):
   self.skipTest('No /dev/shm noexec mount on this host')
  with tempfile.TemporaryDirectory(prefix='h3-bundle-noexec-',dir=base) as tmp:
   self.env['H3CLI_RUNTIME_CACHE']=tmp
   r=self.run_bundle();self.assertNotEqual(r.returncode,0);self.assertIn('check cache mount permits execution',r.stderr)

if __name__=='__main__':unittest.main()
