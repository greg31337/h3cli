#!/usr/bin/env python3
"""Host checks for locked downloads, source snapshots and bundle manifests."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/linux'))
import build
import environment
import package

class Build(unittest.TestCase):
 def test_help_works_without_linux_tools(self):
  r=subprocess.run(['bash',str(ROOT/'scripts/build_linux.sh'),'--help'],text=True,capture_output=True)
  self.assertEqual(r.returncode,0,r.stderr);self.assertIn('--offline',r.stdout);self.assertIn('--fetch-only',r.stdout)
 def test_existing_download_requires_matching_content(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'download';p.write_bytes(b'locked');sha=environment.digest(p)
   with patch.object(environment.urllib.request,'urlopen') as network:
    self.assertEqual(environment.fetch('https://invalid',p,sha),p)
    p.write_bytes(b'changed')
    with self.assertRaisesRegex(ValueError,'checksum mismatch'):environment.fetch('https://invalid',p,sha)
    network.assert_not_called()
 def test_offline_miss_never_accesses_network(self):
  with tempfile.TemporaryDirectory() as tmp,patch.object(environment.urllib.request,'urlopen') as network:
   with self.assertRaisesRegex(ValueError,'Missing offline'):build.cached('https://invalid',Path(tmp)/'missing','0'*64,True)
   network.assert_not_called()
 def test_native_wheel_pins_are_shared(self):
  pins=build.wheels();self.assertEqual(len(pins),4)
  self.assertIn(('nvidia-cublas','13.1.1.3','37936a16db8fe4ac1f065c2139360608a543a09275cb1a1af612e08cfa065436'),pins)
 def test_source_archive_snapshot_excludes_generated_outputs(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);source=root/'source';source.mkdir();(source/'src').mkdir();(source/'src/a.c').write_text('int a;')
   for name in ('bin/program','outputs/result','models/weights','src/a.o','.cuda-build-config','.cuda-build-config.tmp'):
    p=source/name;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(b'not source')
   (source/'src/models').mkdir();(source/'src/models/catalog.json').write_text('{}')
   with patch.object(build.subprocess,'check_output',side_effect=subprocess.CalledProcessError(1,['git'])):
    record=build.snapshot(source,root/'snapshot')
   self.assertEqual(set(record['files']),{'src/a.c','src/models/catalog.json'})
   self.assertEqual((root/'snapshot/src/a.c').stat().st_mtime,build.LOCK['source_date_epoch'])
 def test_seal_checks_contents_and_relocation(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'runtime';(root/'share/h3cli').mkdir(parents=True);(root/'bin').mkdir();p=root/'bin/h3cli';p.write_bytes(b'core')
   package.write_json(root/'share/h3cli/runtime.json',{'files':{'bin/h3cli':{'sha256':package.sha(p),'size':4}}})
   self.assertEqual(package.seal(root),{'sealed_files':2})
   moved=root.with_name('relocated');root.rename(moved)
   self.assertEqual(package.seal(moved),{'sealed_files':2})
   (moved/'bin/h3cli').write_bytes(b'evil')
   with self.assertRaisesRegex(ValueError,'changed runtime'):package.seal(moved)
 def test_abi_floor_examines_imports_not_exports(self):
  output='Version definition section\n Name: GLIBCXX_9.99\nVersion needs section\n Name: GLIBC_2.35\n Name: CXXABI_1.3\n'
  with patch.object(package.subprocess,'check_output',return_value=output):
   self.assertEqual(package.requirements(Path('fixture')),['CXXABI_1.3','GLIBC_2.35'])
 def test_oci_rejects_unpinned_cache(self):
  with tempfile.TemporaryDirectory() as tmp:
   cache=Path(tmp);oci=cache/'images/builder';oci.mkdir(parents=True);(oci/'index.json').write_text('{}')
   with self.assertRaisesRegex(ValueError,'locked origin'):environment.prepare(cache,'builder',offline=True)
 def test_vendor_search_paths_stay_inside_payload(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   library=root/'lib/nvidia/cudnn/lib/libcudnn_ops.so.9'
   package.check_rpaths(root,library,['$ORIGIN:$ORIGIN/../../cu13/lib'])
   for entries in (['/host/lib'],['$ORIGIN:'],['$ORIGIN/../../../../../../outside']):
    with self.subTest(entries=entries),self.assertRaises(ValueError):
     package.check_rpaths(root,library,entries)
 def test_generated_metal_include_is_not_source_input(self):
  sys.path.insert(0,str(ROOT/'tests'))
  import cuda_reference_regression as gate
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'src/metal').mkdir(parents=True)
   (root/'src/metal/gpu.m').write_text('source')
   before=gate.source_files(root)
   (root/'src/metal/native_attention.inc').write_text('generated')
   self.assertEqual(before,gate.source_files(root))
   with patch.object(build.subprocess,'check_output',side_effect=subprocess.CalledProcessError(1,['git'])):
    record=build.snapshot(root,root.parent/(root.name+'-snapshot'))
   self.addCleanup(__import__('shutil').rmtree,root.parent/(root.name+'-snapshot'))
   self.assertNotIn('src/metal/native_attention.inc',record['files'])

if __name__=='__main__':unittest.main()
