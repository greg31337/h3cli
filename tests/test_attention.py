#!/usr/bin/env python3
"""CLI checks independent of model installation or CUDA allocation."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import platform
import subprocess
import tempfile
import unittest
import struct
from pathlib import Path

class AttentionCLI(unittest.TestCase):
    def request(self,args,environment=None):
        env={k:v for k,v in os.environ.items() if not k.startswith('H3_')}
        env.update(environment or {})
        return subprocess.run(['./bin/h3cli','-d','/nonexistent-h3-attention-model',*args],env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=20)
    def test_invalid_selection(self):
        for mode in ('','fp8','sage2','SAGE3','auto','-1'):
            with self.subTest(mode=mode):
                p=self.request(['--cuda-attention',mode,'-p','test'])
                self.assertNotEqual(p.returncode,0);self.assertIn('cuda-attention must be',p.stdout)
    def test_decode_selection_rejected(self):
        p=self.request(['--cuda-attention','default','--decode-av-state','missing.h3av'])
        self.assertNotEqual(p.returncode,0);self.assertIn('decode-only does not use cuda-attention',p.stdout)
    def test_failure_creates_no_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            cache=Path(folder)/'folds'
            p=self.request(['--cuda-attention','sage3','--lora','missing.safetensors','--lora-cache',str(cache),'-p','test'])
            self.assertNotEqual(p.returncode,0);self.assertFalse(cache.exists())
    def test_saved_attention_rejected_before_model_inventory(self):
        from test_sampler_file import entries,build
        # The fingerprint fixture needs a local filesystem with precise file
        # identity timestamps; the validation node's /workspace is networked.
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='h3-attention-resume-') as folder:
            path=Path(folder)/'state.h3sample'
            subprocess.run(['./bin/sampler_tests','--fixture',str(path)],check=True,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env={**os.environ,'TMPDIR':folder})
            parts=entries(path.read_bytes())+[[35,1,1,1,12,bytearray(struct.pack('<3I',1,1,1))]]
            path.write_bytes(build(parts))
            for flags in (['--cuda-attention','default'],):
                p=self.request(['--resume-sampler-state',str(path),*flags])
                self.assertNotEqual(p.returncode,0)
                self.assertIn('resume attention differs from the checkpoint',p.stdout)
                self.assertNotIn('missing required model',p.stdout)
            if platform.system()=='Darwin':
                p=self.request(['--resume-sampler-state',str(path)])
                self.assertIn('cuda-attention requires CUDA',p.stdout)
    def test_bounded_validation_cache_preserves_source(self):
        from attention_cache import prepare,clear_generated
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source';source.mkdir();original=source/'existing.h3q';original.write_bytes(b'original')
            cache=Path(folder)/'owned';prepare(cache,source)
            (cache/'new.h3q').write_bytes(b'new')
            self.assertEqual(clear_generated(cache),3)
            self.assertEqual(original.read_bytes(),b'original')
            self.assertTrue((cache/'existing.h3q').is_symlink())
            self.assertFalse((cache/'new.h3q').exists())
            unowned=Path(folder)/'unowned';unowned.mkdir();(unowned/'keep').write_text('keep')
            with self.assertRaises(RuntimeError):prepare(unowned,source)
            self.assertEqual((unowned/'keep').read_text(),'keep')
    @unittest.skipUnless(platform.system()=='Darwin','Metal build test')
    def test_metal_rejected(self):
        for mode in ('sage2++','sage3'):
            p=self.request(['--cuda-attention',mode,'-p','test'])
            self.assertNotEqual(p.returncode,0);self.assertIn('requires CUDA',p.stdout)

if __name__=='__main__':unittest.main()
