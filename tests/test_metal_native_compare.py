#!/usr/bin/env python3
"""Regression: an unchanged continuation prefix cannot hide suffix failures."""
from array import array
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

def fixture(root,suffix_delta=0.,change_prefix=False):
    root.mkdir();vt,h,w,at=72,2,2,405;nv=24*vt*h*w;na=64*at
    values=array('f',[1.]*(nv+na))
    for ch in range(24):
        for i in range((ch*vt+57)*h*w,(ch+1)*vt*h*w):values[i]+=suffix_delta
    for ch in range(64):
        for i in range(nv+ch*at+320,nv+(ch+1)*at):values[i]+=suffix_delta
    if change_prefix:values[0]=1.+2**-23
    if sys.byteorder!='little':values.byteswap()
    header=bytearray(160);header[:8]=b'H3AV\r\n\x1a\n'
    struct.pack_into('<4I',header,8,1,160,0x01020304,1)
    struct.pack_into('<10I',header,24,32,32,243,vt,h,w,at,24,32,2)
    struct.pack_into('<3Q',header,64,42,nv*4,na*4)
    payload=values.tobytes();header[128:160]=hashlib.sha256(header[:128]+payload).digest()
    blob=bytes(header)+payload;(root/'result.h3av').write_bytes(blob)
    record={'evaluation_contract_pass':True,'av_sha256':hashlib.sha256(blob).hexdigest(),
            'case':'B1','evaluations':1,'blocks':50,'geometry':[32,32,243],
            'weight_precision':'bf16','sampler':'cpu-euler','seed':42,'prompt':'fixture',
            'reference_sha256':None,'continuation_sha256':'source','continuation_context':192,
            'limits_sha256':'frozen','conditioning_identity':'same','conditioning_sha256_after':'same',
            'timing':{'summed_denoise_seconds':1,'steady_median':None},'component_fences':False}
    (root/'record.json').write_text(json.dumps(record))

class ContinuationComparison(unittest.TestCase):
    def compare(self,delta=0.,prefix=False):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root/'reference');fixture(root/'candidate',delta,prefix)
            result=root/'comparison.json'
            run=subprocess.run([sys.executable,str(ROOT/'scripts/metal_native_compare.py'),
                str(root/'reference'),str(root/'candidate'),'--max-relative-l2','.05',
                '--min-cosine','.998','--output',str(result)],capture_output=True,text=True)
            return run,json.loads(result.read_text())
    def test_matching_prefix_and_suffix(self):
        run,result=self.compare();self.assertEqual(run.returncode,0,run.stderr);self.assertTrue(result['pass'])
    def test_prefix_cannot_dilute_suffix(self):
        run,result=self.compare(delta=.1);self.assertNotEqual(run.returncode,0)
        for kind in ('video','audio'):
            self.assertTrue(result['metrics'][kind]['pass'])
            self.assertFalse(result['metrics'][kind+'_suffix']['pass'])
            self.assertTrue(result['metrics'][kind+'_protected_prefix']['pass'])
    def test_one_prefix_bit_must_fail(self):
        run,result=self.compare(prefix=True);self.assertNotEqual(run.returncode,0)
        self.assertTrue(result['metrics']['video']['pass'])
        self.assertFalse(result['metrics']['video_protected_prefix']['pass'])

class WeightComparison(unittest.TestCase):
    def check_ablation(self, explicit=True, attention='dense', minimum=.75):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name,weight in [('reference','bf16'),('candidate','q8')]:
                fixture(root/name)
                path=root/name/'record.json';r=json.loads(path.read_text())
                r.update(backend='metal',attention='dense' if name=='reference' else attention,
                         weight_precision=weight,metal_options={'metal-weight-format':weight,
                         'sol-min-exact':.75 if name=='reference' else minimum})
                path.write_text(json.dumps(r))
            command=[sys.executable,str(ROOT/'scripts/metal_native_compare.py'),
                     str(root/'reference'),str(root/'candidate'),'--output',str(root/'result.json')]
            if explicit:command.append('--allow-weight-format-change')
            return subprocess.run(command,capture_output=True,text=True)
    def test_explicit_weight_ablation(self):
        r=self.check_ablation();self.assertEqual(r.returncode,0,r.stderr)
    def test_weight_change_requires_explicit_flag(self):
        self.assertNotEqual(self.check_ablation(explicit=False).returncode,0)
    def test_attention_gains_cannot_be_attributed_to_weights(self):
        self.assertNotEqual(self.check_ablation(attention='sol').returncode,0)
        self.assertNotEqual(self.check_ablation(minimum=.1).returncode,0)

if __name__=='__main__':unittest.main()
