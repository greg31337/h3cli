#!/usr/bin/env python3
"""Reject invalid cache/backend/test-budget requests before opening a model."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import subprocess
import sys
import unittest

class Preflight(unittest.TestCase):
    def reject(self,args,expected,limit='6'):
        env={**os.environ,'H3_TEST_MAX_EVALUATIONS':limit}
        result=subprocess.run(['./bin/h3cli','-d','/nonexistent/h3-model','-p','ordinary test prompt','--steps','1',*args],
            env=env,text=True,capture_output=True)
        self.assertEqual(result.returncode,2,result.stderr)
        self.assertIn(expected,result.stderr)
        self.assertNotIn('cannot open model',result.stderr)
    def test_invalid_policy(self):
        self.reject(['--backend','unknown'],'backend')
        self.reject(['--metal-attention','sol'],'require --backend metal')
        self.reject(['--metal-attention','sparse'],'attention')
        if sys.platform!='darwin':
            self.reject(['--backend','metal'],'Apple Metal execution')
        self.reject(['--metal-attention-kernel','steel-routed'],'require --backend metal')
    def test_metal_attention_flag(self):
        help_result=subprocess.run(['./bin/h3cli','--help'],text=True,capture_output=True)
        self.assertEqual(help_result.returncode,0)
        self.assertIn('--metal-attention MODE',help_result.stderr)
        self.assertNotIn('--attention',help_result.stderr.split())
        # Parsing succeeds and marks denoising as explicitly selected.
        self.reject(['--metal-attention','dense','--decode-av-state','missing.h3av'],
                    '--backend/--metal-attention select denoising')
        # Check rejection during option parsing, before backend/model checks.
        old=subprocess.run(['./bin/h3cli','--attention','dense','--help'],text=True,capture_output=True)
        self.assertEqual(old.returncode,2,old.stderr)
        if sys.platform!='darwin':
            self.reject(['--metal-attention','dense'],'select Apple Metal execution')
    @unittest.skipUnless(sys.platform=='darwin','native Metal CLI')
    def test_native_options(self):
        base=['--backend','metal']
        for option,value in [('sol-q-block','16'),('sol-kv-block','63'),('sol-tau','nan'),
                             ('sol-tau','-1'),('sol-min-exact','1.1'),('sol-dense-layers','51'),('sol-local-radius','-1'),('sol-dense-steps','1001'),('sol-dense-sigma','1.1'),('sol-dense-sigma','-.5')]:
            self.reject(base+['--'+option,value],'SOL' if value=='nan' or option=='sol-local-radius' else 'Metal')
        self.reject(base+['--metal-attention-kernel','unknown'],'kernel must be')
        self.reject(base+['--metal-attention-dtype','fp32'],'dtype must be')
        self.reject(base+['--metal-attention-layout','unknown'],'layout must be')
        self.reject(base+['--metal-attention-layout','fused'],'Metal')
        self.reject(base+['--metal-tier','unknown'],'tier must be')
        self.reject(base+['--metal-attention-dtype','fp16'],'fp16 requires steel-routed')
        self.reject(base+['--metal-ane','unknown'],'Metal ANE must be')
        mixed=base+['--metal-attention-kernel','steel-routed','--metal-attention-dtype','fp16','--metal-ane','static']
        for args in (['--metal-ane-chunk','33'],['--metal-ane-rows','0'],['--metal-ane-rows','513'],['--metal-ane-rows','20000']):
            self.reject(mixed+args,'ANE requires')
        for args in (['--layers','45'],['--reuse','2'],['--core-reuse','2'],['--token-reduction']):
            self.reject(base+args,'Metal requires')
    def test_cache_options(self):
        self.reject(['--save-conditioning',''],'nonempty paths')
        self.reject(['--load-conditioning',''],'nonempty paths')
        self.reject(['--conditioning-schedule'],'conditioning requires')
        self.reject(['--save-conditioning','cache','--conditioning-schedule','--layers','35'],'all 50 blocks')
        self.reject(['--load-conditioning','cache','--resume-sampler-state','state'],'sampler resume')
        self.reject(['--load-conditioning','cache','--decode-av-state','state'],'decode')
    def test_q8_policy(self):
        self.reject(['--metal-weight-format','q4'],'must be bf16 or q8')
        self.reject(['--metal-weight-format','q8'],'require --backend metal')
        if sys.platform=='darwin':
            base=['--backend','metal','--metal-weight-format','q8']
            self.reject(base+['--ssd-streaming'],'resident packed weights')
            self.reject(base+['--metal-ane','static'],'Q8 requires --metal-ane off')
            self.reject(base+['--metal-q8-kernel','unknown'],'must be mpsgraph or simdgroup')
            self.reject(['--backend','metal','--metal-q8-kernel','simdgroup'],'requires q8')
    def test_budget(self):
        self.reject(['--steps','7'],'test evaluation budget exceeded')
        self.reject(['--steps','1'],'must be 6',limit='5')
        self.reject(['--steps','20','--stop-after-step','7','--save-sampler-state','state'],'test evaluation budget exceeded')

if __name__=='__main__':unittest.main()
