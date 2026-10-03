#!/usr/bin/env python3
"""CPU accounting tests: fixed workload, full scheduler trace and real dispatch."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile
import contextlib
import io
import shutil
import subprocess
import cuda_adaptive_subblock as campaign
from cuda_adaptive_subblock import manifest, trace, dispatch, IDS

class Accounting(unittest.TestCase):
    def setUp(self):
        self.m=manifest(Path(__file__).resolve().parents[1]/'docs/cuda/adaptive-subblock-manifest.json')
        self.steps=[dict(step=i,total=50,video_sigma=1-i/50,audio_sigma=1-i/50,evaluated=1,blocks=50,
                         dense_calls=50,sparse_calls=0,selected=0,possible=0) for i in range(50)]
    def log(self,steps):return '\n'.join('h3_experiment '+json.dumps(s) for s in steps)
    def test_complete(self):
        steps,cache=trace(self.log(self.steps));r=dispatch(self.m['variants'][0],steps,cache)
        self.assertEqual((len(IDS),r['transitions'],r['forwards'],r['blocks']),(12,50,50,2500))
    def test_missing_duplicate_and_nonfinite(self):
        for steps in (self.steps[:-1],self.steps+[self.steps[-1]],self.steps[::-1]):
            with self.assertRaises(AssertionError):trace(self.log(steps))
        self.steps[7]['video_sigma']=float('nan')
        with self.assertRaises(AssertionError):trace(self.log(self.steps))
    def test_sparse_warmup_and_cache(self):
        steps=copy.deepcopy(self.steps)
        for s in steps[10:]:s.update(dense_calls=1,sparse_calls=49)
        dispatch(self.m['variants'][7],steps,[])
        steps[3]['sparse_calls']=1
        with self.assertRaises(AssertionError):dispatch(self.m['variants'][7],steps,[])
        with self.assertRaises(AssertionError):dispatch(self.m['variants'][5],self.steps,[])
    def test_core_reuse_counts_fresh_heads(self):
        steps=copy.deepcopy(self.steps)
        for s in steps:
            if s['step']%4 and s['step']!=49:s.update(blocks=0,dense_calls=0)
        r=dispatch(self.m['variants'][3],steps,[])
        self.assertEqual(r['forwards'],50);self.assertLess(r['blocks'],2500)


class Ledger(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'bin').mkdir();(self.root/'bin/h3cli').write_bytes(b'synthetic executable identity')
        self.out=self.root/'campaign';self.manifest=Path(__file__).resolve().parents[1]/'docs/cuda/adaptive-subblock-manifest.json'
        self.calls=[];self.fail=None
    def tearDown(self):self.temp.cleanup()
    def render(self,source,out,model,m,variant,env):
        self.calls.append(variant['id']);out.mkdir()
        artifact=out/'synthetic.bin';artifact.write_bytes(variant['id'].encode())
        r=dict(passed=self.fail is None,returncode=self.fail or 0,wall_seconds=1,counts={'transitions':50},artifacts={'synthetic.bin':campaign.sha(artifact)})
        campaign.write(out/'result.json',r)
        if self.fail is not None:raise RuntimeError('synthetic interruption')
        return r
    def run_campaign(self,resume=False):
        args=['campaign','--source',str(self.root),'--model',str(self.root),'--manifest',str(self.manifest),'--out',str(self.out)]
        if resume:args+=['--resume']
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch('sys.argv',args));stack.enter_context(patch.dict('os.environ',{},clear=True))
            for name,value in [('model_metadata',{}),('runtime',{}),('prepare',{'seconds':0}),('source_files',{})]:
                stack.enter_context(patch.object(campaign,name,return_value=value))
            stack.enter_context(patch.object(campaign,'render',side_effect=self.render));campaign.main()
    def test_single_success_and_immutable_identity(self):
        self.run_campaign();self.assertEqual(self.calls,IDS)
        self.run_campaign(True);self.assertEqual(self.calls,IDS)
        with self.assertRaises(FileExistsError):self.run_campaign()
        (self.root/'bin/h3cli').write_bytes(b'changed build')
        with self.assertRaises(AssertionError):self.run_campaign(True)
        self.assertEqual(self.calls,IDS)
    def test_duplicate_and_changed_artifacts(self):
        self.run_campaign();case=self.out/'D0'
        shutil.copytree(case/'attempt-001',case/'attempt-002')
        with self.assertRaises(AssertionError):self.run_campaign(True)
        shutil.rmtree(case/'attempt-002');(case/'attempt-001/synthetic.bin').write_bytes(b'changed artifact')
        with self.assertRaises(AssertionError):self.run_campaign(True)
    def test_failure_retention_and_complete_coverage(self):
        self.fail=1
        with self.assertRaises(RuntimeError):self.run_campaign()
        ledger=json.loads((self.out/'ledger.json').read_text())
        self.assertEqual(ledger['planned'],IDS);self.assertFalse(ledger['complete']);self.assertFalse(ledger['cases'][0]['passed'])
        self.fail=None;self.run_campaign(True)
        self.assertFalse(json.loads((self.out/'D0/attempt-001/result.json').read_text())['passed'])
        ledger=json.loads((self.out/'ledger.json').read_text());self.assertTrue(ledger['complete']);self.assertEqual([c['id'] for c in ledger['cases']],IDS)
    def test_completed_render_needs_inspection(self):
        self.fail=0
        with self.assertRaises(RuntimeError):self.run_campaign()
        self.fail=None
        with self.assertRaises(AssertionError):self.run_campaign(True)
        self.assertEqual(self.calls,['D0'])
    def test_exact_media_shape_and_full_decode(self):
        streams=[dict(codec_type='video',width=640,height=480,nb_read_frames='90',avg_frame_rate='24/1',duration='3.75'),
                 dict(codec_type='audio',channels=2,sample_rate='32000',duration='3.775')]
        with patch.object(campaign.subprocess,'check_output',side_effect=lambda *a,**k:json.dumps({'streams':streams})),patch.object(campaign.subprocess,'run',return_value=subprocess.CompletedProcess([],0,stdout='',stderr='')):
            campaign.media(Path('synthetic.mp4'),{})
            streams[0]['nb_read_frames']='89'
            with self.assertRaises(AssertionError):campaign.media(Path('synthetic.mp4'),{})

if __name__=='__main__':unittest.main()
