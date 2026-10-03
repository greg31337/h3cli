#!/usr/bin/env python3
"""CPU checks for mixed-precision accounting and immutable six-video ledgers."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import cuda_adaptive_quant as campaign
from adaptive_quant_report import publish

MANIFEST = Path(__file__).resolve().parents[1]/'docs/cuda/adaptive-quant-manifest.json'


def evidence(variant):
    quant = bool(variant['args']); adaptive = '--adaptive-cache' in variant['args']
    steps = [dict(step=i, evaluated=1, blocks=1 if adaptive and i in (4, 6) else 50,
                  sparse_calls=0) for i in range(50)]
    for s in steps:
        s.update(dense_calls=s['blocks'], quant_calls=0 if not quant or s['blocks']==1 else 196 if adaptive else 200)
    cache = [dict(step=s['step'], decision='hit' if s['blocks']==1 else 'refresh',
                  reason='final' if s['step']==49 else 'synthetic', streak=int(s['blocks']==1)) for s in steps] if adaptive else []
    log = 'CUDA weight planner: resident\n'; sidecar = ''
    if quant:
        precision=variant['group']; recipe=3 if adaptive else 2; mode=1 if precision=='fp8' else 2
        log += f'DiT quantization={precision} recipe={recipe} weights=compressed-resident\n'
        log += f'projection counters requested={precision} recipe={recipe} native_calls={sum(s["quant_calls"] for s in steps)} cache_hits={196 if adaptive else 200} prepared=0\n'
        sidecar += f'denoise_quant {mode} {recipe} '+64*'a'+'\n'
        if adaptive:
            log += 'block 0 projections BF16, blocks 1-49 '+precision
            sidecar += 'adaptive 1 2\n'
    return steps, cache, log, sidecar


class Accounting(unittest.TestCase):
    def test_fixed_matrix(self):
        self.assertEqual(campaign.manifest(MANIFEST)['variants'], campaign.CASES)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'manifest.json'; m=json.loads(MANIFEST.read_text()); m['frames']=89
            p.write_text(json.dumps(m))
            with self.assertRaises(AssertionError):campaign.manifest(p)

    def test_dispatch_all_six_and_invalid_evidence(self):
        for variant in campaign.CASES:
            valid=evidence(variant)
            campaign.validate_dispatch(variant,*valid)
            for bad in ('missing-step','wrong-calls','stream'):
                steps,cache,log,sidecar=copy.deepcopy(valid)
                if bad=='missing-step':steps.pop()
                elif bad=='wrong-calls':steps[4]['quant_calls']+=4
                else:log+='\nCUDA weight planner: stream'
                with self.subTest(variant=variant['id'],bad=bad), self.assertRaises(AssertionError):
                    campaign.validate_dispatch(variant,steps,cache,log,sidecar)
            if '--adaptive-cache' in variant['args']:
                steps,cache,log,sidecar=valid
                for replacement in (sidecar.replace('adaptive 1 2','adaptive 1 1'),sidecar.replace(' 3 ',' 2 ')):
                    with self.assertRaises(AssertionError):campaign.validate_dispatch(variant,steps,cache,log,replacement)
                with self.assertRaises(AssertionError):campaign.validate_dispatch(variant,steps,cache,log.replace('prepared=0','prepared=1'),sidecar)


class Ledger(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name); self.out=self.root/'campaign'
        (self.root/'bin').mkdir(); (self.root/'bin/h3cli').write_bytes(b'synthetic binary')
        self.preparation=self.root/'preparation.json'
        campaign.write(self.preparation,dict(passed=True,records=[dict(name='prepare-'+p,passed=True) for p in ('fp8','nvfp4')]))
        self.calls=[]; self.fail=None
    def tearDown(self):self.temp.cleanup()
    def render(self,source,out,model,m,variant,env):
        self.calls.append(variant['id']); out.mkdir()
        self.assertEqual(env['H3_CUDA_WEIGHT_MODE'],'resident'); self.assertNotIn('H3_TEST_MAX_EVALUATIONS',env)
        original=next(v for v in campaign.CASES if v['id']==variant['id'])
        steps,cache,log,sidecar=evidence(original)
        campaign.write(out/'steps.json',steps); campaign.write(out/'cache.json',cache)
        (out/'render.log').write_text(log); (out/'final.h3av.presentation').write_text(sidecar)
        (out/'synthetic.bin').write_bytes(variant['id'].encode())
        r=dict(passed=self.fail is None,returncode=self.fail or 0,wall_seconds=1,counts={'transitions':50},
               artifacts={'synthetic.bin':campaign.sha(out/'synthetic.bin')})
        campaign.write(out/'result.json',r)
        if self.fail is not None:raise RuntimeError('synthetic failure')
        return r
    def run_campaign(self,resume=False):
        args=['campaign','--source',str(self.root),'--model',str(self.root),'--out',str(self.out),
              '--manifest',str(MANIFEST),'--cache',str(self.root),'--preparation',str(self.preparation)]
        if resume:args+=['--resume']
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch('sys.argv',args)); stack.enter_context(patch.dict('os.environ',{},clear=True))
            for name,value in [('model_metadata',{}),('runtime',{}),('prepare',{}),('source_files',{}),('packed_metadata',{})]:
                stack.enter_context(patch.object(campaign,name,return_value=value))
            stack.enter_context(patch.object(campaign,'render',side_effect=self.render)); campaign.main()
    def test_exactly_six_and_identity(self):
        self.run_campaign(); self.run_campaign(True); self.assertEqual(self.calls,campaign.IDS)
        with self.assertRaises(FileExistsError):self.run_campaign()
        (self.root/'bin/h3cli').write_bytes(b'changed')
        with self.assertRaises(AssertionError):self.run_campaign(True)
    def test_duplicate_and_artifact_tampering(self):
        self.run_campaign(); d=self.out/'F8-D0'
        shutil.copytree(d/'attempt-001',d/'attempt-002')
        with self.assertRaises(AssertionError):self.run_campaign(True)
        shutil.rmtree(d/'attempt-002'); (d/'attempt-001/synthetic.bin').write_bytes(b'changed')
        with self.assertRaises(AssertionError):self.run_campaign(True)
    def test_retains_failure(self):
        self.fail=1
        with self.assertRaises(RuntimeError):self.run_campaign()
        self.fail=None; self.run_campaign(True)
        ledger=json.loads((self.out/'ledger.json').read_text())
        self.assertTrue(ledger['complete']); self.assertEqual(ledger['cases'][0]['attempt_count'],2)
        self.assertFalse(json.loads((self.out/'F8-D0/attempt-001/result.json').read_text())['passed'])
    def test_never_repeats_completed_render(self):
        self.fail=0
        with self.assertRaises(RuntimeError):self.run_campaign()
        self.fail=None
        with self.assertRaises(AssertionError):self.run_campaign(True)
        self.assertEqual(self.calls,['F8-D0'])


class Report(unittest.TestCase):
    def test_triplet_links_and_pending_review(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp); rows=[]
            for case in campaign.CASES:
                d=out/case['id']; d.mkdir()
                for name in ('video.mp4','final.h3av','result.json','steps.json','cache.json','ffprobe.json'):(d/name).touch()
                rows.append(dict(case,directory=case['id'],wall_seconds=1,speed_ratio=1,denoise_seconds=1,hits=0,
                                 blocks=2500,quantized_calls=0,peak_vram_gib=1,peak_rss_gib=1,ssim_min=1,
                                 lpips_max=0,audio_relative_l2=0,historical_similarity_pass=True))
            for name in ('identity.json','ledger.json'):(out/name).write_text('{}')
            d=out/'metrics/pair'; d.mkdir(parents=True); (d/'result.json').write_text('{}'); (d/'frame.png').touch()
            pairs=[dict(id='pair',candidate='F8-A1',reference='F8-D0',summary=dict(ssim_min=1,lpips_max=0),
                        audio=dict(relative_l2=0),worst_images=['metrics/pair/frame.png'])]
            publish(out,rows,pairs,{}, {}, {})
            self.assertEqual(json.loads((out/'report.json').read_text())['human_review'],'pending')
            self.assertEqual(len((out/'report.csv').read_text().splitlines()),7)
            (d/'frame.png').unlink()
            with self.assertRaises(AssertionError):publish(out,rows,pairs,{}, {}, {})


if __name__=='__main__':unittest.main()
