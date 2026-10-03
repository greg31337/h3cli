#!/usr/bin/env python3
"""CPU accounting checks for SubBlock/quantization triplets and shared ledgers."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cuda_adaptive_quant as runner
import cuda_subblock_quant as campaign
from adaptive_quant_report import publish

MANIFEST=Path(__file__).resolve().parents[1]/'docs/cuda/subblock-quant-manifest.json'


def evidence(variant):
    quant=bool(variant['args']);sparse='subblock' in variant['args'];steps=[]
    for i in range(50):
        active=sparse and i>=10
        steps.append(dict(step=i,evaluated=1,blocks=50,quant_calls=200 if quant else 0,
                          dense_calls=1 if active else 50,sparse_calls=49 if active else 0,
                          router_calls=98 if active else 0,protected_calls=98 if active else 0,
                          dense_bypass=1 if active else 50 if sparse else 0,
                          selected=21750671 if active else 0,possible=47811456 if active else 0))
    log='CUDA weight planner: resident\n';sidecar=''
    if quant:
        precision=variant['group'];recipe=4 if sparse else 2;mode=1 if precision=='fp8' else 2
        log+=f'DiT quantization={precision} recipe={recipe} weights=compressed-resident\n'
        log+=f'projection counters requested={precision} recipe={recipe} native_calls=10000 cache_hits=200 prepared=0\n'
        sidecar+=f'denoise_quant {mode} {recipe} '+64*'a'+'\n'
    if sparse:
        log+='DiT attention=subblock recipe=2 plan=1 sparsity=0.75 warmup=10 probe=dense\n'
        sidecar+='attention 4 2 1\nadaptive 0 0\nsubblock 0x1.8p-1\n'
    return steps,[],log,sidecar


class Accounting(unittest.TestCase):
    def test_fixed_setting(self):
        self.assertEqual(campaign.manifest(MANIFEST)['variants'],campaign.CASES)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'manifest.json';m=json.loads(MANIFEST.read_text());m['subblock_sparsity']=.8;p.write_text(json.dumps(m))
            with self.assertRaises(AssertionError):campaign.manifest(p)
    def test_exact_dispatch_and_wrong_policy(self):
        for case in campaign.CASES:
            valid=evidence(case);campaign.validate_dispatch(case,*valid)
            for field in ['quant_calls','blocks','dense_calls','router_calls']:
                steps,cache,log,sidecar=copy.deepcopy(valid);steps[10][field]+=1
                with self.subTest(case=case['id'],field=field),self.assertRaises(AssertionError):
                    campaign.validate_dispatch(case,steps,cache,log,sidecar)
            if case['id'].endswith('S75'):
                for field,wrong in [('router_calls',49),('router_calls',97),('protected_calls',49),('possible',47811455)]:
                    steps,cache,log,sidecar=copy.deepcopy(valid);steps[10][field]=wrong
                    with self.subTest(case=case['id'],field=field,wrong=wrong),self.assertRaises(AssertionError):
                        campaign.validate_dispatch(case,steps,cache,log,sidecar)
                for index in [9,10,49]:
                    steps,cache,log,sidecar=copy.deepcopy(valid);steps[index]['sparse_calls']=49 if index<10 else 0
                    with self.assertRaises(AssertionError):campaign.validate_dispatch(case,steps,cache,log,sidecar)
                steps,cache,log,sidecar=valid
                for text in [sidecar.replace('attention 4 2 1','attention 4 1 1'),sidecar.replace(' 4 '+64*'a',' 2 '+64*'a'),sidecar.replace('0x1.8p-1','0x1.99999ap-1')]:
                    with self.assertRaises(AssertionError):campaign.validate_dispatch(case,steps,cache,log,text)
                with self.assertRaises(AssertionError):campaign.validate_dispatch(case,steps,cache,log.replace('prepared=0','prepared=1'),sidecar)
    def test_shared_ledger_uses_subblock_callbacks(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'campaign';(root/'bin').mkdir();(root/'bin/h3cli').write_bytes(b'synthetic build')
            prep=root/'preparation.json';runner.write(prep,dict(passed=True,records=[dict(name='prepare-'+p,passed=True) for p in ('fp8','nvfp4')]))
            calls=[]
            def render(source,d,model,config,variant,env):
                self.assertEqual(env['H3_CUDA_WEIGHT_MODE'],'resident');calls.append(variant['id']);d.mkdir()
                case=next(c for c in campaign.CASES if c['id']==variant['id']);steps,cache,log,sidecar=evidence(case)
                runner.write(d/'steps.json',steps);runner.write(d/'cache.json',cache);(d/'render.log').write_text(log);(d/'final.h3av.presentation').write_text(sidecar)
                result=dict(passed=True,returncode=0,wall_seconds=1,counts=dict(transitions=50),artifacts={'final.h3av.presentation':runner.sha(d/'final.h3av.presentation')})
                runner.write(d/'result.json',result);return result
            args=['campaign','--source',str(root),'--model',str(root),'--out',str(out),'--manifest',str(MANIFEST),'--cache',str(root),'--preparation',str(prep)]
            with contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()));stack.enter_context(patch.dict('os.environ',{},clear=True))
                for name in ['model_metadata','runtime','prepare','source_files','packed_metadata']:stack.enter_context(patch.object(runner,name,return_value={}))
                stack.enter_context(patch.object(runner,'render',side_effect=render))
                with patch('sys.argv',args):runner.main(comparison=campaign)
                with patch('sys.argv',args+['--resume']):runner.main(comparison=campaign)
                self.assertEqual(calls,campaign.IDS)
                (out/'F8-S75/attempt-001/final.h3av.presentation').write_text('changed')
                with patch('sys.argv',args+['--resume']),self.assertRaises(AssertionError):runner.main(comparison=campaign)
    def test_report_labels_policy_and_links(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);rows=[]
            for case in campaign.CASES:
                d=out/case['id'];d.mkdir()
                for name in ['video.mp4','final.h3av','result.json','steps.json','cache.json','ffprobe.json']:(d/name).touch()
                rows.append(dict(case,directory=case['id'],wall_seconds=1,speed_ratio=1,denoise_seconds=1,hits=0,blocks=2500,
                    quantized_calls=10000,peak_vram_gib=1,peak_rss_gib=1,ssim_min=1,lpips_max=0,audio_relative_l2=0,
                    historical_similarity_pass=True,sparse_calls=1960,density=.45,attention_seconds=1,router_seconds=.1))
            for name in ['identity.json','ledger.json']:(out/name).write_text('{}')
            d=out/'metrics/pair';d.mkdir(parents=True);(d/'result.json').write_text('{}');(d/'frame.png').touch()
            pairs=[dict(id='pair',candidate='F8-S75',reference='F8-Q',summary=dict(ssim_min=1,lpips_max=0),audio=dict(relative_l2=0),worst_images=['metrics/pair/frame.png'])]
            publish(out,rows,pairs,{}, {}, {},subblock=True)
            text=(out/'index.html').read_text();self.assertIn('SubBlock 0.75',text);self.assertNotIn('BF16 block 0',text)
            self.assertNotIn('__TITLE__',text);self.assertEqual(json.loads((out/'report.json').read_text())['human_review'],'pending')
            self.assertEqual(len((out/'report.csv').read_text().splitlines()),7)


if __name__=='__main__':unittest.main()
