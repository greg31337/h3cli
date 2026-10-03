#!/usr/bin/env python3
"""No-GPU checks of the fixed scope, budgets, success ledger and cost accounting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from upscale_campaign import IDS, PAIRS, accounting, budget, completed, manifest, transitions
from cuda_reference_regression import sha, write

ROOT=Path(__file__).resolve().parents[1]


class CampaignTests(unittest.TestCase):
    def test_scope_and_budgets(self):
        m=manifest(ROOT/'docs/experiments/latent-upscale-manifest.json')
        cases=[c for p in m['pairs'] for c in p['cases']]
        self.assertEqual(len(cases),14)
        self.assertEqual(sum(budget(c['method'])=='50' for c in cases),4)
        self.assertEqual(sum(c['evaluations'] for c in cases),220)
        self.assertTrue(all(budget(k)=='6' for k in ('transfer','initialize','refinement','decode','pixel','verification')))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'manifest.json'
            for field,value in [('frames',91),('source_steps',51),('seed',43)]:
                bad=copy.deepcopy(m);bad[field]=value;write(p,bad)
                with self.assertRaises(AssertionError):manifest(p)
            bad=copy.deepcopy(m);bad['pairs'][1]['cases'].pop();write(p,bad)
            with self.assertRaises(AssertionError):manifest(p)

    def test_success_and_failure_retention(self):
        with tempfile.TemporaryDirectory() as d:
            stage=Path(d);first=stage/'attempt-001';first.mkdir()
            write(first/'result.json',dict(passed=False,returncode=1))
            self.assertIsNone(completed(stage))
            second=stage/'attempt-002';second.mkdir();(second/'video.mp4').write_bytes(b'video')
            r=dict(passed=True,returncode=0,artifacts={'video.mp4':sha(second/'video.mp4')})
            write(second/'result.json',r);self.assertEqual(completed(stage)[0],second)
            (second/'video.mp4').write_bytes(b'changed')
            with self.assertRaises(AssertionError):completed(stage)
            (second/'video.mp4').write_bytes(b'video')
            third=stage/'attempt-003';third.mkdir();(third/'video.mp4').write_bytes(b'video');write(third/'result.json',r)
            with self.assertRaises(AssertionError):completed(stage)
        with tempfile.TemporaryDirectory() as d:
            stage=Path(d);attempt=stage/'attempt-001';attempt.mkdir()
            write(attempt/'result.json',dict(passed=False,returncode=0))
            with self.assertRaises(AssertionError):completed(stage)

    def test_transition_dispatch(self):
        rows=[dict(step=i,total=4,evaluated=1,blocks=50,sparse_calls=0,quant_calls=0,audio_sigma=0,video_sigma=.25/(i+1)) for i in range(4)]
        log='\n'.join('h3_experiment '+json.dumps(r) for r in rows)
        self.assertEqual(len(transitions(log,4,True)),4)
        with self.assertRaises(AssertionError):transitions(log,2,True)
        rows[2]['audio_sigma']=.01
        with self.assertRaises(AssertionError):transitions('\n'.join('h3_experiment '+json.dumps(r) for r in rows),4,True)

    def test_shared_costs(self):
        names=IDS+['shared/learned','shared/bilinear','shared/I4-init','shared/U2-init','shared/U4-init']
        r={'768p/'+n:dict(wall_seconds=10.,phase_seconds={}) for n in names}
        r['768p/L0']=dict(wall_seconds=100.,phase_seconds={'audio VAE':2.,'video VAE decode':8.})
        costs=accounting(r,'768p')
        self.assertEqual(costs['U0']['later_job_seconds'],20.)
        self.assertEqual(costs['U2']['later_job_seconds'],30.)
        self.assertEqual(costs['U4']['end_to_end_seconds'],120.)
        self.assertEqual(costs['P0']['end_to_end_seconds'],110.)
        self.assertEqual(costs['D0']['end_to_end_seconds'],10.)
        r['768p/shared/I4-init']['shared_noise_creation_seconds']=.5
        self.assertEqual(accounting(r,'768p')['U2']['later_job_seconds'],30.5)


if __name__=='__main__':unittest.main()
