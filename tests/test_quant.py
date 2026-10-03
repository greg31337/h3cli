#!/usr/bin/env python3
"""Host option and persistent budget contracts; no model/GPU required."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import json
import hashlib
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest
from quant_qualify import gallery
from quant_followup import decisions

ROOT = Path(__file__).resolve().parents[1]


class QuantContracts(unittest.TestCase):
    def test_followup_seam_and_turbo_decisions_are_independent_and_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);rows=[]
            for case in ('continuation-hard','continuation-bridge','turbo-strength-1'):
                row=dict(case=case,outputs={})
                for mode in ('off','fp8','nvfp4'):
                    path=out/f'{case}-{mode}.mp4';path.write_bytes(path.name.encode())
                    row['outputs'][mode]=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                rows.append(row)
            manifest=out/'review-manifest.json';manifest.write_text(json.dumps(dict(rows=rows)))
            pending=decisions(out)
            self.assertTrue(all(v['status']=='pending-or-stale' for v in pending.values()))
            acceptance=dict(gallery_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                            decisions={'continuation':dict(status='accepted',media=pending['continuation']['media'])})
            (out/'acceptance.json').write_text(json.dumps(acceptance))
            self.assertEqual(decisions(out)['continuation']['status'],'accepted')
            self.assertEqual(decisions(out)['turbo']['status'],'pending-or-stale')
            (out/'continuation-hard-off.mp4').write_bytes(b'replaced baseline')
            self.assertEqual(decisions(out)['continuation']['status'],'pending-or-stale')

    def test_gallery_acceptance_is_bound_to_media_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);q=out/'quality';q.mkdir()
            row=dict(name='fixture',role='held-out',prompt='test',outputs={})
            media={}
            for mode in ('off','fp8','nvfp4'):
                path=q/(mode+'.mp4');path.write_bytes(mode.encode())
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                row['outputs'][mode]=dict(path=str(path),sha256=digest)
                media[mode]=digest
            (q/'fixture.json').write_text(json.dumps(row));gallery(out)
            self.assertIn('acceptance pending', (q/'review.html').read_text())
            decision=dict(gallery_manifest_sha256=hashlib.sha256((q/'manifest.json').read_bytes()).hexdigest(),
                          decisions={mode:dict(status='accepted',media={'fixture':media[mode]}) for mode in ('fp8','nvfp4')})
            (q/'acceptance.json').write_text(json.dumps(decision));gallery(out)
            self.assertEqual((q/'review.html').read_text().count('human playback/listening accepted'),2)
            (q/'off.mp4').write_bytes(b'changed baseline');gallery(out)
            self.assertNotIn('human playback/listening accepted', (q/'review.html').read_text())
            (q/'off.mp4').write_bytes(b'off')
            (q/'nvfp4.mp4').write_bytes(b'changed');gallery(out)
            self.assertIn('NVFP4: human playback/listening acceptance pending or stale', (q/'review.html').read_text())
            row['prompt']='changed';(q/'fixture.json').write_text(json.dumps(row));gallery(out)
            self.assertNotIn('human playback/listening accepted', (q/'review.html').read_text())

    def cli(self, *args, env=None):
        return subprocess.run([str(ROOT/'bin/h3cli'), '-d', 'missing-model', *args],
                              env=env, text=True, capture_output=True, timeout=10)

    def test_mode_validation(self):
        r = self.cli('--cuda-denoise-quant', 'FP8')
        self.assertEqual(r.returncode, 2); self.assertIn('off, fp8 or nvfp4', r.stderr)
        r = self.cli('--cuda-denoise-quant-cache', '/tmp/unused')
        self.assertEqual(r.returncode, 2); self.assertIn('requires', r.stderr)
        with tempfile.NamedTemporaryFile() as f:
            r=self.cli('--cuda-denoise-quant','fp8','--cuda-denoise-quant-cache',f.name+'/packed')
            self.assertEqual(r.returncode,2);self.assertIn('cache path',r.stderr)
        if platform.system() == 'Darwin':
            for mode in ('fp8', 'nvfp4'):
                r = self.cli('--cuda-denoise-quant', mode)
                self.assertEqual(r.returncode, 2); self.assertIn('requires the CUDA backend', r.stderr)

    def test_budget_failure_timeout_and_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            base = [sys.executable, str(ROOT/'tests/quant_run.py'), '--out', d, '--timeout', '.1']
            def run(name, script):
                return subprocess.run([*base, '--name', name, '--', sys.executable, '-c', script],
                                      capture_output=True, text=True, timeout=10)
            self.assertEqual(run('pass', 'print("ok")').returncode, 0)
            self.assertEqual(run('failure', 'raise SystemExit(7)').returncode, 1)
            self.assertEqual(run('timeout', 'import time;print("partial",flush=True);time.sleep(5)').returncode, 1)
            ledger = json.loads((out/'budget.json').read_text())
            self.assertEqual(len(ledger['runs']), 3); self.assertGreaterEqual(ledger['seconds'], .1)
            self.assertIn('partial', (out/'timeout.log').read_text())
            self.assertNotEqual(run('pass', 'raise SystemExit(99)').returncode, 0)
            ledger['seconds'] = ledger['limit']-.05
            (out/'budget.json').write_text(json.dumps(ledger))
            self.assertIn('budget exhausted', run('not-launched', 'print("bad")').stderr)
            self.assertFalse((out/'not-launched.log').exists())

    def test_interrupted_reservation_is_charged(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'budget.json'
            path.write_text(json.dumps(dict(limit=3600, seconds=0, runs=[], pending=dict(name='lost',timeout=120))))
            r = subprocess.run([sys.executable,str(ROOT/'tests/quant_run.py'),'--out',d,'--name','next','--timeout','1','--',sys.executable,'-c','pass'],capture_output=True,text=True,timeout=10)
            self.assertEqual(r.returncode,0,r.stderr)
            ledger=json.loads(path.read_text());self.assertGreaterEqual(ledger['seconds'],120)
            self.assertEqual(ledger['runs'][0]['status'],'interrupted-run-reservation-charged')

    def test_explicit_unlimited_preserves_spending_and_case_timeouts(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'budget.json'
            original=dict(name='previous',status='passed',seconds=3599)
            path.write_text(json.dumps(dict(limit=3600,seconds=3599,runs=[original])))
            base=[sys.executable,str(ROOT/'tests/quant_run.py'),'--out',d]
            def run(name,extra,code):
                return subprocess.run([*base,'--name',name,*extra,'--',sys.executable,'-c',code],capture_output=True,text=True,timeout=10)
            self.assertEqual(run('extend',['--unlimited','--timeout','.1'],'pass').returncode,0)
            ledger=json.loads(path.read_text());self.assertIsNone(ledger['limit'])
            self.assertGreater(ledger['seconds'],3599);self.assertEqual(ledger['runs'][0],original)
            self.assertEqual(ledger['extensions'][0]['previous_limit'],3600)
            self.assertEqual(run('still-bounded',['--timeout','.1'],'import time;time.sleep(5)').returncode,1)
            ledger=json.loads(path.read_text());self.assertEqual(ledger['runs'][-1]['status'],'failed')
            self.assertEqual(len(ledger['extensions']),1)
            self.assertNotEqual(run('too-long',['--timeout','121'],'pass').returncode,0)


if __name__ == '__main__':
    unittest.main()
