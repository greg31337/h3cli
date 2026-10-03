#!/usr/bin/env python3
"""Small CPU-only checks for deadline persistence and step/dependency gates."""
import json,pathlib,subprocess,sys,tempfile,time,unittest

RUNNER=pathlib.Path(__file__).with_name('fast_vae_campaign.py').resolve()
class Campaign(unittest.TestCase):
    def test_deadline_and_admission(self):
        with tempfile.TemporaryDirectory(prefix='h3-vae-campaign-') as tmp:
            root=pathlib.Path(tmp);manifest=root/'jobs.json';reports=root/'results'
            def run(jobs,budget='3',ok=True):
                manifest.write_text(json.dumps(jobs))
                p=subprocess.run([sys.executable,str(RUNNER),str(manifest),'--root',str(reports),'--budget-minutes',budget,'--reserve-minutes','1'],capture_output=True,text=True,timeout=30)
                self.assertEqual(p.returncode==0,ok,p.stdout+p.stderr)
            def job(name,code,**kw):return dict(id=name,argv=[sys.executable,'-c',code],**kw)
            run([job('first','pass')]);initial=json.loads((reports/'ledger.json').read_text())
            run([job('second','pass'),job('too-long','raise RuntimeError("must not run")',estimated_seconds=999),job('dependency','raise RuntimeError("must not run")',requires=['missing'])],budget='999')
            final=json.loads((reports/'ledger.json').read_text())
            self.assertEqual(initial['deadline_unix'],final['deadline_unix']);self.assertEqual(final['budget_seconds'],180)
            for name in ['too-long','dependency']:self.assertEqual(json.loads((reports/name/'record.json').read_text())['status'],'deferred')
            self.assertEqual(len(final['entries']),4)
            final['finished_unix']=final['started_unix']+2;(reports/'ledger.json').write_text(json.dumps(final));run([job('closed','pass')],ok=False)

    def test_six_step_provenance(self):
        with tempfile.TemporaryDirectory(prefix='h3-vae-steps-') as tmp:
            root=pathlib.Path(tmp);jobs=[]
            for name,count in [('complete',6),('incomplete',5)]:
                jobs.append(dict(id=name,argv=[sys.executable,'-c',f'import os,sys;assert os.environ["H3_TEST_MAX_EVALUATIONS"]=="6";sys.stderr.write("denoise {count}/6\\n")','--steps','6']))
            p=root/'jobs.json';p.write_text(json.dumps(jobs));result=subprocess.run([sys.executable,str(RUNNER),str(p),'--root',str(root/'records'),'--budget-minutes','3','--reserve-minutes','1'],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads((root/'records/complete/record.json').read_text())['status'],'pass')
            self.assertEqual(json.loads((root/'records/incomplete/record.json').read_text())['status'],'failed-provenance')

    def test_waiting_runner_releases_campaign_lock(self):
        with tempfile.TemporaryDirectory(prefix='h3-vae-handoff-') as tmp:
            root=pathlib.Path(tmp);reports=root/'records';ready=root/'ready'
            waiting=root/'waiting.json';producer=root/'producer.json'
            waiting.write_text(json.dumps([dict(id='waiting',argv=[sys.executable,'-c','pass'],requires=['producer'])]))
            producer.write_text(json.dumps([dict(id='producer',argv=[sys.executable,'-c',
                'import pathlib,sys;pathlib.Path(sys.argv[1]).touch()',str(ready)])]))
            args=[sys.executable,str(RUNNER),'--root',str(reports),'--budget-minutes','3','--reserve-minutes','1']
            child=subprocess.Popen(args+[str(waiting),'--wait-for',str(ready)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                limit=time.monotonic()+10
                while not (reports/'ledger.json').exists() and time.monotonic()<limit:time.sleep(.05)
                self.assertTrue((reports/'ledger.json').is_file())
                initial=json.loads((reports/'ledger.json').read_text())
                result=subprocess.run(args+[str(producer)],capture_output=True,text=True,timeout=15)
                self.assertEqual(result.returncode,0,result.stderr)
                stdout,stderr=child.communicate(timeout=15);self.assertEqual(child.returncode,0,stdout+stderr)
                ledger=json.loads((reports/'ledger.json').read_text())
                self.assertEqual(ledger['deadline_unix'],initial['deadline_unix'])
                self.assertEqual([x['id'] for x in ledger['entries']],['producer','waiting'])
                self.assertTrue(all(x['status']=='pass' for x in ledger['entries']))
            finally:
                if child.poll() is None:child.kill();child.communicate()
if __name__=='__main__':unittest.main()
