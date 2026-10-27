"""Notarization retries retain the accepted bytes and Apple submission IDs."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/macos'))
spec=importlib.util.spec_from_file_location('resumable_notary',ROOT/'scripts/macos/release.py')
notary=importlib.util.module_from_spec(spec);spec.loader.exec_module(notary)


class NotaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.file=self.root/'runtime.zip';self.file.write_bytes(b'accepted code')
        self.calls=[]

    def tearDown(self):self.tmp.cleanup()

    def command(self,argv,**kwargs):
        self.calls.append(argv)
        action=argv[2]
        if action=='history':row={'history':[]}
        elif action=='submit':row={'id':'saved-id','status':'In Progress'}
        elif action=='wait':row={'id':argv[3],'status':'Accepted'}
        elif action=='log':
            Path(argv[-1]).write_text(json.dumps({'status':'Accepted','issues':None}));row={}
        else:raise AssertionError(action)
        return subprocess.CompletedProcess(argv,0,json.dumps(row),'')

    def test_complete_retry_does_not_upload_or_wait_again(self):
        with patch.object(notary.build_process,'run',side_effect=self.command):
            first=notary.notarize(self.file,'profile',self.root,'inner')
            self.calls=[]
            second=notary.notarize(self.file,'profile',self.root,'inner')
        self.assertEqual(first['id'],second['id']);self.assertEqual([x[2] for x in self.calls],['log'])

    def test_interrupted_wait_resumes_same_submission(self):
        def interrupt(argv,**kw):
            if argv[2]=='wait':raise KeyboardInterrupt
            return self.command(argv,**kw)
        with patch.object(notary.build_process,'run',side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):notary.notarize(self.file,'profile',self.root,'inner')
        self.assertEqual(json.loads((self.root/'inner-submission.json').read_text())['id'],'saved-id')
        self.calls=[]
        with patch.object(notary.build_process,'run',side_effect=self.command):
            notary.notarize(self.file,'profile',self.root,'inner')
        self.assertEqual([x[2] for x in self.calls],['wait','log'])

    def test_lost_acknowledgement_reconciles_unique_history_name(self):
        row={'name':'unique-upload.zip','sha256':notary.package.sha(self.file)}
        (self.root/'inner-upload.json').write_text(json.dumps(row))
        def history(argv,**kw):
            if argv[2]=='history':return subprocess.CompletedProcess(argv,0,json.dumps({'history':[dict(name=row['name'],id='existing',status='Accepted')]}),'')
            return self.command(argv,**kw)
        with patch.object(notary.build_process,'run',side_effect=history):
            result=notary.notarize(self.file,'profile',self.root,'inner')
        self.assertEqual(result['id'],'existing');self.assertNotIn('submit',[x[2] for x in self.calls])

    def test_mutated_upload_cannot_reuse_notary_result(self):
        with patch.object(notary.build_process,'run',side_effect=self.command):
            notary.notarize(self.file,'profile',self.root,'inner')
        self.file.write_bytes(b'different code')
        with patch.object(notary.build_process,'run',side_effect=AssertionError('must not contact Apple')):
            with self.assertRaisesRegex(ValueError,'bytes changed'):notary.notarize(self.file,'profile',self.root,'inner')

    def test_error_in_accepted_log_blocks(self):
        def badlog(argv,**kw):
            result=self.command(argv,**kw)
            if argv[2]=='log':Path(argv[-1]).write_text(json.dumps({'status':'Accepted','issues':[{'severity':'error'}]}))
            return result
        with patch.object(notary.build_process,'run',side_effect=badlog):
            with self.assertRaisesRegex(ValueError,'log contains an error'):notary.notarize(self.file,'profile',self.root,'inner')


if __name__=='__main__':unittest.main()
