#!/usr/bin/env python3
"""Reject incomplete/nonfinite diagnostics and stale M6 artifacts without GPU work."""
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from metal_q8_report import validate_tensors, validate_record, av, metric, verify_screens, SCREEN_FILES, sha
from test_metal_native_compare import fixture

class ReportEvidence(unittest.TestCase):
    def test_post_hoc_threshold_edits_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);hashes={}
            for name in SCREEN_FILES:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('frozen')
                hashes[name]=sha(path)
            verify_screens(hashes,root)
            (root/SCREEN_FILES[0]).write_text('relaxed after seeing results')
            with self.assertRaisesRegex(ValueError,'changed after frozen'):
                verify_screens(hashes,root)

    def test_missing_and_nonfinite_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)/'run';fixture(run)
            r=json.loads((run/'record.json').read_text())
            with self.assertRaisesRegex(ValueError,'incomplete step'):
                validate_tensors(run,r)
            (run/'steps').mkdir();r['step_tensor_sha256']={}
            values=av(run/'result.h3av')
            for m,data in zip(('video','audio'),values):
                for kind in ('input','velocity','latent'):
                    name=f'step-001-{m}-{kind}.f32'
                    (run/'steps'/name).write_bytes(data.tobytes())
                    r['step_tensor_sha256'][name]='checked separately'
            validate_tensors(run,r)
            path=run/'steps/step-001-video-velocity.f32'
            data=bytearray(path.read_bytes());struct.pack_into('<f',data,0,math.nan);path.write_bytes(data)
            with self.assertRaisesRegex(ValueError,'invalid diagnostic'):
                validate_tensors(run,r)

    def test_stale_artifact_rejected_before_loading_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp);(run/'run.log').write_text('modified')
            with self.assertRaisesRegex(ValueError,'changed artifact'):
                validate_record(run,{'sha256':{'run.log':'old digest'}},'binary digest')

    def test_nonfinite_final_and_wrong_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)/'run';fixture(run);r=json.loads((run/'record.json').read_text());r['case']='B5'
            r['geometry']=[640,480,243]
            with self.assertRaisesRegex(ValueError,'geometry mismatch'):validate_tensors(run,r)
            r['geometry']=[32,32,243]
            path=run/'result.h3av';data=bytearray(path.read_bytes());struct.pack_into('<f',data,160,math.inf);path.write_bytes(data)
            with self.assertRaisesRegex(ValueError,'nonfinite final'):validate_tensors(run,r)

    def test_nonfinite_cannot_pass_error_screen(self):
        with self.assertRaisesRegex(ValueError,'nonfinite'):metric([1.,math.nan],[1.,1.])

if __name__=='__main__':unittest.main()
