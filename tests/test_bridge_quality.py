"""Synthetic checks for quality diagnostics, independent of model behavior."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from bridge_quality_metrics import PROTOCOL, compare, extract, motion_measurements, sidecar, quality_gate
from run_bridge_quality import sweep
from bridge_completion import chain_visual_review, sha


def moving_patch(jump=0, freeze=False):
    rng=np.random.default_rng(123)
    background=rng.integers(20,100,(96,96,3),dtype=np.uint8)
    patch=rng.integers(130,245,(20,18,3),dtype=np.uint8)
    result=[]
    for frame in range(72):
        image=background.copy()
        x=32+int(.3*(min(frame,23) if freeze else frame))+(jump if frame>=24 else 0)
        image[43:63,x:x+18]=patch
        result.append(image)
    return np.asarray(result)


class QualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol=json.loads(PROTOCOL.read_text())
        cls.steady=motion_measurements(moving_patch(),cls.protocol)

    def test_velocity_discontinuity(self):
        jump=motion_measurements(moving_patch(jump=13),self.protocol)
        self.assertGreater(jump['continuity_score'],self.steady['continuity_score']*1.5)
        self.assertGreater(max(jump['curves']['acceleration'][22:25]),
                           max(self.steady['curves']['acceleration'][22:25])*2)

    def test_freezing_cannot_win(self):
        frozen=motion_measurements(moving_patch(freeze=True),self.protocol)
        result=compare(self.steady,frozen,self.protocol)
        self.assertFalse(result['checks']['motion_retained'])
        self.assertFalse(result['numerical_pass'])

    def test_stability_guards(self):
        identical=compare(self.steady,self.steady,self.protocol,True)
        self.assertTrue(identical['numerical_pass'])
        for key,value,check in [
            ('sharpness',self.steady['sharpness']*.5,'sharpness_retained'),
            ('background_color_delta',self.steady['background_color_delta']+10,'color_stable'),
            ('background_contrast',self.steady['background_contrast']*2,'contrast_stable'),
            ('camera_motion',self.steady['camera_motion']+10,'camera_stable'),
            ('background_deformation',self.steady['background_deformation']+10,'background_stable')]:
            result=compare(self.steady,dict(self.steady,**{key:value}),self.protocol,True)
            self.assertFalse(result['checks'][check],key)

    def test_brightness_measurement(self):
        rgb=moving_patch().astype(np.int16); rgb[24:]+=20
        changed=motion_measurements(np.clip(rgb,0,255).astype(np.uint8),self.protocol)
        self.assertGreater(changed['background_color_delta'],self.steady['background_color_delta']+10)
        self.assertGreater(changed['luminance'][24],self.steady['luminance'][24]+19)

    def test_extract_timeline_and_decimal_names(self):
        self.assertEqual(str(sidecar(Path('ease-out-8-0.50'),'json')),'ease-out-8-0.50.json')
        rgb=moving_patch()
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); extract(rgb,folder)
            self.assertEqual(len(list(folder.glob('[+-]*.png'))),72)
            for offset in [-24,-1,0,47]:
                self.assertTrue(np.array_equal(np.array(Image.open(folder/f'{offset:+04d}.png')),rgb[offset+24]))
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(folder/'window.mp4')]))
            self.assertEqual(probe['streams'][0]['nb_frames'],'72')
            self.assertEqual(probe['streams'][0]['avg_frame_rate'],'24/1')
            self.assertEqual(float(probe['streams'][0]['duration']),3)

    def test_sweep_coverage(self):
        full=sweep(True); pilot=sweep()
        self.assertEqual(len(full),60); self.assertEqual(len(pilot),17)
        for group in [full,pilot]:
            self.assertEqual({x['length'] for x in group},{4,6,8,9,10})
            self.assertEqual({x['strength'] for x in group},{.25,.4,.5,.65})
            self.assertEqual({x['profile'] for x in group},{'stepped','linear','ease-out'})
            self.assertEqual(len({(x['length'],x['strength'],x['profile']) for x in group}),len(group))

    def test_acceptance_requires_confirmation_and_current_visual_reviews(self):
        config=dict(mode='bridge',length=8,strength=.5,profile='stepped')
        report=dict(protocol=self.protocol,protocol_sha256='protocol',cases={},comparisons={})
        selection=dict(config=config,protocol_sha256='protocol'); reviews={}
        required=[('run',72),('run',73),('unchanged',72),('unchanged',73),('look-up',72),('arms',72),('turn-left',72)]
        for action,seed in required:
            name=f'{action}-{seed}-chosen'; hard=f'{action}-{seed}-hard'
            report['cases'][name]=dict(config=config,rgb_sha256=name)
            report['cases'][hard]=dict(config=dict(config,mode='hard'),rgb_sha256=hard)
            report['comparisons'][name]=dict(hard=hard,stability_pass=True,numerical_pass=True,continuity_ratio=.8)
            reviews[name]=dict(reviewer='synthetic test',method='synthetic test',notes='synthetic test',
                               bridge_rgb_sha256=name,hard_rgb_sha256=hard,
                               checks={k:'pass' for k in self.protocol['visual_review_required']})
        self.assertTrue(quality_gate(report,selection,reviews)['passed'])
        review=reviews.pop('run-73-chosen')
        self.assertFalse(quality_gate(report,selection,reviews)['passed'])
        reviews['run-73-chosen']=review
        review['bridge_rgb_sha256']='stale'
        self.assertFalse(quality_gate(report,selection,reviews)['passed'])
        review['bridge_rgb_sha256']='run-73-chosen'
        report['comparisons']['run-73-chosen']['numerical_pass']=False
        self.assertFalse(quality_gate(report,selection,reviews)['passed'])

    def test_chain_requires_current_visual_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); boundaries=[]; reviews={}
            for i in range(3): (root/f'segment{i}.rgb').write_bytes(bytes([i]))
            for i in range(1,3):
                source=root/f'segment{i-1}'; target=root/f'segment{i}'
                boundaries.append(dict(source=str(source),target=str(target)))
                reviews[target.name]=dict(reviewer='synthetic',method='synthetic',notes='synthetic',
                    source_rgb_sha256=sha(sidecar(source,'rgb')),target_rgb_sha256=sha(sidecar(target,'rgb')),
                    checks={k:'pass' for k in self.protocol['visual_review_required']})
            self.assertTrue(chain_visual_review(boundaries,reviews))
            reviews['segment2']['checks']['transition']='uncertain'
            with self.assertRaises(AssertionError): chain_visual_review(boundaries,reviews)
            reviews['segment2']['checks']['transition']='pass'
            sidecar(root/'segment2','rgb').write_bytes(b'changed')
            with self.assertRaises(AssertionError): chain_visual_review(boundaries,reviews)


if __name__=='__main__': unittest.main()
