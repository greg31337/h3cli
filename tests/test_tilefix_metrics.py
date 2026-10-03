#!/usr/bin/env python3
"""Sanity-check diagnostic metrics with analytically controlled images."""
import unittest
import numpy as np
from tilefix_metrics import compare,seams
class Metrics(unittest.TestCase):
    def test_identity(self):
        a=np.random.default_rng(1).random((2,32,48,3),dtype=np.float32)
        r=compare(a,a);self.assertTrue(r['exact']);self.assertAlmostEqual(r['ssim_luma_11x11'],1)
    def test_constant_offset(self):
        a=np.full((2,32,48,3),.25);b=a+.125;r=compare(a,b)
        self.assertAlmostEqual(r['mae'],.125);self.assertAlmostEqual(r['rmse'],.125)
        self.assertAlmostEqual(r['psnr_db'],18.06179973983887)
        self.assertAlmostEqual(r['ssim_luma_11x11'],(2*.25*.375+.0001)/(.25**2+.375**2+.0001))
    def test_seam_and_rotation(self):
        # A true discontinuity at the overlap edge exceeds neighboring slope.
        a=np.broadcast_to(np.linspace(0,.1,64)[None,None,:,None],(2,64,64,3)).copy();a[:,:,32:]+=.2
        plan={'x':{'length':48,'starts':[0,32]},'y':{'length':64,'starts':[0]}}
        r=seams(a,plan);self.assertGreater(r['x']['boundaries'][0]['gradient_ratio'],100)
        self.assertIsNone(r['y']['mean_gradient_ratio'])
        rotated=seams(a.transpose(0,2,1,3),{'x':plan['y'],'y':plan['x']})
        self.assertAlmostEqual(r['x']['mean_gradient_ratio'],rotated['y']['mean_gradient_ratio'],places=3)
        self.assertAlmostEqual(r['x']['mean_luminance_ratio'],rotated['y']['mean_luminance_ratio'],places=3)
if __name__=='__main__':unittest.main()
