#!/usr/bin/env python3
"""Sanity checks that diagnostics distinguish exact/static and changing frames."""
import unittest
import numpy as np
from skimage.metrics import structural_similarity
from upscale_report import detail, gray, resize


class Diagnostics(unittest.TestCase):
    def test_constant_and_edge(self):
        a=np.full((32,64,3),64,np.uint8)
        self.assertEqual(detail(a),0.)
        self.assertEqual(structural_similarity(gray(a),gray(a),data_range=1.),1.)
        self.assertTrue(np.all(resize(a,(16,8))==64))
        b=a.copy();b[:,32:]=192
        self.assertGreater(detail(b),0.)
        self.assertLess(structural_similarity(gray(a),gray(b),data_range=1.),1.)

    def test_range_and_channels(self):
        x=np.array([[[255,0,0],[0,255,0],[0,0,255],[255,255,255]]],np.uint8)
        np.testing.assert_allclose(gray(x),[[.2126,.7152,.0722,1.]],rtol=1e-6)


if __name__=='__main__':unittest.main()
