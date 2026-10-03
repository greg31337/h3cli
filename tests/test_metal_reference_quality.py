"""CPU metric regressions; use requirements-metal-quality.txt."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from metal_reference_quality import audio_metrics, np


class AudioMetrics(unittest.TestCase):
    def signal(self):
        t=np.arange(64000)/32000
        x=.1*np.sin(2*np.pi*440*t)*(.55+.4*np.sin(2*np.pi*3*t))
        return np.stack((x,-x),axis=1)
    def test_exact_stereo_repeat(self):
        a=self.signal();r=audio_metrics(a,a)
        self.assertEqual(r['log_spectrum_mean_abs'],0)
        self.assertEqual(r['envelope_relative_l2'],0)
        self.assertAlmostEqual(r['envelope_correlation'],1)
    def test_stereo_does_not_cancel(self):
        a=self.signal();r=audio_metrics(a,np.zeros_like(a))
        self.assertGreater(r['log_spectrum_mean_abs'],0)
        self.assertEqual(r['envelope_relative_l2'],1)
        self.assertGreater(r['new_dropout_windows'],0)
    def test_length_difference_is_retained(self):
        a=self.signal();self.assertEqual(audio_metrics(a,a[:-128])['sample_count_difference'],128)
    def test_invalid_audio(self):
        a=self.signal();a[100,1]=np.nan
        with self.assertRaises(ValueError):audio_metrics(a,a)
        with self.assertRaises(ValueError):audio_metrics(np.zeros((100,2)),np.zeros((100,2)))
    def test_shift_changes_envelope(self):
        a=self.signal();b=np.concatenate([np.zeros_like(a[:8000]),a[:-8000]])
        r=audio_metrics(a,b)
        self.assertGreater(r['envelope_relative_l2'],.15)
        self.assertLess(r['envelope_correlation'],.98)


if __name__=='__main__':unittest.main()
