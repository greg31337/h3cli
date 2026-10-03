#!/usr/bin/env python3
"""Parser coverage without model loading or numerical rendering comparisons."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import subprocess
import unittest


class ReferenceImageCLI(unittest.TestCase):
    def test_modes(self):
        for mode in ('match', 'high', 'max'):
            with self.subTest(mode=mode):
                p = subprocess.run(['./bin/h3cli', '--ref-image-size', mode, '--help'], capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertIn('high (long edge 2048)', p.stdout + p.stderr)

    def test_invalid_modes(self):
        for mode in ('', 'HIGH', '2048', 'auto', 'low'):
            with self.subTest(mode=mode):
                p = subprocess.run(['./bin/h3cli', '--ref-image-size', mode, '--help'], capture_output=True, text=True)
                self.assertEqual(p.returncode, 2)
                self.assertIn('must be match, high or max', p.stderr)


if __name__ == '__main__':
    unittest.main()
