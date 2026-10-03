#!/usr/bin/env python3
"""Portable campaign configuration must retain per-run GPU identity checks."""
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cuda_sol_campaign as campaign


class CampaignConfiguration(unittest.TestCase):
    def test_model_path_with_spaces_is_one_argument(self):
        try:
            with patch.dict(os.environ, {'H3_MODEL_DIR': '/tmp/model directory',
                                        'H3_TEST_REFERENCE_ROOT': '/tmp/reference directory',
                                        'H3_TEST_QUANT_CACHE': '/tmp/quant directory'}):
                importlib.reload(campaign)
                case = dict(steps=2, prompt='test', width=128, height=128,
                            frames=22, seed=42, image_size='match')
                args = campaign.command(case, 'default', Path('outputs/test'), .75, {})
                self.assertEqual(args[args.index('-d')+1], str(Path('/tmp/model directory').resolve()))
                self.assertEqual(campaign.REFERENCE_ROOT, Path('/tmp/reference directory').resolve())
                self.assertEqual(campaign.QUANT_CACHE, Path('/tmp/quant directory').resolve())
        finally:
            importlib.reload(campaign)

    def test_gpu_identity_stays_bound_to_local_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = dict(binary='hash', native='hash', dense_native='hash',
                          source={}, acceptance='hash', manifest='hash', assets={},
                          model_path=str(campaign.MODEL), gpu_uuid='GPU-0000')
            (root/'identity.json').write_text(json.dumps(record))
            with patch.object(campaign, 'ROOT', root), patch.object(campaign, 'sha', return_value='hash'):
                with patch.object(campaign, 'gpu_uuid', return_value='GPU-0000'):
                    self.assertEqual(campaign.verify(), record)
                with patch.object(campaign, 'gpu_uuid', return_value='GPU-0001'):
                    with self.assertRaisesRegex(AssertionError, 'GPU changed'):
                        campaign.verify()
                with patch.object(campaign, 'gpu_uuid', return_value='GPU-0000'), \
                     patch.object(campaign, 'MODEL', root/'other-model'):
                    with self.assertRaisesRegex(AssertionError, 'Model location changed'):
                        campaign.verify()

    def test_missing_private_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = dict(binary='hash', native='hash', dense_native='hash', source={},
                          acceptance='hash', manifest='hash', assets={}, model_path=str(campaign.MODEL))
            (root/'identity.json').write_text(json.dumps(record))
            with patch.object(campaign, 'ROOT', root), patch.object(campaign, 'sha', return_value='hash'), \
                 patch.object(campaign, 'gpu_uuid', return_value='GPU-0000'):
                with self.assertRaisesRegex(AssertionError, 'GPU changed'):
                    campaign.verify()

    def test_device_query_rejects_multiple_or_invalid_results(self):
        for value in ('', 'GPU-0000\nGPU-0001', 'unknown'):
            with self.subTest(value=value), patch.object(campaign.subprocess, 'check_output', return_value=value):
                with self.assertRaisesRegex(AssertionError, 'identify qualification GPU'):
                    campaign.gpu_uuid()


if __name__ == '__main__':
    unittest.main()
