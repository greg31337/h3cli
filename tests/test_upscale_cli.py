#!/usr/bin/env python3
"""Source mode errors must be reported before expensive model loading."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
binary = Path(os.environ.get('H3_TEST_BINARY', root/'bin/h3cli'))
with tempfile.TemporaryDirectory() as d:
    source = str(Path(d)/'source.h3up')
    common = [str(binary), '-d', '/missing-model', '-p', 'test']
    cases = [(['--state-only'], 'state-only'),
             (['--save-upscale-state', source, '--reuse', '2'], 'dense BF16'),
             (['--save-upscale-state', source, '--stop-after-step', '0'], 'dense BF16'),
             (['--save-upscale-state', source, '-o', source], 'aliases'),
             (['--save-upscale-state', source, '--state-only', '--show'], 'does not deliver'),
             (['--save-upscale-state', source, '--cuda-denoise-quant', 'fp8'], 'dense BF16'),
             (['--save-upscale-state', source, '--adaptive-cache', 'conservative'], 'dense BF16'),
             (['--save-upscale-state', ''], 'filename'),
             (['--save-upscale-state', source, '--first', source], 'aliases')]
    for args, message in cases:
        p = subprocess.run(common+args, capture_output=True, text=True)
        assert p.returncode == 2 and message in p.stderr, (args, p.returncode, p.stderr)
        assert 'cannot open model' not in p.stderr
    fresh = [str(binary), '-d', '/missing-model', '--upscale-state', source,
             '--upscale-model', str(Path(d)/'weights.safetensors')]
    fresh_cases = [(['--upscale-refine-steps', '1'], 'steps must'),
                   (['--upscale-refine-steps', '0', '--upscale-noise', '.25'], 'explicit sigma'),
                   (['--upscale-noise', '0'], 'sigma'),
                   (['--upscale-noise', '.6'], 'sigma'),
                   (['--width', '256'], 'conflict'),
                   (['--reuse', '2'], 'conflict'),
                   (['--cuda-denoise-quant', 'fp8'], 'conflict'),
                   (['--adaptive-cache', 'conservative'], 'conflict'),
                   (['--resume-sampler-state', source], 'conflict'),
                   (['--state-only'], 'needs --save-av-state'),
                   (['-o', source], 'alias'),
                   (['--stop-after-step', '1'], 'checkpoint'),
                   (['--stop-after-step', '5', '--save-sampler-state', str(Path(d)/'out')], 'checkpoint'),
                   (['--upscale-refine-steps', '0', '--save-sampler-state', str(Path(d)/'out')], 'checkpoint'),
                   (['--save-av-state', source], 'alias')]
    for args, message in fresh_cases:
        p = subprocess.run(fresh+args, capture_output=True, text=True)
        assert p.returncode == 2 and message in p.stderr, (args, p.returncode, p.stderr)
        assert 'cannot open model' not in p.stderr
print(f'ok: {len(cases)+len(fresh_cases)} upscale CLI preflight cases')
