#!/usr/bin/env python3
"""Independent binary-mask oracle, checked against ComfyUI on 2026-09-13:
https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py
(forward, per-row timesteps and masked output velocities).

Runs without extra packages; if torch is installed, also evaluates its F32
tensor equations. Compare actual C schedule rows AND packed masked velocities.
"""
import json
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]

def main():
    data = json.loads(subprocess.check_output([str(ROOT / 'bin/continuation_tests'), '--oracle']))
    try:
        import torch
    except ImportError:
        torch = None
    checked = 0
    for step in data:
        for audio, row, kind, time, velocity in step['rows']:
            preserved = row % 93 < 65 if audio else row // 4 < 12
            mask = 0 if preserved else 1
            condition_strength = 1.0 if audio else f32(0.999)
            sigma = f32(step['sigma_a'] if audio else step['sigma_v'])
            # Native continuous-mask equation, specialized to binary masks.
            cap = max(f32(1.0-sigma), condition_strength)
            expected = min(f32(1.0-f32(mask*sigma)), cap)
            assert kind == (1 if audio else 0) + (2 if preserved else 0)
            assert f32(time) == expected, (audio, row, time, expected)
            assert velocity == mask * (3.0 if audio else 2.0)
            if torch is not None:
                ts = (1-torch.tensor(mask)*torch.tensor(sigma)).clamp(max=cap)
                assert ts.item() == expected
            checked += 1
    print(f'ok: {checked} native binary-mask oracle rows; '
          f'backend={"Python + PyTorch" if torch is not None else "Python IEEE F32"}')

if __name__ == '__main__':
    main()
