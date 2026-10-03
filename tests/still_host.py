#!/usr/bin/env python3
"""Run host tests with a sparse metadata fixture; no model download or GPU.

Header provenance: Mamad8/MiniMax-H3-Image-VAE at
c7b9252c73707dba494cf4d99ca45d3f33f561b3. No weight payload is included.
"""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HEADER = Path(__file__).parent / 'fixtures/still-vae-header.json'

def sparse_checkpoint(path, header, truncate=False):
    raw = json.dumps(header, separators=(',', ':')).encode()
    payload = max(t['data_offsets'][1] for k, t in header.items() if k != '__metadata__')
    with Path(path).open('wb') as f:
        f.write(struct.pack('<Q', len(raw)))
        f.write(raw)
        f.truncate(8 + len(raw) + payload - (2 if truncate else 0))

if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='h3-still-host-') as directory:
        path = Path(directory) / 'metadata.safetensors'
        sparse_checkpoint(path, json.loads(HEADER.read_text()))
        subprocess.run([sys.argv[1], 'host', str(path)], check=True)
