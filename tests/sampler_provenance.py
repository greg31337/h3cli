#!/usr/bin/env python3
"""Check resume provenance from a --verbose log against its input file."""
import argparse
import hashlib
from pathlib import Path
import re
import struct


def verify(checkpoint, log):
    with checkpoint.open('rb') as f:
        header = f.read(96)
        assert header[:8] == b'H3SAMPLE'
        count = struct.unpack_from('<I', header, 20)[0]
        table = f.read(72*count)
        parts = {struct.unpack_from('<I', table, i*72)[0]:
                 struct.unpack_from('<Q', table, i*72+24)[0] for i in range(count)}
        f.seek(parts[1]+8); step = struct.unpack('<I', f.read(4))[0]
        operations = 0
        if 31 in parts:
            f.seek(parts[31]); operations = struct.unpack('<I', f.read(4))[0]
        f.seek(0); digest = hashlib.file_digest(f, 'sha256').hexdigest()
    matches = re.findall(r'resume provenance: format=(\d+) step=(\d+) operations=(\d+) checkpoint_sha256=([0-9a-f]{64})', log.read_text())
    assert matches and matches[-1] == ('1', str(step), str(operations+1), digest), (checkpoint, matches)
    print(f'ok: resume provenance {checkpoint.name}: step={step}, operations={operations+1}, complete-file SHA-256 exact')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('log', type=Path)
    args = parser.parse_args()
    verify(args.checkpoint, args.log)
