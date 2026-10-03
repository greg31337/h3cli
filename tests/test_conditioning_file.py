#!/usr/bin/env python3
"""Exercise integrity AND semantically malformed, correctly rehashed containers."""
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

BINARY = os.environ.get('H3_CONDITIONING_TEST_BINARY', './bin/conditioning_tests')

def rehash(blob):
    count = struct.unpack_from('<I', blob, 24)[0]
    for index in range(count):
        entry = 128 + index * 96
        offset, size = struct.unpack_from('<QQ', blob, entry + 40)
        if offset + size <= len(blob):
            blob[entry + 56:entry + 88] = hashlib.sha256(blob[offset:offset + size]).digest()
    blob[32:64] = hashlib.sha256(blob[128:128 + count * 96]).digest()
    blob[64:96] = hashlib.sha256(blob[:64]).digest()

class Containers(unittest.TestCase):
    def test_container_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.h3cond'
            subprocess.run([BINARY, 'fixture', str(path)], check=True, capture_output=True)
            valid = path.read_bytes()
            cases = [valid[:n] for n in (0, 7, 64, 127, 128, len(valid) - 1)]
            cases += [valid + b'junk']
            for position in (0, 8, 16, 24, 32, 64, 96, 128, 156, len(valid) - 1):
                blob = bytearray(valid); blob[position] ^= 1; cases.append(blob)
            count = struct.unpack_from('<I', valid, 24)[0]
            entries = {struct.unpack_from('<I', valid, 128 + i * 96)[0]: 128 + i * 96 for i in range(count)}
            # Recomputed checksums must not hide unknown types, duplicate IDs,
            # overlaps, invalid shapes, invalid tags or non-finite tensors.
            for record, field, value in ((2, 0, 1), (2, 4, 99), (2, 8, 4),
                    (2, 12, 0), (2, 24, 0), (2, 24, 5121), (2, 32, 1),
                    (3, 16, 2), (6, 24, 4), (7, 16, 3), (1001, 0, 1199)):
                blob = bytearray(valid)
                struct.pack_into('<I' if field < 16 else '<Q', blob, entries[record] + field, value)
                rehash(blob); cases.append(blob)
            for record, byte_offset, data in ((2, 0, b'\x80\x7f'), (3, 0, b'\x02'),
                    (4, 0, struct.pack('<f', float('nan'))), (6, 0, struct.pack('<I', 99)),
                    (6, 8, struct.pack('<I', 3)), (7, 0, struct.pack('<I', 2)),
                    (7, 4, struct.pack('<I', 3)), (1001, 0, b'\xc0\x7f')):
                blob = bytearray(valid); offset = struct.unpack_from('<Q', blob, entries[record] + 40)[0]
                blob[offset + byte_offset:offset + byte_offset + len(data)] = data
                rehash(blob); cases.append(blob)
            for index, blob in enumerate(cases):
                path.write_bytes(blob)
                result = subprocess.run([BINARY, 'load', str(path)], capture_output=True)
                self.assertNotEqual(result.returncode, 0, f'malformed case {index}')
                self.assertIn(b'h3cond:', result.stderr, f'case {index} must fail cleanly')
            path.write_bytes(valid)
            subprocess.run([BINARY, 'load', str(path)], check=True)
            print(f'{len(cases)} malformed conditioning containers rejected')

if __name__ == '__main__':
    unittest.main()
