#!/usr/bin/env python3
"""Adversarial clean-source containers, including valid checksums on bad data."""
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from test_sampler_file import entries, build

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get('H3_SAMPLER_TEST_BINARY', ROOT/'bin/sampler_tests'))


def container(parts):
    data = build(parts)
    data[:8] = b'H3UPSRC\1'
    data[64:96] = bytes(32)
    data[64:96] = hashlib.sha256(data).digest()
    return data


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name)/'source.h3up'
        subprocess.run([str(BINARY), '--upscale-fixture', str(cls.path)], cwd=ROOT,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cls.original = cls.path.read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, data, accepted=False):
        self.path.write_bytes(data)
        p = subprocess.run([str(BINARY), '--load-upscale', str(self.path)], capture_output=True)
        self.assertEqual(p.returncode == 0, accepted, p.stderr.decode())

    def test_roundtrip(self):
        self.check(self.original, True)

    def test_required_sections(self):
        for id in (1, 3, 5, 11, 12, 13, 45):
            with self.subTest(id=id):
                self.check(container([p for p in entries(self.original) if p[0] != id]))

    def test_unknown_and_versions(self):
        self.check(container(entries(self.original)+[[250, 1, 0, 1, 0, bytearray()]]), True)
        self.check(container(entries(self.original)+[[250, 1, 1, 1, 0, bytearray()]]))
        for field, value in ((1, 2), (2, 0)):
            parts = entries(self.original)
            next(p for p in parts if p[0] == 45)[field] = value
            self.check(container(parts))

    def test_stage_and_audio(self):
        for offset in (0, 4, 8, 12, 16, 20, 24, 228):
            parts = entries(self.original)
            payload = next(p for p in parts if p[0] == 45)[-1]
            struct.pack_into('<I', payload, offset, 99)
            with self.subTest(offset=offset): self.check(container(parts))
        parts = entries(self.original)
        next(p for p in parts if p[0] == 13)[-1][0] ^= 128
        self.check(container(parts))

    def test_nonfinite(self):
        for id in (5, 12, 13):
            parts = entries(self.original)
            payload = next(p for p in parts if p[0] == id)[-1]
            payload[:2 if id == 5 else 4] = struct.pack('<H', 0x7fc0) if id == 5 else struct.pack('<I', 0x7fc00000)
            self.check(container(parts))

    def test_no_history(self):
        for id in (21, 22, 30):
            self.check(container(entries(self.original)+[[id, 1, 0, 2 if id != 30 else 1, 0, bytearray()]]))

    def test_truncation(self):
        for n in (0, 8, 96, len(self.original)-1): self.check(self.original[:n])


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name)/'refine.h3sample'
        subprocess.run([str(BINARY), '--refinement-fixture', str(cls.path)], cwd=ROOT,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cls.original = cls.path.read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, parts, accepted=False):
        self.path.write_bytes(build(parts))
        p = subprocess.run([str(BINARY), '--load', str(self.path)], capture_output=True)
        self.assertEqual(p.returncode == 0, accepted, p.stderr.decode())

    def test_roundtrip(self):
        self.check(entries(self.original), True)

    def test_required_stage_and_noise(self):
        for id in (21, 44, 45):
            self.check([p for p in entries(self.original) if p[0] != id])
        for field, value in ((1, 1), (1, 3), (2, 0)):
            parts = entries(self.original)
            next(p for p in parts if p[0] == 45)[field] = value
            self.check(parts)

    def test_immutable_payloads(self):
        for id in (5, 13, 21):
            parts = entries(self.original)
            next(p for p in parts if p[0] == id)[-1][0] ^= 1
            self.check(parts)
        for offset in (0, 4, 260, 264, 268, 272, 344, 376):
            parts = entries(self.original)
            payload = next(p for p in parts if p[0] == 45)[-1]
            self.assertEqual(len(payload), 408)
            payload[offset] ^= 1
            self.check(parts)


if __name__ == '__main__':
    unittest.main()
