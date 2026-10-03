#!/usr/bin/env python3
"""Small deterministic robustness tests; Metal tests require a GPU-capable shell."""
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROBE = Path(os.environ.get('H3_BUGFIX1_PROBE', ROOT / 'bin/bugfix1_probe')).resolve()


def fixture(path, header, payload=b'', offset=None):
    raw = header if isinstance(header, bytes) else json.dumps(header, separators=(',', ':')).encode()
    if offset is not None:
        assert offset >= len(raw) + 8
        raw += b' ' * (offset - len(raw) - 8)
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + payload)


class Robustness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='h3 bugfix1 ')
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name).resolve()
        self.cwd = self.dir / 'cwd'
        self.cwd.mkdir()
        self.env = {k: v for k, v in os.environ.items() if not k.startswith('H3_')}

    def run_probe(self, *args, ok=True, binary=PROBE, env=None, cwd=None):
        r = subprocess.run([str(binary), *map(str, args)], cwd=cwd or self.cwd,
                           env=env or self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode == 0, ok, r.stderr)
        return r

    def test_shader_precedence_and_errors(self):
        bindir = self.dir / 'binary with spaces'
        bindir.mkdir()
        binary = bindir / 'probe'
        shutil.copy2(PROBE, binary)
        adjacent = bindir / 'src/metal/shaders.metal'
        adjacent.parent.mkdir(parents=True)
        adjacent.write_text('adjacent')
        local = self.cwd / 'src/metal/shaders.metal'
        local.parent.mkdir(parents=True)
        local.write_text('local')
        self.assertEqual(self.run_probe('shader', binary=binary).stdout.strip(), str(local))
        self.assertEqual(self.run_probe('shader', adjacent, binary=binary).stdout.strip(), str(adjacent))
        preserved = str(bindir / '..' / bindir.name / 'src/metal/shaders.metal')
        self.assertEqual(self.run_probe('shader', preserved, binary=binary).stdout.strip(), preserved)
        self.assertEqual(self.run_probe('shader', env=dict(self.env, H3_SHADER_PATH='src/metal/shaders.metal'), binary=binary).stdout.strip(), 'src/metal/shaders.metal')
        override = self.dir / 'override.metal'
        override.write_text('override')
        env = dict(self.env, H3_SHADER_PATH=str(override))
        self.assertEqual(self.run_probe('shader', adjacent, env=env, binary=binary).stdout.strip(), str(override))
        for value in ['', 'missing.metal', str(self.dir)]:
            r = self.run_probe('shader', adjacent, env=dict(self.env, H3_SHADER_PATH=value), binary=binary, ok=False)
            self.assertIn('H3_SHADER_PATH=', r.stderr)
            self.assertIn('explicit override', r.stderr)
        local.unlink()
        self.assertEqual(self.run_probe('shader', binary=binary).stdout.strip(), str(adjacent))
        relative = os.path.relpath(binary, self.cwd)
        self.assertEqual(self.run_probe('shader', binary=relative).stdout.strip(), str(adjacent))
        link = self.cwd / 'linked probe'
        link.symlink_to(binary)
        self.assertEqual(self.run_probe('shader', binary=link).stdout.strip(), str(adjacent))
        adjacent.unlink()
        r = self.run_probe('shader', binary=binary, ok=False)
        self.assertIn(str(local), r.stderr)
        self.assertIn(str(adjacent), r.stderr)
        self.assertIn('H3_SHADER_PATH', r.stderr)
        self.assertIn('missing-absolute.metal', self.run_probe('shader', self.dir / 'missing-absolute.metal', ok=False).stderr)

    def test_long_build_path(self):
        directory = self.dir
        for i in range(7):
            directory /= f'{i} space ' + 'x' * 90
        directory.mkdir(parents=True)
        binary = directory / 'probe'
        shutil.copy2(PROBE, binary)
        shader = directory / 'src/metal/shaders.metal'
        shader.parent.mkdir(parents=True)
        shader.write_text('shader')
        self.assertGreater(len(str(binary)), 700)
        self.assertEqual(self.run_probe('shader', binary=binary).stdout.strip(), str(shader))

    def test_bin_installation_shader_lookup(self):
        install = self.dir / 'installation with spaces'
        binary = install / 'bin/probe'
        binary.parent.mkdir(parents=True)
        shutil.copy2(PROBE, binary)
        shader = install / 'src/metal/shaders.metal'
        shader.parent.mkdir(parents=True)
        shader.write_text('installed')
        self.assertEqual(self.run_probe('shader', binary=binary).stdout.strip(), str(shader))
        link = self.cwd / 'linked probe'
        link.symlink_to(binary)
        self.assertEqual(self.run_probe('shader', binary=link).stdout.strip(), str(shader))
        adjacent = binary.parent / 'src/metal/shaders.metal'
        adjacent.parent.mkdir(parents=True)
        adjacent.write_text('adjacent')
        self.assertEqual(self.run_probe('shader', binary=binary).stdout.strip(), str(adjacent))
        adjacent.unlink()
        shader.unlink()
        self.assertIn(str(shader), self.run_probe('shader', binary=binary, ok=False).stderr)

    def test_format_validation(self):
        x = {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}
        cases = [
            (b'', b''), (b'{', b''), (b' {}', b''),
            (b'{"__metadata__":{"k":"\xff"}}', b''),
            (b'{"__metadata__":{"k":"\\ud800"}}', b''),
            ({'x': dict(x, data_offsets=[0, 5])}, b'1234'),
            ({'x': dict(x, data_offsets=[4, 0])}, b'1234'),
            ({'x': dict(x, data_offsets=[2**64 - 1, 2**64])}, b'1234'),
            ({'x': dict(x, data_offsets=[2**63, 2**63 + 4])}, b'1234'),
            ({'x': dict(x, shape=[2**63, 2])}, b'1234'),
            ({'x': dict(x, shape=[2])}, b'1234'),
            ({'x': dict(x, dtype='INVALID')}, b'1234'),
            ({'x': x, 'y': x}, b'1234'),
            ({'x': dict(x, data_offsets=[4, 8])}, b'12345678'),
            ({'x': x}, b'12345678'),
            ({'__metadata__': 7}, b''),
            ({'__metadata__': {'k': 7}}, b''),
            ({'__metadata__': {'k': {'nested': 'no'}}}, b''),
            (b'{"__metadata__":{"k":"a","k":"b"}}', b''),
            (b'{"__metadata__":{},"__metadata__":{}}', b''),
            (b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4],}}', b'1234'),
            (b'{"x":{"dtype":"F32","dtype":"F32","shape":[1],"data_offsets":[0,4]}}', b'1234'),
            (b'{"x":{"dtype":"INVALID","dtype":"F32","shape":[1],"data_offsets":[0,4]}}', b'1234'),
            (b'{"x":{"dtype":"F32","shape":[01],"data_offsets":[0,4]}}', b'1234'),
            (b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}', b'12345678'),
        ]
        path = self.dir / 'bad.safetensors'
        for header, payload in cases:
            with self.subTest(header=header):
                fixture(path, header, payload)
                self.run_probe('header', path, ok=False)
        for data in [b'', b'1234567', struct.pack('<Q', 2**64-1), struct.pack('<Q', 100) + b'{}']:
            path.write_bytes(data)
            self.run_probe('header', path, ok=False)
        good = {'__metadata__': {'format': 'pt', 'unicode': '\U0001f600 café'}, 'scalar': dict(x, shape=[]),
                'empty': {'dtype': 'BF16', 'shape': [0, 2], 'data_offsets': [4, 4]}}
        for offset in [None, 1024, 1025]:
            fixture(path, good, b'1234', offset)
            self.assertEqual(self.run_probe('header', path).stdout.strip(), '2')

    @unittest.skipIf(os.environ.get('H3_BUGFIX1_CPU_ONLY'), 'CPU-only invocation')
    def test_mapped_copied_bytes_and_ownership(self):
        payload = struct.pack('<4f4H', 1, -2, 3.25, 0, 0x3f80, 0xc000, 0x4050, 0)
        header = {'x': {'dtype': 'F32', 'shape': [4], 'data_offsets': [0, 16]},
                  'y': {'dtype': 'BF16', 'shape': [4], 'data_offsets': [16, 24]}}
        aligned, unaligned = self.dir / 'aligned.safetensors', self.dir / 'unaligned.safetensors'
        fixture(aligned, header, payload, os.sysconf('SC_PAGESIZE'))
        fixture(unaligned, header, payload, os.sysconf('SC_PAGESIZE') + 1)
        env = dict(self.env, H3_ZERO_COPY_WEIGHTS='1', H3_SHADER_PATH=str(ROOT / 'src/metal/shaders.metal'), H3_PROFILE='1')
        for path, expected in [(aligned, 1), (unaligned, 0)]:
            out = self.dir / 'tensor.raw'
            r = self.run_probe('load', path, 'x', expected, out, env=env)
            self.assertEqual(out.read_bytes(), payload[:16])
            if not expected:
                self.assertIn('per-tensor copied fallback', r.stderr)
            self.run_probe('load', path, 'y', 0, out, env=env)
            self.assertEqual(out.read_bytes(), payload[16:])
        r = self.run_probe('storage-check', aligned, unaligned, env=env)
        self.assertEqual(r.stderr.count(f'released {os.sysconf("SC_PAGESIZE")}-byte weight mapping'), 32)


if __name__ == '__main__':
    unittest.main()
