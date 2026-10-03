import contextlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fold_lora as f

QKV = 'blocks.0.attn.qkv_proj.weight'
OUT = 'blocks.0.attn.out_proj.weight'


def reference_bf16(value):
    # Independent scalar round-to-nearest, ties-to-even implementation.
    result = []
    for x in np.asarray(value, dtype=np.float32).ravel():
        bits, = struct.unpack('<I', struct.pack('<f', x))
        upper, lower = bits >> 16, bits & 65535
        result.append(upper + (lower > 32768 or (lower == 32768 and upper % 2)))
    return np.array(result, dtype='<u2').reshape(np.shape(value))


def save(path, tensors, metadata=None, alignment=64):
    header, parts, offset = {}, [], 0
    if metadata is not None:
        header['__metadata__'] = metadata
    for name, value in tensors.items():
        if isinstance(value, tuple):
            dtype, array = value
        else:
            dtype, array = 'F32', np.asarray(value, dtype='<f4')
        raw = array.tobytes()
        header[name] = dict(dtype=dtype, shape=list(array.shape), data_offsets=[offset, offset+len(raw)])
        offset += len(raw)
        parts.append(raw)
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-(len(raw)+8) % alignment)
    Path(path).write_bytes(struct.pack('<Q', len(raw))+raw+b''.join(parts))


def bf(value):
    return 'BF16', reference_bf16(value)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.base = self.root/'base'
        self.base.mkdir()
        rng = np.random.default_rng(31)
        self.w = rng.normal(size=(6, 4)).astype('f')/4
        self.w2 = rng.normal(size=(4, 2)).astype('f')/4
        save(self.base/'model.safetensors', {QKV: bf(self.w), OUT: bf(self.w2), 'norm.weight': bf(np.ones(4))})
        self.cp = f.Checkpoint(self.base)
        self.w = self.cp.tensors[QKV].read()
        self.w2 = self.cp.tensors[OUT].read()
        self.a = rng.normal(size=(2, 4)).astype('f')/8
        self.b = rng.normal(size=(6, 2)).astype('f')/8
        self.adapter = self.root/'adapter.safetensors'
        self.out = self.root/'folded'
        self.original = {p.name:p.read_bytes() for p in self.base.iterdir()}

    def adapter_file(self, prefix='blocks.0.attn.qkv_proj', metadata=None, extras=None, path=None):
        tensors = {prefix+'.lora_A.weight': self.a, prefix+'.lora_B.weight': self.b}
        tensors.update(extras or {})
        save(path or self.adapter, tensors, metadata)
        return path or self.adapter

    def run_fold(self, adapters=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return f.fold(self.base, adapters or [(self.adapter, 1.0)], self.out,
                          allow_full_copy=True, **kwargs)

    def unchanged(self):
        self.assertEqual(self.original, {p.name:p.read_bytes() for p in self.base.iterdir()})
