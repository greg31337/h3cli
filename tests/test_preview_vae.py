#!/usr/bin/env python3
"""Host-only tiny-model/CLI validation. No installed weights or Python packages needed."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROBE = Path(os.environ.get('H3_PREVIEW_TEST_BINARY', ROOT/'bin/preview_vae_tests')).resolve()

def pattern_weights(path):
    tensors = {}
    payload = bytearray()
    def conv(name, out, inp, kernel, bias=True):
        for suffix, shape in [('weight', [out, inp, kernel, kernel])] + ([('bias', [out])] if bias else []):
            elements = 1
            for n in shape: elements *= n
            begin = len(payload)
            if name == 'decoder.22' and suffix == 'bias':
                payload.extend(struct.pack('<12e', *(i / 16 for i in range(12))))
            else: payload.extend(bytes(elements * 2))
            tensors[f'{name}.{suffix}'] = dict(dtype='F16', shape=shape, data_offsets=[begin, len(payload)])
    conv('decoder.1', 256, 24, 3)
    for index, channel in [(i, 256 if i < 9 else 128 if i < 15 else 64) for i in [3,4,5,9,10,11,15,16,17]]:
        for layer in [0,2,4]: conv(f'decoder.{index}.conv.{layer}', channel, channel * (2 if layer == 0 else 1), 3)
    for name, out, inp, kernel in [('7.conv',256,256,1),('8',128,256,3),('13.conv',256,128,1),('14',64,128,3),('19.conv',128,64,1),('20',64,64,3)]:
        conv(f'decoder.{name}', out, inp, kernel, False)
    conv('decoder.22',12,64,3)
    header = json.dumps(tensors, separators=(',', ':')).encode()
    path.write_bytes(struct.pack('<Q', len(header)) + header + payload)
    return tensors, payload

def temporal_pattern_weights(path):
    """Sparse channel-zero network: current plus previous latent, duplicated in time.

    Only the first memory block has a nonzero residual branch. This supplies
    an independent temporal/layout oracle, including boundaries between batches.
    """
    tensors, payload = pattern_weights(path)
    def center(name, out, inp, value):
        tensor = tensors[name+'.weight']
        _, channels, kernel, _ = tensor['shape']
        index = (out * channels + inp) * kernel * kernel + (kernel // 2) * kernel + kernel // 2
        struct.pack_into('<e', payload, tensor['data_offsets'][0] + index * 2, value)
    center('decoder.1', 0, 0, 1)
    center('decoder.3.conv.0', 0, 256, 1)  # Previous input, rather than current input.
    center('decoder.3.conv.2', 0, 0, 1)
    center('decoder.3.conv.4', 0, 0, 1)
    for name in ['7.conv', '8', '13.conv', '14', '19.conv', '20']:
        center('decoder.'+name, 0, 0, 1)
    center('decoder.13.conv', 128, 0, 1)
    center('decoder.19.conv', 64, 0, 1)
    for out in range(12):
        center('decoder.22', out, 0, 0.125)
    header = json.dumps(tensors, separators=(',', ':')).encode()
    path.write_bytes(struct.pack('<Q', len(header)) + header + payload)

class PreviewVAE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='h3 tiny tests ')
        cls.directory = Path(cls.temp.name)
        cls.weights = cls.directory / 'pattern.safetensors'
        cls.header, cls.payload = pattern_weights(cls.weights)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def run_command(self, args, success=False):
        result = subprocess.run([str(x) for x in args], cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode == 0, success, result.stderr)
        return result.stdout + result.stderr

    def test_valid_architecture_without_encoder(self):
        out = self.run_command([PROBE, 'validate', self.weights], True)
        self.assertEqual(len(out.strip()), 64)

    def test_invalid_models(self):
        path = self.directory/'bad.safetensors'
        header = json.loads(json.dumps(self.header))
        cases = []
        h = dict(header); del h['decoder.1.bias']; cases.append((h, self.payload))
        h = json.loads(json.dumps(header)); h['decoder.1.weight']['shape'][0] = 255; cases.append((h,self.payload))
        h = json.loads(json.dumps(header)); h['decoder.1.weight']['dtype'] = 'BF16'; cases.append((h,self.payload))
        h = json.loads(json.dumps(header)); h['decoder.extra'] = h.pop('decoder.1.bias'); cases.append((h,self.payload))
        nonfinite = bytearray(self.payload); nonfinite[:2] = struct.pack('<H',0x7c00); cases.append((header, nonfinite))
        cases.append((header,self.payload[:-1]))
        for h, data in cases:
            raw = json.dumps(h,separators=(',',':')).encode()
            path.write_bytes(struct.pack('<Q',len(raw))+raw+data)
            self.run_command([PROBE,'validate',path])

    def test_early_cli_errors(self):
        common = [ROOT/'bin/h3cli','-d','missing-model']
        cases = [(['--preview-vae-model','missing'], 'requires'),
                 (['--offline','-p','test','--steps','2','--preview-vae','--preview-vae-model','missing'], 'missing'),
                 (['--decode-full-state'], 'option'),
                 (['--decode-av-state','missing','-p','hello'], 'decode-only'),
                 (['--decode-av-state','missing','--steps','2'], 'decode-only'),
                 (['--decode-av-state','missing','--save-av-state','out'], 'decode-only'),
                 (['--decode-av-state','missing','--stop-after-step','0'], 'decode-only'),
                 (['--decode-av-state','missing','--resume-sampler-state','missing'], 'decode-only')]
        for args, phrase in cases:
            with self.subTest(args=args):
                out = self.run_command(common+args)
                self.assertIn(phrase,out)
                for forbidden in ['text encoder','DiT initialization','Qwen vision']: self.assertNotIn(forbidden,out)

if __name__ == '__main__':
    if len(os.sys.argv)==3 and os.sys.argv[1]=='--write-pattern':
        pattern_weights(Path(os.sys.argv[2]))
    elif len(os.sys.argv)==3 and os.sys.argv[1]=='--write-temporal-pattern':
        temporal_pattern_weights(Path(os.sys.argv[2]))
    else: unittest.main()
