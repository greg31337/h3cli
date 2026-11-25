import unittest
import struct
import tempfile
from pathlib import Path
from long_video_guard import plan, device_admission, inspect_checkpoint, GIB
from long_video_campaign import script_plan

class GuardTests(unittest.TestCase):
    def test_limits(self):
        for f in (22,39,362,447,464):
            self.assertEqual(plan('native',256,256,f)['frames'],f)
        for f in (0,5,363,480,481,719,736,737,1433,1440,1450,1451,2**63):
            with self.assertRaises(ValueError):plan('native',256,256,f)
        for size in ((128,128),(256,512),(1920,1088)):
            with self.assertRaises(ValueError):plan('native',*size,22)
        with self.assertRaises(ValueError):plan('upscale',256,256,22)
        with self.assertRaises(ValueError):plan('native',256,256,464,cuda=False)
        with self.assertRaises(ValueError):plan('native',256,256,464,steps=7)
        with self.assertRaises(ValueError):plan('native',256,256,464,reference_frames=361)
    def test_continuation_merge(self):
        self.assertEqual(plan('continuation',256,256,464,context=447,merged=464)['delivered'],17)
        for context in (448,464,702):
            with self.assertRaises(ValueError):plan('continuation',256,256,464,context=context)
        with self.assertRaises(ValueError):plan('script',256,256,464,merged=928)
    def test_worker_and_merge_plans(self):
        def row(name,frames):return dict(name=name,frames=frames,delivered=frames,command=['--width','256','--height','256','--frames',str(frames),'--steps','2'])
        data=dict(segments=[row('a',464)],merged=[dict(segments=['a'])])
        self.assertEqual(script_plan(data)['frames'],464)
        data['segments'].append(row('b',22));data['merged'][0]['segments'].append('b')
        with self.assertRaises(ValueError):script_plan(data)
        data=dict(segments=[row('a',481)],merged=[])
        with self.assertRaises(ValueError):script_plan(data)

    def test_reserve(self):
        p=plan('native',256,256,464)
        self.assertEqual(device_admission(p,73415,72000),73415*(1<<20)//10)
        with self.assertRaises(ValueError):device_admission(p,73415,1000)

    def test_precision_and_placement(self):
        p=plan('native',256,256,464,precision='nvfp4',placement='resident')
        self.assertEqual(p['precision'],'nvfp4')
        self.assertGreater(p['live_bytes'],plan('native',256,256,464,precision='nvfp4')['live_bytes'])
        device_admission(p,73415,72000)
        with self.assertRaises(ValueError):plan('native',256,256,22,precision='nvfp4',cuda=False)
        with self.assertRaises(ValueError):plan('native',256,256,464,precision='fp8')
        data=dict(segments=[dict(name='a',frames=464,delivered=464,command=['--width','256','--height','256','--frames','464','--steps','2'])],merged=[])
        with self.assertRaises(ValueError):script_plan(data,'nvfp4')
        data['segments'][0]['command']+=['--cuda-denoise-quant','nvfp4']
        self.assertEqual(script_plan(data,'nvfp4')['precision'],'nvfp4')

    def test_checkpoint_identity_without_movie_allocation(self):
        header=bytearray(96);header[:8]=b'H3SAMPLE';struct.pack_into('<I',header,20,1)
        entry=bytearray(72);struct.pack_into('<4I3Q',entry,0,1,1,1,1,172,168,172)
        identity=bytearray(172);struct.pack_into('<i',identity,4,2)
        struct.pack_into('<ii',identity,16,256,256)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'metadata.h3sample'
            for frames in (464,481,736,1450):
                struct.pack_into('<iii',identity,160,256,256,frames)
                path.write_bytes(header+entry+identity)
                if frames==464:self.assertEqual(inspect_checkpoint(path)['frames'],464)
                else:
                    with self.assertRaises(ValueError):inspect_checkpoint(path)
            struct.pack_into('<iii',identity,160,1920,1088,464)
            path.write_bytes(header+entry+identity)
            with self.assertRaises(ValueError):inspect_checkpoint(path)

    def test_sage3_admission_and_worker_identity(self):
        for precision in ('bf16','nvfp4'):
            dense=plan('native',256,256,464,precision=precision)
            sage=plan('native',256,256,464,precision=precision,attention='sage3')
            self.assertEqual(sage['live_bytes']-dense['live_bytes'],512*(1<<20))
            self.assertEqual(sage['attention_workspace_bytes'],512*(1<<20))
            device_admission(sage,73415,72000)
        with self.assertRaises(ValueError):plan('native',256,256,22,attention='sage3',cuda=False)
        with self.assertRaises(ValueError):plan('native',256,256,464,attention='sage2++')
        data=dict(segments=[dict(name='a',frames=464,delivered=464,command=['--width','256','--height','256','--frames','464','--steps','2'])],merged=[])
        with self.assertRaises(ValueError):script_plan(data,attention='sage3')
        data['segments'][0]['command']+=['--cuda-attention','sage3']
        self.assertEqual(script_plan(data,attention='sage3')['attention'],'sage3')

if __name__=='__main__':unittest.main()
