import unittest
from long_video_guard import plan, device_admission, GIB
from long_video_campaign import script_plan

class GuardTests(unittest.TestCase):
    def test_limits(self):
        for f in (22,39,362,447,464):
            self.assertEqual(plan('native',256,256,f)['frames'],f)
        for f in (0,5,363,480,481,719,736,737,2**63):
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

if __name__=='__main__':unittest.main()
