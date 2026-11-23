import unittest
import tempfile
import struct
from pathlib import Path
from large_canvas_guard import plan, select_frames, device_admission, GIB
from large_canvas_campaign import state_shape, script_plan


class GuardTests(unittest.TestCase):
    def test_state_and_script_guards(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'state.h3av'
            header = bytearray(160)
            header[:8] = b'H3AV\r\n\x1a\n'
            for w,h,f,kind,accepted in ((3840,2176,22,'upscale',True),
                    (1920,1088,56,'continuation',False),(1344,768,56,'continuation',True),
                    (3840,2176,39,'upscale',True),(3840,2176,56,'upscale',False),
                    (1920,1088,124,'native',False)):
                struct.pack_into('<3I',header,24,w,h,f)
                path.write_bytes(header)
                if accepted:
                    state_shape(path,kind)
                else:
                    with self.assertRaises(ValueError):state_shape(path,kind)
        segment = dict(command=['--width','1920','--height','1088','--frames','22'],delivered=22)
        self.assertEqual(script_plan(dict(segments=[segment]*2))[1],44)
        with self.assertRaises(ValueError):script_plan(dict(segments=[segment]*4))

    def test_maximum_short_refinement(self):
        for f, expected in ((22, 27), (39, 42)):
            p = plan('upscale', 3840, 2176, f)
            self.assertLessEqual(p['capacity_bytes'], expected * GIB)
            self.assertGreater(p['capacity_bytes'], (expected-1) * GIB)
            self.assertEqual(p['capacity_bytes'], plan('upscale', 2176, 3840, f)['capacity_bytes'])

    def test_forbidden_duration_and_continuation(self):
        for f in (56, 72, 73, 90, 124, 243, 362):
            for kind in ('native', 'upscale'):
                with self.assertRaises(ValueError):
                    plan(kind, 1344, 768, f)
        plan('continuation', 1344, 768, 56)
        with self.assertRaises(ValueError):
            plan('continuation', 1920, 1088, 56)

    def test_actual_rows_and_overflow(self):
        self.assertEqual(select_frames('upscale',3840,2176)['frames'],39)
        self.assertEqual(select_frames('upscale',3840,2176,70000)['frames'],22)
        with self.assertRaises(ValueError):
            plan('upscale', 3840, 2176, 22, rows=1)
        with self.assertRaises(ValueError):
            plan('upscale', 3840, 2176, 22, rows=1 << 63)
        with self.assertRaises(ValueError):
            plan('native', 3840, 2176, 22)
        with self.assertRaises(ValueError):
            plan('upscale', 3840, 2208, 22)
        self.assertLess(plan('upscale', 3840, 2176, 39, refine=0)['capacity_bytes'], 8 * GIB)

    def test_physical_reserve_and_current_free(self):
        self.assertEqual(plan('native',128,64,22)['live_bytes'],7*GIB)
        p=plan('upscale',3840,2176,39)
        self.assertGreater(device_admission(p,73415,72787),7*GIB)
        with self.assertRaises(ValueError):device_admission(p,73415,10000)
        with self.assertRaises(ValueError):
            device_admission(dict(live_bytes=54*GIB,capacity_bytes=60*GIB),73415,72787)


if __name__ == '__main__':
    unittest.main()
