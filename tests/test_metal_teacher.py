"""Malformed teacher inputs must never partially replace a sampler state."""
import ctypes
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class Teacher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.directory=Path(cls.tmp.name)
        library=cls.directory/'teacher.so'
        subprocess.run(['cc','-shared','-fPIC','-I',str(ROOT),str(ROOT/'src/testing/test_teacher.c'),'-o',str(library)],check=True)
        cls.function=ctypes.CDLL(str(library)).h3_test_teacher_load
        cls.function.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.POINTER(ctypes.c_float),ctypes.c_size_t,
                              ctypes.POINTER(ctypes.c_float),ctypes.c_size_t,ctypes.c_char_p,ctypes.c_size_t]
        cls.function.restype=ctypes.c_int
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def check(self,audio,expected,step=1):
        (self.directory/'step-001-video-latent.f32').write_bytes(struct.pack('<2f',3,4))
        (self.directory/'step-001-audio-latent.f32').write_bytes(audio)
        video=(ctypes.c_float*2)(1,2);sound=(ctypes.c_float*2)(5,6);error=ctypes.create_string_buffer(256)
        result=self.function(str(self.directory).encode(),step,video,2,sound,2,error,256)
        self.assertEqual(result,expected,error.value)
        self.assertEqual(list(video),[3,4] if expected else [1,2])
        self.assertEqual(list(sound),[7,8] if expected else [5,6])
    def test_valid(self):self.check(struct.pack('<2f',7,8),1)
    def test_truncated(self):self.check(struct.pack('<f',7),0)
    def test_trailing_data(self):self.check(struct.pack('<3f',7,8,9),0)
    def test_nonfinite(self):self.check(struct.pack('<2f',7,float('nan')),0)
    def test_budget(self):self.check(struct.pack('<2f',7,8),0,step=6)

if __name__=='__main__':unittest.main()
