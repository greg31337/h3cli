#!/usr/bin/env python3
"""Independent FFmpeg/RGB/PCM oracle for the Ref2VA video preprocessing boundary."""
import array
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
BINARY=Path(os.environ.get('H3_REFVIDEO_TEST_BINARY',str(ROOT/'bin/refvideo_tests'))).resolve()

def ffmpeg(*args):
    return subprocess.check_output(['ffmpeg','-v','error','-y',*map(str,args)])

class MediaTests(unittest.TestCase):
    cases=0
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='h3-refvideo-')
        cls.root=Path(cls.temp.name); cls.source=cls.root/'source.mkv'; cls.long=cls.root/'long.mkv'
        # Deliberately 30 fps: native decoding must normalize to 72 frames at 24 fps.
        ffmpeg('-f','lavfi','-i','testsrc=size=48x32:rate=30','-f','lavfi','-i',
               'sine=frequency=317:sample_rate=32000','-t','3','-c:v','ffv1',
               '-c:a','pcm_f32le','-ac','2',cls.source)
        ffmpeg('-f','lavfi','-i','color=size=32x32:rate=24','-frames:v','361','-c:v','ffv1',cls.long)
        cls.wav=cls.root/'supplied.wav'
        ffmpeg('-i',cls.source,'-vn','-c:a','pcm_f32le',cls.wav)
        cls.rgb=ffmpeg('-i',cls.source,'-map','0:v:0','-an','-vf',
                      'fps=24,scale=32:32:flags=lanczos,setsar=1','-f','rawvideo','-pix_fmt','rgb24','pipe:1')
        assert len(cls.rgb)==72*32*32*3
        pcm=ffmpeg('-i',cls.source,'-map','0:a:0','-ac','2','-ar','32000','-f','f32le','pipe:1')
        cls.pcm=array.array('f'); cls.pcm.frombytes(pcm)
        scale=struct.unpack('<f',struct.pack('<f',1/255))[0]
        cls.f32=[struct.pack('<f',n*scale) for n in range(256)]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup(); print(f'ok: {cls.cases} normalized video media cases')

    def decode(self,cap,source=None,audio=None,accepted=True):
        type(self).cases+=1; base=self.root/f'case-{self.cases}'
        command=[str(BINARY),'--decode',str(source or self.source),str(cap),str(base)]
        if audio: command.append(str(audio))
        result=subprocess.run(command,capture_output=True,text=True,cwd=ROOT)
        self.assertEqual(result.returncode==0,accepted,result.stderr)
        return (json.loads(result.stdout) if accepted else result.stderr),base

    def rgb_oracle(self,frames):
        raw=self.rgb[:frames*32*32*3]
        return b''.join(self.f32[value] for c in range(3) for value in raw[c::3])

    def test_normalized_rgb_and_vae_prefix(self):
        for cap,vae,latent in [(48,39,12),(56,56,17),(60,56,17),(72,56,17)]:
            with self.subTest(cap=cap):
                plan,base=self.decode(cap)
                self.assertEqual((plan['frames'],plan['vae_frames'],plan['latent_t']),(cap,vae,latent))
                self.assertEqual(base.with_suffix('.rgb').read_bytes(),self.rgb_oracle(cap))
                # Channels retain the normalized stride, even when the VAE sees fewer frames.
                pixels=base.with_suffix('.rgb').read_bytes(); area=32*32*4
                for channel in range(3):
                    self.assertEqual(pixels[channel*cap*area:(channel*cap+vae)*area],
                                     self.rgb_oracle(vae)[channel*vae*area:(channel+1)*vae*area])

    def test_qwen_pairs_and_timestamps(self):
        for legacy,cap in [(False,60),(False,72)]:
            plan,base=self.decode(cap)
            frames=plan['frames']; indices=list(range(0,frames,12)); pairs=[]; expected=bytearray()
            for start in range(0,len(indices),2):
                first,second=indices[start],indices[min(start+1,len(indices)-1)]
                pairs.append([first,second,(first+second)/48])
                for frame in [first,second]:
                    raw=self.rgb[frame*32*32*3:(frame+1)*32*32*3]
                    for channel in range(3):
                        for value in raw[channel::3]: expected.extend(self.f32[value])
            self.assertEqual(plan['pairs'],pairs)
            self.assertEqual(base.with_suffix('.pairs').read_bytes(),expected)
        self.assertEqual(pairs[-1],[48,60,2.25])

    def test_embedded_and_supplied_soundtracks(self):
        for cap,legacy,expected_samples in [(48,False,64000),(60,False,80000),(72,False,96000)]:
            for audio in [self.source,self.wav]:
                plan,base=self.decode(cap,audio=audio)
                self.assertEqual(plan['soundtrack_samples'],expected_samples)
                pcm=self.pcm[:2*expected_samples]
                expected=(pcm[0::2]+pcm[1::2]).tobytes()
                self.assertEqual(base.with_suffix('.audio').read_bytes(),expected)

    def test_released_limits_and_request_cap(self):
        error,_=self.decode(47,accepted=False); self.assertIn('2–15 seconds',error)
        plan,_=self.decode(360,source=self.long)
        self.assertEqual((plan['frames'],plan['vae_frames'],plan['latent_t']),(360,345,102))
        error,_=self.decode(362,source=self.long,accepted=False); self.assertIn('2–15 seconds',error)
        plan,_=self.decode(60,source=self.long); self.assertEqual(plan['frames'],60)

if __name__=='__main__': unittest.main()
