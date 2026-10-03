#!/usr/bin/env python3
"""Codec/metadata/CLI regressions with synthetic frames, no model evaluations."""
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
FFMPEG=os.environ.get('H3_FFMPEG','ffmpeg')
FFPROBE=os.environ.get('H3_FFPROBE','ffprobe')

class Encoding(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='h3-output-')
        self.path=Path(self.tmp.name)/'video.mp4'
        self.env=dict(os.environ)
        self.env.pop('H3_TEST_REFERENCE_ENCODING',None)
    def tearDown(self): self.tmp.cleanup()
    def encode(self,quality='auto',crf='auto',lossless=0,recipe=0,mode='stream',regression=False,ok=True):
        env=dict(self.env)
        if regression:env['H3_TEST_REFERENCE_ENCODING']='1'
        p=subprocess.run([str(ROOT/'bin/output_encoding'),str(self.path),quality,str(crf),str(lossless),str(recipe),mode],env=env,capture_output=True)
        if not ok:
            self.assertNotEqual(p.returncode,0);return p
        self.assertEqual(p.returncode,0,p.stderr.decode())
        return json.loads(subprocess.check_output([FFPROBE,'-v','error','-show_streams','-of','json',str(self.path)]))['streams']
    def crf(self):
        data=self.path.read_bytes();match=re.search(rb'\bcrf=([\d.]+)',data)
        if match:return float(match[1])
        self.assertRegex(data,rb'rc=cqp.*\bqp=0\b');return 0
    def atoms(self):
        data=self.path.read_bytes();at=0;atoms=[]
        while at+8<=len(data):
            size,kind=struct.unpack('>I4s',data[at:at+8])
            if size==1:size=struct.unpack('>Q',data[at+8:at+16])[0]
            if size==0:size=len(data)-at
            self.assertGreaterEqual(size,8);atoms.append(kind);at+=size
        return atoms
    def test_production_profiles_and_override(self):
        for quality,crf in [('auto',18),('default',25),('maximum',0),('high',5),('medium',22),('low',33)]:
            for recipe in (0,4):
                with self.subTest(quality=quality,recipe=recipe):
                    streams=self.encode(quality,recipe=recipe)
                    self.assertEqual(self.crf(),crf)
                    v,a=streams
                    self.assertEqual((v['codec_name'],v['pix_fmt'],v['color_range']),('h264','yuv420p','tv'))
                    self.assertEqual([v[k] for k in ('color_space','color_transfer','color_primaries')],['bt709']*3)
                    self.assertEqual((a['codec_name'],a['sample_rate'],a['channels']),('aac','32000',2))
                    atoms=self.atoms();self.assertLess(atoms.index(b'moov'),atoms.index(b'mdat'))
        self.encode('low',crf=17);self.assertEqual(self.crf(),17)
    def test_all_writers_and_lossless_rgb(self):
        expected=bytearray((i*37+i//19)%256 for i in range(32*32*4*3))
        for f in range(4):
            for y in range(32):
                for x in range(16):
                    i=(f*32*32+y*32+x)*3
                    expected[i:i+3]=bytes([255 if f==c else 0 for c in range(3)])
        for mode in ('stream','buffered','silent'):
            for recipe in (0,4):
                streams=self.encode('low',lossless=1,recipe=recipe,mode=mode)
                v=streams[0];self.assertEqual(v['pix_fmt'],'gbrp');self.assertEqual(v['color_range'],'pc')
                decoded=subprocess.check_output([FFMPEG,'-v','error','-i',str(self.path),'-map','0:v','-pix_fmt','rgb24','-f','rawvideo','pipe:1'])
                self.assertEqual(decoded,expected)
                self.assertEqual(len(streams),1 if mode=='silent' else 2)
        self.encode(crf=0,lossless=1)
        self.assertIn(b'lossless-video',self.encode(crf=18,lossless=1,ok=False).stderr)
    def test_bt709_matrix(self):
        self.encode('maximum')
        yuv=subprocess.check_output([FFMPEG,'-v','error','-i',str(self.path),'-an','-pix_fmt','yuv420p','-f','rawvideo','pipe:1'])
        for f,y in enumerate((63,173,32,16)):
            self.assertLessEqual(abs(yuv[f*1536+8*32+8]-y),1)
    def test_isolated_regression_profile(self):
        v=self.encode(recipe=4,regression=True)[0]
        self.assertEqual(self.crf(),25)
        self.assertNotIn('color_space',v)
        atoms=self.atoms();self.assertGreater(atoms.index(b'moov'),atoms.index(b'mdat'))
        self.encode('high',recipe=4,regression=True,ok=False)
        self.encode(regression=True,ok=False)
    def test_cli_rejects_invalid_or_inapplicable_options(self):
        for flags in (['--ffmpeg-crf','-1'],['--ffmpeg-crf','52'],['--ffmpeg-crf','1.5'],
                      ['--output-quality','lossless'],['--lossless-video','--ffmpeg-crf','18'],
                      ['--still','--output-quality','high'],['--info','--lossless-video']):
            p=subprocess.run([str(ROOT/'bin/h3cli'),'--offline',*flags],env=self.env,capture_output=True)
            self.assertEqual(p.returncode,2,p.stderr.decode())

if __name__=='__main__':unittest.main()
