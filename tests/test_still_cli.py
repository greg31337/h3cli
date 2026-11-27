#!/usr/bin/env python3
"""Preflight must reject conflicts without attempting any model/GPU load."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import subprocess,unittest,tempfile,struct
from pathlib import Path
class StillCLI(unittest.TestCase):
 def reject(self,args,word):
  p=subprocess.run(['./bin/h3cli',*args],capture_output=True,text=True)
  self.assertNotEqual(p.returncode,0);self.assertIn(word,p.stderr);self.assertNotIn('DiT initialization',p.stderr)
 def test_codec_conflicts(self):
  base=['--decode-still-latent','missing','-o','x.png']
  for args in [['--frames','1'],['--seconds','1'],['--width','256'],['--steps','1'],['--preview-vae'],['--save-av-state','x'],['--still'],['--save-still-latent','x'],['--ref-image','x'],['-p','x'],['--resume-sampler-state','x']]:
   with self.subTest(args=args):self.reject(base+args,'accepts only')
 def test_still_conflicts(self):
  base=['-d','missing','--still','-p','x','-o','x.png']
  for args in [['--frames','5'],['--seconds','1'],['--continue-from','x'],['--save-av-state','x'],['--continue-mode','hard'],['--continue-bridge-steps','8'],['--continue-bridge-profile','linear'],['--keep-continuation-prefix']]:self.reject(base+args,'cannot use')
  for args in [['--preview-vae'],['--ref-audio','x'],['--ref-video','x'],['--first-frame','x'],['--stop-after-step','1'],['--load-conditioning','x'],['--layers','45'],['--reuse','2'],['--width','641'],['-o','x.mp4']]:
   with self.subTest(args=args):self.reject(base+args,'still')
 def test_high_resolution_png(self):
  with tempfile.TemporaryDirectory() as directory:
   for width,height in ((2144,1216),(3872,2176),(2176,3872)):
    path=Path(directory)/'image.png'
    subprocess.run(['./bin/still_probe','png',str(width),str(height),str(path)],check=True,capture_output=True)
    data=path.read_bytes();self.assertEqual(data[:8],b'\x89PNG\r\n\x1a\n');self.assertEqual(struct.unpack('>II',data[16:24]),(width,height))
 def test_video_selection(self):self.reject(['-d','missing','--image-vae','x'],'retired')
 def test_still_count_validation(self):
  base=['--still','-p','x','-d','missing']
  for count in ('0','-1','1.5','2147483648','18446744073709551616'):
   with self.subTest(count=count):self.reject(base+['--still-count',count],'still-count')
  self.reject(['--still-count','2','-p','x','-d','missing'],'requires --still')
  self.reject(base+['--still-count','2','--seed','18446744073709551615'],'overflow')
  self.reject(base+['--still-count','3','--seed','18446744073709551614'],'overflow')
if __name__=='__main__':unittest.main()
