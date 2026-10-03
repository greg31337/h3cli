#!/usr/bin/env python3
"""Preflight must reject conflicts without attempting any model/GPU load."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import subprocess,unittest
class StillCLI(unittest.TestCase):
 def reject(self,args,word):
  p=subprocess.run(['./bin/h3cli',*args],capture_output=True,text=True)
  self.assertNotEqual(p.returncode,0);self.assertIn(word,p.stderr);self.assertNotIn('DiT initialization',p.stderr)
 def test_codec_conflicts(self):
  base=['--decode-still-latent','missing','--image-vae','missing','-o','x.png']
  for args in [['--frames','1'],['--seconds','1'],['--width','256'],['--steps','1'],['--preview-vae'],['--save-av-state','x'],['--still'],['--save-still-latent','x'],['--ref-image','x'],['-p','x'],['--resume-sampler-state','x']]:
   with self.subTest(args=args):self.reject(base+args,'accepts only')
 def test_still_conflicts(self):
  base=['-d','missing','--still','--image-vae','missing','-p','x','-o','x.png']
  for args in [['--frames','5'],['--seconds','1'],['--continue-from','x'],['--save-av-state','x'],['--continue-mode','hard'],['--continue-bridge-steps','8'],['--continue-bridge-profile','linear'],['--keep-continuation-prefix']]:self.reject(base+args,'cannot use')
  for args in [['--preview-vae'],['--ref-audio','x'],['--ref-video','x'],['--first-frame','x'],['--stop-after-step','1'],['--load-conditioning','x'],['--layers','45'],['--reuse','2'],['--width','641'],['-o','x.mp4']]:
   with self.subTest(args=args):self.reject(base+args,'still')
 def test_video_selection(self):self.reject(['-d','missing','--image-vae','x'],'require --still')
if __name__=='__main__':unittest.main()
