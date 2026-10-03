#!/usr/bin/env python3
"""Sparse adversarial artifacts test parsing without reading model payloads."""
import copy,json,subprocess,tempfile,unittest
from still_host import HEADER, sparse_checkpoint
from pathlib import Path
BASE=json.loads(HEADER.read_text())
class Metadata(unittest.TestCase):
 def check(self,h,valid=False,truncate=False):
  with tempfile.TemporaryDirectory(prefix='h3-still-metadata-') as d:
   p=Path(d)/'image.safetensors';sparse_checkpoint(p,h,truncate)
   r=subprocess.run(['./bin/still_probe','validate',str(p)],capture_output=True,text=True)
   self.assertEqual(r.returncode==0,valid,r.stderr)
 def test_valid(self):self.check(BASE,True)
 def test_truncated(self):self.check(BASE,truncate=True)
 def test_missing(self):
  for k in ['h3_t1_direct','h3_t1_format','h3_t1_output_slice','minimax_h3_video_vae']:
   h=copy.deepcopy(BASE);del h['__metadata__'][k];self.check(h)
 def test_whitening(self):
  for key,index,value in [('latents_mean',0,'NaN'),('latents_std',0,0),('latents_std',2,-1),('latents_std',0,1e300)]:
   h=copy.deepcopy(BASE);c=json.loads(h['__metadata__']['minimax_h3_video_vae']);c[key][index]=value;h['__metadata__']['minimax_h3_video_vae']=json.dumps(c);self.check(h)
 def test_architecture(self):
  for k,v in [('causal_encoder',False),('use_3d_conv',False),('vae_ratio',32),('pixel_norm_type','none'),('embed_dim',25),('zq_ch_encoder',48),('padding_mode_t','replicate')]:
   h=copy.deepcopy(BASE);c=json.loads(h['__metadata__']['minimax_h3_video_vae']);c['source_config'][k]=v;h['__metadata__']['minimax_h3_video_vae']=json.dumps(c);self.check(h)
 def test_shape(self):
  h=copy.deepcopy(BASE);h['decoder.proj_out.weight']['shape'][0]=2**63;self.check(h)
 def test_role(self):
  with tempfile.TemporaryDirectory(prefix='h3-still-role-') as d:
   p=Path(d)/'image.safetensors';sparse_checkpoint(p,BASE)
   r=subprocess.run(['./bin/still_probe','store',d],capture_output=True,text=True);self.assertNotEqual(r.returncode,0);self.assertIn('image-only',r.stderr)
 def test_coexisting_and_duplicates(self):
  original={'weight':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}
  with tempfile.TemporaryDirectory(prefix='h3-still-role-') as d:
   p=Path(d);sparse_checkpoint(p/'a.safetensors',original)
   r=subprocess.run(['./bin/still_probe','store',d],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
   sparse_checkpoint(p/'b.safetensors',original)
   r=subprocess.run(['./bin/still_probe','store',d],capture_output=True,text=True);self.assertNotEqual(r.returncode,0);self.assertIn('duplicate',r.stderr)
   (p/'b.safetensors').unlink();sparse_checkpoint(p/'b.safetensors',BASE)
   r=subprocess.run(['./bin/still_probe','store',d],capture_output=True,text=True);self.assertNotEqual(r.returncode,0);self.assertIn('image-only',r.stderr)
if __name__=='__main__':unittest.main()
