#!/usr/bin/env python3
"""Adversarial schema-2 still artifacts and retired-checkpoint role checks."""
import copy
import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from still_host import HEADER, sparse_checkpoint

BASE = {'__metadata__': {'h3_still_schema':'2','h3_latent_space':'h3-normalized-v1',
        'h3_vae_compatibility':'a'*64,'h3_decoder_recipe':'stock-video-repeat5-frame3-f32-v1','h3_model_variant':'FL2VA'},
        'latent':{'dtype':'F32','shape':[1,24,1,2,4],'data_offsets':[0,24*2*4*4]}}
class Metadata(unittest.TestCase):
    def check(self, header, valid=False, truncate=False):
        with tempfile.TemporaryDirectory(prefix='h3-still-metadata-') as directory:
            path=Path(directory)/'latent.safetensors'
            sparse_checkpoint(path,header,truncate)
            result=subprocess.run(['./bin/still_probe','validate',str(path)],capture_output=True,text=True)
            self.assertEqual(result.returncode==0,valid,result.stderr)
    def test_valid(self):
        for variant in ('FL2VA','Ref2VA'):
            header=copy.deepcopy(BASE);header['__metadata__']['h3_model_variant']=variant;self.check(header,True)
    def test_missing(self):
        for key in BASE['__metadata__']:
            header=copy.deepcopy(BASE);del header['__metadata__'][key];self.check(header)
    def test_metadata(self):
        for key,value in (('h3_still_schema','1'),('h3_decoder_recipe','direct-t1'),('h3_model_variant','other'),('h3_latent_space','raw'),('h3_vae_compatibility','g'*64)):
            header=copy.deepcopy(BASE);header['__metadata__'][key]=value;self.check(header)
    def test_shapes_and_dtype(self):
        for index,value in ((0,2),(1,23),(2,5),(3,3),(4,2**63)):
            header=copy.deepcopy(BASE);header['latent']['shape'][index]=value;self.check(header)
        header=copy.deepcopy(BASE);header['latent']['dtype']='F16';self.check(header)
        self.check(BASE,truncate=True)
    def test_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'nan.safetensors';sparse_checkpoint(path,BASE)
            with path.open('r+b') as file:
                length=struct.unpack('<Q',file.read(8))[0];file.seek(8+length);file.write(struct.pack('<f',float('nan')))
            self.assertNotEqual(subprocess.run(['./bin/still_probe','validate',str(path)],capture_output=True).returncode,0)
    def test_normalization_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/'FL2VA/video_vae/config.json';config.parent.mkdir(parents=True)
            (config.parent/'source').mkdir()
            def inspect(mean, std):
                config.write_text(json.dumps(dict(latents_mean=mean,latents_std=std)))
                return subprocess.run(['./bin/still_probe','inspect',directory],capture_output=True,text=True)
            a=inspect([0]*24,[1]*24);b=inspect([.01]*24,[1]*24)
            self.assertEqual(a.returncode,0,a.stderr);self.assertEqual(b.returncode,0,b.stderr)
            self.assertNotEqual(a.stdout,b.stdout)
            self.assertNotEqual(inspect([0]*23,[1]*24).returncode,0)
            self.assertNotEqual(inspect([0]*24,[0]*24).returncode,0)
    def test_retired_weight_role(self):
        with tempfile.TemporaryDirectory() as directory:
            sparse_checkpoint(Path(directory)/'old.safetensors',json.loads(HEADER.read_text()))
            result=subprocess.run(['./bin/still_probe','store',directory],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('image-only',result.stderr)
if __name__=='__main__':unittest.main()
