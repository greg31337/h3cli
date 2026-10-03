#!/usr/bin/env python3
"""Exercise tile policy diagnostics through the decode-only plan command."""
import json, os, subprocess, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class PolicyLogging(unittest.TestCase):
    def test_modes(self):
        for value,tile,policy,warning in [(None,256,'default',False),('256',256,'explicit override',False),
                ('320',320,'explicit override',True),('auto',304,'auto',False)]:
            with self.subTest(value=value):
                env=os.environ.copy();env.pop('H3_VAE_TILE_PIXELS',None);env['H3_PROFILE']='1'
                if value is not None:env['H3_VAE_TILE_PIXELS']=value
                r=subprocess.run([str(ROOT/'bin/tilefix_decode'),'plan','7','768','768','-','-','-'],env=env,capture_output=True,text=True,check=True)
                self.assertEqual(json.loads(r.stdout)['tile'],tile)
                self.assertIn(f'{tile} px ({policy})',r.stderr)
                self.assertEqual('reconstruction may differ' in r.stderr,warning)
    def test_invalid(self):
        for value in ['', 'garbage', '336', '384', '512', '1088', '240', '257']:
            env=os.environ.copy();env['H3_VAE_TILE_PIXELS']=value
            r=subprocess.run([str(ROOT/'bin/tilefix_decode'),'plan','7','768','768','-','-','-'],env=env,capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)
            self.assertIn(f'H3_VAE_TILE_PIXELS={value}',r.stderr)
            self.assertIn('256..320',r.stderr)
            self.assertIn('default is 256',r.stderr)
    def test_quiet(self):
        env=os.environ.copy();env.pop('H3_PROFILE',None)
        r=subprocess.run([str(ROOT/'bin/tilefix_decode'),'plan','7','320','320','-','-','-'],env=env,capture_output=True,text=True,check=True)
        self.assertEqual(r.stderr,'')
if __name__=='__main__':unittest.main()
