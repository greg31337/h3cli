#!/usr/bin/env python3
"""Decode pre-tilefix .h3av/.h3sample artifacts through their existing loaders."""
import json, os, subprocess
from pathlib import Path
from tilefix_validation import sha,ROOT

def main():
    out=ROOT/'outputs/tilefix-validation/saved';out.mkdir(parents=True,exist_ok=True)
    cases={'av':('av',27,256,ROOT/'outputs/resume-validation/suite/t2va/oracle.h3av'),
           'sample':('sample',37,256,ROOT/'outputs/refvideo-integration-validation/released-final.h3sample'),
           'old-policy-av':('av',17,320,ROOT/'outputs/tilefix-validation/generation/fl2va-before.h3av')}
    records={}
    env={k:v for k,v in os.environ.items() if not k.startswith('H3_')}
    for kind,(mode,t,size,source) in cases.items():
        before=sha(source);outputs={}
        weights=ROOT/'models/MiniMax-H3'/('Ref2VA' if kind=='sample' else 'FL2VA')/'video_vae/source'
        for policy in ('default','256','auto','320'):
            dst=out/(kind+'-'+policy+'.f32');e=env.copy()
            if policy!='default':e['H3_VAE_TILE_PIXELS']=policy
            cmd=[str(ROOT/'bin/tilefix_decode'),mode,str(t),str(size),str(size),str(source),str(dst),str(weights)]
            print('saved',kind,policy,flush=True)
            r=subprocess.run(cmd,env=e,cwd=ROOT,capture_output=True,text=True,check=True)
            outputs[policy]={'stats':json.loads(r.stdout),'sha256':sha(dst)}
        # At 256 canvas every policy is one actual 256 tile (no padding).
        if size==256:assert len({v['sha256'] for v in outputs.values()})==1
        else:
            assert outputs['default']['sha256']==outputs['256']['sha256']
            assert outputs['auto']['sha256']==outputs['320']['sha256']
            assert outputs['default']['sha256']==sha(ROOT/'outputs/tilefix-validation/generation/fl2va-after.decoded-rgb')
            assert outputs['auto']['sha256']==sha(ROOT/'outputs/tilefix-validation/generation/fl2va-before.decoded-rgb')
        assert sha(source)==before
        records[kind]={'source':str(source.relative_to(ROOT)),'source_sha256':before,'outputs':outputs}
    (out/'results.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
