#!/usr/bin/env python3
"""Pixel parity with the pinned old decoder and between default/explicit paths."""
import json,os,subprocess
from pathlib import Path
from tilefix_validation import ROOT,sha
from source_tree import historical_source

def main():
    out=ROOT/'outputs/tilefix-validation/compatibility';out.mkdir(parents=True,exist_ok=True)
    baseline=out/'baseline';(baseline/'tests').mkdir(parents=True,exist_ok=True)
    (baseline/'src/vae').mkdir(parents=True,exist_ok=True)
    source=historical_source(ROOT,'59b64c9:h3_video_vae.c')
    (baseline/'src/vae/video_vae.c').write_bytes(source)
    (baseline/'tests/tilefix_decode.c').write_bytes((ROOT/'tests/tilefix_decode.c').read_bytes())
    binary=ROOT/'bin/tilefix_baseline_decode'
    binary.parent.mkdir(parents=True,exist_ok=True)
    cmd=['clang','-O3','-std=c11','-D_DARWIN_C_SOURCE','-I'+str(ROOT),str(baseline/'tests/tilefix_decode.c'),str(ROOT/'bin/libh3.a')]
    for framework in ('Foundation','Metal','MetalPerformanceShaders','MetalPerformanceShadersGraph','Accelerate'):cmd+=['-framework',framework]
    subprocess.run(cmd+['-licucore','-lm','-o',str(binary)],check=True)
    matrix=ROOT/'outputs/tilefix-validation/matrix';records={}
    for name,exe,policy,mode,h,w,reference in [
        ('legacy',binary,None,'resident',576,1024,matrix/'576x1024/auto.f32'),
        ('explicit-256',ROOT/'bin/tilefix_decode','256','decode',320,320,matrix/'320x320/default.f32'),
        ('default-standalone',ROOT/'bin/tilefix_decode',None,'decode',320,320,matrix/'320x320/default.f32')]:
        env={k:v for k,v in os.environ.items() if not k.startswith('H3_')}
        if policy:env['H3_VAE_TILE_PIXELS']=policy
        dst=out/(name+'.f32');latent=matrix/f'{h}x{w}'/'latent.f32'
        command=[str(exe),mode,'7',str(h),str(w),str(latent),str(dst),str(ROOT/'models/MiniMax-H3/FL2VA/video_vae/source')]
        print(name,flush=True);r=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,check=True)
        assert sha(dst)==sha(reference),name
        records[name]={'stats':json.loads(r.stdout),'sha256':sha(dst),'reference':str(reference.relative_to(ROOT)),'exact':True,'command':command}
    (out/'results.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
