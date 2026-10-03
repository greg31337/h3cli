#!/usr/bin/env python3
"""Assert the final uninstrumented production library defaults to reference.

Replays the four captured input presentations and compares final conditioning
byte-for-byte with the earlier explicit-reference encoder runs.
"""
import json,os,subprocess
from pathlib import Path
from scalingfix_prepare import ROOT,OUT,FLAGS
from memory_generation import digest

def main():
    out=OUT/'default';out.mkdir(exist_ok=True);binary=ROOT/'bin/scalingfix_default'
    binary.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['clang','-O3','-std=c11','-D_DARWIN_C_SOURCE','-I',str(ROOT),str(ROOT/'tests/scalingfix_qwen.c'),str(ROOT/'bin/libh3.a'),*FLAGS,'-o',str(binary)],check=True)
    env={k:v for k,v in os.environ.items() if not k.startswith('H3_')};records={}
    for case in ('plain','dialogue','image','video'):
        dst=out/case;dst.mkdir(exist_ok=True)
        weights=ROOT/'models/MiniMax-H3'/('Ref2VA' if case in ('image','video') else 'FL2VA')/'text_encoder'
        result=subprocess.run([str(binary),str(weights),str(OUT/'presentations'/case),str(dst)],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
        got=digest(dst/'final.bf16');expected=digest(OUT/'encoder'/case/'reference/layer-50.bf16');assert got==expected,(case,'unset mode differs from explicit reference')
        records[case]={'stats':json.loads(result.stdout),'sha256':got,'exact':True,'binary_sha256':digest(binary),'mode_environment_present':False}
        print('PASS production default matches explicit reference:',case,flush=True)
    (out/'results.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
