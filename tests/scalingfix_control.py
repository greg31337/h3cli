#!/usr/bin/env python3
"""Build/run real Qwen cancellation and old checkpoint compatibility checks."""
import argparse,subprocess
from pathlib import Path
from scalingfix_prepare import FLAGS,ROOT,OUT

def main():
    p=argparse.ArgumentParser();p.add_argument('--build-only',action='store_true');p.add_argument('--sanitize',action='store_true');a=p.parse_args()
    d=OUT/('control-sanitize' if a.sanitize else 'control');d.mkdir(exist_ok=True)
    s=(ROOT/'src/conditioning/text_encoder.c').read_text();needle='        int layer_ok =';assert s.count(needle)==1;s=s.replace(needle,'        h3_scaling_enter();\n'+needle)
    (d/'encoder.c').write_text('extern void h3_scaling_enter(void);\n'+s)
    flags=['-O1' if a.sanitize else '-O3','-g','-std=c11','-D_DARWIN_C_SOURCE','-I',str(ROOT)]
    if a.sanitize:flags+=['-fsanitize=address,undefined','-fno-omit-frame-pointer']
    (ROOT/'bin').mkdir(exist_ok=True)
    subprocess.run(['clang',*flags,str(ROOT/'tests/scalingfix_control.c'),str(d/'encoder.c'),str(ROOT/'bin/libh3.a'),*FLAGS,'-o',str(ROOT/'bin'/('scalingfix_control_sanitize' if a.sanitize else 'scalingfix_control'))],check=True)
    if not a.build_only:
        with (d/'run.log').open('w') as log:subprocess.run([str(ROOT/'bin'/('scalingfix_control_sanitize' if a.sanitize else 'scalingfix_control'))],cwd=ROOT,stdout=log,stderr=log,check=True)
        print((d/'run.log').read_text())
if __name__=='__main__':main()
