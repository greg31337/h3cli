#!/usr/bin/env python3
"""Run real installed weights, instrumenting actual execution entries in isolated copies."""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sanitize',action='store_true')
    args=p.parse_args()
    out=ROOT/'outputs/memory-validation'/('cancellation-sanitize' if args.sanitize else 'cancellation')
    out.mkdir(parents=True,exist_ok=True)
    flags=['-I',str(ROOT),'-std=c11','-D_DARWIN_C_SOURCE','-g','-O1' if args.sanitize else '-O3']
    if args.sanitize: flags+=['-fsanitize=address,undefined','-fno-omit-frame-pointer']
    objects=[]
    insertions={
        'src/conditioning/text_encoder.c':('        int layer_ok =','        h3_test_enter("text");\n'),
        'src/conditioning/vision_encoder.c':('        vision_block_weights block;','        h3_test_enter("vision");\n'),
        'src/vae/video_vae.c':('    vae_block *weight = &vae->blocks[index];','    h3_test_enter("video");\n'),
        'src/vae/audio_vae.c':('        audio_stage stage = {0};','        h3_test_enter("audio");\n'),
        'src/vae/video_encoder.c':('            next = run_block(encoder, hidden,','            h3_test_enter("video-encoder");\n'),
    }
    for name in [*insertions,'src/denoise/dit.c']:
        source=(ROOT/name).read_text()
        if name in insertions:
            needle,insert=insertions[name]
            assert source.count(needle)==1,(name,needle)
            source=source.replace(needle,insert+needle)
        if name=='src/vae/audio_vae.c':
            needle='        for (int residual = 0; ok && residual < ENCODER_RESIDUALS; residual++)'
            assert source.count(needle)==1
            source=source.replace(needle,'        h3_test_enter("audio-encoder");\n'+needle)
        if name=='src/denoise/dit.c':
            # Insert AFTER each progress gate, immediately before the step body.
            needle='dit->sigmas.steps, error, error_size))) break;'
            assert source.count(needle)==3
            source=source.replace(needle,needle+'\n        h3_test_enter("dit");')
        copy=out/name
        copy.parent.mkdir(parents=True,exist_ok=True)
        copy.write_text('extern void h3_test_enter(const char *name);\n'+source)
        obj=copy.with_suffix('.o'); objects.append(str(obj))
        subprocess.run(['clang',*flags,'-c',str(copy),'-o',str(obj)],check=True)
    binary=ROOT/'bin/memory_cancellation'
    binary.parent.mkdir(parents=True,exist_ok=True)
    command=['clang',*flags,str(ROOT/'tests/memory_cancellation.c'),*objects,
        str(ROOT/'bin/libh3.a'),'-framework','Foundation','-framework','Metal',
        '-framework','MetalPerformanceShaders','-framework','MetalPerformanceShadersGraph',
        '-framework','Accelerate','-framework','CoreML','-framework','CoreVideo',
        '-lc++','-licucore','-lm','-o',str(binary)]
    subprocess.run(command,check=True)
    start=time.monotonic()
    with (out/'test.log').open('w') as log:
        result=subprocess.run([str(binary)],cwd=ROOT,stdout=log,stderr=log)
    print((out/'test.log').read_text(),end='')
    (out/'results.json').write_text(json.dumps({'passed':result.returncode==0,
        'seconds':time.monotonic()-start,'sanitized':args.sanitize,'command':command},indent=2)+'\n')
    result.check_returncode()


if __name__=='__main__': main()
