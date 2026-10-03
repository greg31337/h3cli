#!/usr/bin/env python3
"""Render the native pipeline with the oracle's fixed Qwen boundary, from fresh media.

Only isolated test copies gain text replay and first-velocity capture hooks.
Reference encoders, augmentation, layout, noise, DiT and decoders run normally.
"""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from continuation_regression import instrument,HELPER
from test_sampler_file import entries
from source_tree import copy_source_tree

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    record=json.loads((args.oracle/'preparation.json').read_text());request=record['request']
    original={p[0]:bytes(p[-1]) for p in entries(args.checkpoint.read_bytes())}
    (out/'input.text').write_bytes(original[5])
    source=out/'source';source.mkdir()
    copy_source_tree(ROOT, source)
    instrument(source)
    path=source/'src/engine.c';s=path.read_text()
    for kind in ('video','audio'):
        s=s.replace(f'    regression_buffer("condition-{kind}",condition_{kind}_rows,condition_{kind}_elements*4,1);\n','')
    path.write_text(s)
    path=source/'src/denoise/dit.c';s=path.read_text()
    needle='            if (ok && reuse_interval > 1) {'
    assert s.count(needle)==1
    s=s.replace(needle,'''            if (ok && step == 0) {
                regression_buffer("video_velocity",video_velocity,video_count*4,0);
                regression_buffer("audio_velocity",audio_velocity,audio_count*4,0);
            }
'''+needle)
    path.write_text('#include <stdio.h>\n#include <stdlib.h>\n'+HELPER+s)
    with (out/'build.log').open('w') as log:subprocess.run(['make','-j8','all'],cwd=source,stdout=log,stderr=log,check=True)
    cmd=[str(source/'bin/h3cli'),'-d',str(ROOT/'models/MiniMax-H3'),'-p',record['prompt'],
        '--ref-silent-video',record['reference'],'--save-sampler-state',str(out/'final.h3sample'),'-o',str(out/'native.mp4')]
    for key,flag in {'width':'--width','height':'--height','frames':'--frames','steps':'--steps','seed':'--seed',
                     'dit_layers':'--layers','denoise_reuse':'--reuse','core_reuse':'--core-reuse'}.items():
        cmd += [flag,str(request[key])]
    for key,value in request.items():
        if (key.startswith('use_') or key in ('token_reduction','ssd_streaming')) and value:
            cmd += ['--'+key.replace('_','-')]
    env={k:v for k,v in os.environ.items() if not k.startswith('H3_')}
    env.update(H3_CPU_SAMPLER='1',H3_DISABLE_FUSED_MLP='1',H3_DIT_F32_FINAL='1',
               H3_REGRESSION_REPLAY=str(out/'input'),H3_REGRESSION_DUMP=str(out/'native'))
    with (out/'render.log').open('w') as log:subprocess.run(cmd,cwd=source,env=env,stdout=log,stderr=log,check=True)
    final={p[0]:bytes(p[-1]) for p in entries((out/'final.h3sample').read_bytes())}
    for kind in (5,6,7,8,9,10,11,21,22,32):
        assert original[kind]==final[kind],f'controlled request boundary {kind} changed'
    result={'passed':True,'command':cmd,'fixed_sections':[5,6,7,8,9,10,11,21,22,32],
        'binary_sha256':hashlib.sha256((source/'bin/h3cli').read_bytes()).hexdigest(),
        'original_checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}
    (out/'render-identity.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS complete native render; all controlled conditions/layout/noise match original capture exactly',flush=True)
if __name__=='__main__':main()
