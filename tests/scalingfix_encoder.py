#!/usr/bin/env python3
"""Run identical captured Qwen inputs, preserving all mode and baseline dumps."""
import argparse,json,os,subprocess,time
from pathlib import Path
from memory_generation import digest
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/scalingfix-validation'
def main():
    p=argparse.ArgumentParser();p.add_argument('--only',default='plain,dialogue,image,video');p.add_argument('--modes',default='baseline,legacy,scaled-q,reference,reference-repeat');a=p.parse_args()
    for case in a.only.split(','):
        fixture=OUT/'presentations'/case
        identity={f.name:digest(f) for f in fixture.iterdir() if f.suffix in ('.u64','.u32','.u8','.bf16')}
        for label in a.modes.split(','):
            dest=OUT/'encoder'/case/label;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'result.json').exists():continue
            build=OUT/('baseline' if label=='baseline' else 'native');mode='reference' if label=='reference-repeat' else label
            cwd=build
            if label=='legacy-synchronized':
                mode='legacy';cwd=OUT/'synchronized';cwd.mkdir(exist_ok=True)
                source=(build/'src/metal/shaders.metal').read_text()
                # Older validation builds excluded legacy from the barrier.
                # Current production legacy is already synchronized; retain
                # this label so existing parity reports remain reproducible.
                guard='if constexpr (SCALE_MODE != 2)'
                assert source.count(guard)<=1
                if guard in source:source=source.replace(guard,'if constexpr (true)')
                (cwd/'src/metal').mkdir(parents=True,exist_ok=True)
                (cwd/'src/metal/shaders.metal').write_text(source)
            env={k:v for k,v in os.environ.items() if not k.startswith('H3_')};env['H3_QWEN_GQA_SCALE_MODE']=mode;env['H3_PROFILE']='1'
            cmd=[str(ROOT/'bin'/('scalingfix_qwen_'+build.name)),str(ROOT/'models/MiniMax-H3'/('Ref2VA' if case in ('image','video') else 'FL2VA')/'text_encoder'),str(fixture),str(dest)]
            print(case,label,flush=True);start=time.monotonic()
            with (dest/'run.log').open('w') as log:r=subprocess.run(cmd,cwd=cwd,env=env,stdout=log,stderr=log)
            r.check_returncode();stats=json.loads((dest/'run.log').read_text().splitlines()[-1]);stats['process_seconds']=time.monotonic()-start
            stats['input_hashes']=identity;stats['binary_sha256']=digest(ROOT/'bin'/('scalingfix_qwen_'+build.name));stats['shader_sha256']=digest(cwd/'src/metal/shaders.metal')
            assert all(digest(dest/k)==v for k,v in identity.items()),'presentation changed'
            (dest/'result.json').write_text(json.dumps(stats,indent=2)+'\n');print(stats['seconds'],flush=True)
if __name__=='__main__':main()
