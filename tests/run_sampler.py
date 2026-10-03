#!/usr/bin/env python3
"""Sequential real-model pause/restart oracle for resume tasks T001–T032."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import time
from test_sampler_file import entries

ROOT=Path(__file__).resolve().parents[1]

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def appended(base,ext): return Path(str(base)+'.'+ext)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',default='t2va,ref2va,continuation,reuse2,reuse3,bridge,mixed')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/resume-validation/suite')
    parser.add_argument('--reuse-t2va',type=Path,help='Use the already completed t2va oracle from this exact driver build')
    args=parser.parse_args(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    frozen=out/'build'; frozen.mkdir(exist_ok=True)
    (frozen/'src/metal').mkdir(parents=True,exist_ok=True)
    (frozen/'bin').mkdir(exist_ok=True)
    for file in ['bin/h3cli','bin/sampler_generate','src/metal/shaders.metal']:
        shutil.copy2(ROOT/file,frozen/file)
    manifest={'passed':False,'binary_sha256':digest(frozen/'bin/sampler_generate'),'cli_sha256':digest(frozen/'bin/h3cli'),
              'shader_sha256':digest(frozen/'src/metal/shaders.metal'),'runs':{}}
    env={k:v for k,v in os.environ.items() if not k.startswith('H3_')}; env['H3_CPU_SAMPLER']='1'
    def run(name,cmd,cwd):
        print('running',name,flush=True); started=time.monotonic()
        with (out/(name+'.log')).open('w') as log:
            subprocess.run([str(v) for v in cmd],cwd=cwd,env=env,stdout=log,stderr=log,check=True)
        return {'command':[str(v) for v in cmd],'seconds':time.monotonic()-started}
    for name in args.only.split(','):
        env['H3_TEST_LEGACY_REFVIDEO']='1' if name=='mixed' else '0'
        directory=out/name; directory.mkdir(exist_ok=True)
        (directory/'inputs').mkdir(exist_ok=True)
        (directory/'src/metal').mkdir(parents=True,exist_ok=True)
        for f in ['face1.jpg','body1.jpg']: shutil.copy2(ROOT/'inputs'/f,directory/'inputs'/f)
        for link,target in [('src/metal/shaders.metal',frozen/'src/metal/shaders.metal'),('outputs',ROOT/'outputs')]:
            if not (directory/link).exists(): (directory/link).symlink_to(target)
        source='-'
        if name in ['continuation','bridge']:
            source=directory/'source.h3av'
            shutil.copy2(ROOT/'outputs/resume-validation/baseline/ref2va-original.h3av',source)
        base=directory/'oracle'; resumed=directory/'resumed'
        mode='ref2va' if name.startswith('reuse') or name=='continuation' else name
        reuse=int(name[-1]) if name.startswith('reuse') else 1
        if name=='t2va' and args.reuse_t2va:
            for ext in ['mp4','h3av','trace','h3sample','full.h3sample','preview.mp4','after-preview.h3sample']:
                shutil.copy2(appended(args.reuse_t2va,ext),appended(base,ext))
            record={'reused_from':str(args.reuse_t2va)}
        else:
            record=run(name+'-oracle',[frozen/'bin/sampler_generate',ROOT/'models/MiniMax-H3',base,mode,source,reuse],directory)
        parts={p[0]:p for p in entries(appended(base,'h3sample').read_bytes())}
        assert struct.unpack_from('<4I',parts[1][-1])==(1,20,4,reuse)
        assert struct.unpack_from('<I',parts[11][-1])[0]==20
        av=parts[12][-1]+parts[13][-1]; trace=appended(base,'trace').read_bytes()
        assert len(trace)==21*len(av) and av==trace[4*len(av):5*len(av)]
        assert appended(base,'h3sample').read_bytes()==appended(base,'after-preview.h3sample').read_bytes()
        if name in ['ref2va','mixed']:
            fullparts={p[0]:p for p in entries(appended(base,'full.h3sample').read_bytes())}
            assert 23 in fullparts, 'multimodal presentation diagnostics missing'
        if name=='mixed': assert len(parts[8][-1])>0
        if source!='-':
            assert parts[15][-1][8:40]==bytes.fromhex(digest(source)), 'wrong continuation source fingerprint'
            raw=source.read_bytes()
            vt,lh,lw,at=struct.unpack_from('<4I',raw,36)
            video_bytes=struct.unpack_from('<Q',raw,72)[0]
            vp,ap=struct.unpack_from('<2I',parts[10][-1],68)
            plane=lh*lw*4
            video_hash=hashlib.sha256(b''.join(raw[160+(c*vt+vt-vp)*plane:160+(c+1)*vt*plane] for c in range(24))).digest()
            audio_start=160+video_bytes
            audio_hash=hashlib.sha256(b''.join(raw[audio_start+(c*at+at-ap)*4:audio_start+(c+1)*at*4] for c in range(64))).digest()
            assert parts[15][-1][40:72]==video_hash and parts[15][-1][72:104]==audio_hash, 'raw continuation-tail fingerprint mismatch'
            source.unlink()
        # References and .h3av source disappear before process restart.
        (directory/'inputs').rename(directory/'unavailable-inputs')
        if name=='mixed':
            # The original audio/video paths live under the shared outputs link;
            # use an empty local outputs tree during resume to make them absent.
            (directory/'outputs').unlink(); (directory/'outputs').mkdir()
            (directory/'outputs'/'.h3-model-hashes').symlink_to(ROOT/'outputs/.h3-model-hashes')
        record['resume']=run(name+'-resume',[frozen/'bin/sampler_generate',ROOT/'models/MiniMax-H3',resumed,'resume',appended(base,'h3sample'),appended(base,'trace')],directory)
        for ext in ['h3av','mp4']:
            assert appended(base,ext).read_bytes()==appended(resumed,ext).read_bytes(), name+' '+ext+' changed'
        for label,expected_audio in [('preview',False),('resumed',True)]:
            media=appended(base,'preview.mp4') if label=='preview' else appended(resumed,'mp4')
            streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(media)]))['streams']
            assert {s['codec_type'] for s in streams}==({'audio','video'} if expected_audio else {'video'})
        log=(out/(name+'-resume.log')).read_text()
        assert 'tokenizer, text/vision and reference encoders skipped' in log
        record['hashes']={ext:digest(appended(base,ext)) for ext in ['h3av','mp4','trace','h3sample','full.h3sample']}
        record['passed']=True; manifest['runs'][name]=record
        (out/'results.json').write_text(json.dumps(manifest,indent=2)+'\n')
    manifest['passed']=True; (out/'results.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('ok: all requested uninterrupted/pause/restart comparisons are byte-identical',flush=True)

if __name__=='__main__': main()
