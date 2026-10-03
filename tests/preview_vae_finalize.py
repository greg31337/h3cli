#!/usr/bin/env python3
"""Short tiny-generation -> original-finalization delivery checks, no denoising."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from preview_vae_gallery import inspect

ROOT=Path(os.environ.get('H3_PREVIEW_TEST_ROOT','outputs/preview-vae/metal'))

def decode(path, kind):
    return subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-map',f'0:{kind}',
        '-f','rawvideo' if kind=='v' else 'f32le',*(['-pix_fmt','rgb24'] if kind=='v' else []),'-'])

def main():
    cases=[('invariance/tiny','invariance/default',128,128,22),
           ('continuation/hard-tiny','continuation/hard-full',64,64,51),
           ('continuation/bridge-tiny','continuation/bridge-full',64,64,51)]
    records=[]
    for source,baseline,w,h,frames in cases:
        state=ROOT/(source+'.h3av');before=hashlib.sha256(state.read_bytes()).hexdigest()
        output=ROOT/(source+'-finalized.mp4')
        command=['./bin/h3cli','-d','models/MiniMax-H3','--decode-av-state',str(state),'-o',str(output),'--profile']
        result=subprocess.run(command,capture_output=True,text=True,timeout=90)
        output.with_suffix('.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stderr
        assert 'denoiser calls=0' in result.stderr
        assert not any(phase in result.stderr for phase in ['tokenizer','text encoder','Qwen vision','DiT initialization'])
        assert hashlib.sha256(state.read_bytes()).hexdigest()==before
        inspect(output,frames,w,h)
        # Default CUDA and Metal retain exact delivery checks.
        video_equal=decode(output,'v')==decode(ROOT/(baseline+'.mp4'),'v')
        assert video_equal,(source,'v')
        assert decode(output,'a')==decode(ROOT/(baseline+'.mp4'),'a'),(source,'a')
        records.append(dict(state=str(state),state_sha256=before,output=str(output),mp4_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),command=command,
                            video_byte_identical=video_equal,audio_byte_identical=True,
                            match='exact decoded RGB/audio versus original generation'))
    inspect(ROOT/'invariance/paused-tiny.mp4',22,128,128,audio=False)
    (ROOT/'finalization.json').write_text(json.dumps(records,indent=2)+'\n')
    print('PASS tiny -> full finalize: resize, hard/bridge 39-frame trim, unchanged states, zero denoiser calls, exact audio; silent paused preview; '+
          'exact original RGB')

if __name__=='__main__':main()
