"""Early CLI/script and saved-state rejection; never runs a valid model request."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--binary',type=Path,default=Path('bin/h3cli'))
ap.add_argument('--long-av',type=Path);ap.add_argument('--long-sample',type=Path);ap.add_argument('--long-source',type=Path)
a=ap.parse_args();binary=str(a.binary.resolve());checks=0
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);output=root/'forbidden.mp4'
    def reject(flags,words=()):
        global checks
        p=subprocess.run([binary,'-d',str(root/'missing-model'),'--offline','-o',str(output),*map(str,flags)],capture_output=True,text=True,timeout=15,env=dict(os.environ,H3_OFFLINE='1'))
        assert p.returncode and not output.exists(),p.stdout+p.stderr
        assert 'missing required model' not in p.stderr.lower(),p.stderr
        if words:assert any(w in p.stderr for w in words),p.stderr
        checks+=1
    for frames in ((737,2147483647) if sys.platform!='darwin' else (363,464,736,737,2147483647)):
        reject(['-p','A teapot.','--width','256','--height','256','--frames',frames,'--steps','2'],('frames','limit'))
    for seconds in ('nan','inf','1e99','-1'):
        reject(['-p','A teapot.','--seconds',seconds],('seconds',))
    if sys.platform=='darwin':
        header=bytearray(160);header[:8]=b'H3AV\r\n\x1a\n'
        for offset,value in ((8,3),(12,160),(16,0x01020304),(20,1),(24,256),(28,256),(32,736)):
            struct.pack_into('<I',header,offset,value)
        state=root/'header-only.h3av';state.write_bytes(header)
        Path(str(state)+'.presentation').write_text('H3-PRESENTATION 9\nstate '+64*'0'+'\nvariant 0\n')
        for preview in ([],['--preview-vae']):reject(['--decode-av-state',state,*preview],('CUDA','limit'))
        reject(['-p','A teapot.','--continue-from',state,'--frames','56','--steps','2'],('CUDA','limit'))
        if a.long_av:reject(['--decode-av-state',a.long_av],('CUDA','limit'))
        if a.long_sample:reject(['--resume-sampler-state',a.long_sample],('requirements','duration','limit'))
        if a.long_source:
            for k in (0,2,3,4):reject(['--upscale-state',a.long_source,'--upscale-refine-steps',k],('requirements','duration','limit'))
print(json.dumps({'status':'passed','checks':checks,'models_loaded':False}))
