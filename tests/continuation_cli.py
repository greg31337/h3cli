#!/usr/bin/env python3
"""CLI validation and optional real restart/resume with verified CPU fallback."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'outputs/continuation-validation/acceptance/native-1.h3av')
    parser.add_argument('--positive',action='store_true',help='Render the native-2 case via CLI and verify restart/prefix/fallback behavior')
    args=parser.parse_args(); source=args.source.resolve(); assert source.exists()
    out=source.parent/'cli';out.mkdir(exist_ok=True)
    basic=[str(ROOT/'bin/h3cli'),'-d',str(ROOT/'models/MiniMax-H3'),'-p','validation prompt',
           '--width','256','--height','256','--frames','90','--continue-from',str(source)]
    failures=[(['--continue-context','40'],'39 + 51'),(['--continue-context','141'],'history'),
              (['--frames','39'],'suffix'),(['--width','128'],'geometry'),
              (['--first-frame','inputs/face1.jpg'],'anchors'),(['--last-frame','inputs/body1.jpg'],'anchors'),
              (['--core-reuse','4'],'core-reuse'),(['--token-reduction'],'token reduction'),(['--layers','45'],'layers')]
    log=[]
    def rejected(cmd,expected,env=None):
        r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,env=env)
        assert r.returncode and expected in r.stderr,(r.returncode,r.stderr)
        log.append({'arguments':cmd,'expected':expected,'returncode':r.returncode,'stderr':r.stderr})
    for args_extra,expected in failures: rejected(basic+args_extra,expected)
    reduction_env=os.environ.copy();reduction_env['H3_TOKEN_REDUCTION']='1'
    rejected(basic,'token reduction',reduction_env)
    no_source=basic[:-2]
    rejected(no_source+['--continue-context','39'],'require --continue-from')
    rejected(no_source+['--keep-continuation-prefix'],'require --continue-from')
    blob=bytearray(source.read_bytes()); blob[-1]^=1
    corrupt=out/'corrupt.h3av';corrupt.write_bytes(blob)
    rejected(basic+['--continue-from',str(corrupt)],'checksum')
    blob=bytearray(source.read_bytes());blob[88]^=1
    blob[128:160]=hashlib.sha256(blob[:128]+blob[160:]).digest()
    mismatch=out/'incompatible.h3av';mismatch.write_bytes(blob)
    rejected(basic+['--continue-from',str(mismatch)],'incompatible continuation model/VAE signature')
    blob=bytearray(source.read_bytes());struct.pack_into('<I',blob,8,99)
    version=out/'version.h3av';version.write_bytes(blob)
    rejected(basic+['--continue-from',str(version)],'version')
    (out/'validation.json').write_text(json.dumps(log,indent=2))
    print(f'ok: {len(log)} CLI rejection cases',flush=True)
    if args.positive:
        from run_continuation import CHAIN_PROMPT
        cmd=basic.copy();cmd[cmd.index('-p')+1]=CHAIN_PROMPT
        cmd += ['--seed','43','--steps','20','--ref-image','inputs/face1.jpg','--ref-image','inputs/body1.jpg',
                '--save-av-state',str(out/'resume.h3av'),'-o',str(out/'resume.mp4'),'--profile']
        env=os.environ.copy();env['H3_GPU_SAMPLER']='1'
        with (out/'resume.log').open('w') as f:
            subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=f,env=env,check=True)
        assert 'continuation uses CPU F32 Euler sampler' in (out/'resume.log').read_text()
        from continuation_metrics import state,prefix_bytes
        resumed=state(out/'resume.h3av'); expected=state(source.parent/'native-2.h3av')
        assert resumed[0]==expected[0] and prefix_bytes(resumed)==prefix_bytes(expected)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(out/'resume.mp4')]))
        streams={s['codec_type']:s for s in probe['streams']}
        assert abs(float(streams['video']['start_time']))<1e-6
        assert abs(float(streams['audio']['start_time']))<1/32000
        assert int(streams['video']['nb_frames'])==51
        assert abs(float(streams['audio']['duration'])-2.125)<1/32000
        print('ok: CLI restart/resume, exact protected latents, trimmed AV duration and explicit GPU-request CPU fallback')

if __name__=='__main__': main()
