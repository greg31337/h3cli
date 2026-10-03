#!/usr/bin/env python3
"""Crossed primary clean-state full video VAE replay, with bounded raw storage.

If independently generated clean payloads are identical, verify that fact and
execute the two decoders once on the common payload. This establishes all four
crossed combinations without repeating an identical computation.
"""
import argparse, hashlib, json, os, struct, subprocess, time
from pathlib import Path
import numpy as np
from cuda_sglang_compare import metric, oracle, pack_video, pack_audio


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','oracle-python','model','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--private-cublas',required=True)
    p.add_argument('--case',nargs=3,action='append',required=True,metavar=('ID','ORACLE_RUN','NATIVE_STATE'))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);rows=[]
    def run(cmd,root,name,env=None):
        (root/(name+'-command.json')).write_text(json.dumps(dict(argv=cmd,env=env or {}),indent=2)+'\n')
        start=time.monotonic()
        with (root/(name+'.log')).open('x') as log:
            r=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=os.environ|(env or {}),cwd=a.source)
        if r.returncode:raise RuntimeError(name+' failed: '+str(r.returncode))
        return time.monotonic()-start
    for case,oracle_run,state in a.case:
        if case not in ('C0','C1','C2'):raise ValueError('not a frozen primary')
        root=a.out/case;root.mkdir();row=dict(case=case,oracle_run=oracle_run,native_state=state)
        try:
            spec=json.loads((Path(oracle_run)/'command.json').read_text());blob=Path(state).read_bytes()
            if blob[:8]!=b'H3AV\r\n\x1a\n' or hashlib.sha256(blob[:128]+blob[160:]).digest()!=blob[128:160]:raise ValueError('invalid clean native state')
            width,height,frames=struct.unpack_from('<3I',blob,24)
            if (width,height,frames)!=(640,480,spec['frames']) or spec['case']!=case:raise ValueError('state geometry mismatch')
            nv,na=struct.unpack_from('<2Q',blob,72)
            last=spec['evaluations']-1
            v=np.frombuffer(blob,dtype='<f4',count=nv//4,offset=160)
            au=np.frombuffer(blob,dtype='<f4',count=na//4,offset=160+nv)
            row['clean_video']=metric(pack_video(v),oracle(Path(oracle_run)/'capture',f'step-{last:03d}.video'))
            row['clean_audio']=metric(pack_audio(au),oracle(Path(oracle_run)/'capture',f'step-{last:03d}.audio'))
            if not row['clean_video'].get('exact') or not row['clean_audio'].get('exact'):raise ValueError('crossed replay requires separate differing payloads')
            row['crossed_equivalence']='Both independent clean AV payloads are bitwise identical; each decoder executed once on that common video payload.'
            n=root/'native';n.mkdir();o=root/'oracle'
            row['native_command_seconds']=run([str(a.source/'bin/cuda_sglang_vae'),str(a.model/'source'),state,'legacy',str(n/'rgb.f32'),'1','full'],root,'native',dict(H3_SGLANG_CUBLAS_LIBRARY=a.private_cublas))
            row['oracle_command_seconds']=run([str(a.oracle_python),str(a.source/'tests/cuda_sglang_vae.py'),'--model',str(a.model),'--state',state,'--out',str(o),'--no-capture'],root,'oracle')
            expected=frames*640*480*3*4
            for f in (n/'rgb.f32',o/'rgb.f32'):
                if f.stat().st_size!=expected:raise ValueError('invalid raw video size')
            row['raw_rgb']=metric(np.memmap(n/'rgb.f32',dtype='<f4',mode='r'),np.memmap(o/'rgb.f32',dtype='<f4',mode='r'))
            row['passed']=bool(row['raw_rgb'].get('exact'))
            row['sha256']={str(f.relative_to(root)):digest(f) for f in (n/'rgb.f32',o/'rgb.f32')}
        except (OSError,ValueError,RuntimeError,KeyError) as e:row.update(passed=False,error=str(e))
        rows.append(row);(root/'result.json').write_text(json.dumps(row,indent=2)+'\n')
        result=dict(kind='crossed raw video decoder; no end-to-end timing qualification',source_binary_sha256=digest(a.source/'bin/cuda_sglang_vae'),results=rows,passed=all(r['passed'] for r in rows))
        (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(case,row.get('passed'),row.get('error',''),flush=True)
    return 0 if all(r['passed'] for r in rows) else 1
if __name__=='__main__':raise SystemExit(main())
