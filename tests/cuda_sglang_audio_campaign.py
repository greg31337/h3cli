#!/usr/bin/env python3
"""Serial crossed audio decode and pinned-AAC qualification; no denoising."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
from cuda_sglang_compare import metric

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','oracle-python','model','ffmpeg','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--private-cublas',required=True);p.add_argument('--case',nargs=2,action='append',required=True,metavar=('ID','STATE'))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    rows=[];contract=json.loads(Path(__file__).with_name('cuda_sglang_contract.json').read_text())
    def run(cmd,root,name,env=None):
        (root/(name+'-command.json')).write_text(json.dumps(dict(argv=cmd,env=env or {}),indent=2)+'\n')
        with (root/(name+'.log')).open('w') as log:
            started=time.monotonic();result=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=os.environ| (env or {}))
        if result.returncode:raise RuntimeError(name+' failed: '+str(result.returncode))
        return time.monotonic()-started
    def encoded(pcm,root):
        data=np.fromfile(pcm,'<f4').reshape(2,-1).T.copy();raw=root/'interleaved.f32';raw.write_bytes(data.tobytes())
        run([str(a.ffmpeg),'-nostdin','-v','error','-f','f32le','-ar','32000','-ac','2','-i',str(raw),'-c:a','aac',str(root/'encoded.m4a')],root,'encode')
        run([str(a.ffmpeg),'-nostdin','-v','error','-i',str(root/'encoded.m4a'),'-f','f32le','-acodec','pcm_f32le',str(root/'decoded.f32')],root,'decode')
        return np.fromfile(root/'decoded.f32','<f4')
    for name,state in a.case:
        if name not in ('C0','C1','C2','H0','H1','H2'):raise ValueError('not a frozen primary/held-out case')
        root=a.out/name;root.mkdir();native=root/'native';native.mkdir();oracle=root/'oracle';row=dict(case=name,state=state)
        try:
            row['native_command_seconds']=run([str(a.source/'bin/cuda_sglang_audio'),str(a.model),state,str(native/'pcm.f32')],root,'native',dict(H3_SGLANG_CUBLAS_LIBRARY=a.private_cublas))
            row['oracle_command_seconds']=run([str(a.oracle_python),str(a.source/'tests/cuda_sglang_audio.py'),'--model',str(a.model),'--state',state,'--out',str(oracle)],root,'oracle')
            n=np.fromfile(native/'pcm.f32','<f4');o=np.fromfile(oracle/'pcm.f32','<f4');row['raw_pcm']=metric(n,o)
            row['encoded_pcm']=metric(encoded(native/'pcm.f32',native),encoded(oracle/'pcm.f32',oracle))
            m=row['encoded_pcm'];gate=contract['gates']['decoded_audio'];row['passed']=bool(m.get('compatible') and m.get('finite') and m['relative_l2']<=gate['relative_l2_max'] and m['cosine']>=gate['cosine_min'])
            row['sha256']={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in (native,oracle) for f in folder.iterdir() if f.suffix in ('.f32','.m4a')}
        except (OSError,ValueError,RuntimeError) as e:row.update(passed=False,error=str(e))
        rows.append(row);(root/'result.json').write_text(json.dumps(row,indent=2)+'\n')
        report=dict(kind='crossed audio decoder and codec; no generation/performance qualification',source_binary_sha256=hashlib.sha256((a.source/'bin/cuda_sglang_audio').read_bytes()).hexdigest(),contract=contract,results=rows,passed=all(r['passed'] for r in rows))
        (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(name,json.dumps({k:v for k,v in row.items() if k not in ('sha256',)}),flush=True)
    return 0 if all(r['passed'] for r in rows) else 1
if __name__=='__main__':raise SystemExit(main())
