#!/usr/bin/env python3
"""Finite operands whose FP32 router moments overflow must recover to exact."""
import fcntl,json
from pathlib import Path
import numpy as np
from cuda_sol_campaign import ROOT,execute,verify,save
from cuda_sol_oracle import bf,fp,metadata,write_layout,sha

def main():
    with (ROOT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
        dest=ROOT/'statistic-recovery';dest.mkdir(exist_ok=False)
        x=np.empty((3,128,2,128),dtype='float32');x[0]=1e-10;x[1,:64]=1e20;x[1,64:]=-1e20;x[2]=.7
        raw=bf(x);raw.tofile(dest/'input.qkv');write_layout(dest/'layout.bin',128,metadata(128,32))
        args=['./bin/cuda_sol_native','--input',str(dest/'input.qkv'),'--layout',str(dest/'layout.bin'),
              '--output',str(dest/'out.bf16'),'--routes',str(dest/'routes.bin'),'--stats',str(dest/'native.json'),
              '--sequence','128','--heads','2','--q','32','--minimum','0','--tau','1','--radius','0']
        row=execute('B-statistic-recovery','B',args,60,allow_failure=True)
        stats=json.loads((dest/'native.json').read_text());out=fp(np.fromfile(dest/'out.bf16',dtype='<u2'))
        error=float(np.max(np.abs(out-fp(raw[2,0,0,0]))));mask=np.fromfile(dest/'routes.bin',dtype='uint8')
        result=dict(passed=row['returncode']==0 and stats['ok'] and stats['guards'] and stats['inputs_unchanged']
                    and stats['pairs'][5]>0 and stats['pairs'][1]==0 and bool(np.all(mask==1))
                    and bool(np.isfinite(out).all()) and error<=.01,
                    native=stats,max_constant_error=error,input_sha256=sha(dest/'input.qkv'),layout_sha256=sha(dest/'layout.bin'))
        save(dest/'result.json',result);assert result['passed'],result
if __name__=='__main__':main()
